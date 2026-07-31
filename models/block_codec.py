"""
BlockCodec: 统一的按基础块自描述编解码结构。

设计目标（用户定稿）：
  - 每个基础块捆成一条自描述记录：[结构码] + [各叶子: 定长系数 δ + EDWB 残差]
  - unsplit 块 = 1 个叶子（整块），是分裂的特例，不浪费槽位
  - 结构码：隐式二叉树形状，1 字节即可（支持到深度 4）
  - 系数 δ：定长（每叶子 K_fixed+2 个），利于批处理 GEMM + 行寻址
  - 残差：保留 EDWB 动态位宽（每叶子 R_min + bits + packed），不定长（保压缩比）
  - 随机访问 O(1)：structure_code[block_id] + block_offset 前缀和直接寻址

统一表示（FourierCoeffs）：
  base 系数 = decoder(z)（隐式），patch 系数 = δ（显式），二者解码后都是
  (v0, v1, a[K]) 一组傅里叶系数，走同一个 synthesize：
      output(t) = (1-t)v0 + t v1 + Σ_k a_k sin(π k t)

结构码枚举（叶子划分数 = C(d)，C(0)=1, C(d)=1+C(d-1)^2）：
  depth=2 (100→25): 5 种   → 1 字节够（3 bit）
      0: [全块]  1: [半][半]  2: [半][1/4][1/4]  3: [1/4][1/4][半]  4: 全 1/4
  depth=3 (256→32): 26 种  → 1 字节够（5 bit）
  depth=4         : 677 种 → 超过 256，需 2 字节
  code_bytes 由 TreeCodebook 按 n_codes 自动选（<=256→1B, 否则 2B）。
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import torch

from models.patch_split import build_patch_basis


# =============================================================================
# 隐式二叉树：结构码 <-> 叶子划分
# =============================================================================

def enumerate_partitions(depth: int) -> List[Tuple[float, ...]]:
    """
    枚举 [0,1) 区间在二分树下、最大深度 depth 的所有叶子划分。
    每个划分用叶子长度(相对块长的比例)的元组表示，按时间顺序。

    递归：P(0) = {(1.0,)}；P(d) = {不分} ∪ {左划分⊕右划分(各半)}。
    去重并排序，得到稳定的"结构码 → 划分"映射。
    """
    cache: Dict[int, List[Tuple[float, ...]]] = {}

    def rec(d: int) -> List[Tuple[float, ...]]:
        if d in cache:
            return cache[d]
        result = [(1.0,)]  # 不分裂：整段一个叶子
        if d > 0:
            sub = rec(d - 1)
            for left in sub:
                for right in sub:
                    # 左右各占半，叶子长度减半后拼接
                    merged = tuple(x * 0.5 for x in left) + tuple(x * 0.5 for x in right)
                    result.append(merged)
        # 去重(保序)
        seen = set()
        uniq = []
        for p in result:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        cache[d] = uniq
        return uniq

    return rec(depth)


class TreeCodebook:
    """结构码 <-> 叶子划分 的双向表。code_bytes 按划分数自动选（<=256→1B, 否则 2B）。"""

    def __init__(self, max_depth: int):
        self.max_depth = max_depth
        self.partitions = enumerate_partitions(max_depth)   # List[Tuple[float,...]]
        self.code_of = {p: i for i, p in enumerate(self.partitions)}
        self.n_codes = len(self.partitions)
        # 结构码落盘字节：1B 覆盖 <=256 种（depth<=3），否则 2B（depth=4=677 种）
        if self.n_codes <= 256:
            self.code_bytes = 1
        elif self.n_codes <= 65536:
            self.code_bytes = 2
        else:
            raise ValueError(f"max_depth={max_depth} 产生 {self.n_codes} 种划分，超过 2 字节上限。")

    def leaves(self, code: int) -> Tuple[float, ...]:
        """结构码 → 叶子长度比例元组。"""
        return self.partitions[code]

    def num_leaves(self, code: int) -> int:
        return len(self.partitions[code])

    def leaf_bounds(self, code: int, block_start: int, block_len: int) -> List[Tuple[int, int]]:
        """结构码 → 叶子的 [start,end) 绝对边界列表（按时间顺序）。"""
        bounds = []
        t = block_start
        for frac in self.partitions[code]:
            seg_len = int(round(frac * block_len))
            bounds.append((t, t + seg_len))
            t += seg_len
        # 末段对齐到块尾（消除 round 误差）
        if bounds:
            s, _ = bounds[-1]
            bounds[-1] = (s, block_start + block_len)
        return bounds

    def locate_leaf(self, code: int, block_start: int, block_len: int, t: int) -> int:
        """O(1)/O(叶子数) 定位 t 落在该块第几个叶子（叶子数 <= 2^depth，常数）。"""
        rel = t - block_start
        acc = 0
        for i, frac in enumerate(self.partitions[code]):
            seg_len = int(round(frac * block_len))
            if rel < acc + seg_len:
                return i
            acc += seg_len
        return len(self.partitions[code]) - 1


# =============================================================================
# 统一傅里叶系数
# =============================================================================

@dataclass
class FourierCoeffs:
    """
    统一傅里叶系数：base(decoder 隐式) 与 patch(存储显式) 的共同表示。

    统一约定：系数为合一向量 δ ∈ [K+2]，对应基底 Φ = build_patch_basis(L, K)：
        Φ 列 = [(1-t), t, sin(π1t), ..., sin(πKt)]
        output(t) = Φ(t) · δ
    base 与 patch 都表示成这同一个 δ（base 的 δ 来自 decoder，patch 的 δ 直接存），
    走同一个 synthesize / 同一个 build_patch_basis，真正"一套"。
    """
    delta: torch.Tensor       # [K+2] 或 [B, K+2]

    @staticmethod
    def synthesize(delta: torch.Tensor, seg_len: int, K: int,
                   device=None, dtype=torch.float32) -> torch.Tensor:
        """
        合成波形：output = Φ · δ。base 与 patch 共用此内核（与批处理同一基底）。

        Args:
            delta: [K+2]（单段）或 [B, K+2]（批量，一次 GEMM）
            seg_len: 段长（点数）
            K: DST 频率数
        Returns:
            [seg_len] 或 [B, seg_len]
        """
        if device is None:
            device = delta.device
        Phi = build_patch_basis(seg_len, K, device=device, dtype=delta.dtype)  # [L, K+2] 缓存
        if delta.dim() == 1:
            return Phi @ delta                          # [L]
        return delta @ Phi.transpose(0, 1)              # [B, L] 一次 GEMM

    @staticmethod
    def from_decoder(v0, v1, a) -> torch.Tensor:
        """把 decoder 输出的 (v0, v1, a[K]) 打包成统一 δ=[v0, v1, a_1..a_K]。"""
        v0 = v0.reshape(-1)
        v1 = v1.reshape(-1)
        if a.dim() == 1:
            return torch.cat([v0, v1, a], dim=0)                    # [K+2]
        return torch.cat([v0.unsqueeze(-1), v1.unsqueeze(-1), a], dim=-1)  # [B, K+2]


# =============================================================================
# 叶子记录
# =============================================================================

@dataclass
class LeafRecord:
    """单个叶子：定长系数 δ + EDWB 残差元信息。"""
    start: int
    end: int
    delta_q: torch.Tensor       # [K_fixed+2] 定长（含 v0,v1,a_1..a_K）
    # EDWB 残差（变长位宽，保压缩比）
    res_min: float = 0.0
    res_bits: int = 0
    res_bytes: int = 0          # ceil(seg_len*bits/8) + header
    coeff_bytes: int = 0        # 定长系数字节


@dataclass
class BlockRecord:
    """一个基础块的自描述记录：结构码 + 叶子列表。"""
    block_id: int
    block_start: int
    block_len: int
    code: int                    # 结构码（隐式树形状）
    leaves: List[LeafRecord] = field(default_factory=list)
    left_id: int = -1            # 块左边界 grid 节点（base 预测用）
    right_id: int = -1           # 块右边界 grid 节点

    def total_bytes(self, include_code: bool = True, code_bytes: int = 1) -> int:
        b = sum(lf.coeff_bytes + lf.res_bytes for lf in self.leaves)
        if include_code:
            b += code_bytes      # 结构码字节（1 或 2，由 codebook 决定）
        return b


# =============================================================================
# BlockCodec：持有所有块记录，提供 O(1) 随机访问 + 批处理重建
# =============================================================================

class BlockCodec:
    """
    统一块编解码器。

    存储（每基础块密集寻址）：
        structure_code[block_id]  : 1 字节
        block_offset[block_id]    : 叶子在全局叶子数组中的起始下标（前缀和）
    全局叶子数组：
        coeff_pool  : [total_leaves, K+2]  定长系数（行寻址）
        leaf_meta   : 每叶子 (block_id, start, end, res_min, res_bits)
    """

    def __init__(self, base_block_size: int, min_resolution: int, K_fixed: int,
                 max_depth: int = None):
        self.base_block_size = base_block_size
        self.min_resolution = min_resolution
        self.K_fixed = K_fixed
        # max_depth：默认由 base/min 推理论上限；也可显式传入（如用 RDO 回收后的
        # 实测最大树高），以收紧结构码位宽（实测高度常 < 理论上限 → 码更短）。
        if max_depth is None:
            md = 0
            ratio = base_block_size // min_resolution
            while (1 << md) < ratio:
                md += 1
            max_depth = md
        self.max_depth = max_depth
        self.codebook = TreeCodebook(max_depth)

        self.blocks: Dict[int, BlockRecord] = {}   # block_id -> record
        # 寻址结构（commit 后由 finalize() 构建）
        self._block_offset: Dict[int, int] = {}
        self._coeff_pool: Optional[torch.Tensor] = None
        self._leaf_meta: List[dict] = []
        self._finalized = False
        self.residual_codec = None

    # ----- 写入 -----
    def add_block(self, record: BlockRecord):
        self.blocks[record.block_id] = record
        self._finalized = False

    def finalize(self, device=None, dtype=torch.float32):
        """构建 O(1) 寻址结构：前缀和 + 紧凑系数池。"""
        self._block_offset = {}
        rows = []
        self._leaf_meta = []
        cursor = 0
        for bid in sorted(self.blocks.keys()):
            rec = self.blocks[bid]
            self._block_offset[bid] = cursor
            for lf in rec.leaves:
                rows.append(lf.delta_q)
                self._leaf_meta.append({
                    'block_id': bid, 'start': lf.start, 'end': lf.end,
                    'res_min': lf.res_min, 'res_bits': lf.res_bits,
                })
                cursor += 1
        if rows:
            self._coeff_pool = torch.stack(rows, dim=0).to(device=device, dtype=dtype)
        else:
            self._coeff_pool = torch.zeros(0, self.K_fixed + 2, device=device, dtype=dtype)
        self._finalized = True
        return self

    # ----- O(1) 随机访问辅助 -----
    def attach_residual_codec(self, residual_codec):
        """挂载 ResidualCodec（按全局叶子行号对齐）。访问时用 row 取残差。"""
        self.residual_codec = residual_codec
        return self

    def leaf_global_index(self, block_id: int, leaf_idx_in_block: int) -> int:
        """全局叶子行号 = block_offset[block_id] + 块内叶子序号。O(1)。"""
        return self._block_offset[block_id] + leaf_idx_in_block

    def locate(self, t: int) -> Tuple[int, int, Tuple[int, int]]:
        """
        定位时间点 t：返回 (block_id, global_leaf_row, (leaf_start, leaf_end))。
        直接用存储的叶子边界（rec.leaves[i].start/end），避免与结构码重算不一致
        （末块截断 / _nearest_code 重映射时，重算边界会和实际存储不符）。
        """
        block_id = t // self.base_block_size
        rec = self.blocks[block_id]
        leaf_i = 0
        for i, lf in enumerate(rec.leaves):
            if lf.start <= t < lf.end:
                leaf_i = i
                break
        else:
            leaf_i = len(rec.leaves) - 1   # 落在末尾（边界保护）
        lf = rec.leaves[leaf_i]
        row = self.leaf_global_index(block_id, leaf_i)
        return block_id, row, (lf.start, lf.end)

    # ----- 统计 -----
    def total_bytes(self) -> dict:
        n_blocks = len(self.blocks)
        n_leaves = sum(len(r.leaves) for r in self.blocks.values())
        code_bytes = n_blocks * self.codebook.code_bytes
        coeff_bytes = sum(lf.coeff_bytes for r in self.blocks.values() for lf in r.leaves)
        res_bytes = sum(lf.res_bytes for r in self.blocks.values() for lf in r.leaves)
        return {
            'num_blocks': n_blocks,
            'num_leaves': n_leaves,
            'code_bytes': code_bytes,
            'coeff_bytes': coeff_bytes,
            'residual_bytes': res_bytes,
            'total_bytes': code_bytes + coeff_bytes + res_bytes,
        }

    # ----- 批处理重建（同长叶子分桶 GEMM）-----
    def reconstruct_patch_batched(self) -> Dict[int, torch.Tensor]:
        """
        把所有叶子的 patch 修正按 seg_len 分桶，每桶一次 GEMM。
        返回 {global_leaf_row: patch_curve[seg_len]}（仅 DST+ramp 部分，不含 base）。
        """
        if not self._finalized:
            self.finalize(device=self._coeff_pool.device if self._coeff_pool is not None else None)
        from collections import defaultdict
        buckets = defaultdict(list)   # seg_len -> [(row, delta_q)]
        for row, meta in enumerate(self._leaf_meta):
            seg_len = meta['end'] - meta['start']
            buckets[seg_len].append((row, self._coeff_pool[row]))
        out = {}
        K = self.K_fixed
        for seg_len, items in buckets.items():
            if seg_len <= 0:
                continue
            mat = torch.stack([it[1] for it in items], dim=0)              # [M, K+2]
            # 统一合成内核：base 与 patch 共用 FourierCoeffs.synthesize
            recon = FourierCoeffs.synthesize(mat, seg_len, K)              # [M, seg_len] 一次 GEMM
            for i, (row, _) in enumerate(items):
                out[row] = recon[i]
        return out
