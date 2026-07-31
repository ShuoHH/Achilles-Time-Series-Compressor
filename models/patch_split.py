"""
Parent-Anchored Additive Patch Split (根治共享 decoder 污染的 split 表示)

动机
----
现有 split：child 通过共享 decoder + 新增 z_M 节点重解码。分裂阶段一旦训练
decoder，未被分裂的块会被难块带偏（全局污染），压缩比不升反降。学习率方案无法
根治，因为这是"共享状态被改"的结构问题。

本方案：split 阶段不碰 decoder / normalizer / quantizer / 旧 token。child 表示为
冻结 parent 预测之上的**加性局部修正**：

    y_c(t) = y_p(t_parent段)  +  Δ_c(t_child)

    Δ_c(t) = (1-t)·d0 + t·d1 + Σ_{k=1..K} d_k · sin(π·k·t)

基底与 FourierDecoder 输出形式逐字一致（ramp + 半周期 DST，ω_k=πk）。
Δ_c 对系数 δ_c=[d0,d1,d_1..d_K] 线性，可闭式最小二乘求解，不存在"不收敛"。

性质
----
1. 未分裂块输出逐比特不变 → unsplit residual bytes delta ≡ 0（数学恒等，非调参）。
2. 不新增 z_M 节点（FourierDecoder 只用 z_start，child 复用 parent 预测）。
   split 额外成本 = 每个 child 的 (K+2) 个量化系数 + 小量化头。
3. 随机访问仍 O(1)：定位 parent → 算 parent 预测 → 加 patch → 加 EDWB 残差。

字节核算全部复用 FallbackDict 真实 EDWB 公式，保证与最终落盘一致。
"""

import math
from dataclasses import dataclass
from typing import Optional, List, Dict

import torch

from models.fallback_dict import FallbackDict


# 量化头开销（int8 需存 min + scale 作 FP16）
_INT8_HEADER = 4   # min(FP16) + scale(FP16)
# 每个 patch 条目的元数据（标记被 patch 的 child + K + mode），保守计 2B
PATCH_META_BYTES = 2


# 基底缓存：key=(length,K,device,dtype) → Tensor[length,K+2]。运行时内存，不落盘。
_PATCH_BASIS_CACHE: dict = {}


def build_patch_basis(length: int, K: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    构建 patch 基底 B(t) ∈ [length, K+2]，列为：
        [(1-t), t, sin(π·1·t), ..., sin(π·K·t)]
    与 FourierDecoder 的 ramp + DST 形式一致（端点 sin 项为 0）。

    带缓存：同 (length, K, device, dtype) 复用，避免随机访问/批量重建时重复构造。
    缓存的是模型常数（不落盘、不计入压缩），只占运行时少量内存。
    """
    key = (length, K, str(device), str(dtype))
    cached = _PATCH_BASIS_CACHE.get(key)
    if cached is not None:
        return cached
    t = torch.linspace(0.0, 1.0, length, device=device, dtype=dtype)  # [L]
    cols = [(1.0 - t), t]
    for k in range(1, K + 1):
        cols.append(torch.sin(math.pi * k * t))
    B = torch.stack(cols, dim=1)  # [L, K+2]
    _PATCH_BASIS_CACHE[key] = B
    return B


def quantize_coeffs(delta: torch.Tensor, mode: str):
    """
    量化 patch 系数。

    Returns: (delta_q_dequantized, meta)
        delta_q_dequantized: 反量化后的系数（用于重建，反映真实落盘精度）
        meta: dict（落盘所需，用于核算/复现）
    """
    if mode == 'fp16':
        dq = delta.to(torch.float16).to(delta.dtype)
        return dq, {'mode': 'fp16'}
    elif mode == 'int8':
        vmin = delta.min()
        vmax = delta.max()
        scale = (vmax - vmin) / 255.0
        if scale <= 0:
            # 全相等：仅存 min，scale=0
            return torch.full_like(delta, float(vmin)), {'mode': 'int8', 'min': float(vmin), 'scale': 0.0}
        q = torch.round((delta - vmin) / scale).clamp(0, 255)
        dq = q * scale + vmin
        return dq, {'mode': 'int8', 'min': float(vmin), 'scale': float(scale)}
    else:
        raise ValueError(f"unknown quant mode: {mode}")


def patch_coeff_bytes(K: int, mode: str) -> int:
    """单个 child 的 patch 系数字节数（含量化头）。"""
    n_coeff = K + 2
    if mode == 'fp16':
        return n_coeff * 2
    elif mode == 'int8':
        return n_coeff * 1 + _INT8_HEADER
    else:
        raise ValueError(f"unknown quant mode: {mode}")


@dataclass
class ChildPatchFit:
    """单个 child 半块的 patch 拟合结果。"""
    length: int
    K: int
    mode: str
    delta_q: torch.Tensor          # 反量化后的系数（重建用）
    meta: dict
    residual_span: float
    residual_bytes: int            # EDWB 残差字节（量化 patch 后）
    coeff_bytes: int               # patch 系数字节
    eps: float

    @property
    def total_bytes(self) -> int:
        return self.residual_bytes + self.coeff_bytes


def fit_child_patch(target: torch.Tensor, eps: float, K: int, mode: str) -> ChildPatchFit:
    """
    对单个 child 半块拟合 patch。

    Args:
        target: 该 child 段的 (真值 - 冻结parent预测) 残差 [L]
        eps:    该 child 的 EDWB 误差容限（归一化空间）
        K:      DST 频率数
        mode:   'int8' | 'fp16'
    """
    L = target.shape[0]
    device, dtype = target.device, target.dtype
    B = build_patch_basis(L, K, device=device, dtype=dtype)  # [L, K+2]

    # 闭式最小二乘 δ* = argmin ||B δ - target||
    # torch 1.7 无 torch.linalg.lstsq，改用正规方程 (BᵀB + λI) δ = Bᵀy，
    # 加极小 ridge 保数值稳定，torch.solve 在 1.7 可用。
    n = B.shape[1]
    BtB = B.transpose(0, 1) @ B                                   # [n, n]
    Bty = B.transpose(0, 1) @ target.unsqueeze(-1)               # [n, 1]
    ridge = 1e-8 * torch.eye(n, device=device, dtype=dtype)
    try:
        sol = torch.solve(Bty, BtB + ridge).solution.squeeze(-1)  # [n]
    except Exception:
        sol = torch.pinverse(B) @ target                          # 兜底

    # 量化（反映真实落盘精度）
    delta_q, meta = quantize_coeffs(sol, mode)

    recon = B @ delta_q                      # [L]
    resid = target - recon                   # patch 后的最终残差
    span = (resid.max() - resid.min()).item()
    res_bytes = FallbackDict.estimate_bitwidth_cost(span, L, eps)
    c_bytes = patch_coeff_bytes(K, mode)

    return ChildPatchFit(
        length=L, K=K, mode=mode, delta_q=delta_q, meta=meta,
        residual_span=span, residual_bytes=res_bytes, coeff_bytes=c_bytes, eps=eps,
    )


def fit_child_best(target: torch.Tensor, eps: float,
                   K_list=(0, 4, 8, 16), modes=('int8', 'fp16')) -> ChildPatchFit:
    """枚举 (K, mode)，返回 total_bytes 最小的拟合。"""
    best = None
    for mode in modes:
        for K in K_list:
            if K + 2 > target.shape[0]:
                continue  # 系数数不能超过点数
            fit = fit_child_patch(target, eps, K, mode)
            if best is None or fit.total_bytes < best.total_bytes:
                best = fit
    if best is None:
        # 极短块兜底：K=0 int8
        best = fit_child_patch(target, eps, 0, 'int8')
    return best


def fit_child_fixedK(target: torch.Tensor, eps: float, K: int,
                     modes=('int8', 'fp16')) -> ChildPatchFit:
    """固定 K（定长系数），仅在 mode 间选更省者。用于定长 patch 方案。"""
    best = None
    for mode in modes:
        if K + 2 > target.shape[0]:
            continue
        fit = fit_child_patch(target, eps, K, mode)
        if best is None or fit.total_bytes < best.total_bytes:
            best = fit
    if best is None:
        best = fit_child_patch(target, eps, 0, 'int8')
    return best


@dataclass
class SplitPatchResult:
    """一个 parent 块的 patch-split 评估结果。"""
    start: int
    end: int
    mid: int
    parent_len: int
    parent_span: float
    parent_bits: int
    parent_residual_bytes: int     # 不分裂的旧成本
    left_fit: ChildPatchFit
    right_fit: ChildPatchFit
    new_total_bytes: int           # 分裂后新成本（含 patch 系数 + meta）
    net_gain: int                  # old - new
    accept: bool
    # 精度（保留每块提升情况）
    parent_mae: float
    child_mae: float


def evaluate_parent_patch_split(
    parent_pred: torch.Tensor,     # 冻结 parent 在 [start,end) 的预测 [parent_len]
    true_full: torch.Tensor,       # 真值 [parent_len]（归一化空间）
    start: int, end: int, mid: int,
    eps_parent: float, eps_left: float, eps_right: float,
    K_list=(0, 4, 8, 16), modes=('int8', 'fp16'),
) -> SplitPatchResult:
    """
    对一个 parent 块评估 patch-split 的字节收益（report-only，不改任何共享状态）。

    old = parent_residual_bytes（不分裂）
    new = left.total_bytes + right.total_bytes + PATCH_META_BYTES
    accept = new < old
    """
    parent_len = end - start
    l_local = mid - start
    r_local = end - mid

    parent_resid = true_full - parent_pred
    parent_span = (parent_resid.max() - parent_resid.min()).item()
    parent_bits = FallbackDict.compute_bits(parent_span, eps_parent)
    parent_bytes = FallbackDict.estimate_bitwidth_cost(parent_span, parent_len, eps_parent)
    parent_mae = parent_resid.abs().mean().item()

    # child 目标 = 真值 - 冻结 parent 预测（对应半段）
    left_target = (true_full[:l_local] - parent_pred[:l_local])
    right_target = (true_full[l_local:] - parent_pred[l_local:])

    left_fit = fit_child_best(left_target, eps_left, K_list, modes)
    right_fit = fit_child_best(right_target, eps_right, K_list, modes)

    new_total = left_fit.total_bytes + right_fit.total_bytes + PATCH_META_BYTES
    net_gain = parent_bytes - new_total
    accept = net_gain > 0

    # child MAE（重建后）：用于保留精度提升报告
    Bl = build_patch_basis(l_local, left_fit.K, device=parent_pred.device, dtype=parent_pred.dtype)
    Br = build_patch_basis(r_local, right_fit.K, device=parent_pred.device, dtype=parent_pred.dtype)
    l_recon = parent_pred[:l_local] + Bl @ left_fit.delta_q
    r_recon = parent_pred[l_local:] + Br @ right_fit.delta_q
    l_mae = (true_full[:l_local] - l_recon).abs().mean().item()
    r_mae = (true_full[l_local:] - r_recon).abs().mean().item()
    child_mae = (l_mae * l_local + r_mae * r_local) / parent_len

    return SplitPatchResult(
        start=start, end=end, mid=mid, parent_len=parent_len,
        parent_span=parent_span, parent_bits=parent_bits,
        parent_residual_bytes=parent_bytes,
        left_fit=left_fit, right_fit=right_fit,
        new_total_bytes=new_total, net_gain=net_gain, accept=accept,
        parent_mae=parent_mae, child_mae=child_mae,
    )


# =============================================================================
# Ablation: 整块加系数 (whole-block) vs 分裂 (split)
#
# 审稿人质疑："既然是残差修正，为什么要分裂？直接对整块拟合更高阶傅里叶
# 不就行了？" 本对照在【相同系数预算】下比较两种方案，用同一套 EDWB 公式、
# 同一个 parent 预测、同一个 eps，隔离出"分裂"本身的独立价值。
#
# A) whole-block: 对整个 parent 残差拟合一个 (2K+2) 阶 patch（不分裂）。
#                 系数数 = 2K+2，与 B 的两个 child 各 (K+2) 近似相等。
# B) split:       左右各拟合一个 (K+2) 阶 patch（evaluate_parent_patch_split）。
#
# 关键差异：A 的傅里叶基是【全局】的（整块 span 由全块最坏点主导，EDWB 按
# 最坏 span 收费）；B 把残差按时间隔离成两段，每段 span 独立 → 难的那半不再
# 污染好的那半。若 B 在相同系数预算下字节更少，即证明分裂的 span 隔离价值，
# 而非"多给了系数"。
# =============================================================================

@dataclass
class WholeBlockResult:
    start: int
    end: int
    block_len: int
    fit: ChildPatchFit          # 整块单 patch
    parent_residual_bytes: int
    new_total_bytes: int        # patch 系数 + 残差
    net_gain: int
    accept: bool


def evaluate_whole_block_patch(
    parent_pred: torch.Tensor, true_full: torch.Tensor,
    start: int, end: int, eps_parent: float,
    K_whole_list, modes=('int8', 'fp16'),
) -> WholeBlockResult:
    """
    A) 整块加系数：不分裂，对整个 parent 残差拟合一个高阶 patch。

    Args:
        K_whole_list: 整块允许的频率数候选（应≈ 2*split_K，保证系数预算可比）
    """
    block_len = end - start
    parent_resid = true_full - parent_pred
    parent_span = (parent_resid.max() - parent_resid.min()).item()
    parent_bytes = FallbackDict.estimate_bitwidth_cost(parent_span, block_len, eps_parent)

    target = true_full - parent_pred
    fit = fit_child_best(target, eps_parent, K_list=K_whole_list, modes=modes)
    # 整块只有一个 patch 条目（meta 计一次）
    new_total = fit.total_bytes + PATCH_META_BYTES
    net_gain = parent_bytes - new_total
    return WholeBlockResult(
        start=start, end=end, block_len=block_len, fit=fit,
        parent_residual_bytes=parent_bytes,
        new_total_bytes=new_total, net_gain=net_gain, accept=net_gain > 0,
    )


# =============================================================================
# 多层 patch + 自下而上 RDO 剪枝（离线核验）
#
# 模型：自适应分段深度（segmentation tree），不是叠加层。
#   - 把 base 块递归二分到 max_depth，每个区间独立在 (true - parent_pred) 的
#     对应段上闭式解一个单层 patch（与现单层 split 的 child 拟合方式完全一致）。
#   - 自下而上 DP：每个内部节点比较
#       保留分裂 = 左子树最优字节 + 右子树最优字节
#       合并     = 本区间作为一个单层 patch 的字节
#     取更小者。DP 自动决定每个区间的最优分段深度。
#   - 这与 VectorGC.dp_prune 的"保留分裂 vs 合并"同构，只是代价从"grid 节点"
#     换成"patch δ + 残差"。每个叶子是一段独立 patch，互不重叠，无层间扣减问题。
# =============================================================================

@dataclass
class MultiLayerNode:
    start: int
    end: int
    single_fit: ChildPatchFit        # 本区间作为单层 patch 的拟合
    single_bytes: int                # 本区间单层总字节(δ + 残差 + meta)
    best_bytes: int                  # DP 最优(可能来自子树)
    is_split: bool                   # DP 决策：是否分裂
    left: 'MultiLayerNode' = None
    right: 'MultiLayerNode' = None
    depth_used: int = 0              # 该区间 DP 后实际用到的最大深度
    full_bytes: int = 0              # 不回收(强制切到 max_depth)的字节
    full_leaves: int = 1             # 不回收时的叶子数(最细分段数)


def _fit_interval_single(parent_pred, true_full, start, end, blk_start,
                         eps, K_list, modes):
    """把 [start,end) 区间作为单层 patch 拟合(在 parent 预测残差上)。"""
    l0 = start - blk_start
    l1 = end - blk_start
    target = true_full[l0:l1] - parent_pred[l0:l1]
    fit = fit_child_best(target, eps, K_list=K_list, modes=modes)
    return fit


def build_multilayer_tree(parent_pred, true_full, start, end, blk_start,
                          eps_fn, K_list, modes, min_len, depth, max_depth) -> MultiLayerNode:
    """
    递归构建多层 patch 树 + 自下而上 DP 剪枝。

    同时记录 full_bytes / full_leaves（强制切到 max_depth、不回收的代价），
    与 best_bytes / 剪枝后叶子数对比即得"回收量"。

    Args:
        parent_pred: 整个 base 块的冻结预测 [base_len]
        true_full:   整个 base 块真值 [base_len]
        start, end:  当前区间(绝对时间)
        blk_start:   base 块起始(用于切片)
        eps_fn:      eps_fn(t0, t1) -> 该区间 eps
        min_len:     最小区间长度(不再细分)
        depth/max_depth: 当前/最大深度
    Returns: MultiLayerNode(已含 DP 最优)
    """
    seg_len = end - start
    eps = eps_fn(start, end)
    single_fit = _fit_interval_single(parent_pred, true_full, start, end,
                                      blk_start, eps, K_list, modes)
    single_bytes = single_fit.total_bytes + PATCH_META_BYTES

    node = MultiLayerNode(
        start=start, end=end, single_fit=single_fit,
        single_bytes=single_bytes, best_bytes=single_bytes,
        is_split=False, depth_used=depth,
        full_bytes=single_bytes, full_leaves=1,
    )

    # 触底：达到最大深度或长度不足
    if depth >= max_depth or seg_len < 2 * min_len:
        return node

    mid = start + seg_len // 2
    left = build_multilayer_tree(parent_pred, true_full, start, mid, blk_start,
                                 eps_fn, K_list, modes, min_len, depth + 1, max_depth)
    right = build_multilayer_tree(parent_pred, true_full, mid, end, blk_start,
                                  eps_fn, K_list, modes, min_len, depth + 1, max_depth)

    # 不回收(强制切到底)的代价 = 左右子树全切叶子之和
    node.full_bytes = left.full_bytes + right.full_bytes
    node.full_leaves = left.full_leaves + right.full_leaves

    split_bytes = left.best_bytes + right.best_bytes
    # 自下而上 DP：保留分裂 vs 合并单层
    if split_bytes < single_bytes:
        node.best_bytes = split_bytes
        node.is_split = True
        node.left = left
        node.right = right
        node.depth_used = max(left.depth_used, right.depth_used)
    # else: 合并单层(默认)，best_bytes=single_bytes，不挂子树

    return node


def count_leaves_and_depth(node: MultiLayerNode):
    """统计 DP 后该树的叶子数(实际 patch 段数)和最大深度。"""
    if not node.is_split:
        return 1, node.depth_used
    ln, ld = count_leaves_and_depth(node.left)
    rn, rd = count_leaves_and_depth(node.right)
    return ln + rn, max(ld, rd)


def collect_leaf_segments(node: MultiLayerNode) -> List[dict]:
    """
    从 DP 剪枝后的树中提取最终保留的叶子段（自下而上回收后的结果）。

    每个叶子 = 一段独立 patch，返回：
        [{'start','end','fit': ChildPatchFit}, ...]
    这些段拼接覆盖整个 base 块，是回收冗余分裂层后的最优分段。
    """
    if not node.is_split:
        return [{'start': node.start, 'end': node.end, 'fit': node.single_fit}]
    return collect_leaf_segments(node.left) + collect_leaf_segments(node.right)


def tree_height_after_pruning(node: MultiLayerNode) -> int:
    """
    RDO 回收（DP 剪枝）后该块的实际树高。

    height = 从根到最深保留叶子的分裂次数：
      - 未分裂(叶子) → 0
      - 分裂 → 1 + max(左, 右)
    用于统计实际需要的 max_depth，进而决定结构码位宽。
    """
    if not node.is_split:
        return 0
    return 1 + max(tree_height_after_pruning(node.left),
                   tree_height_after_pruning(node.right))


class PatchSplitManager:
    """
    已提交的 parent-anchored patch split 的持有者（加性覆盖层）。

    统一分段模型（segment-based）：
    - 每个被 patch 的 base 块存一组【分段】，每段一个独立 patch fit。
      单层分裂 = 2 段；多层（自下而上 RDO 回收后）= N 段（N≥1）。
    - 不改 index_table、不分配 z_M、不训练 decoder。
    - 解码/核算：被 patch 的块 → 逐段 (parent 预测 + Φ·δ) 拼接；未 patch 块不变。
    - 按 base 块区间寻址，O(1) 定位。
    """

    def __init__(self):
        # (start, end) -> List[ {'start','end','fit': ChildPatchFit} ]
        # 段按时间顺序排列，拼接覆盖整个 [start, end)
        self.entries: Dict[tuple, list] = {}

    def is_patched(self, start: int, end: int) -> bool:
        return (start, end) in self.entries

    def commit(self, result: 'SplitPatchResult'):
        """提交一个单层 patch split（兼容接口）：转成左右两段。"""
        self.entries[(result.start, result.end)] = [
            {'start': result.start, 'end': result.mid, 'fit': result.left_fit},
            {'start': result.mid, 'end': result.end, 'fit': result.right_fit},
        ]

    def commit_segments(self, start: int, end: int, segments: list):
        """
        提交多段 patch（多层 RDO 回收后的结果）。

        Args:
            start, end: base 块区间
            segments: [{'start','end','fit'}, ...]，按时间排列，拼接覆盖 [start,end)
        """
        self.entries[(start, end)] = segments

    def get(self, start: int, end: int) -> Optional[list]:
        return self.entries.get((start, end), None)

    def reconstruct_block(self, parent_pred: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """
        用已提交的分段 patch 重建被 patch 的 base 块。

        Args:
            parent_pred: 冻结 parent 在 [start,end) 的预测 [block_len]
            start, end: base 块区间
        Returns:
            重建波形 [block_len]；若该块未被 patch，原样返回 parent_pred。
        """
        segs = self.entries.get((start, end), None)
        if segs is None:
            return parent_pred
        out = parent_pred.clone()
        for seg in segs:
            s0 = seg['start'] - start
            s1 = seg['end'] - start
            fit = seg['fit']
            B = build_patch_basis(s1 - s0, fit.K, device=parent_pred.device, dtype=parent_pred.dtype)
            out[s0:s1] = parent_pred[s0:s1] + B @ fit.delta_q
        return out

    def patch_bytes(self, start: int, end: int) -> int:
        """该块所有段的 patch 系数字节（含 1 个 meta，记录段数 + 各段 K/mode）。"""
        segs = self.entries.get((start, end), None)
        if segs is None:
            return 0
        return sum(seg['fit'].coeff_bytes for seg in segs) + PATCH_META_BYTES

    def child_residual_bytes(self, start: int, end: int) -> int:
        """该块所有段的残差字节之和。"""
        segs = self.entries.get((start, end), None)
        if segs is None:
            return 0
        return sum(seg['fit'].residual_bytes for seg in segs)

    def num_segments(self, start: int, end: int) -> int:
        segs = self.entries.get((start, end), None)
        return len(segs) if segs else 0

    def reconstruct_batched(self, block_preds: dict) -> dict:
        """
        批量重建多个被 patch 的块（保住批处理特性）。

        把所有块的所有 patch 段按 (seg_len, K) 分桶，每桶一次 GEMM
        （δ_matrix[M, K+2] @ Φ^T[K+2, seg_len]），而非逐段循环。
        这与 base 解码的"按块长分桶 + 每桶一次 GEMM"同构。

        Args:
            block_preds: {(start,end): parent_pred[block_len]}  冻结 base 预测
        Returns:
            {(start,end): reconstructed[block_len]}  仅含传入且被 patch 的块
        """
        # 1. 收集所有段，按 (seg_len, K) 分桶
        #    bucket[(seg_len,K)] = list of (block_key, s0, s1, delta_q)
        from collections import defaultdict
        buckets = defaultdict(list)
        out = {}
        for key, pred in block_preds.items():
            segs = self.entries.get(key, None)
            if segs is None:
                continue
            out[key] = pred.clone()
            start = key[0]
            for seg in segs:
                s0 = seg['start'] - start
                s1 = seg['end'] - start
                fit = seg['fit']
                buckets[(s1 - s0, fit.K)].append((key, s0, s1, fit.delta_q))

        # 2. 每桶一次 GEMM
        for (seg_len, K), items in buckets.items():
            if seg_len <= 0:
                continue
            dev = items[0][3].device
            dtype = items[0][3].dtype
            Phi = build_patch_basis(seg_len, K, device=dev, dtype=dtype)  # [seg_len, K+2] 缓存
            delta_mat = torch.stack([it[3] for it in items], dim=0)        # [M, K+2]
            recon = delta_mat @ Phi.transpose(0, 1)                        # [M, seg_len] 一次 GEMM
            # 散射回各自块
            for i, (key, s0, s1, _) in enumerate(items):
                out[key][s0:s1] = out[key][s0:s1] + recon[i]
        return out

    def total_patches(self) -> int:
        return len(self.entries)

    def summary(self) -> dict:
        n = len(self.entries)
        patch_b = sum(self.patch_bytes(s, e) for (s, e) in self.entries)
        child_res_b = sum(self.child_residual_bytes(s, e) for (s, e) in self.entries)
        total_segs = sum(len(v) for v in self.entries.values())
        return {
            'num_patches': n,
            'total_segments': total_segs,
            'total_patch_bytes': patch_b,
            'total_child_residual_bytes': child_res_b,
        }
