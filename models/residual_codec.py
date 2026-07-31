"""
ResidualCodec: EDWB 残差的位流编码 + O(1) 随机访问 + 批处理解包（方案B）。

设计（与讨论一致）：
  - 每个叶子段的残差按 *该段自己的* 动态位宽 b 量化（不共用，per-leaf 位宽）：
        q = round((r - r_min) / (2ε))   ∈ [0, 2^b - 1]
        r̂ = r_min + q · 2ε              （反量化，|r - r̂| ≤ ε）
    b = 0 时该段不存 payload，残差全部用 r_min 近似（0-bit 奇迹）。
  - 存储布局（密集寻址，零冗余偏移）：
        leaf_bits[row]      : uint8   每叶子位宽 b（块头逻辑上属于该块，物理上扁平存）
        leaf_rmin[row]      : float16 每叶子 r_min
        block_bit_offset[bid]: int64  第 bid 块残差码流的起始 *比特* 位置（块级前缀和）
        bitstream           : 紧凑位流，按 (block, leaf) 顺序逐段 b 比特打包
    叶子段长由树码恢复（不存）；块内叶子起点 = 块级偏移 + Σ(前面叶子 len×b)（cumsum，可并行）。
  - O(1) 随机点访问：算出该点所在叶子的比特起点 + 段内偏移，直接读 b 比特。
  - 批处理：按位宽 b 分桶，同位宽的点向量化解包。
"""

import math
from typing import List, Optional

import numpy as np
import torch


class ResidualCodec:
    """EDWB 残差位流编解码器（与 BlockCodec 叶子顺序对齐，按行号 row 寻址）。"""

    def __init__(self, eps: float):
        self.eps = float(eps)
        self.step = 2.0 * float(eps)            # 量化桶宽
        self.leaf_bits: Optional[np.ndarray] = None     # [num_leaves] uint8
        self.leaf_rmin: Optional[np.ndarray] = None     # [num_leaves] float32(存float16精度)
        self.leaf_len: Optional[np.ndarray] = None      # [num_leaves] int32 段长
        self.leaf_bitpos: Optional[np.ndarray] = None   # [num_leaves+1] int64 每叶子比特起点(全局前缀和)
        self.bitstream: Optional[np.ndarray] = None     # uint8 packed
        self._finalized = False

    # ---------------------------------------------------------------- 编码
    @staticmethod
    def _compute_bits(span: float, eps: float) -> int:
        if eps <= 0:
            return 16
        bucket = 2.0 * eps
        if span <= bucket:
            return 0
        return min(int(math.ceil(math.log2(math.ceil(span / bucket)))), 16)

    def encode(self, leaf_residuals: List[np.ndarray]):
        """
        leaf_residuals: 按全局叶子行号顺序排列的残差数组列表，
                        第 row 个元素是该叶子段的残差向量（归一化空间，true - (base+patch)）。
        向量化打包：先逐叶子量化得整数码，再用 numpy 一次性展开成比特、packbits。
        """
        n = len(leaf_residuals)
        bits = np.zeros(n, dtype=np.uint8)
        rmin = np.zeros(n, dtype=np.float32)
        llen = np.zeros(n, dtype=np.int32)
        bitpos = np.zeros(n + 1, dtype=np.int64)

        # 第一遍：量化 + 统计总比特，收集每叶子的 (codes, b)
        codes_list: List[np.ndarray] = []
        total_bits = 0
        for i, r in enumerate(leaf_residuals):
            r = np.asarray(r, dtype=np.float64).reshape(-1)
            L = r.size
            llen[i] = L
            if L == 0:
                codes_list.append(np.zeros(0, dtype=np.int64))
                bitpos[i + 1] = total_bits
                continue
            rmn = float(r.min())
            span = float(r.max() - r.min())
            b = self._compute_bits(span, self.eps)
            bits[i] = b
            rmin[i] = np.float32(rmn)
            if b > 0:
                q = np.round((r - rmn) / self.step).astype(np.int64)
                q = np.clip(q, 0, (1 << b) - 1)
            else:
                q = np.zeros(L, dtype=np.int64)
            codes_list.append(q)
            total_bits += L * b
            bitpos[i + 1] = total_bits

        # 第二遍：向量化展开成全局比特数组，再 packbits（MSB-first）
        bit_arr = np.zeros(int(total_bits), dtype=np.uint8)
        cursor = 0
        for i, q in enumerate(codes_list):
            b = int(bits[i])
            if b == 0 or q.size == 0:
                continue
            L = q.size
            # 每个码展开成 b 个比特（MSB-first）：[L, b]
            shifts = np.arange(b - 1, -1, -1, dtype=np.int64)
            exp = ((q[:, None] >> shifts[None, :]) & 1).astype(np.uint8)  # [L, b]
            seg = exp.reshape(-1)
            bit_arr[cursor:cursor + seg.size] = seg
            cursor += seg.size
        stream = np.packbits(bit_arr) if bit_arr.size else np.zeros(0, dtype=np.uint8)

        self.leaf_bits = bits
        self.leaf_rmin = rmin
        self.leaf_len = llen
        self.leaf_bitpos = bitpos
        self.bitstream = stream
        self._finalized = True
        return self

    # ---------------------------------------------------------------- 单点解码
    def _read_code(self, bitpos: int, b: int) -> int:
        v = 0
        for _ in range(b):
            bit = (self.bitstream[bitpos >> 3] >> (7 - (bitpos & 7))) & 1
            v = (v << 1) | int(bit)
            bitpos += 1
        return v

    def decode_point(self, row: int, offset_in_leaf: int) -> float:
        """第 row 个叶子段内偏移 offset 处的残差反量化值。O(b)。"""
        b = int(self.leaf_bits[row])
        rmn = float(self.leaf_rmin[row])
        if b == 0:
            return rmn
        bitpos = int(self.leaf_bitpos[row]) + offset_in_leaf * b
        q = self._read_code(bitpos, b)
        return rmn + q * self.step

    def decode_leaf(self, row: int) -> np.ndarray:
        """整段解出该叶子残差向量（向量化，避免逐点 Python 循环）。"""
        b = int(self.leaf_bits[row])
        L = int(self.leaf_len[row])
        rmn = float(self.leaf_rmin[row])
        if b == 0 or L == 0:
            return np.full(L, rmn, dtype=np.float32)
        return self._decode_one_vectorized(row, b)

    # ---------------------------------------------------------------- 批处理解码
    def decode_leaves_batched(self, rows: List[int]) -> dict:
        """
        批量解出一组叶子的残差，按位宽 b 分桶向量化。
        返回 {row: np.ndarray[L]}。
        """
        out = {}
        from collections import defaultdict
        bucket = defaultdict(list)        # b -> [row,...]
        for row in rows:
            bucket[int(self.leaf_bits[row])].append(row)
        for b, rws in bucket.items():
            if b == 0:
                for row in rws:
                    L = int(self.leaf_len[row])
                    out[row] = np.full(L, float(self.leaf_rmin[row]), dtype=np.float32)
                continue
            for row in rws:
                out[row] = self._decode_one_vectorized(row, b)
        return out

    def _decode_one_vectorized(self, row: int, b: int) -> np.ndarray:
        """向量化解一个叶子段（同位宽 b）：用 numpy 位运算批量取码。"""
        L = int(self.leaf_len[row])
        rmn = float(self.leaf_rmin[row])
        base = int(self.leaf_bitpos[row])
        # 每个点的起始比特位置
        starts = base + np.arange(L, dtype=np.int64) * b
        # 取每个码的 b 个比特：构造 [L, b] 的比特位置矩阵
        bit_idx = starts[:, None] + np.arange(b, dtype=np.int64)[None, :]   # [L, b]
        byte_idx = bit_idx >> 3
        bit_in_byte = 7 - (bit_idx & 7)
        bits_val = (self.bitstream[byte_idx] >> bit_in_byte) & 1            # [L, b]
        # MSB-first 组装：weight = 2^(b-1 .. 0)
        weights = (1 << np.arange(b - 1, -1, -1, dtype=np.int64))[None, :]  # [1, b]
        q = (bits_val.astype(np.int64) * weights).sum(axis=1)               # [L]
        return (rmn + q * self.step).astype(np.float32)

    # ---------------------------------------------------------------- 统计
    def total_bytes(self) -> dict:
        n = len(self.leaf_bits) if self.leaf_bits is not None else 0
        header = n * 3                      # 每叶子 rmin(2B fp16) + bits(1B)
        body = len(self.bitstream) if self.bitstream is not None else 0
        return {'num_leaves': n, 'header_bytes': header,
                'bitstream_bytes': body, 'total_bytes': header + body}
