"""
EntropyResidualCodec: 残差的【熵编码】版本（对照 ResidualCodec 的 EDWB 定宽版）。

目的：给消融里的 `entropy` 变体测【真实全量解压吞吐】，而不是只估字节数。

与 EDWB(ResidualCodec) 的对照关系
--------------------------------------------------------------------------
  相同：残差用 bucket = 2ε 分桶量化，保证 |r - r̂| ≤ ε（误差有界口径一致）。
  不同：落盘/解码方式
    EDWB  : 每点固定 ceil(log2 桶数) bits，位流可 numpy 向量化批量解包 → 快、支持 O(1) 随机访问。
    熵编码: 每 leaf 的 bin 序列用 range coder（算术编码族）编码，解码【本质顺序】
            （每个符号依赖前一符号后的解码器状态），无法向量化、无法 O(1) 随机访问。
  → 本 codec 正是要把"熵编码顺序解码"的真实开销测出来，与 EDWB 向量化解码对照。

实现说明
--------------------------------------------------------------------------
  - Range coder：32-bit 无进位（Subbotin 风格），字节级输出，静态频率模型（per-leaf）。
  - 每个 leaf 独立：把该 leaf 的 bin 值 relabel 成稠密符号 0..K-1（残差集中 → K 很小），
    统计频率，range 编码符号序列。解码时逐符号还原 → 反 relabel → 反量化。
  - 压缩比不由本 codec 决定（消融用 entropy_cost 的香农下界报 ratio）；本 codec 只负责
    提供一个【正确且优化程度合理】的顺序解码器来测 decode 吞吐。

接口对齐 ResidualCodec：encode(leaf_residuals) / decode_leaf(row) / decode_leaves_batched(rows)。
"""

import numpy as np

_TOP = 1 << 24
_BOT = 1 << 16
_MASK = (1 << 32) - 1


# ============================================================ Range coder 核心
class _RangeEncoder:
    """32-bit 无进位 range 编码器（Subbotin 风格），字节级输出。"""

    def __init__(self):
        self.low = 0
        self.rng = _MASK
        self.out = bytearray()

    def encode(self, cum, freq, tot):
        # 依据累计频率把区间收窄到 [cum, cum+freq)/tot
        self.rng //= tot
        self.low = (self.low + cum * self.rng) & _MASK
        self.rng = (self.rng * freq) & _MASK
        self._renorm()

    def _renorm(self):
        while True:
            if (self.low ^ (self.low + self.rng)) & _MASK < _TOP:
                pass
            elif self.rng < _BOT:
                # 下溢：强制收缩 range，避免精度耗尽
                self.rng = ((-self.low) & (_BOT - 1))
            else:
                break
            self.out.append((self.low >> 24) & 0xFF)
            self.low = (self.low << 8) & _MASK
            self.rng = (self.rng << 8) & _MASK

    def finish(self):
        for _ in range(4):
            self.out.append((self.low >> 24) & 0xFF)
            self.low = (self.low << 8) & _MASK
        return bytes(self.out)


class _RangeDecoder:
    """与 _RangeEncoder 配对的解码器。"""

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.low = 0
        self.rng = _MASK
        self.code = 0
        for _ in range(4):
            self.code = ((self.code << 8) | self._next_byte()) & _MASK

    def _next_byte(self):
        if self.pos < len(self.data):
            b = self.data[self.pos]
            self.pos += 1
            return b
        return 0

    def get_freq(self, tot):
        """返回当前落在 [0, tot) 的目标累计频率值。"""
        self.rng //= tot
        return ((self.code - self.low) & _MASK) // self.rng

    def decode_update(self, cum, freq):
        self.low = (self.low + cum * self.rng) & _MASK
        self.rng = (self.rng * freq) & _MASK
        self._renorm()

    def _renorm(self):
        while True:
            if (self.low ^ (self.low + self.rng)) & _MASK < _TOP:
                pass
            elif self.rng < _BOT:
                self.rng = ((-self.low) & (_BOT - 1))
            else:
                break
            self.code = ((self.code << 8) | self._next_byte()) & _MASK
            self.low = (self.low << 8) & _MASK
            self.rng = (self.rng << 8) & _MASK


# ============================================================ 残差熵 codec
class EntropyResidualCodec:
    """残差熵编码器（per-leaf 静态频率 range coding）。仅用于 decode 吞吐测量。"""

    def __init__(self, eps: float):
        self.eps = float(eps)
        self.step = 2.0 * float(eps)
        # per-leaf 元数据
        self.leaf_rmin = None      # [n] float32
        self.leaf_len = None       # [n] int32
        self.leaf_syms = None      # list[np.ndarray] 每 leaf 的稠密符号→bin 值映射(u)
        self.leaf_freqs = None     # list[np.ndarray] 每 leaf 的符号频率(counts)
        self.leaf_streams = None   # list[bytes]      每 leaf 的 range 码流
        self._finalized = False

    # -------------------------------------------------------- 编码
    def encode(self, leaf_residuals):
        n = len(leaf_residuals)
        rmin = np.zeros(n, dtype=np.float32)
        llen = np.zeros(n, dtype=np.int32)
        syms_list, freqs_list, streams = [], [], []
        for i, r in enumerate(leaf_residuals):
            r = np.asarray(r, dtype=np.float64).reshape(-1)
            L = r.size
            llen[i] = L
            if L == 0:
                rmin[i] = 0.0
                syms_list.append(np.zeros(0, dtype=np.int64))
                freqs_list.append(np.zeros(0, dtype=np.int64))
                streams.append(b'')
                continue
            rmn = float(r.min())
            rmin[i] = np.float32(rmn)
            q = np.round((r - rmn) / self.step).astype(np.int64)
            q = np.clip(q, 0, None)
            # relabel 到稠密符号 0..K-1（残差集中 → K 小）
            u, inv = np.unique(q, return_inverse=True)   # u=distinct bins, inv=符号序列
            counts = np.bincount(inv, minlength=u.size).astype(np.int64)
            syms_list.append(u)
            freqs_list.append(counts)
            if u.size <= 1:
                streams.append(b'')   # 单符号：无需码流，解码直接填充
                continue
            # range 编码符号序列（静态模型：cumfreq 由 counts 得）
            cum = np.zeros(u.size + 1, dtype=np.int64)
            np.cumsum(counts, out=cum[1:])
            tot = int(cum[-1])
            enc = _RangeEncoder()
            cum_l = cum.tolist(); cnt_l = counts.tolist(); inv_l = inv.tolist()
            for s in inv_l:
                enc.encode(cum_l[s], cnt_l[s], tot)
            streams.append(enc.finish())
        self.leaf_rmin = rmin
        self.leaf_len = llen
        self.leaf_syms = syms_list
        self.leaf_freqs = freqs_list
        self.leaf_streams = streams
        self._finalized = True
        return self

    # -------------------------------------------------------- 解码（顺序，本质不可向量化）
    def decode_leaf(self, row: int) -> np.ndarray:
        row = int(row)
        L = int(self.leaf_len[row])
        rmn = float(self.leaf_rmin[row])
        if L == 0:
            return np.zeros(0, dtype=np.float32)
        u = self.leaf_syms[row]
        if u.size <= 1:
            # 单符号：全段同值
            val = rmn + (float(u[0]) * self.step if u.size == 1 else 0.0)
            return np.full(L, val, dtype=np.float32)
        counts = self.leaf_freqs[row]
        cum = np.zeros(u.size + 1, dtype=np.int64)
        np.cumsum(counts, out=cum[1:])
        tot = int(cum[-1])
        cum_l = cum.tolist(); cnt_l = counts.tolist(); u_l = u.tolist()
        K = u.size
        dec = _RangeDecoder(self.leaf_streams[row])
        out_q = np.empty(L, dtype=np.int64)
        for i in range(L):
            f = dec.get_freq(tot)
            # 找符号 s 使 cum[s] <= f < cum[s+1]（线性；残差符号少，K 小）
            s = 0
            while s + 1 < K and cum_l[s + 1] <= f:
                s += 1
            dec.decode_update(cum_l[s], cnt_l[s])
            out_q[i] = u_l[s]
        return (rmn + out_q * self.step).astype(np.float32)

    def decode_leaves_batched(self, rows) -> dict:
        # 熵编码无法真正批处理（顺序依赖）；逐 leaf 解，接口对齐 ResidualCodec。
        return {int(r): self.decode_leaf(int(r)) for r in rows}

    def total_bytes(self) -> dict:
        n = len(self.leaf_len) if self.leaf_len is not None else 0
        body = sum(len(s) for s in self.leaf_streams) if self.leaf_streams else 0
        return {'num_leaves': n, 'bitstream_bytes': body}


# ============================================================ 本地自测
if __name__ == '__main__':
    rng = np.random.RandomState(0)
    eps = 0.05
    codec = EntropyResidualCodec(eps)
    # 构造若干 leaf：集中在 0 附近的残差（典型分布）+ 少量离群
    leaves = []
    for _ in range(50):
        L = int(rng.randint(64, 512))
        r = rng.normal(0, 0.15, L)
        if rng.random_sample() < 0.3:
            r[rng.randint(0, L)] += rng.normal(0, 1.0)  # 离群
        leaves.append(r.astype(np.float32))
    codec.encode(leaves)
    ok = True
    for i, r in enumerate(leaves):
        rec = codec.decode_leaf(i)
        # 反量化值应满足 |r - rec| <= eps（量化误差有界）
        err = np.abs(r - rec).max()
        # 且 decode 出的量化 bin 应与 encode 时一致（roundtrip 精确）
        rmn = float(np.asarray(r, np.float64).min())
        q_enc = np.round((np.asarray(r, np.float64) - rmn) / (2 * eps)).astype(np.int64)
        q_enc = np.clip(q_enc, 0, None)
        q_dec = np.round((rec - rmn) / (2 * eps)).astype(np.int64)
        if not np.array_equal(q_enc, q_dec):
            ok = False
            print(f"[FAIL] leaf {i}: bin mismatch, maxerr={err:.4f}")
        elif err > eps + 1e-6:
            ok = False
            print(f"[FAIL] leaf {i}: err {err:.4f} > eps {eps}")
    print("ROUNDTRIP OK" if ok else "ROUNDTRIP FAILED")
