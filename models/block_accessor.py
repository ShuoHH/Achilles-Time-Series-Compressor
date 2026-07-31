"""
BlockAccessor: 基于 BlockCodec 的随机访问读取器（与数据结构解耦）。

职责分离：
  - BlockCodec  ：纯数据结构（结构码 + 定长系数池 + 残差 meta + O(1) 寻址）
  - BlockAccessor：访问逻辑（单点 / 区间 / 批量），持有 codec + decode/residual 回调

访问路径（统一 base + patch + 残差）：
  query_point(t):
    block_id = t // base_block_size                  # O(1)
    leaf_idx = codec.locate(t)                       # O(1) 隐式树定位
    seg = decode_block(left,right)[切片]             # base 预测
        + Φ·δ(coeff_pool[row])                       # patch 修正(统一 synthesize)
        + 残差[row]                                   # EDWB(可选)
    return seg[t - leaf_start]

  query_range / query_batch：覆盖/去重叶子后批量解码再散射。

batch 友好：同段长叶子可分桶一次 GEMM（reconstruct_patch_batched 提供）。
decode_fn / residual_fn 由外部注入，Accessor 不依赖 exp / 训练代码。
"""

from typing import Callable, Optional, Dict, Tuple, List

import torch

from models.block_codec import BlockCodec, FourierCoeffs


class BlockAccessor:
    """
    随机访问读取器。

    Args:
        codec: 已 finalize 的 BlockCodec
        decode_fn: base 预测回调，签名
            decode_fn(left_id, right_id, block_len, block_start) -> Tensor[block_len]
            返回整块的冻结 base 预测（解码整块，访问时切片）。
        residual_fn: 可选 EDWB 残差回调，签名
            residual_fn(global_leaf_row, seg_len) -> Tensor[seg_len] | None
            未接时叶子只用 base + patch（误差不保证上界）。
        block_cache: 是否缓存整块 base 预测（区间/批量访问显著提速）。
    """

    def __init__(self, codec: BlockCodec,
                 decode_fn: Callable,
                 residual_fn: Optional[Callable] = None,
                 block_cache: bool = True):
        self.codec = codec
        self.decode_fn = decode_fn
        self.residual_fn = residual_fn
        self._use_cache = block_cache
        self._block_pred_cache: Dict[int, torch.Tensor] = {}
        self._resid_cache: Dict[int, torch.Tensor] = {}   # row -> 残差张量(目标设备，解一次复用)

    def _get_residual(self, row: int, L: int, device) -> Optional[torch.Tensor]:
        """取第 row 个叶子残差（缓存到目标设备，避免重复解码 + 重复 CPU→GPU 搬运）。"""
        if self.residual_fn is None:
            return None
        cached = self._resid_cache.get(row)
        if cached is not None:
            return cached
        r = self.residual_fn(row, L)
        if r is None:
            return None
        if r.device != device:
            r = r.to(device)
        if self._use_cache:
            self._resid_cache[row] = r
        return r

    # ----- 内部 -----
    def _block_pred(self, block_id: int) -> torch.Tensor:
        """整块 base 预测（带缓存，块内多点/区间复用，摊薄 decoder 开销）。"""
        if self._use_cache and block_id in self._block_pred_cache:
            return self._block_pred_cache[block_id]
        rec = self.codec.blocks[block_id]
        pred = self.decode_fn(rec.left_id, rec.right_id, rec.block_len, rec.block_start)
        if self._use_cache:
            self._block_pred_cache[block_id] = pred
        return pred

    def _leaf_recon(self, block_id: int, leaf_idx: int) -> Tuple[torch.Tensor, int]:
        """重建单个叶子波形 = base 切片 + Φ·δ + 残差。返回 (seg, leaf_start)。"""
        codec = self.codec
        rec = codec.blocks[block_id]
        lf = rec.leaves[leaf_idx]
        block_pred = self._block_pred(block_id)
        lo = lf.start - rec.block_start
        hi = min(lf.end - rec.block_start, block_pred.shape[0])
        L = hi - lo
        seg = block_pred[lo:lo + L].clone()
        row = codec.leaf_global_index(block_id, leaf_idx)
        delta = codec._coeff_pool[row]
        if delta.abs().sum() > 0:
            seg = seg + FourierCoeffs.synthesize(delta, L, codec.K_fixed)
        if self.residual_fn is not None:
            r = self._get_residual(row, L, seg.device)
            if r is not None:
                m = min(seg.shape[0], r.shape[0])
                if m > 0:
                    seg[:m] = seg[:m] + r[:m]
        return seg, lf.start

    def clear_cache(self):
        self._block_pred_cache.clear()
        self._resid_cache.clear()

    # ----- 单点 -----
    def query_point(self, t: int) -> float:
        codec = self.codec
        block_id, row, (s, e) = codec.locate(int(t))
        leaf_idx = row - codec._block_offset[block_id]
        seg, leaf_start = self._leaf_recon(block_id, leaf_idx)
        idx = min(int(t) - leaf_start, seg.shape[0]-1)
        return float(seg[idx])

    # ----- 区间 -----
    def query_range(self, t_start: int, t_end: int) -> torch.Tensor:
        if t_end <= t_start:
            return torch.empty(0)
        codec = self.codec
        out = torch.zeros(t_end - t_start)
        t = t_start
        while t < t_end:
            block_id, row, (s, e) = codec.locate(t)
            leaf_idx = row - codec._block_offset[block_id]
            seg, leaf_start = self._leaf_recon(block_id, leaf_idx)
            a = max(s, t_start)
            b = min(e, t_end)
            b = min(b, leaf_start + seg.shape[0])
            out[a - t_start:b - t_start] = seg[a - leaf_start:b - leaf_start]
            t = e
        return out

    # ----- 批量散点 -----
    def query_batch(self, times) -> torch.Tensor:
        codec = self.codec
        out = torch.zeros(len(times))
        leaf_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, int]] = {}
        for i, t in enumerate(times):
            ti = int(t)
            block_id, row, (s, e) = codec.locate(ti)
            leaf_idx = row - codec._block_offset[block_id]
            key = (block_id, leaf_idx)
            if key not in leaf_cache:
                leaf_cache[key] = self._leaf_recon(block_id, leaf_idx)
            seg, leaf_start = leaf_cache[key]
            idx = min(ti - leaf_start, seg.shape[0]-1)
            out[i] = seg[idx]
        return out

    # ----- 全量重建 -----
    def decompress_all(self) -> torch.Tensor:
        codec = self.codec
        T = codec.base_block_size * len(codec.blocks)  # 近似；末块可能短
        # 用实际块边界拼接
        max_t = max(r.block_start + r.block_len for r in codec.blocks.values())
        out = torch.zeros(max_t)
        for bid, rec in codec.blocks.items():
            for li in range(len(rec.leaves)):
                seg, leaf_start = self._leaf_recon(bid, li)
                out[leaf_start:leaf_start + seg.shape[0]] = seg
        return out

    # =========================================================================
    # 快路径：分桶 GEMM（前向次数 ∝ 唯一块数 / 段数，而非查询数）
    #
    # 需要注入批量基预测回调 decode_blocks_fn：
    #   decode_blocks_fn(left_ids[K], right_ids[K], block_len, offsets[K]) -> [K, block_len]
    # 同块长一组，一次 decoder.decode_batch（一次 GEMM）。
    # =========================================================================

    def attach_batched_decode(self, decode_blocks_fn):
        """注入批量基预测回调（同块长一次 GEMM）。"""
        self.decode_blocks_fn = decode_blocks_fn
        return self

    def _reconstruct_blocks(self, block_ids):
        """批量重建给定块完整波形（base 分桶GEMM + patch 分桶GEMM + 残差）。"""
        from collections import defaultdict
        codec = self.codec
        block_ids = list(dict.fromkeys(int(b) for b in block_ids))   # 去重保序

        # 1) base：按 block_len 分桶，每桶一次 decode_batch
        len_bucket = defaultdict(list)
        for bid in block_ids:
            len_bucket[codec.blocks[bid].block_len].append(bid)
        block_pred = {}
        dev = codec._coeff_pool.device if codec._coeff_pool is not None else None
        for blen, bids in len_bucket.items():
            recs = [codec.blocks[b] for b in bids]
            left = torch.tensor([r.left_id for r in recs], device=dev)
            right = torch.tensor([r.right_id for r in recs], device=dev)
            offs = torch.tensor([r.block_start % codec.base_block_size for r in recs], device=dev)
            preds = self.decode_blocks_fn(left, right, blen, offs)   # [K, blen]
            for i, b in enumerate(bids):
                block_pred[b] = preds[i].clone()

        # 2) patch：覆盖叶子按 seg_len 分桶，每桶一次 GEMM
        seg_bucket = defaultdict(list)
        for bid in block_ids:
            rec = codec.blocks[bid]
            base_off = codec._block_offset[bid]
            for li, lf in enumerate(rec.leaves):
                delta = codec._coeff_pool[base_off + li]
                if delta.abs().sum() > 0:
                    seg_bucket[lf.end - lf.start].append((bid, lf.start - rec.block_start, delta))
        for seg_len, items in seg_bucket.items():
            mat = torch.stack([it[2] for it in items], dim=0)
            recon = FourierCoeffs.synthesize(mat, seg_len, codec.K_fixed)
            for i, (bid, lo, _) in enumerate(items):
                block_pred[bid][lo:lo + seg_len] += recon[i]

        # 3) 残差（可选）
        if self.residual_fn is not None:
            for bid in block_ids:
                rec = codec.blocks[bid]
                base_off = codec._block_offset[bid]
                for li, lf in enumerate(rec.leaves):
                    r = self.residual_fn(base_off + li, lf.end - lf.start)
                    if r is not None:
                        lo = lf.start - rec.block_start
                        if r.device != block_pred[bid].device:
                            r = r.to(block_pred[bid].device)
                        avail = block_pred[bid].shape[0] - lo
                        m = min(int(r.shape[0]), avail)
                        if m > 0:
                            block_pred[bid][lo:lo + m] += r[:m]
        return block_pred

    def query_batch_fast(self, times) -> torch.Tensor:
        """随机点快路径：去重块 → 分桶GEMM → 散射取点。"""
        codec = self.codec
        times = [int(t) for t in times]
        bids = [t // codec.base_block_size for t in times]
        preds = self._reconstruct_blocks(bids)
        out = torch.zeros(len(times))
        for i, t in enumerate(times):
            rec = codec.blocks[t // codec.base_block_size]
            out[i] = preds[t // codec.base_block_size][t - rec.block_start]
        return out

    def query_range_fast(self, t_start: int, t_end: int) -> torch.Tensor:
        """随机段快路径：段跨到的块整体分桶GEMM → 拼接裁剪。"""
        if t_end <= t_start:
            return torch.empty(0)
        codec = self.codec
        bbs = codec.base_block_size
        bids = list(range(t_start // bbs, (t_end - 1) // bbs + 1))
        preds = self._reconstruct_blocks(bids)
        out = torch.zeros(t_end - t_start)
        for bid in bids:
            rec = codec.blocks[bid]
            bs = rec.block_start
            a = max(bs, t_start)
            b = min(bs + rec.block_len, t_end)
            out[a - t_start:b - t_start] = preds[bid][a - bs:b - bs]
        return out

    def decompress_all_fast(self) -> torch.Tensor:
        """全量解压快路径：所有块分桶GEMM一次性重建。"""
        codec = self.codec
        preds = self._reconstruct_blocks(list(codec.blocks.keys()))
        max_t = max(r.block_start + r.block_len for r in codec.blocks.values())
        out = torch.zeros(max_t)
        for bid, rec in codec.blocks.items():
            out[rec.block_start:rec.block_start + rec.block_len] = preds[bid]
        return out
