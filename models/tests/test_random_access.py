"""随机访问测试（BlockAccessor）：query_point / query_range / query_batch 自洽。"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
from models.block_codec import (
    BlockCodec, BlockRecord, LeafRecord, FourierCoeffs,
)
from models.block_accessor import BlockAccessor


def make_accessor():
    K = 4
    Kp = K + 2
    codec = BlockCodec(base_block_size=100, min_resolution=25, K_fixed=K)
    cb = codec.codebook
    torch.manual_seed(0)
    recs = [
        BlockRecord(0, 0, 100, cb.code_of[(1.0,)],
                    [LeafRecord(0, 100, torch.zeros(Kp), coeff_bytes=0)],
                    left_id=0, right_id=1),
        BlockRecord(1, 100, 100, cb.code_of[(0.5, 0.5)],
                    [LeafRecord(100, 150, torch.randn(Kp), coeff_bytes=Kp),
                     LeafRecord(150, 200, torch.randn(Kp), coeff_bytes=Kp)],
                    left_id=1, right_id=2),
        BlockRecord(2, 200, 100, cb.code_of[(0.5, 0.25, 0.25)],
                    [LeafRecord(200, 250, torch.randn(Kp), coeff_bytes=Kp),
                     LeafRecord(250, 275, torch.randn(Kp), coeff_bytes=Kp),
                     LeafRecord(275, 300, torch.randn(Kp), coeff_bytes=Kp)],
                    left_id=2, right_id=3),
    ]
    for r in recs:
        codec.add_block(r)
    codec.finalize()

    def decode_fn(left_id, right_id, block_len, block_start):
        t = torch.linspace(0, 1, block_len)
        return left_id + t   # 每块不同基线
    acc = BlockAccessor(codec, decode_fn)

    # 批量基预测回调（与单块一致：left_id + linspace）
    def decode_blocks_fn(left_ids, right_ids, block_len, offsets):
        t = torch.linspace(0, 1, block_len)
        return left_ids.float().unsqueeze(1) + t.unsqueeze(0)   # [K, block_len]
    acc.attach_batched_decode(decode_blocks_fn)
    return acc, codec, K


def ref_leaf(codec, K, bid, leaf_idx):
    rec = codec.blocks[bid]
    lf = rec.leaves[leaf_idx]
    L = lf.end - lf.start
    t = torch.linspace(0, 1, rec.block_len)
    base = (rec.left_id + t)[lf.start - rec.block_start: lf.start - rec.block_start + L].clone()
    row = codec.leaf_global_index(bid, leaf_idx)
    delta = codec._coeff_pool[row]
    if delta.abs().sum() > 0:
        base = base + FourierCoeffs.synthesize(delta, L, K)
    return base


def test_point():
    acc, codec, K = make_accessor()
    for t in [10, 120, 170, 210, 260, 290]:
        bid, row, (s, e) = codec.locate(t)
        li = row - codec._block_offset[bid]
        ref = ref_leaf(codec, K, bid, li)
        assert abs(acc.query_point(t) - ref[t - s].item()) < 1e-5, t
    print("query_point OK")


def test_range():
    acc, codec, K = make_accessor()
    out = acc.query_range(120, 290)
    assert out.shape[0] == 170
    for t in [120, 150, 199, 200, 260, 289]:
        assert abs(out[t - 120].item() - acc.query_point(t)) < 1e-5, t
    print("query_range OK")


def test_batch():
    acc, codec, K = make_accessor()
    times = [290, 10, 170, 170, 250, 120]
    out = acc.query_batch(times)
    for i, t in enumerate(times):
        assert abs(out[i].item() - acc.query_point(t)) < 1e-5, t
    print("query_batch OK (dedup)")


def test_decompress_all_matches_points():
    acc, codec, K = make_accessor()
    full = acc.decompress_all()
    for t in [5, 130, 199, 240, 299]:
        assert abs(full[t].item() - acc.query_point(t)) < 1e-5, t
    print("decompress_all OK")


def test_fast_equals_naive():
    acc, codec, K = make_accessor()
    # 点：fast == naive
    times = [290, 10, 170, 250, 120, 199]
    f = acc.query_batch_fast(times)
    n = acc.query_batch(times)
    assert torch.allclose(f, n, atol=1e-5), (f, n)
    # 段：fast == naive
    fr = acc.query_range_fast(120, 290)
    nr = acc.query_range(120, 290)
    assert torch.allclose(fr, nr, atol=1e-5)
    # 全量：fast == naive
    fa = acc.decompress_all_fast()
    na = acc.decompress_all()
    assert torch.allclose(fa, na, atol=1e-5)
    print("fast == naive (point/range/all) OK")


if __name__ == '__main__':
    test_point()
    test_range()
    test_batch()
    test_decompress_all_matches_points()
    test_fast_equals_naive()
    print("All random-access tests passed.")
