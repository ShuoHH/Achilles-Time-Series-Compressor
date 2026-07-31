"""BlockCodec 测试：结构码枚举、叶子边界、O(1) 定位、批处理重建。"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
from models.block_codec import (
    enumerate_partitions, TreeCodebook, BlockCodec, BlockRecord, LeafRecord,
    FourierCoeffs,
)


def test_depth2_has_5_codes():
    # base=100, min=25 → depth=2 → 用户说的 5 种划分
    parts = enumerate_partitions(2)
    print("depth2 partitions:", parts)
    assert len(parts) == 5, f"expected 5, got {len(parts)}"
    # 不分裂必须是 code 0
    assert parts[0] == (1.0,)


def test_depth3_fits_one_byte():
    # base=256, min=32 → depth=3 → 26 种，1 字节够
    cb = TreeCodebook(3)
    assert cb.n_codes == 26, cb.n_codes
    assert cb.n_codes <= 256


def test_leaf_bounds():
    cb = TreeCodebook(2)
    # 找到 [50][25][25] 这种划分的 code
    target = (0.5, 0.25, 0.25)
    code = cb.code_of[target]
    bounds = cb.leaf_bounds(code, block_start=0, block_len=100)
    assert bounds == [(0, 50), (50, 75), (75, 100)], bounds


def test_locate_leaf():
    cb = TreeCodebook(2)
    code = cb.code_of[(0.5, 0.25, 0.25)]   # [0,50)[50,75)[75,100)
    assert cb.locate_leaf(code, 0, 100, 10) == 0    # 在 [0,50)
    assert cb.locate_leaf(code, 0, 100, 60) == 1    # 在 [50,75)
    assert cb.locate_leaf(code, 0, 100, 90) == 2    # 在 [75,100)


def test_codec_locate_and_offset():
    K = 8
    codec = BlockCodec(base_block_size=100, min_resolution=25, K_fixed=K)
    cb = codec.codebook
    # 块0: 不分裂(code 0, 1 叶子)
    code0 = cb.code_of[(1.0,)]
    rec0 = BlockRecord(block_id=0, block_start=0, block_len=100, code=code0,
                       leaves=[LeafRecord(0, 100, torch.zeros(K + 2), coeff_bytes=(K+2))])
    # 块1: [50][25][25] (3 叶子)
    code1 = cb.code_of[(0.5, 0.25, 0.25)]
    rec1 = BlockRecord(block_id=1, block_start=100, block_len=100, code=code1,
                       leaves=[LeafRecord(100, 150, torch.ones(K + 2), coeff_bytes=(K+2)),
                               LeafRecord(150, 175, torch.ones(K + 2) * 2, coeff_bytes=(K+2)),
                               LeafRecord(175, 200, torch.ones(K + 2) * 3, coeff_bytes=(K+2))])
    codec.add_block(rec0)
    codec.add_block(rec1)
    codec.finalize()

    # block_offset: 块0 从行0, 块1 从行1
    assert codec._block_offset[0] == 0
    assert codec._block_offset[1] == 1

    # locate t=170 → 块1, 叶子1 ([150,175)), 全局行 = 1 + 1 = 2
    bid, row, (s, e) = codec.locate(170)
    assert bid == 1 and (s, e) == (150, 175) and row == 2, (bid, row, s, e)
    # 该行系数应为全 2
    assert torch.allclose(codec._coeff_pool[2], torch.ones(K + 2) * 2)


def test_batched_reconstruct_matches():
    K = 4
    codec = BlockCodec(base_block_size=100, min_resolution=25, K_fixed=K)
    cb = codec.codebook
    torch.manual_seed(0)
    code1 = cb.code_of[(0.5, 0.25, 0.25)]
    leaves = [
        LeafRecord(100, 150, torch.randn(K + 2), coeff_bytes=K+2),
        LeafRecord(150, 175, torch.randn(K + 2), coeff_bytes=K+2),
        LeafRecord(175, 200, torch.randn(K + 2), coeff_bytes=K+2),
    ]
    codec.add_block(BlockRecord(1, 100, 100, code1, leaves))
    codec.finalize()
    out = codec.reconstruct_patch_batched()
    # 校验：每个叶子用统一 FourierCoeffs.synthesize 单独算应一致
    from models.block_codec import FourierCoeffs
    for row, meta in enumerate(codec._leaf_meta):
        seg_len = meta['end'] - meta['start']
        delta = codec._coeff_pool[row]
        ref = FourierCoeffs.synthesize(delta, seg_len, K)
        assert torch.allclose(out[row], ref, atol=1e-5), row
    print("batched patch reconstruct OK")


def test_depth4_uses_2_bytes():
    # base=512, min=32 → depth=4 → 677 种 → 需 2 字节
    cb = TreeCodebook(4)
    assert cb.n_codes == 677, cb.n_codes
    assert cb.code_bytes == 2


def test_synthesize_batch_equals_single():
    from models.block_codec import FourierCoeffs
    K, L = 4, 64
    torch.manual_seed(1)
    deltas = torch.randn(5, K + 2)
    batch = FourierCoeffs.synthesize(deltas, L, K)          # [5, L]
    for i in range(5):
        single = FourierCoeffs.synthesize(deltas[i], L, K)  # [L]
        assert torch.allclose(batch[i], single, atol=1e-5), i
    print("synthesize batch == single OK")


if __name__ == '__main__':
    test_depth2_has_5_codes()
    test_depth3_fits_one_byte()
    test_depth4_uses_2_bytes()
    test_synthesize_batch_equals_single()
    test_leaf_bounds()
    test_locate_leaf()
    test_codec_locate_and_offset()
    test_batched_reconstruct_matches()
    print("All block_codec tests passed.")
