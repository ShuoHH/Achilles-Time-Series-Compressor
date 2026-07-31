"""block_payload 测试：serialize→deserialize 往返一致 + payload 尺寸==解析式计数。"""
import os, sys, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import torch

from models.block_codec import BlockCodec, BlockRecord, LeafRecord
from models.residual_codec import ResidualCodec
from models.patch_split import PATCH_META_BYTES, _INT8_HEADER
from models import block_payload as BP


def build_codec():
    K = 4; Kp = K + 2
    codec = BlockCodec(base_block_size=100, min_resolution=25, K_fixed=K)
    cb = codec.codebook
    torch.manual_seed(0)
    # block0 未分裂(1叶子,无系数); block1 [50,50]; block2 [50,25,25]
    recs = [
        BlockRecord(0, 0, 100, cb.code_of[(1.0,)],
                    [LeafRecord(0, 100, torch.zeros(Kp), coeff_bytes=0)],
                    left_id=0, right_id=1),
        BlockRecord(1, 100, 100, cb.code_of[(0.5, 0.5)],
                    [LeafRecord(100, 150, torch.rand(Kp) * 2 - 1, coeff_bytes=Kp + _INT8_HEADER),
                     LeafRecord(150, 200, torch.rand(Kp) * 2 - 1, coeff_bytes=Kp + _INT8_HEADER)],
                    left_id=1, right_id=2),
        BlockRecord(2, 200, 100, cb.code_of[(0.5, 0.25, 0.25)],
                    [LeafRecord(200, 250, torch.rand(Kp) * 2 - 1, coeff_bytes=Kp + _INT8_HEADER),
                     LeafRecord(250, 275, torch.rand(Kp) * 2 - 1, coeff_bytes=Kp + _INT8_HEADER),
                     LeafRecord(275, 300, torch.rand(Kp) * 2 - 1, coeff_bytes=Kp + _INT8_HEADER)],
                    left_id=2, right_id=3),
    ]
    for r in recs:
        codec.add_block(r)
    codec.finalize()

    # 残差：按全局叶子顺序，各叶子长度 [100,50,50,50,25,25]，构造不同 span
    eps = 0.05
    rng = np.random.RandomState(1)
    lens = [100, 50, 50, 50, 25, 25]
    leaf_res = []
    for i, L in enumerate(lens):
        if i == 3:
            leaf_res.append(np.zeros(L, dtype=np.float32))          # 0-bit
        else:
            leaf_res.append((rng.rand(L).astype(np.float32) - 0.5) * (0.3 * (i + 1)))
    rc = ResidualCodec(eps).encode(leaf_res)
    codec.attach_residual_codec(rc)
    return codec, K


def test_roundtrip_and_size():
    codec, K = build_codec()
    rc = codec.residual_codec
    Kp = K + 2

    blob = BP.serialize(codec)
    out = BP.deserialize(blob)

    # --- 系数往返（int8 精度容差）---
    cp0 = codec._coeff_pool
    cp1 = out['coeff_pool']
    assert cp1.shape == cp0.shape, (cp1.shape, cp0.shape)
    max_coeff_err = (cp1 - cp0).abs().max().item()
    assert max_coeff_err < 0.02, f"coeff roundtrip err {max_coeff_err}"

    # --- 残差往返：q 精确保留，误差仅来自 res_rmin 的 fp16 存储精度(解析式口径就是 2B)---
    for row in range(out['num_leaves']):
        ref = rc.decode_leaf(row)                          # 内存 float32 rmin
        got = out['residual'][row]                          # payload fp16 rmin
        err = np.abs(ref - got).max() if ref.size else 0.0
        assert err < 1e-3, f"leaf {row} residual err {err}"   # fp16(rmin) 舍入 << eps

    # --- payload 尺寸 == 解析式计数（残差 + 系数 + patch meta）---
    expected = 0
    for row in range(len(rc.leaf_bits)):
        b = int(rc.leaf_bits[row]); L = int(rc.leaf_len[row])
        expected += BP._RES_HEADER + math.ceil(L * b / 8)      # 3 + ceil(L*b/8)
    # 被 patch 的叶子(block1,2 共 5 个)带系数; 被 patch 的块(2 个)各 2B meta
    n_patched_leaves = 2 + 3
    n_patched_blocks = 2
    expected += n_patched_leaves * (Kp + _INT8_HEADER)
    expected += n_patched_blocks * PATCH_META_BYTES

    assert BP.payload_size(blob) == expected, (BP.payload_size(blob), expected)
    print(f"payload={BP.payload_size(blob)}B == analytical={expected}B  "
          f"coeff_err={max_coeff_err:.2e}  residual exact OK")


if __name__ == '__main__':
    test_roundtrip_and_size()
    print("All block_payload tests passed.")
