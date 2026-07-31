"""ResidualCodec 测试：误差界 ≤ ε、单点==整段==批处理、O(1) 定位自洽。"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
from models.residual_codec import ResidualCodec


def make_leaves(seed=0):
    rng = np.random.RandomState(seed)
    leaves = [
        rng.normal(0, 1.0, size=50).astype(np.float32),    # 大 span
        rng.normal(0, 0.001, size=25).astype(np.float32),  # 小 span(可能 0-bit)
        rng.uniform(-0.5, 0.5, size=25).astype(np.float32),
        np.zeros(30, dtype=np.float32),                    # 全零 → 0-bit
        rng.normal(0, 5.0, size=64).astype(np.float32),    # 很大 span
    ]
    return leaves


def test_error_bound():
    eps = 0.05
    leaves = make_leaves()
    codec = ResidualCodec(eps).encode(leaves)
    for row, r in enumerate(leaves):
        rec = codec.decode_leaf(row)
        err = np.abs(rec - r).max() if r.size else 0.0
        assert err <= eps + 1e-6, f"leaf {row} err {err} > eps {eps}"
    print("error bound <= eps OK")


def test_point_eq_leaf():
    eps = 0.05
    leaves = make_leaves(1)
    codec = ResidualCodec(eps).encode(leaves)
    for row, r in enumerate(leaves):
        full = codec.decode_leaf(row)
        for j in range(r.size):
            assert abs(codec.decode_point(row, j) - full[j]) < 1e-6, (row, j)
    print("decode_point == decode_leaf OK")


def test_batched_eq_single():
    eps = 0.03
    leaves = make_leaves(2)
    codec = ResidualCodec(eps).encode(leaves)
    rows = list(range(len(leaves)))
    batched = codec.decode_leaves_batched(rows)
    for row in rows:
        full = codec.decode_leaf(row)
        assert np.allclose(batched[row], full, atol=1e-6), row
    print("batched == single OK")


def test_bits_per_leaf_independent():
    """验证位宽是 per-leaf 独立的（大 span 段位宽 > 小 span 段）。"""
    eps = 0.05
    leaves = make_leaves(3)
    codec = ResidualCodec(eps).encode(leaves)
    bits = codec.leaf_bits
    # leaf3 全零 → 0 bit；leaf4 大 span → 位宽最大
    assert bits[3] == 0, bits
    assert bits[4] == bits.max(), bits
    print(f"per-leaf bits OK: {list(bits)}")


if __name__ == '__main__':
    test_error_bound()
    test_point_eq_leaf()
    test_batched_eq_single()
    test_bits_per_leaf_independent()
    print("All residual-codec tests passed.")
