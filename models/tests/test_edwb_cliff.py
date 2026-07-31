"""
Minimal tests for bitwidth-cliff split helpers.

Verifies that the cliff stats reuse the REAL EDWB formula (FallbackDict) and
that the bitwidth threshold / one-bit payload-saving behaviour matches spec.

Run:  pytest cross_models/tests/test_edwb_cliff.py -q
"""

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from models.fallback_dict import FallbackDict


# --- Reference reimplementation of the cliff stats (mirrors _edwb_cliff_stats) ---
# We keep this standalone so the test does not require building the full Exp class.
def edwb_cliff_stats(span, length, eps):
    bits = FallbackDict.compute_bits(span, eps)
    cost = FallbackDict.estimate_bitwidth_cost(span, length, eps)
    if bits <= 0:
        return {'bits': bits, 'cost': cost, 'next_threshold': 0.0,
                'needed_drop': 0.0, 'needed_drop_ratio': 0.0, 'one_bit_saving': 0}
    next_threshold = 2.0 * eps * (2 ** (bits - 1))
    needed_drop = max(0.0, span - next_threshold)
    needed_drop_ratio = needed_drop / max(span, 1e-12)
    payload_b = math.ceil(length * bits / 8)
    payload_b1 = math.ceil(length * (bits - 1) / 8) if bits - 1 > 0 else 0
    return {'bits': bits, 'cost': cost, 'next_threshold': next_threshold,
            'needed_drop': needed_drop, 'needed_drop_ratio': needed_drop_ratio,
            'one_bit_saving': payload_b - payload_b1}


def test_compute_bits_matches_formula():
    eps = 0.01
    # span just above 2*eps*2^(b-1) -> bits = b ; just below -> bits = b-1
    for b in range(1, 8):
        thr = 2 * eps * (2 ** (b - 1))      # boundary: span<=thr -> b-1 ; >thr -> b
        assert FallbackDict.compute_bits(thr * 0.999, eps) == b - 1
        assert FallbackDict.compute_bits(thr * 1.001, eps) == b


def test_zero_bit():
    eps = 0.01
    # span <= 2*eps -> 0 bits (0-bit miracle)
    assert FallbackDict.compute_bits(2 * eps * 0.5, eps) == 0
    s = edwb_cliff_stats(2 * eps * 0.5, 256, eps)
    assert s['bits'] == 0
    assert s['one_bit_saving'] == 0


def test_one_bit_payload_saving_512():
    # block_len=512, dropping 1 bit saves 512/8 = 64 bytes
    eps = 0.01
    # choose span so bits=5 (well above threshold, span/2eps in (16,32])
    span = 2 * eps * 20  # q=20 -> bits=ceil(log2(20))=5
    s = edwb_cliff_stats(span, 512, eps)
    assert s['bits'] == 5
    assert s['one_bit_saving'] == 64


def test_one_bit_payload_saving_256():
    eps = 0.01
    span = 2 * eps * 20  # bits=5
    s = edwb_cliff_stats(span, 256, eps)
    assert s['bits'] == 5
    assert s['one_bit_saving'] == 32


def test_needed_drop_ratio_smaller_when_closer_to_cliff():
    eps = 0.01
    bits = 5
    thr = 2 * eps * (2 ** (bits - 1))   # next threshold for dropping to bits-1
    near = edwb_cliff_stats(thr * 1.01, 256, eps)    # just above threshold
    far = edwb_cliff_stats(thr * 1.9, 256, eps)      # far above threshold
    assert near['bits'] == bits and far['bits'] == bits
    assert near['needed_drop_ratio'] < far['needed_drop_ratio']


if __name__ == '__main__':
    test_compute_bits_matches_formula()
    test_zero_bit()
    test_one_bit_payload_saving_512()
    test_one_bit_payload_saving_256()
    test_needed_drop_ratio_smaller_when_closer_to_cliff()
    print("All cliff tests passed.")
