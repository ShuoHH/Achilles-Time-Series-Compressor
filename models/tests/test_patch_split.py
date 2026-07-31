"""
Functional tests for parent-anchored additive patch split.

Run:  python cross_models/tests/test_patch_split.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch

from models.patch_split import (
    build_patch_basis, quantize_coeffs, patch_coeff_bytes,
    fit_child_patch, fit_child_best, evaluate_parent_patch_split,
)
from models.fallback_dict import FallbackDict


def test_basis_shape_and_endpoints():
    B = build_patch_basis(256, 8)
    assert B.shape == (256, 10)               # K+2 columns
    # ramp endpoints: row0 = [1,0,...], last = [0,1,...]
    assert abs(B[0, 0].item() - 1.0) < 1e-6 and abs(B[0, 1].item()) < 1e-6
    assert abs(B[-1, 0].item()) < 1e-6 and abs(B[-1, 1].item() - 1.0) < 1e-6
    # sin(pi*k*t) endpoints are 0
    assert B[0, 2:].abs().max().item() < 1e-5
    assert B[-1, 2:].abs().max().item() < 1e-5


def test_coeff_bytes():
    # int8: (K+2)*1 + 4 header ; fp16: (K+2)*2
    assert patch_coeff_bytes(8, 'int8') == 10 + 4
    assert patch_coeff_bytes(8, 'fp16') == 20
    assert patch_coeff_bytes(0, 'int8') == 2 + 4


def test_lstsq_fits_linear_combo_exactly():
    # target that is exactly representable by the basis -> residual ~ 0
    L, K = 256, 8
    B = build_patch_basis(L, K)
    true_delta = torch.randn(K + 2)
    target = B @ true_delta
    fit = fit_child_patch(target, eps=0.01, K=K, mode='fp16')
    # fp16 quant of a clean target -> tiny residual span
    assert fit.residual_span < 1e-1


def test_byte_gate_accepts_when_residual_drops_enough():
    # parent residual with large span (needs many bits); patch removes a smooth
    # ramp+sin component so child residual span collapses -> bytes drop.
    L = 512
    eps = 0.01
    t = torch.linspace(0, 1, L)
    parent_pred = torch.zeros(L)
    # true = smooth low-freq signal exactly in patch space + tiny noise
    true_full = 3.0 * t + 1.5 * torch.sin(math.pi * 2 * t)
    res = evaluate_parent_patch_split(
        parent_pred, true_full, start=0, end=L, mid=L // 2,
        eps_parent=eps, eps_left=eps, eps_right=eps,
    )
    # parent residual span huge (=true span); after patch each child ~exact
    assert res.parent_bits > 0
    assert res.accept
    assert res.net_gain > 0


def test_byte_gate_rejects_pure_noise():
    # white noise can't be captured by a few smooth basis funcs ->
    # child residual span ~ parent span, but we pay patch coeff bytes -> reject.
    torch.manual_seed(0)
    L = 512
    eps = 0.001
    parent_pred = torch.zeros(L)
    true_full = torch.randn(L) * 0.5     # high-freq noise
    res = evaluate_parent_patch_split(
        parent_pred, true_full, start=0, end=L, mid=L // 2,
        eps_parent=eps, eps_left=eps, eps_right=eps,
    )
    # not strictly guaranteed, but for white noise the patch rarely pays off
    assert not res.accept


if __name__ == '__main__':
    test_basis_shape_and_endpoints()
    test_coeff_bytes()
    test_lstsq_fits_linear_combo_exactly()
    test_byte_gate_accepts_when_residual_drops_enough()
    test_byte_gate_rejects_pure_noise()
    print("All patch-split tests passed.")
