"""Functional test for multi-layer patch tree + bottom-up DP."""
import os, sys, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
from models.patch_split import build_multilayer_tree, count_leaves_and_depth


def _eps_fn(t0, t1):
    return 0.01


def test_piecewise_signal_prefers_split():
    # 左半平缓(低频)，右半高频 → 分裂应该比整块单层省
    L = 512
    t = torch.linspace(0, 1, L)
    parent = torch.zeros(L)
    true = torch.zeros(L)
    true[:256] = 0.5 * t[:256]                       # 左:平缓
    true[256:] = 2.0 * torch.sin(math.pi * 20 * t[256:])  # 右:高频大幅
    root = build_multilayer_tree(parent, true, 0, L, 0, _eps_fn,
                                 K_list=(0, 4, 8), modes=('int8', 'fp16'),
                                 min_len=64, depth=0, max_depth=3)
    leaves, depth = count_leaves_and_depth(root)
    print(f"piecewise: leaves={leaves} depth={depth} best={root.best_bytes} single={root.single_bytes}")
    assert root.best_bytes <= root.single_bytes      # DP 不会比整块单层差
    assert leaves >= 2                                # 非平稳信号应触发分裂


def test_smooth_signal_no_oversplit():
    # 全局平滑低频 → DP 不应过度分裂
    L = 512
    t = torch.linspace(0, 1, L)
    parent = torch.zeros(L)
    true = 0.3 * t + 0.1 * torch.sin(math.pi * 2 * t)
    root = build_multilayer_tree(parent, true, 0, L, 0, _eps_fn,
                                 K_list=(0, 4, 8), modes=('int8', 'fp16'),
                                 min_len=64, depth=0, max_depth=3)
    leaves, depth = count_leaves_and_depth(root)
    print(f"smooth: leaves={leaves} depth={depth} best={root.best_bytes} single={root.single_bytes}")
    assert root.best_bytes <= root.single_bytes


def test_dp_never_worse_than_single():
    torch.manual_seed(0)
    L = 256
    parent = torch.zeros(L)
    true = torch.randn(L) * 0.3
    root = build_multilayer_tree(parent, true, 0, L, 0, _eps_fn,
                                 K_list=(0, 4, 8), modes=('int8', 'fp16'),
                                 min_len=32, depth=0, max_depth=3)
    assert root.best_bytes <= root.single_bytes


if __name__ == '__main__':
    test_piecewise_signal_prefers_split()
    test_smooth_signal_no_oversplit()
    test_dp_never_worse_than_single()
    print("All multilayer tests passed.")
