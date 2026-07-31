"""验证批量 GEMM 重建与逐块重建结果一致（定长/变长都适用）。"""
import os, sys, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
from models.patch_split import (
    PatchSplitManager, fit_child_fixedK, fit_child_best,
)


def _make_manager(fixed_K=None):
    mgr = PatchSplitManager()
    torch.manual_seed(0)
    block_preds = {}
    for bi in range(5):
        start = bi * 256
        end = start + 256
        pred = torch.randn(256)
        block_preds[(start, end)] = pred
        true = pred + torch.randn(256) * 0.2
        mid = start + 128
        segs = []
        for (s0, s1) in [(start, mid), (mid, end)]:
            l0, l1 = s0 - start, s1 - start
            target = true[l0:l1] - pred[l0:l1]
            if fixed_K is not None:
                fit = fit_child_fixedK(target, 0.01, fixed_K, modes=('int8', 'fp16'))
            else:
                fit = fit_child_best(target, 0.01, K_list=(0, 4, 8), modes=('int8', 'fp16'))
            segs.append({'start': s0, 'end': s1, 'fit': fit})
        mgr.commit_segments(start, end, segs)
    return mgr, block_preds


def test_batched_matches_perblock_variable():
    mgr, preds = _make_manager(fixed_K=None)
    batched = mgr.reconstruct_batched(preds)
    for key, pred in preds.items():
        per = mgr.reconstruct_block(pred, key[0], key[1])
        assert torch.allclose(per, batched[key], atol=1e-5), f"mismatch {key}"
    print("variable-K: batched == per-block OK")


def test_batched_matches_perblock_fixedK():
    mgr, preds = _make_manager(fixed_K=8)
    batched = mgr.reconstruct_batched(preds)
    for key, pred in preds.items():
        per = mgr.reconstruct_block(pred, key[0], key[1])
        assert torch.allclose(per, batched[key], atol=1e-5), f"mismatch {key}"
    # 定长：所有段 K 相同 → 同一个桶 → 一次 GEMM
    print("fixed-K=8: batched == per-block OK")


if __name__ == '__main__':
    test_batched_matches_perblock_variable()
    test_batched_matches_perblock_fixedK()
    print("All batched-patch tests passed.")
