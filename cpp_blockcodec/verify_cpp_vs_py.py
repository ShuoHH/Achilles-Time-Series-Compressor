"""
比对 C++ dump_blockcodec 输出 vs Python BlockAccessor.decompress_all，验证数值一致。

用法:
  1. python export_blockcodec_cpp.py --ckpt <ckpt> --out <export_dir>
  2. ./dump_blockcodec <export_dir> cpp_full.bin
  3. python cpp_blockcodec/verify_cpp_vs_py.py --ckpt <ckpt> --export <export_dir> --cpp cpp_full.bin
"""
import argparse, json, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from exp.exp_neurts import Exp_NeurTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--export', required=True)
    ap.add_argument('--cpp', required=True, help='C++ dump 的 float32 .bin')
    a = ap.parse_args()

    with open(os.path.join(a.ckpt, 'args.json')) as f:
        args = argparse.Namespace(**json.load(f))
    args.skip_benchmark = True; args.use_multi_gpu = False
    args.use_gpu = torch.cuda.is_available()
    exp = Exp_NeurTS(args)
    sd = torch.load(os.path.join(a.ckpt, 'checkpoint.pth'), map_location=exp.device)
    exp.model.load_state_dict(sd, strict=False)
    with open(os.path.join(a.ckpt, 'manager_state.pkl'), 'rb') as f:
        ms = pickle.load(f)
    exp.manager.patch_counter = ms['patch_counter']
    for i, (lid, rid) in enumerate(ms['index_table']):
        if i < len(exp.manager.index_table):
            exp.manager.index_table[i].left_id = lid
            exp.manager.index_table[i].right_id = rid
    exp._refresh_quant_params()
    # 与导出脚本一致：定长 K、残差落盘开启（C++ 含残差，Python 参考也要含残差才可比）
    pfk = getattr(args, 'patch_fixed_K', 8)
    Kf = (pfk,)
    modes = (getattr(args, 'patch_fixed_mode', 'int8'),)
    _max_depth = getattr(args, 'patch_max_depth', 3)
    _eval_thr = getattr(args, 'eval_threshold', getattr(args, 'split_threshold', 0.05))
    _err_mode = getattr(args, 'error_mode', 'absolute')
    exp._codec_build_residual = True
    exp.multilayer_patch_eval(error_threshold=_eval_thr, error_mode=_err_mode,
                              K_list=Kf, modes=modes, max_depth=_max_depth, commit=True)

    py_full = exp.block_accessor.decompress_all().cpu().numpy().astype(np.float32)
    cpp_full = np.fromfile(a.cpp, dtype=np.float32)
    n = min(len(py_full), len(cpp_full))
    py_full, cpp_full = py_full[:n], cpp_full[:n]
    diff = np.abs(py_full - cpp_full)
    print(f"compared {n} points")
    print(f"  max abs diff = {diff.max():.6e}")
    print(f"  mean abs diff = {diff.mean():.6e}")
    print(f"  {'PASS' if diff.max() < 1e-3 else 'FAIL'} (tol 1e-3, fp32+libm 差异)")


if __name__ == '__main__':
    main()
