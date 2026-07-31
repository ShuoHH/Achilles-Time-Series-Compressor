"""
批量压缩比：扫 all_checkpoints 下所有 checkpoint，逐个加载权重 → 重建 patch 树
（multilayer commit，闭式拟合不重训）→ 算 baseline / patch 压缩比 → 汇总成表。

每个 checkpoint 独立 try/except，单个失败不影响其余，最后报成功/失败清单。

用法：
  python batch_compression_ratio.py                              # 默认 all_checkpoints, ε=0.05
  python batch_compression_ratio.py --threshold 0.1
  python batch_compression_ratio.py --filter BT                  # 只跑名字含 BT 的
  python batch_compression_ratio.py --csv ratios.csv             # 导出 CSV
  python batch_compression_ratio.py --thresholds 0.01,0.05,0.1   # 一个模型扫多个阈值

压缩比口径（与 compression_ratio_with_patches 一致）：
  ratio   = 原始 / (grid + index + 残差[+patch系数])           —— 不含 decoder 权重
  ratio_w = 原始 / (上 + decoder 权重)                          —— 含共享 decoder 权重
"""
import os, sys, json, pickle, argparse, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from exp.exp_neurts import Exp_NeurTS


def load_exp(ckpt_dir):
    with open(os.path.join(ckpt_dir, 'args.json')) as f:
        args = argparse.Namespace(**json.load(f))
    args.skip_benchmark = True
    args.use_multi_gpu = False
    args.use_gpu = torch.cuda.is_available()
    exp = Exp_NeurTS(args)
    sd = torch.load(os.path.join(ckpt_dir, 'checkpoint.pth'), map_location=exp.device)
    exp.model.load_state_dict(sd, strict=False)
    with open(os.path.join(ckpt_dir, 'manager_state.pkl'), 'rb') as f:
        ms = pickle.load(f)
    exp.manager.patch_counter = ms['patch_counter']
    for i, (lid, rid) in enumerate(ms['index_table']):
        if i < len(exp.manager.index_table):
            exp.manager.index_table[i].left_id = lid
            exp.manager.index_table[i].right_id = rid
    exp._refresh_quant_params()
    return exp, args


def run_one(ckpt_dir, threshold, patch_fixed_K, patch_fixed_mode='int8'):
    """
    加载 + 重建树(commit) + 算两个压缩比：
      - quantized   : 权重量化（grid 走 fake-quant，体积按 quant_bits/8）→ 落盘真实压缩比
      - unquantized : 权重不量化（grid 用 float32 原值，体积按 4 B/dim）→ 理论上界对照

    两版用同一棵已 commit 的树（树结构与量化无关），只切换 grid 量化开关与
    args.quant_bits（决定体积口径 + 重建是否走 fake-quant）。
    """
    exp, args = load_exp(ckpt_dir)
    _max_depth = getattr(args, 'patch_max_depth', 3)
    _err_mode = getattr(args, 'error_mode', 'absolute')
    gs = exp.grid_storage
    orig_qb = getattr(args, 'quant_bits', 8)
    orig_qen = getattr(gs, '_quantization_enabled', False)

    # ---- 量化版：保持训练时的量化设置（quant_bits>0 且开 fake-quant）----
    args.quant_bits = orig_qb if orig_qb and orig_qb > 0 else 8
    if hasattr(gs, '_quantization_enabled'):
        gs._quantization_enabled = True
    exp._refresh_quant_params()
    exp.multilayer_patch_eval(
        error_threshold=threshold, error_mode=_err_mode,
        K_list=(patch_fixed_K,), modes=(patch_fixed_mode,),
        max_depth=_max_depth, commit=True,
    )
    r_quant = exp.compression_ratio_with_patches(error_threshold=threshold, error_mode=_err_mode)

    # ---- 不量化版：grid 用 float32 原值重建，体积按 4 B/dim（quant_bits=0）----
    # 重新加载一个干净的 exp，避免量化版的 commit 状态串味（树会重建）。
    exp2, args2 = load_exp(ckpt_dir)
    args2.quant_bits = 0  # 0 → compression_ratio_with_patches 用 4 B/dim
    if hasattr(exp2.grid_storage, '_quantization_enabled'):
        exp2.grid_storage._quantization_enabled = False
    exp2.multilayer_patch_eval(
        error_threshold=threshold, error_mode=_err_mode,
        K_list=(patch_fixed_K,), modes=(patch_fixed_mode,),
        max_depth=_max_depth, commit=True,
    )
    r_float = exp2.compression_ratio_with_patches(error_threshold=threshold, error_mode=_err_mode)

    return r_quant, r_float


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='all_checkpoints', help='checkpoint 根目录')
    ap.add_argument('--threshold', type=float, default=0.05)
    ap.add_argument('--thresholds', default='', help='逗号分隔多阈值，覆盖 --threshold（一个模型扫多个 ε）')
    ap.add_argument('--patch_fixed_K', type=int, default=8)
    ap.add_argument('--filter', default='', help='只跑名字含该子串的 checkpoint')
    ap.add_argument('--csv', default='', help='导出 CSV 路径')
    a = ap.parse_args()

    thresholds = [float(x) for x in a.thresholds.split(',') if x.strip()] if a.thresholds \
                 else [a.threshold]

    # 找出所有合法 checkpoint 目录
    ckpts = []
    for name in sorted(os.listdir(a.root)):
        d = os.path.join(a.root, name)
        if not os.path.isdir(d):
            continue
        if a.filter and a.filter not in name:
            continue
        if os.path.exists(os.path.join(d, 'checkpoint.pth')) and \
           os.path.exists(os.path.join(d, 'args.json')):
            ckpts.append((name, d))

    print(f"找到 {len(ckpts)} 个 checkpoint（root={a.root}, filter='{a.filter}'）")
    print(f"阈值 = {thresholds}, 定长 K={a.patch_fixed_K}\n")

    # results[name][thr] = {'quant': r_quant, 'float': r_float}
    results, failed = {}, []
    for i, (name, d) in enumerate(ckpts):
        results[name] = {}
        for thr in thresholds:
            print("=" * 70)
            print(f"[{i+1}/{len(ckpts)}] {name}  (ε={thr})")
            print("=" * 70)
            try:
                r_quant, r_float = run_one(d, thr, a.patch_fixed_K)
                if r_quant is None and r_float is None:
                    failed.append((name, thr, "no committed patches"))
                    print("  [SKIP] no committed patches")
                else:
                    results[name][thr] = {'quant': r_quant, 'float': r_float}
                    if r_quant:
                        print(f"  [quantized ] base={r_quant['base_ratio_w']:.3f}x  "
                              f"patch={r_quant['patch_ratio_w']:.3f}x  patched={r_quant['n_patched']}")
                    if r_float:
                        print(f"  [unquantized] base={r_float['base_ratio_w']:.3f}x  "
                              f"patch={r_float['patch_ratio_w']:.3f}x")
            except Exception as e:
                failed.append((name, thr, str(e)))
                print(f"  [FAILED] {e}")
                traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- 汇总表（每个阈值一张）----
    # 报含权重口径（base_ratio_w / patch_ratio_w）：量化版 vs 不量化版。
    for thr in thresholds:
        print("\n" + "=" * 110)
        print(f"SUMMARY — Compression ratio with weights (ε={thr})  "
              f"[Q=量化权重, F=float32权重]")
        print("=" * 110)
        hdr = (f"{'dataset':<40} {'Q_base':>8} {'Q_patch':>8} "
               f"{'F_base':>8} {'F_patch':>8} {'patched':>8}")
        print(hdr)
        print("-" * len(hdr))
        for name in results:
            cell = results[name].get(thr)
            if not cell:
                print(f"{name:<40} {'--':>8} {'--':>8} {'--':>8} {'--':>8} {'--':>8}")
                continue
            q, fl = cell['quant'], cell['float']
            qb = f"{q['base_ratio_w']:.3f}" if q else "--"
            qp = f"{q['patch_ratio_w']:.3f}" if q else "--"
            fb = f"{fl['base_ratio_w']:.3f}" if fl else "--"
            fp = f"{fl['patch_ratio_w']:.3f}" if fl else "--"
            npat = q['n_patched'] if q else (fl['n_patched'] if fl else 0)
            print(f"{name:<40} {qb:>8} {qp:>8} {fb:>8} {fp:>8} {npat:>8}")

    if failed:
        print("\n" + "=" * 70)
        print(f"FAILED / SKIPPED ({len(failed)}):")
        for name, thr, err in failed:
            print(f"  {name} (ε={thr}): {err}")

    # ---- CSV ----
    if a.csv:
        import csv
        with open(a.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['dataset', 'threshold',
                        'quant_base_ratio', 'quant_patch_ratio',
                        'quant_base_ratio_w', 'quant_patch_ratio_w',
                        'float_base_ratio', 'float_patch_ratio',
                        'float_base_ratio_w', 'float_patch_ratio_w',
                        'n_patched', 'decoder_params'])
            for name in results:
                for thr in thresholds:
                    cell = results[name].get(thr)
                    if not cell:
                        continue
                    q, fl = cell['quant'], cell['float']
                    def g(d, k):
                        return d[k] if d else ''
                    w.writerow([name, thr,
                                g(q, 'base_ratio'), g(q, 'patch_ratio'),
                                g(q, 'base_ratio_w'), g(q, 'patch_ratio_w'),
                                g(fl, 'base_ratio'), g(fl, 'patch_ratio'),
                                g(fl, 'base_ratio_w'), g(fl, 'patch_ratio_w'),
                                g(q or fl, 'n_patched'), g(q or fl, 'decoder_params')])
        print(f"\n[CSV] 已导出 → {a.csv}")


if __name__ == '__main__':
    main()
