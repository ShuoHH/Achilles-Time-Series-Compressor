"""
随机访问 Benchmark（阿克琉斯 patch 版 BlockCodec/BlockAccessor）。

复现表格指标：
  - 随机点访问 (point access): N ∈ {1,100,1000,10000,100000}   → ns/query
  - 随机段访问 (range scan):    W ∈ {10,100,1000,10000,100000} → MB/s
  - 解压缩 (decompression):     全序列                          → MB/s

流程：加载权重 → multilayer commit 构建 BlockCodec → 注入访问回调
      → 固定 seed 生成 query set → warmup + 多轮计时取均值。

注意：当前残差未接（base+patch），测的是访问速度量级；接残差只是每段多一次
      查表+加法，不改速度量级。
"""
import os, sys, json, pickle, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
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


def build_accessor(exp, args, threshold=None, patch_fixed_K=8, patch_fixed_mode='int8'):
    # C++ 对齐：默认走定长 K（BlockCodec 用定长系数池）。旧 args.json 缺字段时兜底。
    Kf = (patch_fixed_K,)
    modes = (patch_fixed_mode,)
    _max_depth = getattr(args, 'patch_max_depth', 3)
    _eval_thr = threshold if threshold is not None else \
                getattr(args, 'eval_threshold', getattr(args, 'split_threshold', 0.05))
    _err_mode = getattr(args, 'error_mode', 'absolute')
    print(f"  [build_accessor] eval_threshold={_eval_thr}, error_mode={_err_mode}, "
          f"max_depth={_max_depth}, K_fixed={patch_fixed_K}, mode={patch_fixed_mode}")
    exp._codec_build_residual = True   # 随机访问需要残差落盘（误差精确重建）
    exp.multilayer_patch_eval(
        error_threshold=_eval_thr,
        error_mode=_err_mode,
        K_list=Kf, modes=modes,
        max_depth=_max_depth,
        commit=True,
    )
    assert exp.block_accessor is not None, "BlockAccessor 未构建"
    return exp.block_accessor


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, rounds, warmup=2, reset=None):
    """
    每轮单独计时取均值。reset（若提供）在每轮 fn() 前执行但**不计入**耗时，
    用于清缓存 → 保证每轮都是"冷"执行。否则 warmup 那次就把 _block_pred_cache
    填满，后续所有轮全是读缓存，测的是缓存读取速度而非真实解码。
    """
    for _ in range(warmup):
        if reset is not None:
            reset()
        fn()
    _sync()
    total = 0.0
    for _ in range(rounds):
        if reset is not None:
            reset()          # 不计时：清缓存，保证冷解码
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        total += time.perf_counter() - t0
    return total / rounds


# 批处理版：查询点去重到唯一块，按块长分桶一次 decode_batch（batch=K 一次推理），
# 再散射取点/段。对应"阿克琉斯"真实批处理路径（100 点→20 块→一次批量解）。
def bench_point(acc, T, sizes, seed=42, rounds=5):
    rng = np.random.default_rng(seed)
    print("\n" + "=" * 64)
    print("POINT ACCESS (per-block decode, COLD: cache cleared each round)")
    print("=" * 64)
    print(f"  {'N':>8} | {'total ms':>10} | {'ns/query':>12}")
    print("  " + "-" * 40)
    res = {}
    for n in sizes:
        times = rng.integers(0, T, size=n).tolist()
        sec = timed(lambda: acc.query_batch(times), rounds=rounds, warmup=2,
                    reset=acc.clear_cache)
        ns = sec / n * 1e9
        print(f"  {n:>8} | {sec*1e3:>10.3f} | {ns:>12.1f}")
        res[n] = ns
        acc.clear_cache()
    return res


def bench_range(acc, T, widths, seed=43, bytes_per_val=4):
    rng = np.random.default_rng(seed)
    print("\n" + "=" * 64)
    print("RANGE SCAN (per-block decode, COLD: cache cleared each round)")
    print("=" * 64)
    print(f"  {'W':>8} | {'ms/query':>10} | {'MB/s':>10}")
    print("  " + "-" * 40)
    res = {}
    for w in widths:
        if w > T:
            continue
        starts = rng.integers(0, T - w, size=5).tolist()
        def run():
            for s in starts:
                acc.query_range(int(s), int(s) + w)
        sec = timed(run, rounds=2, warmup=1, reset=acc.clear_cache) / len(starts)
        mbps = (w * bytes_per_val) / sec / 1e6
        print(f"  {w:>8} | {sec*1e3:>10.4f} | {mbps:>10.2f}")
        res[w] = mbps
        acc.clear_cache()
    return res


def bench_decompress(acc, T, rounds=3, bytes_per_val=4):
    print("\n" + "=" * 64)
    print("DECOMPRESSION (full series, per-block decode, COLD: cache cleared each round)")
    print("=" * 64)
    # 每轮清缓存 → 真实冷解码（否则后续轮全读缓存，速度虚高）
    sec = timed(lambda: acc.decompress_all(), rounds=rounds, warmup=1,
                reset=acc.clear_cache)
    mbps = (T * bytes_per_val) / sec / 1e6
    print(f"  T={T} | {sec*1e3:.2f} ms | {mbps:.2f} MB/s")
    return mbps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=r'checkpoints/NeurTS_BT_bs512_mr32_hd256_rb5_itr0')
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--threshold', type=float, default=None,
                    help='重建树用的误差阈值 ε（不传则用 args.json 的值，再无则 0.05）')
    ap.add_argument('--patch_fixed_K', type=int, default=8, help='定长 patch K，默认 8')
    a = ap.parse_args()

    print("Loading model + building BlockCodec ...")
    exp, args = load_exp(a.ckpt)
    acc = build_accessor(exp, args, threshold=a.threshold, patch_fixed_K=a.patch_fixed_K)
    T = exp.manager.total_length
    st = exp.block_codec.total_bytes()
    print(f"\nBlockCodec: {st['num_blocks']} blocks, {st['num_leaves']} leaves, "
          f"code {exp.block_codec.codebook.code_bytes}B/block, T={T}")

    pr = bench_point(acc, T, [1, 100, 1000, 10000, 100000], rounds=a.rounds)
    rr = bench_range(acc, T, [10, 100, 1000, 10000, 100000])
    dr = bench_decompress(acc, T)

    print("\n" + "=" * 64)
    print("SUMMARY (阿克琉斯 python, patch BlockCodec)")
    print("=" * 64)
    print("  Point access (ns/query):")
    for n, v in pr.items():
        print(f"    N={n:>7}: {v:>12.1f} ns")
    print("  Range scan (MB/s):")
    for w, v in rr.items():
        print(f"    W={w:>7}: {v:>10.2f} MB/s")
    print(f"  Decompression: {dr:.2f} MB/s")
    print("=" * 64)


if __name__ == '__main__':
    main()
