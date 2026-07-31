"""
NeurTS: Neural Time-Series Storage System - Main Entry Point

基于 Crossformer 训练框架的 NeurTS 时序压缩系统。

Usage:
    python main_neurts.py --data city_temperature-fixed --root_path ./datasets/
"""

import argparse
import os
import torch

torch.manual_seed(42)


from exp.exp_neurts import Exp_NeurTS
from utils.tools import string_split

# =============================================================================
# 参数解析
# =============================================================================

parser = argparse.ArgumentParser(description='NeurTS: Neural Time-Series Storage')

# 数据参数
parser.add_argument('--data', type=str, required=True, default='city_temperature-fixed', 
                    help='dataset name')
parser.add_argument('--root_path', type=str, default='./datasets/', 
                    help='root path of the data file')
parser.add_argument('--data_path', type=str, default='city_temperature-fixed.csv', 
                    help='data file name')
parser.add_argument('--data_col', type=int, default=0, 
                    help='data column index (0-indexed)')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', 
                    help='location to store model checkpoints')

# NeurTS 核心参数
parser.add_argument('--base_block_size', type=int, default=256,
                    help='base block size for grid nodes')
parser.add_argument('--min_resolution', type=int, default=32,
                    help='minimum resolution for index table')
parser.add_argument('--max_patch_nodes', type=int, default=31111,
                    help='maximum number of patch nodes')

# 模型参数
parser.add_argument('--decoder_type', type=str, default='fourier',
                    help='decoder architecture: fourier | tcn | gaussian_mlp | attn')
parser.add_argument('--transformer_nhead', type=int, default=4,
                    help='[attn] number of attention heads; hidden_dim must be divisible. Default 4.')
parser.add_argument('--transformer_ffn_mult', type=int, default=4,
                    help='[attn] FFN hidden = hidden_dim * mult. Default 4.')
parser.add_argument('--trend_dim', type=int, default=1, 
                    help='trend feature dimension')
parser.add_argument('--context_dim', type=int, default=15,
                    help='context feature dimension. total dim = trend_dim + context_dim.')
parser.add_argument('--hidden_dim', type=int, default=64,
                    help='decoder hidden dimension')
parser.add_argument('--num_freqs', type=int, default=64,
                    help='[fourier] number of Fourier frequency components F. '
                         'Output = Σ_k a_k·sin(2πkt)+b_k·cos(2πkt). '
                         'Nyquist limit = block_size//2. Default 64.')
parser.add_argument('--pe_dim', type=int, default=32,
                    help='positional encoding dimension (used by tcn/attn decoders). Default 32.')
parser.add_argument('--num_res_blocks', type=int, default=5,
                    help='number of layers/blocks in decoder')
parser.add_argument('--kernel_size', type=int, default=5,
                    help='convolution kernel size (for conv-based decoders)')
parser.add_argument('--dropout', type=float, default=0.1, 
                    help='dropout rate')

# 训练参数
parser.add_argument('--num_workers', type=int, default=0, 
                    help='data loader num workers')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=256,
                    help='batch size for evaluation/split (reduce if GPU OOM)')
parser.add_argument('--train_epochs', type=int, default=50,
                    help='training epochs')
parser.add_argument('--patience', type=int, default=100,
                    help='early stopping patience (pretrain phase)')
parser.add_argument('--finetune_patience', type=int, default=100,
                    help='early stopping patience (finetune phase, more tolerant)')
parser.add_argument('--learning_rate', type=float, default=1e-3, 
                    help='initial learning rate')
parser.add_argument('--lradj', type=str, default='none', 
                    help='learning rate adjustment type (none=constant LR, type1/type2=step decay)')
parser.add_argument('--itr', type=int, default=1, 
                    help='number of experiment iterations')

# 训练阶段参数
parser.add_argument('--pretrain_epochs', type=int, default=200,
                    help='phase 1: pretrain epochs (fixed structure)')
parser.add_argument('--final_finetune_epochs', type=int, default=500,
                    help='phase 2: final convergence epochs (vectors only)')
parser.add_argument('--error_mode', type=str, default='relative', choices=['absolute', 'relative'],
                    help='error mode: "relative" = percentage error |e|/|x| (default), "absolute" = absolute error in original units')
parser.add_argument('--finetune_lr_decay_start', type=int, default=60,
                    help='epoch at which cosine LR decay starts during finetune phases. Default 60.')
parser.add_argument('--pretrain_lr_decay_start', type=int, default=100,
                    help='epoch at which cosine LR decay starts during pretrain and final_finetune phases. '
                         'Default 100.')
parser.add_argument('--split_threshold', type=float, default=0.10,
                    help='error threshold for patch commit. relative mode: 0.10=10%%; absolute mode: 3.0=3°F')
parser.add_argument('--eval_threshold', type=float, default=0.10,
                    help='error threshold for evaluation. relative mode: 0.10=10%%; absolute mode: 3.0=3°F')
parser.add_argument('--multiscale_pretrain_weight', type=float, default=0.0,
                    help='multi-scale loss weight during pretrain phase. 0.0 = disabled.')
parser.add_argument('--multiscale_finetune_weight', type=float, default=0.0,
                    help='multi-scale loss weight during finetune phase. 0.0 = disabled.')

# 量化参数
parser.add_argument('--quant_bits', type=int, default=8,
                    help='quantization bits for grid vectors (0 = disable quantization)')
parser.add_argument('--residual_groups', type=int, default=1,
                    help='sub-group quantization groups for residuals. 1=disabled (whole-block span). '
                         '4=split each block into 4 groups, each with independent span/bits. '
                         'Groups with < 16 pts are merged. Default 1.')
parser.add_argument('--patch_split_eval', action='store_true', default=False,
                    help='[offline check] after pretrain, run parent-anchored additive patch-split '
                         'evaluation: for each candidate block, fit an additive Fourier/DST patch '
                         'on top of the FROZEN parent prediction (closed-form least squares, no '
                         'decoder/grid/index/token change), quantize it, and recompute REAL EDWB '
                         'bytes. Reports per-block ACCEPT/REJECT and global raw vs rate-gated net '
                         'bytes. Unsplit blocks are unchanged by construction. Report-only.')
parser.add_argument('--patch_split_K', type=str, default='0,4,8,16',
                    help='[--patch_split_eval] comma-separated DST frequency counts K to enumerate. '
                         'Default 0,4,8,16.')
parser.add_argument('--patch_split_modes', type=str, default='int8,fp16',
                    help='[--patch_split_eval] comma-separated quant modes to enumerate. '
                         'Default int8,fp16.')
parser.add_argument('--patch_split_commit', action='store_true', default=False,
                    help='[integration] after pretrain, COMMIT all byte-gate-accepted '
                         'parent-anchored patches (additive overlay; decoder/grid/index/tokens '
                         'untouched, no z_M nodes) and report the real compression ratio with '
                         'patches vs the no-patch baseline. This is the non-polluting split path: '
                         'unsplit blocks are bit-exact unchanged. When set, the polluting '
                         'progressive split phase is skipped.')
parser.add_argument('--ablation_split_vs_whole', action='store_true', default=False,
                    help='[ablation] after pretrain, compare SPLIT (left/right child patches) '
                         'against WHOLE-BLOCK higher-order coeffs under a MATCHED coefficient '
                         'budget (whole_K ≈ 2×split_K). Answers "why split instead of just adding '
                         'coefficients?" If split wins under equal budget, the gain comes from '
                         'EDWB span isolation, not from more coefficients. Report-only.')
parser.add_argument('--multilayer_patch_eval', action='store_true', default=False,
                    help='[offline check] after pretrain, evaluate MULTI-LAYER patch splitting '
                         'with bottom-up RDO pruning: recursively bisect each base block to '
                         '--patch_max_depth, fit a patch per segment, and run bottom-up DP to '
                         'pick the optimal split depth per interval. Compares multi-layer vs '
                         'single-layer total bytes to decide if deeper splits are worth it. '
                         'Report-only, no training, no shared-state change.')
parser.add_argument('--patch_max_depth', type=int, default=3,
                    help='[--multilayer_patch_eval] max bisection depth (1=single cut, '
                         '2=down to 1/4, 3=down to 1/8...). Default 3.')
parser.add_argument('--multilayer_patch_commit', action='store_true', default=False,
                    help='[integration] after pretrain, COMMIT the multi-layer patch result '
                         '(bottom-up RDO pruned segments) into the patch manager and report the '
                         'real compression ratio with patches. Non-polluting; skips the '
                         'progressive split phase. Uses --patch_max_depth.')
parser.add_argument('--patch_K_sweep', action='store_true', default=False,
                    help='[调研] after pretrain, compare variable-K vs fixed-K patch byte cost '
                         'to choose the fixed K for the batched (定长) patch scheme. Report-only.')
parser.add_argument('--patch_fixed_K', type=int, default=8,
                    help='fixed number of DST frequencies per patch segment for the BATCHED '
                         '(定长) patch scheme. Makes all patch segments equal-length so they '
                         'fold into one GEMM (preserves batching). Default 8. Use --patch_K_sweep '
                         'to pick the best value.')
parser.add_argument('--patch_fixed_mode', type=str, default='int8',
                    help='quant mode for fixed-K patch coeffs (int8|fp16). Default int8.')
parser.add_argument('--patch_fixed', action='store_true', default=False,
                    help='use the BATCHED (定长) patch scheme in commit paths: all segments use '
                         '--patch_fixed_K frequencies and --patch_fixed_mode, so they fold into '
                         'one GEMM per segment-length bucket (preserves batching). When off, '
                         'commit uses variable-K (smaller bytes, but breaks batching).')

# Benchmark 参数
parser.add_argument('--query_dir', type=str, default='./query_sets/',
                    help='directory for fixed query sets (generated once, shared across all methods)')
parser.add_argument('--skip_benchmark', action='store_true', default=False,
                    help='skip random access benchmark after training')
parser.add_argument('--run_spectral_oracle', action='store_true', default=False,
                    help='run DST spectral concentration oracle after training (FourierDecoder only)')
parser.add_argument('--oracle_K', type=str, default='8,16,32',
                    help='comma-separated K values for spectral oracle, e.g. 8,16,32')
parser.add_argument('--benchmark_seed', type=int, default=42,
                    help='RNG seed for query set generation (must match other baselines)')
parser.add_argument('--benchmark_rounds_point', type=int, default=10,
                    help='rounds for point random access benchmark')
parser.add_argument('--benchmark_rounds_full', type=int, default=50,
                    help='rounds for full decompression benchmark')

# GPU 参数
parser.add_argument('--use_gpu', type=bool, default=True, 
                    help='use GPU')
parser.add_argument('--gpu', type=int, default=0, 
                    help='GPU device id')
parser.add_argument('--use_multi_gpu', action='store_true', default=False,
                    help='use multiple GPUs')
parser.add_argument('--devices', type=str, default='0,1,2,3', 
                    help='device ids for multi-GPU')

args = parser.parse_args()

# =============================================================================
# GPU 配置
# =============================================================================

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

# =============================================================================
# 数据集配置
# =============================================================================

data_parser = {
    'city_temperature-fixed': {
        'data': 'city_temperature-fixed.csv', 
        'data_col': 2
    },
    'BT': {
        'data': 'BT.csv', 
        'data_col': 1
    },
    'BW': {
        'data': 'BW.csv', 
        'data_col': 1
    },
}

if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.data_col = data_info.get('data_col', 0)

# =============================================================================
# 主程序
# =============================================================================

print('=' * 60)
print('NeurTS: Neural Time-Series Storage System')
print('=' * 60)
print('\nArgs:')
for k, v in vars(args).items():
    print(f'    {k}: {v}')
print()

for ii in range(args.itr):
    # 实验命名
    setting = 'NeurTS_{}_bs{}_mr{}_hd{}_rb{}_itr{}'.format(
        args.data,
        args.base_block_size,
        args.min_resolution,
        args.hidden_dim,
        args.num_res_blocks,
        ii
    )
    
    print('>' * 60)
    print(f'Start training: {setting}')
    print('>' * 60)
    
    # 创建实验
    exp = Exp_NeurTS(args)
    
    # =========================================================================
    # 阶段 1：预训练（固定结构，不分裂）
    # pretrain_epochs=0 → 跳过预训练，直接进入渐进式分裂（feature_strip 推荐）
    # =========================================================================
    print('\n' + '=' * 60)
    print('PHASE 1: Pretrain (Fixed Structure)')
    print('=' * 60)
    
    if args.pretrain_epochs > 0:
        exp.train(setting, epochs=args.pretrain_epochs, phase='pretrain')
        
        # 预训练后评估
        print('\n[Pretrain] Evaluation:')
        exp.evaluate(setting)
        exp.compute_compression_ratio()
        
        # 预训练后详细报告（所有块的误差统计）
        print('\n[Pretrain] Detailed Report:')
        exp.final_evaluation(error_threshold=args.eval_threshold, error_mode=args.error_mode)

        # 离线 patch-split 字节核验（--patch_split_eval；冻结 decoder，不改共享状态）
        if args.patch_split_eval:
            _K_list = tuple(int(x) for x in args.patch_split_K.split(',') if x.strip() != '')
            _modes = tuple(x.strip() for x in args.patch_split_modes.split(',') if x.strip() != '')
            exp.patch_split_eval(
                error_threshold=args.eval_threshold,
                error_mode=args.error_mode,
                K_list=_K_list,
                modes=_modes,
            )

        # Ablation：分裂 vs 整块加系数（相同系数预算）
        if args.ablation_split_vs_whole:
            _K_list = tuple(int(x) for x in args.patch_split_K.split(',') if x.strip() != '')
            _modes = tuple(x.strip() for x in args.patch_split_modes.split(',') if x.strip() != '')
            exp.ablation_split_vs_whole(
                error_threshold=args.eval_threshold,
                error_mode=args.error_mode,
                split_K_list=_K_list,
                modes=_modes,
            )

        # 多层 patch + 自下而上 RDO 剪枝（离线核验）
        if args.multilayer_patch_eval:
            _K_list = tuple(int(x) for x in args.patch_split_K.split(',') if x.strip() != '')
            _modes = tuple(x.strip() for x in args.patch_split_modes.split(',') if x.strip() != '')
            exp.multilayer_patch_eval(
                error_threshold=args.eval_threshold,
                error_mode=args.error_mode,
                K_list=_K_list,
                modes=_modes,
                max_depth=args.patch_max_depth,
            )

        # 定长 K 选择调研（variable vs fixed-K）
        if args.patch_K_sweep:
            _K_list = tuple(int(x) for x in args.patch_split_K.split(',') if x.strip() != '')
            _modes = tuple(x.strip() for x in args.patch_split_modes.split(',') if x.strip() != '')
            exp.patch_K_sweep(
                error_threshold=args.eval_threshold,
                error_mode=args.error_mode,
                K_candidates=_K_list,
                modes=_modes,
            )
    else:
        print(f'  [Pretrain] Skipped (pretrain_epochs=0). Entering progressive split directly.')
        print(f'  Decoder type: {args.decoder_type}')

    # =========================================================================
    # 接入路径：commit patch-split（非污染分裂）。设置后跳过渐进式分裂。
    # =========================================================================
    if args.patch_split_commit:
        print('\n' + '=' * 60)
        print('PATCH-SPLIT COMMIT (non-polluting split path)')
        print('=' * 60)
        if args.patch_fixed:
            _K_list = (args.patch_fixed_K,)
            _modes = (args.patch_fixed_mode,)
            print(f'  [定长 batched] fixed K={args.patch_fixed_K}, mode={args.patch_fixed_mode}')
        else:
            _K_list = tuple(int(x) for x in args.patch_split_K.split(',') if x.strip() != '')
            _modes = tuple(x.strip() for x in args.patch_split_modes.split(',') if x.strip() != '')
        exp.patch_split_eval(
            error_threshold=args.eval_threshold,
            error_mode=args.error_mode,
            K_list=_K_list,
            modes=_modes,
            commit=True,
        )
        exp.compression_ratio_with_patches(
            error_threshold=args.eval_threshold,
            error_mode=args.error_mode,
        )
        print('\n[Patch-Split Commit] Progressive split phase skipped (non-polluting path used).')

    # 多层 patch + 自下而上回收 COMMIT（非污染分裂）。
    if args.multilayer_patch_commit:
        print('\n' + '=' * 60)
        print('MULTI-LAYER PATCH COMMIT (bottom-up RDO pruned, non-polluting)')
        print('=' * 60)
        if args.patch_fixed:
            _K_list = (args.patch_fixed_K,)
            _modes = (args.patch_fixed_mode,)
            print(f'  [定长 batched] fixed K={args.patch_fixed_K}, mode={args.patch_fixed_mode}')
        else:
            _K_list = tuple(int(x) for x in args.patch_split_K.split(',') if x.strip() != '')
            _modes = tuple(x.strip() for x in args.patch_split_modes.split(',') if x.strip() != '')
        exp.multilayer_patch_eval(
            error_threshold=args.eval_threshold,
            error_mode=args.error_mode,
            K_list=_K_list,
            modes=_modes,
            max_depth=args.patch_max_depth,
            commit=True,
        )
        exp.compression_ratio_with_patches(
            error_threshold=args.eval_threshold,
            error_mode=args.error_mode,
        )
        print('\n[Multi-Layer Commit] Progressive split phase skipped (non-polluting path used).')

    # =========================================================================
    # 阶段 2：收尾收敛（固定结构，只优化向量）
    # =========================================================================
    print('\n' + '=' * 60)
    print('PHASE 2: Final Convergence (vectors only, no multiscale)')
    print('=' * 60)
    
    exp.train(setting, epochs=args.final_finetune_epochs,
              phase='final_finetune',
              finetune_only_vectors=True)
    
    print('\n[Phase 2] Evaluation after final convergence:')
    exp.evaluate(setting)
    exp.compute_compression_ratio()
    exp.final_evaluation(error_threshold=args.eval_threshold, error_mode=args.error_mode)
    
    # =========================================================================
    # 最终处理
    # =========================================================================
    print('\n' + '=' * 60)
    print('Final Processing')
    print('=' * 60)
    
    # 裁剪未使用的 patch 节点
    exp.manager.trim_unused_patch_nodes()
    
    # 最终评估（量化前）
    print('\n[Final] Pre-Quantization Evaluation:')
    exp.evaluate(setting)
    exp.compute_compression_ratio()
    
    # 详细最终报告（量化前）
    print('\n[Final] Pre-Quantization Detailed Report:')
    pre_quant_report = exp.final_evaluation(error_threshold=args.eval_threshold, error_mode=args.error_mode)
    
    # =========================================================================
    # 构建兜底字典（Tier 2/3）
    # =========================================================================
    print('\n' + '=' * 60)
    print('Building Fallback Dictionary (Tier 2/3)')
    print('=' * 60)
    fb_dict = exp.build_fallback_dict(
        error_threshold=args.eval_threshold,
        error_mode=args.error_mode
    )
    
    # =========================================================================
    # 量化处理
    # =========================================================================
    if args.quant_bits > 0:
        print('\n' + '=' * 60)
        print(f'Quantization ({args.quant_bits}-bit)')
        print('=' * 60)
        
        # 执行量化
        exp.quantize_grid()
        
        # 量化后评估
        print('\n[Final] Post-Quantization Evaluation:')
        exp.evaluate(setting)
        
        # 详细最终报告（量化后）
        print('\n[Final] Post-Quantization Detailed Report:')
        post_quant_report = exp.final_evaluation(error_threshold=args.eval_threshold, error_mode=args.error_mode)
        
        # 对比量化前后
        print('\n[Quantization Impact]')
        print(f"    MAE: {pre_quant_report['mae']:.6f} -> {post_quant_report['mae']:.6f} "
              f"(+{(post_quant_report['mae'] - pre_quant_report['mae']):.6f})")
        print(f"    Max Error: {pre_quant_report['max_abs_error']:.6f} -> {post_quant_report['max_abs_error']:.6f} "
              f"(+{(post_quant_report['max_abs_error'] - pre_quant_report['max_abs_error']):.6f})")
        print(f"    Compliance: {pre_quant_report['compliance_rate']*100:.4f}% -> {post_quant_report['compliance_rate']*100:.4f}%")
        
        final_report = post_quant_report
        final_report['pre_quant_mae'] = pre_quant_report['mae']
        final_report['pre_quant_max_error'] = pre_quant_report['max_abs_error']
    else:
        final_report = pre_quant_report
    
    # patch 节点存储方案测试（量化后，向量已最终化）
    if exp._parentage_map:
        print('\n' + '=' * 60)
        print('Patch Node Storage Analysis')
        print('=' * 60)
        exp.report_z_delta_stats()
        exp.test_reduced_storage(setting)

    # 保存报告
    vis_dir = './visualizations'
    os.makedirs(vis_dir, exist_ok=True)
    import json
    report_path = os.path.join(vis_dir, 'final_report.json')
    with open(report_path, 'w') as f:
        # level_stats 和 block_size_stats 的 key 是 int，需要转换
        save_report = {k: v for k, v in final_report.items() if k not in ['level_stats', 'block_size_stats']}
        save_report['level_stats'] = {str(k): v for k, v in final_report.get('level_stats', {}).items()}
        save_report['block_size_stats'] = {str(k): v for k, v in final_report.get('block_size_stats', {}).items()}
        json.dump(save_report, f, indent=2)
    print(f"\n[Report] Saved to {report_path}")

    # =========================================================================
    # 构建并保存 Accessor（训练后只需做一次）
    # =========================================================================
    print('\n' + '=' * 60)
    print('Building & Saving NeurTSAccessor')
    print('=' * 60)
    from neurts_accessor import NeurTSAccessor
    accessor = NeurTSAccessor.from_exp(exp)
    accessor_path = os.path.join('./checkpoints', f'{setting}_accessor.pt')
    accessor.save(accessor_path)
    print(f'[Accessor] Saved → {accessor_path}')
    print('[Accessor] Load next time with:')
    print(f'    from neurts_accessor import NeurTSAccessor')
    print(f'    accessor = NeurTSAccessor.from_file("{accessor_path}")')

    # =========================================================================
    # Spectral Oracle（可选，仅 FourierDecoder 有意义）
    # =========================================================================
    if args.run_spectral_oracle:
        print('\n' + '=' * 60)
        print('Spectral Oracle Analysis')
        print('=' * 60)
        K_values = tuple(int(k) for k in args.oracle_K.split(','))
        exp.spectral_oracle(K_values=K_values)

    # =========================================================================
    # 随机访问 Benchmark（固定 query set，与所有 baseline 统一标准）
    # =========================================================================
    if not args.skip_benchmark:
        print('\n' + '=' * 60)
        print('Random Access Benchmark (NeaTS-style, fixed query sets)')
        print('=' * 60)

        from neurts_accessor import generate_query_sets
        import os

        query_dir = args.query_dir
        seed      = args.benchmark_seed
        T         = accessor.T

        # 生成 query set（如果文件已存在则跳过）
        point_file = os.path.join(query_dir, f'point_N1000000_seed{seed}.npy')
        if not os.path.exists(point_file):
            print(f'[Benchmark] Generating query sets in {query_dir} ...')
            generate_query_sets(query_dir, total_length=T, seed=seed)
        else:
            print(f'[Benchmark] Reusing existing query sets in {query_dir}')

        # 一键跑完所有指标
        results = accessor.run_all_benchmarks(
            query_dir       = query_dir,
            bytes_per_value = 4,
            seed            = seed,
            n_point         = 1_000_000,
            n_range         = 10_000,
            rounds_point    = args.benchmark_rounds_point,
            rounds_full     = args.benchmark_rounds_full,
            mode            = 'per_round_cold',
        )

        # 摘要打印
        print('\n[Benchmark Summary]')
        if results.get('point'):
            pr = results['point']
            print(f'  Point random access : {pr["random_access_MBps"]:.2f} MB/s'
                  f'  ({pr["avg_ns_per_query"]:.1f} ns/query)')
        if results.get('full_decomp'):
            fd = results['full_decomp']
            print(f'  Full decompression  : {fd["full_decompression_MBps"]:.2f} MB/s')
        if results.get('range'):
            tops = sorted(results['range'], key=lambda r: r['window'])
            print(f'  Range throughput    : ' +
                  '  '.join(f'W{r["window"]}={r["MBps"]:.1f}MB/s' for r in tops[:4]))
    else:
        print('\n[Benchmark] Skipped (--skip_benchmark). '
              f'Run later with: accessor.run_all_benchmarks("{args.query_dir}")')

print('\n' + '=' * 60)
print('All experiments completed!')
print('=' * 60)
