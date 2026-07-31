"""
export_blockcodec_cpp.py
========================

把训练好并 commit 了 patch 的 NeurTS 模型（含 BlockCodec）导出成
C++ accessor 可 mmap 的二进制文件（patch 版，区别于旧版 z_M 分裂导出）。

输出目录布局
-----------
    meta.bin            固定 68 字节头（见 BINARY FORMAT）
    meta.json           人类可读副本 + partitions（C++ 复刻隐式树用）
    grid.bin            float32 [num_nodes, feature_dim]  base+patch latent
    decoder.bin         FourierDecoder 权重（同旧版格式）
    struct_code.bin     uint8/uint16 [num_blocks]         结构码(隐式树形状)
    block_offset.bin    int32   [num_blocks+1]            叶子前缀和(O(1)行寻址)
    block_meta.bin      int32   [num_blocks, 3]           (block_start,left_id,right_id)
    coeff_pool.bin      float32 [num_leaves, K_fixed+2]   定长系数 δ

BINARY FORMAT (meta.bin, little-endian, 无 padding) — 68 字节
    0   int32  format_version (=2)
    4   int32  T
    8   int32  base_block_size
    12  int32  min_resolution
    16  int32  max_depth
    20  int32  n_codes
    24  int32  code_bytes (1 或 2)
    28  int32  num_blocks
    32  int32  num_leaves
    36  int32  num_nodes
    40  int32  feature_dim
    44  int32  K_fixed
    48  int32  dec_in_dim
    52  int32  dec_hidden_dim
    56  int32  dec_num_freqs
    60  float32 scaler_mean
    64  float32 scaler_std
"""

import argparse
import json
import os
import struct

import numpy as np
import torch

_FORMAT_VERSION = 3


def _f32(t):
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    return np.ascontiguousarray(t, dtype=np.float32)


def _extract_decoder(model):
    dec = model.decoder
    if type(dec).__name__ != 'FourierDecoder':
        raise RuntimeError(f"only FourierDecoder supported, got {type(dec).__name__}")
    seq = dec.to_coeff
    return {
        'in_dim': int(dec.in_dim),
        'hidden_dim': int(seq[0].out_features),
        'num_freqs': int(dec.num_freqs),
        'w': {
            'to_v0_W': _f32(dec.to_v0.weight), 'to_v0_b': _f32(dec.to_v0.bias),
            'to_v1_W': _f32(dec.to_v1.weight), 'to_v1_b': _f32(dec.to_v1.bias),
            'to_coeff0_W': _f32(seq[0].weight), 'to_coeff0_b': _f32(seq[0].bias),
            'to_coeff2_W': _f32(seq[2].weight), 'to_coeff2_b': _f32(seq[2].bias),
            'freqs': _f32(dec.freqs),
        }
    }


def export_blockcodec(exp, out_dir):
    """从已构建 block_codec 的 exp 导出 C++ 二进制。"""
    codec = exp.block_codec
    if codec is None:
        raise RuntimeError("exp.block_codec is None; 先跑 multilayer commit 构建 codec")
    os.makedirs(out_dir, exist_ok=True)

    gs = exp.grid_storage
    cb = codec.codebook
    block_ids = sorted(codec.blocks.keys())
    num_blocks = len(block_ids)
    feature_dim = gs.trend_dim + gs.context_dim

    # grid.bin
    base = _f32(gs.base_grid.detach())
    patch = _f32(gs.patch_grid.detach())
    grid = np.ascontiguousarray(np.concatenate([base, patch], axis=0), dtype=np.float32)
    grid.tofile(os.path.join(out_dir, 'grid.bin'))
    num_nodes = grid.shape[0]

    # decoder.bin
    dec = _extract_decoder(exp.model)
    with open(os.path.join(out_dir, 'decoder.bin'), 'wb') as fp:
        fp.write(struct.pack('<iiii', dec['in_dim'], dec['hidden_dim'], dec['num_freqs'], 0))
        for k in ('to_v0_W', 'to_v0_b', 'to_v1_W', 'to_v1_b',
                  'to_coeff0_W', 'to_coeff0_b', 'to_coeff2_W', 'to_coeff2_b', 'freqs'):
            dec['w'][k].tofile(fp)

    # struct_code / block_offset / block_meta / coeff_pool
    code_dtype = np.uint16 if cb.code_bytes == 2 else np.uint8
    codes = np.zeros(num_blocks, dtype=code_dtype)
    offsets = np.zeros(num_blocks + 1, dtype=np.int32)
    block_meta = np.zeros((num_blocks, 3), dtype=np.int32)
    coeff_rows = []
    cursor = 0
    for bi, bid in enumerate(block_ids):
        rec = codec.blocks[bid]
        codes[bi] = rec.code
        offsets[bi] = cursor
        block_meta[bi] = [rec.block_start, rec.left_id, rec.right_id]
        for lf in rec.leaves:
            coeff_rows.append(_f32(lf.delta_q))
            cursor += 1
    offsets[num_blocks] = cursor
    num_leaves = cursor

    codes.tofile(os.path.join(out_dir, 'struct_code.bin'))
    offsets.tofile(os.path.join(out_dir, 'block_offset.bin'))
    block_meta.tofile(os.path.join(out_dir, 'block_meta.bin'))
    if coeff_rows:
        coeff_pool = np.ascontiguousarray(np.stack(coeff_rows, axis=0), dtype=np.float32)
    else:
        coeff_pool = np.zeros((0, codec.K_fixed + 2), dtype=np.float32)
    coeff_pool.tofile(os.path.join(out_dir, 'coeff_pool.bin'))

    # ── 残差文件（方案B）：从 ResidualCodec 落盘，供 C++ 精确重建（误差≤ε）──
    rc = getattr(codec, 'residual_codec', None)
    has_residual = 1 if rc is not None else 0
    res_stream_bytes = 0
    res_step = 0.0
    if rc is not None:
        np.ascontiguousarray(rc.leaf_bits, dtype=np.uint8).tofile(
            os.path.join(out_dir, 'residual_bits.bin'))       # [num_leaves] uint8 位宽
        np.ascontiguousarray(rc.leaf_rmin, dtype=np.float32).tofile(
            os.path.join(out_dir, 'residual_rmin.bin'))       # [num_leaves] float32 r_min
        np.ascontiguousarray(rc.leaf_len, dtype=np.int32).tofile(
            os.path.join(out_dir, 'residual_len.bin'))        # [num_leaves] int32 段长
        np.ascontiguousarray(rc.leaf_bitpos, dtype=np.int64).tofile(
            os.path.join(out_dir, 'residual_bitpos.bin'))     # [num_leaves+1] int64 比特起点前缀和
        np.ascontiguousarray(rc.bitstream, dtype=np.uint8).tofile(
            os.path.join(out_dir, 'residual_stream.bin'))     # packbits 位流
        res_stream_bytes = int(rc.bitstream.size)
        res_step = float(rc.step)                             # 量化步长 2ε

    # meta.bin（format v3：在 v2 的 15i2f 基础上追加 1i(has_residual) 1i(res_stream_bytes) 1f(res_step)）
    scaler = exp.data_loader.scaler
    mean = float(scaler.mean.item() if hasattr(scaler.mean, 'item') else scaler.mean)
    std = float(scaler.std.item() if hasattr(scaler.std, 'item') else scaler.std)
    meta_buf = struct.pack(
        '<15i2f' + 'iif',
        _FORMAT_VERSION, int(exp.manager.total_length),
        int(codec.base_block_size), int(codec.min_resolution),
        int(codec.max_depth), int(cb.n_codes), int(cb.code_bytes),
        int(num_blocks), int(num_leaves), int(num_nodes), int(feature_dim),
        int(codec.K_fixed), int(dec['in_dim']), int(dec['hidden_dim']),
        int(dec['num_freqs']), mean, std,
        int(has_residual), int(res_stream_bytes), float(res_step),
    )
    assert len(meta_buf) == 80, f"meta.bin must be 80 bytes (v3), got {len(meta_buf)}"
    with open(os.path.join(out_dir, 'meta.bin'), 'wb') as fp:
        fp.write(meta_buf)

    # meta.json
    meta_json = {
        'format_version': _FORMAT_VERSION,
        'T': int(exp.manager.total_length),
        'base_block_size': int(codec.base_block_size),
        'min_resolution': int(codec.min_resolution),
        'max_depth': int(codec.max_depth),
        'n_codes': int(cb.n_codes), 'code_bytes': int(cb.code_bytes),
        'num_blocks': num_blocks, 'num_leaves': num_leaves,
        'num_nodes': num_nodes, 'feature_dim': feature_dim,
        'K_fixed': int(codec.K_fixed),
        'decoder': {'in_dim': dec['in_dim'], 'hidden_dim': dec['hidden_dim'],
                    'num_freqs': dec['num_freqs']},
        'scaler': {'mean': mean, 'std': std},
        'residual': {'has_residual': has_residual,
                     'stream_bytes': res_stream_bytes, 'step': res_step},
        'partitions': [list(p) for p in cb.partitions],
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as fp:
        json.dump(meta_json, fp, indent=2)

    total = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir))
    print(f"[export-blockcodec] {num_blocks} blocks, {num_leaves} leaves, "
          f"K_fixed={codec.K_fixed}, code_bytes={cb.code_bytes}")
    print(f"[export-blockcodec] grid {num_nodes}x{feature_dim}, "
          f"coeff_pool {num_leaves}x{codec.K_fixed + 2}")
    print(f"[export-blockcodec] DONE -> {out_dir}  total {total / 1024:.2f} KB")
    return out_dir


if __name__ == '__main__':
    import pickle, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from exp.exp_neurts import Exp_NeurTS

    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--threshold', type=float, default=None,
                    help='重建树用的误差阈值 ε（不传则用 args.json 的 eval_threshold/split_threshold，再无则 0.05）')
    ap.add_argument('--patch_fixed_K', type=int, default=8,
                    help='定长 patch 系数频率数 K（导出给 C++ 的 BlockCodec 用定长，默认 8）')
    ap.add_argument('--patch_fixed_mode', type=str, default='int8',
                    help='定长 patch 量化模式 int8|fp16（默认 int8）')
    a = ap.parse_args()

    with open(os.path.join(a.ckpt, 'args.json')) as f:
        args = argparse.Namespace(**json.load(f))
    args.skip_benchmark = True
    args.use_multi_gpu = False
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

    # C++ BlockCodec 需要定长 K：强制走 patch_fixed 路径，K/mode 由命令行给（不依赖旧 args.json）
    Kf = (a.patch_fixed_K,)
    modes = (a.patch_fixed_mode,)
    # 旧 checkpoint 的 args.json 可能缺这些后加的字段，用默认值兜底（不改 args.json、不重训）
    _max_depth = getattr(args, 'patch_max_depth', 3)
    _eval_thr = a.threshold if a.threshold is not None else \
                getattr(args, 'eval_threshold', getattr(args, 'split_threshold', 0.05))
    _err_mode = getattr(args, 'error_mode', 'absolute')
    print(f"  [export] eval_threshold={_eval_thr}, error_mode={_err_mode}, "
          f"max_depth={_max_depth}, K_fixed={a.patch_fixed_K}, mode={a.patch_fixed_mode}")
    exp._codec_build_residual = True   # 导出 C++ 需要残差落盘
    exp.multilayer_patch_eval(error_threshold=_eval_thr, error_mode=_err_mode,
                              K_list=Kf, modes=modes, max_depth=_max_depth, commit=True)
    export_blockcodec(exp, a.out)
