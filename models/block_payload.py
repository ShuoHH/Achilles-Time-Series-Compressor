"""
统一 payload 序列化（方式 A）：把 BlockCodec(+ResidualCodec) 的存储序列化成
"一个主表 + 一个局部 payload" 的规范格式，替代原来"表 + 字典"的表述。

设计
----
主表（按 block_id 扁平，无字典/哈希）：
    tree_code[b]        隐式树结构码
    payload_offset[b]   该块数据在 payload 中的起始字节

局部 payload（单一连续缓冲，按 block_id、块内按叶子顺序组织）：
    对每个被 patch 的块，先 2B 元数据（段数 = PATCH_META_BYTES）；
    然后每个叶子依次：
        res_bits   : 1B  uint8   残差位宽 b
        res_rmin   : 2B  fp16     残差最小值
        [仅 patched 叶子] coeff : (K+2)×int8 + coeff_min(fp16) + coeff_scale(fp16)  = (K+2)+4 B
        residual   : ceil(L·b/8) B  EDWB 残差码流（逐叶子字节对齐，MSB-first）

要点
----
- 逐叶子字节对齐的残差 + int8 系数 → payload 尺寸**精确等于**解析式压缩比里
  estimate_bitwidth_cost / patch_coeff_bytes 的计数（证明该比值可被真实序列化达到）。
- "被 patch ⟺ 分裂(叶子数>1)"：单叶子块不带系数（与解析式一致，只有分裂块计系数）。
- 本模块只做序列化/反序列化；随机访问仍走 BlockCodec 的快数组（load 时解包），
  访问速度与压缩比都不受影响。
"""

import math
from typing import Dict, Tuple

import numpy as np
import torch

from models.block_codec import BlockCodec, TreeCodebook
from models.patch_split import PATCH_META_BYTES, _INT8_HEADER

_COEFF_HEADER = _INT8_HEADER      # 4B: coeff_min(fp16) + coeff_scale(fp16)
_RES_HEADER = 3                   # 1B res_bits + 2B res_rmin


def _leaf_len_list(codebook: TreeCodebook, code: int, block_start: int, block_len: int):
    return [(e - s) for (s, e) in codebook.leaf_bounds(code, block_start, block_len)]


def serialize(codec: BlockCodec) -> dict:
    """
    序列化为 {tree_codes, payload_offsets, payload(bytes), meta}。
    meta 含反序列化所需的全局常量（base_block_size / max_depth / K_fixed / res_step / T）。
    """
    if not codec._finalized:
        codec.finalize()
    rc = codec.residual_codec
    step = float(rc.step) if rc is not None else 0.0
    K = codec.K_fixed
    n_coeff = K + 2
    bbs = codec.base_block_size

    block_ids = sorted(codec.blocks.keys())
    T = max(r.block_start + r.block_len for r in codec.blocks.values())

    tree_codes = np.zeros(len(block_ids), dtype=np.uint16)
    payload_offsets = np.zeros(len(block_ids), dtype=np.uint32)
    buf = bytearray()

    for bi, bid in enumerate(block_ids):
        rec = codec.blocks[bid]
        tree_codes[bi] = rec.code
        payload_offsets[bi] = len(buf)
        n_leaves = len(rec.leaves)
        patched = n_leaves > 1

        if patched:
            buf += int(n_leaves).to_bytes(PATCH_META_BYTES, 'little')  # 2B 段数元数据

        for li, lf in enumerate(rec.leaves):
            row = codec.leaf_global_index(bid, li)
            L = lf.end - lf.start

            # --- 残差元数据 + 系数 ---
            if rc is not None:
                b = int(rc.leaf_bits[row]); rmin = float(rc.leaf_rmin[row])
            else:
                b = 0; rmin = 0.0
            buf += np.uint8(b).tobytes()
            buf += np.float16(rmin).tobytes()

            if patched:
                d = codec._coeff_pool[row].detach().cpu().numpy().astype(np.float64)
                vmin = float(d.min()); vmax = float(d.max())
                scale = (vmax - vmin) / 255.0 if vmax > vmin else 0.0
                if scale > 0:
                    q8 = np.rint((d - vmin) / scale).clip(0, 255).astype(np.uint8)
                else:
                    q8 = np.zeros(n_coeff, dtype=np.uint8)
                buf += q8.tobytes()                       # (K+2) B
                buf += np.float16(vmin).tobytes()         # 2B
                buf += np.float16(scale).tobytes()        # 2B

            # --- EDWB 残差码流（逐叶子字节对齐）---
            if rc is not None and b > 0:
                r_hat = rc.decode_leaf(row)               # 反量化 [L]
                q = np.rint((r_hat - rmin) / step).astype(np.int64)
                q = np.clip(q, 0, (1 << b) - 1)
                shifts = np.arange(b - 1, -1, -1, dtype=np.int64)
                bits = ((q[:, None] >> shifts[None, :]) & 1).astype(np.uint8).reshape(-1)
                buf += np.packbits(bits).tobytes()        # ceil(L*b/8) B

    return {
        'tree_codes': tree_codes,
        'payload_offsets': payload_offsets,
        'payload': bytes(buf),
        'meta': {
            'base_block_size': bbs, 'max_depth': codec.max_depth,
            'K_fixed': K, 'res_step': step, 'T': T,
            'num_blocks': len(block_ids), 'block_ids': block_ids,
        },
    }


def deserialize(blob: dict) -> dict:
    """
    从 serialize() 的产物还原 {coeff_pool[num_leaves,K+2], residual{row->np[L]}}，
    用于校验/重载。行号与原 codec 的全局叶子顺序一致。
    """
    meta = blob['meta']
    payload = blob['payload']
    tree_codes = blob['tree_codes']
    offsets = blob['payload_offsets']
    K = meta['K_fixed']; n_coeff = K + 2
    bbs = meta['base_block_size']; T = meta['T']; step = meta['res_step']
    cb = TreeCodebook(meta['max_depth'])
    block_ids = meta['block_ids']

    coeff_rows = []
    residual = {}
    row = 0
    for bi, bid in enumerate(block_ids):
        p = int(offsets[bi])
        code = int(tree_codes[bi])
        block_start = bid * bbs
        block_len = min(bbs, T - block_start)
        lens = _leaf_len_list(cb, code, block_start, block_len)
        patched = len(lens) > 1

        if patched:
            p += PATCH_META_BYTES   # 跳过 2B 段数元数据

        for L in lens:
            b = int(np.frombuffer(payload, np.uint8, 1, p)[0]); p += 1
            rmin = float(np.frombuffer(payload, np.float16, 1, p)[0]); p += 2

            if patched:
                q8 = np.frombuffer(payload, np.uint8, n_coeff, p).astype(np.float32); p += n_coeff
                vmin = float(np.frombuffer(payload, np.float16, 1, p)[0]); p += 2
                scale = float(np.frombuffer(payload, np.float16, 1, p)[0]); p += 2
                d_hat = vmin + q8 * scale
                coeff_rows.append(torch.from_numpy(d_hat.astype(np.float32)))
            else:
                coeff_rows.append(torch.zeros(n_coeff, dtype=torch.float32))

            if b > 0:
                nbytes = math.ceil(L * b / 8)
                raw = np.frombuffer(payload, np.uint8, nbytes, p); p += nbytes
                bits = np.unpackbits(raw)[: L * b].reshape(L, b)
                weights = (1 << np.arange(b - 1, -1, -1, dtype=np.int64))[None, :]
                q = (bits.astype(np.int64) * weights).sum(axis=1)
                residual[row] = (rmin + q.astype(np.float32) * step).astype(np.float32)
            else:
                residual[row] = np.full(L, rmin, dtype=np.float32)
            row += 1

    coeff_pool = torch.stack(coeff_rows, dim=0) if coeff_rows else torch.zeros(0, n_coeff)
    return {'coeff_pool': coeff_pool, 'residual': residual, 'num_leaves': row}


def payload_size(blob: dict) -> int:
    return len(blob['payload'])
