"""
FallbackDict: Tier 2/3 兜底字典

误差兜底方案的核心数据结构。以块的 Left_ID 为 Key，存储两种类型的兜底条目：

Tier 2 (PATCH) - 稀疏修补：
    仅记录少量坏点的 offset 和 residual，解码时先跑神经网络再叠加修补。
    Dict[left_id] = {"type": "PATCH", "offsets": [5, 12, 28], "residuals": [0.2, -0.3, 0.18]}

Tier 3 (RAW) - 整块直通：
    整个块的原始数据全部记录，解码时直接查字典，不走神经网络。
    Dict[left_id] = {"type": "RAW", "data": [0.5, 0.8, -0.1, ...]}

代价模型（落盘二进制格式）：
    - PATCH 条目头部: left_id(2B) + type(1B) + block_len(2B) + num_offsets(1B, uint8) + res_min(2B) + res_scale(2B) = 10B
    - PATCH 每个坏点: offset(1B, uint8) + quantized_residual(1B, int8) = 2 bytes
    - RAW 条目头部: left_id(2B) + type(1B) + block_len(2B) = 5B
    - RAW 每个点: 2 bytes (FP16，保证精度)
    - 一个 Grid 节点 = feature_dim × 1 byte (8-bit quantized)，如 32d → 32 bytes

    PATCH 残差量化方案：
    - 每个 PATCH 条目存储 res_min(FP16) 和 res_scale(FP16) 作为量化参数
    - 量化：quant_val = round((residual - res_min) / res_scale * 255)
    - 反量化：residual ≈ quant_val / 255 * res_scale + res_min
    - 量化误差 ≤ res_scale / 510，远小于误差阈值 ε，不影响可控性

Vector GC 规则：
    - 一个中间节点可被回收，当且仅当其左右两侧的块都不再需要神经网络解码（都是 RAW）
    - 回收后相邻 RAW 块可合并，减少字典条目
    - GC 后 patch_grid 会出现空洞，需要在最终落盘前做一次 compaction（重排 + 更新 ID）
"""

import math

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class FallbackDict:
    """Tier 2/3 兜底字典：存储神经网络解码失败的块的补救数据"""
    
    # 代价常量（落盘二进制格式）
    PATCH_BYTES_PER_POINT = 2   # offset(1B, uint8) + quantized_residual(1B, int8)
    RAW_BYTES_PER_POINT = 2     # FP16 raw value (保证精度)
    
    # 条目元数据开销（每个字典条目的固定开销）
    PATCH_ENTRY_OVERHEAD = 10   # left_id(2) + type(1) + block_len(2) + num_offsets(1, uint8) + res_min(2) + res_scale(2)
    RAW_ENTRY_OVERHEAD = 5      # left_id(2) + type(1) + block_len(2)
    
    # 隐式二叉树索引（中分 + bitmask 编码）
    # 中分 → 树结构完全隐式，只需 1 byte/base_block 的 bitmask 标记哪些层级 active
    # 256-point base block → 最多 7 层分裂 → 7 bits < 1 byte
    INDEX_PER_BASE_BLOCK = 1    # 每个 base block 的索引开销 (bitmask, 1B)
    
    # 动态位宽量化 (Error-Bounded Dynamic Bit-width Quantization)
    # 条目头: R_min(2B, FP16) + bits_per_point(1B, uint8) = 3B
    # 条目体: ceil(block_len * bits / 8) bytes (packed bit array)
    BITWIDTH_HEADER = 3         # R_min(2B) + bits(1B)
    
    def __init__(self, node_cost_bytes: int = 32):
        """
        Args:
            node_cost_bytes: 一个 Grid 节点的字节开销（8-bit quant 下 = feature_dim）
        """
        self.entries: Dict[int, dict] = {}  # left_id -> entry
        self.node_cost_bytes = node_cost_bytes
    
    # =========================================================================
    # 条目操作
    # =========================================================================
    
    def add_patch(self, left_id: int, block_len: int, 
                  offsets: List[int], residuals: List[float]):
        """
        添加 Tier 2 稀疏修补条目。
        
        Args:
            left_id: 块的左边界节点 ID（字典 Key）
            block_len: 块长度
            offsets: 坏点在块内的偏移位置列表
            residuals: 对应位置的残差值列表（true - decoded）
        """
        self.entries[left_id] = {
            'type': 'PATCH',
            'block_len': block_len,
            'offsets': list(offsets),
            'residuals': list(residuals),
        }
    
    def add_raw(self, left_id: int, block_len: int, data: List[float]):
        """
        添加 Tier 3 整块直通条目。
        
        Args:
            left_id: 块的左边界节点 ID（字典 Key）
            block_len: 块长度
            data: 块内全部原始数据点
        """
        self.entries[left_id] = {
            'type': 'RAW',
            'block_len': block_len,
            'data': list(data),
        }
    
    def remove(self, left_id: int):
        """移除条目"""
        if left_id in self.entries:
            del self.entries[left_id]
    
    def get(self, left_id: int) -> Optional[dict]:
        """获取条目，不存在返回 None"""
        return self.entries.get(left_id, None)
    
    def has(self, left_id: int) -> bool:
        """检查是否存在条目"""
        return left_id in self.entries
    
    def get_type(self, left_id: int) -> Optional[str]:
        """获取条目类型：'PATCH'/'RAW'/None"""
        entry = self.entries.get(left_id)
        return entry['type'] if entry else None
    
    def is_raw(self, left_id: int) -> bool:
        """判断是否为 RAW 条目"""
        entry = self.entries.get(left_id)
        return entry is not None and entry['type'] == 'RAW'
    
    def is_patch(self, left_id: int) -> bool:
        """判断是否为 PATCH 条目"""
        entry = self.entries.get(left_id)
        return entry is not None and entry['type'] == 'PATCH'
    
    # =========================================================================
    # 代价计算
    # =========================================================================
    
    def entry_cost(self, left_id: int) -> int:
        """单条目的字节开销（含元数据）"""
        entry = self.entries.get(left_id)
        if entry is None:
            return 0
        if entry['type'] == 'PATCH':
            return self.PATCH_ENTRY_OVERHEAD + len(entry['offsets']) * self.PATCH_BYTES_PER_POINT
        else:  # RAW
            return self.RAW_ENTRY_OVERHEAD + entry['block_len'] * self.RAW_BYTES_PER_POINT
    
    def total_cost(self) -> int:
        """字典全部条目的总字节开销（含元数据）"""
        return sum(self.entry_cost(lid) for lid in self.entries)
    
    @classmethod
    def estimate_patch_cost(cls, bad_point_count: int, include_overhead: bool = True) -> int:
        """估算 PATCH 代价（字节）"""
        data_cost = bad_point_count * cls.PATCH_BYTES_PER_POINT
        return (cls.PATCH_ENTRY_OVERHEAD + data_cost) if include_overhead else data_cost
    
    @classmethod
    def estimate_raw_cost(cls, block_len: int, include_overhead: bool = True) -> int:
        """估算 RAW 代价（字节）"""
        data_cost = block_len * cls.RAW_BYTES_PER_POINT
        return (cls.RAW_ENTRY_OVERHEAD + data_cost) if include_overhead else data_cost
    
    def estimate_node_cost(self) -> int:
        """一个 Grid 节点的代价（字节）"""
        return self.node_cost_bytes
    
    @staticmethod
    def compute_bits(span: float, epsilon: float) -> int:
        """
        根据残差极差和误差容限计算每点所需位宽。
        
        bucket_size = 2 * epsilon（保证量化误差 ≤ epsilon）
        num_buckets = ceil(span / bucket_size)
        bits = ceil(log2(num_buckets))
        
        特殊情况：span ≤ 2*epsilon → 0 bits（0-bit 奇迹：仅存 R_min 即可）
        
        Args:
            span: 残差极差 R_max - R_min（归一化空间）
            epsilon: 误差容限（归一化空间）
        Returns:
            bits_per_point: 0~8
        """
        if epsilon <= 0:
            return 16
        bucket_size = 2.0 * epsilon
        if span <= bucket_size:
            return 0
        num_buckets = math.ceil(span / bucket_size)
        bits = math.ceil(math.log2(num_buckets))
        return min(bits, 16)  # cap at 16 bits (8-bit truncates large span, breaks error bound)
    
    @classmethod
    def estimate_bitwidth_cost(cls, span: float, block_len: int, epsilon: float) -> int:
        """
        估算动态位宽条目的字节代价。
        
        cost = BITWIDTH_HEADER(3B) + ceil(block_len * bits / 8)
        0-bit 时仅 3B header，无 body。
        
        Args:
            span: 残差极差（归一化空间）
            block_len: 块长度（点数）
            epsilon: 误差容限（归一化空间）
        Returns:
            总字节代价
        """
        bits = cls.compute_bits(span, epsilon)
        body_bytes = math.ceil(block_len * bits / 8) if bits > 0 else 0
        return cls.BITWIDTH_HEADER + body_bytes

    @classmethod
    def estimate_bitwidth_cost_grouped(cls, residual, epsilon: float,
                                       num_groups: int,
                                       min_group_size: int = 16) -> int:
        """
        Sub-group 量化代价：把残差分成 num_groups 组，每组独立计算 span 和 bits。

        相比单 span 方案，离群点只影响自己所在的组，其余组不受影响，
        可显著降低 avg bits（以每组多 1 个 3B header 为代价）。

        effective_groups = max(1, min(num_groups, block_len // min_group_size))
        当 effective_groups==1 时等价于 estimate_bitwidth_cost。

        Args:
            residual : Tensor 或 list/ndarray，残差序列（带符号）
            epsilon  : 误差容限（归一化空间）
            num_groups     : 期望组数（典型值 4）
            min_group_size : 每组最小点数，防止过细分组（默认 16）
        Returns:
            总字节代价（所有组的 header + body 之和）
        """
        import torch as _torch
        n = len(residual)
        effective_g = max(1, min(num_groups, n // min_group_size))

        if effective_g == 1:
            if isinstance(residual, _torch.Tensor):
                span = (residual.max() - residual.min()).item()
            else:
                span = float(max(residual)) - float(min(residual))
            return cls.estimate_bitwidth_cost(span, n, epsilon)

        group_size = math.ceil(n / effective_g)
        total_cost = 0
        for g in range(effective_g):
            r_g = residual[g * group_size: (g + 1) * group_size]
            glen = len(r_g)
            if glen == 0:
                continue
            if isinstance(r_g, _torch.Tensor):
                span_g = (r_g.max() - r_g.min()).item()
            else:
                span_g = float(max(r_g)) - float(min(r_g))
            bits_g = cls.compute_bits(span_g, epsilon)
            body_g = math.ceil(glen * bits_g / 8) if bits_g > 0 else 0
            total_cost += cls.BITWIDTH_HEADER + body_g
        return total_cost
    
    # =========================================================================
    # 合并操作（Vector GC 使用）
    # =========================================================================
    
    def merge_adjacent_raw(self, left_id_1: int, left_id_2: int, 
                           merged_left_id: int):
        """
        合并两个相邻的 RAW 条目为一个。
        
        前提：left_id_1 的块紧邻 left_id_2 的块（它们共享一个中间节点）。
        合并后使用 merged_left_id 作为新 Key（通常等于 left_id_1）。
        
        Args:
            left_id_1: 第一个 RAW 块的 left_id
            left_id_2: 第二个 RAW 块的 left_id
            merged_left_id: 合并后的 left_id（通常为 left_id_1）
        """
        entry1 = self.entries.get(left_id_1)
        entry2 = self.entries.get(left_id_2)
        
        if entry1 is None or entry2 is None:
            raise ValueError(f"Both entries must exist: {left_id_1}, {left_id_2}")
        if entry1['type'] != 'RAW' or entry2['type'] != 'RAW':
            raise ValueError(f"Both entries must be RAW for merge")
        
        merged_data = entry1['data'] + entry2['data']
        merged_len = entry1['block_len'] + entry2['block_len']
        
        # 删除旧条目
        del self.entries[left_id_1]
        del self.entries[left_id_2]
        
        # 创建合并后的条目
        self.entries[merged_left_id] = {
            'type': 'RAW',
            'block_len': merged_len,
            'data': merged_data,
        }
    
    # =========================================================================
    # 解码重构（给定 decoded output，叠加字典修补）
    # =========================================================================
    
    def reconstruct(self, left_id: int, decoded_output: torch.Tensor) -> torch.Tensor:
        """
        用字典条目修补或替换神经网络的解码输出。
        
        - Tier 1（无条目）：直接返回 decoded_output
        - Tier 2（PATCH）：在坏点位置叠加残差
        - Tier 3（RAW）：完全替换为原始数据
        
        Args:
            left_id: 块的 left_id
            decoded_output: 神经网络解码输出，shape [block_len] 或 [1, block_len]
            
        Returns:
            修补后的输出，shape 同 decoded_output
        """
        entry = self.entries.get(left_id)
        if entry is None:
            return decoded_output
        
        result = decoded_output.clone()
        squeeze = False
        if result.dim() == 1:
            result = result.unsqueeze(0)
            squeeze = True
        
        if entry['type'] == 'PATCH':
            offsets = entry['offsets']
            residuals = torch.tensor(entry['residuals'], 
                                     dtype=result.dtype, device=result.device)
            for i, offset in enumerate(offsets):
                if offset < result.shape[-1]:
                    result[..., offset] += residuals[i]
        elif entry['type'] == 'RAW':
            raw_data = torch.tensor(entry['data'], 
                                    dtype=result.dtype, device=result.device)
            result[..., :len(raw_data)] = raw_data
        
        if squeeze:
            result = result.squeeze(0)
        return result
    
    # =========================================================================
    # 统计与摘要
    # =========================================================================
    
    def summary(self) -> dict:
        """统计摘要"""
        patch_entries = [e for e in self.entries.values() if e['type'] == 'PATCH']
        raw_entries = [e for e in self.entries.values() if e['type'] == 'RAW']
        
        total_patch_points = sum(len(e['offsets']) for e in patch_entries)
        total_raw_points = sum(e['block_len'] for e in raw_entries)
        
        patch_bytes = len(patch_entries) * self.PATCH_ENTRY_OVERHEAD + total_patch_points * self.PATCH_BYTES_PER_POINT
        raw_bytes = len(raw_entries) * self.RAW_ENTRY_OVERHEAD + total_raw_points * self.RAW_BYTES_PER_POINT
        
        return {
            'total_entries': len(self.entries),
            'patch_entries': len(patch_entries),
            'raw_entries': len(raw_entries),
            'total_patch_points': total_patch_points,
            'total_raw_points': total_raw_points,
            'patch_bytes': patch_bytes,
            'raw_bytes': raw_bytes,
            'total_bytes': patch_bytes + raw_bytes,
            'node_cost_bytes': self.node_cost_bytes,
        }
    
    def print_summary(self):
        """打印统计摘要"""
        s = self.summary()
        print("\n[Fallback Dictionary Summary]")
        print(f"    Total Entries:     {s['total_entries']}")
        print(f"    PATCH entries:     {s['patch_entries']} ({s['total_patch_points']} bad points)")
        print(f"    RAW entries:       {s['raw_entries']} ({s['total_raw_points']} raw points)")
        print(f"    PATCH cost:        {s['patch_bytes']/1024:.2f} KB")
        print(f"    RAW cost:          {s['raw_bytes']/1024:.2f} KB")
        print(f"    Total dict cost:   {s['total_bytes']/1024:.2f} KB")
        print(f"    Node unit cost:    {s['node_cost_bytes']} bytes/node")
    
    # =========================================================================
    # 序列化（落盘用）
    # =========================================================================
    
    def state_dict(self) -> dict:
        """导出为可序列化的字典"""
        return {
            'entries': self.entries.copy(),
            'node_cost_bytes': self.node_cost_bytes,
        }
    
    def load_state_dict(self, state: dict):
        """从字典恢复"""
        self.entries = state['entries']
        self.node_cost_bytes = state.get('node_cost_bytes', self.node_cost_bytes)
    
    def __len__(self):
        return len(self.entries)
    
    def __repr__(self):
        s = self.summary()
        return (f"FallbackDict(entries={s['total_entries']}, "
                f"patch={s['patch_entries']}, raw={s['raw_entries']}, "
                f"cost={s['total_bytes']/1024:.2f}KB)")
