"""
Patch Registry: 坏点补丁注册表

用于存储无法被 TCN 精确拟合的异常点残差，实现 100% 误差可控的解码。

核心设计：
1. 以 Left_ID 为键的直接映射，与 IndexTable 的槽位对应
2. 每条记录包含：Count（坏点数量）+ Payload（偏移量+残差的紧凑数组）
3. 偏移量使用 uint8（相对于块左边界），残差使用 float16/float32

使用流程：
1. 训练完成后调用 collect_patches() 收集所有块的坏点
2. 推理时调用 apply_patches() 将残差加到基础波形上

示例（base_block_size=256, min_resolution=32）：
- 块 [0, 256) 由 Z0, Z1 定义
- 发现 3 个坏点：时间步 10, 100, 250
- 存储：Patch_Registry[Z0] = {count: 3, patches: [(10, res1), (100, res2), (250, res3)]}
- 推理时：output[10] += res1, output[100] += res2, output[250] += res3
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class PatchRecord:
    """单个块的补丁记录"""
    count: int  # 坏点数量
    offsets: List[int]  # 相对偏移量（相对于块左边界）
    residuals: List[float]  # 残差值
    
    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            'count': self.count,
            'offsets': self.offsets,
            'residuals': self.residuals
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'PatchRecord':
        """从字典反序列化"""
        return cls(
            count=d['count'],
            offsets=d['offsets'],
            residuals=d['residuals']
        )
    
    def get_memory_bytes(self) -> int:
        """计算内存占用（字节）"""
        # count: 2 bytes (uint16)
        # offsets: 1 byte each (uint8)
        # residuals: 2 bytes each (float16)
        return 2 + self.count * (1 + 2)


class PatchRegistry:
    """
    补丁注册表：以 Left_ID 为键存储坏点补丁。
    
    设计原则：
    1. 与 Grid Storage 并行存储，不修改原有数据结构
    2. 支持高效的键值查询
    3. 支持序列化/反序列化
    4. 支持量化压缩
    """
    
    def __init__(self, error_threshold: float = 0.1, use_float16: bool = True):
        """
        初始化补丁注册表。
        
        Args:
            error_threshold: 误差阈值，超过此值的点被视为坏点
            use_float16: 是否使用 float16 存储残差（节省空间）
        """
        self.error_threshold = error_threshold
        self.use_float16 = use_float16
        
        # 核心存储：Left_ID -> PatchRecord
        self._registry: Dict[int, PatchRecord] = {}
        
        # 统计信息
        self._total_patches = 0
        self._blocks_with_patches = 0
    
    def register(self, left_id: int, offsets: List[int], residuals: List[float]):
        """
        注册一个块的补丁。
        
        Args:
            left_id: 块的左边界节点 ID
            offsets: 坏点的相对偏移量列表
            residuals: 对应的残差值列表
        """
        if len(offsets) == 0:
            return
        
        assert len(offsets) == len(residuals), "offsets and residuals must have same length"
        
        # 如果使用 float16，转换残差
        if self.use_float16:
            residuals = [float(np.float16(r)) for r in residuals]
        
        record = PatchRecord(
            count=len(offsets),
            offsets=offsets,
            residuals=residuals
        )
        
        self._registry[left_id] = record
        self._total_patches += len(offsets)
        self._blocks_with_patches += 1
    
    def get(self, left_id: int) -> Optional[PatchRecord]:
        """
        获取指定块的补丁记录。
        
        Args:
            left_id: 块的左边界节点 ID
            
        Returns:
            PatchRecord 或 None（如果没有补丁）
        """
        return self._registry.get(left_id, None)
    
    def has_patches(self, left_id: int) -> bool:
        """检查指定块是否有补丁"""
        return left_id in self._registry
    
    def apply_to_output(
        self,
        output: torch.Tensor,
        left_id: int,
        block_start: int = 0
    ) -> torch.Tensor:
        """
        将补丁应用到输出张量。
        
        Args:
            output: 基础波形输出 [block_len] 或 [1, block_len]
            left_id: 块的左边界节点 ID
            block_start: 块的起始时间（用于计算绝对位置）
            
        Returns:
            修正后的输出张量
        """
        record = self.get(left_id)
        if record is None:
            return output
        
        # 确保输出是可修改的
        output = output.clone()
        
        # 应用补丁
        squeeze = False
        if output.dim() == 1:
            squeeze = True
            output = output.unsqueeze(0)
        
        for offset, residual in zip(record.offsets, record.residuals):
            if offset < output.shape[-1]:
                output[..., offset] += residual
        
        if squeeze:
            output = output.squeeze(0)
        
        return output
    
    def clear(self):
        """清空所有补丁"""
        self._registry.clear()
        self._total_patches = 0
        self._blocks_with_patches = 0
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        if self._blocks_with_patches == 0:
            avg_patches = 0
        else:
            avg_patches = self._total_patches / self._blocks_with_patches
        
        return {
            'total_patches': self._total_patches,
            'blocks_with_patches': self._blocks_with_patches,
            'total_blocks_registered': len(self._registry),
            'avg_patches_per_block': avg_patches,
            'memory_bytes': self.get_memory_usage(),
        }
    
    def get_memory_usage(self) -> int:
        """计算总内存占用（字节）"""
        total = 0
        for record in self._registry.values():
            total += record.get_memory_bytes()
        return total
    
    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            'error_threshold': self.error_threshold,
            'use_float16': self.use_float16,
            'registry': {str(k): v.to_dict() for k, v in self._registry.items()},
            'total_patches': self._total_patches,
            'blocks_with_patches': self._blocks_with_patches,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'PatchRegistry':
        """从字典反序列化"""
        registry = cls(
            error_threshold=d['error_threshold'],
            use_float16=d['use_float16']
        )
        registry._total_patches = d['total_patches']
        registry._blocks_with_patches = d['blocks_with_patches']
        registry._registry = {
            int(k): PatchRecord.from_dict(v) 
            for k, v in d['registry'].items()
        }
        return registry
    
    def save(self, path: str):
        """保存到文件"""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'PatchRegistry':
        """从文件加载"""
        import json
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


class PatchCollector:
    """
    补丁收集器：训练结束后遍历所有块，收集坏点残差。
    
    使用方法：
    ```python
    collector = PatchCollector(model, manager, raw_data, device)
    registry = collector.collect(error_threshold=0.1)
    ```
    """
    
    def __init__(
        self,
        model: nn.Module,
        manager,  # NeurTSManager
        raw_data: torch.Tensor,
        device: torch.device,
        base_block_size: int = 256
    ):
        """
        初始化补丁收集器。
        
        Args:
            model: 训练好的 NeurTS 模型
            manager: NeurTSManager 实例
            raw_data: 原始时间序列数据
            device: 计算设备
            base_block_size: 基础块大小
        """
        self.model = model
        self.manager = manager
        self.raw_data = raw_data
        self.device = device
        self.base_block_size = base_block_size
    
    def collect(
        self,
        error_threshold: float = 0.1,
        use_float16: bool = True,
        verbose: bool = True
    ) -> PatchRegistry:
        """
        收集所有块的坏点补丁。
        
        Args:
            error_threshold: 误差阈值
            use_float16: 是否使用 float16 存储残差
            verbose: 是否打印详细信息
            
        Returns:
            填充好的 PatchRegistry
        """
        registry = PatchRegistry(
            error_threshold=error_threshold,
            use_float16=use_float16
        )
        
        self.model.eval()
        all_blocks = self.manager.get_all_unique_blocks()
        
        if verbose:
            print(f"\n[PatchCollector] Collecting patches from {len(all_blocks)} blocks...")
            print(f"    Error threshold: {error_threshold}")
        
        total_bad_points = 0
        blocks_with_patches = 0
        
        with torch.no_grad():
            # 批量处理所有块
            left_ids = torch.tensor([b[2] for b in all_blocks], device=self.device)
            right_ids = torch.tensor([b[3] for b in all_blocks], device=self.device)
            
            # 固定长度解码
            outputs = self.model(left_ids, right_ids)  # [batch, 1, base_block_size]
            outputs = outputs.squeeze(1)  # [batch, base_block_size]
            
            for i, (start_time, end_time, left_id, right_id, level_code) in enumerate(all_blocks):
                block_len = end_time - start_time
                output = outputs[i, :block_len]
                true = self.raw_data[start_time:end_time]
                
                # 计算残差
                residual = true - output
                abs_error = torch.abs(residual)
                
                # 找出坏点（误差超过阈值）
                bad_mask = abs_error >= error_threshold
                bad_indices = torch.where(bad_mask)[0]
                
                if len(bad_indices) > 0:
                    # 收集坏点信息
                    offsets = bad_indices.cpu().tolist()
                    residuals = residual[bad_indices].cpu().tolist()
                    
                    # 注册补丁
                    registry.register(left_id, offsets, residuals)
                    
                    total_bad_points += len(offsets)
                    blocks_with_patches += 1
        
        if verbose:
            stats = registry.get_statistics()
            print(f"[PatchCollector] Collection complete:")
            print(f"    Blocks with patches: {stats['blocks_with_patches']}/{len(all_blocks)}")
            print(f"    Total patches: {stats['total_patches']}")
            print(f"    Avg patches per block: {stats['avg_patches_per_block']:.2f}")
            print(f"    Memory usage: {stats['memory_bytes'] / 1024:.2f} KB")
        
        return registry


def apply_patches_to_reconstruction(
    output: torch.Tensor,
    left_ids: torch.Tensor,
    registry: PatchRegistry,
    block_lens: Optional[List[int]] = None
) -> torch.Tensor:
    """
    批量应用补丁到重构输出。
    
    Args:
        output: 批量输出 [batch, block_size] 或 [batch, 1, block_size]
        left_ids: 左边界节点 ID [batch]
        registry: 补丁注册表
        block_lens: 每个块的实际长度（可选）
        
    Returns:
        修正后的输出
    """
    output = output.clone()
    
    squeeze = False
    if output.dim() == 3:
        squeeze = True
        output = output.squeeze(1)
    
    batch_size = output.shape[0]
    
    for i in range(batch_size):
        left_id = left_ids[i].item()
        record = registry.get(left_id)
        
        if record is not None:
            max_len = output.shape[-1]
            if block_lens is not None:
                max_len = min(max_len, block_lens[i])
            
            for offset, residual in zip(record.offsets, record.residuals):
                if offset < max_len:
                    output[i, offset] += residual
    
    if squeeze:
        output = output.unsqueeze(1)
    
    return output
