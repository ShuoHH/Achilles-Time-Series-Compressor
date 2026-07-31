"""
NeurTS Dataset: 任务驱动数据集 (Legacy)

注意：此模块为旧版兼容接口，基于 task_list 驱动。
新项目请使用 data_loader.py 中的 NeurTSDataset（基于 Level Code 的掩码对齐方案）。
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple


class NeurTSTaskDataset(Dataset):
    """
    [Legacy] 任务驱动数据集：完全由task_list驱动，支持变长波形块。
    
    警告：此类为旧版兼容接口。新项目请使用 data_loader.NeurTSDataset。
    
    核心思想：由 task_list 指定每个样本的 (start_tick, end_tick, left_id, right_id)。
    由于支持递归分裂，切出来的波形长度可能不一（如64, 32, 16），
    因此需要将其Zero-Pad到固定的max_block_size。
    
    Args:
        raw_data: 原始时序数据，形状为 [total_length] 或 [total_length, channels]
        task_list: 任务列表 [(start_tick, end_tick, left_id, right_id), ...]
        max_block_size: 最大块长度，用于padding
    """
    
    def __init__(
        self,
        raw_data: torch.Tensor,
        task_list: List[Tuple[int, int, int, int]],
        max_block_size: int
    ):
        self.raw_data = raw_data
        self.task_list = task_list
        self.max_block_size = max_block_size
    
    def __len__(self) -> int:
        return len(self.task_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        获取单个训练样本。
        
        Returns:
            padded_waveform: Zero-Padded波形 [max_block_size]
            valid_length_mask: 有效长度掩码 [max_block_size], 1表示有效, 0表示padding
            left_id: 左边界节点ID
            right_id: 右边界节点ID
        """
        start_tick, end_tick, left_id, right_id = self.task_list[idx]
        
        # 切片原始数据
        waveform = self.raw_data[start_tick:end_tick]
        valid_length = end_tick - start_tick
        
        # Zero-Pad到max_block_size
        padded_waveform = torch.zeros(self.max_block_size, dtype=self.raw_data.dtype)
        padded_waveform[:valid_length] = waveform
        
        # 创建有效长度掩码
        valid_length_mask = torch.zeros(self.max_block_size, dtype=torch.float32)
        valid_length_mask[:valid_length] = 1.0
        
        return padded_waveform, valid_length_mask, left_id, right_id
    
    def update_task_list(self, new_task_list: List[Tuple[int, int, int, int]]):
        """更新任务列表（用于递归分裂后刷新数据集）。"""
        self.task_list = new_task_list
    
    def get_block_size_distribution(self) -> dict:
        """
        获取当前任务列表中块大小的分布统计。
        
        Returns:
            包含min, max, avg等统计信息的字典
        """
        block_sizes = [task[1] - task[0] for task in self.task_list]
        
        if not block_sizes:
            return {"min": 0, "max": 0, "avg": 0, "count": 0}
        
        return {
            "min": min(block_sizes),
            "max": max(block_sizes),
            "avg": sum(block_sizes) / len(block_sizes),
            "count": len(block_sizes)
        }
