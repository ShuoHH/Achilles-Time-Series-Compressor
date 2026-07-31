import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from utils.tools import StandardScaler

import warnings
warnings.filterwarnings('ignore')

from typing import List, Tuple, Optional


# =============================================================================
# NeurTS Data Loader: 数据加载与预处理
# =============================================================================

class NeurTSDataLoader:
    """
    NeurTS 数据加载器：负责从文件加载数据并进行预处理。
    
    职责：
    1. 从CSV文件读取原始时序数据
    2. 数据标准化
    3. 提供原始数据供GlobalGridStorage初始化
    
    与其他组件的协作：
    - GlobalGridStorage: 使用 get_raw_data() 初始化grid
    - GridManager: 直接使用 get_raw_data() 初始化索引表
    - NeurTSDataset: 使用 get_raw_data() 和 manager 进行训练采样
    """
    
    def __init__(
        self,
        root_path: str,
        data_path: str,
        block_size: int = 64,
        scale: bool = True,
        data_col: int = 2,
    ):
        """
        初始化数据加载器。
        
        Args:
            root_path: 数据根目录
            data_path: 数据文件名
            block_size: 初始块大小
            scale: 是否进行标准化
            data_col: 数据所在列索引（0-indexed）
        """
        self.root_path = root_path
        self.data_path = data_path
        self.block_size = block_size
        self.scale = scale
        self.data_col = data_col
        
        self.scaler = None
        self.data = None
        self.raw_len = 0
        self.num_blocks = 0
        
        # 加载数据
        self._load_data()
    
    def _load_data(self):
        """从CSV文件加载并预处理数据。"""
        file_path = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(file_path, header=None, usecols=[self.data_col])
        
        self.raw_len = len(df_raw)
        
        # 标准化
        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(df_raw.values)
            data_np = self.scaler.transform(df_raw.values)
        else:
            data_np = df_raw.values
        
        # 转换为Tensor并展平为1D
        self.data = torch.from_numpy(data_np).float().squeeze()
        
        # 计算Block数量
        self.num_blocks = self.raw_len // self.block_size
        
        print(f"[NeurTSDataLoader] Loaded {self.raw_len} samples, "
              f"block_size={self.block_size}, num_blocks={self.num_blocks}")
    
    def get_raw_data(self) -> torch.Tensor:
        """
        获取原始数据（用于GlobalGridStorage初始化）。
        
        Returns:
            原始时序数据 [total_length]
        """
        return self.data
    
    def get_num_base_nodes(self) -> int:
        """
        获取基础网格节点数量。
        
        Returns:
            num_blocks + 1（N个块需要N+1个边界节点）
        """
        return self.num_blocks + 1
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        反标准化（用于可视化）。
        
        Args:
            data: 标准化后的数据
            
        Returns:
            原始尺度的数据
        """
        if self.scaler is None:
            return data
        
        data_np = data.cpu().numpy()
        if data_np.ndim == 1:
            data_np = data_np.reshape(-1, 1)
        
        return torch.from_numpy(self.scaler.inverse_transform(data_np)).squeeze()


# =============================================================================
# NeurTSDataset: 基于层级码的掩码对齐数据集
# =============================================================================

class NeurTSDataset(Dataset):
    """
    基于层级码 (Level Code) 的掩码对齐数据集。
    
    核心设计：
    - 每个样本对应一个 min_resolution 长度的时间窗口
    - 通过 level_code 计算当前块的实际长度
    - 使用 mask 标记当前窗口在块内的位置
    - TCN 输入固定为 base_block_size，通过 mask 实现对齐
    
    示例 (base=100, min=25):
    - 查询 t=125 (idx=5)
    - 若未分裂 (level=0, len=100):
        - win_start = 125
        - offset = 125 % 100 = 25
        - mask[25:50] = 1 (表示这是大块的第二部分)
    - 若已分裂 (level=2, len=25):
        - block_len = 25
        - offset = 125 % 25 = 0
        - mask[0:25] = 1 (表示这是一个独立的小块)
    
    返回格式：
        (left_id, right_id, ground_truth, mask, offset, block_len)
        - left_id: 左边界Grid节点ID (int)
        - right_id: 右边界Grid节点ID (int)
        - ground_truth: 当前块的真实数据 [base_block_size]，前 block_len 为有效値
        - mask: 对齐掩码 [base_block_size]，前 block_len 为 1
        - offset: 块起始时间在基础块内的物理偏移 (int)，用于全局 t_global 计算
        - block_len: 块的实际长度 (int)
    
    Args:
        raw_data: 原始时序数据 [total_length]
        manager: GridManager 实例
    """
    
    def __init__(
        self,
        raw_data: torch.Tensor,
        manager,  # GridManager
        grid_storage = None,  # 保留参数兼容性，但不再使用
    ):
        super().__init__()
        
        self.raw_data = raw_data
        self.manager = manager
        self.grid_storage = grid_storage
        
        self.min_resolution = manager.min_resolution
        self.base_block_size = manager.base_block_size
        self.num_slots = manager.num_slots
        
        print(f"[NeurTSDataset] Initialized: num_slots={self.num_slots}, "
              f"base_block_size={self.base_block_size}, min_resolution={self.min_resolution}")
    
    def __len__(self) -> int:
        """返回槽位数量（每个槽位对应一个 min_resolution 窗口）。"""
        return self.num_slots
    
    def __getitem__(self, idx: int):
        """
        获取第 idx 个槽位对应的 block 训练样本。
        
        核心逻辑（支持 Argmax 非中点分裂）：
        1. 查表获取当前槽位的 (left_id, right_id, level_code)
        2. 向左遍历找到该 block 的起始槽位
        3. 向右遍历找到该 block 的结束槽位
        4. 通过遍历结果计算 block 实际长度（不再依赖 level_code）
        5. 构建 mask：前 block_len 为 1，后面为 0
        6. 取 ground_truth：从 block 起始位置取 block_len 长度
        
        Args:
            idx: 槽位索引
            
        Returns:
            tuple: (left_id, right_id, ground_truth, mask, offset)
                - left_id: 左边界 Grid 节点 ID
                - right_id: 右边界 Grid 节点 ID
                - ground_truth: 当前 block 的真实数据 [block_len]，padding 到 [base_block_size]
                - mask: 对齐掩码 [base_block_size]，前 block_len 为 1
                - offset: 块起始时间在基础块内的物理偏移，用于全局 t_global
                - block_len: 块的实际长度
        """
        # 1. 查表获取块信息
        entry = self.manager.index_table[idx]
        left_id = entry.left_id
        right_id = entry.right_id
        level_code = entry.level_code
        
        # 2. 向左遍历找到该 block 的起始槽位
        block_start_idx = idx
        while block_start_idx > 0:
            prev = self.manager.index_table[block_start_idx - 1]
            if (prev.left_id == left_id and prev.right_id == right_id and 
                prev.level_code == level_code):
                block_start_idx -= 1
            else:
                break
        
        # 3. 向右遍历找到该 block 的结束槽位
        block_end_idx = idx
        while block_end_idx < self.num_slots - 1:
            next_entry = self.manager.index_table[block_end_idx + 1]
            if (next_entry.left_id == left_id and next_entry.right_id == right_id and
                next_entry.level_code == level_code):
                block_end_idx += 1
            else:
                break
        
        # 4. 通过遍历结果计算 block 实际长度（支持非中点分裂）
        block_start_time = block_start_idx * self.min_resolution
        block_end_time = (block_end_idx + 1) * self.min_resolution
        block_len = block_end_time - block_start_time
        
        offset = block_start_time  # 块在时序中的绝对起始坐标（acorn1d 使用全局归一化）
        
        # 5. 构建 Mask
        # TCN 输出固定为 base_block_size，mask 标记前 block_len 为有效区域
        mask = torch.zeros(self.base_block_size, dtype=torch.float32)
        effective_len = min(block_len, self.base_block_size)
        mask[:effective_len] = 1.0
        
        # 6. 读取 ground_truth：从 block 起始位置取 block_len 长度，padding 到 base_block_size
        # 注意：末尾边界检查，防止超出 raw_data 长度
        ground_truth = torch.zeros(self.base_block_size, dtype=self.raw_data.dtype)
        actual_end = min(block_start_time + block_len, len(self.raw_data))
        actual_len = min(actual_end - block_start_time, self.base_block_size)
        if actual_len > 0:
            ground_truth[:actual_len] = self.raw_data[block_start_time:block_start_time + actual_len]
            # 如果实际长度小于 effective_len，更新 mask
            if actual_len < effective_len:
                mask[actual_len:effective_len] = 0.0
        
        # 7. 返回（包含 block_len 用于动态解码）
        return left_id, right_id, ground_truth, mask, offset, block_len
    
    def get_sample_info(self, idx: int) -> dict:
        """
        获取样本的详细信息（用于调试）。
        
        Args:
            idx: 槽位索引
            
        Returns:
            包含样本详细信息的字典
        """
        entry = self.manager.index_table[idx]
        win_start = idx * self.min_resolution
        
        # 找到块起始位置（向左遍历）
        block_start_idx = idx
        while block_start_idx > 0:
            prev = self.manager.index_table[block_start_idx - 1]
            if (prev.left_id == entry.left_id and prev.right_id == entry.right_id and 
                prev.level_code == entry.level_code):
                block_start_idx -= 1
            else:
                break
        
        # 找到块结束位置（向右遍历）
        block_end_idx = idx
        while block_end_idx < self.num_slots - 1:
            next_entry = self.manager.index_table[block_end_idx + 1]
            if (next_entry.left_id == entry.left_id and next_entry.right_id == entry.right_id and
                next_entry.level_code == entry.level_code):
                block_end_idx += 1
            else:
                break
        
        block_start_time = block_start_idx * self.min_resolution
        block_end_time = (block_end_idx + 1) * self.min_resolution
        block_len = block_end_time - block_start_time
        offset = win_start - block_start_time
        
        return {
            "idx": idx,
            "win_start": win_start,
            "win_end": win_start + self.min_resolution,
            "left_id": entry.left_id,
            "right_id": entry.right_id,
            "level_code": entry.level_code,
            "block_len": block_len,
            "block_start_time": block_start_time,
            "block_end_time": block_end_time,
            "offset": offset,
        }


if __name__ == '__main__':
    # 测试 NeurTSDataset
    print("=" * 60)
    print("Test: NeurTSDataset with Level Code")
    print("=" * 60)
    
    from models.grid import GlobalGridStorage
    from models.neurts_model import NeurTSModel
    from models.neurts_manager import GridManager
    
    total_length = 400
    base_block_size = 100
    min_resolution = 25
    
    raw_data = torch.sin(torch.linspace(0, 8 * 3.14159, total_length))
    
    grid_storage = GlobalGridStorage(
        raw_data=raw_data,
        block_size=base_block_size,
        trend_dim=1,
        context_dim=31,
        max_patch_nodes=100
    )
    
    model = NeurTSModel(
        grid_storage=grid_storage,
        max_block_size=base_block_size,
        hidden_dim=32
    )
    
    manager = GridManager(
        model=model,
        raw_data=raw_data,
        base_block_size=base_block_size,
        min_resolution=min_resolution
    )
    
    dataset = NeurTSDataset(
        raw_data=raw_data,
        manager=manager,
        grid_storage=grid_storage
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # 测试 __getitem__
    left_id, right_id, gt, mask, offset = dataset[0]
    print(f"Sample 0: left={left_id}, right={right_id}, gt.shape={gt.shape}, mask.sum()={mask.sum().item()}")
    
    # 测试分裂
    print("\nSplitting at t=50...")
    new_id = manager.split_block(50)
    print(f"New node ID: {new_id}")
    
    left_id, right_id, gt, mask, offset = dataset[1]
    print(f"Sample 1 after split: left={left_id}, right={right_id}, mask.sum()={mask.sum().item()}")
    
    print("\nTest passed!")