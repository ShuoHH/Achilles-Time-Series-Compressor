"""
BaseDecoder: 所有解码器的抽象基类。

所有解码器必须实现统一的 forward 接口：
    forward(z_start, z_end, block_size) -> [Batch, 1, block_size]

构造函数统一签名：
    __init__(dim, trend_dim, hidden_dim, **kwargs)

其中 kwargs 可包含各解码器特有的参数（pe_dim, num_blocks, kernel_size, dropout 等）。
"""

import torch
import torch.nn as nn
from abc import abstractmethod


class BaseDecoder(nn.Module):
    """
    解码器抽象基类。
    
    所有解码器子类必须实现 forward 方法，将左右边界状态向量
    解码为指定长度的波形。
    
    Args:
        dim (int): 状态向量总维度（z_start, z_end 的维度）
        trend_dim (int): 趋势分量维度
        hidden_dim (int): 隐藏层维度
    """
    
    def __init__(self, dim: int, trend_dim: int, hidden_dim: int, **kwargs):
        super().__init__()
        self.dim = dim
        self.trend_dim = trend_dim
        self.hidden_dim = hidden_dim
    
    @abstractmethod
    def forward(
        self,
        z_start: torch.Tensor,
        z_end: torch.Tensor,
        block_size: int,
        offsets: torch.Tensor = None,
        t_local: torch.Tensor = None
    ) -> torch.Tensor:
        """
        解码波形块。
        
        Args:
            z_start: 起始状态向量 [Batch, Dim]
            z_end: 终止状态向量 [Batch, Dim]
            block_size: 输出序列长度
            offsets: 各样本块在基础块内的物理偏移 [Batch]，用于全局物理时间计算
            
        Returns:
            重构波形 [Batch, 1, block_size]
        """
        raise NotImplementedError
