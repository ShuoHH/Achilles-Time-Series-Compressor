"""
TCN 解码器：基于时间卷积网络的波形重构。

包含:
- ResBlock1D: 一维残差块（膨胀卷积 + replicate padding）
- ResolutionInvariantPE: 分辨率不变位置编码（绝对步长，消除频率畸变）
- TcnDecoder: 双流解码器（动态边界融合 + TCN 精修）

数据流向:
1. 动态边界融合: boundary(t) = α(t)*proj(z_start) + β(t)*proj(z_end)
2. 位置编码: 添加正弦 PE
3. TCN 精修: ResBlock1D 堆叠（指数膨胀扩大感受野）
4. 输出: 直接输出波形
"""

import math

import torch
import torch.nn as nn

from .base_decoder import BaseDecoder


class ResBlock1D(nn.Module):
    """
    一维残差块。
    
    结构：Conv1D → Norm → ELU → Dropout → Conv1D → Norm → ELU + Skip Connection
    
    Args:
        channels (int): 输入与输出通道数。
        kernel_size (int): 卷积核尺寸，默认值为 3。
        dilation (int): 膨胀系数，默认值为 1。
        dropout (float): Dropout概率，默认值为 0.1。
    """
    
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super(ResBlock1D, self).__init__()
        
        padding = dilation * (kernel_size - 1) // 2
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, padding_mode='replicate', bias=True)
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, padding_mode='replicate', bias=True)
        self.norm2 = nn.GroupNorm(1, channels)
        
        self.elu = nn.ELU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Channels, Time]
        Returns:
            [Batch, Channels, Time]
        """
        residual = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.elu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.elu(out)
        
        return out + residual


class FiLMResBlock1D(nn.Module):
    """
    FiLM 条件调制一维残差块。
    
    在每个残差块内部，用边界向量对特征做仿射调制：
        out = (1 + gamma) * norm(conv(x)) + beta
    
    这迫使网络在每一层都持续依赖边界向量，而非只在输入端注入一次。
    当块分裂后新增 z_mid 时，调制信号在每层都会改变，让分裂对 TCN 也有效。
    
    FiLM 投影零初始化：初始时 gamma=0, beta=0，等价于原始 ResBlock1D。
    
    Args:
        channels (int): 输入与输出通道数。
        boundary_dim (int): 边界条件向量维度（通常为 2 * hidden_dim）。
        kernel_size (int): 卷积核尺寸，默认值为 3。
        dilation (int): 膨胀系数，默认值为 1。
        dropout (float): Dropout概率，默认值为 0.1。
    """
    
    def __init__(self, channels: int, boundary_dim: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super(FiLMResBlock1D, self).__init__()
        
        padding = dilation * (kernel_size - 1) // 2
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, padding_mode='replicate', bias=True)
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, padding_mode='replicate', bias=True)
        self.norm2 = nn.GroupNorm(1, channels)
        
        self.elu = nn.ELU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        
        # FiLM 投影：boundary_cond → (gamma, beta)，残差式调制
        self.film = nn.Linear(boundary_dim, channels * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
    
    def forward(self, x: torch.Tensor, boundary_cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Channels, Time]
            boundary_cond: [Batch, boundary_dim]  （每块一个全局条件向量）
        Returns:
            [Batch, Channels, Time]
        """
        film_params = self.film(boundary_cond)          # [B, channels*2]
        gamma, beta = film_params.chunk(2, dim=-1)      # [B, channels] each
        gamma = gamma.unsqueeze(-1)                     # [B, channels, 1]
        beta  = beta.unsqueeze(-1)                      # [B, channels, 1]
        
        residual = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = (1.0 + gamma) * out + beta               # FiLM 调制
        out = self.elu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = (1.0 + gamma) * out + beta               # FiLM 调制
        out = self.elu(out)
        
        return out + residual


class ResolutionInvariantPE(nn.Module):
    """
    Resolution-invariant positional encoding with absolute step.
    
    Uses arange (step=1) instead of linspace, so the PE frequency pattern
    remains identical regardless of block size. This prevents "frequency
    aliasing" when blocks are split: Conv1d kernels trained on step=1 PE
    continue to work correctly on shorter sub-blocks.
    
    Before (linspace * scale):
      256-block: step=1.0, 128-block: step=2.0  -> kernel mismatch!
    After (arange):
      256-block: step=1.0, 128-block: step=1.0  -> seamless transfer
    
    Args:
        d_model (int): PE dimension.
    """
    
    def __init__(self, d_model: int):
        super(ResolutionInvariantPE, self).__init__()
        self.d_model = d_model
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Args:
            seq_len: sequence length
        Returns:
            [1, d_model, seq_len] (channel-first for Conv1D)
        """
        device = self.div_term.device
        
        # Absolute integer positions: step is always 1, regardless of seq_len
        position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)  # [seq_len, 1]
        
        pe = torch.zeros(seq_len, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(position * self.div_term)
        if self.d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * self.div_term[:self.d_model // 2])
        else:
            pe[:, 1::2] = torch.cos(position * self.div_term)
        
        return pe.T.unsqueeze(0)  # [1, d_model, seq_len]


class TcnDecoder(BaseDecoder):
    """
    TCN 解码器：从边界状态向量直接重构波形。
    
    数据流向:
    1. 动态边界融合: 时间变化的边界权重
       - α(t) = 1 - t/T (靠近起点时权重大)
       - β(t) = t/T     (靠近终点时权重大)
       - boundary(t) = α(t) * proj(z_start) + β(t) * proj(z_end)
    2. 位置编码: 添加正弦PE
    3. TCN 精修: ResBlock1D堆叠
    4. 输出: 直接输出波形
    
    Args:
        dim (int): 状态向量总维度。
        trend_dim (int): 保留参数以兼容旧代码，但不再使用。
        hidden_dim (int): 隐藏层维度。
        pe_dim (int): 位置编码维度，默认值：32。
        num_blocks (int): ResBlock1D层数，默认值：4。
        kernel_size (int): 卷积核尺寸，默认值：3。
        dropout (float): Dropout概率，默认值：0.1。
    """
    
    def __init__(
        self,
        dim: int,
        trend_dim: int,
        hidden_dim: int,
        pe_dim: int = 32,
        num_blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(dim, trend_dim, hidden_dim)
        
        self.pe_dim = pe_dim
        self.dim = dim
        
        # 位置编码 (resolution-invariant: step=1 regardless of block size)
        self.pos_encoder = ResolutionInvariantPE(d_model=pe_dim)
        
        # 输入投影: (PE:pe_dim + Boundary:dim) → hidden_dim
        # 边界使用 SIREN 式线性插值，无可学习参数，与 SIREN 输入统一
        input_channels = pe_dim + dim
        self.input_proj = nn.Conv1d(input_channels, hidden_dim, kernel_size=1)
        
        # FiLM 条件向量维度：拼接原始 z_start / z_end（dim*2）
        boundary_dim = dim * 2
        
        # FiLMResBlock1D 堆叠，使用指数膨胀扩大感受野
        # 每层都注入边界条件，强制网络持续依赖边界向量
        self.res_blocks = nn.ModuleList([
            FiLMResBlock1D(
                channels=hidden_dim,
                boundary_dim=boundary_dim,
                kernel_size=kernel_size,
                dilation=2 ** i,  # 1, 2, 4, 8, 16, 32, 64, 128, ...
                dropout=dropout
            )
            for i in range(num_blocks)
        ])
        
        # 输出投影: hidden_dim → 1 (残差)
        self.output_proj = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        
        # 输出层零初始化：确保初始输出 ≈ 0
        # 避免 hidden_dim 较大时残差块放大信号导致初始 loss 爆炸
        # （标准做法：GPT-2, ResNet 等均采用类似策略）
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self,
        z_start: torch.Tensor,
        z_end: torch.Tensor,
        block_size: int,
        offsets: torch.Tensor = None,
        t_local: torch.Tensor = None,
        z_aux: torch.Tensor = None,
        aux_mask: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        解码波形块。
        
        Args:
            z_start: 起始状态向量 [Batch, Dim]
            z_end: 终止状态向量 [Batch, Dim]
            block_size: 输出序列长度
            offsets: 各样本块在基础块内的物理偏移 [Batch]，暂未使用
            t_local/z_aux/aux_mask: 接口兼容（TCN 单 token，不使用 aux；忽略）
            
        Returns:
            重构波形 [Batch, 1, block_size]
        """
        batch_size = z_start.shape[0]
        device = z_start.device
        
        # Step 1: SIREN 式线性插值边界（无可学习参数，与 SIREN 输入统一）
        # boundary(t) = (1-t)*z_start + t*z_end，逐点插值
        t = torch.linspace(0, 1, block_size, device=device)  # [block_size]
        alpha = (1 - t).view(1, 1, block_size)  # [1, 1, block_size]
        beta  = t.view(1, 1, block_size)         # [1, 1, block_size]
        
        z_start_exp = z_start.unsqueeze(-1)  # [Batch, dim, 1]
        z_end_exp   = z_end.unsqueeze(-1)    # [Batch, dim, 1]
        boundary_expanded = alpha * z_start_exp + beta * z_end_exp  # [Batch, dim, block_size]
        
        # Step 2: 位置编码
        pe = self.pos_encoder(block_size).expand(batch_size, -1, -1)
        
        # Step 3: 拼接并通过 TCN
        combined = torch.cat([pe, boundary_expanded], dim=1)  # [Batch, pe_dim + dim, block_size]
        x = self.input_proj(combined)
        
        # FiLM 条件：拼接原始 z_start / z_end（dim*2）
        boundary_cond = torch.cat([z_start, z_end], dim=-1)  # [B, 2*dim]
        
        for res_block in self.res_blocks:
            x = res_block(x, boundary_cond)
        
        # Step 4: 直接输出波形
        output = self.output_proj(x)
        
        return output
