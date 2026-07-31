"""
GaussianMLP 解码器：非频率隐式神经表示（INR）。

动机：SIREN 用周期激活 sin(ω·x)，本质是"频率先验的 MLP"，与显式傅里叶基同属
      频率派。为在模型消融中提供"非频率"对照，本解码器与 SIREN 结构完全一致
      （纯 FiLM、输入仅时间坐标 t），仅把激活函数从 sin 换成高斯 exp(-(a·x)²)。
      这样可干净地隔离变量：周期激活 vs 非周期激活。

高斯激活 INR 参考：Chng et al., "GARF: Gaussian Activated Radiance Fields", 2022。
高斯凸包（局部支撑）拟合高频，无任何频率/周期先验。

接口与 SIREN 一致：forward(z_start, z_end, block_size, offsets, ...) -> [B,1,T]。
条件方式：FiLM 零初始化，z 逐层调制。
"""

import numpy as np
import torch
import torch.nn as nn

from .base_decoder import BaseDecoder


class FiLMGaussianLayer(nn.Module):
    """
    FiLM 条件调制高斯层：Linear → FiLM → 高斯激活 exp(-(a·x)²)。

    out = (1 + gamma) * linear(x) + beta
    output = exp(-(scale · out)²)   [非末层] / out [末层，线性]

    FiLM 投影零初始化：初始 gamma=0, beta=0。

    Args:
        in_features:  输入维度
        out_features: 输出维度
        boundary_dim: FiLM 条件向量维度
        scale:        高斯带宽（越大凸包越窄→有效频率越高）
        is_last:      是否末层（线性输出，无激活）
    """

    def __init__(self, in_features, out_features, boundary_dim, scale=1.0, is_last=False):
        super().__init__()
        self.scale = scale
        self.is_last = is_last
        self.linear = nn.Linear(in_features, out_features)

        # FiLM 投影：零初始化，初始等价于无调制
        self.film = nn.Linear(boundary_dim, out_features * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, boundary_cond):
        out = self.linear(x)                               # [N, out]
        gamma, beta = self.film(boundary_cond).chunk(2, dim=-1)
        out = (1.0 + gamma) * out + beta                   # FiLM 调制
        if self.is_last:
            return out
        return torch.exp(-(self.scale * out) ** 2)         # 高斯激活（非频率）


class GaussianMLPDecoder(BaseDecoder):
    """
    非频率 MLP INR：结构同 SIREN（纯 FiLM，输入仅 t），激活换为高斯。

    Args:
        dim (int):        状态向量总维度
        trend_dim (int):  保留接口
        hidden_dim (int): 隐藏层维度
        num_blocks (int): 层数
        gauss_scale (float):        隐层高斯带宽，默认 1.0
        gauss_scale_first (float):  首层高斯带宽（略大以提升表达），默认 3.0
        base_block_size (int):      基础块长，用于全局时间坐标（配合 offsets）
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
        gauss_scale: float = 1.0,
        gauss_scale_first: float = 3.0,
        base_block_size: int = 256,
        **kwargs
    ):
        super().__init__(dim, trend_dim, hidden_dim)
        self.num_layers = num_blocks
        self.base_block_size = float(base_block_size)
        self.boundary_dim = dim                            # 单向量，仅用 z_start

        input_dim = 1                                      # 纯 FiLM：主干只吃时间坐标 t
        layers = []
        layers.append(FiLMGaussianLayer(input_dim, hidden_dim, self.boundary_dim,
                                        scale=gauss_scale_first, is_last=False))
        for _ in range(self.num_layers - 2):
            layers.append(FiLMGaussianLayer(hidden_dim, hidden_dim, self.boundary_dim,
                                            scale=gauss_scale, is_last=False))
        layers.append(FiLMGaussianLayer(hidden_dim, 1, self.boundary_dim,
                                        scale=gauss_scale, is_last=True))
        self.layers = nn.ModuleList(layers)

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
        batch_size = z_start.shape[0]
        device = z_start.device

        # 全局物理时间坐标（与 SIREN 完全一致，支持 offset/分裂）
        tl = torch.linspace(0, 1, block_size, device=device)
        stride = block_size / self.base_block_size
        if offsets is not None:
            start_ratio = offsets.float() / self.base_block_size
            t_global = start_ratio.unsqueeze(1) + stride * tl.unsqueeze(0)   # [B, T]
            t_global = t_global.unsqueeze(-1)                               # [B, T, 1]
        else:
            t_global = tl.view(1, block_size, 1).expand(batch_size, -1, -1)

        cond = z_start.unsqueeze(1).expand(-1, block_size, -1).reshape(-1, self.boundary_dim)
        x = t_global.reshape(-1, 1)
        for layer in self.layers:
            x = layer(x, cond)
        output = x.view(batch_size, block_size, 1).permute(0, 2, 1)          # [B, 1, T]
        return output
