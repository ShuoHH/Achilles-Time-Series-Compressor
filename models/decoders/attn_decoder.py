"""
Attention（Transformer）解码器：基于自注意力的隐式表征波形重构。

动机：为模型消融提供"注意力架构"对照（区别于频率派 fourier/siren、卷积派 tcn、
      非频率 MLP gaussian）。块内各位置作为 token，自注意力建模位置间依赖，
      z 通过 FiLM 逐层调制（与 siren/tcn 一致，保证分裂时 z_mid 对波形有控制力）。

数据流向：
  1. 边界线性插值 boundary(t) = (1-t)·z_start + t·z_end（逐点，无参数）
  2. 分辨率不变位置编码（arange step=1，分裂无频率畸变）
  3. 拼接投影到 d_model → [B, T, d_model]
  4. N 层 (Self-Attention + FFN)，每层 FiLM(z_start‖z_end) 调制
  5. 输出投影 d_model → 1（零初始化，稳定起步）

接口：forward(z_start, z_end, block_size, offsets, ...) -> [B, 1, T]。
"""

import math

import torch
import torch.nn as nn

from .base_decoder import BaseDecoder


class ResolutionInvariantPE(nn.Module):
    """分辨率不变位置编码（绝对步长 step=1，块分裂后频率模式不变）。"""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, seq_len: int) -> torch.Tensor:
        device = self.div_term.device
        position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        pe = torch.zeros(seq_len, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(position * self.div_term)
        if self.d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * self.div_term[:self.d_model // 2])
        else:
            pe[:, 1::2] = torch.cos(position * self.div_term)
        return pe.unsqueeze(0)  # [1, seq_len, d_model]


class FiLMTransformerLayer(nn.Module):
    """
    FiLM 条件调制的 Transformer 层：Self-Attn → FiLM → FFN → FiLM。

    z 通过 FiLM(gamma,beta) 在每层调制（沿时间轴广播），零初始化。
    """

    def __init__(self, d_model, nhead, ffn_mult, boundary_dim, dropout=0.1):
        super().__init__()
        # 不用 batch_first（兼容旧版 PyTorch）；前向内部转置为 [T,B,d]
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # 两组 FiLM：注意力后 + FFN 后
        self.film = nn.Linear(boundary_dim, d_model * 4)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, cond):
        # x: [B, T, d]; cond: [B, boundary_dim]
        g1, b1, g2, b2 = self.film(cond).chunk(4, dim=-1)   # each [B, d]
        g1 = g1.unsqueeze(1); b1 = b1.unsqueeze(1)
        g2 = g2.unsqueeze(1); b2 = b2.unsqueeze(1)

        xt = x.transpose(0, 1)                              # [B,T,d] -> [T,B,d]
        a, _ = self.attn(xt, xt, xt, need_weights=False)
        a = a.transpose(0, 1)                               # [T,B,d] -> [B,T,d]
        x = self.norm1(x + self.dropout(a))
        x = (1.0 + g1) * x + b1                             # FiLM 调制

        f = self.ffn(x)
        x = self.norm2(x + f)
        x = (1.0 + g2) * x + b2                             # FiLM 调制
        return x


class AttnDecoder(BaseDecoder):
    """
    注意力（Transformer）解码器。

    Args:
        dim (int):        状态向量总维度
        hidden_dim (int): d_model
        pe_dim (int):     位置编码维度
        num_blocks (int): Transformer 层数
        nhead (int):      注意力头数（hidden_dim 需可整除）
        ffn_mult (int):   FFN 隐藏 = d_model * ffn_mult
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
        nhead: int = 4,
        ffn_mult: int = 4,
        **kwargs
    ):
        super().__init__(dim, trend_dim, hidden_dim)
        self.dim = dim
        self.pe_dim = pe_dim

        # nhead 需整除 d_model
        if hidden_dim % nhead != 0:
            for h in (4, 2, 1):
                if hidden_dim % h == 0:
                    nhead = h; break
        self.pos_encoder = ResolutionInvariantPE(d_model=pe_dim)

        # 输入投影：(PE:pe_dim + Boundary:dim) → d_model
        self.input_proj = nn.Linear(pe_dim + dim, hidden_dim)

        boundary_dim = dim * 2                              # FiLM 条件：z_start‖z_end
        self.layers = nn.ModuleList([
            FiLMTransformerLayer(hidden_dim, nhead, ffn_mult, boundary_dim, dropout)
            for _ in range(num_blocks)
        ])

        self.output_proj = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.output_proj.weight)             # 零初始化，稳定起步
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
        batch_size = z_start.shape[0]
        device = z_start.device

        # Step 1: 边界线性插值 [B, T, dim]
        t = torch.linspace(0, 1, block_size, device=device).view(1, block_size, 1)
        boundary = (1 - t) * z_start.unsqueeze(1) + t * z_end.unsqueeze(1)   # [B, T, dim]

        # Step 2: 位置编码 [1, T, pe_dim] → [B, T, pe_dim]
        pe = self.pos_encoder(block_size).expand(batch_size, -1, -1)

        # Step 3: 拼接投影到 d_model
        x = self.input_proj(torch.cat([pe, boundary], dim=-1))              # [B, T, d]

        # Step 4: FiLM 条件（z_start‖z_end），逐层注意力
        cond = torch.cat([z_start, z_end], dim=-1)                          # [B, 2*dim]
        for layer in self.layers:
            x = layer(x, cond)

        # Step 5: 输出 [B, T, 1] → [B, 1, T]
        output = self.output_proj(x).permute(0, 2, 1)
        return output
