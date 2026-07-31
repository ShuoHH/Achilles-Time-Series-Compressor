"""
FourierDecoder: 显式傅里叶合成解码器（线性坡道 + DST）。

架构分解（三层设计）：

  ① 欧拉正交展开 (Euler)：
        A·sin(ωt + φ) = a·sin(ωt) + b·cos(ωt)
     → 直接预测 a_k, b_k，消除 Phase Wrapping 问题。

  ② 半周期正弦基底 (DST)：
        ω_k = π·k（不是 2π·k！）
        sin(πk·0) = 0,  sin(πk·1) = sin(kπ) = 0   ← 端点自然为零
     → 消除"强制周期性诅咒"，杜绝 Gibbs 现象。

  ③ 线性坡道 (Ramp)：
        ramp(t) = (1-t)·v0 + t·v1
     → v0 和 v1 均由 z_start 预测（每块只有一个自己的向量）。
     → z_end（right_id 对应的向量）属于相邻下一块，不应使用。
     → z_start 同时编码块的起止值和频谱内容。

最终公式：
    output(t) = [(1-t)·v0 + t·v1]  +  Σ_k a_k · sin(π·k·t)
                ↑ 线性坡道（趋势）       ↑ DST 振荡（端点=0）

多 token 谱细化（CONCAT 路径，max_aux_tokens > 0 启用）：
    z_in = [z_start; z_aux_1; z_aux_2; ...; z_aux_{max_M}]   ← 长度恒为 (max_M+1)*D
    v0/v1/a 由 z_in 经过同一组（更宽的）头部直接产生。
    缺位 aux 用 0 填充；aux 列权重 zero-init，pretrain 期间收到 0 梯度恒保 0。
    数学等价：M=1 路径 == 旧窄 MLP，无任何附加偏置。

CONCAT 与 SUM 的本质区别：
    SUM:    a = MLP(z_start) + MLP(z_aux_1) + ... + MLP(z_aux_M)
            → SiLU 在每个 token 上独立施加，token 间无交叉混合
            → 表达力上限 ⊊ 单 token 大 D 的 MLP（实验已验证）
    CONCAT: a = MLP([z_start; z_aux_1; ...; z_aux_M])
            → SiLU 前 W1 完成跨 token 的线性混合
            → 表达力等价于 D × (1+M) 维输入的单 MLP
            → 这才是"按块自适应扩 D"的正确实现
"""

import math
import torch
import torch.nn as nn

from .base_decoder import BaseDecoder


class FourierDecoder(BaseDecoder):
    """
    线性坡道 + 离散正弦变换（DST）解码器。

    z_start → 块起点值 v0 + 块终点值 v1 + DST 振荡系数 a_k
    z_end   → 不使用（属于相邻块，仅为接口兼容传入）

    Args:
        dim (int): z 向量维度
        trend_dim (int): 接口兼容，不使用
        hidden_dim (int): MLP 隐层宽度，用于非线性频谱系数预测
        num_freqs (int): DST 正弦分量数 F，建议 base_block_size // 8。
            奈奎斯特上限 = block_size // 2。
        base_block_size (int): 接口兼容
        total_length (int): 接口兼容
    """

    def __init__(
        self,
        dim: int,
        trend_dim: int,
        hidden_dim: int,
        num_freqs: int = 64,
        base_block_size: int = 512,
        total_length: int = 100000,
        max_aux_tokens: int = 0,
        **kwargs
    ):
        super().__init__(dim, trend_dim, hidden_dim)

        # 本地坐标解码器（t∈[0,1]），分裂判决走最大误差准则，不用位宽代价模型
        self.scale_agnostic = True

        self.num_freqs = num_freqs
        self.base_block_size = float(base_block_size)
        self.total_length = float(max(total_length, 1))

        # 多 token 谱细化（CONCAT 路径）的最大辅助 token 数
        # = 0  → 退化为单 token 解码器，行为完全等价旧版
        # > 0  → 输入扩展到 (1+max_M)*D，aux 列 zero-init，按需启用
        self.max_aux = int(max_aux_tokens)
        self.dim = dim
        self.in_dim = dim * (1 + self.max_aux)

        # 边界值预测：v0 = 块起点，v1 = 块终点，均来自 z_start (+ z_aux 拼接)
        self.to_v0 = nn.Linear(self.in_dim, 1)
        self.to_v1 = nn.Linear(self.in_dim, 1)

        # DST 振荡系数：z_in → MLP → a_k
        _h = max(hidden_dim, num_freqs)
        self.to_coeff = nn.Sequential(
            nn.Linear(self.in_dim, _h),
            nn.SiLU(),
            nn.Linear(_h, num_freqs),
        )

        # 固定频率网格 ω_k = π·k（半周期！不是 2π·k）
        freqs = torch.arange(1, num_freqs + 1).float() * math.pi
        self.register_buffer('freqs', freqs)  # [F]

        # 初始化：
        #   to_v0 / to_v1 整张 weight + bias = 0（旧版约定保留）
        #   to_coeff 输出层小随机；输入层 z_start 部分用默认 kaiming，aux 部分 = 0
        nn.init.zeros_(self.to_v0.weight)
        nn.init.zeros_(self.to_v0.bias)
        nn.init.zeros_(self.to_v1.weight)
        nn.init.zeros_(self.to_v1.bias)
        nn.init.normal_(self.to_coeff[-1].weight, std=0.01)
        nn.init.zeros_(self.to_coeff[-1].bias)
        if self.max_aux > 0:
            with torch.no_grad():
                # to_coeff[0].weight 形状 [hidden, in_dim]
                # in_dim = dim + max_aux*dim → 后 max_aux*dim 列对应 aux token
                #
                # 关键：不能与 z_aux 同时严格为 0，否则形成"双零死路径"：
                #   warmup 阶段 decoder 冻结，W_aux=0 → ∂loss/∂z_aux = W_aux^T·g = 0
                #   → z_aux 永远学不动；进入 finetune 时 z_aux 仍 = 0
                #   → ∂loss/∂W_aux = g·z_aux^T = 0
                #   → 两者互锁在 0，aux 通道彻底失活。
                #
                # 解：W_aux 用小随机（LoRA-A 风格 std=0.01），z_aux 初始 0：
                #   forward: W_aux · 0 = 0，baseline 严格不变 ✓
                #   z_aux 梯度: W_aux^T · g ≠ 0，warmup 即可学 ✓
                #   pretrain 期间 z_aux ≡ 0 → W_aux 梯度恒为 0 → 该子矩阵自然保持
                #     初始小随机值不变（不会被破坏），与"pretrain 不污染"目标一致。
                nn.init.normal_(
                    self.to_coeff[0].weight.data[:, dim:], mean=0.0, std=0.01
                )

    def forward(
        self,
        z_start: torch.Tensor,
        z_end: torch.Tensor,
        block_size: int,
        offsets: torch.Tensor = None,
        t_local: torch.Tensor = None,
        z_aux: torch.Tensor = None,
        aux_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            z_start: [B, dim]  本块的主 z 向量
            z_end:   [B, dim]  接口兼容，不使用（属于相邻块）
            block_size: 解码点数
            offsets: 接口兼容
            t_local: [K] 块内归一化坐标 ∈ [0,1]；None → 全量 linspace
            z_aux:   可选，多 token 谱细化的辅助向量
                     None / 空    → 等价 M=1 单 token
                     [B, M, dim] → 与 z_start 拼接作为宽 MLP 输入；M ≤ max_aux 时
                                    在 max_aux 轴上零填充对齐；M > max_aux 时截断
            aux_mask: [B, M]  1=有效，0=padding；mask 为 0 的 token 强制清零

        Returns:
            [B, 1, T]，T = block_size 或 K
        """
        device = z_start.device
        dtype = z_start.dtype

        if t_local is None:
            t_local = torch.linspace(0, 1, block_size, device=device)  # [T]
        T = t_local.shape[0]

        B = z_start.shape[0]
        D = self.dim

        # ── 构建宽输入 z_in = [z_start; aux_1; aux_2; ...; aux_{max_M}] ──
        if self.max_aux == 0:
            # 严格等价旧窄 MLP：直接喂 z_start
            z_in = z_start  # [B, D]
        else:
            # 准备 aux 部分：填充 / 截断到 max_aux
            if z_aux is None or z_aux.numel() == 0:
                z_aux_pad = torch.zeros(B, self.max_aux, D, device=device, dtype=dtype)
            else:
                z_aux_pad = z_aux
                if aux_mask is not None:
                    # mask 为 0 的位置强制为 0（防止 padding 节点的随机内容污染）
                    z_aux_pad = z_aux_pad * aux_mask.unsqueeze(-1).to(dtype)
                cur_M = z_aux_pad.shape[1]
                if cur_M < self.max_aux:
                    pad = torch.zeros(B, self.max_aux - cur_M, D, device=device, dtype=dtype)
                    z_aux_pad = torch.cat([z_aux_pad, pad], dim=1)
                elif cur_M > self.max_aux:
                    z_aux_pad = z_aux_pad[:, :self.max_aux]

            # 拼接：[B, D + max_aux*D]
            z_in = torch.cat([z_start, z_aux_pad.reshape(B, -1)], dim=-1)

        # 边界值与 DST 系数（一次前向，CONCAT 跨 token 在 W1 处线性混合）
        v0 = self.to_v0(z_in)          # [B, 1]
        v1 = self.to_v1(z_in)          # [B, 1]
        a  = self.to_coeff(z_in)       # [B, F]

        # 线性坡道：承载趋势，无周期性约束
        t_e = t_local.view(1, T)           # [1, T]
        ramp = (1 - t_e) * v0 + t_e * v1   # [B, T]

        # DST 正弦叠加：sin(πk·t)，端点=0，无 Gibbs 现象
        w_t = self.freqs.view(1, 1, -1) * t_local.view(1, T, 1)  # [1, T, F]
        osc = (a.unsqueeze(1) * torch.sin(w_t)).sum(-1)           # [B, T]

        return (ramp + osc).unsqueeze(1)  # [B, 1, T]
