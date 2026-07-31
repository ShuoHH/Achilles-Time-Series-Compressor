import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# GlobalGridStorage: 用于 NeurTS 时间序列压缩系统的全局状态特征网格
# dataloader应该传进来index方便取样
# =============================================================================

class GlobalGridStorage(nn.Module):
    """
    全局状态特征网格存储模块 (Global State-Space Grid Storage)
    
    核心思想：
    - 不存储原始时间序列数据，而是存储一个稀疏的"状态特征网格"
    - 解码时，利用相邻的两个 Grid 节点 (z_k, z_{k+1}) 通过线性插值生成引导信号
    - 维护一个全局共享的大 Grid 表，作为可学习参数
    
    双层网格架构（支持递归分裂）：
    - base_grid: 基础网格，存储初始粗粒度节点（从raw_data初始化）
    - patch_grid: 补丁网格，存储递归分裂产生的细粒度节点（动态分配）
    
    Grid 结构：
    - Shape: [Total_Nodes, Feature_Dim]
    - Total_Nodes = ceil(Total_Data_Length / Block_Size)
    - Feature_Dim = trend_dim + context_dim
    
    特征解耦：
    - z_trend (趋势特征): 从原始数据初始化，捕捉数据的整体轮廓
    - z_context (上下文特征): 随机初始化，代表 TCN 的隐状态
    
    ID分配规则：
    - ID < num_base_nodes: 从base_grid查询
    - ID >= num_base_nodes: 从patch_grid查询（需减去偏移量）
    """
    
    def __init__(
        self,
        raw_data: torch.Tensor,
        block_size: int = 1000,
        trend_dim: int = 1,
        context_dim: int = 63,
        max_patch_nodes: int = 1000,
        init_mode: str = 'endpoint',  # 'endpoint' | 'average' | 'max'
        context_init_std: float = 0.02,
    ):
        """
        初始化 GlobalGridStorage
        
        Args:
            raw_data: 原始时间序列张量, shape [T] 或 [T, D] (D=1 for univariate)
            block_size: 基础块长度，每 block_size 个点对应一个 Grid 节点
            trend_dim: 趋势特征维度 (通常为 1，对应标量数值)
            context_dim: 上下文特征维度 (TCN 隐状态维度)
            max_patch_nodes: 补丁网格的最大节点数量（预分配）
            init_mode: z_trend 初始化模式
                - 'endpoint': 取每个 block 的端点值 (默认，适合插值重建)
                - 'average': 取每个 block 的平均值
                - 'max': 取每个 block 的最大值
            context_init_std: z_context 随机初始化的标准差
        """
        super().__init__()
        
        # 保存配置
        self.block_size = block_size
        self.trend_dim = trend_dim
        self.context_dim = context_dim
        self.feature_dim = trend_dim + context_dim
        self.max_patch_nodes = max_patch_nodes
        self.context_init_std = context_init_std
        
        # 计算 Grid 节点数量
        self.total_length = raw_data.shape[0]    
        self.num_base_nodes = (self.total_length + block_size - 1) // block_size + 1
        
        # Base Grid: 从对based——grid表进行初始化
        base_grid_table = self._build_grid_table(raw_data, init_mode, context_init_std)
        self.base_id_threshold=base_grid_table.shape[0]-1
        self.base_grid = nn.Parameter(base_grid_table)
        
        # Patch Grid: 预分配，用于递归分裂产生的节点
        patch_grid_table = torch.zeros(max_patch_nodes, self.feature_dim)
        nn.init.normal_(patch_grid_table, mean=0.0, std=context_init_std)
        self.patch_grid = nn.Parameter(patch_grid_table)
        
        # 兼容旧API
        self.num_nodes = self.num_base_nodes
        
    def _build_grid_table(
        self,
        raw_data: torch.Tensor,
        init_mode: str,
        context_init_std: float,
    ) -> torch.Tensor:
        """
        构建并智能初始化 Grid 表
        
        初始化策略 (冷启动加速收敛):
        1. z_trend: 从原始数据采样/池化，使初始状态下线性插值就能大致拟合数据轮廓
        2. z_context: Xavier Uniform 或 Normal 随机初始化
        
        Returns:
            grid_table: shape [num_nodes, feature_dim]
        """
        # 确保 raw_data 是 2D: [T, D]
        if raw_data.dim() == 1:
            raw_data = raw_data.unsqueeze(-1)  # [T] -> [T, 1]
        
        T, D = raw_data.shape
        
        # =====================================================================
        # Step 1: 初始化 z_trend (趋势特征)
        # =====================================================================
        # 目标: 从原始数据中提取每个 block 的代表值
        # 这样初始状态下，线性插值就能大致拟合数据的轮廓
        
        if init_mode == 'endpoint':
            # 端点采样: 取每个 block 起始位置的值
            # 对于插值重建任务，端点值是最直接的选择
            # 采样索引: 0, block_size, 2*block_size, ...  这个就是为趋势的采样点做标记
            sample_indices = torch.arange(0, T, self.block_size, device=raw_data.device)
            
            # 确保最后一个节点对应数据末尾
            if sample_indices[-1] != T - 1:
                sample_indices = torch.cat([sample_indices, torch.tensor([T - 1], device=raw_data.device)])
            
            # 向量化采样
            z_trend = raw_data[sample_indices]  # [num_nodes, D]
            
        elif init_mode == 'average':
            # 平均池化: 取每个 block 的均值
            # 使用 unfold + mean 实现高效向量化
            # 先 padding 到 block_size 的整数倍
            pad_len = (self.block_size - T % self.block_size) % self.block_size
            if pad_len > 0:
                # 用最后一个值填充
                raw_data_padded = F.pad(raw_data.T, (0, pad_len), mode='replicate').T
            else:
                raw_data_padded = raw_data
            
            # reshape 并计算均值: [T_padded, D] -> [num_blocks, block_size, D] -> [num_blocks, D]
            num_blocks = raw_data_padded.shape[0] // self.block_size
            z_trend = raw_data_padded.view(num_blocks, self.block_size, D).mean(dim=1)
            
            # 添加末尾节点 (用最后一个 block 的均值)
            z_trend = torch.cat([z_trend, z_trend[-1:]], dim=0)
            
        elif init_mode == 'max':
            # 最大池化: 取每个 block 的最大值
            pad_len = (self.block_size - T % self.block_size) % self.block_size
            if pad_len > 0:
                raw_data_padded = F.pad(raw_data.T, (0, pad_len), mode='replicate').T
            else:
                raw_data_padded = raw_data
            
            num_blocks = raw_data_padded.shape[0] // self.block_size
            z_trend = raw_data_padded.view(num_blocks, self.block_size, D).max(dim=1)[0]
            
            # 添加末尾节点
            z_trend = torch.cat([z_trend, z_trend[-1:]], dim=0)
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}. Choose from 'endpoint', 'average', 'max'")
        
        # 调整 z_trend 维度以匹配 trend_dim
        # 如果 trend_dim > D，则复制填充；如果 trend_dim < D，则截断
        if self.trend_dim > D:
            # 复制第一个通道填充
            z_trend = z_trend[:, :1].expand(-1, self.trend_dim)
        elif self.trend_dim < D:
            z_trend = z_trend[:, :self.trend_dim]
        
        # 确保节点数量正确
        if z_trend.shape[0] < self.num_base_nodes:
            # 用最后一个值填充
            pad_nodes = self.num_base_nodes - z_trend.shape[0]
            z_trend = torch.cat([z_trend, z_trend[-1:].expand(pad_nodes, -1)], dim=0)
        elif z_trend.shape[0] > self.num_base_nodes:
            z_trend = z_trend[:self.num_base_nodes]
        
        # =====================================================================
        # Step 2: 初始化 z_context (上下文特征)
        # =====================================================================
        # 使用 Xavier Uniform 初始化，这是 TCN 隐状态的标准做法
        # 因为上下文特征无法直接从原数据观测，需要通过训练学习
        
        z_context = torch.empty(self.num_base_nodes, self.context_dim, device=raw_data.device)
        
        # Xavier Uniform: U(-a, a) where a = sqrt(6 / (fan_in + fan_out))
        # 这里简化为 Normal 初始化，更适合时间序列场景
        nn.init.normal_(z_context, mean=0.0, std=context_init_std)
        
        # =====================================================================
        # Step 3: 拼接 z_trend 和 z_context
        # =====================================================================
        # Grid 结构: [z_trend | z_context]
        # 这样在访问时可以方便地分离两部分
        
        grid_table = torch.cat([z_trend, z_context], dim=-1)  # [num_nodes, feature_dim]
        
        return grid_table
    
    def get_vectors(self, ids: torch.Tensor) -> torch.Tensor:
        """
        根据ID从base_grid或patch_grid中查询向量。
        
        ID < num_base_nodes: 从base_grid查询
        ID >= num_base_nodes: 从patch_grid查询（需减去偏移量）
        
        Args:
            ids: 节点ID张量 [Batch]
            
        Returns:
            状态向量 [Batch, feature_dim]
        """
        is_base = ids < self.num_base_nodes
        
        batch_size = ids.shape[0]
        device = ids.device
        vectors = torch.zeros(batch_size, self.feature_dim, device=device)
        
        if is_base.any():
            base_ids = ids[is_base]
            vectors[is_base] = self.base_grid[base_ids]
        
        if (~is_base).any():
            patch_ids = ids[~is_base] - self.num_base_nodes
            vectors[~is_base] = self.patch_grid[patch_ids]
        
        return vectors
    
    def get_trend(self, ids: torch.Tensor = None) -> torch.Tensor:
        """
        获取趋势特征部分。
        
        Args:
            ids: 可选，节点ID。如果为None，返回base_grid的全部trend。
            
        Returns:
            z_trend [N, trend_dim]
        """
        if ids is None:
            return self.base_grid[:, :self.trend_dim]
        return self.get_vectors(ids)[:, :self.trend_dim]
    
    def get_context(self, ids: torch.Tensor = None) -> torch.Tensor:
        """
        获取上下文特征部分。
        
        Args:
            ids: 可选，节点ID。如果为None，返回base_grid的全部context。
            
        Returns:
            z_context [N, context_dim]
        """
        if ids is None:
            return self.base_grid[:, self.trend_dim:]
        return self.get_vectors(ids)[:, self.trend_dim:]
    
    def init_patch_node(self, patch_id: int, trend_value: float, 
                        left_id: int = None, right_id: int = None,
                        alpha: float = 0.5):
        """
        初始化一个patch节点。
        
        Context 部分直接继承 z_left（ACORN-style）：
        - 新节点作为右子块的 z_left，从父块 z_left 出发优化
        - right_id / alpha 保留接口但不参与计算
        
        Args:
            patch_id: patch节点的全局ID (>= num_base_nodes)
            trend_value: 用于初始化trend部分的值
            left_id: 可选，左邻居节点ID，新节点继承其 context
            right_id: 保留接口，单向量设计下不使用
            alpha: 保留接口，单向量设计下不使用
        """
        local_id = patch_id - self.num_base_nodes
        with torch.no_grad():
            # 初始化 trend 部分
            self.patch_grid[local_id, :self.trend_dim] = trend_value
            
            # 如果提供了左邻居ID，context 部分直接继承 z_left（ACORN-style）
            # 单向量设计：decoder 只用 z_left，新节点作为右子块的 z_left，
            # 从父块的 z_left 出发优化，而不是从线性插值的"陌生"向量出发
            if left_id is not None:
                left_vec = self.get_vectors(torch.tensor([left_id], device=self.patch_grid.device))
                self.patch_grid[local_id, self.trend_dim:] = left_vec[0, self.trend_dim:]
    
    def forward(self, left_ids: torch.Tensor, right_ids: torch.Tensor) -> tuple:
        """
        前向传播：获取左右边界向量（量化感知）。
        
        启用量化时：
        - 训练模式：STE fake quantize（前向用量化值，梯度直通）
        - 评估模式：确定性 fake quantize（与最终 quantize_and_freeze 一致）
        - 已真量化：直接返回（无需 fake）
        
        Args:
            left_ids: 左边界节点ID [Batch]
            right_ids: 右边界节点ID [Batch]
        
        Returns:
            (left_vectors, right_vectors): 各为 [Batch, feature_dim]
        """
        left_vectors = self.get_vectors(left_ids)
        right_vectors = self.get_vectors(right_ids)
        
        # 量化感知：根据 _quant_mode 选择训练时的量化策略
        if hasattr(self, '_quantization_enabled') and self._quantization_enabled:
            if self.training:
                mode = getattr(self, '_quant_mode', 'ste')
                if mode == 'noise':
                    # TCN 兼容模式：随机均匀噪声（更柔和，避免残差块放大阶梯扰动）
                    left_vectors = self._simulate_quantization_noise(left_vectors)
                    right_vectors = self._simulate_quantization_noise(right_vectors)
                else:
                    # SIREN 默认模式：STE fake quantize（精确量化感知）
                    left_vectors = self._ste_fake_quantize(left_vectors)
                    right_vectors = self._ste_fake_quantize(right_vectors)
            else:
                left_vectors = self.fake_quantize(left_vectors)
                right_vectors = self.fake_quantize(right_vectors)
        
        return left_vectors, right_vectors
    
    def trim_patch_grid(self, used_patch_count: int) -> 'GlobalGridStorage':
        """
        裁剪 patch_grid，只保留已使用的行，释放未使用的内存。
        
        注意：此操作会创建一个新的 Parameter，原有的梯度历史会丢失。
        建议在训练完成后、保存模型前调用。
        
        Args:
            used_patch_count: 实际使用的 patch 节点数量
            
        Returns:
            self: 返回自身以支持链式调用
        """
        if used_patch_count <= 0:
            # 没有使用任何 patch 节点，创建一个空的 patch_grid
            new_patch_grid = torch.zeros(0, self.feature_dim, 
                                         device=self.patch_grid.device,
                                         dtype=self.patch_grid.dtype)
        elif used_patch_count >= self.max_patch_nodes:
            # 全部使用，无需裁剪
            print(f"[GlobalGridStorage] No trimming needed: all {self.max_patch_nodes} patch nodes are used.")
            return self
        else:
            # 只保留前 used_patch_count 行
            new_patch_grid = self.patch_grid.data[:used_patch_count].clone()
        
        # 更新 patch_grid Parameter
        del self.patch_grid
        self.patch_grid = nn.Parameter(new_patch_grid)
        
        old_max = self.max_patch_nodes
        self.max_patch_nodes = used_patch_count
        
        print(f"[GlobalGridStorage] Trimmed patch_grid: {old_max} -> {used_patch_count} nodes, "
              f"freed {(old_max - used_patch_count) * self.feature_dim * 4 / 1024:.2f} KB")
        
        return self
    
    def get_memory_usage(self) -> dict:
        """
        获取当前内存使用情况。
        
        Returns:
            包含各部分内存使用量的字典（单位：字节）
        """
        base_bytes = self.base_grid.numel() * self.base_grid.element_size()
        patch_bytes = self.patch_grid.numel() * self.patch_grid.element_size()
        
        return {
            "base_grid_bytes": base_bytes,
            "patch_grid_bytes": patch_bytes,
            "total_bytes": base_bytes + patch_bytes,
            "base_grid_kb": base_bytes / 1024,
            "patch_grid_kb": patch_bytes / 1024,
            "total_kb": (base_bytes + patch_bytes) / 1024,
            "num_base_nodes": self.num_base_nodes,
            "num_patch_nodes": self.patch_grid.shape[0],
            "feature_dim": self.feature_dim,
        }
    
    def extra_repr(self) -> str:
        return (
            f"num_base_nodes={self.num_base_nodes}, "
            f"max_patch_nodes={self.max_patch_nodes}, "
            f"feature_dim={self.feature_dim} (trend={self.trend_dim}, context={self.context_dim}), "
            f"block_size={self.block_size}, "
            f"total_length={self.total_length}"
        )
    
    # =========================================================================
    # 量化相关方法
    # =========================================================================
    
    def setup_quantization(self, num_bits: int = 8):
        """
        设置量化参数。
        
        Args:
            num_bits: 量化位数，默认 8-bit (256 bins)
        """
        self.num_bits = num_bits
        self.n_quant_bins = 2 ** num_bits
        self._quantization_enabled = True
        self._is_quantized = False
        
        # 量化训练模式: 'ste' = STE fake quantize (SIREN), 'noise' = 随机噪声 (TCN)
        if not hasattr(self, '_quant_mode'):
            self._quant_mode = 'ste'
        
        # 量化范围将在 quantize_and_freeze 时根据实际数据动态计算
        self.register_buffer('quant_min', torch.tensor(0.0))
        self.register_buffer('quant_max', torch.tensor(1.0))
        self.register_buffer('quant_scale', torch.tensor(1.0))
        
        print(f"[GlobalGridStorage] Quantization setup: {num_bits}-bit ({self.n_quant_bins} bins), mode={self._quant_mode}")
    
    def fake_quantize(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        确定性 fake quantize：量化→反量化 round-trip。
        
        eval 模式使用全局量化参数（与最终 quantize_and_freeze 一致）；
        train 模式使用 per-tensor 动态范围。
        
        Args:
            vectors: 输入向量 [..., feature_dim]
            
        Returns:
            fake-quantized 向量（float32 但精度 = num_bits）
        """
        if hasattr(self, '_is_quantized') and self._is_quantized:
            return vectors  # 已经真量化，无需 fake
        
        if not self.training and hasattr(self, 'quant_scale') and self.quant_scale > 0:
            # eval 模式：使用全局参数，保证与最终量化结果一致
            v_min = self.quant_min
            scale = self.quant_scale
        else:
            # train 模式：per-tensor 动态范围
            v_min = vectors.min()
            v_max = vectors.max()
            scale = (v_max - v_min) / (self.n_quant_bins - 1)
            if scale == 0:
                return vectors
        
        quantized_idx = torch.round((vectors - v_min) / scale)
        quantized_idx = quantized_idx.clamp(0, self.n_quant_bins - 1)
        dequantized = quantized_idx * scale + v_min
        return dequantized
    
    def _simulate_quantization_noise(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        训练时模拟量化噪声（随机均匀噪声）。
        
        比 STE 更柔和：不 snap 到格点，而是在量化步长范围内添加随机扰动。
        适合对输入扰动敏感的解码器（如 TCN 残差网络）。
        
        Args:
            vectors: 输入向量 [..., feature_dim]
            
        Returns:
            加噪后的向量
        """
        with torch.no_grad():
            v_min = vectors.min()
            v_max = vectors.max()
            step = (v_max - v_min) / (self.n_quant_bins - 1)
        
        noise = (torch.rand_like(vectors) - 0.5) * step
        return vectors + noise
    
    def _ste_fake_quantize(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Straight-Through Estimator (STE) fake quantize：
        前向用量化后的值，反向梯度直接流过（当作恒等函数）。
        
        Args:
            vectors: 输入向量 [..., feature_dim]
            
        Returns:
            STE fake-quantized 向量
        """
        dequantized = self.fake_quantize(vectors)
        # STE: forward uses dequantized, backward flows through vectors
        return vectors + (dequantized - vectors).detach()
    
    @torch.no_grad()
    def compute_quantization_params(self):
        """
        根据当前 Grid 数据计算量化参数（min, max, scale）。
        
        应在训练完成后、量化前调用。
        """
        # 合并 base_grid 和已使用的 patch_grid
        all_data = torch.cat([self.base_grid.data, self.patch_grid.data], dim=0)
        
        self.quant_min = all_data.min()
        self.quant_max = all_data.max()
        self.quant_scale = (self.quant_max - self.quant_min) / (self.n_quant_bins - 1)
        
        print(f"[GlobalGridStorage] Quantization params: min={self.quant_min:.6f}, "
              f"max={self.quant_max:.6f}, scale={self.quant_scale:.6f}")
    
    @torch.no_grad()
    def quantize_and_freeze(self):
        """
        对 Grid 进行量化并冻结参数。
        
        量化公式：
        1. quantized_idx = round((value - min) / scale)
        2. dequantized = quantized_idx * scale + min
        
        调用后 Grid 不再可训练。
        """
        if not hasattr(self, '_quantization_enabled') or not self._quantization_enabled:
            print("[GlobalGridStorage] Warning: quantization not setup, call setup_quantization first")
            return
        
        # 计算量化参数
        self.compute_quantization_params()
        
        # 量化 base_grid
        base_quantized_idx = torch.round((self.base_grid.data - self.quant_min) / self.quant_scale)
        base_quantized_idx.clamp_(0, self.n_quant_bins - 1)
        self.base_grid.data = base_quantized_idx * self.quant_scale + self.quant_min
        self.base_grid.requires_grad_(False)
        
        # 量化 patch_grid
        patch_quantized_idx = torch.round((self.patch_grid.data - self.quant_min) / self.quant_scale)
        patch_quantized_idx.clamp_(0, self.n_quant_bins - 1)
        self.patch_grid.data = patch_quantized_idx * self.quant_scale + self.quant_min
        self.patch_grid.requires_grad_(False)
        
        self._is_quantized = True
        
        print(f"[GlobalGridStorage] Grid quantized to {self.num_bits}-bit and frozen")
    
    @torch.no_grad()
    def clamp_values(self):
        """
        将 Grid 值裁剪到量化范围内。
        
        可在训练过程中定期调用，防止值超出量化范围。
        """
        if not hasattr(self, 'quant_min'):
            return
        
        self.base_grid.data.clamp_(self.quant_min, self.quant_max)
        self.patch_grid.data.clamp_(self.quant_min, self.quant_max)
    
    def get_quantized_memory_usage(self) -> dict:
        """
        获取量化后的内存使用情况（理论值）。
        
        Returns:
            包含量化后内存使用量的字典
        """
        if not hasattr(self, 'num_bits'):
            return self.get_memory_usage()
        
        num_base = self.num_base_nodes
        num_patch = self.patch_grid.shape[0]
        total_elements = (num_base + num_patch) * self.feature_dim
        
        # 量化后每个元素占用 num_bits 位
        quantized_bytes = total_elements * self.num_bits / 8
        # 还需要存储 min, scale (2 个 float32)
        metadata_bytes = 2 * 4
        
        original = self.get_memory_usage()
        
        return {
            "original_bytes": original['total_bytes'],
            "quantized_bytes": quantized_bytes + metadata_bytes,
            "compression_ratio": original['total_bytes'] / (quantized_bytes + metadata_bytes),
            "num_bits": self.num_bits,
            "total_elements": total_elements,
        }


class Grid(nn.Module):
    """
    Base class for Feature level grids
    """
    def __init__(self, channels, h, w, quantization, circular=False):
        super().__init__()
        self.h = h
        self.w = w
        self.channels = channels
        self.n_quant_bins = 2**quantization
        self.quant_left = -(self.n_quant_bins-1) / (2*self.n_quant_bins)
        self.quant_right = (self.n_quant_bins) / (2*self.n_quant_bins)
        self.circular = circular

        self.grid = nn.Parameter(torch.randn(1, channels, h, w)/100)

    def resample(self, coordinate_start, h, w, stride, support_resolution_h, support_resolution_w, quantize=False):
        raise NotImplementedError

    def simulate_quantization(self):
        uniform = (torch.rand_like(self.grid) / self.n_quant_bins) - 1/(2*self.n_quant_bins)
        return self.grid + uniform

    @torch.no_grad()
    def quantize_grid_and_freeze(self):
        bin_edges = torch.linspace(self.quant_left, self.quant_right, steps=self.n_quant_bins+1, device=self.grid.device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        bin_indices = torch.bucketize(self.grid, bin_edges, right=True) - 1
        bin_indices.clamp_(0, self.n_quant_bins - 1) # ensure within range

        self.grid.data = bin_centers[bin_indices]
        self.grid.requires_grad_(False)

    @torch.no_grad()
    def clamp_values(self):
        self.grid.data.clamp_(self.quant_left, self.quant_right)

    def forward(self, coordinate_start, h, w, stride, support_resolution_h, support_resolution_w, quantize=False):
        return self.resample(coordinate_start, h, w, stride, support_resolution_h, support_resolution_w, quantize=quantize)

def test_global_grid_storage_basic():
    print("=" * 60)
    print("Test 1: 基本构建与 shape 检查")

    T = 100
    raw_data = torch.linspace(0, 1, T)  # 简单线性数据
    block_size = 10
    context_dim = 4

    grid = GlobalGridStorage(
        raw_data=raw_data,
        block_size=block_size,
        trend_dim=1,
        context_dim=context_dim,
        max_patch_nodes=50,
        init_mode="endpoint"
    )

    expected_num_nodes = (T + block_size - 1) // block_size + 1

    print(f"num_base_nodes: {grid.num_base_nodes}")
    print(f"expected: {expected_num_nodes}")
    print(f"base_grid shape: {grid.base_grid.shape}")
    print(f"patch_grid shape: {grid.patch_grid.shape}")

    assert grid.num_base_nodes == expected_num_nodes
    assert grid.base_grid.shape == (expected_num_nodes, 1 + context_dim)

    print("✔ 基本构建通过\n")
    
    # Test 2: 测试 get_vectors
    print("=" * 60)
    print("Test 2: get_vectors 测试")
    
    ids = torch.tensor([0, 1, grid.num_base_nodes, grid.num_base_nodes + 1])
    vectors = grid.get_vectors(ids)
    print(f"get_vectors output shape: {vectors.shape}")
    assert vectors.shape == (len(ids), grid.feature_dim)
    print("✔ get_vectors 通过\n")
    
    # Test 3: 测试 init_patch_node
    print("=" * 60)
    print("Test 3: init_patch_node 测试")
    
    patch_id_to_init = grid.num_base_nodes + 5
    trend_val = 0.77
    grid.init_patch_node(patch_id_to_init, trend_val)
    retrieved_trend = grid.get_vectors(torch.tensor([patch_id_to_init]))[:, :grid.trend_dim]
    print(f"Initialized patch node {patch_id_to_init} trend to {trend_val}, retrieved: {retrieved_trend.item():.4f}")
    assert torch.isclose(retrieved_trend.squeeze().float(), torch.tensor(trend_val).float())
    print("✔ init_patch_node 通过\n")
    
    # Test 4: 测试 trim_patch_grid
    print("=" * 60)
    print("Test 4: trim_patch_grid 测试")
    
    mem_before = grid.get_memory_usage()
    print(f"Before trim: {mem_before['num_patch_nodes']} patch nodes, {mem_before['patch_grid_kb']:.2f} KB")
    
    used_count = 10
    grid.trim_patch_grid(used_count)
    
    mem_after = grid.get_memory_usage()
    print(f"After trim: {mem_after['num_patch_nodes']} patch nodes, {mem_after['patch_grid_kb']:.2f} KB")
    
    assert mem_after['num_patch_nodes'] == used_count
    print("✔ trim_patch_grid 通过\n")


##参考
    @torch.no_grad()
    def quantize_grid_and_freeze(self):
        bin_edges = torch.linspace(self.quant_left, self.quant_right, steps=self.n_quant_bins+1, device=self.grid.device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        bin_indices = torch.bucketize(self.grid, bin_edges, right=True) - 1
        bin_indices.clamp_(0, self.n_quant_bins - 1) # ensure within range

        self.grid.data = bin_centers[bin_indices]
        self.grid.requires_grad_(False)

    @torch.no_grad()
    def clamp_values(self):
        self.grid.data.clamp_(self.quant_left, self.quant_right)

if __name__ == "__main__":
    test_global_grid_storage_basic()