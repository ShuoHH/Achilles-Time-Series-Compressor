"""
NeurTS Model: 主模型类

整合GlobalGridStorage和可插拔解码器，实现完整的编解码流程。
解码器通过 decoder_type 参数选择，支持多种网络架构对比实验。
"""

import torch
import torch.nn as nn

from .grid import GlobalGridStorage
from .decoders import create_decoder


class NeurTSModel(nn.Module):
    """
    NeurTS：神经时序存储系统 - 完整模型。
    
    整合Grid存储与可插拔解码器，实现从边界状态向量到波形的并行重构。
    
    数据流向:
    Grid (z) → Decoder(z_start, z_end, block_size) → Output [Batch, 1, block_size]
    
    Args:
        grid_storage: GlobalGridStorage实例，管理base_grid和patch_grid
        max_block_size: 最大波形块长度（用于padding）
        decoder_type: 解码器类型（'siren' 等，见 decoders/__init__.py）
        hidden_dim: 解码器隐藏层维度
        pe_dim: 位置编码维度，默认值：32
        num_res_blocks: 网络层数/残差块数，默认值：4
        kernel_size: 卷积核尺寸（部分解码器使用），默认值：3
        dropout: Dropout概率，默认值：0.1
        **decoder_kwargs: 传递给特定解码器的额外参数
    """
    
    def __init__(
        self,
        grid_storage: GlobalGridStorage,
        max_block_size: int,
        hidden_dim: int,
        decoder_type: str = 'siren',
        pe_dim: int = 32,
        num_res_blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        aux_dim: int = -1,
        **decoder_kwargs
    ):
        super(NeurTSModel, self).__init__()
        
        self.grid_storage = grid_storage
        self.max_block_size = max_block_size
        self.decoder_type = decoder_type
        
        # 从grid_storage获取维度信息
        self.total_dim = grid_storage.feature_dim
        self.trend_dim = grid_storage.trend_dim
        self.context_dim = grid_storage.context_dim
        
        # ── Aux Token 异构维度（Path A：filler-zero）─────────────────
        # aux_dim < total_dim 时：aux 向量仅前 aux_dim 维有效，其余强制为 0。
        # 效果：
        #   - 主 token 仍占 total_dim 字节
        #   - aux token 实际信息量 = aux_dim 字节（存储成本减半/四分之一）
        #   - 解码器输入维度不变（D + max_aux*D），尾部零位对应权重接收 0 梯度，
        #     自然保持初始值不变，对前向贡献始终为 0。
        # aux_dim <= 0 或 >= total_dim 时退化为等维行为（向后兼容）。
        if aux_dim is None or aux_dim <= 0 or aux_dim >= self.total_dim:
            self.aux_dim = self.total_dim
        else:
            self.aux_dim = int(aux_dim)
        
        # 通过工厂函数创建解码器
        self.decoder = create_decoder(
            decoder_type=decoder_type,
            dim=self.total_dim,
            trend_dim=self.trend_dim,
            hidden_dim=hidden_dim,
            pe_dim=pe_dim,
            num_blocks=num_res_blocks,
            kernel_size=kernel_size,
            dropout=dropout,
            base_block_size=max_block_size,
            **decoder_kwargs
        )
        
        # ── Aux Token Lookup（多 token 谱细化）─────────────────────
        # 由 GridManager 注入：aux_lookup(left_id, right_id) -> tuple[int]
        # 默认 None，行为与原版完全一致（向后兼容）
        # 设置后，decode_batch / forward / decode_single 自动查表并传入 z_aux
        self.aux_lookup = None
    
    def zero_init_node(self, node_id: int) -> None:
        """
        将指定 patch 节点的存储向量置零。
        
        用于 aux token 初始化：保证训练初期 aux token 对输出贡献为零，
        从而 multi-token 微调从一个等价于 base 模型的起点出发。
        
        Args:
            node_id: 全局节点 ID（必须 >= num_base_nodes）
        """
        gs = self.grid_storage
        if node_id < gs.num_base_nodes:
            raise ValueError(f"zero_init_node only for patch nodes, got {node_id}")
        local_id = node_id - gs.num_base_nodes
        with torch.no_grad():
            gs.patch_grid.data[local_id].zero_()
    
    def _gather_aux(
        self,
        left_ids: torch.Tensor,
        right_ids: torch.Tensor,
    ) -> tuple:
        """
        根据 (left_ids, right_ids) 查 aux 表，构建 z_aux 和 aux_mask。
        
        返回 (z_aux, aux_mask)；若该 batch 全部无 aux token，返回 (None, None)。
        """
        if self.aux_lookup is None:
            return None, None
        
        device = left_ids.device
        l_list = left_ids.detach().cpu().tolist()
        r_list = right_ids.detach().cpu().tolist()
        
        aux_lists = [self.aux_lookup(int(l), int(r)) for l, r in zip(l_list, r_list)]
        if not any(aux_lists):
            return None, None
        
        B = len(aux_lists)
        M_max = max(len(a) for a in aux_lists)
        D = self.total_dim
        
        # 收集所有 aux id 一次性查询，效率最高
        flat_ids: list = []
        positions: list = []  # (batch_idx, m_idx)
        for i, aux_ids in enumerate(aux_lists):
            for m, aid in enumerate(aux_ids):
                flat_ids.append(int(aid))
                positions.append((i, m))
        
        z_aux = torch.zeros(B, M_max, D, device=device)
        aux_mask = torch.zeros(B, M_max, device=device)
        
        if flat_ids:
            flat_ids_t = torch.tensor(flat_ids, device=device, dtype=torch.long)
            # 走 grid_storage 的量化感知通路：复用 forward(left_ids, right_ids)
            # 这里把同一份 ID 同时当 left/right 传入，取 left 即可
            flat_vecs, _ = self.grid_storage(flat_ids_t, flat_ids_t)  # [N, D]
            for k, (i, m) in enumerate(positions):
                z_aux[i, m] = flat_vecs[k]
                aux_mask[i, m] = 1.0
        
        # ── Path A: 异构维度约束 ────────────────────────────────────
        # 若 aux_dim < total_dim，强制将 aux 向量的尾部 [aux_dim:] 位清零。
        # 用乘法 mask 而非 in-place 切片赋值，autograd 行为最干净：
        #   ① 前向：尾部贡献 = z_aux[..., aux_dim:] * 0 = 0
        #   ② 反向：dL/dz_aux[..., aux_dim:] = dL/d(z_aux*mask)[..., aux_dim:] * 0 = 0
        #          → grid_storage 中 aux 节点的尾部维度保持零初始
        #   ③ 序列化：存档时只需保存前 aux_dim 维，压缩成本相应降低。
        if self.aux_dim < D:
            keep_mask = torch.zeros(D, device=device, dtype=z_aux.dtype)
            keep_mask[:self.aux_dim] = 1.0
            z_aux = z_aux * keep_mask  # broadcast: [B,M,D] * [D] → [B,M,D]
        
        return z_aux, aux_mask
    
    def forward(
        self,
        left_ids: torch.Tensor,
        right_ids: torch.Tensor,
        block_sizes: torch.Tensor = None,
        offsets: torch.Tensor = None
    ) -> torch.Tensor:
        """
        训练阶段的前向传播。
        
        数据流向:
        Grid(left_ids, right_ids) → (z_start, z_end) → Decoder(z_start, z_end, block_size) → Output
        
        Args:
            left_ids: 左边界节点ID [Batch]
            right_ids: 右边界节点ID [Batch]
            block_sizes: 每个样本的实际波形长度 [Batch]（可选，用于变长解码）
            offsets: 各样本块起始绝对位置 [Batch]（acorn1d 解码器使用）
            
        Returns:
            重构波形 [Batch, 1, max_block_size]
        """
        # 从grid获取边界向量
        left_vec, right_vec = self.grid_storage(left_ids, right_ids)
        
        # Aux token 查表（multi-token 谱细化；aux_lookup=None 时返回 None,None）
        z_aux, aux_mask = self._gather_aux(left_ids, right_ids)
        
        # 解码
        output = self.decoder(
            left_vec, right_vec, self.max_block_size,
            offsets=offsets, z_aux=z_aux, aux_mask=aux_mask,
        )
        
        return output
    
    def decode_single(
        self,
        left_id: int,
        right_id: int,
        block_size: int,
        offset: int = 0
    ) -> torch.Tensor:
        """
        解码单个波形块（推理用）。
        
        Args:
            left_id: 左边界节点ID
            right_id: 右边界节点ID
            block_size: 实际波形长度
            
        Returns:
            重构波形 [1, block_size]
        """
        device = self.grid_storage.base_grid.device
        
        left_ids = torch.tensor([left_id], device=device)
        right_ids = torch.tensor([right_id], device=device)
        
        # 通过 forward() 获取向量，确保量化感知（fake quantize）生效
        left_vec, right_vec = self.grid_storage(left_ids, right_ids)
        
        # Aux token 查表
        z_aux, aux_mask = self._gather_aux(left_ids, right_ids)
        
        offset_t = torch.tensor([offset], device=device)
        with torch.no_grad():
            output = self.decoder(
                left_vec, right_vec, block_size,
                offsets=offset_t, z_aux=z_aux, aux_mask=aux_mask,
            )
        
        return output.squeeze(0)  # [1, block_size]
    
    def decode_batch(
        self,
        left_ids: torch.Tensor,
        right_ids: torch.Tensor,
        block_size: int,
        offsets: torch.Tensor = None
    ) -> torch.Tensor:
        """
        批量解码指定长度的波形块（训练用，支持动态块长度）。
        
        与 forward 的区别：forward 始终输出 max_block_size，
        而 decode_batch 输出指定的 block_size，使位置编码和 Hermite 插值完整覆盖 [0, 1]。
        
        Args:
            left_ids: 左边界节点ID [Batch]
            right_ids: 右边界节点ID [Batch]
            block_size: 输出波形长度
            offsets: 各样本块在基础块内的物理偏移 [Batch]
            
        Returns:
            重构波形 [Batch, 1, block_size]
        """
        # 从 grid 获取边界向量（训练时会自动添加量化噪声）
        left_vec, right_vec = self.grid_storage(left_ids, right_ids)
        
        # Aux token 查表（multi-token 谱细化；aux_lookup=None 时返回 None,None）
        z_aux, aux_mask = self._gather_aux(left_ids, right_ids)
        
        # 用指定长度解码
        output = self.decoder(
            left_vec, right_vec, block_size,
            offsets=offsets, z_aux=z_aux, aux_mask=aux_mask,
        )
        
        return output
    
    def init_patch_node(self, patch_id: int, trend_value: float,
                        left_id: int = None, right_id: int = None,
                        alpha: float = 0.5):
        """
        初始化一个patch节点（委托给grid_storage）。
        
        Context 部分直接继承 z_left（ACORN-style）：
        - 新节点作为右子块的 z_left，从父块的 z_left 出发优化
        - right_id / alpha 保留接口但不参与计算
        
        Args:
            patch_id: patch节点的全局ID
            trend_value: 用于初始化trend部分的値
            left_id: 可选，左邻居节点ID，用于继承context
            right_id: 保留接口，单向量设计下不使用
            alpha: 保留接口，单向量设计下不使用
        """
        self.grid_storage.init_patch_node(patch_id, trend_value, left_id, right_id, alpha)
    
    @property
    def num_base_nodes(self) -> int:
        """基础网格节点数量。"""
        return self.grid_storage.num_base_nodes
    
    @property
    def max_patch_nodes(self) -> int:
        """补丁网格最大节点数量。"""
        return self.grid_storage.max_patch_nodes
    
    @property
    def block_size(self) -> int:
        """初始块大小。"""
        return self.grid_storage.block_size
    
    def trim_patch_grid(self, used_patch_count: int):
        """
        裁剪 patch_grid，只保留已使用的行（委托给grid_storage）。
        
        Args:
            used_patch_count: 实际使用的 patch 节点数量
        """
        self.grid_storage.trim_patch_grid(used_patch_count)
    
    def get_memory_usage(self) -> dict:
        """获取当前内存使用情况（委托给grid_storage）。"""
        return self.grid_storage.get_memory_usage()
