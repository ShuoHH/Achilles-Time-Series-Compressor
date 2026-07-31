"""

NeurTS Experiment: 训练实验类



基于 Crossformer 的训练框架，适配 NeurTS 时序压缩系统。

"""



import math



from exp.exp_basic import Exp_Basic

from models.neurts import GlobalGridStorage, NeurTSModel, GridManager

from models.neurts_dataset import NeurTSTaskDataset

from models.patch_registry import PatchRegistry, PatchCollector, apply_patches_to_reconstruction

from models.fallback_dict import FallbackDict

from models.patch_split import evaluate_parent_patch_split, PatchSplitManager, evaluate_whole_block_patch, build_multilayer_tree, count_leaves_and_depth, tree_height_after_pruning
from models.patch_split import fit_child_best, fit_child_fixedK, build_patch_basis
from models.block_codec import BlockCodec, BlockRecord, LeafRecord, FourierCoeffs

from data.data_loader import NeurTSDataLoader, NeurTSDataset



from utils.tools import EarlyStopping, adjust_learning_rate



import numpy as np

import torch

import torch.nn as nn

from torch import optim

from torch.utils.data import DataLoader, Sampler

from collections import defaultdict

import random as _random

from torch.nn import DataParallel



import os

import time

import json

import pickle

import matplotlib.pyplot as plt











class SameLengthBatchSampler(Sampler):

    """

    同长度批采样器：确保每个 batch 内所有样本块长度相同。

    

    优势：消除 _process_one_batch 中的分组循环，每个 batch 只需一次 decode_batch 调用。

    保持逆长度加权采样，训练逻辑不变。

    """

    

    def __init__(self, manager, batch_size, weights=None):

        """

        Args:

            manager: GridManager 实例

            batch_size: batch 大小

            weights: 每个槽位的采样权重（逆长度加权）

        """

        self.batch_size = batch_size

        

        # 按块长度分组槽位索引

        self.groups = defaultdict(list)  # length -> [slot_idx]

        self.group_weights = defaultdict(list)  # length -> [weight]

        

        num_slots = manager.num_slots

        min_res = manager.min_resolution

        

        idx = 0

        while idx < num_slots:

            entry = manager.index_table[idx]

            left_id, right_id, level_code = entry.left_id, entry.right_id, entry.level_code

            

            block_start_idx = idx

            block_end_idx = idx

            while block_end_idx < num_slots - 1:

                next_entry = manager.index_table[block_end_idx + 1]

                if (next_entry.left_id == left_id and next_entry.right_id == right_id and

                    next_entry.level_code == level_code):

                    block_end_idx += 1

                else:

                    break

            

            block_len = (block_end_idx - block_start_idx + 1) * min_res

            

            for i in range(block_start_idx, block_end_idx + 1):

                self.groups[block_len].append(i)

                if weights is not None:

                    self.group_weights[block_len].append(weights[i].item() if torch.is_tensor(weights[i]) else weights[i])

            

            idx = block_end_idx + 1

        

        self.total_slots = num_slots

        

        # 预计算每组采样数量（按组权重占比分配，保持总 epoch 大小不变）

        total_weight = sum(sum(ws) for ws in self.group_weights.values()) if weights is not None else num_slots

        self.group_num_samples = {}

        for length, indices in self.groups.items():

            if weights is not None:

                group_weight = sum(self.group_weights[length])

                self.group_num_samples[length] = max(1, round(num_slots * group_weight / total_weight))

            else:

                self.group_num_samples[length] = len(indices)

    

    def __iter__(self):

        all_batches = []

        

        for length, indices in self.groups.items():

            num_samples = self.group_num_samples[length]

            

            if self.group_weights.get(length):

                # 按权重采样

                w = torch.tensor(self.group_weights[length], dtype=torch.float32)

                sampled_pos = torch.multinomial(w, num_samples=num_samples, replacement=True)

                sampled = [indices[p] for p in sampled_pos]

            else:

                sampled = [indices[i % len(indices)] for i in range(num_samples)]

                _random.shuffle(sampled)

            

            # 切分为 batch

            for i in range(0, len(sampled), self.batch_size):

                all_batches.append(sampled[i:i + self.batch_size])

        

        # 打乱 batch 顺序（不同长度的 batch 交替出现）

        _random.shuffle(all_batches)

        

        for batch in all_batches:

            yield batch

    

    def __len__(self):

        total = sum(self.group_num_samples.values())

        return (total + self.batch_size - 1) // self.batch_size





class Exp_NeurTS(Exp_Basic):

    """

    NeurTS 训练实验类

    

    核心流程：

    1. 加载数据 -> NeurTSDataLoader

    2. 构建模型 -> GlobalGridStorage + NeurTSModel + GridManager

    3. 创建数据集 -> NeurTSDataset (基于 Level Code 的掩码对齐)

    4. 训练循环 -> 重构损失优化

    5. 可选：自适应分裂 -> GridManager.split_block()

    """

    

    def __init__(self, args):

        # 先保存 args，因为 _build_model 需要用到

        self.args = args

        self.device = self._acquire_device()

        

        # 加载数据

        self.data_loader = self._load_data()

        self.raw_data = self.data_loader.get_raw_data().to(self.device)

        

        # 构建模型组件

        self.grid_storage, self.model, self.manager = self._build_model()

        

        # 创建数据集

        self.train_dataset = self._create_dataset()

        

        # 优化器持久化：避免每轮微调重置优化器状态

        self._optimizer = None

        self._optimizer_mode = None  # 'full' or 'finetune'

        

        # Tier 2/3 兜底字典

        self.fallback_dict = None  # 在 build_fallback_dict() 中初始化

        

        # Vector GC 结果（纯只读分析，供压缩比计算使用）

        self.gc_result = None  # 在 run_vector_gc() 中填充

        

        # 多尺度预训练权重（pretrain 阶段激活，其余阶段为 0）

        self._multiscale_weight = 0.0

        

        # 前向分裂日志：记录每次分裂的 (start, end) -> (mid, new_node_id)

        # 用于 Vector GC 的 DP 剪枝，GC 完成后销毁，不写入磁盘

        self.split_history = {}

        

        self._measured_r = None  # 用于探测机制：标记是否已有过分裂历史

        

        # 自适应折扣：记录每轮分裂时的预测子块 BW 代价，用于校准下轮 discount

        # key: (start_time, end_time), value: {'predicted_child_bw': float, 'eps': float}

        self._split_predictions = {}

        self._adaptive_discount = None  # 由实测数据计算的折扣因子

        

        # 亲子映射：记录每个新节点对应的父块左节点 ID，用于统计 Δz 大小

        # key: new_node_id, value: (old_left_id, level_code)

        self._parentage_map = {}

        

        # Active-Dim 约束缓存：split_vec_dim < feature_dim 时启用

        # key: local_patch_id (= new_node_id - num_base_nodes)

        # val: frozen_tensor  [feature_dim - split_vec_dim]，永远等于父节点对应 dims

        self._frozen_patch_values = {}

        

        # 分裂前后比对：adaptive_split 填充，report_split_results 消费

        self._presplit_stats = {}

        # Split 字节转化全局累加（--split_byte_gate）：raw vs rate-gated 双账本

        self._split_byte_stats = None

        # Parent-anchored patch split 持有者（接入用；None=未启用）

        self.patch_manager = None

        # 统一块编解码结构（BlockCodec：结构码+定长系数+EDWB残差，O(1)访问）

        self.block_codec = None

        # 随机访问器（BlockAccessor，独立模块）

        self.block_accessor = None

    

    def _load_data(self):

        """加载原始数据"""

        data_loader = NeurTSDataLoader(

            root_path=self.args.root_path,

            data_path=self.args.data_path,

            block_size=self.args.base_block_size,

            scale=True,

            data_col=self.args.data_col if hasattr(self.args, 'data_col') else 0

        )

        return data_loader

    

    def _build_model(self):

        """构建 NeurTS 模型组件"""

        args = self.args

        

        # 1. GlobalGridStorage: 双层网格存储

        grid_storage = GlobalGridStorage(

            raw_data=self.raw_data,

            block_size=args.base_block_size,

            trend_dim=args.trend_dim,

            context_dim=args.context_dim,

            max_patch_nodes=args.max_patch_nodes,

            init_mode='endpoint',

            context_init_std=0.02

        ).to(self.device)

        

        # 设置量化（如果启用）

        decoder_type = getattr(args, 'decoder_type', 'siren')

        if hasattr(args, 'quant_bits') and args.quant_bits > 0:

            # TCN 用随机噪声模式（柔和），SIREN 用 STE 模式（精确）

            grid_storage._quant_mode = 'noise' if decoder_type == 'tcn' else 'ste'

            grid_storage.setup_quantization(num_bits=args.quant_bits)

        

        # 2. NeurTSModel: 可插拔解码器

        model = NeurTSModel(

            grid_storage=grid_storage,

            max_block_size=args.base_block_size,

            hidden_dim=args.hidden_dim,

            decoder_type=decoder_type,

            pe_dim=args.pe_dim,

            num_res_blocks=args.num_res_blocks,

            kernel_size=args.kernel_size,

            dropout=args.dropout,

            strip_k=getattr(args, 'strip_k', 4),

            strip_c=getattr(args, 'strip_c', 16),

            hyper_hidden=getattr(args, 'hyper_hidden', 128),

            num_freqs=getattr(args, 'num_freqs', 64),

            max_aux_tokens=getattr(args, 'max_aux_tokens', 0),

            use_length_cond=getattr(args, 'use_length_cond', False),

            len_embed_dim=getattr(args, 'len_embed_dim', 16),

            aux_dim=getattr(args, 'aux_dim', -1),

            total_length=len(self.raw_data),

            transformer_nhead=getattr(args, 'transformer_nhead', 4),

            transformer_ffn_mult=getattr(args, 'transformer_ffn_mult', 4),

        ).to(self.device)

        

        # 3. GridManager: 索引管理器

        manager = GridManager(

            model=model,

            raw_data=self.raw_data,

            base_block_size=args.base_block_size,

            min_resolution=args.min_resolution

        )

        

        # 注入 aux token 查表回调：decode_batch / forward / decode_single 内部自动用

        # 行为：未调用 manager.add_aux_token 之前，_aux_map 为空，查表恒返回 ()，

        # 解码器收到 z_aux=None，与原版完全一致（向后兼容）

        model.aux_lookup = manager.get_aux_ids

        

        # 多 GPU 支持

        if args.use_multi_gpu and args.use_gpu:

            model = nn.DataParallel(model, device_ids=args.device_ids)

        

        print(f"[Exp_NeurTS] Model built:")

        print(f"    Base nodes: {grid_storage.num_base_nodes}")

        print(f"    Max patch nodes: {args.max_patch_nodes}")

        print(f"    Feature dim: {grid_storage.feature_dim}")

        print(f"    Index slots: {manager.num_slots}")

        

        return grid_storage, model, manager

    

    def _create_dataset(self):

        """创建训练数据集"""

        dataset = NeurTSDataset(

            raw_data=self.raw_data,

            manager=self.manager,

            grid_storage=self.grid_storage

        )

        return dataset

    

    def _get_data_loader(self, shuffle=True):

        """

        获取 DataLoader。

        

        训练时使用 SameLengthBatchSampler：每个 batch 只含同一种块长度，

        消除 _process_one_batch 中的分组循环开销，保持逆长度加权采样。

        """

        if shuffle:

            weights = self._compute_sample_weights()

            batch_sampler = SameLengthBatchSampler(

                manager=self.manager,

                batch_size=self.args.batch_size,

                weights=weights

            )

            return DataLoader(

                self.train_dataset,

                batch_sampler=batch_sampler,

                num_workers=self.args.num_workers,

            )

        else:

            return DataLoader(

                self.train_dataset,

                batch_size=self.args.batch_size,

                shuffle=False,

                num_workers=self.args.num_workers,

                drop_last=False

            )

    

    def _compute_sample_weights(self):

        """

        计算逆长度加权的采样权重。

        

        短块的权重更高，补偿其槽位少导致的训练不充分问题。

        权重 = base_block_size / block_len

        

        例如 (base=100, min=25):

        - 长度 100 的块：权重 = 100/100 = 1.0

        - 长度 25 的块：权重 = 100/25 = 4.0（被采样概率是长块的 4 倍）

        """

        weights = []

        num_slots = self.manager.num_slots

        min_res = self.manager.min_resolution

        base_size = self.manager.base_block_size

        

        idx = 0

        while idx < num_slots:

            entry = self.manager.index_table[idx]

            left_id, right_id, level_code = entry.left_id, entry.right_id, entry.level_code

            

            # 找到该块覆盖的所有槽位

            block_start_idx = idx

            block_end_idx = idx

            while block_end_idx < num_slots - 1:

                next_entry = self.manager.index_table[block_end_idx + 1]

                if (next_entry.left_id == left_id and next_entry.right_id == right_id and

                    next_entry.level_code == level_code):

                    block_end_idx += 1

                else:

                    break

            

            # 计算块长度和权重

            block_len = (block_end_idx - block_start_idx + 1) * min_res

            weight = base_size / block_len  # 逆长度加权

            

            # 为该块的所有槽位分配相同权重

            for _ in range(block_end_idx - block_start_idx + 1):

                weights.append(weight)

            

            idx = block_end_idx + 1

        

        return torch.tensor(weights, dtype=torch.float32)

    

    def _select_optimizer(self, finetune_only_vectors=False, force_new=False, freeze_decoder=False):

        """

        选择优化器（支持持久化，避免每轮微调重置优化器状态）。

        

        微调模式采用四级差异化学习率：

        - patch_grid（新分裂向量）：使用完整学习率，需要快速收敛

        - base_grid（已学好的基础向量）：使用 0.05x 学习率，微调适配新向量

        - decoder FiLM 层：使用 0.10x 学习率，适配新向量分布

        - decoder 核心 sin 权重：使用 0.005x 学习率，极度保守

        

        分裂轮 decoder 保护模式（freeze_decoder=True）：

        - 共享 decoder 在预训练时已充分拟合全量数据；分裂只是给难处增加

          局部向量自由度，不应再改 decoder。若让 decoder 继续训练，新分裂的

          难块会把共享 decoder 带偏，连累所有未分裂的大块（原始长度拟合变差）。

        - 因此冻结 decoder（split_decoder_lr_ratio<=0）或大幅压低其 LR，

          只训练 per-node 的 grid 向量（base_grid + patch_grid，彼此不污染）。

        

        Args:

            finetune_only_vectors: 如果为 True，四级差异化学习率训练所有组件

            force_new: 如果为 True，强制创建新优化器（用于预训练阶段）

            freeze_decoder: 如果为 True，分裂轮冻结/压低 decoder LR，保护已收敛大块

        """

        if finetune_only_vectors:

            current_mode = 'finetune'

        elif freeze_decoder:

            current_mode = 'split_frozen'

        else:

            current_mode = 'full'

        base_lr = self.args.learning_rate

        finetune_base_lr_ratio   = 0.05   # base_grid

        finetune_film_lr_ratio   = 0.10   # FiLM 调制权重

        finetune_core_lr_ratio   = 0.005  # 核心 sin 线性权重

        # 分裂轮 decoder LR 系数：<=0 表示硬冻结，>0 表示软冻结（按比例压低）

        split_dec_ratio = getattr(self.args, 'split_decoder_lr_ratio', 0.0)

        

        # 如果模式相同且优化器已存在，复用优化器（保持动量等状态）

        if not force_new and self._optimizer is not None and self._optimizer_mode == current_mode:

            # 重置学习率到初始值（修复跨 round 的 LR 衰减残留）

            if finetune_only_vectors:

                self._optimizer.param_groups[0]['lr'] = base_lr                             # patch_grid

                self._optimizer.param_groups[1]['lr'] = base_lr * finetune_base_lr_ratio    # base_grid

                self._optimizer.param_groups[2]['lr'] = base_lr * finetune_film_lr_ratio    # FiLM

                self._optimizer.param_groups[3]['lr'] = base_lr * finetune_core_lr_ratio    # core sin

            else:

                for pg in self._optimizer.param_groups:

                    pg['lr'] = base_lr

            # 清除 initial_lr 标记，让 adjust_learning_rate 用新值作为基准

            for pg in self._optimizer.param_groups:

                pg.pop('initial_lr', None)

            print(f"    [Optimizer] Reusing {current_mode} optimizer (preserving momentum, LR reset)")

            return self._optimizer

        

        # 需要创建新优化器

        if finetune_only_vectors:

            # 所有组件都可训练，四级差异化学习率

            for param in self.model.parameters():

                param.requires_grad = True

            

            # 分离 FiLM 调制权重 vs 核心 sin 线性权重

            # feature_strip 等无 FiLM 解码器：将全部 decoder 参数放入 film slot（适中 LR），core slot 留空

            film_params = [p for n, p in self.model.decoder.named_parameters() if 'film' in n]

            core_params = [p for n, p in self.model.decoder.named_parameters() if 'film' not in n]

            has_film = len(film_params) > 0

            if has_film:

                g2_params = film_params

                g3_params = core_params

            else:

                g2_params = list(self.model.decoder.parameters())

                g3_params = []

            

            model_optim = optim.Adam([

                {'params': [self.model.grid_storage.patch_grid], 'lr': base_lr},                        # 新节点：全速

                {'params': [self.model.grid_storage.base_grid], 'lr': base_lr * finetune_base_lr_ratio},# 旧节点：5%

                {'params': g2_params, 'lr': base_lr * finetune_film_lr_ratio},                          # FiLM / tiny MLP：10%

                {'params': g3_params, 'lr': base_lr * finetune_core_lr_ratio},                          # sin 核心（无 FiLM 时为空）：0.5%

            ])

            if has_film:

                print(f"    [Finetune/FiLM] patch_grid lr={base_lr:.2e}, base_grid lr={base_lr*finetune_base_lr_ratio:.2e}, "

                      f"film lr={base_lr*finetune_film_lr_ratio:.2e}, core lr={base_lr*finetune_core_lr_ratio:.2e}")

            else:

                print(f"    [Finetune/NoFiLM] patch_grid lr={base_lr:.2e}, base_grid lr={base_lr*finetune_base_lr_ratio:.2e}, "

                      f"decoder(tiny MLP) lr={base_lr*finetune_film_lr_ratio:.2e}")

        elif freeze_decoder:

            # 分裂轮 decoder 保护模式：只训 grid 向量，冻结/压低 decoder。

            gs = self.model.grid_storage

            if split_dec_ratio <= 0.0:

                # 硬冻结 decoder：彻底锁定共享权重，未分裂大块的拟合不可能再退化。

                for p in self.model.decoder.parameters():

                    p.requires_grad = False

                gs.base_grid.requires_grad_(True)

                gs.patch_grid.requires_grad_(True)

                model_optim = optim.Adam([

                    {'params': [gs.patch_grid], 'lr': base_lr},   # 新节点：全速

                    {'params': [gs.base_grid],  'lr': base_lr},   # 旧节点向量：全速（per-node，不污染 decoder）

                ])

                print(f"    [Split/FrozenDecoder] decoder FROZEN; patch_grid & base_grid lr={base_lr:.2e}")

            else:

                # 软冻结：decoder 仍可微调，但 LR 大幅压低，限制其漂移幅度。

                for param in self.model.parameters():

                    param.requires_grad = True

                model_optim = optim.Adam([

                    {'params': [gs.patch_grid], 'lr': base_lr},

                    {'params': [gs.base_grid],  'lr': base_lr},

                    {'params': list(self.model.decoder.parameters()), 'lr': base_lr * split_dec_ratio},

                ])

                print(f"    [Split/SoftFreeze] decoder lr={base_lr*split_dec_ratio:.2e} "
                      f"({split_dec_ratio:.3f}x), patch_grid & base_grid lr={base_lr:.2e}")

        else:

            # 确保所有参数都可训练

            for param in self.model.parameters():

                param.requires_grad = True

            model_optim = optim.Adam(list(self.model.parameters()), lr=base_lr)

        

        # 保存优化器状态

        self._optimizer = model_optim

        self._optimizer_mode = current_mode

        print(f"    [Optimizer] Created new {current_mode} optimizer")

        

        return model_optim

    

    def _unfreeze_decoder(self):

        """解冻 decoder，用于 pretrain 阶段或需要全量训练时。"""

        for param in self.model.decoder.parameters():

            param.requires_grad = True

    def _compute_split_related_rows(self):

        """计算"新分裂相关"的向量行 id，用于行级分层 LR。

        每次分裂涉及两个生效向量（decoder 只用 left_id = z_start）：
          - 右子块 z_start = 新节点 z_M（patch_grid）
          - 左子块 z_start = old_left_id（父块原左节点，base/patch_grid 均可能）
        扫描 index_table，凡 right_id 命中本轮新分裂节点集合的 slot，其 left_id
        即左子块向量；新节点本身也计入。

        Returns:
            (base_rows, patch_rows): base_grid / patch_grid 中"分裂相关"的本地行索引集合。
        """

        gs = self.model.grid_storage

        num_base = gs.num_base_nodes

        split_node_ids = getattr(self, '_last_split_node_ids', set())

        base_rows = set()

        patch_rows = set()

        if not split_node_ids:

            return base_rows, patch_rows

        for nid in split_node_ids:

            if nid >= num_base:

                patch_rows.add(int(nid) - num_base)

            else:

                base_rows.add(int(nid))

        for i in range(self.manager.num_slots):

            entry = self.manager.index_table[i]

            if entry.right_id in split_node_ids:

                lid = entry.left_id

                if lid >= num_base:

                    patch_rows.add(int(lid) - num_base)

                else:

                    base_rows.add(int(lid))

        return base_rows, patch_rows

    def _build_split_row_masks(self, base_rows, patch_rows, other_ratio):

        """逐行 LR 缩放掩码：分裂相关行=1.0，其余=other_ratio。可直接乘 .grad。"""

        gs = self.model.grid_storage

        dev = gs.base_grid.device

        base_mask = torch.full((gs.base_grid.shape[0], 1), float(other_ratio),

                               device=dev, dtype=gs.base_grid.dtype)

        patch_mask = None

        if gs.patch_grid is not None and gs.patch_grid.shape[0] > 0:

            patch_mask = torch.full((gs.patch_grid.shape[0], 1), float(other_ratio),

                                    device=dev, dtype=gs.patch_grid.dtype)

        for r in base_rows:

            if 0 <= r < base_mask.shape[0]:

                base_mask[r, 0] = 1.0

        if patch_mask is not None:

            for r in patch_rows:

                if 0 <= r < patch_mask.shape[0]:

                    patch_mask[r, 0] = 1.0

        return base_mask, patch_mask

    

    def _select_criterion(self):

        """选择损失函数"""

        return nn.MSELoss()

    

    def _process_one_batch(self, batch):

        """

        处理一个 batch 的数据（动态长度解码）

        

        改进：按块长度分组，每组独立解码，使位置编码完整覆盖 [0, 1]。

        这样每个块的位置编码语义一致：第一个点=0，最后一个点=1。

        

        Args:

            batch: (left_id, right_id, ground_truth, mask, offset, block_len)

                - ground_truth: [batch, base_block_size]，前 block_len 为真实值，后面为 0

                - mask: [batch, base_block_size]，前 block_len 为 1，后面为 0

                - offset: 当前始终为 0（保留接口一致性）

                - block_len: [batch]，每个样本的实际块长度

            

        Returns:

            loss: 重构损失

            pred: 预测波形

            true: 真实波形

        """

        left_ids, right_ids, ground_truth, mask, offsets, block_lens = batch

        

        # 转换为 tensor 并移动到设备

        if not isinstance(left_ids, torch.Tensor):

            left_ids = torch.tensor(left_ids)

        if not isinstance(right_ids, torch.Tensor):

            right_ids = torch.tensor(right_ids)

        if not isinstance(block_lens, torch.Tensor):

            block_lens = torch.tensor(block_lens)

        if not isinstance(offsets, torch.Tensor):

            offsets = torch.tensor(offsets)

            

        left_ids = left_ids.to(self.device)

        right_ids = right_ids.to(self.device)

        ground_truth = ground_truth.to(self.device)

        block_lens = block_lens.to(self.device)

        offsets = offsets.to(self.device)

        

        # 按块长度分组，每组独立解码

        unique_lens = torch.unique(block_lens)

        total_loss = 0.0

        total_count = 0

        

        for bl in unique_lens:

            bl_int = bl.item()

            group_mask = (block_lens == bl)

            group_indices = torch.where(group_mask)[0]

            

            # 提取该组的数据

            group_left = left_ids[group_indices]

            group_right = right_ids[group_indices]

            group_truth = ground_truth[group_indices, :bl_int]  # 只取有效部分

            group_offsets = offsets[group_indices]              # 物理偏移 [group_size]

            

            # 动态长度解码（传入物理偏移用于全局时间坐标）

            group_output = self.model.decode_batch(group_left, group_right, bl_int, group_offsets)

            group_output = group_output.squeeze(1)  # [group_size, bl_int]

            

            # 计算损失

            group_loss = ((group_output - group_truth) ** 2).sum()

            total_loss += group_loss

            total_count += group_indices.numel() * bl_int

        

        if total_count > 0:

            loss = total_loss / total_count

        else:

            loss = torch.tensor(0.0, device=self.device)

        

        # 多尺度正则 loss：L1（半块）+ L2（四分之一块）强制 decoder 多尺度工作

        # L1 对所有 bl>=8 块计算；L2 仅对 bl>=64（子块>=16 点）块计算，权重 0.5

        # scale_agnostic 解码器（如 feature_strip）忽略 offsets，左右半块调用返回相同输出，

        # 梯度互相对立，跳过多尺度 loss。

        _decoder_scale_agnostic = getattr(self.model.decoder, 'scale_agnostic', False)

        if self._multiscale_weight > 0 and not _decoder_scale_agnostic:

            ms_l1_total = 0.0

            ms_l1_count = 0

            ms_l2_total = 0.0

            ms_l2_count = 0

            for bl in unique_lens:

                bl_int = bl.item()

                if bl_int < 8:

                    continue

                half = bl_int // 2

                group_mask = (block_lens == bl)

                group_indices = torch.where(group_mask)[0]

                group_left = left_ids[group_indices]

                group_right = right_ids[group_indices]

                group_truth = ground_truth[group_indices, :bl_int]

                group_offsets = offsets[group_indices]

                

                # L1：左半段

                out_left = self.model.decode_batch(group_left, group_right, half, group_offsets)

                out_left = out_left.squeeze(1)

                ms_l1_total += ((out_left - group_truth[:, :half]) ** 2).sum()

                # L1：右半段

                out_right = self.model.decode_batch(group_left, group_right, half, group_offsets + half)

                out_right = out_right.squeeze(1)

                ms_l1_total += ((out_right - group_truth[:, half:half*2]) ** 2).sum()

                ms_l1_count += group_indices.numel() * half * 2

                

                # L2：四等分（仅 bl >= 64，保证每段 >= 16 点）

                if bl_int >= 64:

                    quarter = bl_int // 4

                    for seg in range(4):

                        out_q = self.model.decode_batch(

                            group_left, group_right, quarter, group_offsets + seg * quarter)

                        out_q = out_q.squeeze(1)

                        ms_l2_total += ((out_q - group_truth[:, seg*quarter:(seg+1)*quarter]) ** 2).sum()

                    ms_l2_count += group_indices.numel() * quarter * 4

            

            ms_total = 0.0

            if ms_l1_count > 0:

                ms_total += ms_l1_total / ms_l1_count

            if ms_l2_count > 0:

                ms_total += 0.5 * (ms_l2_total / ms_l2_count)

            loss = loss + self._multiscale_weight * ms_total

        

        return loss, None, None  # pred/true 不再返回（动态长度无法拼接）

    

    def train(self, setting, epochs: int = None, phase: str = "train", finetune_only_vectors: bool = False):

        """

        训练主循环

        

        Args:

            setting: 实验命名

            epochs: 训练轮数，默认使用 args.train_epochs

            phase: 训练阶段标识，用于日志输出 ("pretrain" / "finetune" / "train")

            finetune_only_vectors: 如果为 True，使用三级差异化学习率

        """

        args = self.args

        num_epochs = epochs if epochs is not None else args.train_epochs

        

        # 创建保存路径

        path = os.path.join(args.checkpoints, setting)

        if not os.path.exists(path):

            os.makedirs(path)

        

        # 保存配置（仅首次）

        args_path = os.path.join(path, "args.json")

        if not os.path.exists(args_path):

            with open(args_path, 'w') as f:

                json.dump(vars(args), f, indent=True)

        

        # 保存 scaler 统计信息（仅首次）

        scale_path = os.path.join(path, "scale_statistic.pkl")

        if self.data_loader.scaler is not None and not os.path.exists(scale_path):

            scale_statistic = {

                'mean': self.data_loader.scaler.mean.tolist() if hasattr(self.data_loader.scaler.mean, 'tolist') else self.data_loader.scaler.mean,

                'std': self.data_loader.scaler.std.tolist() if hasattr(self.data_loader.scaler.std, 'tolist') else self.data_loader.scaler.std

            }

            with open(scale_path, 'wb') as f:

                pickle.dump(scale_statistic, f)

        

        # 获取 DataLoader（每次训练都重新获取，因为分裂后数据集可能变化）

        train_loader = self._get_data_loader(shuffle=True)

        train_steps = len(train_loader)

        

        # Early stopping（微调阶段使用更宽松的 patience）

        finetune_phase = phase.startswith('finetune') or phase.startswith('progressive')

        patience = getattr(args, 'finetune_patience', args.patience) if finetune_phase else args.patience

        early_stopping = EarlyStopping(patience=patience, verbose=True)

        

        # 优化器和损失函数

        # 预训练阶段(phase='pretrain')强制创建新优化器，微调阶段复用优化器保持动量

        force_new_optimizer = (phase == 'pretrain')

        # 分裂轮（progressive）decoder 保护：冻结/压低 decoder LR，避免难块把

        # 共享 decoder 带偏、连累未分裂的大块。由 --freeze_decoder_in_split 开关控制。

        _is_progressive = phase.startswith('progressive')

        _freeze_dec = bool(getattr(args, 'freeze_decoder_in_split', False)) and _is_progressive

        # decoder 保护模式下每轮新建优化器（force_new）：丢弃上一轮残留的动量，

        # 做干净的 warm restart（与 pretrain→finetune 切换同理），避免大动量把

        # 已收敛权重踹飞。

        if _freeze_dec:

            force_new_optimizer = True

        # 分裂轮启用 Phase-3 差异化 LR（finetune_only_vectors=True）时，也每轮

        # 新建优化器做热重启 —— Phase 3 的有效性正来自"新建 finetune 优化器"

        # （消融显示它把 6.14x 精修回 6.57x）。复用旧动量会削弱这一效果。

        if _is_progressive and finetune_only_vectors:

            force_new_optimizer = True

        model_optim = self._select_optimizer(finetune_only_vectors=finetune_only_vectors,

                                             force_new=force_new_optimizer,

                                             freeze_decoder=_freeze_dec)

        # 若本轮未冻结 decoder（或处于非分裂阶段），确保 decoder 在上一轮被冻结后

        # 重新解冻（_select_optimizer 的硬冻结分支会把 requires_grad 置 False）。

        if not _freeze_dec:

            self._unfreeze_decoder()

        # ── 行级分层 LR（--split_layered_lr）──────────────────────────────
        # 分裂轮专用：新分裂的两个向量（z_M + 左子块 old_left_id）满速学，
        # 其余所有 grid 行 + decoder 用 other_ratio×lr。通过"专用优化器
        # (base/patch 满速 + decoder 低速) + 逐行梯度缩放(非分裂行 ×other_ratio)"
        # 实现张量内部的行级差异 LR。
        _layered = bool(getattr(args, 'split_layered_lr', False)) and _is_progressive

        _layered_base_mask = None

        _layered_patch_mask = None

        if _layered:

            _other_ratio = float(getattr(args, 'split_other_lr_ratio', 0.05))

            gs = self.model.grid_storage

            for p in self.model.parameters():

                p.requires_grad = True

            _layered_groups = [

                {'params': [gs.base_grid],  'lr': args.learning_rate},

                {'params': [gs.patch_grid], 'lr': args.learning_rate},

                {'params': list(self.model.decoder.parameters()),

                 'lr': args.learning_rate * _other_ratio},

            ]

            model_optim = optim.Adam(_layered_groups)

            self._optimizer = model_optim

            self._optimizer_mode = 'split_layered'

            _base_rows, _patch_rows = self._compute_split_related_rows()

            _layered_base_mask, _layered_patch_mask = self._build_split_row_masks(

                _base_rows, _patch_rows, _other_ratio)

            print(f"    [Split/LayeredLR] {len(_base_rows)} base + {len(_patch_rows)} patch rows "

                  f"@ full lr={args.learning_rate:.2e}; other rows & decoder "

                  f"@ {_other_ratio:.3f}x = {args.learning_rate * _other_ratio:.2e}")

        criterion = self._select_criterion()

        

        # 多尺度正则权重：从 args 读取，支持命令行关闭（0.0）

        _ms_pretrain = getattr(args, 'multiscale_pretrain_weight', 0.3)

        _ms_finetune = getattr(args, 'multiscale_finetune_weight', 0.1)

        if phase == 'pretrain':

            self._multiscale_weight = _ms_pretrain

        elif phase == 'final_finetune':

            self._multiscale_weight = 0.0

        else:  # progressive_rN / finetune_rN

            self._multiscale_weight = _ms_finetune

        

        num_blocks = len(self.manager.get_all_unique_blocks())

        print(f"\n[{phase.upper()}] Starting {phase} for {num_epochs} epochs...")

        print(f"    Current blocks: {num_blocks}")

        print(f"    Total samples: {len(self.train_dataset)}")

        print(f"    Batch size: {args.batch_size}")

        print(f"    Steps per epoch: {train_steps}")

        

        for epoch in range(num_epochs):

            time_now = time.time()

            iter_count = 0

            train_loss = []

            

            self.model.train()

            epoch_time = time.time()

            

            for i, batch in enumerate(train_loader):

                iter_count += 1

                model_optim.zero_grad()

                

                loss, pred, true = self._process_one_batch(batch)

                train_loss.append(loss.item())

                

                if (i + 1) % 100 == 0:

                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")

                    speed = (time.time() - time_now) / iter_count

                    left_time = speed * ((num_epochs - epoch) * train_steps - i)

                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')

                    iter_count = 0

                    time_now = time.time()

                

                loss.backward()

                # 行级分层 LR：非分裂行的梯度按 other_ratio 缩放，
                # 等效"分裂相关行满速、其余行低速"。decoder 的低速由其 param
                # group 的低 lr 实现，这里只缩 grid 两个张量的非分裂行。
                if _layered_base_mask is not None:

                    gs = self.model.grid_storage

                    if gs.base_grid.grad is not None:

                        gs.base_grid.grad.mul_(_layered_base_mask)

                    if _layered_patch_mask is not None and gs.patch_grid.grad is not None:

                        gs.patch_grid.grad.mul_(_layered_patch_mask)

                model_optim.step()

                

                # Active-Dim 约束：还原 split 节点的 frozen dims

                # 必须在 step() 之后执行，以对抗 Adam 动量引起的漂移

                if self._frozen_patch_values:

                    _pg = self.model.grid_storage.patch_grid.data

                    for _lid, _fv in self._frozen_patch_values.items():

                        _pg[_lid, -_fv.shape[0]:] = _fv

            

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s")

            train_loss_avg = np.average(train_loss)

            

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss_avg:.7f}")

            

            # Early stopping

            early_stopping(train_loss_avg, self.model, path)

            if early_stopping.early_stop:

                print("Early stopping")

                break

            

            # 分阶段 cosine 衰减：微调 60 轮后开始，预训练/final 100 轮后开始
            if finetune_phase:
                _decay_start = getattr(args, 'finetune_lr_decay_start', 60)
            else:
                _decay_start = getattr(args, 'pretrain_lr_decay_start', 100)
            _ep = epoch + 1
            if _ep >= _decay_start:
                _progress = (_ep - _decay_start) / max(1, num_epochs - _decay_start)
                _factor   = max(0.5 * (1.0 + math.cos(math.pi * min(_progress, 1.0))), 0.01)
                for _pg in model_optim.param_groups:
                    if 'initial_lr' not in _pg:
                        _pg['initial_lr'] = _pg['lr']
                    _pg['lr'] = _pg['initial_lr'] * _factor
                if _ep == _decay_start:
                    print(f"    [LR] Cosine decay started (epoch {_decay_start}): factor={_factor:.4f}")
            else:
                adjust_learning_rate(model_optim, _ep, args)

        

        # 加载最佳模型

        best_model_path = os.path.join(path, 'checkpoint.pth')

        if os.path.exists(best_model_path):

            self.model.load_state_dict(torch.load(best_model_path))

        

        # 保存最终模型

        state_dict = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()

        torch.save(state_dict, os.path.join(path, 'checkpoint.pth'))

        

        # 保存 GridManager 状态（仅存储 left_id, right_id，不存 level_code）

        manager_state = {

            'patch_counter': self.manager.patch_counter,

            'index_table': [(e.left_id, e.right_id) for e in self.manager.index_table]

        }

        with open(os.path.join(path, 'manager_state.pkl'), 'wb') as f:

            pickle.dump(manager_state, f)

        

        print(f"\n[{phase.upper()}] Completed. Model saved to {path}")

        

        return self.model

    

    def refresh_dataset(self):

        """分裂后刷新数据集，重新创建 DataLoader"""

        self.train_dataset = self._create_dataset()

        print(f"[Dataset] Refreshed: {len(self.train_dataset)} samples")

    

    def evaluate(self, setting):

        """评估模型：计算整体重构误差"""

        self.model.eval()

        self._refresh_quant_params()

        

        test_loader = self._get_data_loader(shuffle=False)

        total_loss = []

        

        with torch.no_grad():

            for batch in test_loader:

                loss, pred, true = self._process_one_batch(batch)

                total_loss.append(loss.item())

        

        avg_loss = np.average(total_loss)

        print(f"[Evaluate] Average reconstruction MSE: {avg_loss:.7f}")

        

        return avg_loss

    

    def reconstruct_full(self):

        """重构完整时间序列（动态长度解码）"""

        self.model.eval()

        self._refresh_quant_params()

        

        all_blocks = self.manager.get_all_unique_blocks()

        reconstructed = torch.zeros(self.manager.total_length, device=self.device)

        

        # 按块长度分组，每组独立解码

        from collections import defaultdict

        len_groups = defaultdict(list)

        for i, (start_time, end_time, left_id, right_id, level_code) in enumerate(all_blocks):

            block_len = end_time - start_time

            len_groups[block_len].append((i, start_time, end_time, left_id, right_id))

        

        eval_bs = getattr(self.args, 'eval_batch_size', 256)

        with torch.no_grad():

            for block_len, group in len_groups.items():

                for chunk_start in range(0, len(group), eval_bs):

                    chunk = group[chunk_start:chunk_start + eval_bs]

                    indices, starts, ends, lefts, rights = zip(*chunk)

                    

                    left_ids = torch.tensor(lefts, device=self.device)

                    right_ids = torch.tensor(rights, device=self.device)

                    blk_offsets = torch.tensor(

                        [s % self.args.base_block_size for s in starts], device=self.device)

                    

                    # 动态长度解码（传入物理偏移用于全局时间坐标）

                    outputs = self.model.decode_batch(left_ids, right_ids, block_len, blk_offsets)

                    outputs = outputs.squeeze(1)  # [chunk_size, block_len]

                    

                    for j, (start_time, end_time) in enumerate(zip(starts, ends)):

                        reconstructed[start_time:end_time] = outputs[j]

        

        return reconstructed

    

    def spectral_oracle(self, K_values=(8, 16, 32), max_blocks=None):
        """
        Oracle: 分析 FourierDecoder 残差的 DST 集中度。

        对每个块计算：
          residual r = raw_data - decoder_output
          DST(r) → top-K 系数占总能量的比例

        结论：
          集中度 > 0.7 → 谱细化有效，K 个系数能压缩大量残差信息
          集中度 < 0.5 → 残差接近白噪声，谱细化无优势

        Args:
            K_values : tuple[int]  评估的 K 档位，默认 (8, 16, 32)
            max_blocks : int | None  限制分析块数（None = 全量）
        """
        import numpy as np
        try:
            from scipy.fft import dst as scipy_dst
        except ImportError:
            raise ImportError("scipy 未安装，请 pip install scipy")

        self.model.eval()
        all_blocks = self.manager.get_all_unique_blocks()
        if max_blocks is not None:
            all_blocks = all_blocks[:max_blocks]

        # 获取误差阈值（用于区分难块/易块）
        threshold   = getattr(self.args, 'eval_threshold', 1.0)
        error_mode  = getattr(self.args, 'error_mode', 'absolute')
        eval_bs     = getattr(self.args, 'eval_batch_size', 256)

        # 按块长分组批量解码
        from collections import defaultdict
        len_groups = defaultdict(list)
        for entry in all_blocks:
            start_time, end_time, left_id, right_id, level_code = entry
            bl = end_time - start_time
            len_groups[bl].append((start_time, end_time, left_id, right_id))

        # 结果累积
        records = []  # dict per block

        with torch.no_grad():
            for block_len, group in len_groups.items():
                for chunk_start in range(0, len(group), eval_bs):
                    chunk   = group[chunk_start:chunk_start + eval_bs]
                    starts  = [c[0] for c in chunk]
                    ends    = [c[1] for c in chunk]
                    lefts   = [c[2] for c in chunk]
                    rights  = [c[3] for c in chunk]

                    left_ids  = torch.tensor(lefts,  device=self.device)
                    right_ids = torch.tensor(rights, device=self.device)
                    blk_off   = torch.tensor(
                        [s % self.args.base_block_size for s in starts],
                        device=self.device)

                    outputs = self.model.decode_batch(
                        left_ids, right_ids, block_len, blk_off
                    ).squeeze(1)  # [B, T]

                    for j, (st, en) in enumerate(zip(starts, ends)):
                        pred = outputs[j]                    # [T] normalised space
                        true = self.raw_data[st:en]          # [T] normalised space

                        r      = (true - pred).cpu().float().numpy()  # [T]
                        max_e  = float(np.abs(r).max())
                        mae    = float(np.abs(r).mean())
                        is_hard = max_e > threshold

                        # DST-1 analysis: coefficients c_k = Σ r_n·sin(π(n+1)(k+1)/(N+1))
                        # scipy dst type=1 matches FourierDecoder's half-period sine basis
                        coeffs  = scipy_dst(r, type=1)
                        energy  = np.abs(coeffs)
                        total_e = energy.sum() + 1e-12
                        order   = np.argsort(energy)[::-1]

                        conc = {}
                        cumulative = 0.0
                        for rank, idx in enumerate(order):
                            cumulative += energy[idx]
                            k_check = rank + 1
                            if k_check in K_values:
                                conc[k_check] = cumulative / total_e
                            if k_check >= max(K_values):
                                break

                        records.append(dict(
                            start=st, end=en, block_len=block_len,
                            max_err=max_e, mae=mae, is_hard=is_hard,
                            conc=conc,
                        ))

        # ── 统计报告 ──────────────────────────────────────────────────
        hard   = [r for r in records if r['is_hard']]
        easy   = [r for r in records if not r['is_hard']]
        n_tot  = len(records)

        print(f"\n{'='*62}")
        print(f"  Spectral Oracle  (threshold={threshold}, mode={error_mode})")
        print(f"{'='*62}")
        print(f"  Total blocks : {n_tot}   Hard : {len(hard)} ({len(hard)/n_tot*100:.1f}%)   "
              f"Easy : {len(easy)} ({len(easy)/n_tot*100:.1f}%)")
        print(f"  {'K':>4}  {'Hard conc (avg)':>18}  {'Easy conc (avg)':>18}  {'All conc (avg)':>16}")
        print(f"  {'-'*62}")
        for K in sorted(K_values):
            hc = float(np.mean([r['conc'][K] for r in hard]))   if hard   else float('nan')
            ec = float(np.mean([r['conc'][K] for r in easy]))   if easy   else float('nan')
            ac = float(np.mean([r['conc'][K] for r in records]))
            verdict = "✓ 有效" if hc > 0.70 else ("△ 边缘" if hc > 0.50 else "✗ 无效")
            print(f"  K={K:<3}  {hc:>17.1%}  {ec:>17.1%}  {ac:>15.1%}   {verdict}")

        # 按块长汇总 hard-block 集中度
        if hard:
            print(f"\n  [Hard blocks DST concentration by block size, K={max(K_values)}]")
            bl_groups = defaultdict(list)
            for r in hard:
                bl_groups[r['block_len']].append(r['conc'][max(K_values)])
            for bl in sorted(bl_groups):
                vals = bl_groups[bl]
                print(f"    size={bl:<6}  n={len(vals):<5}  avg={np.mean(vals):.1%}  "
                      f"p50={np.percentile(vals,50):.1%}  p25={np.percentile(vals,25):.1%}")

        print(f"{'='*62}\n")
        return records


    def compute_compression_ratio(self):

        """

        计算压缩比（快速估算，不含 Tier 2/3 兜底开销）。

        

        报告 Grid 和 Index 的实际大小和理论压缩大小。

        完整的压缩比（含 Tier 2/3）由 final_evaluation() 报告。

        

        大小计算明细：

        - Grid (float32): total_nodes × vec_dim × 4 bytes        → 当前内存

        - Grid (8-bit):    total_nodes × vec_dim × 1 byte         → 量化后

        - Index (per-slot): num_slots × 4 bytes (2×uint16)        → 最细粒度表实际大小

        - Index (per-block): num_blocks × 5 bytes                 → 落盘压缩格式

        - total_nodes = num_base_nodes + patch_counter             → 实际使用节点数

        - num_slots = total_length // min_resolution               → 最细粒度槽位数

        """

        stats = self.manager.get_statistics()

        total_nodes = stats['total_nodes']

        num_unique_blocks = stats['num_unique_blocks']

        num_slots = stats['num_slots']

        vec_dim = self.grid_storage.trend_dim + self.grid_storage.context_dim

        

        original_bytes = self.manager.total_length * 4  # float32

        

        # Grid 大小
        _quant_bits = getattr(self.args, 'quant_bits', 8)
        # bytes per stored dim: quantised -> quant_bits/8 (e.g. 1.0 for
        # 8-bit, 2.0 for 16-bit); unquantised (quant_bits == 0) -> 4.0
        # (raw float32, the actual on-disk cost when quantisation is off).
        _bytes_per_dim = (_quant_bits / 8) if _quant_bits > 0 else 4.0

        # Aux 异构维度（Path A）
        _model_aux_dim = getattr(self.model, 'aux_dim', vec_dim)
        _aux_count = self.manager.get_aux_stats().get('total_aux_tokens', 0) \
            if hasattr(self.manager, 'get_aux_stats') else 0
        _main_count = total_nodes - _aux_count

        grid_f32 = (_main_count * vec_dim + _aux_count * _model_aux_dim) * 4

        grid_quantized = int(
            _main_count * vec_dim * _bytes_per_dim
            + _aux_count * _model_aux_dim * _bytes_per_dim
        )

        

        # Index 大小

        index_slot = num_slots * 4             # per-slot: left_id(u16) + right_id(u16) = 4 bytes

        index_block = num_unique_blocks * 5    # per-block: ~5 bytes (left_id + right_id + len)

        

        # 压缩比（用 quant_bits grid + per-block index，最优理论值）

        compressed_bytes = grid_quantized + index_block

        ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else float('inf')

        

        print(f"\n[Compression Stats (excl. Tier 2/3)]")

        print(f"    Original data:       {original_bytes / 1024:>10.2f} KB ({self.manager.total_length} points x float32)")

        if _aux_count > 0 and _model_aux_dim < vec_dim:
            print(f"    Grid (float32):      {grid_f32 / 1024:>10.2f} KB "
                  f"(main {_main_count}n×{vec_dim}d + aux {_aux_count}n×{_model_aux_dim}d, ×4B)")
            print(f"    Grid ({_quant_bits}-bit):      {grid_quantized / 1024:>10.2f} KB "
                  f"(main {_main_count}n×{vec_dim}d + aux {_aux_count}n×{_model_aux_dim}d, "
                  f"×{_bytes_per_dim:.1f}B)")
        else:
            print(f"    Grid (float32):      {grid_f32 / 1024:>10.2f} KB ({total_nodes} nodes x {vec_dim}d x 4B)")
            print(f"    Grid ({_quant_bits}-bit):      {grid_quantized / 1024:>10.2f} KB ({total_nodes} nodes x {vec_dim}d x {_bytes_per_dim:.1f}B)")

        print(f"    Index (per-slot):    {index_slot / 1024:>10.2f} KB ({num_slots} slots x 4B)")

        print(f"    Index (per-block):   {index_block / 1024:>10.2f} KB ({num_unique_blocks} blocks x 5B)")

        print(f"    Base ratio ({_quant_bits}-bit+per-block): {ratio:.2f}x")

        

        return ratio

    

    def _compute_block_residual(self, t_start, t_end):

        """计算指定时间区间 [t_start, t_end) 的残差向量 (true - pred)。用于自适应折扣校准。"""

        try:

            block_len = t_end - t_start

            true_data = self.raw_data[t_start:t_end]

            slot = t_start // self.manager.min_resolution

            entry = self.manager.index_table[slot]

            offset = t_start % self.args.base_block_size

            with torch.no_grad():

                pred = self.model.decode_single(

                    entry.left_id, entry.right_id, block_len, offset=offset).squeeze()

            return (true_data - pred)

        except Exception:

            return None

    

    def _refresh_quant_params(self):

        """刷新全局量化参数，确保 eval 模式下 fake_quantize 使用正确的 min/max/scale。"""

        if hasattr(self.grid_storage, '_quantization_enabled') and self.grid_storage._quantization_enabled:

            if not (hasattr(self.grid_storage, '_is_quantized') and self.grid_storage._is_quantized):

                self.grid_storage.compute_quantization_params()

    

    @staticmethod

    def _edwb_cliff_stats(span: float, length: int, eps: float):

        """

        单个残差块的 EDWB 位宽 cliff 统计（复用 FallbackDict 真实公式）。

        

        全部复用 FallbackDict.compute_bits / estimate_bitwidth_cost，

        不重新实现 EDWB，保证与最终落盘核算一致。

        

        Returns: dict

            bits              : 当前位宽

            cost              : 当前 EDWB 字节代价（header + payload）

            next_threshold    : 降到 (bits-1) 所需的 span 上界 = 2ε·2^(bits-1)

            needed_drop       : max(0, span - next_threshold)

            needed_drop_ratio : needed_drop / max(span, 1e-12)，越小越接近台阶

            one_bit_saving    : 位宽降 1 档的 payload 字节节省

        """

        bits = FallbackDict.compute_bits(span, eps)

        cost = FallbackDict.estimate_bitwidth_cost(span, length, eps)

        if bits <= 0:

            return {'bits': bits, 'cost': cost, 'next_threshold': 0.0,

                    'needed_drop': 0.0, 'needed_drop_ratio': 0.0,

                    'one_bit_saving': 0}

        next_threshold = 2.0 * eps * (2 ** (bits - 1))

        needed_drop = max(0.0, span - next_threshold)

        needed_drop_ratio = needed_drop / max(span, 1e-12)

        # payload 字节差：当前 bits 档 vs 低一档

        import math as _math

        payload_b = _math.ceil(length * bits / 8)

        payload_b1 = _math.ceil(length * (bits - 1) / 8) if bits - 1 > 0 else 0

        one_bit_saving = payload_b - payload_b1

        return {'bits': bits, 'cost': cost, 'next_threshold': next_threshold,

                'needed_drop': needed_drop, 'needed_drop_ratio': needed_drop_ratio,

                'one_bit_saving': one_bit_saving}

    

    def _cliff_split_score(self, residual, block_len: int, eps: float,

                           z_M_cost: int, gamma: float = 8.0):

        """

        基于「当前 base-model 残差」估计一次中点分裂的字节转化潜力。

        

        不使用分裂后训练结果（无 oracle），只把父块残差在中点切两半分别用

        EDWB 公式计费：

            coding_only_gain = parent_cost - (left_cost + right_cost)

            base_net         = coding_only_gain - z_M_cost

            child_potential  = Σ_{L,R} one_bit_saving · exp(-gamma · needed_drop_ratio)

            score            = base_net + child_potential

        

        score>0 表示：要么光重新分块编码就已省字节（base_net>0），

        要么子块已贴近下一档位宽台阶、稍加训练即可降 1 bit（child_potential 大）。

        

        Args:

            residual: 父块带符号残差张量 [block_len]

            block_len: 父块长度

            eps: 该块的 EDWB 误差容限（归一化空间）

            z_M_cost: 新增 z_M 节点字节成本

            gamma: cliff 潜力衰减率

        Returns: dict（含 score 及全部中间量，供报告复用）

        """

        import math as _math

        mid = block_len // 2

        parent_span = (residual.max() - residual.min()).item()

        left_res = residual[:mid]

        right_res = residual[mid:]

        left_span = (left_res.max() - left_res.min()).item() if mid > 0 else 0.0

        right_span = (right_res.max() - right_res.min()).item() if (block_len - mid) > 0 else 0.0

        

        parent_cost = FallbackDict.estimate_bitwidth_cost(parent_span, block_len, eps)

        left_stats = self._edwb_cliff_stats(left_span, mid, eps)

        right_stats = self._edwb_cliff_stats(right_span, block_len - mid, eps)

        

        coding_only_gain = parent_cost - (left_stats['cost'] + right_stats['cost'])

        base_net = coding_only_gain - z_M_cost

        child_potential = (

            left_stats['one_bit_saving'] * _math.exp(-gamma * left_stats['needed_drop_ratio'])

            + right_stats['one_bit_saving'] * _math.exp(-gamma * right_stats['needed_drop_ratio'])

        )

        score = base_net + child_potential

        return {

            'score': score,

            'parent_cost': parent_cost,

            'parent_span': parent_span,

            'parent_bits': FallbackDict.compute_bits(parent_span, eps),

            'left_cost': left_stats['cost'], 'right_cost': right_stats['cost'],

            'left_bits': left_stats['bits'], 'right_bits': right_stats['bits'],

            'left_needed_drop_ratio': left_stats['needed_drop_ratio'],

            'right_needed_drop_ratio': right_stats['needed_drop_ratio'],

            'coding_only_gain': coding_only_gain,

            'base_net': base_net,

            'child_potential': child_potential,

            'z_M_cost': z_M_cost,

        }

    

    def adaptive_split(self, error_threshold=0.10, max_splits=100, error_mode='relative', retrain_discount=1.0, min_split_bits=4, min_split_savings=0.0):

        """

        自适应分裂：动态位宽代价驱动 + 中分（Bitwidth Cost Driven, Midpoint Split）。

        

        中分优势：树结构完全隐式，索引仅需 1B bitmask/base_block。

        判决逻辑（直接对比位宽代价）：

        - keep_cost   = BITWIDTH_HEADER + ceil(block_len × bits / 8)

        - split_cost  = NODE + discount × (bitwidth_cost(left) + bitwidth_cost(right))

        - savings     = keep_cost - split_cost  （预期残差节省 - 分裂代价）

        - 当 savings > min_split_savings × keep_cost 时分裂

          （min_split_savings=0 时退化为 split_cost < keep_cost，与旧行为一致）

        

        自适应折扣（Adaptive Discount）：

        - 首轮：discount = retrain_discount（命令行参数，默认 1.0 保守）

        - 后续轮：discount = measured_discount（上轮实测 actual/predicted）

        - 系统自动校准：重训收益大 → discount 降 → 更多分裂

        -                 重训无效  → discount ≈ 1.0 → 停止分裂

        

        Args:

            error_threshold: 坏点误差阈值。relative模式下为比例（0.10=10%），absolute模式下为原始单位

            max_splits: 最大分裂次数

            error_mode: 'relative'（百分比误差）或 'absolute'（绝对误差）

            retrain_discount: 首轮折扣因子∈(0,1]。后续轮被自适应值覆盖。默认 1.0（保守）。

            min_split_bits: 仅分裂位宽 >= 该阈值的块。默认 4。

            min_split_savings: 最小净收益比例∈[0,1)。仅当预期节省 (keep_cost-split_cost)

                超过 keep_cost 的该比例时才分裂，用于过滤折扣假设可能失真的边际分裂。

                0.0（默认）= 无过滤，只要净收益为正即分裂。

        """

        # 获取 scaler 参数

        scaler = self.data_loader.scaler

        if scaler is not None:

            std_val = scaler.std.item() if hasattr(scaler.std, 'item') else float(scaler.std)

            mean_val = scaler.mean.item() if hasattr(scaler.mean, 'item') else float(scaler.mean)

        else:

            std_val, mean_val = 1.0, 0.0

        

        if error_mode == 'absolute':

            norm_threshold = error_threshold / std_val

            thresh_desc = f"{error_threshold} (orig) / {norm_threshold:.4f} (norm), mode=absolute"

        else:

            norm_threshold = None  # relative mode: per-point comparison

            thresh_desc = f"{error_threshold*100:.1f}% relative error"

        

        # ================================================================

        # 自适应折扣：用上轮分裂的实测结果校准本轮 discount

        # ================================================================

        if self._split_predictions and self._adaptive_discount is None:

            # 上轮有分裂记录，计算 measured_discount

            ratios = []

            for (s, e), pred in self._split_predictions.items():

                if (s, e) not in self.split_history:

                    continue

                split_t, _ = self.split_history[(s, e)]

                eps = pred['eps']

                # 计算实际子块 BW 代价

                left_res = self._compute_block_residual(s, split_t)

                right_res = self._compute_block_residual(split_t, e)

                if left_res is not None and right_res is not None:

                    left_span = (left_res.max() - left_res.min()).item()

                    right_span = (right_res.max() - right_res.min()).item()

                    actual_bw = (FallbackDict.estimate_bitwidth_cost(left_span, split_t - s, eps) +

                                 FallbackDict.estimate_bitwidth_cost(right_span, e - split_t, eps))

                    predicted_bw = pred['predicted_child_bw']

                    if predicted_bw > 0:

                        ratios.append(actual_bw / predicted_bw)

            if ratios:

                # discount = 实测 actual/predicted 子块残差比。
                #   < 1 : 重训有效，子块残差比预测更小 → 鼓励分裂
                #   > 1 : 重训未兑现，子块残差比预测更大 → split_cost 被抬高，
                #         自动抑制分裂（甚至 split_cost > keep_cost 直接不分）
                # 上限放到 1.5（而非旧的 0.9）：旧上限把 discount 焊死在乐观区，
                # 成本模型永远无法判定"分裂在亏损"，导致每轮顶配额过度分裂。
                self._adaptive_discount = float(np.clip(np.mean(ratios), 0.5, 1.5))

                print(f"[Adaptive Discount] measured from {len(ratios)} splits: "

                      f"avg(actual/predicted) = {np.mean(ratios):.4f} → discount = {self._adaptive_discount:.4f}")

        

        # 确定本轮使用的 discount

        if self._adaptive_discount is not None:

            effective_discount = self._adaptive_discount

        else:

            effective_discount = retrain_discount

        

        # 解码器是否为尺度无关模式（feature_strip 等）

        # 尺度无关：最大误差准则 + 误差点吸附分裂，代替位宽代价模型 + 中分

        # 尺度敏感（siren）：保持原有位宽代价模型 + 中分逻辑

        _use_max_error = getattr(self.model.decoder, 'scale_agnostic', False)

        

        all_blocks = self.manager.get_all_unique_blocks()

        total_blocks_before = len(all_blocks)

        print(f"\n[Adaptive Split] Current blocks: {total_blocks_before}, threshold={thresh_desc}, "

              f"max_splits={max_splits}, discount={effective_discount:.4f}")

        

        self.model.eval()

        self._refresh_quant_params()

        

        # 筛选可分裂的块

        splittable_blocks = []

        for block in all_blocks:

            start_time, end_time, left_id, right_id, level_code = block

            block_len = end_time - start_time

            if block_len >= 2 * self.manager.min_resolution and level_code < self.manager.max_level:

                splittable_blocks.append(block)

        

        if not splittable_blocks:

            print(f"    No splittable blocks.")

            return 0

        

        # 按块长度分组，每组独立解码（动态长度解码）

        block_errors = {}  # block -> (max_error, error_curve, bad_point_count, block_len)

        

        from collections import defaultdict

        len_groups = defaultdict(list)

        for block in splittable_blocks:

            start_time, end_time, left_id, right_id, level_code = block

            block_len = end_time - start_time

            len_groups[block_len].append(block)

        

        eval_bs = getattr(self.args, 'eval_batch_size', 256)

        with torch.no_grad():

            for block_len, group in len_groups.items():

                # mini-batch 分片，防止显存溢出

                # 尺度无关解码器（feature_strip）：half-block 两次查询输出相同，improvement_ratio 无意义，跳过

                compute_imp = (block_len >= 32) and not _use_max_error

                half_len = block_len // 2

                for chunk_start in range(0, len(group), eval_bs):

                    chunk = group[chunk_start:chunk_start + eval_bs]

                    left_ids = torch.tensor([b[2] for b in chunk], device=self.device)

                    right_ids = torch.tensor([b[3] for b in chunk], device=self.device)

                    blk_offsets = torch.tensor(

                        [b[0] % self.args.base_block_size for b in chunk], device=self.device)

                    

                    # 动态长度解码（传入物理偏移用于全局时间坐标）

                    outputs = self.model.decode_batch(left_ids, right_ids, block_len, blk_offsets)

                    outputs = outputs.squeeze(1)  # [chunk_size, block_len]

                    

                    # 半块解码：用于 improvement_ratio（仅限尺度敏感解码器 siren）

                    if compute_imp:

                        out_left_h = self.model.decode_batch(left_ids, right_ids, half_len, blk_offsets)

                        out_left_h = out_left_h.squeeze(1)  # [chunk, half_len]

                        out_right_h = self.model.decode_batch(left_ids, right_ids, half_len, blk_offsets + half_len)

                        out_right_h = out_right_h.squeeze(1)  # [chunk, half_len]

                    

                    for i, block in enumerate(chunk):

                        start_time, end_time, left_id, right_id, level_code = block

                        output = outputs[i]

                        true = self.raw_data[start_time:end_time]

                        residual = true - output  # 带符号残差

                        error_curve = torch.abs(residual)  # 绝对误差

                        block_max_error = error_curve.max().item()

                        # 最大误差点局部索引（用于误差点吸附分裂）

                        max_err_idx = int(error_curve.argmax().item())

                        # 残差极差（位宽代价模型核心，尺度敏感解码器使用）

                        span = (residual.max() - residual.min()).item()

                        # 统计坏点数量（根据误差模式）+ 计算 per-block epsilon

                        if error_mode == 'relative':

                            true_orig = true * std_val + mean_val

                            error_orig = error_curve * std_val

                            rel_error = error_orig / torch.clamp(torch.abs(true_orig), min=1.0)

                            bad_point_count = (rel_error >= error_threshold).sum().item()

                            # relative 模式：epsilon = threshold × min(|true_orig|) / std

                            min_denom = torch.clamp(torch.abs(true_orig), min=1.0).min().item()

                            blk_epsilon = error_threshold * min_denom / std_val

                            # 最大相对误差（最大误差准则使用）

                            criterion_err = rel_error.max().item()

                        else:

                            bad_point_count = (error_curve >= norm_threshold).sum().item()

                            blk_epsilon = norm_threshold

                            # 最大绝对误差（原始单位，最大误差准则使用）

                            criterion_err = block_max_error * std_val

                        

                        # improvement_ratio = 半块MSE / 全块MSE（仅限尺度敏感解码器）

                        if compute_imp:

                            true_l = self.raw_data[start_time:start_time + half_len]

                            true_r = self.raw_data[start_time + half_len:start_time + half_len * 2]

                            full_mse = (residual ** 2).mean().item()

                            left_mse = ((out_left_h[i] - true_l) ** 2).mean().item()

                            right_mse = ((out_right_h[i] - true_r) ** 2).mean().item()

                            imp_ratio = (left_mse + right_mse) / (2.0 * full_mse + 1e-9)

                        else:

                            imp_ratio = 1.0

                        

                        block_errors[block] = (block_max_error, error_curve, bad_point_count, block_len, residual, span, blk_epsilon, imp_ratio, max_err_idx, criterion_err)

        

        # ================================================================

        # 分裂决策（两种模式）

        # ================================================================

        to_split = []

        

        if _use_max_error:

            # ------------------------------------------------------------

            # 最大误差准则（feature_strip / fourier 等尺度无关解码器）

            # 插值框架理论：分裂使最大误差可减半，直接以误差超阈值为触发条件

            # entry 格式: (block, criterion_err, max_err_idx)

            # ------------------------------------------------------------

            _cliff_select = getattr(self.args, 'split_cliff_select', False)

            if not _cliff_select:

                # 旧逻辑：纯最大误差选块（默认，保证复现）

                keep_ok = 0

                for block, (max_err, error_curve, n_bad, blk_len, residual, span, eps, imp_ratio, max_err_idx, criterion_err) in block_errors.items():

                    if criterion_err > error_threshold:

                        to_split.append((block, criterion_err, max_err_idx))

                    else:

                        keep_ok += 1

                # 按最大误差从大到小排序（优先处理误差最严重的块）

                to_split.sort(key=lambda x: x[1], reverse=True)

                print(f"    {keep_ok} blocks: criterion ≤ threshold (fine)")

                print(f"    {len(to_split)} blocks: max error > threshold, will split")

            else:

                # ────────────────────────────────────────────────────────

                # Bitwidth-cliff-aware 选块（--split_cliff_select）

                #

                # 动机：max_error 选块与 EDWB 字节台阶错位 —— 误差大 ≠ span 接近

                # 下一档位宽阈值。EDWB 残差字节是台阶函数：

                #   bits = ceil(log2(ceil(span / 2ε)))，只有 span 跨过 2ε·2^(b-1)

                #   才降 1 bit、字节才真降。本选块器只用「当前 base-model 残差」

                #   （中点切两半，不用分裂后 oracle）估计字节转化潜力：

                #     coding_only_gain = parent_cost - (left_cost + right_cost)

                #     base_net         = coding_only_gain - z_M_cost

                #     child_potential  = Σ one_bit_saving · exp(-γ · needed_drop_ratio)

                #     score            = base_net + child_potential

                #   仅保留 criterion_err>阈值 且 score>0 的块，按 score 降序分裂。

                # ────────────────────────────────────────────────────────

                _gamma = getattr(self.args, 'split_cliff_gamma', 8.0)

                vec_dim = self.grid_storage.trend_dim + self.grid_storage.context_dim

                _qb = getattr(self.args, 'quant_bits', 8)

                _bpd = (_qb / 8) if _qb > 0 else 4.0

                z_M_cost = int(vec_dim * _bpd)  # 新增 z_M 节点的字节成本

                keep_ok = 0          # criterion ≤ 阈值（精度已达标）

                err_ok_no_byte = 0   # 误差超阈值但 score ≤ 0（精度可提但字节救不回）

                for block, (max_err, error_curve, n_bad, blk_len, residual, span, eps, imp_ratio, max_err_idx, criterion_err) in block_errors.items():

                    if criterion_err <= error_threshold:

                        keep_ok += 1

                        continue

                    cliff = self._cliff_split_score(residual, blk_len, eps, z_M_cost, _gamma)

                    if cliff['score'] > 0:

                        to_split.append((block, cliff['score'], max_err_idx))

                    else:

                        err_ok_no_byte += 1

                # 按字节转化分数从高到低排序（临门一脚的块优先）

                to_split.sort(key=lambda x: x[1], reverse=True)

                print(f"    {keep_ok} blocks: criterion ≤ threshold (fine)")

                print(f"    {err_ok_no_byte} blocks: error > threshold but no positive byte saving (skipped, cliff-aware)")

                print(f"    {len(to_split)} blocks: error > threshold AND cliff-aware byte saving > 0, will split")

                if not to_split:

                    print(f"    [Cliff] No positive byte-saving split candidates found this round.")

        

        else:

            # ------------------------------------------------------------

            # 位宽代价模型（siren 等尺度敏感解码器）

            # 直接对比 keep vs split 的位宽代价，无需估计治愈率 R

            # entry 格式: (block, max_err, max_err_idx, savings)

            # ------------------------------------------------------------

            vec_dim = self.grid_storage.trend_dim + self.grid_storage.context_dim

            _quant_bits = getattr(self.args, 'quant_bits', 8)

            # bytes per stored z_M vector: quantised -> quant_bits/8;
            # unquantised (quant_bits == 0) -> float32 (4 B/dim).
            _bpd = (_quant_bits / 8) if _quant_bits > 0 else 4.0
            node_cost = int(vec_dim * _bpd)  # bytes per new z_M vector

            _num_groups = getattr(self.args, 'residual_groups', 1)

            keep_ok = 0

            zero_bit = 0

            marginal_filtered = 0  # 净收益为正但未过 min_split_savings 门槛的块数

            

            imp_threshold = getattr(self.args, 'improvement_threshold', 0.95)

            for block, (max_err, error_curve, n_bad, blk_len, residual, span, eps, imp_ratio, max_err_idx, criterion_err) in block_errors.items():

                if _num_groups > 1:
                    keep_cost = FallbackDict.estimate_bitwidth_cost_grouped(residual, eps, _num_groups)
                else:
                    keep_cost = FallbackDict.estimate_bitwidth_cost(span, blk_len, eps)

                keep_bits = FallbackDict.compute_bits(span, eps)

                if keep_bits == 0:

                    zero_bit += 1

                    continue

                if keep_bits < min_split_bits:

                    keep_ok += 1

                    continue

                if imp_ratio >= imp_threshold:

                    keep_ok += 1

                    continue

                mid_idx = blk_len // 2

                left_span  = (residual[:mid_idx].max() - residual[:mid_idx].min()).item()

                right_span = (residual[mid_idx:].max()  - residual[mid_idx:].min()).item()

                if _num_groups > 1:
                    bw_left  = FallbackDict.estimate_bitwidth_cost_grouped(residual[:mid_idx],  eps, _num_groups)
                    bw_right = FallbackDict.estimate_bitwidth_cost_grouped(residual[mid_idx:],  eps, _num_groups)
                else:
                    bw_left  = FallbackDict.estimate_bitwidth_cost(left_span,  mid_idx,           eps)
                    bw_right = FallbackDict.estimate_bitwidth_cost(right_span, blk_len - mid_idx, eps)

                split_cost = node_cost + effective_discount * (bw_left + bw_right)

                # 净收益 = 残差节省 - 分裂代价；要求超过 keep_cost 的 min_split_savings 比例
                # min_split_savings=0 → 退化为 split_cost < keep_cost（旧行为）
                savings = keep_cost - split_cost
                if savings > min_split_savings * keep_cost:

                    to_split.append((block, max_err, max_err_idx, savings))

                else:

                    keep_ok += 1
                    # 净收益为正但未过门槛 → 记为被 min_split_savings 挡下的边际分裂
                    if savings > 0 and min_split_savings > 0:
                        marginal_filtered += 1

            

            # 按节省字节数从大到小排序

            to_split.sort(key=lambda x: x[3], reverse=True)

            if zero_bit > 0:

                print(f"    {zero_bit} blocks: 0-bit (perfect)")

            if keep_ok > 0:

                print(f"    {keep_ok} blocks: keep (split_cost >= keep_cost)")

            if marginal_filtered > 0:

                print(f"    ({marginal_filtered} of them: positive savings but below "
                      f"min_split_savings={min_split_savings:.2f}×keep_cost gate)")

            print(f"    {len(to_split)} blocks: split cost-effective")

            

            # 探测分裂：首轮无历史时强制少量分裂（位宽模型保守偏置补偿）

            need_probe = (self._measured_r is None)

            if not to_split and need_probe:

                probe_candidates = []

                for block, (max_err, error_curve, n_bad, blk_len, residual, span, eps, imp_ratio, max_err_idx, criterion_err) in block_errors.items():

                    bits = FallbackDict.compute_bits(span, eps)

                    if bits > 0:

                        probe_candidates.append((block, max_err, max_err_idx, span))

                probe_candidates.sort(key=lambda x: x[3], reverse=True)

                n_probe = max(10, min(50, len(probe_candidates) // 200))

                to_split = [(b, me, mi, 0) for b, me, mi, sp in probe_candidates[:n_probe]]

                if to_split:

                    self._measured_r = 0.0

                    print(f"    [Probe] Force-splitting top {len(to_split)} worst-span blocks")

        

        # 限制最大分裂数（保存截断部分作为控制组）
        # 控制组：与被选中块质量最接近（同样超过阈值但因 max_splits 配额未被选中）
        control_candidates = to_split[max_splits:]  # 截断前先保存
        to_split = to_split[:max_splits]

        # 存储控制组基线 MAE（finetune 前）
        self._control_group_stats = {}
        for entry in control_candidates[:200]:  # 最多保存200个避免内存浪费
            block = entry[0]
            start_time, end_time, left_id, right_id, level_code = block
            max_err, error_curve, n_bad, blk_len, residual, span, eps, imp_ratio, max_err_idx, criterion_err = block_errors[block]
            self._control_group_stats[(start_time, end_time)] = {
                'mae':      error_curve.mean().item(),
                'left_id':  left_id,
                'right_id': right_id,
                'size':     blk_len,
            }

        if not to_split:

            max_err = max(v[0] for v in block_errors.values()) if block_errors else 0

            max_span = max(v[5] for v in block_errors.values()) if block_errors else 0

            print(f"    Max error {max_err:.6f}, max span {max_span:.6f}, no splits needed.")

            return 0

        

        # 清空上轮预测，准备记录本轮

        self._split_predictions = {}

        self._adaptive_discount = None  # 下轮重新从实测计算

        self._last_split_node_ids = set()  # 记录本轮新增的 z_mid 节点 ID

        

        # ── 分裂前统计：收集并打印待分裂块的误差信息 ──────────────────

        self._presplit_stats = {}

        pre_maes, pre_bad_pcts, pre_max_errs = [], [], []

        for entry in to_split:

            block = entry[0]

            start_time, end_time, left_id, right_id, level_code = block

            max_err, error_curve, n_bad, blk_len, residual, span, eps, imp_ratio, max_err_idx, criterion_err = block_errors[block]

            block_mae = error_curve.mean().item()

            bad_pct = n_bad / blk_len if blk_len > 0 else 0.0

            if error_mode == 'relative':

                _true_orig = self.raw_data[start_time:end_time] * std_val + mean_val

                _rel = error_curve * std_val / torch.clamp(torch.abs(_true_orig), min=1.0)

                max_rel_err  = _rel.max().item()

                mean_rel_err = _rel.mean().item()

            else:

                max_rel_err  = max_err * std_val

                mean_rel_err = block_mae * std_val

            self._presplit_stats[(start_time, end_time)] = {

                'mae': block_mae, 'bad_pct': bad_pct, 'max_err': max_err,

                'size': blk_len, 'left_id': left_id, 'right_id': right_id,

                'max_rel_err': max_rel_err, 'mean_rel_err': mean_rel_err,

                'level_code': level_code,

                'span': span, 'eps': blk_epsilon,

            }

            pre_maes.append(block_mae)

            pre_bad_pcts.append(bad_pct)

            pre_max_errs.append(max_err)

        print(f"\n    [Pre-split stats of {len(to_split)} blocks]")

        print(f"      MAE(norm): avg={np.mean(pre_maes):.5f}  max={np.max(pre_maes):.5f}")

        if std_val != 1.0:

            print(f"      MAE(orig): avg={np.mean(pre_maes)*std_val:.3f}  max={np.max(pre_maes)*std_val:.3f}  (data std={std_val:.2f})")

        print(f"      BadPct:    avg={np.mean(pre_bad_pcts)*100:.1f}%  max={np.max(pre_bad_pcts)*100:.1f}%")

        print(f"      MaxErr:    avg={np.mean(pre_max_errs):.5f}  max={np.max(pre_max_errs):.5f}")

        

        # 对每个超阈值的块各分裂一次

        split_count = 0

        min_res = self.manager.min_resolution

        for entry in to_split:

            block = entry[0]

            start_time, end_time, left_id, right_id, level_code = block

            try:

                # ── 中分（midpoint split，所有模式统一）──────────────

                mid_time   = (start_time + end_time) // 2

                split_time = (mid_time // min_res) * min_res

                

                # 边界保护：确保左右子块各至少 min_resolution 个点

                split_time = max(start_time + min_res, min(split_time, end_time - min_res))

                new_node_id = self.manager.split_block(split_time)

                split_count += 1

                self._last_split_node_ids.add(new_node_id)

                self._parentage_map[new_node_id] = (left_id, level_code)

                

                # Active-Dim 约束：缓存父节点的 frozen dims（训练时锁死这些维度）

                _svd = getattr(self.args, 'split_vec_dim', None)

                _fdim = self.model.grid_storage.feature_dim

                if _svd is not None and _svd < _fdim:

                    with torch.no_grad():

                        _z_par = self.model.grid_storage.get_vectors(

                            torch.tensor([left_id], device=self.device))[0]

                        _lid = new_node_id - self.model.grid_storage.num_base_nodes

                        self._frozen_patch_values[_lid] = _z_par[_svd:].detach().clone()

                

                # 前向分裂日志：O(1) 记账，为 Vector GC 构建隐式二叉树

                self.split_history[(start_time, end_time)] = (split_time, new_node_id)

                

                # 位宽代价模型：记录预测子块 BW 代价（下轮自适应折扣校准，仅限 siren 模式）

                if not _use_max_error:

                    _, _, _, blk_len, residual, span, eps, _, _, _ = block_errors[block]

                    split_idx  = split_time - start_time

                    left_span  = (residual[:split_idx].max() - residual[:split_idx].min()).item()

                    right_span = (residual[split_idx:].max()  - residual[split_idx:].min()).item()

                    pred_bw = (FallbackDict.estimate_bitwidth_cost(left_span,  split_idx,            eps) +

                               FallbackDict.estimate_bitwidth_cost(right_span, blk_len - split_idx,  eps))

                    self._split_predictions[(start_time, end_time)] = {

                        'predicted_child_bw': pred_bw, 'eps': eps

                    }

            except ValueError:

                continue

        

        total_blocks_after = len(self.manager.get_all_unique_blocks())

        print(f"[Adaptive Split] Completed: {split_count} splits, blocks: {total_blocks_before} -> {total_blocks_after} (+{total_blocks_after - total_blocks_before})")

        if split_count > 0:

            self._optimizer = None  # 结构变化：强制下次 train() 重建优化器以包含新 z_new

        return split_count

    

    def report_split_results(self):

        """

        比对分裂前后的重建质量，评估分裂是否有效。

        应在 adaptive_split + 微调之后调用。

        """

        if not self._presplit_stats:

            return

        

        scaler = self.data_loader.scaler

        if scaler is not None:

            std_val  = scaler.std.item()  if hasattr(scaler.std,  'item') else float(scaler.std)

            mean_val = scaler.mean.item() if hasattr(scaler.mean, 'item') else float(scaler.mean)

        else:

            std_val, mean_val = 1.0, 0.0

        error_mode = getattr(self.args, 'error_mode', 'relative')



        def _rel_errors(out, true_slice):

            """返回 (max_rel_err, mean_rel_err)"""

            abs_err  = torch.abs(out - true_slice)

            if error_mode == 'relative':

                true_orig = true_slice * std_val + mean_val

                rel = abs_err * std_val / torch.clamp(torch.abs(true_orig), min=1.0)

                return rel.max().item(), rel.mean().item()

            else:

                return (abs_err * std_val).max().item(), (abs_err * std_val).mean().item()



        results = []

        self.model.eval()

        with torch.no_grad():

            for (start, end), pre in self._presplit_stats.items():

                if (start, end) not in self.split_history:

                    continue

                split_t, new_node_id = self.split_history[(start, end)]

                left_id  = pre['left_id']

                right_id = pre['right_id']

                llen = split_t - start

                rlen = end - split_t

                if llen <= 0 or rlen <= 0:

                    continue

                try:

                    l_lids = torch.tensor([left_id],     device=self.device)

                    l_rids = torch.tensor([new_node_id], device=self.device)

                    l_off  = torch.tensor([start % self.args.base_block_size], device=self.device)

                    l_out  = self.model.decode_batch(l_lids, l_rids, llen, l_off).squeeze()

                    l_mae  = torch.abs(l_out - self.raw_data[start:split_t]).mean().item()

                    l_max_rel, l_mean_rel = _rel_errors(l_out, self.raw_data[start:split_t])



                    r_lids = torch.tensor([new_node_id], device=self.device)

                    r_rids = torch.tensor([right_id],    device=self.device)

                    r_off  = torch.tensor([split_t % self.args.base_block_size], device=self.device)

                    r_out  = self.model.decode_batch(r_lids, r_rids, rlen, r_off).squeeze()

                    r_mae  = torch.abs(r_out - self.raw_data[split_t:end]).mean().item()

                    r_max_rel, r_mean_rel = _rel_errors(r_out, self.raw_data[split_t:end])



                    child_mae      = (l_mae * llen + r_mae * rlen) / (llen + rlen)

                    child_max_rel  = max(l_max_rel, r_max_rel)

                    child_mean_rel = (l_mean_rel * llen + r_mean_rel * rlen) / (llen + rlen)

                    # ── Shadow z_L: evaluate parent quality with shadow-trained z_L ──
                    # Shadow was computed from snapshot + trained on parent data under
                    # current decoder — the fairest "no-split" baseline for comparison.
                    _shadow_zL = getattr(self, '_zL_shadows', {}).get((start, end))
                    if _shadow_zL is not None:
                        _gs = self.model.grid_storage
                        if left_id < _gs.num_base_nodes:
                            _orig_zL = _gs.base_grid.data[left_id].clone()
                            _gs.base_grid.data[left_id] = _shadow_zL
                            _is_base = True
                        else:
                            _llocal  = left_id - _gs.num_base_nodes
                            _orig_zL = _gs.patch_grid.data[_llocal].clone()
                            _gs.patch_grid.data[_llocal] = _shadow_zL
                            _is_base = False
                        _p_out = self.model.decode_batch(
                            torch.tensor([left_id],  device=self.device),
                            torch.tensor([right_id], device=self.device),
                            end - start,
                            torch.tensor([start % self.args.base_block_size], device=self.device)
                        ).squeeze()
                        _p_tgt = self.raw_data[start:end]
                        if _p_tgt.device != _p_out.device:
                            _p_tgt = _p_tgt.to(_p_out.device)
                        _p_res = _p_tgt - _p_out
                        _shadow_pspan = (_p_res.max() - _p_res.min()).item()
                        parent_mae    = torch.abs(_p_res).mean().item()
                        if _is_base:
                            _gs.base_grid.data[left_id] = _orig_zL
                        else:
                            _gs.patch_grid.data[_llocal] = _orig_zL
                        _pspan_override = _shadow_pspan
                    else:
                        parent_mae      = pre['mae']
                        _pspan_override = None

                    par_max_rel    = pre.get('max_rel_err',  None)

                    par_mean_rel   = pre.get('mean_rel_err', None)

                    improv_pct     = (parent_mae - child_mae) / (parent_mae + 1e-9) * 100

                    rel_improv_pct = ((par_max_rel - child_max_rel) / (par_max_rel + 1e-9) * 100

                                      if par_max_rel is not None else None)

                    # ── Storage cost analysis ──────────────────────────────────
                    _eps_par = pre.get('eps', 1e-4)
                    _pspan   = _pspan_override if _pspan_override is not None else pre.get('span', 0.0)
                    _qb = getattr(self.args, 'quant_bits', 8)
                    _bpd = (_qb / 8) if _qb > 0 else 4.0
                    _z_cost  = int(self.model.grid_storage.feature_dim * _bpd)  # bytes for z_M
                    _l_res   = self.raw_data[start:split_t] - l_out
                    _r_res   = self.raw_data[split_t:end]   - r_out
                    _l_span  = (_l_res.max() - _l_res.min()).item()
                    _r_span  = (_r_res.max() - _r_res.min()).item()
                    # Per-child epsilon: recompute from actual child signal range
                    # In relative mode each child may have a different min |signal|.
                    # In absolute mode epsilon is the same for all blocks.
                    _error_threshold = getattr(self.args, 'error_threshold', 0.1)
                    if error_mode == 'relative' and std_val > 0:
                        _l_true_orig = self.raw_data[start:split_t] * std_val + mean_val
                        _r_true_orig = self.raw_data[split_t:end]   * std_val + mean_val
                        _l_min_denom = torch.clamp(torch.abs(_l_true_orig), min=1.0).min().item()
                        _r_min_denom = torch.clamp(torch.abs(_r_true_orig), min=1.0).min().item()
                        _eps_l = _error_threshold * _l_min_denom / std_val
                        _eps_r = _error_threshold * _r_min_denom / std_val
                    else:
                        _eps_l = _eps_par   # absolute mode: eps same for all blocks
                        _eps_r = _eps_par
                    _par_cost = FallbackDict.estimate_bitwidth_cost(_pspan, end - start, _eps_par)
                    _l_cost   = FallbackDict.estimate_bitwidth_cost(_l_span, llen, _eps_l)
                    _r_cost   = FallbackDict.estimate_bitwidth_cost(_r_span, rlen, _eps_r)
                    _net_saving = _par_cost - (_z_cost + _l_cost + _r_cost)
                    # bits/point (theoretical) for span comparison
                    import math
                    _par_bits   = math.ceil(math.log2(_pspan / (_eps_par + 1e-12) + 1)) if _pspan > _eps_par else 0
                    _l_bits     = math.ceil(math.log2(_l_span / (_eps_l  + 1e-12) + 1)) if _l_span > _eps_l  else 0
                    _r_bits     = math.ceil(math.log2(_r_span / (_eps_r  + 1e-12) + 1)) if _r_span > _eps_r  else 0
                    _child_bits = (_l_bits * llen + _r_bits * rlen) / (end - start)

                    results.append({

                        'start': start, 'end': end, 'size': end - start,

                        'parent_mae': parent_mae,

                        'l_mae': l_mae, 'r_mae': r_mae, 'child_mae': child_mae,

                        'improv_pct': improv_pct,

                        'par_max_rel':   par_max_rel,

                        'par_mean_rel':  par_mean_rel,

                        'l_max_rel':     l_max_rel,   'r_max_rel':    r_max_rel,

                        'l_mean_rel':    l_mean_rel,  'r_mean_rel':   r_mean_rel,

                        'child_max_rel': child_max_rel, 'child_mean_rel': child_mean_rel,

                        'rel_improv_pct': rel_improv_pct,

                        'par_cost': _par_cost, 'l_cost': _l_cost, 'r_cost': _r_cost,
                        'z_M_cost': _z_cost,   'net_saving': _net_saving,
                        # span & bits tracking (key link: span → bitwidth → compression ratio)
                        'par_span': _pspan, 'l_span': _l_span, 'r_span': _r_span,
                        'par_bits': _par_bits, 'child_bits': _child_bits,

                    })

                except Exception:

                    continue

        

        if not results:

            print("[Split Results] No comparable blocks found.")

            return

        

        improved = sum(1 for r in results if r['improv_pct'] > 5.0)

        worsened = sum(1 for r in results if r['improv_pct'] < -5.0)

        avg_improv       = np.mean([r['improv_pct']    for r in results])

        avg_parent_mae   = np.mean([r['parent_mae']    for r in results])

        avg_child_mae    = np.mean([r['child_mae']     for r in results])

        # span & bits: length-weighted averages (fair comparison across block sizes)
        total_pts        = sum(r['size'] for r in results)
        avg_par_span     = sum(r['par_span'] * r['size'] for r in results) / (total_pts + 1e-9)
        # child span: length-weighted average of (l_span*llen + r_span*rlen) across all split blocks
        avg_child_span   = sum(r['l_span']*(r['size']//2) + r['r_span']*(r['size'] - r['size']//2)
                               for r in results) / (total_pts + 1e-9)
        avg_par_bits     = sum(r['par_bits']   * r['size'] for r in results) / (total_pts + 1e-9)
        avg_child_bits   = sum(r['child_bits'] * r['size'] for r in results) / (total_pts + 1e-9)
        span_improv_pct  = (avg_par_span - avg_child_span) / (avg_par_span + 1e-9) * 100

        has_rel = results[0]['par_max_rel'] is not None

        if has_rel:

            avg_par_max_rel   = np.mean([r['par_max_rel']   for r in results])

            avg_child_max_rel = np.mean([r['child_max_rel'] for r in results])

            avg_par_mean_rel  = np.mean([r['par_mean_rel']  for r in results])

            avg_child_mean_rel= np.mean([r['child_mean_rel']for r in results])

            rel_improved = sum(1 for r in results if (r['rel_improv_pct'] or 0) > 5.0)

            rel_worsened = sum(1 for r in results if (r['rel_improv_pct'] or 0) < -5.0)



        W = 90

        print("\n" + "=" * W)

        print("SPLIT QUALITY REPORT  (pre-split → post-finetune)")

        print("=" * W)

        print(f"  Blocks compared  : {len(results)}")

        print(f"  MAE improved(>5%): {improved}  |  worsened: {worsened}")

        print(f"  Avg parent MAE   : {avg_parent_mae:.5f}  →  child: {avg_child_mae:.5f}  (norm)"

              + (f"  =  {avg_parent_mae*std_val:.3f} → {avg_child_mae*std_val:.3f} (orig)" if std_val != 1.0 else ""))

        print(f"  Avg MAE improv   : {avg_improv:+.1f}%")
        print(f"")
        print(f"  ── Span & Bits (compression ratio drivers) ──")
        print(f"  Avg residual span: {avg_par_span:.5f}  →  {avg_child_span:.5f}  ({span_improv_pct:+.1f}%)")
        print(f"  Avg bits/point   : {avg_par_bits:.2f}  →  {avg_child_bits:.2f}  "
              f"(Δ {avg_child_bits - avg_par_bits:+.2f} bits/pt)")
        _bits_z_overhead = sum(r['z_M_cost'] * 8 / r['size'] for r in results) / len(results)
        print(f"  z_M bits overhead: +{_bits_z_overhead:.2f} bits/pt (amortized per parent block)")
        print(f"  Net bits change  : {avg_child_bits + _bits_z_overhead - avg_par_bits:+.2f} bits/pt  "
              f"({'saves' if avg_child_bits + _bits_z_overhead < avg_par_bits else 'costs'} storage)")

        if has_rel:

            lbl = "RelErr" if error_mode == 'relative' else "AbsErr(orig)"

            print(f"  {lbl} improved(>5%): {rel_improved}  |  worsened: {rel_worsened}")

            if error_mode == 'relative':
                print(f"  Avg par MaxRel   : {avg_par_max_rel*100:.2f}%  →  child: {avg_child_max_rel*100:.2f}%  "
                      f"(Δ {(avg_child_max_rel-avg_par_max_rel)*100:+.2f}pp)")
                print(f"  Avg par MeanRel  : {avg_par_mean_rel*100:.2f}%  →  child: {avg_child_mean_rel*100:.2f}%  "
                      f"(Δ {(avg_child_mean_rel-avg_par_mean_rel)*100:+.2f}pp)")
            else:
                print(f"  Avg par MaxErr   : {avg_par_max_rel:.4f}  →  child: {avg_child_max_rel:.4f}  "
                      f"(Δ {avg_child_max_rel-avg_par_max_rel:+.4f}, orig units)")
                print(f"  Avg par MeanErr  : {avg_par_mean_rel:.4f}  →  child: {avg_child_mean_rel:.4f}  "
                      f"(Δ {avg_child_mean_rel-avg_par_mean_rel:+.4f}, orig units)")



        # 全部分裂块按最大相对误差改善幅度排序打印

        sort_key = 'rel_improv_pct' if has_rel else 'improv_pct'

        all_sorted = sorted(results, key=lambda r: (r[sort_key] or 0), reverse=True)

        if has_rel:
            if error_mode == 'relative':
                mx_lbl, mn_lbl = 'parMxRel', 'parMnRel'
                lx_lbl, ln_lbl = 'lMxRel ', 'lMnRel '
                rx_lbl, rn_lbl = 'rMxRel ', 'rMnRel '
            else:
                mx_lbl, mn_lbl = 'parMxErr', 'parMnErr'
                lx_lbl, ln_lbl = 'lMxErr  ', 'lMnErr  '
                rx_lbl, rn_lbl = 'rMxErr  ', 'rMnErr  '
            hdr = (f"  {'[start,end]':>13} {'sz':>5} | "
                   f"{'parMAE':>7} {'lMAE':>7} {'rMAE':>7} | "
                   f"{mx_lbl:>9} {lx_lbl:>8} {rx_lbl:>8} | "
                   f"{mn_lbl:>9} {ln_lbl:>8} {rn_lbl:>8} | "
                   f"{'MAEimp':>7} {'MxImp':>7}")
        else:
            hdr = (f"  {'[start,end]':>13} {'sz':>5} | "
                   f"{'parent':>8} {'left':>8} {'right':>8} | {'improv':>7}")
        print(f"\n  ── All {len(all_sorted)} split blocks (sorted by MaxErr improvement):")
        print(hdr)
        for r in all_sorted:
            flag = " ✗" if (r[sort_key] or 0) < -5.0 else (" ✓" if (r[sort_key] or 0) > 5.0 else "")
            if has_rel:
                if error_mode == 'relative':
                    mx_fmt = (f"{r['par_max_rel']*100:8.2f}% {r['l_max_rel']*100:7.2f}% {r['r_max_rel']*100:7.2f}% | "
                              f"{r['par_mean_rel']*100:8.2f}% {r['l_mean_rel']*100:7.2f}% {r['r_mean_rel']*100:7.2f}%")
                else:
                    mx_fmt = (f"{r['par_max_rel']:9.4f}  {r['l_max_rel']:8.4f}  {r['r_max_rel']:8.4f}  | "
                              f"{r['par_mean_rel']:9.4f}  {r['l_mean_rel']:8.4f}  {r['r_mean_rel']:8.4f} ")
                print(f"  [{r['start']:5d},{r['end']:5d}] {r['size']:5d} | "
                      f"{r['parent_mae']:.5f} {r['l_mae']:.5f} {r['r_mae']:.5f} | "
                      f"{mx_fmt} | "
                      f"{r['improv_pct']:+6.1f}% {(r['rel_improv_pct'] or 0):+6.1f}%{flag}")

            else:

                print(f"  [{r['start']:5d},{r['end']:5d}] {r['size']:5d} | "

                      f"{r['parent_mae']:.5f} {r['l_mae']:.5f} {r['r_mae']:.5f} | "

                      f"{r['improv_pct']:+.1f}%{flag}")

        print("=" * W)

        # ── 控制组对比：真实分裂净收益 ──────────────────────────────────────────
        ctrl = getattr(self, '_control_group_stats', {})
        if ctrl:
            ctrl_rows = []
            with torch.no_grad():
                for (cs, ce), cpre in ctrl.items():
                    try:
                        c_out = self.model.decode_batch(
                            torch.tensor([cpre['left_id']],  device=self.device),
                            torch.tensor([cpre['right_id']], device=self.device),
                            ce - cs,
                            torch.tensor([cs % self.args.base_block_size], device=self.device)
                        ).squeeze()
                        c_mae = torch.abs(c_out - self.raw_data[cs:ce]).mean().item()
                        ctrl_rows.append({'pre': cpre['mae'], 'post': c_mae,
                                          'improv': (cpre['mae'] - c_mae) / (cpre['mae'] + 1e-9) * 100})
                    except Exception:
                        continue
            if ctrl_rows and results:
                ctrl_improv  = float(np.mean([r['improv']     for r in ctrl_rows]))
                split_improv = float(np.mean([r['improv_pct'] for r in results]))
                net_benefit  = split_improv - ctrl_improv
                ctrl_pre  = float(np.mean([r['pre']  for r in ctrl_rows]))
                ctrl_post = float(np.mean([r['post'] for r in ctrl_rows]))
                print(f"\n  ── Control Group vs Split Group (net split benefit) ──")
                print(f"  {'Group':<14} {'n':>4}  {'pre-MAE':>8}  {'post-MAE':>9}  {'improv':>8}")
                print(f"  {'Split blocks':<14} {len(results):>4}  "
                      f"{np.mean([r['parent_mae'] for r in results]):>8.5f}  "
                      f"{np.mean([(r['l_mae']+r['r_mae'])/2 for r in results]):>9.5f}  "
                      f"{split_improv:>+7.1f}%")
                print(f"  {'Control (noSplit)':<14} {len(ctrl_rows):>4}  "
                      f"{ctrl_pre:>8.5f}  {ctrl_post:>9.5f}  {ctrl_improv:>+7.1f}%")
                print(f"  {'─'*60}")
                marker = '✓' if net_benefit > 5.0 else ('△' if net_benefit > 0 else '✗')
                print(f"  Net split benefit = {split_improv:+.1f}% - {ctrl_improv:+.1f}% = {net_benefit:+.1f}%  {marker}")
                if net_benefit <= 0:
                    print(f"  [Warn] Split benefit ≤ 0: training alone explains all improvement this round.")
                elif net_benefit < 5.0:
                    print(f"  [Note] Marginal split benefit (<5%): consider raising threshold or fewer splits.")

                # ── Per-block rollback (storage-cost criterion) ───────────────
                # net_saving = parent_cost - (z_M_cost + l_child_cost + r_child_cost)
                # Keep split only if net_saving >= min_saving (default 0 bytes)
                min_saving = getattr(self.args, 'rollback_min_saving', 0)

                # Cost summary across all splits
                total_par_cost  = sum(r.get('par_cost',  0) for r in results)
                total_chld_cost = sum(r.get('l_cost', 0) + r.get('r_cost', 0) + r.get('z_M_cost', 0) for r in results)
                total_net       = total_par_cost - total_chld_cost
                print(f"\n  ── Storage cost summary ──")
                print(f"  Parent residual cost (pre-split):       {total_par_cost:>8} B")
                print(f"  Children + z_M cost  (post-finetune):   {total_chld_cost:>8} B")
                print(f"  Total net saving:                       {total_net:>+8} B  "
                      f"({'✓ saves space' if total_net > 0 else '✗ wastes space'})")

                enable_rollback = getattr(self.args, 'enable_rollback', False)
                if not enable_rollback:
                    print(f"  [Rollback] Disabled (enable_rollback=False). All {len(results)} splits kept.")
                else:
                    rollback_cands = [(r['start'], r['end'],
                                       r['improv_pct'], r.get('net_saving', 0),
                                       r.get('par_cost', 0),
                                       r.get('l_cost', 0) + r.get('r_cost', 0) + r.get('z_M_cost', 0))
                                      for r in results if r.get('net_saving', 0) < min_saving]
                    if rollback_cands:
                        print(f"\n  ── Per-block rollback "
                              f"({len(rollback_cands)} splits with net_saving < {min_saving}B) ──")
                        print(f"  {'[start,end]':>14}  {'improv':>7}  {'par_B':>6}  {'chld_B':>7}  {'save_B':>7}  status")
                        rolled = 0
                        for s, e, imp, nsav, pcost, ccost in sorted(rollback_cands, key=lambda x: x[3]):
                            ok  = self._rollback_split(s, e)
                            tag = "✓ rolled" if ok else "✗ failed"
                            print(f"  [{s:6d},{e:6d}]  {imp:>+6.1f}%  {pcost:>6}  {ccost:>7}  {nsav:>+7}  {tag}")
                            if ok:
                                rolled += 1
                        print(f"  Rolled back {rolled}/{len(rollback_cands)} splits  "
                              f"(remaining active splits: {len(self.split_history)})")
                    else:
                        print(f"  All {len(results)} splits kept (all net_saving ≥ {min_saving}B)")
                print("=" * W)
        else:
            print(f"\n  [Note] No control group available this round "
                  f"(all candidates were selected for splitting).")



    def report_split_byte_conversion(self):

        """

        Split 字节转化报告（--split_byte_gate）。

        

        分裂 + 微调之后调用。对本轮每个 split 块，用 EDWB 真实公式重算字节：

            old: 不分裂时父块残差代价（优先 shadow z_L 作为公平的 no-split 基线）

            new: z_M 节点 + 左右子块残差代价

            netGain = old - new

        ACCEPT(netGain>0) / REJECT(netGain<=0)。同时保留每块的 MAE/MaxErr 提升，

        额外给出"精度提升但字节未省"的块。仅报告，不回滚（与 enable_rollback 独立）。

        

        全局 raw vs rate-gated 双账本累加到 self._split_byte_stats，供最终 summary 使用。

        """

        if not self._presplit_stats:

            return

        if not getattr(self.args, 'split_byte_gate', False):

            return

        

        scaler = self.data_loader.scaler

        if scaler is not None:

            std_val  = scaler.std.item()  if hasattr(scaler.std,  'item') else float(scaler.std)

            mean_val = scaler.mean.item() if hasattr(scaler.mean, 'item') else float(scaler.mean)

        else:

            std_val, mean_val = 1.0, 0.0

        error_mode = getattr(self.args, 'error_mode', 'relative')

        _error_threshold = getattr(self.args, 'split_threshold', 0.1)

        

        vec_dim = self.grid_storage.trend_dim + self.grid_storage.context_dim

        _qb = getattr(self.args, 'quant_bits', 8)

        _bpd = (_qb / 8) if _qb > 0 else 4.0

        z_M_cost = int(vec_dim * _bpd)

        

        def _block_eps(t0, t1):

            """该子块的 EDWB epsilon（与 final_evaluation 口径一致）。"""

            if error_mode == 'absolute':

                return _error_threshold / std_val if std_val > 0 else _error_threshold

            true_orig = self.raw_data[t0:t1] * std_val + mean_val

            min_denom = torch.clamp(torch.abs(true_orig), min=1.0).min().item()

            return _error_threshold * min_denom / std_val if std_val > 0 else _error_threshold

        

        def _max_rel(out, t0, t1):

            true = self.raw_data[t0:t1]

            abs_err = torch.abs(out - true)

            if error_mode == 'relative':

                true_orig = true * std_val + mean_val

                return (abs_err * std_val / torch.clamp(torch.abs(true_orig), min=1.0)).max().item()

            return (abs_err * std_val).max().item()

        

        rows = []

        self.model.eval()

        with torch.no_grad():

            for (start, end), pre in self._presplit_stats.items():

                if (start, end) not in self.split_history:

                    continue

                split_t, new_node_id = self.split_history[(start, end)]

                left_id, right_id = pre['left_id'], pre['right_id']

                llen, rlen = split_t - start, end - split_t

                if llen <= 0 or rlen <= 0:

                    continue

                

                # ── 新成本：左右子块（当前模型，分裂后）──

                l_out = self.model.decode_batch(

                    torch.tensor([left_id], device=self.device),

                    torch.tensor([new_node_id], device=self.device),

                    llen,

                    torch.tensor([start % self.args.base_block_size], device=self.device)).squeeze()

                r_out = self.model.decode_batch(

                    torch.tensor([new_node_id], device=self.device),

                    torch.tensor([right_id], device=self.device),

                    rlen,

                    torch.tensor([split_t % self.args.base_block_size], device=self.device)).squeeze()

                l_res = self.raw_data[start:split_t] - l_out

                r_res = self.raw_data[split_t:end] - r_out

                l_span = (l_res.max() - l_res.min()).item()

                r_span = (r_res.max() - r_res.min()).item()

                eps_l, eps_r = _block_eps(start, split_t), _block_eps(split_t, end)

                new_l_bytes = FallbackDict.estimate_bitwidth_cost(l_span, llen, eps_l)

                new_r_bytes = FallbackDict.estimate_bitwidth_cost(r_span, rlen, eps_r)

                l_bits = FallbackDict.compute_bits(l_span, eps_l)

                r_bits = FallbackDict.compute_bits(r_span, eps_r)

                

                # ── 旧成本：不分裂的父块（优先 shadow z_L 公平基线）──

                eps_par = _block_eps(start, end)

                _shadow_zL = getattr(self, '_zL_shadows', {}).get((start, end))

                gs = self.model.grid_storage

                if _shadow_zL is not None:

                    if left_id < gs.num_base_nodes:

                        _orig = gs.base_grid.data[left_id].clone()

                        gs.base_grid.data[left_id] = _shadow_zL

                        _is_base = True

                    else:

                        _ll = left_id - gs.num_base_nodes

                        _orig = gs.patch_grid.data[_ll].clone()

                        gs.patch_grid.data[_ll] = _shadow_zL

                        _is_base = False

                    p_out = self.model.decode_batch(

                        torch.tensor([left_id], device=self.device),

                        torch.tensor([right_id], device=self.device),

                        end - start,

                        torch.tensor([start % self.args.base_block_size], device=self.device)).squeeze()

                    p_res = self.raw_data[start:end] - p_out

                    par_span = (p_res.max() - p_res.min()).item()

                    par_mae = torch.abs(p_res).mean().item()

                    par_max_rel = _max_rel(p_out, start, end)

                    if _is_base:

                        gs.base_grid.data[left_id] = _orig

                    else:

                        gs.patch_grid.data[_ll] = _orig

                else:

                    par_span = pre.get('span', 0.0)

                    par_mae = pre.get('mae', 0.0)

                    par_max_rel = pre.get('max_rel_err', 0.0)

                par_bits = FallbackDict.compute_bits(par_span, eps_par)

                old_parent_bytes = FallbackDict.estimate_bitwidth_cost(par_span, end - start, eps_par)

                

                # ── 字节账 ──

                old_total = old_parent_bytes                       # 不分裂：仅父残差（z_L 共享，不计）

                new_total = z_M_cost + new_l_bytes + new_r_bytes    # 分裂：+1 z_M 节点 + 左右残差

                net_gain = old_total - new_total

                accept = net_gain > 0

                

                # ── 精度提升（保留每块提升情况）──

                child_mae = (torch.abs(l_res).mean().item() * llen +

                             torch.abs(r_res).mean().item() * rlen) / (llen + rlen)

                child_max_rel = max(_max_rel(l_out, start, split_t),

                                    _max_rel(r_out, split_t, end))

                mae_imp = (par_mae - child_mae) / (par_mae + 1e-9) * 100

                mxr_imp = (par_max_rel - child_max_rel) / (par_max_rel + 1e-9) * 100

                

                rows.append({

                    'start': start, 'end': end, 'plen': end - start, 'clen': llen,

                    'par_bits': par_bits, 'l_bits': l_bits, 'r_bits': r_bits,

                    'old_res_bytes': old_parent_bytes,

                    'new_res_bytes': new_l_bytes + new_r_bytes,

                    'z_M_cost': z_M_cost,

                    'net_gain': net_gain,

                    'par_mae': par_mae, 'child_mae': child_mae,

                    'par_max_rel': par_max_rel, 'child_max_rel': child_max_rel,

                    'mae_imp': mae_imp, 'mxr_imp': mxr_imp,

                    'accept': accept,

                })

        

        if not rows:

            print("[Split Byte Conversion] No comparable split blocks.")

            return

        

        # ── per-split 表（保留提升 + 新增字节）──

        W = 118

        print("\n" + "=" * W)

        print("SPLIT BYTE-CONVERSION REPORT  (real EDWB bytes; report-only, no rollback)")

        print("=" * W)

        hdr = (f"  {'[start,end]':>15} {'plen':>5} | {'bits p->L/R':>12} | "

               f"{'resB old->new':>14} | {'zM':>4} | {'netGain':>8} | "

               f"{'MAEimp':>7} {'MxImp':>7} | status")

        print(hdr)

        print("  " + "-" * (W - 2))

        for r in sorted(rows, key=lambda x: x['net_gain'], reverse=True):

            status = "ACCEPT" if r['accept'] else "REJECT"

            flag = ""

            if (not r['accept']) and r['mae_imp'] > 5.0:

                flag = "  <- acc+ but no bytes"

            print(f"  [{r['start']:7d},{r['end']:7d}] {r['plen']:5d} | "

                  f"{r['par_bits']:>2}->{r['l_bits']:>2}/{r['r_bits']:<2}    | "

                  f"{r['old_res_bytes']:>5}->{r['new_res_bytes']:<5}    | "

                  f"{r['z_M_cost']:>4} | {r['net_gain']:>+8} | "

                  f"{r['mae_imp']:>+6.1f}% {r['mxr_imp']:>+6.1f}% | {status}{flag}")

        

        # ── 本轮 raw vs rate-gated 双账本 ──

        n_total = len(rows)

        n_accept = sum(1 for r in rows if r['accept'])

        n_reject = n_total - n_accept

        raw_extra = sum(r['z_M_cost'] for r in rows)

        raw_res_saved = sum(r['old_res_bytes'] - r['new_res_bytes'] for r in rows)

        raw_net = sum(r['net_gain'] for r in rows)

        gated_extra = sum(r['z_M_cost'] for r in rows if r['accept'])

        gated_res_saved = sum(r['old_res_bytes'] - r['new_res_bytes'] for r in rows if r['accept'])

        gated_net = sum(r['net_gain'] for r in rows if r['accept'])

        bits_drop = sum(1 for r in rows if min(r['l_bits'], r['r_bits']) < r['par_bits'])

        res_drop = sum(1 for r in rows if r['new_res_bytes'] < r['old_res_bytes'])

        mae_imp_but_reject = sum(1 for r in rows if r['mae_imp'] > 5.0 and not r['accept'])

        

        print("  " + "-" * (W - 2))

        print(f"  This round: {n_total} splits | ACCEPT {n_accept} | REJECT {n_reject}")

        print(f"    [raw]        z_M extra {raw_extra:+d}B | residual saved {raw_res_saved:+d}B | net {raw_net:+d}B")

        print(f"    [rate-gated] z_M extra {gated_extra:+d}B | residual saved {gated_res_saved:+d}B | net {gated_net:+d}B")

        print(f"    bits-dropped blocks: {bits_drop}/{n_total} | residual-bytes-dropped: {res_drop}/{n_total}")

        print(f"    MAE improved(>5%) but REJECTED (no net bytes): {mae_imp_but_reject}")

        print("=" * W)

        

        # ── 全局累加（最终 summary 用）──

        if not hasattr(self, '_split_byte_stats') or self._split_byte_stats is None:

            self._split_byte_stats = {

                'total': 0, 'accept': 0, 'reject': 0,

                'raw_extra': 0, 'raw_res_saved': 0, 'raw_net': 0,

                'gated_extra': 0, 'gated_res_saved': 0, 'gated_net': 0,

                'bits_drop': 0, 'res_drop': 0, 'mae_imp_but_reject': 0,

            }

        S = self._split_byte_stats

        S['total'] += n_total;       S['accept'] += n_accept;  S['reject'] += n_reject

        S['raw_extra'] += raw_extra; S['raw_res_saved'] += raw_res_saved; S['raw_net'] += raw_net

        S['gated_extra'] += gated_extra; S['gated_res_saved'] += gated_res_saved; S['gated_net'] += gated_net

        S['bits_drop'] += bits_drop; S['res_drop'] += res_drop

        S['mae_imp_but_reject'] += mae_imp_but_reject

    

    def report_split_byte_summary(self):

        """

        全局 split 字节转化 summary（训练全部结束后调用，--split_byte_gate）。

        汇总各轮 report_split_byte_conversion 的 raw vs rate-gated 双账本。

        """

        if not getattr(self.args, 'split_byte_gate', False):

            return

        S = getattr(self, '_split_byte_stats', None)

        if not S or S['total'] == 0:

            print("\n[Split Byte Summary] No splits with byte accounting recorded.")

            return

        W = 70

        print("\n" + "=" * W)

        print("GLOBAL SPLIT BYTE-CONVERSION SUMMARY")

        print("=" * W)

        print(f"  split_cliff_select : {getattr(self.args, 'split_cliff_select', False)}")

        print(f"  split_byte_gate    : {getattr(self.args, 'split_byte_gate', False)}")

        print(f"  total split blocks : {S['total']}")

        print(f"    accepted (netGain>0): {S['accept']}")

        print(f"    rejected (netGain<=0): {S['reject']}")

        print(f"  bitwidth dropped   : {S['bits_drop']}/{S['total']} blocks (>=1 child bits < parent)")

        print(f"  residual-bytes drop: {S['res_drop']}/{S['total']} blocks")

        print(f"  MAE improved but REJECTED (no net bytes): {S['mae_imp_but_reject']}")

        print(f"\n  [Split raw]        (all trained splits counted)")

        print(f"    z_M extra bytes   : {S['raw_extra']:+d} B")

        print(f"    residual saved    : {S['raw_res_saved']:+d} B")

        print(f"    net gain          : {S['raw_net']:+d} B  "

              f"({'saves' if S['raw_net'] > 0 else 'wastes'} space)")

        print(f"\n  [Split rate-gated] (only netGain>0 splits counted)")

        print(f"    z_M extra bytes   : {S['gated_extra']:+d} B")

        print(f"    residual saved    : {S['gated_res_saved']:+d} B")

        print(f"    net gain          : {S['gated_net']:+d} B  "

              f"({'saves' if S['gated_net'] > 0 else 'wastes'} space)")

        print("=" * W)



    def patch_K_sweep(self, error_threshold: float = 0.05, error_mode: str = 'relative',

                      K_candidates=(0, 4, 8, 16), modes=('int8', 'fp16'),

                      max_blocks: int = None):

        """

        定长 K 选择调研：对比"变长(自由选 K)" vs "固定 K∈K_candidates"的总字节。

        

        目的：定长 patch 方案要固定 K（保住批处理）。本统计告诉你固定到哪个 K

        字节损失最小。纯计算、不训练。每个候选块中点切两半，左右各：

          - variable: fit_child_best（自由选 K）

          - fixed-K : fit_child_fixedK（强制该 K）

        汇总 ACCEPT 块上各方案的残差+系数总字节。

        """

        self.model.eval()

        self._refresh_quant_params()

        scaler = self.data_loader.scaler

        if scaler is not None:

            std_val = scaler.std.item() if hasattr(scaler.std, 'item') else float(scaler.std)

            mean_val = scaler.mean.item() if hasattr(scaler.mean, 'item') else float(scaler.mean)

        else:

            std_val, mean_val = 1.0, 0.0

        norm_threshold = error_threshold / std_val if error_mode == 'absolute' else None

        

        def _block_eps(t0, t1):

            if error_mode == 'absolute':

                return norm_threshold

            true_orig = self.raw_data[t0:t1] * std_val + mean_val

            min_denom = torch.clamp(torch.abs(true_orig), min=1.0).min().item()

            return error_threshold * min_denom / std_val if std_val > 0 else error_threshold

        

        all_blocks = self.manager.get_all_unique_blocks()

        min_res = self.manager.min_resolution

        candidates = [b for b in all_blocks

                      if (b[1] - b[0]) >= 2 * min_res and b[4] < self.manager.max_level]

        if max_blocks is not None:

            candidates = candidates[:max_blocks]

        

        # 累加器：variable + 每个固定 K

        var_bytes = 0

        var_K_hist = {}

        fixed_bytes = {K: 0 for K in K_candidates}

        n_seg = 0

        

        with torch.no_grad():

            for (start_time, end_time, left_id, right_id, level_code) in candidates:

                blk_len = end_time - start_time

                mid = start_time + (blk_len // 2 // min_res) * min_res

                if mid <= start_time or mid >= end_time:

                    mid = start_time + blk_len // 2

                parent_pred = self.model.decode_single(

                    left_id, right_id, blk_len,

                    offset=start_time % self.args.base_block_size).squeeze()

                true_full = self.raw_data[start_time:end_time]

                if true_full.device != parent_pred.device:

                    true_full = true_full.to(parent_pred.device)

                

                for (s0, s1) in [(start_time, mid), (mid, end_time)]:

                    l0, l1 = s0 - start_time, s1 - start_time

                    target = true_full[l0:l1] - parent_pred[l0:l1]

                    eps = _block_eps(s0, s1)

                    # variable

                    vf = fit_child_best(target, eps, K_list=K_candidates, modes=modes)

                    var_bytes += vf.total_bytes

                    var_K_hist[vf.K] = var_K_hist.get(vf.K, 0) + 1

                    # fixed-K

                    for K in K_candidates:

                        ff = fit_child_fixedK(target, eps, K, modes=modes)

                        fixed_bytes[K] += ff.total_bytes

                    n_seg += 1

        

        if n_seg == 0:

            print("[Patch K-Sweep] No candidate segments.")

            return {}

        

        W = 64

        print("\n" + "=" * W)

        print("PATCH K-SWEEP  (variable vs fixed-K, total bytes on segments)")

        print("=" * W)

        print(f"  Segments evaluated : {n_seg}")

        print(f"  Variable-K (free)  : {var_bytes} B  (optimal, baseline)")

        print(f"  Variable-K chosen distribution:")

        for K in sorted(var_K_hist.keys()):

            print(f"      K={K:>2}: {var_K_hist[K]} segs ({100*var_K_hist[K]/n_seg:.1f}%)")

        print(f"  Fixed-K total bytes (loss vs variable):")

        best_fixed_K = None

        best_fixed_val = None

        for K in K_candidates:

            loss = fixed_bytes[K] - var_bytes

            loss_pct = 100 * loss / max(var_bytes, 1)

            print(f"      K={K:>2}: {fixed_bytes[K]:>8} B  (+{loss} B, +{loss_pct:.1f}%)")

            if best_fixed_val is None or fixed_bytes[K] < best_fixed_val:

                best_fixed_val = fixed_bytes[K]

                best_fixed_K = K

        print(f"  → Best fixed K = {best_fixed_K} "

              f"(loss {best_fixed_val - var_bytes:+d} B, "

              f"{100*(best_fixed_val-var_bytes)/max(var_bytes,1):.1f}% over variable)")

        print(f"  NOTE: fixed-K enables batched GEMM (regular patch pool) at small byte cost.")

        print("=" * W)

        

        return {

            'n_seg': n_seg, 'var_bytes': var_bytes, 'var_K_hist': var_K_hist,

            'fixed_bytes': fixed_bytes, 'best_fixed_K': best_fixed_K,

        }



    def patch_split_eval(self, error_threshold: float = 0.05, error_mode: str = 'relative',

                         K_list=(0, 4, 8, 16), modes=('int8', 'fp16'),

                         max_blocks: int = None, commit: bool = False):

        """

        Parent-Anchored Additive Patch Split —— 字节核验（report-only）+ 可选提交接入。

        

        对当前冻结模型的每个候选块评估「parent 预测 + 加性 patch」方案：

        不训练、不改 decoder / grid / index / 旧 token，因此未分裂块逐比特不变

        （unsplit delta 恒为 0）。每个 child 在冻结 parent 预测上闭式拟合 patch 系数，

        量化后用真实 EDWB 公式核算字节，exact byte gate 决定接受/拒绝。

        

        全部复用 FallbackDict 公式，与最终落盘核算一致。

        

        Args:

            error_threshold: 误差阈值（relative=比例，absolute=原始单位）

            error_mode: 'relative' | 'absolute'

            K_list: 枚举的 DST 频率数

            modes: 枚举的量化模式

            max_blocks: 仅评估前 N 个块（调试用，None=全部）

            commit: True 时把所有 ACCEPT 的 patch 提交到 self.patch_manager（接入）

        Returns:

            统计 dict

        """

        self.model.eval()

        self._refresh_quant_params()

        

        scaler = self.data_loader.scaler

        if scaler is not None:

            std_val = scaler.std.item() if hasattr(scaler.std, 'item') else float(scaler.std)

            mean_val = scaler.mean.item() if hasattr(scaler.mean, 'item') else float(scaler.mean)

        else:

            std_val, mean_val = 1.0, 0.0

        norm_threshold = error_threshold / std_val if error_mode == 'absolute' else None

        

        def _block_eps(t0, t1):

            if error_mode == 'absolute':

                return norm_threshold

            true_orig = self.raw_data[t0:t1] * std_val + mean_val

            min_denom = torch.clamp(torch.abs(true_orig), min=1.0).min().item()

            return error_threshold * min_denom / std_val if std_val > 0 else error_threshold

        

        all_blocks = self.manager.get_all_unique_blocks()

        min_res = self.manager.min_resolution

        

        # 仅评估可分裂块（长度 >= 2*min_res 且未达最大层级）

        candidates = [b for b in all_blocks

                      if (b[1] - b[0]) >= 2 * min_res and b[4] < self.manager.max_level]

        if max_blocks is not None:

            candidates = candidates[:max_blocks]

        

        results = []

        with torch.no_grad():

            for (start_time, end_time, left_id, right_id, level_code) in candidates:

                parent_len = end_time - start_time

                mid = start_time + (parent_len // 2 // min_res) * min_res

                if mid <= start_time or mid >= end_time:

                    mid = start_time + parent_len // 2

                # 冻结 parent 预测（开 fake-quant，匹配落盘）

                parent_pred = self.model.decode_single(

                    left_id, right_id, parent_len,

                    offset=start_time % self.args.base_block_size).squeeze()

                true_full = self.raw_data[start_time:end_time]

                if true_full.device != parent_pred.device:

                    true_full = true_full.to(parent_pred.device)

                

                res = evaluate_parent_patch_split(

                    parent_pred=parent_pred, true_full=true_full,

                    start=start_time, end=end_time, mid=mid,

                    eps_parent=_block_eps(start_time, end_time),

                    eps_left=_block_eps(start_time, mid),

                    eps_right=_block_eps(mid, end_time),

                    K_list=K_list, modes=modes,

                )

                results.append(res)

        

        if not results:

            print("[Patch-Split Eval] No splittable candidates.")

            return {}

        

        # ── 接入：把 ACCEPT 的 patch 提交到 manager（commit=True）──

        if commit:

            if self.patch_manager is None:

                self.patch_manager = PatchSplitManager()

            for r in results:

                if r.accept:

                    self.patch_manager.commit(r)

        

        # ── 统计 ──

        n_total = len(results)

        accepted = [r for r in results if r.accept]

        n_acc = len(accepted)

        # 全局账本：只接受赚钱的 split（未分裂块成本不变，故全局净 = Σ accepted net_gain）

        gated_net = sum(r.net_gain for r in accepted)

        gated_res_saved = sum(r.parent_residual_bytes - (r.left_fit.residual_bytes + r.right_fit.residual_bytes)

                              for r in accepted)

        gated_patch_bytes = sum(r.left_fit.coeff_bytes + r.right_fit.coeff_bytes for r in accepted)

        raw_net = sum(r.net_gain for r in results)

        mae_imp_but_reject = sum(

            1 for r in results

            if (not r.accept) and r.parent_mae > 0 and

               (r.parent_mae - r.child_mae) / (r.parent_mae + 1e-9) > 0.05)

        

        W = 124

        print("\n" + "=" * W)

        print("PARENT-ANCHORED PATCH-SPLIT EVALUATION  (frozen decoder, no shared-state change; report-only)")

        print("=" * W)

        hdr = (f"  {'[start,end]':>15} {'plen':>5} | {'pBits':>5} | "

               f"{'K L/R':>7} {'mode L/R':>10} | "

               f"{'resB old->new':>15} | {'patchB':>7} | {'netGain':>8} | {'MAEimp':>7} | status")

        print(hdr)

        print("  " + "-" * (W - 2))

        for r in sorted(results, key=lambda x: x.net_gain, reverse=True)[:60]:

            mae_imp = (r.parent_mae - r.child_mae) / (r.parent_mae + 1e-9) * 100

            new_res = r.left_fit.residual_bytes + r.right_fit.residual_bytes

            patchB = r.left_fit.coeff_bytes + r.right_fit.coeff_bytes

            status = "ACCEPT" if r.accept else "REJECT"

            flag = "  <- acc+ no bytes" if (not r.accept and mae_imp > 5.0) else ""

            print(f"  [{r.start:7d},{r.end:7d}] {r.parent_len:5d} | {r.parent_bits:>5} | "

                  f"{r.left_fit.K:>3}/{r.right_fit.K:<3} "

                  f"{r.left_fit.mode:>4}/{r.right_fit.mode:<4} | "

                  f"{r.parent_residual_bytes:>6}->{new_res:<6}  | {patchB:>7} | "

                  f"{r.net_gain:>+8} | {mae_imp:>+6.1f}% | {status}{flag}")

        

        print("  " + "-" * (W - 2))

        print(f"  Candidates evaluated : {n_total}")

        print(f"    ACCEPT (netGain>0) : {n_acc}")

        print(f"    REJECT             : {n_total - n_acc}")

        print(f"    MAE improved(>5%) but REJECTED: {mae_imp_but_reject}")

        print(f"  [raw]        all splits net      : {raw_net:+d} B")

        print(f"  [rate-gated] accepted-only net   : {gated_net:+d} B  "

              f"({'saves' if gated_net > 0 else 'wastes'} space)")

        print(f"               residual saved      : {gated_res_saved:+d} B")

        print(f"               patch coeff bytes   : {gated_patch_bytes:+d} B")

        print(f"  NOTE: unsplit blocks unchanged by construction (decoder/grid/index untouched).")

        print(f"        global total bytes change  = -[rate-gated net] = {-gated_net:+d} B")

        print("=" * W)

        

        return {

            'n_total': n_total, 'n_accept': n_acc,

            'raw_net': raw_net, 'gated_net': gated_net,

            'gated_res_saved': gated_res_saved, 'gated_patch_bytes': gated_patch_bytes,

            'mae_imp_but_reject': mae_imp_but_reject,

            'results': results,

        }



    def _build_block_codec(self):

        """

        把 multilayer commit 收集的 _codec_blocks 转成统一 BlockCodec。

        

        每块：分段比例 → 结构码；每叶子系数 δ pad/截到 K_fixed+2（定长）；

        unsplit 块 = code 0 单叶子（δ 全零，仅靠 base + 残差）。

        残差只记 meta（res_bits 由 span 推），实际残差数据落盘时再补。

        """

        bbs = self.args.base_block_size

        min_res = self.manager.min_resolution

        # 用最深叶子推断 K_fixed（取已提交段的系数维度；无则 0）

        K_fixed = None

        for info in self._codec_blocks.values():

            if info['segs']:

                K_fixed = info['segs'][0]['fit'].delta_q.shape[0] - 2

                break

        if K_fixed is None:

            K_fixed = 0

        

        # 实测最大树高：用 RDO 回收后各块的最细叶子推断，收紧结构码位宽

        # height = log2(block_len / min_leaf_len)；取所有块最大值

        import math as _m

        actual_max_h = 0

        for info in self._codec_blocks.values():

            segs = info['segs']

            if not segs:

                continue

            blen = info['block_len']

            min_leaf = min(s['end'] - s['start'] for s in segs)

            if min_leaf > 0:

                h = int(round(_m.log2(blen / min_leaf)))

                actual_max_h = max(actual_max_h, h)

        # 用实测高度建 codebook（而非 base/min 理论上限），码更短

        codec = BlockCodec(base_block_size=bbs, min_resolution=min_res,

                           K_fixed=K_fixed, max_depth=actual_max_h)

        cb = codec.codebook

        device = self.grid_storage.base_grid.device

        Kp = K_fixed + 2

        

        for bid in sorted(self._codec_blocks.keys()):

            info = self._codec_blocks[bid]

            bstart, blen = info['block_start'], info['block_len']

            segs = info['segs']

            if not segs:

                # unsplit：整块单叶子，code 0，δ 全零

                frac = (1.0,)

                leaves_seg = [{'start': bstart, 'end': bstart + blen,

                               'fit': None}]

            else:

                frac = tuple(round((s['end'] - s['start']) / blen, 6) for s in segs)

                leaves_seg = segs

            # 分段比例 → 结构码（找最接近的合法划分）

            code = cb.code_of.get(frac, None)

            if code is None:

                # 比例不在码本：仅末尾不完整块(block_len != base)会出现非2幂次切分。

                # 此时 _nearest_code 选出的码本边界与实际叶子边界不符，会导致

                # locate / 随机访问越界。根治：把这种块直接降级为 unsplit 单叶子，

                # 结构码恒与存储叶子一致（牺牲该单块的 patch，仅末块1块，可忽略）。

                if blen != bbs:

                    code = 0  # 整块单叶子

                    leaves_seg = [{'start': bstart, 'end': bstart + blen,

                                   'fit': None}]

                else:

                    code = self._nearest_code(cb, frac)

            # 构建叶子记录

            leaf_recs = []

            for s in leaves_seg:

                seg_len = s['end'] - s['start']

                if s['fit'] is None:

                    delta = torch.zeros(Kp, device=device)

                    coeff_b = 0

                    res_bits = 0

                else:

                    dq = s['fit'].delta_q.to(device)

                    # pad/截到定长 Kp

                    if dq.shape[0] < Kp:

                        dq = torch.cat([dq, torch.zeros(Kp - dq.shape[0], device=device)])

                    elif dq.shape[0] > Kp:

                        dq = dq[:Kp]

                    delta = dq

                    coeff_b = s['fit'].coeff_bytes

                    res_bits = getattr(s['fit'], 'res_bits', 0) if hasattr(s['fit'], 'res_bits') else 0

                leaf_recs.append(LeafRecord(

                    start=s['start'], end=s['end'], delta_q=delta,

                    res_bits=res_bits,

                    res_bytes=(s['fit'].residual_bytes if s['fit'] is not None else 0),

                    coeff_bytes=coeff_b,

                ))

            codec.add_block(BlockRecord(block_id=bid, block_start=bstart,

                                        block_len=blen, code=code, leaves=leaf_recs,

                                        left_id=info['left_id'],

                                        right_id=info.get('right_id', -1)))

        

        codec.finalize(device=device)

        # ── 残差落盘（方案B）：仅随机访问场景需要（默认关，压缩比不触发，省时间）──
        # 由 self._codec_build_residual 控制：bench_random_access / 导出时置 True。
        # 与压缩比口径一致：残差 = 真值 - (base 预测 + patch 修正)，按叶子顺序量化打包。
        if getattr(self, '_codec_build_residual', False):
          try:
            from models.residual_codec import ResidualCodec
            import numpy as _np
            eps_fn = getattr(self, '_codec_eps_fn', None)
            if eps_fn is not None:
                leaf_residuals = []
                with torch.no_grad():
                    for bid in sorted(codec.blocks.keys()):
                        rec = codec.blocks[bid]
                        # base 预测（整块）
                        base_pred = self.model.decode_single(
                            rec.left_id, rec.right_id, rec.block_len,
                            offset=rec.block_start % bbs).squeeze()
                        true_full = self.raw_data[rec.block_start:rec.block_start + rec.block_len]
                        if true_full.device != base_pred.device:
                            true_full = true_full.to(base_pred.device)
                        for li, lf in enumerate(rec.leaves):
                            lo = lf.start - rec.block_start
                            hi = min(lf.end - rec.block_start, base_pred.shape[0])
                            seg_base = base_pred[lo:hi]
                            seg_true = true_full[lo:hi]
                            L = seg_base.shape[0]
                            # patch 修正
                            row = codec.leaf_global_index(bid, li)
                            delta = codec._coeff_pool[row]
                            if delta.abs().sum() > 0:
                                from models.block_codec import FourierCoeffs
                                seg_base = seg_base + FourierCoeffs.synthesize(delta, L, codec.K_fixed)
                            resid = (seg_true - seg_base).detach().cpu().numpy().astype(_np.float64)
                            leaf_residuals.append(resid)
                # 单一 eps（absolute 模式）；relative 模式用每块 eps 的近似（取整体 norm_threshold 兜底）
                eps_val = self._codec_norm_threshold
                if eps_val is None:
                    # relative：用首块 eps 近似（仅用于位宽估计，误差界仍按块保证）
                    eps_val = eps_fn(0, min(bbs, self.manager.total_length))
                rcodec = ResidualCodec(float(eps_val)).encode(leaf_residuals)
                codec.attach_residual_codec(rcodec)
                rstat = rcodec.total_bytes()
                print(f"  [ResidualCodec] {rstat['num_leaves']} leaves, "
                      f"header={rstat['header_bytes']}B, bitstream={rstat['bitstream_bytes']}B, "
                      f"total={rstat['total_bytes']}B")
          except Exception as _e:
            print(f"  [ResidualCodec] skipped ({_e})")

        # 随机访问器（独立模块 BlockAccessor）：注入 base 预测回调

        from models.block_accessor import BlockAccessor

        bbs2 = self.args.base_block_size

        def _decode_fn(left_id, right_id, block_len, block_start):

            with torch.no_grad():

                return self.model.decode_single(

                    left_id, right_id, block_len,

                    offset=block_start % bbs2).squeeze()

        def _decode_blocks_fn(left_ids, right_ids, block_len, offsets):

            # 同块长一组，一次 decode_batch（一次 GEMM）

            with torch.no_grad():

                out = self.model.decode_batch(left_ids, right_ids, block_len, offsets)

                return out.squeeze(1)   # [K, block_len]

        # 残差回调：从 ResidualCodec 按全局行号取该叶子残差（接入访问路径）

        _rc = getattr(codec, 'residual_codec', None)

        def _residual_fn(row, seg_len):

            if _rc is None:

                return None

            import numpy as _np2

            r = _rc.decode_leaf(int(row))

            return torch.from_numpy(_np2.ascontiguousarray(r, dtype=_np2.float32))

        self.block_codec = codec

        self.block_accessor = BlockAccessor(codec, _decode_fn,

                                            residual_fn=(_residual_fn if _rc is not None else None))

        self.block_accessor.attach_batched_decode(_decode_blocks_fn)

        stats = codec.total_bytes()

        print(f"\n  [BlockCodec] built: {stats['num_blocks']} blocks, "

              f"{stats['num_leaves']} leaves, code {cb.code_bytes}B/block "

              f"(max_depth={codec.max_depth}, {cb.n_codes} partitions)")

        print(f"    code bytes={stats['code_bytes']}, coeff bytes={stats['coeff_bytes']}, "

              f"residual bytes={stats['residual_bytes']}, total={stats['total_bytes']}")

    

    @staticmethod

    def _nearest_code(cb, frac):

        """分段比例不在码本中时，找叶子数相同、各叶子比例最接近的合法划分。"""

        best, best_d = 0, float('inf')

        for code, part in enumerate(cb.partitions):

            if len(part) != len(frac):

                continue

            d = sum(abs(a - b) for a, b in zip(part, frac))

            if d < best_d:

                best_d, best = d, code

        return best

    

    def multilayer_patch_eval(self, error_threshold: float = 0.05, error_mode: str = 'relative',

                              K_list=(0, 4, 8), modes=('int8', 'fp16'),

                              max_depth: int = 3, max_blocks: int = None,

                              commit: bool = False):

        """

        多层 patch + 自下而上 RDO 剪枝（离线核验 + 可选提交接入）。

        

        对每个 base 块递归二分到 max_depth，每段独立闭式拟合单层 patch；自下而上 DP

        比较"保留分裂 vs 合并单层"，自动选每个区间的最优分段深度（= 自下而上回收

        冗余分裂层）。与单层(只切一刀)对比，回答"多层是否值得"。纯计算、不训练、

        不改任何共享状态。commit=True 时把 DP 回收后的最优分段提交到 patch_manager。

        

        Args:

            K_list: 每段枚举的频率数

            modes: 量化模式

            max_depth: 最大分裂深度(1=单层, 2=最多切到1/4, 3=1/8...)

            max_blocks: 仅评估前 N 个(调试)

            commit: True 时提交回收后的分段（仅提交比 no-patch 更省的块）

        """

        self.model.eval()

        self._refresh_quant_params()

        

        scaler = self.data_loader.scaler

        if scaler is not None:

            std_val = scaler.std.item() if hasattr(scaler.std, 'item') else float(scaler.std)

            mean_val = scaler.mean.item() if hasattr(scaler.mean, 'item') else float(scaler.mean)

        else:

            std_val, mean_val = 1.0, 0.0

        norm_threshold = error_threshold / std_val if error_mode == 'absolute' else None

        

        def _eps_fn(t0, t1):

            if error_mode == 'absolute':

                return norm_threshold

            true_orig = self.raw_data[t0:t1] * std_val + mean_val

            min_denom = torch.clamp(torch.abs(true_orig), min=1.0).min().item()

            return error_threshold * min_denom / std_val if std_val > 0 else error_threshold

        

        all_blocks = self.manager.get_all_unique_blocks()

        min_res = self.manager.min_resolution

        candidates = [b for b in all_blocks

                      if (b[1] - b[0]) >= 2 * min_res and b[4] < self.manager.max_level]

        if max_blocks is not None:

            candidates = candidates[:max_blocks]

        

        if commit and self.patch_manager is None:

            self.patch_manager = PatchSplitManager()

        # 收集 codec 块信息（commit 时填充，循环后统一构建 BlockCodec）

        self._codec_blocks = {}

        # 保存残差量化所需的 eps 信息（供 _build_block_codec 重算叶子残差落盘）

        self._codec_eps_fn = _eps_fn

        self._codec_norm_threshold = norm_threshold

        self._codec_error_mode = error_mode

        def block_id_for(t):

            return t // self.args.base_block_size

        

        # 统计：各深度被 DP 选中的叶子分布、单层 vs 多层总字节

        depth_leaf_hist = {}     # depth -> 叶子数

        total_no_patch = 0       # 不分裂(整块残差)字节

        total_single = 0         # 单层(只切一刀)最优字节

        total_multi = 0          # 多层 DP 最优字节

        n_multi_deeper = 0       # DP 选择 >1 层的块数

        n_committed = 0          # 实际提交的块数

        total_full_bytes = 0     # 不回收(强制切到 max_depth)的字节

        total_full_leaves = 0    # 不回收时的总段数

        total_kept_leaves = 0    # 回收后保留的总段数

        height_hist = {}         # RDO 回收后实际树高 -> 块数

        

        with torch.no_grad():

            for (start_time, end_time, left_id, right_id, level_code) in candidates:

                blk_len = end_time - start_time

                parent_pred = self.model.decode_single(

                    left_id, right_id, blk_len,

                    offset=start_time % self.args.base_block_size).squeeze()

                true_full = self.raw_data[start_time:end_time]

                if true_full.device != parent_pred.device:

                    true_full = true_full.to(parent_pred.device)

                

                # 不分裂基线

                resid0 = true_full - parent_pred

                span0 = (resid0.max() - resid0.min()).item()

                eps0 = _eps_fn(start_time, end_time)

                base_bytes = FallbackDict.estimate_bitwidth_cost(span0, blk_len, eps0)

                total_no_patch += base_bytes

                

                # 多层 DP 树

                root = build_multilayer_tree(

                    parent_pred, true_full, start_time, end_time, start_time,

                    _eps_fn, K_list, modes, min_res, depth=0, max_depth=max_depth)

                total_multi += root.best_bytes

                

                # 单层(max_depth=1)最优

                root1 = build_multilayer_tree(

                    parent_pred, true_full, start_time, end_time, start_time,

                    _eps_fn, K_list, modes, min_res, depth=0, max_depth=1)

                total_single += root1.best_bytes

                

                leaves, used_depth = count_leaves_and_depth(root)

                depth_leaf_hist[used_depth] = depth_leaf_hist.get(used_depth, 0) + 1

                if leaves > 2:

                    n_multi_deeper += 1

                

                # 回收量统计：不回收(全切到 max_depth) vs 回收后(DP best)

                total_full_bytes += root.full_bytes

                total_full_leaves += root.full_leaves

                total_kept_leaves += leaves

                

                # RDO 回收后实际树高（决定结构码位宽）

                _h = tree_height_after_pruning(root)

                height_hist[_h] = height_hist.get(_h, 0) + 1

                

                # 接入：提交回收后的最优分段（仅当比 no-patch 更省）

                if commit:

                    from models.patch_split import collect_leaf_segments

                    # 多层 DP 最优字节(含 meta) vs 不分裂的残差字节

                    if root.best_bytes < base_bytes:

                        segs = collect_leaf_segments(root)

                        self.patch_manager.commit_segments(start_time, end_time, segs)

                        n_committed += 1

                        self._codec_blocks[block_id_for(start_time)] = {

                            'block_start': start_time, 'block_len': blk_len,

                            'left_id': left_id, 'right_id': right_id, 'segs': segs,

                        }

                    else:

                        # 未提交 → 整块单叶子(code 0)，无 patch 系数(全零)，base+残差兜底

                        self._codec_blocks[block_id_for(start_time)] = {

                            'block_start': start_time, 'block_len': blk_len,

                            'left_id': left_id, 'right_id': right_id, 'segs': None,

                        }

        

        # ── 构建统一 BlockCodec（仅 commit 时）──

        if commit and self._codec_blocks:

            self._build_block_codec()

        

        W = 78

        print("\n" + "=" * W)

        print(f"MULTI-LAYER PATCH + BOTTOM-UP RDO PRUNING  (max_depth={max_depth}"

              + (", COMMIT" if commit else "") + ")")

        print("=" * W)

        print(f"  Candidate base blocks : {len(candidates)}")

        print(f"  Blocks DP chose >1 layer (>2 segments): {n_multi_deeper}")

        if commit:

            print(f"  Blocks committed (multi-layer, net-positive): {n_committed}")

        print(f"  DP depth distribution (max depth reached per block):")

        for d in sorted(depth_leaf_hist.keys()):

            print(f"    depth {d}: {depth_leaf_hist[d]} blocks")

        print(f"\n  Residual+patch bytes on candidate blocks:")

        print(f"    no-patch (baseline)   : {total_no_patch} B")

        print(f"    single-layer (1 cut)  : {total_single} B  "

              f"(Δ vs baseline {total_no_patch - total_single:+d})")

        print(f"    multi-layer (DP)      : {total_multi} B  "

              f"(Δ vs baseline {total_no_patch - total_multi:+d})")

        _multi_gain = total_single - total_multi

        print(f"  Multi-layer extra saving over single-layer: {_multi_gain:+d} B "

              f"({'worth it' if _multi_gain > 0 else 'NOT worth it'})")

        # ── 自下而上回收量（RDO pruning）──

        _reclaimed_segs = total_full_leaves - total_kept_leaves

        _reclaimed_bytes = total_full_bytes - total_multi

        print(f"\n  ── Bottom-up RDO reclamation (over-split → merge) ──")

        print(f"    segments before pruning (full split to depth {max_depth}): {total_full_leaves}")

        print(f"    segments after  pruning (DP optimal)                    : {total_kept_leaves}")

        print(f"    reclaimed segments: {_reclaimed_segs} "

              f"({100.0 * _reclaimed_segs / max(total_full_leaves,1):.1f}% of full-split segments)")

        print(f"    bytes before pruning: {total_full_bytes} B")

        print(f"    bytes after  pruning: {total_multi} B")

        print(f"    reclaimed bytes:      {_reclaimed_bytes} B "

              f"(saved by merging redundant splits)")

        # ── 树高分布（决定结构码位宽）──

        _n = sum(height_hist.values())

        _max_h = max(height_hist.keys()) if height_hist else 0

        # 覆盖 99% 块所需高度

        _cum = 0

        _h99 = _max_h

        for h in sorted(height_hist.keys()):

            _cum += height_hist[h]

            if _cum >= 0.99 * _n:

                _h99 = h

                break

        def _codes_for_depth(d):

            c = 1

            for _ in range(d):

                c = 1 + c * c

            return c

        _codes_max = _codes_for_depth(_max_h)

        _code_bytes = 1 if _codes_max <= 256 else (2 if _codes_max <= 65536 else 3)

        print(f"\n  ── Tree height after RDO pruning (decides structure-code width) ──")

        for h in sorted(height_hist.keys()):

            print(f"    height {h}: {height_hist[h]} blocks ({100*height_hist[h]/_n:.1f}%)")

        print(f"    max height = {_max_h}  → partitions={_codes_max} → structure code {_code_bytes} byte(s)")

        print(f"    height covering 99% blocks = {_h99} "

              f"(partitions={_codes_for_depth(_h99)}, "

              f"{1 if _codes_for_depth(_h99)<=256 else 2} byte if capped here)")

        print(f"\n  Interpretation:")

        print(f"    multi >> single → deeper splits pay off, multi-layer + bottom-up GC justified.")

        print(f"    multi ≈ single  → single layer suffices; multi-layer is over-design.")

        print("=" * W)

        

        return {

            'n_candidates': len(candidates),

            'n_multi_deeper': n_multi_deeper,

            'n_committed': n_committed,

            'total_no_patch': total_no_patch,

            'total_single': total_single,

            'total_multi': total_multi,

            'multi_gain_over_single': _multi_gain,

            'reclaimed_segments': _reclaimed_segs,

            'reclaimed_bytes': _reclaimed_bytes,

            'full_leaves': total_full_leaves,

            'kept_leaves': total_kept_leaves,

            'height_hist': height_hist,

            'depth_hist': depth_leaf_hist,

        }



    def ablation_split_vs_whole(self, error_threshold: float = 0.05, error_mode: str = 'relative',

                                split_K_list=(0, 4, 8), modes=('int8', 'fp16'),

                                max_blocks: int = None):

        """

        Ablation：相同系数预算下，分裂(B) vs 整块加系数(A)。

        

        回答审稿人质疑"既然是残差修正，为何要分裂"。两方案共享同一 parent 预测、

        同一 eps、同一 EDWB 公式；A 的频率预算 ≈ 2×split_K（与 B 的两个 child 系数

        总数可比）。逐块对比 net_gain，统计 B 严格优于 A 的块数与字节差。

        

        关键：若 B 在【相同系数预算】下普遍更省字节，证明收益来自分裂的 span 隔离

        （EDWB 按块内最坏 span 收费），而非"多给了系数"。

        

        Args:

            split_K_list: B（分裂）每个 child 枚举的频率数

            modes: 量化模式

            max_blocks: 仅评估前 N 个（调试）

        """

        self.model.eval()

        self._refresh_quant_params()

        

        scaler = self.data_loader.scaler

        if scaler is not None:

            std_val = scaler.std.item() if hasattr(scaler.std, 'item') else float(scaler.std)

            mean_val = scaler.mean.item() if hasattr(scaler.mean, 'item') else float(scaler.mean)

        else:

            std_val, mean_val = 1.0, 0.0

        norm_threshold = error_threshold / std_val if error_mode == 'absolute' else None

        

        def _block_eps(t0, t1):

            if error_mode == 'absolute':

                return norm_threshold

            true_orig = self.raw_data[t0:t1] * std_val + mean_val

            min_denom = torch.clamp(torch.abs(true_orig), min=1.0).min().item()

            return error_threshold * min_denom / std_val if std_val > 0 else error_threshold

        

        # A（整块）频率预算 ≈ 2×split_K，保证系数总数与 B 可比

        whole_K_list = tuple(sorted(set(2 * k for k in split_K_list if k > 0)) or (8,))

        

        all_blocks = self.manager.get_all_unique_blocks()

        min_res = self.manager.min_resolution

        candidates = [b for b in all_blocks

                      if (b[1] - b[0]) >= 2 * min_res and b[4] < self.manager.max_level]

        if max_blocks is not None:

            candidates = candidates[:max_blocks]

        

        rows = []

        with torch.no_grad():

            for (start_time, end_time, left_id, right_id, level_code) in candidates:

                blk_len = end_time - start_time

                mid = start_time + (blk_len // 2 // min_res) * min_res

                if mid <= start_time or mid >= end_time:

                    mid = start_time + blk_len // 2

                parent_pred = self.model.decode_single(

                    left_id, right_id, blk_len,

                    offset=start_time % self.args.base_block_size).squeeze()

                true_full = self.raw_data[start_time:end_time]

                if true_full.device != parent_pred.device:

                    true_full = true_full.to(parent_pred.device)

                

                # B: 分裂

                b = evaluate_parent_patch_split(

                    parent_pred=parent_pred, true_full=true_full,

                    start=start_time, end=end_time, mid=mid,

                    eps_parent=_block_eps(start_time, end_time),

                    eps_left=_block_eps(start_time, mid),

                    eps_right=_block_eps(mid, end_time),

                    K_list=split_K_list, modes=modes,

                )

                # A: 整块加系数（预算 ≈ 2×split_K）

                a = evaluate_whole_block_patch(

                    parent_pred=parent_pred, true_full=true_full,

                    start=start_time, end=end_time,

                    eps_parent=_block_eps(start_time, end_time),

                    K_whole_list=whole_K_list, modes=modes,

                )

                rows.append((a, b))

        

        if not rows:

            print("[Ablation] No candidates.")

            return {}

        

        # 仅统计"至少一方值得修正"的块（两者都不 accept 的块对结论无意义）

        useful = [(a, b) for (a, b) in rows if a.accept or b.accept]

        n = len(useful)

        b_better = sum(1 for (a, b) in useful if b.new_total_bytes < a.new_total_bytes)

        a_better = sum(1 for (a, b) in useful if a.new_total_bytes < b.new_total_bytes)

        tie = n - b_better - a_better

        sum_a = sum(a.new_total_bytes for (a, b) in useful)

        sum_b = sum(b.new_total_bytes for (a, b) in useful)

        # 仅在两者都接受的块上比净收益（最公平的子集）

        both = [(a, b) for (a, b) in useful if a.accept and b.accept]

        sum_a_both = sum(a.new_total_bytes for (a, b) in both)

        sum_b_both = sum(b.new_total_bytes for (a, b) in both)

        

        W = 100

        print("\n" + "=" * W)

        print("ABLATION: SPLIT (B) vs WHOLE-BLOCK-COEFFS (A)  [matched coefficient budget]")

        print("=" * W)

        print(f"  split_K (B per-child) : {split_K_list}")

        print(f"  whole_K (A budget)    : {whole_K_list}  (≈ 2×split_K, comparable total coeffs)")

        print(f"  Blocks (>=1 accepts)  : {n}")

        print(f"    B better (fewer bytes): {b_better}")

        print(f"    A better              : {a_better}")

        print(f"    tie                   : {tie}")

        print(f"  Total bytes on these blocks:")

        print(f"    A (whole-block) : {sum_a} B")

        print(f"    B (split)       : {sum_b} B   (Δ {sum_a - sum_b:+d} B, "

              f"{'B wins' if sum_b < sum_a else 'A wins'})")

        if both:

            print(f"  On blocks where BOTH accept ({len(both)}):")

            print(f"    A : {sum_a_both} B | B : {sum_b_both} B  (Δ {sum_a_both - sum_b_both:+d} B)")

        print(f"\n  Interpretation:")

        print(f"    B<A under matched budget  → split has independent value (span isolation),")

        print(f"                                 not just 'more coefficients'.")

        print(f"    B≈A                        → split unnecessary; reframe as adaptive coeffs.")

        print("=" * W)

        

        return {

            'n': n, 'b_better': b_better, 'a_better': a_better, 'tie': tie,

            'sum_a': sum_a, 'sum_b': sum_b,

            'sum_a_both': sum_a_both, 'sum_b_both': sum_b_both,

            'whole_K_list': whole_K_list, 'split_K_list': split_K_list,

        }



    def compression_ratio_with_patches(self, error_threshold: float = 0.05,

                                       error_mode: str = 'relative'):

        """

        含 patch-split 的真实压缩比核算（接入后的最终体积）。

        

        与 final_evaluation 同口径（grid + index + EDWB 残差），但：

        - 被 patch 的 parent：残差 = 左右 child 残差和；额外加 patch 系数字节。

        - 未被 patch 的块：残差 = 整块 bitwidth_cost（不变）。

        - grid 节点数不增加（patch 不分配 z_M）。

        

        同时打印 baseline（无 patch）对照，直观看出 patch 净收益。

        """

        if self.patch_manager is None or self.patch_manager.total_patches() == 0:

            print("\n[Patch Compression] No committed patches; nothing to account.")

            return None

        

        self.model.eval()

        self._refresh_quant_params()

        

        scaler = self.data_loader.scaler

        if scaler is not None:

            std_val = scaler.std.item() if hasattr(scaler.std, 'item') else float(scaler.std)

            mean_val = scaler.mean.item() if hasattr(scaler.mean, 'item') else float(scaler.mean)

        else:

            std_val, mean_val = 1.0, 0.0

        norm_threshold = error_threshold / std_val if error_mode == 'absolute' else None

        

        def _block_eps(t0, t1):

            if error_mode == 'absolute':

                return norm_threshold

            true_orig = self.raw_data[t0:t1] * std_val + mean_val

            min_denom = torch.clamp(torch.abs(true_orig), min=1.0).min().item()

            return error_threshold * min_denom / std_val if std_val > 0 else error_threshold

        

        all_blocks = self.manager.get_all_unique_blocks()

        baseline_resid = 0      # 无 patch：每块整块 bitwidth_cost

        patched_resid = 0       # 有 patch：被 patch 块用 child 残差，其余不变

        patch_coeff_total = 0

        n_patched = 0

        

        with torch.no_grad():

            for (start_time, end_time, left_id, right_id, level_code) in all_blocks:

                blk_len = end_time - start_time

                eps = _block_eps(start_time, end_time)

                parent_pred = self.model.decode_single(

                    left_id, right_id, blk_len,

                    offset=start_time % self.args.base_block_size).squeeze()

                true_full = self.raw_data[start_time:end_time]

                if true_full.device != parent_pred.device:

                    true_full = true_full.to(parent_pred.device)

                

                # baseline：整块残差代价

                span = (true_full - parent_pred).abs()

                resid = true_full - parent_pred

                blk_span = (resid.max() - resid.min()).item()

                base_cost = FallbackDict.estimate_bitwidth_cost(blk_span, blk_len, eps)

                baseline_resid += base_cost

                

                # patched

                entry = self.patch_manager.get(start_time, end_time)

                if entry is not None:

                    patched_resid += self.patch_manager.child_residual_bytes(start_time, end_time)

                    patch_coeff_total += self.patch_manager.patch_bytes(start_time, end_time)

                    n_patched += 1

                else:

                    patched_resid += base_cost

        

        # grid + index（patch 不增加节点）

        vec_dim = self.grid_storage.trend_dim + self.grid_storage.context_dim

        total_nodes = self.grid_storage.num_base_nodes + self.manager.patch_counter

        _qb = getattr(self.args, 'quant_bits', 8)

        _bpd = (_qb / 8) if _qb > 0 else 4.0

        grid_bytes = int(total_nodes * vec_dim * _bpd)

        base_block_size = self.manager.base_block_size

        num_base_blocks = math.ceil(self.manager.total_length / base_block_size)

        index_bytes = num_base_blocks * FallbackDict.INDEX_PER_BASE_BLOCK

        

        original_bytes = self.manager.total_length * 4

        

        base_total = grid_bytes + index_bytes + baseline_resid

        patch_total = grid_bytes + index_bytes + patched_resid + patch_coeff_total

        base_ratio = original_bytes / base_total if base_total > 0 else float('inf')

        patch_ratio = original_bytes / patch_total if patch_total > 0 else float('inf')

        

        # 模型权重（共享 decoder）存储代价：一次性计入，按 quant_bits 量化

        decoder_params = sum(p.numel() for p in self.model.decoder.parameters())

        decoder_bytes = int(decoder_params * _bpd)

        base_total_w = base_total + decoder_bytes

        patch_total_w = patch_total + decoder_bytes

        base_ratio_w = original_bytes / base_total_w if base_total_w > 0 else float('inf')

        patch_ratio_w = original_bytes / patch_total_w if patch_total_w > 0 else float('inf')

        

        W = 70

        print("\n" + "=" * W)

        print("COMPRESSION WITH PATCH-SPLIT (committed patches)")

        print("=" * W)

        print(f"  Committed patches    : {n_patched}")

        print(f"  Original data        : {original_bytes/1024:>10.2f} KB")

        print(f"  Grid ({_qb}-bit)        : {grid_bytes/1024:>10.2f} KB ({total_nodes} nodes, unchanged)")

        print(f"  Index (bitmask)      : {index_bytes/1024:>10.2f} KB")

        print(f"  Decoder weights      : {decoder_bytes/1024:>10.2f} KB ({decoder_params} params x {_bpd:.1f}B, shared once)")

        print(f"  --- Baseline (no patch) ---")

        print(f"    Residual (EDWB)    : {baseline_resid/1024:>10.2f} KB")

        print(f"    Total              : {base_total/1024:>10.2f} KB  → ratio {base_ratio:.3f}x")

        print(f"    Total (+weights)   : {base_total_w/1024:>10.2f} KB  → ratio {base_ratio_w:.3f}x")

        print(f"  --- With patch-split ---")

        print(f"    Residual (EDWB)    : {patched_resid/1024:>10.2f} KB")

        print(f"    Patch coeffs       : {patch_coeff_total/1024:>10.2f} KB")

        print(f"    Total              : {patch_total/1024:>10.2f} KB  → ratio {patch_ratio:.3f}x")

        print(f"    Total (+weights)   : {patch_total_w/1024:>10.2f} KB  → ratio {patch_ratio_w:.3f}x")

        _delta = base_total - patch_total

        print(f"  Net bytes saved      : {_delta:+d} B "

              f"({'patch wins' if _delta > 0 else 'patch loses'})")

        print("=" * W)

        

        return {

            'base_total': base_total, 'patch_total': patch_total,

            'base_ratio': base_ratio, 'patch_ratio': patch_ratio,

            'n_patched': n_patched, 'net_saved': _delta,

            'decoder_bytes': decoder_bytes, 'decoder_params': decoder_params,

            'base_total_w': base_total_w, 'patch_total_w': patch_total_w,

            'base_ratio_w': base_ratio_w, 'patch_ratio_w': patch_ratio_w,

        }



    def _rollback_split(self, start: int, end: int) -> bool:
        """
        回收一次分裂，执行以下四步：

        Step 0: 还原 z_L（优先 shadow，fallback snapshot）
            shadow z_L 由 _update_zL_shadows() 在 finetune 后计算：
            从 warmup 前快照出发，用当前 decoder 在父块数据上训练，
            是与当前 decoder 匹配的最优父块表示，不含子块污染。
            若 shadow 不可用则退到 warmup 前快照（冷启动）。

        Step 1: 回退 index_table
            将 [start, end) 范围内所有槽位恢复为 (left_id, right_id, orig_level)，
            使 z_L 重新成为完整父块的左边界节点。

        Step 2: 释放 z_M
            将 new_node_id 归还到 manager._free_patch_ids（free-list），
            下次分裂可直接复用该 patch_grid 槽位，patch_counter 不增加。

        Step 3: 从 split_history 移除记录

        Returns:
            True  — rollback succeeded
            False — data missing (presplit_stats / split_history / level_code)
        """
        if (start, end) not in self._presplit_stats:
            return False
        if (start, end) not in self.split_history:
            return False

        ppre         = self._presplit_stats[(start, end)]
        left_id      = ppre['left_id']
        right_id     = ppre['right_id']
        orig_level   = ppre.get('level_code')
        if orig_level is None:
            return False

        # Read new_node_id (z_M) before removing from split_history
        _, new_node_id = self.split_history[(start, end)]

        # Step 0: Restore z_L from shadow (parent-trained, uncontaminated by child specialization).
        # Shadow z_L was computed by _update_zL_shadows() starting from the pre-warmup snapshot
        # and fine-tuned on the parent block with the current decoder — the cleanest restoration.
        # Fall back to pre-warmup snapshot if shadow is unavailable.
        gs = self.model.grid_storage
        _z_restore = getattr(self, '_zL_shadows',   {}).get((start, end))
        if _z_restore is None:
            _z_restore = getattr(self, '_zL_snapshots', {}).get((start, end))
        if _z_restore is not None:
            with torch.no_grad():
                if left_id < gs.num_base_nodes:
                    gs.base_grid.data[left_id] = _z_restore.clone()
                else:
                    gs.patch_grid.data[left_id - gs.num_base_nodes] = _z_restore.clone()

        # Step 1: Revert index_table — z_L covers full [start, end) again
        # slots: [start//min_res,  end//min_res - 1] inclusive
        min_res         = self.manager.min_resolution
        block_start_idx = start // min_res
        block_end_idx   = (end   // min_res) - 1
        for i in range(block_start_idx, block_end_idx + 1):
            self.manager.index_table[i].left_id    = left_id
            self.manager.index_table[i].right_id   = right_id
            self.manager.index_table[i].level_code = orig_level

        # Step 2: Clear z_M data in patch_grid + release slot to free-list
        # init_patch_node wrote trend + context into patch_grid[local_idx] at split time;
        # we must zero it out here to fully undo the z vector table modification.
        gs = self.model.grid_storage
        z_M_local = new_node_id - gs.num_base_nodes
        if 0 <= z_M_local < gs.patch_grid.shape[0]:
            with torch.no_grad():
                gs.patch_grid.data[z_M_local].zero_()
        self.manager.release_patch_node(new_node_id)

        # Step 3: Remove from split_history
        del self.split_history[(start, end)]

        return True

    def _update_zL_shadows(self, shadow_steps: int = 400):
        """
        finetune 后计算 shadow z_L：

          1. 从 warmup 前的快照出发（未被子块训练污染）
          2. 冻结 decoder，在当前 decoder 下用全块数据训练 z_L
          3. 将结果保存为 _zL_shadows
          4. 恢复 base_grid/patch_grid 中实际的（子块特化）z_L

        _zL_shadows 的两个用途：
          - report_split_results(): 用 shadow 评估父块质量，得到公平的
            "不分裂" 基准，与分裂后子块质量做对比，收益计算更准确
          - _rollback_split(): 回滚时用 shadow 还原 z_L，无污染
        """
        presplit  = getattr(self, '_presplit_stats', {})
        snapshots = getattr(self, '_zL_snapshots',  {})

        split_items = [
            (s, e, pre)
            for (s, e), pre in presplit.items()
            if (s, e) in self.split_history and (s, e) in snapshots
        ]
        if not split_items or shadow_steps <= 0:
            self._zL_shadows = {}
            return

        print(f"\n[Shadow] Computing shadow z_L for {len(split_items)} split blocks, "
              f"{shadow_steps} steps (decoder frozen)...")

        gs = self.model.grid_storage

        # ── 1. 保存当前（子块特化）z_L ──────────────────────────────────────
        child_zL = {}
        for s, e, pre in split_items:
            lid = pre['left_id']
            if lid < gs.num_base_nodes:
                child_zL[(s, e)] = gs.base_grid.data[lid].clone()
            else:
                child_zL[(s, e)] = gs.patch_grid.data[lid - gs.num_base_nodes].clone()

        # ── 2. 用快照覆盖 z_L（shadow 的训练起点）───────────────────────────
        for s, e, pre in split_items:
            lid  = pre['left_id']
            snap = snapshots[(s, e)]
            if lid < gs.num_base_nodes:
                gs.base_grid.data[lid] = snap.clone()
            else:
                gs.patch_grid.data[lid - gs.num_base_nodes] = snap.clone()

        # ── 3. 冻结 decoder，在父块数据上训练 z_L ───────────────────────────
        for param in self.model.decoder.parameters():
            param.requires_grad = False
        gs.base_grid.requires_grad_(True)

        shadow_optim = torch.optim.Adam(
            [gs.base_grid, gs.patch_grid],
            lr=self.args.learning_rate
        )
        self.model.train()
        import random as _random
        total_loss  = 0.0
        log_interval = max(1, shadow_steps // 5)

        for step in range(shadow_steps):
            s, e, pre = _random.choice(split_items)
            shadow_optim.zero_grad()
            l_ids  = torch.tensor([pre['left_id']],  device=self.device)
            r_ids  = torch.tensor([pre['right_id']], device=self.device)
            off    = torch.tensor([s % self.args.base_block_size], device=self.device)
            output = self.model.decode_batch(l_ids, r_ids, e - s, off).squeeze()
            target = self.raw_data[s:e]
            if target.device != output.device:
                target = target.to(output.device)
            loss = torch.nn.functional.mse_loss(output, target)
            if loss.requires_grad:
                loss.backward()
                shadow_optim.step()
            total_loss += loss.item()
            if (step + 1) % log_interval == 0:
                print(f"    [Shadow] step {step+1:>4}/{shadow_steps}  "
                      f"avg_loss={total_loss/(step+1):.6f}")

        print(f"    [Shadow] Done. avg_loss={total_loss/shadow_steps:.6f}")

        # ── 4. 保存 shadow z_L，同时评估 shadow 父块质量 ────────────────────
        self._zL_shadows = {}
        self.model.eval()
        with torch.no_grad():
            shadow_maes = []
            for s, e, pre in split_items:
                lid = pre['left_id']
                if lid < gs.num_base_nodes:
                    self._zL_shadows[(s, e)] = gs.base_grid.data[lid].clone()
                else:
                    self._zL_shadows[(s, e)] = gs.patch_grid.data[lid - gs.num_base_nodes].clone()
                out = self.model.decode_batch(
                    torch.tensor([lid],           device=self.device),
                    torch.tensor([pre['right_id']], device=self.device),
                    e - s,
                    torch.tensor([s % self.args.base_block_size], device=self.device)
                ).squeeze()
                tgt = self.raw_data[s:e]
                if tgt.device != out.device:
                    tgt = tgt.to(out.device)
                shadow_maes.append(torch.abs(out - tgt).mean().item())
            if shadow_maes:
                print(f"  [Shadow] Shadow parent quality: "
                      f"avgMAE={sum(shadow_maes)/len(shadow_maes):.5f}  "
                      f"maxMAE={max(shadow_maes):.5f}")

        # ── 5. 恢复子块特化 z_L ─────────────────────────────────────────────
        for s, e, pre in split_items:
            lid  = pre['left_id']
            cval = child_zL[(s, e)]
            if lid < gs.num_base_nodes:
                gs.base_grid.data[lid] = cval
            else:
                gs.patch_grid.data[lid - gs.num_base_nodes] = cval

        # 解冻 decoder
        for param in self.model.decoder.parameters():
            param.requires_grad = True

    def report_z_delta_stats(self):

        """

        报告所有分裂节点 z_mid 与其父节点 z_left 的逐维偏差。

        用于评估 Δz 差值编码方案（4-bit）是否可行。

        

        关键指标：

          max_dim_delta  : 单维最大绝对偏差（决定需要几 bit 才能表示 Δz）

          mean_dim_delta : 单维平均绝对偏差

          z_left_range   : 父节点的单维最大绝对值（作为参考量程）

          ratio          : max_dim_delta / z_left_range

                           < 0.25 → 4-bit 可行；> 0.5 → 与全量存储无异

        """

        if not self._parentage_map:

            print("[Z-Delta Stats] 暂无分裂记录")

            return

        

        gs = self.model.grid_storage

        by_level = {}

        dim_deltas_all = []  # 每个节点的 diff 向量，用于跨节点逐维统计

        

        with torch.no_grad():

            for new_id, (old_left_id, level) in self._parentage_map.items():

                z_new = gs.get_vectors(torch.tensor([new_id],      device=self.device))[0]

                z_old = gs.get_vectors(torch.tensor([old_left_id], device=self.device))[0]

                diff           = (z_new - z_old).abs()          # 逐维绝对偏差 [vec_dim]

                max_dim_delta  = diff.max().item()               # 最大维度偏差

                mean_dim_delta = diff.mean().item()              # 平均维度偏差

                z_left_range   = z_old.abs().max().item()        # 父节点单维最大绝对值

                ratio          = max_dim_delta / (z_left_range + 1e-8)

                if level not in by_level:

                    by_level[level] = []

                by_level[level].append({

                    'max_dim':   max_dim_delta,

                    'mean_dim':  mean_dim_delta,

                    'z_range':   z_left_range,

                    'ratio':     ratio,

                })

                dim_deltas_all.append(diff.cpu().numpy())

        

        print("\n[Z-Delta Stats]  Δz = z_mid_trained - z_left_init，逐维绝对偏差")

        print(f"    {'Level':>6} {'Count':>6} {'MaxDimΔ':>10} {'MeanDimΔ':>10} "

              f"{'z_leftRange':>12} {'MaxΔ/Range':>12}  判断")

        print("    " + "-" * 72)

        for lv in sorted(by_level.keys()):

            es = by_level[lv]

            max_d  = np.mean([e['max_dim']  for e in es])

            mean_d = np.mean([e['mean_dim'] for e in es])

            zr     = np.mean([e['z_range']  for e in es])

            ratio  = np.mean([e['ratio']    for e in es])

            verdict = '✓ 4-bit可行' if ratio < 0.25 else ('△ 勉强' if ratio < 0.5 else '✗ 不够')

            print(f"    {lv:>6} {len(es):>6} {max_d:>10.4f} {mean_d:>10.4f} "

                  f"{zr:>12.4f} {ratio:>12.4f}  {verdict}")

        

        all_entries = [e for v in by_level.values() for e in v]

        all_ratio   = np.mean([e['ratio'] for e in all_entries])

        all_max_d   = np.mean([e['max_dim'] for e in all_entries])

        print(f"    {'ALL':>6} {len(all_entries):>6} {all_max_d:>10.4f}" +

              f"{'':>10} {'':>12} {all_ratio:>12.4f}")

        print(f"    → 建议: ratio<0.25 可用4-bit; ratio>0.5 建议降vec_dim而非降bit宽")

        

        # ── 逐维度汇总（跨所有分裂节点）────────────────────────────────────────────────────────────

        if dim_deltas_all:

            dim_matrix = np.stack(dim_deltas_all, axis=0)  # [N_nodes, vec_dim]

            dim_mean   = dim_matrix.mean(axis=0)            # 每维平均 |Δz|

            dim_max    = dim_matrix.max(axis=0)             # 每维最大 |Δz|

            dim_rank   = np.argsort(dim_mean)[::-1]         # 按均值降序

            vec_dim    = dim_matrix.shape[1]

            global_mean = dim_mean.mean()

            print(f"\n    [Per-Dim Breakdown]  跨所有 {len(dim_deltas_all)} 个分裂节点")

            print(f"    （Δ = |z_mid_trained - z_left|，z_left 为被分裂块的左边界节点）")

            print(f"    {'Dim':>5} {'MeanΔ':>10} {'MaxΔ':>10}  活跃程度")

            print("    " + "-" * 42)

            for d in dim_rank:

                bar = '█' * int(dim_mean[d] / (dim_mean.max() + 1e-8) * 20)

                print(f"    {d:>5} {dim_mean[d]:>10.4f} {dim_max[d]:>10.4f}  {bar}")

            # 活跃维度：均值偏差 > 全局均值的 2 倍

            active  = [int(d) for d in range(vec_dim) if dim_mean[d] > global_mean * 2.0]

            stable  = [int(d) for d in range(vec_dim) if d not in active]

            active_pct = sum(dim_mean[d] for d in active) / (dim_mean.sum() + 1e-8) * 100

            print(f"\n    活跃维度（MeanΔ > 全局均值×2）: {active}  占总偏差 {active_pct:.1f}%")

            print(f"    稳定维度: {stable}")



    def test_reduced_storage(self, setting):

        """

        模拟三种降维存储方案，评估对重构质量的影响（不改变训练结果）。



        方案 A (dim0_only):

            只保留 Dim 0（取 z_child 训练值），其余 7 维继承父节点 z_parent。

            等价存储代价：1B / split_node。



        方案 B (n_keep_inherit):

            保留前 n_keep=vec_dim//2-1=3 维取 z_child 训练值，

            其余 5 维继承父节点 z_parent（不存储差值）。

            等价存储代价：3B / split_node。



        方案 C (n_keep_mean):

            保留前 n_keep=3 维取 z_child 训练值，

            其余 5 维用 z_child 后半段的均值广播填充（存 1 个均值）。

            等价存储代价：(3+1)B = 4B / split_node。

        """

        if not self._parentage_map:

            print("[ReducedStorage] 暂无分裂节点，跳过")

            return



        gs = self.model.grid_storage

        original_patch = gs.patch_grid.data.clone()

        vec_dim = gs.feature_dim

        n_keep  = vec_dim // 2 - 1   # = 3 for vec_dim=8



        # ── 打印各 level 的分裂节点数量 ─────────────────────────────────

        from collections import Counter

        level_counts = Counter(lv for (_, lv) in self._parentage_map.values())

        total_nodes  = len(self._parentage_map)

        print(f"\n[ReducedStorage] 共 {total_nodes} 个分裂节点  vec_dim={vec_dim}  n_keep={n_keep}")

        print(f"    {'Level':>6} {'Count':>8}  说明")

        for lv in sorted(level_counts):

            desc = f"{'base→L'+str(lv+1)}"

            print(f"    {lv:>6} {level_counts[lv]:>8}  {desc}")



        schemes = [

            ('dim0_only',       f'方案A: 只存Dim0              (1B/node)'),

            ('n_keep_inherit',  f'方案B: 前{n_keep}维child+后继承parent ({n_keep}B/node)'),

            ('n_keep_mean',     f'方案C: 前{n_keep}维child+后child均值  ({n_keep+1}B/node)'),

        ]



        for scheme, label in schemes:

            with torch.no_grad():
                gs.patch_grid.data.copy_(original_patch)  # 每次方案前先还原，防止上一方案污染 z_child

            with torch.no_grad():

                for new_id, (old_left_id, _) in self._parentage_map.items():

                    z_child  = gs.get_vectors(torch.tensor([new_id],      device=self.device))[0]

                    z_parent = gs.get_vectors(torch.tensor([old_left_id], device=self.device))[0]

                    reduced  = z_parent.clone()   # 默认：所有维度继承父节点



                    if scheme == 'dim0_only':

                        reduced[0] = z_child[0]                     # 仅覆盖 Dim 0

                    elif scheme == 'n_keep_inherit':

                        reduced[:n_keep] = z_child[:n_keep]         # 覆盖前 n_keep 维，其余仍为 parent

                    else:  # n_keep_mean

                        reduced[:n_keep] = z_child[:n_keep]         # 前 n_keep 维取 child 训练值

                        reduced[n_keep:] = z_child[n_keep:].mean()  # 后半段取 child 自身均值



                    gs.patch_grid.data[new_id - gs.num_base_nodes] = reduced



            print(f"\n    {label}")

            self.evaluate(setting)



        # 还原

        with torch.no_grad():

            gs.patch_grid.data.copy_(original_patch)

        print("[ReducedStorage] 原始向量已还原")

        # ── 逐维消融：向量级代理（快速，不运行完整 evaluate）────────────────────
        # 对每个维度 d，只存 z_child[d]，其余继承 z_parent
        # 指标：VecMAE = mean||z_reduced - z_child|| （越低 = 该维度信息量越大）
        print(f"\n[Per-Dim Ablation]  快速代理：只存该维度时 z_reduced vs z_child 的平均误差")
        print(f"    （越低 = 该维度越能代表 patch 节点的变化）")
        print(f"    {'Rank':>5} {'Dim':>5} {'VecMAE':>10} {'信息占比%':>12}")
        print("    " + "-" * 38)
        with torch.no_grad():
            _zc_list, _zp_list = [], []
            for _nid, (_lid, _lvl) in self._parentage_map.items():
                _zc_list.append(gs.get_vectors(torch.tensor([_nid],  device=self.device))[0])
                _zp_list.append(gs.get_vectors(torch.tensor([_lid],  device=self.device))[0])
            _zc = torch.stack(_zc_list, dim=0)  # [N, D]
            _zp = torch.stack(_zp_list, dim=0)  # [N, D]
            _baseline = (_zc - _zp).abs().mean().item()  # 全部继承 parent 时的误差
            _dim_maes = []
            for _d in range(vec_dim):
                _reduced = _zp.clone()
                _reduced[:, _d] = _zc[:, _d]
                _dim_maes.append((_reduced - _zc).abs().mean().item())
            _dim_order = np.argsort(_dim_maes)  # MAE 越低 = 该维度越重要
            print(f"    {'base':>5} {'all_parent':>5} {_baseline:>10.4f} (基准)")
            for _rank, _d in enumerate(_dim_order):
                _pct = (_baseline - _dim_maes[_d]) / (_baseline + 1e-8) * 100
                print(f"    {_rank+1:>5} {_d:>5} {_dim_maes[_d]:>10.4f} {_pct:>11.1f}%")


    # ------------------------------------------------------------------
    def z_span_refine(self, refine_steps: int):
        """
        Post-finetune decoder-frozen z-span refinement.

        Runs refine_steps steps on the FULL dataset with:
          - decoder FROZEN  (gradients only flow to z vectors)
          - ALL z vectors trainable (base_grid + patch_grid)
          - loss = MSE + warmup_span_weight * span_loss

        Unlike warmup (which only touches split-adjacent slots), this phase
        covers every block in the dataset, cumulatively pushing marginal
        5-bit blocks toward 4-bit over successive split rounds.
        """
        _span_weight = getattr(self.args, 'warmup_span_weight', 0.0)
        if refine_steps <= 0 or _span_weight <= 0:
            return

        _span_tau        = getattr(self.args, 'span_tau',        0.1)
        _span_gamma      = getattr(self.args, 'span_gamma',      0.95)
        _span_min_bits   = getattr(self.args, 'span_min_bits',   5)
        _span_max_margin = getattr(self.args, 'span_max_margin', 1.3)
        _refine_lr       = self.args.learning_rate * 0.1

        print(f"\n[ZRefine] Decoder-frozen z-span refinement: {refine_steps} steps, "
              f"lr={_refine_lr:.2e}")
        print(f"  Span: weight={_span_weight}, tau={_span_tau}, gamma={_span_gamma}, "
              f"min_bits={_span_min_bits}, max_margin={_span_max_margin}")

        # Freeze decoder
        for param in self.model.decoder.parameters():
            param.requires_grad = False
        self.model.grid_storage.base_grid.requires_grad_(True)

        refine_optim = torch.optim.Adam(
            [self.model.grid_storage.patch_grid,
             self.model.grid_storage.base_grid],
            lr=_refine_lr
        )

        refine_loader = self._get_train_loader(shuffle=True)
        self.model.train()

        step = 0
        total_loss      = 0.0
        window_sum      = 0.0
        window_span_sum = 0.0
        window_cnt      = 0
        prev_window_avg = None
        plateau_count   = 0
        loader_iter     = iter(refine_loader)

        plateau_patience  = getattr(self.args, 'warmup_plateau_patience',  20)
        plateau_min_delta = getattr(self.args, 'warmup_plateau_min_delta', 1e-5)
        plateau_window    = max(10, plateau_patience // 2)
        stop_reason       = f"max_steps={refine_steps}"

        while step < refine_steps:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(refine_loader)
                batch = next(loader_iter)

            refine_optim.zero_grad()
            loss, _, _ = self._process_one_batch(batch)
            span_loss = self._compute_warmup_span_loss(
                batch, _span_tau, _span_gamma, _span_min_bits, _span_max_margin)
            span_val = span_loss.item()
            if span_loss.requires_grad:
                loss = loss + _span_weight * span_loss
            if not loss.requires_grad:
                step += 1
                continue
            loss.backward()
            refine_optim.step()

            loss_val         = loss.item()
            total_loss      += loss_val
            window_sum      += loss_val
            window_span_sum += span_val
            window_cnt      += 1
            step            += 1

            if window_cnt >= plateau_window:
                cur_avg  = window_sum / window_cnt
                span_win = window_span_sum / window_cnt
                print(f"    [ZRefine] step {step:>5}  window_avg={cur_avg:.6f}  "
                      f"total_avg={total_loss/step:.6f}  span_win={span_win:.5f}")
                if prev_window_avg is not None:
                    improvement = prev_window_avg - cur_avg
                    if improvement < plateau_min_delta:
                        plateau_count += 1
                        if plateau_count >= 2:
                            stop_reason = (f"plateau ({prev_window_avg:.6f}"
                                           f"→{cur_avg:.6f}, Δ={improvement:.2e})")
                            break
                    else:
                        plateau_count = 0
                prev_window_avg = cur_avg
                window_sum      = 0.0
                window_span_sum = 0.0
                window_cnt      = 0

        avg_loss = total_loss / max(step, 1)
        print(f"    [ZRefine] Done at step {step}. avg_loss={avg_loss:.6f}  "
              f"stop={stop_reason}")

        # Restore decoder
        for param in self.model.decoder.parameters():
            param.requires_grad = True
        self.model.grid_storage.base_grid.requires_grad_(True)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    def targeted_span_training(self, error_threshold: float, error_mode: str,
                               margin_hi: float = 1.1,
                               max_steps: int = 300,
                               span_weight: float = 0.5,
                               min_bits: int = 5,
                               max_mae_ratio: float = 1.2):
        """
        边缘块定向训练：扫描所有块，找出 margin 刚好在 (1.0, margin_hi] 的块，
        对它们专门做 decoder-frozen 定向优化，推动 span 降过阈值从而少用一个 bit。

        原理：margin = hard_span / target_span，target_span = 2ε·2^(b-1)
        margin 在 (1.0, 1.1] 的块只需 span 缩减 <10% 即可降一档。
        每块节省约 block_len/8 bytes。

        Args:
            error_threshold: 误差阈值
            error_mode:       'relative' or 'absolute'
            margin_hi:        只处理 margin <= margin_hi 的块 (default 1.1)
            max_steps:        每块最大优化步数 (default 300)
            span_weight:      MSE loss 中 span hinge 的权重 (default 0.5)
            min_bits:         只处理 bits >= min_bits 的块 (default 5)
            max_mae_ratio:    MAE 增幅超过此倍数时回滚该块 (default 1.2)
        """
        _tau       = getattr(self.args, 'span_tau',   0.1)
        _gamma     = getattr(self.args, 'span_gamma', 0.95)
        _refine_lr = self.args.learning_rate * 0.05

        scaler   = getattr(self.data_loader, 'scaler', None)
        std_val  = (scaler.std.item()  if hasattr(scaler.std,  'item') else float(scaler.std))  if scaler is not None else 1.0
        mean_val = (scaler.mean.item() if hasattr(scaler.mean, 'item') else float(scaler.mean)) if scaler is not None else 0.0

        if error_mode == 'absolute':
            norm_threshold = error_threshold / std_val
        else:
            norm_threshold = None

        # ── Phase 1: 扫描所有块，收集边缘块 ──────────────────────────────────
        all_blocks = self.manager.get_all_unique_blocks()
        self.model.eval()
        self._refresh_quant_params()

        marginal = []   # list of dicts

        with torch.no_grad():
            for start_time, end_time, left_id, right_id, level_code in all_blocks:
                bl  = end_time - start_time
                off = start_time % self.args.base_block_size

                pred = self.model.decode_batch(
                    torch.tensor([left_id],  device=self.device),
                    torch.tensor([right_id], device=self.device),
                    bl,
                    torch.tensor([off], device=self.device)
                ).squeeze()

                true     = self.raw_data[start_time:end_time]
                residual = true - pred
                blk_span = (residual.max() - residual.min()).item()

                if error_mode == 'absolute':
                    blk_eps = norm_threshold
                else:
                    true_orig = true * std_val + mean_val
                    min_denom = torch.clamp(torch.abs(true_orig), min=1.0).min().item()
                    blk_eps   = error_threshold * min_denom / std_val

                blk_bits = FallbackDict.compute_bits(blk_span, blk_eps)
                if blk_bits < min_bits:
                    continue

                target_span = 2.0 * blk_eps * float(2 ** (blk_bits - 1))
                if target_span <= 0:
                    continue

                margin = blk_span / target_span
                if margin <= 1.0 or margin > margin_hi:
                    continue

                init_mae = (pred - true).abs().mean().item()
                marginal.append({
                    'start':       start_time,
                    'end':         end_time,
                    'left_id':     left_id,
                    'right_id':    right_id,
                    'bl':          bl,
                    'off':         off,
                    'bits':        blk_bits,
                    'target_span': target_span,
                    'blk_eps':     blk_eps,
                    'margin':      margin,
                    'init_mae':    init_mae,
                })

        if not marginal:
            print(f"\n[TargetedSpan] No marginal blocks found "
                  f"(bits>={min_bits}, margin in (1.0, {margin_hi}]).")
            return

        # Sort by margin ascending: easiest wins first
        marginal.sort(key=lambda x: x['margin'])

        print(f"\n[TargetedSpan] Found {len(marginal)} marginal blocks "
              f"(bits>={min_bits}, margin in (1.0, {margin_hi}]).")
        print(f"  lr={_refine_lr:.2e}, span_weight={span_weight}, "
              f"max_steps={max_steps}, tau={_tau}")

        # ── Phase 2: Decoder frozen, per-block targeted optimization ─────────
        for param in self.model.decoder.parameters():
            param.requires_grad = False
        self.model.grid_storage.base_grid.requires_grad_(True)

        gs = self.model.grid_storage
        self.model.train()

        success_count   = 0
        total_bit_bytes = 0.0

        for info in marginal:
            start    = info['start']
            end      = info['end']
            lid      = info['left_id']
            rid      = info['right_id']
            bl       = info['bl']
            off      = info['off']
            bits     = info['bits']
            tgt_span = info['target_span']
            eps      = info['blk_eps']
            init_mae = info['init_mae']

            true     = self.raw_data[start:end]
            off_t    = torch.tensor([off],  device=self.device)
            lid_t    = torch.tensor([lid],  device=self.device)
            rid_t    = torch.tensor([rid],  device=self.device)

            # Save z vectors for potential rollback
            num_base = gs.num_base_nodes
            def _save(nid):
                if nid < num_base:
                    return gs.base_grid.data[nid].clone()
                return gs.patch_grid.data[nid - num_base].clone()
            def _restore(nid, saved):
                if nid < num_base:
                    gs.base_grid.data[nid].copy_(saved)
                else:
                    gs.patch_grid.data[nid - num_base].copy_(saved)

            saved_l = _save(lid)
            saved_r = _save(rid)

            # Fresh per-block Adam (zero momentum avoids cross-block interference)
            blk_optim = torch.optim.Adam(
                [gs.base_grid, gs.patch_grid], lr=_refine_lr)

            achieved = False
            for _ in range(max_steps):
                blk_optim.zero_grad()
                pred     = self.model.decode_batch(lid_t, rid_t, bl, off_t).squeeze()
                residual = pred - true

                with torch.no_grad():
                    hard_span = (residual.max() - residual.min()).item()
                if hard_span < tgt_span:
                    achieved = True
                    break

                # MSE keeps reconstruction quality
                mse_loss = (residual ** 2).mean()
                # Soft-span hinge pushes span below target
                soft_max    =  _tau * torch.logsumexp( residual / _tau, dim=0)
                soft_min    = -_tau * torch.logsumexp(-residual / _tau, dim=0)
                soft_span_v = soft_max - soft_min
                excess      = (soft_span_v - _gamma * tgt_span) / tgt_span
                span_loss   = excess.clamp(min=0.0) ** 2

                (mse_loss + span_weight * span_loss).backward()
                blk_optim.step()

            # Verify + MAE safety rollback
            with torch.no_grad():
                final_pred  = self.model.decode_batch(lid_t, rid_t, bl, off_t).squeeze()
                final_res   = final_pred - true
                final_span  = (final_res.max() - final_res.min()).item()
                final_bits  = FallbackDict.compute_bits(final_span, eps)
                final_mae   = final_res.abs().mean().item()

            if final_mae > init_mae * max_mae_ratio:
                # Quality degradation too large → rollback
                _restore(lid, saved_l)
                _restore(rid, saved_r)
                status = f"✗ rollback (MAE {init_mae:.5f}→{final_mae:.5f})"
            elif final_bits < bits:
                success_count   += 1
                byte_saving = FallbackDict.estimate_bitwidth_cost(
                    final_span, bl, eps) - FallbackDict.estimate_bitwidth_cost(
                    info['margin'] * tgt_span, bl, eps)
                total_bit_bytes += abs(byte_saving)
                status = (f"✓ {bits}→{final_bits}-bit  "
                          f"margin {info['margin']:.3f}→{final_span/tgt_span:.3f}  "
                          f"MAE {init_mae:.5f}→{final_mae:.5f}")
            else:
                status = (f"✗ still {bits}-bit  "
                          f"margin {info['margin']:.3f}→{final_span/tgt_span:.3f}")

            print(f"  [{start}:{end}] len={bl} {status}")

        # Restore decoder
        for param in self.model.decoder.parameters():
            param.requires_grad = True
        self.model.grid_storage.base_grid.requires_grad_(True)

        print(f"\n[TargetedSpan] Done: {success_count}/{len(marginal)} blocks reduced. "
              f"Estimated savings ≈ {total_bit_bytes:.0f} bytes.")
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    def _compute_warmup_span_loss(self, batch, tau: float, gamma: float,
                                  min_bits: int, max_margin: float) -> torch.Tensor:
        """
        Per-block span hinge loss for warmup.

        For each unique (left_id, right_id, block_len) in the batch:
          1. Reconstruct full block → residual r = pred - truth
          2. hard_span (detached) → current bits b
          3. target_span T = 2ε·2^(b-1)   (max span fitting b-1 bits)
          4. Active mask: bits >= min_bits  AND  1.0 <= margin <= max_margin
          5. soft_span via logsumexp; L = hinge((soft-γT)/T)², weighted by block_len

        Returns a scalar tensor (0.0 if no active blocks in this batch).
        """
        left_ids, right_ids, ground_truth, mask, offsets, block_lens = batch
        dev = self.device

        def _t(x):
            return x.to(dev) if isinstance(x, torch.Tensor) else torch.tensor(x, device=dev)

        left_ids     = _t(left_ids)
        right_ids    = _t(right_ids)
        ground_truth = _t(ground_truth).float()
        block_lens   = _t(block_lens)
        offsets      = _t(offsets)

        error_threshold = getattr(self.args, 'error_threshold', 0.02)
        error_mode      = getattr(self.args, 'error_mode', 'relative')
        scaler   = getattr(self.data_loader, 'scaler', None)
        std_val  = (scaler.std.item()  if hasattr(scaler.std,  'item') else float(scaler.std))  if scaler is not None else 1.0
        mean_val = (scaler.mean.item() if hasattr(scaler.mean, 'item') else float(scaler.mean)) if scaler is not None else 0.0

        # Deduplicate: each unique (left_id, right_id, block_len) = one block
        keys = torch.stack([left_ids, right_ids, block_lens.long()], dim=1)
        unique_keys, inv_idx = torch.unique(keys, dim=0, return_inverse=True)

        span_terms   = []
        total_weight = 0.0

        for k in range(len(unique_keys)):
            lid = unique_keys[k, 0].item()
            rid = unique_keys[k, 1].item()
            bl  = int(unique_keys[k, 2].item())

            # All slots of the same block share the same ground_truth – take first
            slot_idx = (inv_idx == k).nonzero(as_tuple=True)[0][0]
            gt_k  = ground_truth[slot_idx, :bl]          # [bl] normalised
            off_k = offsets[slot_idx : slot_idx + 1]     # [1]

            # Full-block decode (differentiable through z vectors)
            pred_k     = self.model.decode_batch(
                torch.tensor([lid], device=dev),
                torch.tensor([rid], device=dev),
                bl, off_k).squeeze()                     # [bl]
            residual_k = pred_k - gt_k                   # [bl]

            # Hard span & bits for active mask (detached – no grad through mask)
            with torch.no_grad():
                hard_span = (residual_k.max() - residual_k.min()).item()

            # Per-block epsilon in normalised space
            if error_mode == 'absolute':
                blk_eps = error_threshold / std_val
            else:
                gt_orig   = gt_k.detach() * std_val + mean_val
                min_denom = max(gt_orig.abs().min().item(), 1.0)
                blk_eps   = error_threshold * min_denom / std_val

            blk_bits = FallbackDict.compute_bits(hard_span, blk_eps)
            if blk_bits < min_bits:
                continue

            target_span = 2.0 * blk_eps * float(2 ** (blk_bits - 1))
            if target_span <= 0:
                continue

            margin = hard_span / target_span
            if margin < 1.0 or margin > max_margin:
                continue

            # Soft span via logsumexp (smooth differentiable max/min)
            soft_max    =  tau * torch.logsumexp( residual_k / tau, dim=0)
            soft_min    = -tau * torch.logsumexp(-residual_k / tau, dim=0)
            soft_span_k = soft_max - soft_min

            # One-sided hinge: penalise only when soft_span > γ·target_span
            excess  = (soft_span_k - gamma * target_span) / target_span
            loss_k  = excess.clamp(min=0.0) ** 2

            span_terms.append(loss_k * bl)
            total_weight += bl

        if span_terms and total_weight > 0:
            return torch.stack(span_terms).sum() / total_weight
        return torch.zeros(1, device=dev).squeeze()
    # ------------------------------------------------------------------

    def warmup_patch_nodes(self, warmup_steps: int = 50):

        """

        分裂后预热阶段：冻结 decoder，优化所有分裂相关 z 向量。

        

        分裂后两类向量需要适配：

          - z_M（新节点，patch_grid）：冷启动，从头收敛

          - z_L（左子块的唯一向量 = left_id）：其系数为全块训练，

            现在需要重新编码左半块内容

          注：z_R 是右侧相邻块的向量（right_id），decoder 不使用 right_id，

              所以 z_R 梯度为零，即使包含在优化器里也不会被更新。

        

        relevant_indices 包含左子块和右子块（通过 z_M 的 left/right_id 位置查找），

        优化器覆盖 patch_grid + base_grid（z_L 可能在 base_grid 中）。

        

        应在 adaptive_split 之后、正式 finetune 之前调用。

        

        Args:

            warmup_steps: 预热步数，建议 30-100

        """

        if warmup_steps <= 0:

            return

        

        print(f"\n[Warmup] Freezing decoder, optimizing all split-adjacent z vectors for {warmup_steps} steps...")

        

        # 找出两类子块 slot：

        #   右子块：left_id  in split_node_ids（z_M 作为起点）

        #   左子块：right_id in split_node_ids（z_M 作为终点）

        split_node_ids = getattr(self, '_last_split_node_ids', set())

        relevant_indices = []

        if split_node_ids:

            for i in range(self.manager.num_slots):

                entry = self.manager.index_table[i]

                if entry.left_id in split_node_ids or entry.right_id in split_node_ids:

                    relevant_indices.append(i)

        

        if not relevant_indices:

            print("  [Warmup] No new split nodes found, skipping warmup.")

            return

        

        print(f"  [Warmup] {len(relevant_indices)} slots (both children) involve new split nodes "

              f"(out of {self.manager.num_slots} total)")

        

        # 冻结 decoder 所有参数（共享权重不参与 warmup）

        for param in self.model.decoder.parameters():

            param.requires_grad = False

        

        # base_grid 解冻：z_L/z_R（父块边界向量）需要重新适配子块尺度

        # 新 Adam 实例从零初始化 moment，对不涉及 split 的向量梯度为零 → 实际不更新

        self.model.grid_storage.base_grid.requires_grad_(True)

        

        # 同时优化 patch_grid（z_M）和 base_grid（z_L, z_R）

        warmup_optim = torch.optim.Adam(

            [self.model.grid_storage.patch_grid,

             self.model.grid_storage.base_grid],

            lr=self.args.learning_rate

        )

        

        # 只从新分裂 slot 中采样（ground_truth 已 padding，collate 无问题）

        # num_workers=0：避免 Windows 多进程冷启动问题

        from torch.utils.data import SubsetRandomSampler

        warmup_loader = DataLoader(

            self.train_dataset,

            batch_size=min(self.args.batch_size, len(relevant_indices)),

            sampler=SubsetRandomSampler(relevant_indices),

            num_workers=0,

        )

        # ── Pre-warmup 快照：在 z_L 被 warmup 改动前，先测量父块全长重建质量 ──
        self._pre_warmup_parent_qual = {}
        presplit = getattr(self, '_presplit_stats', {})
        self.model.eval()
        with torch.no_grad():
            for (ps, pe), ppre in presplit.items():
                if (ps, pe) not in self.split_history:
                    continue
                try:
                    pl_ids = torch.tensor([ppre['left_id']],  device=self.device)
                    pr_ids = torch.tensor([ppre['right_id']], device=self.device)
                    p_off  = torch.tensor([ps % self.args.base_block_size], device=self.device)
                    p_out  = self.model.decode_batch(pl_ids, pr_ids, pe - ps, p_off).squeeze()
                    p_err  = torch.abs(p_out - self.raw_data[ps:pe])
                    self._pre_warmup_parent_qual[(ps, pe)] = {
                        'mae':     p_err.mean().item(),
                        'max_err': p_err.max().item(),
                    }
                except Exception:
                    pass
        if self._pre_warmup_parent_qual:
            snap_maes = [v['mae']     for v in self._pre_warmup_parent_qual.values()]
            snap_maxe = [v['max_err'] for v in self._pre_warmup_parent_qual.values()]
            print(f"  [Warmup] Pre-warmup parent@full-block ({len(snap_maes)} splits): "
                  f"avgMAE={np.mean(snap_maes):.5f}  avgMaxErr={np.mean(snap_maxe):.5f}  "
                  f"maxMaxErr={np.max(snap_maxe):.5f}")

        # z_L 快照：warmup 前保存父块 z_L，作为 shadow z_L 的训练起点。
        # finetune 后由 _update_zL_shadows() 从快照出发、在当前 decoder 下
        # 训练父块，得到 shadow z_L，用于精确收益对比和回滚时的无污染还原。
        self._zL_snapshots = {}
        _snap_gs  = self.model.grid_storage
        _snap_pre = getattr(self, '_presplit_stats', {})
        for (_sp, _ep), _ppre in _snap_pre.items():
            if (_sp, _ep) not in self.split_history:
                continue
            _lid = _ppre['left_id']
            if _lid < _snap_gs.num_base_nodes:
                self._zL_snapshots[(_sp, _ep)] = _snap_gs.base_grid.data[_lid].clone()
            else:
                _local = _lid - _snap_gs.num_base_nodes
                self._zL_snapshots[(_sp, _ep)] = _snap_gs.patch_grid.data[_local].clone()
        if self._zL_snapshots:
            print(f"  [Warmup] Saved z_L snapshots for {len(self._zL_snapshots)} parent blocks.")

        self.model.train()

        step = 0
        total_loss  = 0.0
        window_sum  = 0.0
        window_cnt  = 0
        prev_window_avg = None
        plateau_count   = 0
        loader_iter = iter(warmup_loader)

        plateau_patience  = getattr(self.args, 'warmup_plateau_patience',  20)
        plateau_min_delta = getattr(self.args, 'warmup_plateau_min_delta', 1e-5)
        plateau_window    = max(10, plateau_patience // 2)
        log_interval      = plateau_window

        # Span loss config
        _span_weight    = getattr(self.args, 'warmup_span_weight', 0.0)
        _span_tau       = getattr(self.args, 'span_tau',       0.1)
        _span_gamma     = getattr(self.args, 'span_gamma',     0.95)
        _span_min_bits  = getattr(self.args, 'span_min_bits',  6)
        _span_max_margin= getattr(self.args, 'span_max_margin',1.3)
        if _span_weight > 0:
            print(f"  [Warmup] Span loss: weight={_span_weight}, tau={_span_tau}, "
                  f"gamma={_span_gamma}, min_bits={_span_min_bits}, max_margin={_span_max_margin}")

        stop_reason = f"max_steps={warmup_steps}"
        total_span_sum  = 0.0

        while step < warmup_steps:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(warmup_loader)
                batch = next(loader_iter)

            warmup_optim.zero_grad()
            loss, _, _ = self._process_one_batch(batch)
            span_val = 0.0
            if _span_weight > 0:
                span_loss = self._compute_warmup_span_loss(
                    batch, _span_tau, _span_gamma, _span_min_bits, _span_max_margin)
                if span_loss.requires_grad:
                    loss = loss + _span_weight * span_loss
                span_val = span_loss.item()
            if not loss.requires_grad:
                step += 1
                continue
            loss.backward()
            warmup_optim.step()
            loss_val    = loss.item()
            total_span_sum += span_val
            total_loss  += loss_val
            window_sum  += loss_val
            window_cnt  += 1
            step        += 1

            if window_cnt >= plateau_window:
                cur_avg = window_sum / window_cnt
                if step >= log_interval:
                    _span_info = (f"  span_avg={total_span_sum/step:.5f}"
                                  if _span_weight > 0 else "")
                    print(f"    [Warmup] step {step:>5}  window_avg={cur_avg:.6f}  "
                          f"total_avg={total_loss/step:.6f}{_span_info}")
                if prev_window_avg is not None:
                    improvement = prev_window_avg - cur_avg
                    if improvement < plateau_min_delta:
                        plateau_count += 1
                        if plateau_count >= 2:
                            stop_reason = (f"plateau (window_avg {prev_window_avg:.6f}"
                                           f"→{cur_avg:.6f}, Δ={improvement:.2e})")
                            break
                    else:
                        plateau_count = 0
                prev_window_avg = cur_avg
                window_sum = 0.0
                window_cnt = 0

        avg_loss = total_loss / max(step, 1)
        print(f"    [Warmup] Done at step {step}. avg_loss={avg_loss:.6f}  stop={stop_reason}")

        # ── Post-warmup 块级评估：父块 vs 子块，含 max error ──────────────────
        self.model.eval()
        post_rows = []
        with torch.no_grad():
            for (ps, pe), ppre in presplit.items():
                if (ps, pe) not in self.split_history:
                    continue
                split_t, new_node_id = self.split_history[(ps, pe)]
                left_id, right_id = ppre['left_id'], ppre['right_id']
                llen, rlen = split_t - ps, pe - split_t
                if llen <= 0 or rlen <= 0:
                    continue
                try:
                    # 父块 (post-warmup, z_L trained as left-child)
                    p_out = self.model.decode_batch(
                        torch.tensor([left_id],  device=self.device),
                        torch.tensor([right_id], device=self.device),
                        pe - ps,
                        torch.tensor([ps % self.args.base_block_size], device=self.device)
                    ).squeeze()
                    p_err = torch.abs(p_out - self.raw_data[ps:pe])
                    # 左子块
                    l_out = self.model.decode_batch(
                        torch.tensor([left_id],     device=self.device),
                        torch.tensor([new_node_id], device=self.device),
                        llen,
                        torch.tensor([ps % self.args.base_block_size], device=self.device)
                    ).squeeze()
                    l_err = torch.abs(l_out - self.raw_data[ps:split_t])
                    # 右子块
                    r_out = self.model.decode_batch(
                        torch.tensor([new_node_id], device=self.device),
                        torch.tensor([right_id],    device=self.device),
                        rlen,
                        torch.tensor([split_t % self.args.base_block_size], device=self.device)
                    ).squeeze()
                    r_err = torch.abs(r_out - self.raw_data[split_t:pe])
                    child_mae     = (l_err.mean() * llen + r_err.mean() * rlen).item() / (llen + rlen)
                    child_max_err = max(l_err.max().item(), r_err.max().item())
                    snap = self._pre_warmup_parent_qual.get((ps, pe), {})
                    post_rows.append({
                        'start': ps, 'end': pe,
                        'snap_mae':     snap.get('mae',     float('nan')),
                        'snap_max_err': snap.get('max_err', float('nan')),
                        'post_par_mae':     p_err.mean().item(),
                        'post_par_max_err': p_err.max().item(),
                        'child_mae':     child_mae,
                        'child_max_err': child_max_err,
                    })
                except Exception:
                    continue
        if post_rows:
            def _nm(lst): return np.nanmean(lst)
            sm  = [r['snap_mae']         for r in post_rows]
            sme = [r['snap_max_err']     for r in post_rows]
            pm  = [r['post_par_mae']     for r in post_rows]
            pme = [r['post_par_max_err'] for r in post_rows]
            cm  = [r['child_mae']        for r in post_rows]
            cme = [r['child_max_err']    for r in post_rows]
            spl_mae_gain = (_nm(sm)  - _nm(cm))  / (_nm(sm)  + 1e-9) * 100
            spl_max_gain = (_nm(sme) - _nm(cme)) / (_nm(sme) + 1e-9) * 100
            zl_adapt_mae = (_nm(sm)  - _nm(pm))  / (_nm(sm)  + 1e-9) * 100  # diagnostic
            zl_adapt_max = (_nm(sme) - _nm(pme)) / (_nm(sme) + 1e-9) * 100  # diagnostic
            print(f"  [Warmup] Post-warmup block eval ({len(post_rows)} splits):")
            print(f"    {'Metric':<10} {'ParentPre':>10} {'ParentPost':>11} {'Children':>10}")
            print(f"    {'MAE':<10} {_nm(sm):>10.5f} {_nm(pm):>11.5f} {_nm(cm):>10.5f}")
            print(f"    {'MaxErr':<10} {_nm(sme):>10.5f} {_nm(pme):>11.5f} {_nm(cme):>10.5f}")
            print(f"    Split benefit (ParentPre→Children): MAE {spl_mae_gain:+.1f}%  MaxErr {spl_max_gain:+.1f}%")
            print(f"    [diag] z_L self-adapt during warmup: MAE {zl_adapt_mae:+.1f}%  MaxErr {zl_adapt_max:+.1f}%"
                  f"  (warmup is part of split op; no-split = no warmup)")
        self._post_warmup_rows = post_rows  # 供 report_split_results 引用

        # 恢复所有参数可训练
        for param in self.model.decoder.parameters():
            param.requires_grad = True
        self.model.grid_storage.base_grid.requires_grad_(True)



    def quantize_grid(self):

        """

        对 Grid 进行量化并冻结。

        

        应在训练完成后、最终评估前调用。

        """

        if hasattr(self.grid_storage, '_quantization_enabled') and self.grid_storage._quantization_enabled:

            self.grid_storage.quantize_and_freeze()

            

            # 打印量化后的内存使用情况

            quant_mem = self.grid_storage.get_quantized_memory_usage()

            print(f"\n[Quantization] Memory usage:")

            print(f"    Original: {quant_mem['original_bytes'] / 1024:.2f} KB")

            print(f"    Quantized: {quant_mem['quantized_bytes'] / 1024:.2f} KB")

            print(f"    Compression ratio: {quant_mem['compression_ratio']:.2f}x")

        else:

            print("[Quantization] Warning: quantization not enabled, skipping")

    

    def run_vector_gc(self, error_threshold: float = 0.10, error_mode: str = 'relative') -> dict:

        """

        执行 Vector GC：基于 RDO 的 DP 剪枝（纯只读分析）。

        

        利用前向分裂日志 (split_history) 构建隐式二叉树，

        后序遍历对比三份账单（保留分裂 / 合并BITWIDTH / 合并RAW），

        统计可回收的冗余节点数量和节省字节数。

        

        不修改 index_table / split_history / patch_grid。

        结果存储在 self.gc_result 供下游 final_evaluation 压缩比计算使用。

        

        Args:

            error_threshold: 坏点误差阈值

            error_mode: 'relative' 或 'absolute'

            

        Returns:

            GC 统计结果字典

        """

        from models.vector_gc import VectorGC

        

        # 获取 scaler 参数

        scaler = self.data_loader.scaler

        if scaler is not None:

            std_val = scaler.std.item() if hasattr(scaler.std, 'item') else float(scaler.std)

            mean_val = scaler.mean.item() if hasattr(scaler.mean, 'item') else float(scaler.mean)

        else:

            std_val, mean_val = 1.0, 0.0

        

        # 确保 eval 模式 + 量化感知

        self.model.eval()

        self._refresh_quant_params()

        

        # 解包装 DataParallel（decode_single 是自定义方法，DataParallel 不转发）

        model = self.model.module if isinstance(self.model, DataParallel) else self.model

        

        gc = VectorGC(

            manager=self.manager,

            model=model,

            grid_storage=self.grid_storage,

            raw_data=self.raw_data,

            split_history=self.split_history,

            error_threshold=error_threshold,

            error_mode=error_mode,

            std_val=std_val,

            mean_val=mean_val,

        )

        

        result = gc.run()

        

        # 存储 GC 结果供下游压缩比计算使用

        self.gc_result = result

        

        # GC 完成后销毁账本（纯只读分析已完成，账本不再需要）

        self.split_history.clear()

        

        return result

    

    def final_evaluation(self, error_threshold: float = 0.10, error_mode: str = 'relative') -> dict:

        """

        最终评估：遍历所有唯一块，计算详细的重构质量报告。

        

        支持两种误差模式：

        - relative: 百分比误差 |error|/|true_orig| >= threshold（论文标准）

        - absolute: 绝对误差（原始空间） >= threshold

        采用动态位宽量化（EBDQ）代价模型：span → bits → bitwidth_cost。

        包含 Bitwidth 分布统计和理论压缩比估算。

        

        Args:

            error_threshold: 坏点误差阈值。relative模式下为比例（0.10=10%），absolute模式下为原始单位

            error_mode: 'relative'（百分比误差）或 'absolute'（绝对误差）

            

        Returns:

            dict: 包含关键指标（compliance_rate, mae, bitwidth_stats 等）

        """

        self.model.eval()

        self._refresh_quant_params()

        

        # 获取 scaler 参数

        scaler = self.data_loader.scaler

        if scaler is not None:

            std_val = scaler.std.item() if hasattr(scaler.std, 'item') else float(scaler.std)

            mean_val = scaler.mean.item() if hasattr(scaler.mean, 'item') else float(scaler.mean)

        else:

            std_val, mean_val = 1.0, 0.0

        

        if error_mode == 'absolute':

            norm_threshold = error_threshold / std_val

            thresh_desc = f"{error_threshold} (orig) / {norm_threshold:.4f} (norm), mode=absolute"

        else:

            norm_threshold = None  # relative mode uses per-point threshold

            thresh_desc = f"{error_threshold*100:.1f}% relative error"

        

        all_blocks = self.manager.get_all_unique_blocks()

        total_length = self.manager.total_length

        

        # 用于存储逐点误差

        all_abs_errors = torch.zeros(total_length, device=self.device)

        all_rel_errors = torch.zeros(total_length, device=self.device)

        

        total_blocks = len(all_blocks)

        block_stats = []  # 每个块的统计信息
        # Sub-group sweep accumulators（在 block 循环内计算，避免重存 residual）
        _sweep_bw = {1: 0, 2: 0, 4: 0, 8: 0}  # groups -> accumulated bw bytes

        

        print("\n" + "=" * 70)

        print("FINAL EVALUATION REPORT")

        print("=" * 70)

        print(f"Error Threshold: {thresh_desc}, Tier logic: pure bit-cost driven")

        print(f"Total Blocks: {total_blocks}")

        print(f"Total Data Points: {total_length}")

        print("-" * 70)

        

        # 按块长度分组，每组独立解码

        from collections import defaultdict

        len_groups = defaultdict(list)

        for i, (start_time, end_time, left_id, right_id, level_code) in enumerate(all_blocks):

            block_len = end_time - start_time

            len_groups[block_len].append((i, start_time, end_time, left_id, right_id, level_code))

        

        eval_bs = getattr(self.args, 'eval_batch_size', 256)

        with torch.no_grad():

            for block_len, group in len_groups.items():

                for chunk_start in range(0, len(group), eval_bs):

                    chunk = group[chunk_start:chunk_start + eval_bs]

                    indices, starts, ends, lefts, rights, levels = zip(*chunk)

                    

                    left_ids = torch.tensor(lefts, device=self.device)

                    right_ids = torch.tensor(rights, device=self.device)

                    blk_offsets = torch.tensor(

                        [s % self.args.base_block_size for s in starts], device=self.device)

                    

                    # 动态长度解码（传入物理偏移用于全局时间坐标）

                    outputs = self.model.decode_batch(left_ids, right_ids, block_len, blk_offsets)

                    outputs = outputs.squeeze(1)  # [chunk_size, block_len]

                    

                    for j, (start_time, end_time, level_code) in enumerate(zip(starts, ends, levels)):

                        output = outputs[j]

                        

                        # 真实值

                        true = self.raw_data[start_time:end_time]

                        

                        # 计算绝对误差（归一化空间）

                        abs_error = torch.abs(output - true)

                        all_abs_errors[start_time:end_time] = abs_error

                        

                        # 计算相对误差（原始空间）

                        true_orig = true * std_val + mean_val

                        error_orig = abs_error * std_val

                        rel_error = error_orig / torch.clamp(torch.abs(true_orig), min=1.0)

                        all_rel_errors[start_time:end_time] = rel_error

                        

                        # 块级统计

                        block_max_error = abs_error.max().item()

                        block_mae = abs_error.mean().item()

                        block_max_rel_error = rel_error.max().item()

                        block_mean_rel_error = rel_error.mean().item()

                        

                        # 坏点判定（根据误差模式）

                        if error_mode == 'relative':

                            block_bad_points = (rel_error >= error_threshold).sum().item()

                        else:

                            block_bad_points = (abs_error >= norm_threshold).sum().item()

                        

                        # 动态位宽代价模型

                        residual = true - output  # 带符号残差（归一化空间）

                        blk_span = (residual.max() - residual.min()).item()

                        

                        # 计算 per-block epsilon

                        if error_mode == 'absolute':

                            blk_eps = norm_threshold

                        else:

                            min_denom = torch.clamp(torch.abs(true_orig), min=1.0).min().item()

                            blk_eps = error_threshold * min_denom / std_val

                        

                        blk_bits = FallbackDict.compute_bits(blk_span, blk_eps)

                        bitwidth_cost = FallbackDict.estimate_bitwidth_cost(blk_span, block_len, blk_eps)

                        _rg = getattr(self.args, 'residual_groups', 1)
                        bitwidth_cost_grouped = (FallbackDict.estimate_bitwidth_cost_grouped(residual, blk_eps, _rg)
                                                 if _rg > 1 else bitwidth_cost)
                        # Sub-group sweep: compute all candidate group counts in-place
                        _sweep_bw[1] += bitwidth_cost
                        for _sg in (2, 4, 8):
                            _sweep_bw[_sg] += FallbackDict.estimate_bitwidth_cost_grouped(residual, blk_eps, _sg)

                        raw_cost = FallbackDict.RAW_ENTRY_OVERHEAD + block_len * FallbackDict.RAW_BYTES_PER_POINT

                        

                        # 残差绝对值分位点（用于评估百分位量化方案收益）

                        abs_res_np = residual.abs().cpu().numpy()

                        pct_vals = np.percentile(abs_res_np, [50, 75, 85, 90, 95])

                        

                        block_stats.append({

                            'start': start_time,

                            'end': end_time,

                            'level': level_code,

                            'max_error': block_max_error,

                            'mae': block_mae,

                            'max_rel_error': block_max_rel_error,

                            'mean_rel_error': block_mean_rel_error,

                            'bad_points': block_bad_points,

                            'span': blk_span,

                            'bits': blk_bits,

                            'bitwidth_cost': bitwidth_cost,

                            'bitwidth_cost_grouped': bitwidth_cost_grouped,

                            'raw_cost': raw_cost,

                            'pct50': pct_vals[0],

                            'pct75': pct_vals[1],

                            'pct85': pct_vals[2],

                            'pct90': pct_vals[3],

                            'pct95': pct_vals[4],

                        })

        

        # =====================================================================

        # 全局统计

        # =====================================================================

        if error_mode == 'relative':

            compliant_mask = all_rel_errors < error_threshold

        else:

            compliant_mask = all_abs_errors < norm_threshold

        compliant_points = compliant_mask.sum().item()

        total_points = total_length

        

        compliance_rate = compliant_points / total_points

        max_abs_error = all_abs_errors.max().item()

        mae = all_abs_errors.mean().item()

        mse = (all_abs_errors ** 2).mean().item()

        mae_orig = mae * std_val

        max_abs_error_orig = max_abs_error * std_val

        mean_rel_error = all_rel_errors.mean().item()

        max_rel_error = all_rel_errors.max().item()

        

        # =====================================================================

        # 打印报告

        # =====================================================================

        print("\n[Global Statistics]")

        print(f"    Compliance Rate:     {compliance_rate * 100:.4f}% ({compliant_points}/{total_points})")

        print(f"    Max Absolute Error:  {max_abs_error:.6f} (norm) / {max_abs_error_orig:.4f} (orig)")

        print(f"    Mean Absolute Error: {mae:.6f} (norm) / {mae_orig:.4f} (orig)")

        print(f"    Mean Squared Error:  {mse:.6f} (norm)")

        print(f"    Mean Relative Error: {mean_rel_error*100:.4f}%")

        print(f"    Max Relative Error:  {max_rel_error*100:.4f}%")

        

        # =====================================================================

        # 位宽分布统计

        # =====================================================================

        bits_dist = {}  # bits -> {'count': N, 'points': N, 'total_cost': N, 'total_span': N}

        for b in block_stats:

            bits = b['bits']

            bl = b['end'] - b['start']

            if bits not in bits_dist:

                bits_dist[bits] = {'count': 0, 'points': 0, 'total_cost': 0, 'total_span': 0.0}

            bits_dist[bits]['count'] += 1

            bits_dist[bits]['points'] += bl

            bits_dist[bits]['total_cost'] += b['bitwidth_cost']

            bits_dist[bits]['total_span'] += b['span']

        

        print("\n[Bitwidth Distribution]")

        print(f"    {'Bits':>5} {'Blocks':>8} {'Rate':>8} {'Points':>10} {'TotalCost':>10} {'AvgSpan':>10}")

        print("    " + "-" * 60)

        for bits in sorted(bits_dist.keys()):

            d = bits_dist[bits]

            rate = d['count'] / total_blocks * 100 if total_blocks > 0 else 0

            avg_span = d['total_span'] / d['count'] if d['count'] > 0 else 0

            label = "0-bit" if bits == 0 else f"{bits}-bit"

            print(f"    {label:>5} {d['count']:>8} {rate:>7.2f}% {d['points']:>10} {d['total_cost']:>9}B {avg_span:>10.6f}")

        

        # 最差的几个块（按 span 排序）

        worst_blocks = sorted(block_stats, key=lambda x: x['span'], reverse=True)[:5]

        print("\n[Top 5 Worst Blocks (by span)]")

        print(f"    {'Start':>8} {'End':>8} {'Level':>6} {'Span':>10} {'Bits':>5} {'BWCost':>7} {'BadPts':>7} {'MAE':>10}")

        print("    " + "-" * 80)

        for b in worst_blocks:

            print(f"    {b['start']:>8} {b['end']:>8} {b['level']:>6} "

                  f"{b['span']:>10.6f} {b['bits']:>5} {b['bitwidth_cost']:>7} {b['bad_points']:>7} {b['mae']:>10.6f}")

        

        # 层级分布统计

        level_stats = {}

        for b in block_stats:

            lv = b['level']

            if lv not in level_stats:

                level_stats[lv] = {'count': 0, 'zero_bit': 0, 'total_mae': 0, 'total_bits': 0}

            level_stats[lv]['count'] += 1

            level_stats[lv]['total_mae'] += b['mae']

            level_stats[lv]['total_bits'] += b['bits']

            if b['bits'] == 0:

                level_stats[lv]['zero_bit'] += 1

        

        print("\n[Level Distribution]")

        print(f"    {'Level':>6} {'Blocks':>8} {'0-bit':>8} {'0-bitRate':>10} {'AvgBits':>8} {'AvgMAE':>12}")

        print("    " + "-" * 60)

        for lv in sorted(level_stats.keys()):

            s = level_stats[lv]

            avg_mae = s['total_mae'] / s['count'] if s['count'] > 0 else 0

            avg_bits = s['total_bits'] / s['count'] if s['count'] > 0 else 0

            zb_rate = s['zero_bit'] / s['count'] * 100 if s['count'] > 0 else 0

            print(f"    {lv:>6} {s['count']:>8} {s['zero_bit']:>8} {zb_rate:>9.2f}% {avg_bits:>8.2f} {avg_mae:>12.6f}")

        

        # 块长度分布统计

        block_size_stats = {}

        for b in block_stats:

            block_len = b['end'] - b['start']

            if block_len not in block_size_stats:

                block_size_stats[block_len] = {'count': 0, 'zero_bit': 0, 'total_mae': 0, 'total_bits': 0, 'total_bw_cost': 0, 'total_rel_error': 0, 'total_max_error': 0, 'total_bad_pts': 0, 'total_pct50': 0, 'total_pct75': 0, 'total_pct85': 0, 'total_pct90': 0, 'total_pct95': 0}

            block_size_stats[block_len]['count'] += 1

            block_size_stats[block_len]['total_mae'] += b['mae']

            block_size_stats[block_len]['total_bits'] += b['bits']

            block_size_stats[block_len]['total_bw_cost'] += b['bitwidth_cost']

            block_size_stats[block_len]['total_rel_error'] += b['mean_rel_error']

            block_size_stats[block_len]['total_max_error'] += b['max_error']

            block_size_stats[block_len]['total_bad_pts'] += b['bad_points']

            block_size_stats[block_len]['total_pct50'] += b['pct50']

            block_size_stats[block_len]['total_pct75'] += b['pct75']

            block_size_stats[block_len]['total_pct85'] += b['pct85']

            block_size_stats[block_len]['total_pct90'] += b['pct90']

            block_size_stats[block_len]['total_pct95'] += b['pct95']

            if b['bits'] == 0:

                block_size_stats[block_len]['zero_bit'] += 1

        

        print("\n[Block Size Distribution]")

        print(f"    {'Size':>8} {'Blocks':>8} {'0-bit':>6} {'0bRate':>8} {'AvgBits':>8} {'AvgBWB':>8} {'AvgMAE':>10} {'AvgMaxE':>10} {'AvgBadPct':>10} {'AvgRelE':>9} {'P50':>8} {'P75':>8} {'P85':>8} {'P90':>8} {'P95':>8}")

        print("    " + "-" * 150)

        for size in sorted(block_size_stats.keys(), reverse=True):

            s = block_size_stats[size]

            n = s['count'] if s['count'] > 0 else 1

            avg_mae = s['total_mae'] / n

            avg_bits = s['total_bits'] / n

            avg_bw = s['total_bw_cost'] / n

            avg_rel = s['total_rel_error'] / n

            avg_max_e = s['total_max_error'] / n

            avg_bad_pct = s['total_bad_pts'] / (n * size) * 100

            zb_rate = s['zero_bit'] / n * 100

            p50 = s['total_pct50'] / n

            p75 = s['total_pct75'] / n

            p85 = s['total_pct85'] / n

            p90 = s['total_pct90'] / n

            p95 = s['total_pct95'] / n

            print(f"    {size:>8} {s['count']:>8} {s['zero_bit']:>6} {zb_rate:>7.2f}% {avg_bits:>8.2f} {avg_bw:>8.1f}B {avg_mae:>10.6f} {avg_max_e:>10.6f} {avg_bad_pct:>9.2f}% {avg_rel*100:>8.4f}% {p50:>8.4f} {p75:>8.4f} {p85:>8.4f} {p90:>8.4f} {p95:>8.4f}")

        

        # =====================================================================

        # 压缩比估算（动态位宽量化）

        # 总体积 = Grid(8-bit) + Index(per-block) + ΣBitwidth_cost(per block)

        # =====================================================================

        vec_dim = self.grid_storage.trend_dim + self.grid_storage.context_dim

        total_nodes = self.grid_storage.num_base_nodes + self.manager.patch_counter

        num_slots = self.manager.num_slots

        

        # Vector GC 回收量（如果已执行 GC）

        gc_killed = 0

        if hasattr(self, 'gc_result') and self.gc_result is not None:

            gc_killed = self.gc_result.get('total_killed', 0)

        effective_nodes = total_nodes - gc_killed

        

        _quant_bits = getattr(self.args, 'quant_bits', 8)
        # bytes per stored dim: quantised -> quant_bits/8; unquantised
        # (quant_bits == 0) -> 4.0 (raw float32, the real on-disk cost).
        _bytes_per_dim = (_quant_bits / 8) if _quant_bits > 0 else 4.0

        # Aux 异构维度（Path A）：aux 节点仅前 model.aux_dim 维有效
        _model_aux_dim = getattr(self.model, 'aux_dim', vec_dim)
        _aux_count = self.manager.get_aux_stats().get('total_aux_tokens', 0) \
            if hasattr(self.manager, 'get_aux_stats') else 0
        _main_count = total_nodes - _aux_count
        _eff_main_count = effective_nodes - _aux_count  # GC 不会回收 aux 节点

        grid_f32 = (_main_count * vec_dim + _aux_count * _model_aux_dim) * 4

        grid_quantized = int(
            _main_count * vec_dim * _bytes_per_dim
            + _aux_count * _model_aux_dim * _bytes_per_dim
        )

        grid_quantized_gc = int(
            _eff_main_count * vec_dim * _bytes_per_dim
            + _aux_count * _model_aux_dim * _bytes_per_dim
        )

        # 中分隐式二叉树索引：1 byte bitmask per base_block

        base_block_size = self.manager.base_block_size

        num_base_blocks = math.ceil(self.manager.total_length / base_block_size)

        index_bitmask = num_base_blocks * FallbackDict.INDEX_PER_BASE_BLOCK

        

        # 动态位宽总代价

        total_bw_bytes = sum(b['bitwidth_cost'] for b in block_stats)

        total_bw_bytes_grouped = sum(b['bitwidth_cost_grouped'] for b in block_stats)

        zero_bit_blocks = sum(1 for b in block_stats if b['bits'] == 0)

        _rg = getattr(self.args, 'residual_groups', 1)

        

        original_bytes = total_length * 4

        

        # 压缩比：quant_bits grid + bitmask index + bitwidth correction

        total_compressed = grid_quantized + index_bitmask + total_bw_bytes

        ratio = original_bytes / total_compressed if total_compressed > 0 else float('inf')

        

        total_compressed_gc = grid_quantized_gc + index_bitmask + total_bw_bytes

        ratio_gc = original_bytes / total_compressed_gc if total_compressed_gc > 0 else float('inf')

        # Sub-group 压缩比：当 residual_groups > 1 时作为 primary 指标
        total_compressed_grp = grid_quantized + index_bitmask + total_bw_bytes_grouped
        ratio_grp = original_bytes / total_compressed_grp if total_compressed_grp > 0 else float('inf')

        # 选择 primary 指标：使用 grouped（_rg=1 时等价于 ungrouped）
        total_compressed_primary = total_compressed_grp
        ratio_primary = ratio_grp

        

        print("\n[Compression Size Breakdown (Dynamic Bitwidth + Midpoint)]")

        print(f"    Original data:         {original_bytes/1024:>10.2f} KB ({total_length} pts x 4B)")

        if _aux_count > 0 and _model_aux_dim < vec_dim:
            # 异构 grid：分别列出 main 和 aux 的开销
            print(f"    Grid (float32):        {grid_f32/1024:>10.2f} KB "
                  f"(main {_main_count}n×{vec_dim}d + aux {_aux_count}n×{_model_aux_dim}d, ×4B)")
            print(f"    Grid ({_quant_bits}-bit):          {grid_quantized/1024:>10.2f} KB "
                  f"(main {_main_count}n×{vec_dim}d + aux {_aux_count}n×{_model_aux_dim}d, "
                  f"×{_bytes_per_dim:.1f}B)")
        else:
            print(f"    Grid (float32):        {grid_f32/1024:>10.2f} KB ({total_nodes} nodes x {vec_dim}d x 4B)")
            print(f"    Grid ({_quant_bits}-bit):          {grid_quantized/1024:>10.2f} KB ({total_nodes} nodes x {vec_dim}d x {_bytes_per_dim:.1f}B)")

        if gc_killed > 0:

            print(f"    Grid ({_quant_bits}-bit, GC):      {grid_quantized_gc/1024:>10.2f} KB ({effective_nodes} nodes, -{gc_killed} dead)")

        print(f"    Index (bitmask):       {index_bitmask/1024:>10.2f} KB ({num_base_blocks} base_blocks x {FallbackDict.INDEX_PER_BASE_BLOCK}B)")

        # Sub-group 扫描表：显示不同分组数下的理论上限
        print(f"    Bitwidth correction:   {total_bw_bytes/1024:>10.2f} KB ({total_blocks} blocks, {zero_bit_blocks} are 0-bit)")
        print(f"    [Sub-group sweep (residual only, header overhead included)]")
        for _sg in (1, 2, 4, 8):
            _sg_bw   = _sweep_bw[_sg]
            _sg_tc   = grid_quantized + index_bitmask + _sg_bw
            _sg_r    = original_bytes / _sg_tc if _sg_tc > 0 else float('inf')
            _sg_save = total_bw_bytes - _sg_bw
            _marker  = " ← active" if _sg == _rg else ""
            print(f"      groups={_sg}: {_sg_bw/1024:>8.2f} KB (Δ{_sg_save/1024:+.2f} KB) → {_sg_r:.3f}x{_marker}")

        print(f"  [Total ({_rg}-group BW)]:    {total_compressed_primary/1024:>10.2f} KB → ratio {ratio_primary:.2f}x")

        if gc_killed > 0:

            print(f"  [Total with GC]:         {total_compressed_gc/1024:>10.2f} KB \u2192 ratio {ratio_gc:.2f}x")

        

        print("\n" + "=" * 70)

        print("END OF EVALUATION REPORT")

        print("=" * 70)

        

        # =====================================================================

        # 返回结果字典

        # =====================================================================

        result = {

            'compliance_rate': compliance_rate,

            'compliant_points': compliant_points,

            'total_points': total_points,

            'max_abs_error': max_abs_error,

            'max_abs_error_orig': max_abs_error_orig,

            'mae': mae,

            'mae_orig': mae_orig,

            'mse': mse,

            'mean_rel_error': mean_rel_error,

            'max_rel_error': max_rel_error,

            'total_blocks': total_blocks,

            'error_threshold': error_threshold,

            'error_mode': error_mode,

            'std_val': std_val,

            'level_stats': level_stats,

            'block_size_stats': block_size_stats,

            'bitwidth_stats': {

                'bits_distribution': bits_dist,

                'zero_bit_blocks': zero_bit_blocks,

                'total_bw_bytes': total_bw_bytes,
                'total_bw_bytes_grouped': total_bw_bytes_grouped,

                'ratio': ratio_primary,  # grouped when residual_groups > 1

                'gc_killed': gc_killed,

                'effective_nodes': effective_nodes,

                'ratio_with_gc': ratio_gc,

            },

        }

        

        return result

    

    def build_fallback_dict(self, error_threshold: float = 0.10, 

                            error_mode: str = 'relative') -> FallbackDict:

        """

        构建 Tier 2/3 兜底字典：纯比特代价驱动（Pure Bit-Cost Driven）。

        

        流程：

        1. 遍历所有块，解码并计算误差

        2. bad_points == 0 → Tier 1（无条目，纯神经网络）

        3. patch_cost < raw_cost → Tier 2（PATCH：记录坏点 offset + residual）

        4. patch_cost >= raw_cost → Tier 3（RAW：记录全部原始数据）

        

        其中 patch_cost = bad_pts × 5B, raw_cost = block_len × 2B (FP16)

        

        Args:

            error_threshold: 坏点误差阈值（relative: 比例, absolute: 原始单位）

            error_mode: 'relative' 或 'absolute'

            

        Returns:

            FallbackDict: 构建好的兜底字典

        """

        self.model.eval()

        self._refresh_quant_params()

        

        # 获取 scaler 参数

        scaler = self.data_loader.scaler

        if scaler is not None:

            std_val = scaler.std.item() if hasattr(scaler.std, 'item') else float(scaler.std)

            mean_val = scaler.mean.item() if hasattr(scaler.mean, 'item') else float(scaler.mean)

        else:

            std_val, mean_val = 1.0, 0.0

        

        if error_mode == 'absolute':

            norm_threshold = error_threshold / std_val

        else:

            norm_threshold = None  # relative mode: per-point comparison

        

        # 节点代价 = feature_dim × bytes_per_dim（由 quant_bits 决定）

        vec_dim = self.grid_storage.trend_dim + self.grid_storage.context_dim

        _qb = getattr(self.args, 'quant_bits', 8)

        # unquantised (quant_bits == 0) -> float32 (4 B/dim) node cost
        _bpd = (_qb / 8) if _qb > 0 else 4.0

        fb_dict = FallbackDict(node_cost_bytes=int(vec_dim * _bpd))

        

        all_blocks = self.manager.get_all_unique_blocks()

        

        tier1_count, tier2_count, tier3_count = 0, 0, 0

        

        print("\n" + "=" * 70)

        print("BUILDING FALLBACK DICTIONARY")

        print("=" * 70)

        

        # 按块长度分组解码

        len_groups = defaultdict(list)

        for block in all_blocks:

            start_time, end_time, left_id, right_id, level_code = block

            block_len = end_time - start_time

            len_groups[block_len].append(block)

        

        eval_bs = getattr(self.args, 'eval_batch_size', 256)

        with torch.no_grad():

            for block_len, group in len_groups.items():

                for chunk_start in range(0, len(group), eval_bs):

                    chunk = group[chunk_start:chunk_start + eval_bs]

                    left_ids = torch.tensor([b[2] for b in chunk], device=self.device)

                    right_ids = torch.tensor([b[3] for b in chunk], device=self.device)

                    blk_offsets = torch.tensor(

                        [b[0] % self.args.base_block_size for b in chunk], device=self.device)

                    

                    outputs = self.model.decode_batch(left_ids, right_ids, block_len, blk_offsets)

                    outputs = outputs.squeeze(1)  # [chunk_size, block_len]

                    

                    raw_cost = FallbackDict.RAW_ENTRY_OVERHEAD + block_len * FallbackDict.RAW_BYTES_PER_POINT

                    

                    for i, block in enumerate(chunk):

                        start_time, end_time, left_id, right_id, level_code = block

                        output = outputs[i]

                        true = self.raw_data[start_time:end_time]

                        abs_error = torch.abs(output - true)

                        

                        # 坏点判定

                        if error_mode == 'relative':

                            true_orig = true * std_val + mean_val

                            error_orig = abs_error * std_val

                            rel_error = error_orig / torch.clamp(torch.abs(true_orig), min=1.0)

                            bad_mask = rel_error >= error_threshold

                        else:

                            bad_mask = abs_error >= norm_threshold

                        

                        bad_point_count = bad_mask.sum().item()

                        patch_cost = FallbackDict.PATCH_ENTRY_OVERHEAD + bad_point_count * FallbackDict.PATCH_BYTES_PER_POINT

                        

                        if bad_point_count == 0:

                            # Tier 1: 完美块，无需兜底

                            tier1_count += 1

                        elif patch_cost < raw_cost:

                            # Tier 2: PATCH 更便宜（稀疏修补）

                            offsets = torch.where(bad_mask)[0].cpu().tolist()

                            residuals = (true - output)[bad_mask].cpu().tolist()

                            fb_dict.add_patch(left_id, block_len, offsets, residuals)

                            tier2_count += 1

                        else:

                            # Tier 3: RAW 更便宜（整块存储）

                            raw_data = true.cpu().tolist()

                            fb_dict.add_raw(left_id, block_len, raw_data)

                            tier3_count += 1

        

        total = tier1_count + tier2_count + tier3_count

        print(f"    Tier 1 (Neural Net):   {tier1_count:>6} ({tier1_count/total*100:.2f}%)")

        print(f"    Tier 2 (PATCH):        {tier2_count:>6} ({tier2_count/total*100:.2f}%)")

        print(f"    Tier 3 (RAW):          {tier3_count:>6} ({tier3_count/total*100:.2f}%)")

        fb_dict.print_summary()

        

        # =================================================================

        # Vector GC 统计（仅统计，不实际执行回收）

        # 规则：一个 patch 节点可回收 ⟺ 其左右两侧的块都是 RAW

        # =================================================================

        num_base = self.grid_storage.num_base_nodes

        

        # 建立 right_id → left_id 映射，用于查找某节点左侧块的 left_id

        right_to_left = {}

        for block in all_blocks:

            _, _, left_id, right_id, _ = block

            right_to_left[right_id] = left_id

        

        # 收集所有在 index_table 中实际使用的 patch node

        used_patch_nodes = set()

        for block in all_blocks:

            _, _, left_id, right_id, _ = block

            if left_id >= num_base:

                used_patch_nodes.add(left_id)

            if right_id >= num_base:

                used_patch_nodes.add(right_id)

        

        reclaimable_nodes = []

        for node_id in used_patch_nodes:

            # 右侧块：left_id == node_id → 在字典中查 key=node_id

            right_block_is_raw = fb_dict.is_raw(node_id)

            # 左侧块：right_id == node_id → 找其 left_id，在字典中查

            left_block_left_id = right_to_left.get(node_id)

            if left_block_left_id is not None:

                left_block_is_raw = fb_dict.is_raw(left_block_left_id)

            else:

                left_block_is_raw = False

            

            if left_block_is_raw and right_block_is_raw:

                reclaimable_nodes.append(node_id)

        

        gc_savings = len(reclaimable_nodes) * vec_dim  # bytes saved

        print(f"\n[Vector GC Estimate]")

        print(f"    Used patch nodes:      {len(used_patch_nodes)}")

        print(f"    Reclaimable nodes:     {len(reclaimable_nodes)} (both sides RAW)")

        print(f"    Potential savings:     {gc_savings/1024:.2f} KB ({gc_savings} bytes)")

        if len(used_patch_nodes) > 0:

            print(f"    Reclaim rate:          {len(reclaimable_nodes)/len(used_patch_nodes)*100:.2f}%")

        

        print("=" * 70)

        

        self.fallback_dict = fb_dict

        return fb_dict

    

    def visualize_reconstruction(self, block_idx: int = 0, save_path: str = None):

        """

        可视化重构效果：展示 TCN 重构结果和误差分布。

        

        展示内容：

        1. 上图：原始数据 vs TCN 重构

        2. 下图：误差分布

        

        Args:

            block_idx: 要可视化的块索引（从 get_all_unique_blocks 中选择）

            save_path: 保存路径，如果为 None 则直接显示

        """

        self.model.eval()

        self._refresh_quant_params()

        

        all_blocks = self.manager.get_all_unique_blocks()

        if block_idx >= len(all_blocks):

            print(f"[Visualize] block_idx {block_idx} out of range, max is {len(all_blocks) - 1}")

            block_idx = 0

        

        start_time, end_time, left_id, right_id, level_code = all_blocks[block_idx]

        block_len = end_time - start_time

        

        print(f"[Visualize] Block {block_idx}: [{start_time}, {end_time}), len={block_len}, "

              f"left_id={left_id}, right_id={right_id}, level={level_code}")

        

        with torch.no_grad():

            # 获取左右边界向量

            left_ids = torch.tensor([left_id], device=self.device)

            right_ids = torch.tensor([right_id], device=self.device)

            

            base_block_size = self.args.base_block_size

            

            # 完整前向传播获取最终输出 - 固定长度解码再截取

            output_full = self.model(left_ids, right_ids)

            output = output_full.squeeze()[:block_len]  # 截取有效部分

            

            # 真实数据

            true = self.raw_data[start_time:end_time]

        

        # 转换为 numpy

        output_np = output.cpu().numpy()

        true_np = true.cpu().numpy()

        error_np = (output - true).cpu().numpy()

        

        # 绘图

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        t = np.arange(start_time, end_time)

        

        # 上图：原始数据 vs 重构

        ax1 = axes[0]

        ax1.plot(t, true_np, 'b-', label='Ground Truth', linewidth=1.5, alpha=0.8)

        ax1.plot(t, output_np, 'g-', label='Reconstruction (TCN)', linewidth=1.5, alpha=0.8)

        ax1.set_ylabel('Value')

        ax1.set_title(f'Block [{start_time}, {end_time}) - Ground Truth vs Reconstruction')

        ax1.legend(loc='upper right')

        ax1.grid(True, alpha=0.3)

        

        # 计算误差统计

        output_mse = ((output - true) ** 2).mean().item()

        output_mae = torch.abs(output - true).mean().item()

        output_max = torch.abs(output - true).max().item()

        ax1.text(0.02, 0.98, f'MSE: {output_mse:.6f}\nMAE: {output_mae:.6f}\nMax: {output_max:.6f}', 

                 transform=ax1.transAxes, verticalalignment='top', fontsize=10,

                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        

        # 下图：误差分布

        ax2 = axes[1]

        ax2.plot(t, error_np, 'r-', label='Error (Pred - GT)', linewidth=1.5, alpha=0.8)

        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)

        ax2.set_xlabel('Time')

        ax2.set_ylabel('Error')

        ax2.set_title('Reconstruction Error')

        ax2.legend(loc='upper right')

        ax2.grid(True, alpha=0.3)

        

        plt.tight_layout()

        

        if save_path:

            plt.savefig(save_path, dpi=150, bbox_inches='tight')

            print(f"[Visualize] Saved to {save_path}")

        else:

            plt.show()

        

        plt.close()

        

        # 统计不能被拟合的点（误差超过阈值）

        abs_error = torch.abs(output - true)

        

        # 获取标准化参数

        scaler = self.data_loader.scaler

        if scaler is not None:

            mean_val = scaler.mean.item() if hasattr(scaler.mean, 'item') else float(scaler.mean)

            std_val = scaler.std.item() if hasattr(scaler.std, 'item') else float(scaler.std)

        else:

            mean_val, std_val = 0.0, 1.0

        

        print(f"\n[Visualize] Block {block_idx} Error Distribution:")

        print(f"  Total points: {block_len}")

        print(f"  Standardization: mean={mean_val:.4f}, std={std_val:.4f}")

        print(f"  --- Standardized Scale ---")

        

        thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]

        for thresh in thresholds:

            violation_count = (abs_error > thresh).sum().item()

            violation_rate = violation_count / block_len * 100

            # 转换为原始尺度

            thresh_original = thresh * std_val

            print(f"  |error| > {thresh:.2f} (original: {thresh_original:.4f}): {violation_count} points ({violation_rate:.1f}%)")

        

        # 输出原始尺度的误差统计

        abs_error_original = abs_error * std_val

        print(f"  --- Original Scale ---")

        print(f"  Max |error|: {abs_error.max().item():.4f} (original: {abs_error_original.max().item():.4f})")

        print(f"  Mean |error|: {abs_error.mean().item():.4f} (original: {abs_error_original.mean().item():.4f})")

        print(f"  MSE: {output_mse:.6f} (original: {output_mse * std_val**2:.6f})")

        

        return {

            'output_mse': output_mse,

            'output_mae': output_mae,

            'output_max': output_max,

            'block_len': block_len,

            'abs_error': abs_error.cpu().numpy(),

            'mean': mean_val,

            'std': std_val,

        }

    

    def visualize_full_reconstruction(self, num_blocks: int = 5, save_path: str = None):

        """

        可视化多个块的重构效果（按误差从大到小排序）。

        

        Args:

            num_blocks: 要可视化的块数量

            save_path: 保存路径前缀，如果为 None 则直接显示

        """

        self.model.eval()

        self._refresh_quant_params()

        

        all_blocks = self.manager.get_all_unique_blocks()

        

        # 计算每个块的误差并排序（固定长度解码）

        block_errors = []

        with torch.no_grad():

            # 批量处理所有块

            left_ids = torch.tensor([b[2] for b in all_blocks], device=self.device)

            right_ids = torch.tensor([b[3] for b in all_blocks], device=self.device)

            

            outputs = self.model(left_ids, right_ids)  # [batch, 1, base_block_size]

            outputs = outputs.squeeze(1)  # [batch, base_block_size]

            

            for i, (start_time, end_time, left_id, right_id, level_code) in enumerate(all_blocks):

                block_len = end_time - start_time

                output = outputs[i, :block_len]

                true = self.raw_data[start_time:end_time]

                

                mse = ((output - true) ** 2).mean().item()

                block_errors.append((i, mse))

        

        # 按误差排序

        block_errors.sort(key=lambda x: x[1], reverse=True)

        

        print(f"\n[Visualize] Top {num_blocks} blocks by error:")

        for rank, (idx, mse) in enumerate(block_errors[:num_blocks]):

            start_time, end_time, _, _, level = all_blocks[idx]

            print(f"  #{rank+1}: Block {idx} [{start_time}-{end_time}], MSE={mse:.6f}, level={level}")

            

            if save_path:

                path = f"{save_path}_block{idx}.png"

            else:

                path = None

            self.visualize_reconstruction(block_idx=idx, save_path=path)

    

    # =========================================================================

    # Patch Registry: 坏点补丁收集与应用

    # =========================================================================

    

    def collect_patches(self, error_threshold: float = 0.1, use_float16: bool = True) -> PatchRegistry:

        """

        收集所有块的坏点补丁。

        

        应在训练完全结束后调用（分裂阶段结束后）。

        遍历所有块，找出误差超过阈值的点，记录其残差。

        

        Args:

            error_threshold: 误差阈值，超过此值的点被视为坏点

            use_float16: 是否使用 float16 存储残差（节省空间）

            

        Returns:

            PatchRegistry: 填充好的补丁注册表

        """

        collector = PatchCollector(

            model=self.model,

            manager=self.manager,

            raw_data=self.raw_data,

            device=self.device,

            base_block_size=self.args.base_block_size

        )

        

        self.patch_registry = collector.collect(

            error_threshold=error_threshold,

            use_float16=use_float16,

            verbose=True

        )

        

        return self.patch_registry

    

    def reconstruct_with_patches(self) -> torch.Tensor:

        """

        使用补丁修正的完整重构。

        

        在基础 TCN 重构的基础上，应用补丁修正坏点。

        

        Returns:

            修正后的完整重构序列

        """

        if not hasattr(self, 'patch_registry') or self.patch_registry is None:

            print("[Warning] No patch registry found, using base reconstruction")

            return self.reconstruct_full()

        

        self.model.eval()

        self._refresh_quant_params()

        

        all_blocks = self.manager.get_all_unique_blocks()

        reconstructed = torch.zeros(self.manager.total_length, device=self.device)

        

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

                

                # 应用补丁修正

                output = self.patch_registry.apply_to_output(output, left_id)

                

                reconstructed[start_time:end_time] = output

        

        return reconstructed

    

    def evaluate_with_patches(self, error_threshold: float = 0.1) -> dict:

        """

        使用补丁修正后的评估。

        

        Args:

            error_threshold: 误差阈值

            

        Returns:

            评估结果字典

        """

        if not hasattr(self, 'patch_registry') or self.patch_registry is None:

            print("[Warning] No patch registry found")

            return {}

        

        self.model.eval()

        self._refresh_quant_params()

        

        all_blocks = self.manager.get_all_unique_blocks()

        total_length = self.manager.total_length

        

        all_abs_errors = torch.zeros(total_length, device=self.device)

        

        print("\n" + "=" * 70)

        print("EVALUATION WITH PATCHES")

        print("=" * 70)

        

        with torch.no_grad():

            left_ids = torch.tensor([b[2] for b in all_blocks], device=self.device)

            right_ids = torch.tensor([b[3] for b in all_blocks], device=self.device)

            

            outputs = self.model(left_ids, right_ids)

            outputs = outputs.squeeze(1)

            

            for i, (start_time, end_time, left_id, right_id, level_code) in enumerate(all_blocks):

                block_len = end_time - start_time

                output = outputs[i, :block_len]

                

                # 应用补丁修正

                output = self.patch_registry.apply_to_output(output, left_id)

                

                true = self.raw_data[start_time:end_time]

                abs_error = torch.abs(output - true)

                all_abs_errors[start_time:end_time] = abs_error

        

        # 统计

        compliant_mask = all_abs_errors < error_threshold

        compliant_points = compliant_mask.sum().item()

        total_points = total_length

        

        compliance_rate = compliant_points / total_points

        max_abs_error = all_abs_errors.max().item()

        mae = all_abs_errors.mean().item()

        mse = (all_abs_errors ** 2).mean().item()

        

        # 补丁统计

        patch_stats = self.patch_registry.get_statistics()

        

        print(f"\n[With Patches Statistics]")

        print(f"    Compliance Rate:     {compliance_rate * 100:.4f}% ({compliant_points}/{total_points})")

        print(f"    Max Absolute Error:  {max_abs_error:.6f} (L_inf)")

        print(f"    Mean Absolute Error: {mae:.6f} (MAE)")

        print(f"    Mean Squared Error:  {mse:.6f} (MSE)")

        print(f"\n[Patch Registry Statistics]")

        print(f"    Total patches:       {patch_stats['total_patches']}")

        print(f"    Blocks with patches: {patch_stats['blocks_with_patches']}")

        print(f"    Memory usage:        {patch_stats['memory_bytes'] / 1024:.2f} KB")

        print("=" * 70)

        

        return {

            'compliance_rate': compliance_rate,

            'compliant_points': compliant_points,

            'total_points': total_points,

            'max_abs_error': max_abs_error,

            'mae': mae,

            'mse': mse,

            'patch_stats': patch_stats,

        }

    

    def save_patch_registry(self, path: str):

        """保存补丁注册表到文件"""

        if hasattr(self, 'patch_registry') and self.patch_registry is not None:

            self.patch_registry.save(path)

            print(f"[PatchRegistry] Saved to {path}")

        else:

            print("[Warning] No patch registry to save")

    

    def load_patch_registry(self, path: str):

        """从文件加载补丁注册表"""

        self.patch_registry = PatchRegistry.load(path)

        print(f"[PatchRegistry] Loaded from {path}")

        stats = self.patch_registry.get_statistics()

        print(f"    Total patches: {stats['total_patches']}")

        print(f"    Memory usage: {stats['memory_bytes'] / 1024:.2f} KB")

