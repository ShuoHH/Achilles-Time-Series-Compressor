"""
Vector GC: RDO-based DP Pruning (Rate-Distortion Optimized Tree Pruning)

基于率失真优化的动态规划树剪枝。

核心架构：前向分裂日志 (Forward Split Logging)
    分裂时实时记账 split_history[(start, end)] = (mid, node_id)，
    一行 O(1) 代码就在内存中构建了完整的隐式二叉树。
    GC 完成后销毁账本，不写入磁盘，存储开销为 0。

DP 剪枝算法（纯只读分析，不修改 index_table / split_history）：
    对每个 base_block_size 长度的 Base Block，后序遍历隐式二叉树。
    在每个内部节点，对比三份账单：
    A) 保留分裂 = NODE_COST + cost_left_optimal + cost_right_optimal
    B) 合并 BITWIDTH = BITWIDTH_HEADER + ceil(length * bits / 8)（动态位宽量化）
    C) 合并 RAW   = RAW_OH + length * RAW_PP
    选最小值。若 B 或 C 胜出，标记中间节点为 dead，统计节省字节数。
    最终输出 dead_nodes 集合和节省量，供下游压缩比计算使用。

物理常数：
    NODE_COST       = vec_dim bytes (e.g. 32d * 1B = 32 bytes)
    BITWIDTH_HEADER = 3 bytes/entry (R_min FP16 + bits uint8)
    RAW_OH          = 5 bytes/entry (left_id + type + block_len)
    RAW_PP          = 2 bytes/point (FP16)
"""

import torch
from typing import Tuple, Dict, List, Set

from models.fallback_dict import FallbackDict


class VectorGC:
    """
    基于前向分裂日志的 DP 剪枝器。
    
    使用方法：
        gc = VectorGC(manager, model, grid_storage, raw_data, split_history, ...)
        result = gc.run()
    
    Args:
        manager: GridManager 实例
        model: NeurTSModel 实例（已解包装，由 run_vector_gc 负责 DataParallel 解包）
        grid_storage: GlobalGridStorage 实例（直接传入，避免 DataParallel 属性访问问题）
        raw_data: 原始时序数据 [total_length]（标准化后）
        split_history: 前向分裂日志 {(start, end): (mid, node_id)}
        error_threshold: 坏点误差阈值
        error_mode: 'relative' 或 'absolute'
        std_val: 标准化的 std（用于误差转换）
        mean_val: 标准化的 mean（用于误差转换）
    """
    
    def __init__(
        self,
        manager,
        model,
        grid_storage,
        raw_data: torch.Tensor,
        split_history: Dict[Tuple[int, int], Tuple[int, int]],
        error_threshold: float = 0.10,
        error_mode: str = 'relative',
        std_val: float = 1.0,
        mean_val: float = 0.0,
    ):
        self.manager = manager
        self.model = model
        self.grid_storage = grid_storage
        self.raw_data = raw_data
        self.split_history = split_history
        self.error_threshold = error_threshold
        self.error_mode = error_mode
        self.std_val = std_val
        self.mean_val = mean_val
        
        self.index_table = manager.index_table
        self.min_resolution = manager.min_resolution
        self.base_block_size = manager.base_block_size
        self.total_length = manager.total_length
        self.device = raw_data.device
        
        # 物理常数
        vec_dim = grid_storage.trend_dim + grid_storage.context_dim
        # 节点代价：量化启用时按 num_bits/8，未量化时按 float32 (4 B/dim)
        if getattr(grid_storage, '_quantization_enabled', False):
            _bpd = getattr(grid_storage, 'num_bits', 8) / 8
        else:
            _bpd = 4.0
        self.NODE_COST = int(vec_dim * _bpd)  # bytes per vector node
        self.BW_HEADER = FallbackDict.BITWIDTH_HEADER  # 3 bytes/entry header
        self.RAW_COST_PP = FallbackDict.RAW_BYTES_PER_POINT       # 2 bytes/point
        self.RAW_OH = FallbackDict.RAW_ENTRY_OVERHEAD              # 5 bytes/entry
        
        # 绝对误差模式的标准化阈值
        if error_mode == 'absolute':
            self.norm_threshold = error_threshold / std_val
        else:
            self.norm_threshold = None
        
        # GC 结果
        self.dead_nodes: Set[int] = set()       # 被判死刑的节点 ID
        self.merge_decisions: List[dict] = []   # 合并决策记录
    
    # =========================================================================
    # 辅助函数
    # =========================================================================
    
    def _get_interval_boundary_ids(self, t_start: int, t_end: int) -> Tuple[int, int]:
        """
        获取区间 [t_start, t_end) 的首尾边界节点 ID。
        
        利用 index_table 的性质：
        - 最左槽位的 left_id = 区间左边界节点
        - 最右槽位的 right_id = 区间右边界节点
        """
        start_slot = t_start // self.min_resolution
        end_slot = t_end // self.min_resolution - 1  # 最右槽位（含）
        return self.index_table[start_slot].left_id, self.index_table[end_slot].right_id
    
    def _compute_block_bitwidth_cost(self, t_start: int, t_end: int,
                                      left_id: int, right_id: int) -> float:
        """
        对区间 [t_start, t_end) 用指定边界节点重新推理，计算动态位宽代价。
        
        推理经过 Grid.forward()，自动应用 fake quantize。
        返回 BITWIDTH 条目的字节代价。
        """
        block_len = t_end - t_start
        
        with torch.no_grad():
            output = self.model.decode_single(left_id, right_id, block_len)
            output = output.squeeze()  # [block_len]
        
        true = self.raw_data[t_start:t_end]
        residual = true - output  # 带符号残差
        span = (residual.max() - residual.min()).item()
        
        # 计算 per-block epsilon
        if self.error_mode == 'absolute':
            eps = self.norm_threshold
        else:
            true_orig = true * self.std_val + self.mean_val
            min_denom = torch.clamp(torch.abs(true_orig), min=1.0).min().item()
            eps = self.error_threshold * min_denom / self.std_val
        
        return float(FallbackDict.estimate_bitwidth_cost(span, block_len, eps))
    
    def _collect_subtree_nodes(self, start: int, end: int) -> List[int]:
        """
        递归收集子树 [start, end) 内所有中间节点的 ID。
        
        不包含首尾边界节点（它们属于父级），只收集分裂产生的内部节点。
        """
        if (start, end) not in self.split_history:
            return []  # 叶子，无内部节点
        
        mid, node_id = self.split_history[(start, end)]
        nodes = [node_id]
        nodes.extend(self._collect_subtree_nodes(start, mid))
        nodes.extend(self._collect_subtree_nodes(mid, end))
        return nodes
    
    
    # =========================================================================
    # DP 剪枝核心
    # =========================================================================
    
    def dp_prune(self, start: int, end: int) -> float:
        """
        后序遍历 DP 剪枝（递归核心）。
        
        基于前向分裂日志的隐式二叉树：
        - (start, end) in split_history -> 内部节点，有子树
        - (start, end) not in split_history -> 叶子节点
        
        三份账单：
        A) cost_keep  = NODE_COST + cost_left + cost_right（保留分裂）
        B) cost_merge_patch = PATCH_OH + recalculated_bad_points * PATCH_PP（合并 PATCH）
        C) cost_merge_raw   = RAW_OH + length * RAW_PP（合并 RAW）
        
        Args:
            start: 区间起始时间（含）
            end:   区间结束时间（不含）
            
        Returns:
            该子树的最优字节成本
        """
        length = end - start
        left_id, right_id = self._get_interval_boundary_ids(start, end)
        
        # =================================================================
        # Step 1: 触底 — 叶子节点（未被进一步切碎）
        # =================================================================
        if (start, end) not in self.split_history:
            cost_bw = self._compute_block_bitwidth_cost(start, end, left_id, right_id)
            cost_raw = float(self.RAW_OH + length * self.RAW_COST_PP)
            best_cost = min(cost_bw, cost_raw)
            return best_cost
        
        # =================================================================
        # Step 2: 向子树要账（自底向上）
        # =================================================================
        mid, split_node_id = self.split_history[(start, end)]
        
        cost_left = self.dp_prune(start, mid)
        cost_right = self.dp_prune(mid, end)
        
        # =================================================================
        # Step 3: 终极三选一审判
        # =================================================================
        
        # 账单 A：保留分裂（NODE_COST = 分裂点节点的存储代价）
        cost_keep = self.NODE_COST + cost_left + cost_right
        
        # 账单 B：合并为一整块 BITWIDTH（用大区间首尾边界节点重新推理）
        cost_merge_bw = self._compute_block_bitwidth_cost(start, end, left_id, right_id)
        
        # 账单 C：合并为一整块 RAW
        cost_merge_raw = float(self.RAW_OH + length * self.RAW_COST_PP)
        
        # 三选一
        best_cost = min(cost_keep, cost_merge_bw, cost_merge_raw)
        
        if best_cost >= cost_keep:
            # 保留分裂是最优或并列最优，什么都不做
            return cost_keep
        
        # ====== 触发 Vector GC（纯统计，不修改任何状态）======
        
        # a. 收集子树内所有中间节点 ID（不含首尾边界）
        internal_nodes = self._collect_subtree_nodes(start, end)
        for nid in internal_nodes:
            self.dead_nodes.add(nid)
        
        # b. 确定合并后的决策类型
        if best_cost == cost_merge_bw:
            decision = 'merge_bw'
        else:
            decision = 'merge_raw'
        
        # c. 记录合并决策（不修改 index_table 和 split_history）
        self.merge_decisions.append({
            'start': start, 'end': end, 'length': length,
            'decision': decision,
            'cost_keep': cost_keep,
            'cost_merge_bw': cost_merge_bw,
            'cost_merge_raw': cost_merge_raw,
            'best_cost': best_cost,
            'saved_bytes': cost_keep - best_cost,
            'killed_nodes': len(internal_nodes),
        })
        
        return best_cost
    
    # =========================================================================
    # 顶层入口
    # =========================================================================
    
    def run(self) -> Dict:
        """
        对所有 Base Block 执行 DP 剪枝。
        
        流程：
        1. 遍历时间轴，按 base_block_size 步长切分
        2. 每段调用 dp_prune 递归算账
        3. 汇总统计：回收节点数、节省字节数
        
        注意：纯只读分析，不修改 index_table / split_history / patch_grid。
        只标记 dead nodes、统计回收量，供下游压缩比计算使用。
        
        Returns:
            统计结果字典
        """
        # 索引表实际覆盖的时间范围（可能小于 total_length）
        covered_length = len(self.index_table) * self.min_resolution
        num_full_blocks = covered_length // self.base_block_size
        has_partial = (covered_length % self.base_block_size) > 0
        num_base_blocks = num_full_blocks + (1 if has_partial else 0)
        
        print(f"\n{'=' * 60}")
        print(f"Vector GC: RDO-based DP Pruning")
        print(f"{'=' * 60}")
        print(f"    Base blocks:      {num_base_blocks} ({num_full_blocks} full" +
              (f" + 1 partial)" if has_partial else ")"))
        print(f"    Split history:    {len(self.split_history)} entries")
        print(f"    NODE_COST:        {self.NODE_COST} bytes")
        print(f"    BITWIDTH_HEADER:  {self.BW_HEADER}B/entry")
        print(f"    RAW_COST:         {self.RAW_OH}B/entry + {self.RAW_COST_PP}B/point")
        
        if len(self.split_history) == 0:
            print(f"    No splits recorded, nothing to prune.")
            return {'total_killed': 0, 'total_saved_bytes': 0}
        
        self.model.eval()
        total_cost = 0.0
        
        for t_start in range(0, covered_length, self.base_block_size):
            t_end = min(t_start + self.base_block_size, covered_length)
            block_cost = self.dp_prune(t_start, t_end)
            total_cost += block_cost
        
        # =====================================================================
        # 战报
        # =====================================================================
        total_killed = len(self.dead_nodes)
        total_saved = sum(d['saved_bytes'] for d in self.merge_decisions)
        total_merges = len(self.merge_decisions)
        
        merge_bw = sum(1 for d in self.merge_decisions if d['decision'] == 'merge_bw')
        merge_raw = sum(1 for d in self.merge_decisions if d['decision'] == 'merge_raw')
        
        print(f"\n{'*' * 60}")
        print(f"  Vector GC 战报")
        print(f"{'*' * 60}")
        print(f"    猎杀冗余节点:    {total_killed} 个")
        print(f"    释放存储空间:    {total_killed * self.NODE_COST} bytes "
              f"({total_killed * self.NODE_COST / 1024:.2f} KB)")
        print(f"    合并决策总数:    {total_merges}")
        print(f"      - 合并 BITWIDTH: {merge_bw} (位宽编码比分裂便宜)")
        print(f"      - 合并 RAW:      {merge_raw} (整块 RAW 更便宜)")
        print(f"    总成本节省:      {total_saved:.0f} bytes ({total_saved / 1024:.2f} KB)")
        print(f"    优化后总成本:    {total_cost:.0f} bytes ({total_cost / 1024:.2f} KB)")
        print(f"    剩余分裂记录:    {len(self.split_history)} entries")
        print(f"{'*' * 60}")
        
        return {
            'total_killed': total_killed,
            'dead_nodes': self.dead_nodes,
            'total_saved_bytes': total_saved,
            'total_cost': total_cost,
            'total_merges': total_merges,
            'merge_bw': merge_bw,
            'merge_raw': merge_raw,
            'merge_decisions': self.merge_decisions,
        }
