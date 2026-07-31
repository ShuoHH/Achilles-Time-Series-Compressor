"""
NeurTS Manager: 自适应索引管理器

采用定长最小粒度索引表 (Fixed-Granularity Index Table) 方案。
通过 O(1) 算术计算直接访问，无需遍历或二分查找。
"""

import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .neurts_model import NeurTSModel


@dataclass
class IndexEntry:
    """索引表条目：存储每个最小粒度槽位的边界ID和层级码。
    
    Attributes:
        left_id: 左边界Grid节点ID
        right_id: 右边界Grid节点ID
        level_code: 层级码 (0=Base, 1=Split1次, 2=Split2次...)
    """
    left_id: int
    right_id: int
    level_code: int = 0


class GridManager:
    """
    自适应索引管理器：基于定长最小粒度索引表的 O(1) 查询方案。
    
    核心设计：
    - 废弃 task_list 动态列表方案
    - 引入 index_table 定长数组，长度为 total_length // min_resolution
    - index_table[i] 隐式对应时间段 [i * min_resolution, (i+1) * min_resolution]
    - 查询复杂度严格 O(1): idx = t // min_resolution
    
    示例 (total_length=200, base_block_size=100, min_resolution=25):
    - Grid IDs: 0 (t=0), 1 (t=100), 2 (t=200)
    - index_table 长度为 8 (200/25):
        Idx 0 (0-25):    (left=0, right=1, level=0)  <- Base Block 0-100
        Idx 1 (25-50):   (left=0, right=1, level=0)  <- Base Block 0-100
        Idx 2 (50-75):   (left=0, right=1, level=0)  <- Base Block 0-100
        Idx 3 (75-100):  (left=0, right=1, level=0)  <- Base Block 0-100
        ...
    
    层级映射 (Level Mapping):
    - level=0 -> 长度 base_block_size (100)
    - level=1 -> 长度 base_block_size/2 (50)
    - level=2 -> 长度 base_block_size/4 (25) = min_resolution
    
    Args:
        model: NeurTSModel实例
        raw_data: 原始时序数据 [total_length]
        base_block_size: 基础块大小（初始Grid节点间距）
        min_resolution: 最小粒度（索引表的时间分辨率）
    """
    
    def __init__(
        self,
        model: NeurTSModel,
        raw_data: torch.Tensor,
        base_block_size: int,
        min_resolution: int = 50
    ):
        self.model = model
        self.raw_data = raw_data
        self.base_block_size = base_block_size
        self.min_resolution = min_resolution
        self.total_length = len(raw_data)
        
        # 当前已分配的patch节点数量
        self.patch_counter = 0
        
        # 回滚释放的patch节点全局ID池（free-list），优先复用
        self._free_patch_ids: List[int] = []
        
        # ── Aux Token 表（多 token 谱细化）─────────────────────────
        # 设计：一个块的所有 slot 共享同一份 aux 列表
        # key = (left_id, right_id, level_code)；value = tuple[int]（aux 节点全局 ID）
        # 一个块可以有 0..M_max 个 aux token（默认 0，向后兼容）
        # 分裂时：旧块的 aux token 被释放回 free pool（子块从零开始）
        self._aux_map: dict = {}
        
        # 计算索引表长度
        self.num_slots = self.total_length // min_resolution
        
        # 计算最大层级 (base_block_size / min_resolution = 2^max_level)
        self.max_level = self._compute_max_level()
        
        # 初始化定长索引表
        self.index_table: List[IndexEntry] = self._create_initial_index_table()
        
        print(f"[GridManager] Initialized: total_length={self.total_length}, "
              f"base_block_size={base_block_size}, min_resolution={min_resolution}, "
              f"num_slots={self.num_slots}, max_level={self.max_level}")
    
    def _compute_max_level(self) -> int:
        """
        计算最大层级数。
        
        base_block_size / min_resolution = 2^max_level
        例如: 100/25 = 4 = 2^2, max_level=2
        
        Returns:
            max_level: 最大层级数
        """
        ratio = self.base_block_size // self.min_resolution
        level = 0
        while (1 << level) < ratio:
            level += 1
        return level
    
    def get_length_by_level(self, level: int) -> int:
        """
        根据层级码获取对应的块长度。
        
        level=0 -> base_block_size
        level=1 -> base_block_size / 2
        level=2 -> base_block_size / 4
        ...
        
        Args:
            level: 层级码
            
        Returns:
            块长度
        """
        return self.base_block_size >> level  # 等价于 base_block_size / (2^level)
    
    def _create_initial_index_table(self) -> List[IndexEntry]:
        """
        创建初始索引表（基于base_block_size的均匀划分）。
        
        每个base block覆盖 base_block_size // min_resolution 个槽位。
        
        Returns:
            index_table: 定长索引表
        """
        index_table = []
        slots_per_base_block = self.base_block_size // self.min_resolution
        
        for slot_idx in range(self.num_slots):
            # 计算该槽位属于哪个base block
            time_start = slot_idx * self.min_resolution
            base_block_idx = time_start // self.base_block_size
            
            # base block的左右边界Grid ID
            left_id = base_block_idx
            right_id = base_block_idx + 1
            index_table.append(IndexEntry(left_id=left_id, right_id=right_id, level_code=0))
        
        return index_table
    
    def get_block_info(self, t: int) -> IndexEntry:
        """
        O(1) 查询：获取时间点 t 所属块的边界信息。
        
        时间复杂度: O(1) - 直接通过 idx = t // min_resolution 计算索引。
        
        Args:
            t: 时间点
            
        Returns:
            IndexEntry: 包含 (left_id, right_id, level_code)
        """
        idx = t // self.min_resolution
        if idx >= self.num_slots:
            idx = self.num_slots - 1
        return self.index_table[idx]
    
    def _allocate_patch_node(self) -> int:
        """
        分配一个新的patch节点ID。
        
        Returns:
            新节点的全局ID
        """
        # 优先复用因回滚释放的槽位
        if self._free_patch_ids:
            recycled_id = self._free_patch_ids.pop()
            return recycled_id
        
        new_id = self.model.num_base_nodes + self.patch_counter
        self.patch_counter += 1
        
        if self.patch_counter > self.model.max_patch_nodes:
            raise RuntimeError(f"Patch grid overflow: allocated {self.patch_counter} nodes, "
                             f"max is {self.model.max_patch_nodes}")
        
        return new_id
    
    def release_patch_node(self, global_id: int) -> None:
        """
        将一个 patch 节点的全局 ID 归还到 free-list，供后续分裂复用。
        
        调用前提：调用方已确保 index_table 中没有任何槽位引用该 ID。
        """
        self._free_patch_ids.append(global_id)

    # ──────────────────────────────────────────────────────────────
    # Aux Token 操作（多 token 谱细化）
    # ──────────────────────────────────────────────────────────────
    def add_aux_token(self, time: int, count: int = 1) -> List[int]:
        """
        给覆盖时间点 `time` 的块追加 `count` 个 aux token。
        
        Aux token 的存储复用 patch_grid（与分裂中点共用）。
        Aux token 初始化为零向量（保证未训练前对输出贡献为零）。
        同一个块的所有 slot 共享这份 aux 列表。
        
        Args:
            time: 块内任意时间点
            count: 追加的 token 数量
        
        Returns:
            new_ids: 新分配的 aux token 全局 ID 列表
        """
        info = self.get_block_info(time)
        key = (info.left_id, info.right_id, info.level_code)
        
        new_ids: List[int] = []
        for _ in range(count):
            new_id = self._allocate_patch_node()
            # 零初始化：保证训练初期 aux token 对输出贡献为零
            self.model.zero_init_node(new_id)
            new_ids.append(new_id)
        
        existing = self._aux_map.get(key, ())
        self._aux_map[key] = existing + tuple(new_ids)
        return new_ids
    
    def add_aux_tokens_to_block(
        self,
        left_id: int,
        right_id: int,
        level_code: int,
        count: int = 1,
    ) -> List[int]:
        """
        与 add_aux_token 等价，但通过 (left_id, right_id, level_code) 直接定位块。
        用于循环遍历 unique_blocks 时调用。
        """
        key = (left_id, right_id, level_code)
        new_ids: List[int] = []
        for _ in range(count):
            new_id = self._allocate_patch_node()
            self.model.zero_init_node(new_id)
            new_ids.append(new_id)
        existing = self._aux_map.get(key, ())
        self._aux_map[key] = existing + tuple(new_ids)
        return new_ids
    
    def get_aux_ids(self, left_id: int, right_id: int, level_code: int = None) -> tuple:
        """
        查询指定块的 aux token ID 列表。
        
        decode_batch 内部调用：从 (left_id, right_id) 反查 aux token 列表。
        
        Args:
            left_id: 块的左边界 grid 节点 ID
            right_id: 块的右边界 grid 节点 ID
            level_code: 可选；若不传，按 (left_id, right_id) 模糊匹配（取第一个）
        
        Returns:
            tuple[int]: aux token 全局 ID 列表（可能为空）
        """
        if level_code is not None:
            return self._aux_map.get((left_id, right_id, level_code), ())
        # 模糊匹配（同一对 left/right 通常 level 唯一）
        for k, v in self._aux_map.items():
            if k[0] == left_id and k[1] == right_id:
                return v
        return ()
    
    def clear_aux_tokens(self, left_id: int, right_id: int, level_code: int) -> int:
        """
        移除指定块的所有 aux token，将其 ID 归还 free pool。
        
        Returns:
            释放的 aux token 数量
        """
        key = (left_id, right_id, level_code)
        if key not in self._aux_map:
            return 0
        aux_ids = self._aux_map.pop(key)
        for aid in aux_ids:
            self.release_patch_node(aid)
        return len(aux_ids)
    
    def get_aux_stats(self) -> dict:
        """统计 aux token 分布。"""
        if not self._aux_map:
            return {"num_blocks_with_aux": 0, "total_aux_tokens": 0, "M_distribution": {}}
        m_dist: dict = {}
        total = 0
        for v in self._aux_map.values():
            m = len(v)
            m_dist[m] = m_dist.get(m, 0) + 1
            total += m
        return {
            "num_blocks_with_aux": len(self._aux_map),
            "total_aux_tokens": total,
            "M_distribution": m_dist,
        }

    def split_block_at_time(self, split_time: int, trend_value: Optional[float] = None) -> int:
        """
        在指定时间点分裂块：插入新Grid节点，更新索引表和层级码。
        
        这是底层分裂方法，直接在指定的 split_time 处分裂。
        split_time 必须是 min_resolution 的整数倍。
        
        分裂逻辑（以 split_time=25, base=100, min=25 为例）：
        1. 计算 split_idx = 25 // 25 = 1
        2. 找到当前覆盖该位置的块（通过向左扫描找到 block_start_idx）
        3. 分配新的 patch node ID
        4. 更新 index_table：
           - 左半区 [block_start_idx, split_idx): right_id = new_id
           - 右半区 [split_idx, block_end_idx]: left_id = new_id
           - 所有槽位的 level_code 不变（因为分裂后块长度不再是 2 的幂次）
        
        Args:
            split_time: 分裂时间点（必须是 min_resolution 的整数倍）
            trend_value: 新节点的trend初始值（默认使用raw_data[split_time]）
            
        Returns:
            新分配的节点ID
        """
        # 验证分裂点对齐
        if split_time % self.min_resolution != 0:
            raise ValueError(f"split_time {split_time} must be aligned to min_resolution {self.min_resolution}")
        
        split_idx = split_time // self.min_resolution
        
        if split_idx <= 0 or split_idx >= self.num_slots:
            raise ValueError(f"split_idx {split_idx} out of valid range [1, {self.num_slots - 1})")
        
        # 获取当前块信息
        current_entry = self.index_table[split_idx]
        old_left_id = current_entry.left_id
        old_right_id = current_entry.right_id
        current_level = current_entry.level_code
        
        # 检查是否已达到最大层级
        if current_level >= self.max_level:
            raise ValueError(f"Cannot split further: already at max level {self.max_level}")
        
        # 分裂前：若该块挂有 aux token，全部释放回 free pool（子块从零开始）
        old_key = (old_left_id, old_right_id, current_level)
        if old_key in self._aux_map:
            for aux_id in self._aux_map.pop(old_key):
                self.release_patch_node(aux_id)
        
        # === 先找到当前块覆盖的所有槽位范围（需要用于计算 alpha）===
        block_start_idx = split_idx
        block_end_idx = split_idx
        
        # 向左扫描
        while block_start_idx > 0:
            prev_entry = self.index_table[block_start_idx - 1]
            if (prev_entry.left_id == old_left_id and 
                prev_entry.right_id == old_right_id and
                prev_entry.level_code == current_level):
                block_start_idx -= 1
            else:
                break
        
        # 向右扫描
        while block_end_idx < self.num_slots - 1:
            next_entry = self.index_table[block_end_idx + 1]
            if (next_entry.left_id == old_left_id and 
                next_entry.right_id == old_right_id and
                next_entry.level_code == current_level):
                block_end_idx += 1
            else:
                break
        
        # 计算块的物理时间范围
        block_start_time = block_start_idx * self.min_resolution
        block_end_time = (block_end_idx + 1) * self.min_resolution
        
        # 计算 alpha：分裂点的相对位置 (0~1)
        # alpha = (split_time - block_start) / (block_end - block_start)
        block_len = block_end_time - block_start_time
        alpha = (split_time - block_start_time) / block_len if block_len > 0 else 0.5
        
        # 分配新节点
        new_node_id = self._allocate_patch_node()
        
        # 初始化新节点（trend + ACORN-style 继承 z_left 的 context）
        if trend_value is None:
            trend_value = self.raw_data[split_time].item()
        self.model.init_patch_node(new_node_id, trend_value, old_left_id, old_right_id, alpha)
        
        # 计算新层级（分裂后层级 +1）
        new_level = current_level + 1
        
        # 更新左半部分：[block_start_idx, split_idx) -> right_id = new_node_id, level_code = new_level
        for i in range(block_start_idx, split_idx):
            self.index_table[i].right_id = new_node_id
            self.index_table[i].level_code = new_level
        
        # 更新右半部分：[split_idx, block_end_idx] -> left_id = new_node_id, level_code = new_level
        for i in range(split_idx, block_end_idx + 1):
            self.index_table[i].left_id = new_node_id
            self.index_table[i].level_code = new_level
        
        return new_node_id
    
    def split_block(self, split_time: int, trend_value: Optional[float] = None) -> int:
        """
        [兼容接口] 在指定时间点分裂块。
        
        直接调用 split_block_at_time。
        
        Args:
            split_time: 分裂时间点（必须是 min_resolution 的整数倍）
            trend_value: 新节点的trend初始值（默认使用raw_data[split_time]）
            
        Returns:
            新分配的节点ID
        """
        return self.split_block_at_time(split_time, trend_value)
    
    def compute_split_point_argmax(
        self,
        start_time: int,
        end_time: int,
        error_curve: torch.Tensor,
        min_edge_distance: int = None
    ) -> int:
        """
        基于误差曲线计算最优分裂点（Argmax 误差导向）。
        
        逻辑步骤：
        1. 找到误差最大值的时间索引：raw_split_idx = argmax(error_curve)
        2. 网格吸附 (Snapping)：对齐到 min_resolution 的整数倍
        3. 边界熔断保护：如果分裂点太靠近边缘，回退到二分法
        
        Args:
            start_time: 块的起始时间
            end_time: 块的结束时间
            error_curve: 误差曲线 [block_len]，abs(y_pred - y_true)
            min_edge_distance: 最小边缘距离（默认为 min_resolution）
            
        Returns:
            split_time: 对齐后的分裂时间点（绝对时间）
        """
        if min_edge_distance is None:
            min_edge_distance = self.min_resolution
        
        block_len = end_time - start_time
        
        # Step 1: 找到误差最大值的相对索引
        raw_split_idx = torch.argmax(error_curve).item()
        
        # Step 2: 转换为绝对时间
        raw_split_time = start_time + raw_split_idx
        
        # Step 3: 网格吸附 (Snapping) - 对齐到 min_resolution 的整数倍
        # 使用四舍五入方式
        snapped_split_time = round(raw_split_time / self.min_resolution) * self.min_resolution
        
        # 记录是否使用了 Argmax 还是回退到二分法
        used_argmax = True
        midpoint = (start_time + end_time) // 2
        midpoint_snapped = (midpoint // self.min_resolution) * self.min_resolution
        
        # Step 4: 边界熔断保护
        # 分裂点必须严格在 (start_time, end_time) 内部
        # 且分裂后两个子块的长度都必须 >= min_resolution
        if snapped_split_time <= start_time or snapped_split_time >= end_time:
            # 分裂点在边界上或边界外，回退到二分法
            snapped_split_time = midpoint_snapped
            used_argmax = False
        
        # 检查分裂后的子块长度是否 >= min_resolution
        left_len = snapped_split_time - start_time
        right_len = end_time - snapped_split_time
        
        if left_len < self.min_resolution or right_len < self.min_resolution:
            # 子块太小，回退到二分法
            snapped_split_time = midpoint_snapped
            used_argmax = False
            
            # 再次检查
            left_len = snapped_split_time - start_time
            right_len = end_time - snapped_split_time
            
            if left_len < self.min_resolution or right_len < self.min_resolution:
                raise ValueError(f"Cannot find valid split point for block [{start_time}, {end_time}): "
                               f"block too small (len={end_time - start_time}, min_resolution={self.min_resolution})")
        
        # 调试日志：显示 Argmax 是否生效
        if used_argmax and snapped_split_time != midpoint_snapped:
            print(f"      [Argmax] raw={raw_split_time}, snapped={snapped_split_time}, midpoint={midpoint_snapped}")
        elif not used_argmax:
            print(f"      [Fallback] argmax={raw_split_time} -> midpoint={midpoint_snapped}")
        
        return snapped_split_time
    
    def get_block_range(self, t: int) -> Tuple[int, int]:
        """
        获取时间点 t 所属块的实际时间范围。
        
        通过扫描相邻槽位找到连续的同ID和level区域。
        
        Args:
            t: 时间点
            
        Returns:
            (start_time, end_time): 块的时间范围
        """
        idx = t // self.min_resolution
        if idx >= self.num_slots:
            idx = self.num_slots - 1
        
        entry = self.index_table[idx]
        left_id, right_id, level_code = entry.left_id, entry.right_id, entry.level_code
        
        # 向左扫描
        start_idx = idx
        while start_idx > 0:
            prev = self.index_table[start_idx - 1]
            if (prev.left_id == left_id and prev.right_id == right_id and 
                prev.level_code == level_code):
                start_idx -= 1
            else:
                break
        
        # 向右扫描
        end_idx = idx
        while end_idx < self.num_slots - 1:
            next_entry = self.index_table[end_idx + 1]
            if (next_entry.left_id == left_id and next_entry.right_id == right_id and
                next_entry.level_code == level_code):
                end_idx += 1
            else:
                break
        
        start_time = start_idx * self.min_resolution
        end_time = (end_idx + 1) * self.min_resolution
        
        return start_time, end_time
    
    def get_num_slots(self) -> int:
        """获取索引表槽位数量。"""
        return self.num_slots
    
    def get_num_patch_nodes(self) -> int:
        """获取已分配的patch节点数量。"""
        return self.patch_counter
    
    def get_statistics(self) -> dict:
        """
        获取当前网格的统计信息。
        
        Returns:
            包含槽位数量、节点数量、层级分布等信息的字典
        """
        # 统计不同的块和层级分布
        unique_blocks = set()
        level_distribution = {}
        
        for entry in self.index_table:
            unique_blocks.add((entry.left_id, entry.right_id, entry.level_code))
            level_distribution[entry.level_code] = level_distribution.get(entry.level_code, 0) + 1
        
        return {
            "num_slots": self.num_slots,
            "num_unique_blocks": len(unique_blocks),
            "num_base_nodes": self.model.num_base_nodes,
            "num_patch_nodes": self.patch_counter,
            "total_nodes": self.model.num_base_nodes + self.patch_counter,
            "min_resolution": self.min_resolution,
            "base_block_size": self.base_block_size,
            "max_level": self.max_level,
            "level_distribution": level_distribution,
        }
    
    def get_all_unique_blocks(self) -> List[Tuple[int, int, int, int, int]]:
        """
        获取所有唯一的块信息。
        
        Returns:
            blocks: [(start_time, end_time, left_id, right_id, level_code), ...]
        """
        blocks = []
        i = 0
        
        while i < self.num_slots:
            entry = self.index_table[i]
            left_id, right_id, level_code = entry.left_id, entry.right_id, entry.level_code
            start_idx = i
            
            # 找到连续相同ID和level的槽位
            while i < self.num_slots:
                curr = self.index_table[i]
                if (curr.left_id == left_id and 
                    curr.right_id == right_id and
                    curr.level_code == level_code):
                    i += 1
                else:
                    break
            
            end_idx = i
            start_time = start_idx * self.min_resolution
            end_time = end_idx * self.min_resolution
            
            blocks.append((start_time, end_time, left_id, right_id, level_code))
        
        return blocks
    
    def get_task_list(self) -> List[Tuple[int, int, int, int]]:
        """
        兼容旧接口：返回类似task_list格式的块列表（不含level_code）。
        
        Returns:
            task_list: [(start_time, end_time, left_id, right_id), ...]
        """
        blocks = self.get_all_unique_blocks()
        return [(s, e, l, r) for s, e, l, r, _ in blocks]
    
    def trim_unused_patch_nodes(self):
        """
        裁剪未使用的 patch 节点，释放内存。
        
        调用 GlobalGridStorage.trim_patch_grid() 方法，
        只保留实际分配的 patch 节点。
        
        建议在训练完成后、保存模型前调用。
        """
        self.model.trim_patch_grid(self.patch_counter)
