"""
NeurTS: Neural Time-Series Storage System

统一入口模块，导出所有NeurTS组件。

文件结构:
- grid.py: GlobalGridStorage (双层网格存储)
- decoders/: 可插拔解码器（BaseDecoder, FourierDecoder, ...）
- neurts_model.py: NeurTSModel (主模型，通过 decoder_type 选择解码器)
- neurts_dataset.py: NeurTSTaskDataset (任务驱动数据集)
- neurts_manager.py: GridManager (索引管理器)

使用方式:
    from models.neurts import (
        GlobalGridStorage,
        NeurTSModel,
        NeurTSTaskDataset,
        GridManager
    )
"""

from .grid import GlobalGridStorage
from .neurts_model import NeurTSModel
from .neurts_dataset import NeurTSTaskDataset
from .neurts_manager import GridManager, IndexEntry

__all__ = [
    'GlobalGridStorage',
    'NeurTSModel',
    'NeurTSTaskDataset',
    'GridManager',
    'IndexEntry',
]


# ============================================================
# 测试代码：验证各组件协同工作
# ============================================================

def test_neurts_system():
    """测试NeurTS系统各组件的协同工作。"""
    import torch
    
    print("=" * 60)
    print("NeurTS System Integration Test")
    print("=" * 60)
    
    # 1. 创建模拟原始数据
    print("\n[1] Creating mock raw data...")
    total_length = 6400
    base_block_size = 100
    min_resolution = 50
    raw_data = torch.sin(torch.linspace(0, 20 * 3.14159, total_length))
    print(f"    Raw data shape: {raw_data.shape}")
    print(f"    Base block size: {base_block_size}")
    print(f"    Min resolution: {min_resolution}")
    
    # 2. 创建GlobalGridStorage
    print("\n[2] Creating GlobalGridStorage...")
    grid_storage = GlobalGridStorage(
        raw_data=raw_data,
        block_size=base_block_size,
        trend_dim=1,
        context_dim=63,
        max_patch_nodes=500,
        init_mode='endpoint'
    )
    print(f"    Base nodes: {grid_storage.num_base_nodes}")
    print(f"    Max patch nodes: {grid_storage.max_patch_nodes}")
    print(f"    Feature dim: {grid_storage.feature_dim}")
    
    # 3. 创建NeurTSModel
    print("\n[3] Creating NeurTSModel...")
    model = NeurTSModel(
        grid_storage=grid_storage,
        max_block_size=base_block_size,
        hidden_dim=64,
        pe_dim=32,
        num_res_blocks=4
    )
    print(f"    Model created successfully")
    
    # 4. 测试前向传播
    print("\n[4] Testing forward pass...")
    left_ids = torch.tensor([0, 1, 2, 3])
    right_ids = torch.tensor([1, 2, 3, 4])
    output = model(left_ids, right_ids)
    print(f"    Input: left_ids={left_ids.tolist()}, right_ids={right_ids.tolist()}")
    print(f"    Output shape: {output.shape}")
    assert output.shape == (4, 1, base_block_size)
    
    # 5. 创建GridManager
    print("\n[5] Creating GridManager...")
    manager = GridManager(
        model=model,
        raw_data=raw_data,
        base_block_size=base_block_size,
        min_resolution=min_resolution
    )
    stats = manager.get_statistics()
    print(f"    Num slots: {stats['num_slots']}")
    print(f"    Num unique blocks: {stats['num_unique_blocks']}")
    print(f"    Min resolution: {stats['min_resolution']}")
    
    # 6. 测试 O(1) 查询
    print("\n[6] Testing O(1) get_block_info...")
    for t in [0, 50, 100, 150]:
        entry = manager.get_block_info(t)
        print(f"    t={t}: left_id={entry.left_id}, right_id={entry.right_id}")
    
    # 7. 创建NeurTSTaskDataset
    print("\n[7] Creating NeurTSTaskDataset...")
    dataset = NeurTSTaskDataset(
        raw_data=raw_data,
        task_list=manager.get_task_list(),
        max_block_size=base_block_size
    )
    print(f"    Dataset length: {len(dataset)}")
    
    # 8. 测试数据集采样
    print("\n[8] Testing dataset sampling...")
    padded_waveform, valid_mask, left_id, right_id = dataset[0]
    print(f"    Sample 0: waveform shape={padded_waveform.shape}, "
          f"valid_length={int(valid_mask.sum())}, "
          f"left_id={left_id}, right_id={right_id}")
    
    # 9. 测试分裂操作
    print("\n[9] Testing split_block...")
    split_time = 150
    entry_before = manager.get_block_info(split_time)
    print(f"    Before split at t={split_time}: level={entry_before.level_code}")
    
    new_node_id = manager.split_block(split_time)
    print(f"    Split at t={split_time}, new node ID: {new_node_id}")
    
    # 10. 验证分裂后的索引表
    print("\n[10] Verifying index table after split...")
    for t in [100, 125, 150, 175]:
        entry = manager.get_block_info(t)
        block_len = manager.get_length_by_level(entry.level_code)
        print(f"    t={t}: left_id={entry.left_id}, right_id={entry.right_id}, "
              f"level={entry.level_code}, block_len={block_len}")
    
    stats_after = manager.get_statistics()
    print(f"\n    Stats after split:")
    print(f"    - Num unique blocks: {stats_after['num_unique_blocks']}")
    print(f"    - Num patch nodes: {stats_after['num_patch_nodes']}")
    print(f"    - Level distribution: {stats_after['level_distribution']}")
    
    # 11. 测试内存裁剪
    print("\n[11] Testing patch grid trimming...")
    mem_before = model.get_memory_usage()
    print(f"    Before trim: {mem_before['num_patch_nodes']} patch nodes, "
          f"{mem_before['patch_grid_kb']:.2f} KB")
    
    manager.trim_unused_patch_nodes()
    
    mem_after = model.get_memory_usage()
    print(f"    After trim: {mem_after['num_patch_nodes']} patch nodes, "
          f"{mem_after['patch_grid_kb']:.2f} KB")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_neurts_system()
