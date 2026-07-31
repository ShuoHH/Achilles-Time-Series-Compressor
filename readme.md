# 阿克琉斯: Neural Time-Series Storage System

阿克琉斯是一个面向一维时间序列的**神经压缩与随机访问**系统。它把长序列按固定分辨率切分成块，为每个块学习一个低维隐向量（z 向量），再用一个共享的神经解码器把 `(隐向量, 块内位置)` 映射回原始信号。压缩后的数据由「解码器权重 + 每块隐向量 + 索引表 + 可选残差」构成，支持**随机点查询**和**范围查询**，并提供高性能 C++ codec 用于对齐吞吐指标。


## 核心组件

| 模块 | 说明 |
| --- | --- |
| `main_neurts.py` | 训练入口 |
| `neurts_accessor.py` | 随机访问器（从训练好的实验或导出文件构建，编译索引表为连续数组） |
| `cross_exp/exp_neurts.py` | 训练 / 评估实验流程 |
| `cross_models/` | 解码器、分块 codec、残差编码、网格管理等核心实现 |
| `cross_models/decoders/` | 可切换的解码器架构（fourier / siren / tcn / feature_strip / acorn1d / transformer） |
| `cross_models/block_codec.py`, `block_accessor.py` | 分块编解码与随机访问的 Python 参考实现 |
| `cpp_blockcodec/` | 上述 codec 的高性能 C++ 实现（分桶 GEMM，支持 OpenBLAS 加速） |
| `export_blockcodec_cpp.py` | 把训练好的 checkpoint 导出为 C++ 读取的二进制 |
| `bench_random_access.py` | 随机访问 benchmark（点查询 ns/query、范围查询 / 解压 MB/s） |


## 环境依赖

- Python 3.7+
- torch, numpy, pandas, einops, matplotlib
- （访问）OpenBLAS + CMake，用于编译 C++ codec

```
pip install -r requirements.txt
```

## 快速开始

### 1. 准备数据
把时间序列数据集（CSV）放入 `datasets/`，用 `--data_col` 指定要压缩的列。

### 2. 训练 / 压缩
以下是与 `all_checkpoints/` 中已训练模型一致的典型配置（傅里叶解码器，单阶段训练）：

```
python -u main_neurts.py --data BT --data_path BT.csv --data_col 1 --base_block_size 512 --min_resolution 32 --decoder_type fourier --num_freqs 256 --total_dim 32 --hidden_dim 256 --pretrain_epochs 2000 --split_threshold 0.05 --eval_threshold 0.05 --error_mode absolute --batch_size 32 --learning_rate 1e-3 --quant_bits 8 --patch_split_commit > BT.out
```

训练完成后，checkpoint 保存在 `checkpoints/NeurTS_<data>_bs<block>_mr<res>_hd<hidden>_rb<blocks>_itr0/`，
其中包含 `args.json`、`checkpoint.pth`、`manager_state.pkl`、`scale_statistic.pkl` 四个文件。

主要参数（完整列表见 `python main_neurts.py --help`）：

| 参数 | 说明 | 典型值 |
| --- | --- | --- |
| `--data` / `--data_path` / `--data_col` | 数据集名 / 文件名 / 列索引 | — |
| `--base_block_size` | 基础块大小 | 512 |
| `--min_resolution` | 索引表最小分辨率 | 32 / 64 |
| `--decoder_type` | 解码器架构 | `fourier` |
| `--hidden_dim` | 解码器隐藏维度 | 256 |
| `--num_freqs` | 傅里叶频率分量数 F（Nyquist 上限 = block_size/2） | 256 |
| `--context_dim` / `--trend_dim` | 上下文 / 趋势特征维度 | 31 / 1 |
| `--num_res_blocks` | 解码器层数 | 5 |
| `--error_mode` | 误差模式：`absolute` 绝对误差 / `relative` 相对百分比 | `absolute` |
| `--eval_threshold` | 评估用误差阈值 | 0.05 |
| `--quant_bits` | 隐向量量化位宽 | 8 |
| `--learning_rate` | 学习率 | 1e-3 |

## 随机访问 Benchmark

Python 版：
```
python bench_random_access.py --ckpt checkpoints/NeurTS_BT_bs512_mr32_hd256_rb5_itr0
```

C++ 版（对齐论文吞吐指标）——详细流程见 [`cpp_blockcodec/README.md`](cpp_blockcodec/README.md)：
```bash
# 1. 导出二进制（需训练好的 checkpoint）
python export_blockcodec_cpp.py --ckpt checkpoints/<ckpt_name> --out checkpoints/<name>_cpp

# 2. 编译 C++（默认启用 BLAS 加速）
cd cpp_blockcodec && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j

# 3. 跑 benchmark（随机点 ns/query、随机段 MB/s、解压 MB/s）
./bench_blockcodec ../../checkpoints/<name>_cpp

# 4. 数值比对，确认 C++ 与 Python 一致
python cpp_blockcodec/verify_cpp_vs_py.py --ckpt ... --export ... --cpp /tmp/cpp_full.bin
```

## 分析与作图脚本

| 脚本 | 用途 |
| --- | --- |
| `batch_compression_ratio.py` | 批量统计压缩比 |
| `component_breakdown.py` | 存储 / 时间开销分解 |
| `plot_pie.py` | 存储占比 + 时间占比饼图 |
| `plot_model_ablation_scatter.py` | 解码器消融的压缩比 vs 速度散点图 |




## 致谢



随机访问解码器设计参考了 ACORN（Adaptive Coordinate Networks）的自适应坐标网络思想，感谢原作者的工作。
