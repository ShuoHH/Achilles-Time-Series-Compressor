# BlockCodec patch 版 随机访问（C++ + Python）

patch 版数据结构（结构码隐式树 + 定长系数池 + EDWB 残差）的随机访问实现。
C++ 版复用原版 NeurTS 的分桶 GEMM 加速（base 与 patch 各一次 cblas_sgemm）。

## 文件
- `include/bc/meta.hpp`, `include/bc/blockcodec.hpp` — 头
- `src/blockcodec.cpp` — accessor 实现（码本/分桶GEMM/单点/段/解压）
- `src/bench_main.cpp` — benchmark（ns/query + MB/s）
- `src/profile_main.cpp` — 时间消耗分布（各阶段 ASCII 柱状图，仅分析用）
- `src/dump_main.cpp` — 导出 decompress_all 结果供数值比对
- `CMakeLists.txt`, `verify_cpp_vs_py.py`

对应 Python 版：`../cross_models/block_codec.py` + `../cross_models/block_accessor.py`
导出脚本：`../export_blockcodec_cpp.py`（生成 C++ 读取的二进制）

## 流程（服务器上）

### 1. 导出二进制（Python，需训练好且 patch_fixed 的 checkpoint）
```bash
python export_blockcodec_cpp.py \
    --ckpt checkpoints/NeurTS_BT_bs512_mr32_hd256_rb5_itr0 \
    --out  checkpoints/BT_blockcodec_cpp
```
产出：meta.bin grid.bin decoder.bin struct_code.bin block_offset.bin
      block_meta.bin coeff_pool.bin meta.json

### 2. 编译 C++
```bash
cd cpp_blockcodec
mkdir build && cd build
# 有 OpenBLAS（强烈推荐，提速 30-100x）：
cmake .. -DBC_USE_BLAS=ON -DCMAKE_BUILD_TYPE=Release
# 无 BLAS（仍可跑，慢）：
# cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

### 3. 跑 benchmark（对齐论文表格）
```bash
./bench_blockcodec ../../checkpoints/BT_blockcodec_cpp
```
输出：随机点 ns/query（N=1/100/1k/10k/100k）、随机段 MB/s（W=10..100k）、解压 MB/s。

### 4. 数值比对（确认 C++ == Python）
```bash
./dump_blockcodec ../../checkpoints/BT_blockcodec_cpp /tmp/cpp_full.bin
cd ../..
python cpp_blockcodec/verify_cpp_vs_py.py \
    --ckpt checkpoints/NeurTS_BT_bs512_mr32_hd256_rb5_itr0 \
    --export checkpoints/BT_blockcodec_cpp \
    --cpp /tmp/cpp_full.bin
```
期望：max abs diff < 1e-3（fp32 + libm 差异范围内）→ PASS。

## Python 版 benchmark（对照 C++）
```bash
python bench_random_access.py --ckpt checkpoints/NeurTS_BT_bs512_mr32_hd256_rb5_itr0
```

## 时间消耗分布（profile，定位瓶颈）
```bash
./profile_blockcodec ../../checkpoints/BT_blockcodec_cpp
```
把随机访问拆成 5 个阶段，打印 ASCII 柱状分布图：
- `dedup/loc`  : 块去重 / locate（block_id=t/base，整型运算）
- `base_mlp`   : base 系数 MLP（to_v0/v1 + to_coeff 两次 linear）
- `base_gemm`  : base 振荡 A·sinB^T 分桶 GEMM + ramp 组装
- `patch_gemm` : patch δ·Φ^T 分桶 GEMM + 叠加
- `scatter`    : 从重建块按查询点/段取值

判读：base_gemm/patch_gemm 占比高 → 计算密集（GEMM，符合预期）；
dedup/scatter 占比高 → 访存密集（O(N) 整型/散射）。
注：计时只在 profile 路径启用，正常 `bench_blockcodec` 零开销，对比表的公平性不受影响。

## 注意
- 残差（EDWB）当前未接入访问路径，重建为 base + patch；接残差只是每段多一次
  查表+加法，不改速度量级（但会让误差严格可控）。
- `-march=native` 在服务器本机编译没问题；跨机分发需改成具体架构。
