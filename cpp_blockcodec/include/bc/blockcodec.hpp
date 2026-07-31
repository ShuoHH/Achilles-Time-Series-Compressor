// BlockCodec patch 版 C++ accessor —— 头文件（声明）。
//
// 复用原版 NeurTS C++ 的 fourier_forward_batched 加速思路：
//   base 预测：K 个块按块长分桶，A[K,F] @ sin_basis^T[F,T] 一次 cblas_sgemm。
//   patch 修正：叶子按段长分桶，δ[M,K+2] @ Phi^T[K+2,seg_len] 一次 cblas_sgemm。
//   前向次数 ∝ 唯一块数 / 段数，而非查询数。
//
// 随机访问全程 O(1) 定位：
//   block_id = t / base_block_size
//   code     = struct_code[block_id]                 (隐式树形状)
//   leaf_lo  = block_offset[block_id]                 (前缀和，行寻址)
//   叶子边界 = codebook.leaf_bounds(code)             (隐式树解出，不存边界)
#pragma once

#include "bc/meta.hpp"
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

namespace bc {

// ---- 分阶段计时（仅 profile 时启用，正常访问路径零开销）----
// 把随机访问拆成各阶段，定位瓶颈在哪：
//   dedup    : 块去重 / locate（block_id = t/base，O(N) 整型运算）
//   base_mlp : base 系数 MLP（to_v0/v1 + to_coeff 两次 linear）
//   base_gemm: base 振荡 A·sinB^T 分桶 GEMM + ramp 组装
//   patch_gemm: patch δ·Φ^T 分桶 GEMM + 叠加
//   residual : EDWB 残差解包（per-leaf 位宽解码）+ 叠加（误差精确重建）
//   scatter  : 从重建块按查询点/段散射取值
struct AccessProfile {
    double t_dedup = 0, t_base_mlp = 0, t_base_gemm = 0,
           t_patch_gemm = 0, t_residual = 0, t_scatter = 0;
    double t_parallel_recon = 0;   // 并行执行 base||patch 的墙钟时间（仅并行模式填）
    long   n_unique_blocks = 0, n_queries = 0, n_leaves_touched = 0;
    bool   was_parallel = false;
    void reset() { *this = AccessProfile{}; }
    double total_recon() const {
        return t_dedup + t_base_mlp + t_base_gemm + t_patch_gemm + t_residual;
    }
    double total() const { return total_recon() + t_scatter; }
};

// ---- FourierDecoder 权重（与原版 decoder.bin 同布局）----
struct DecoderWeights {
    int32_t in_dim = 0, hidden_dim = 0, num_freqs = 0;
    std::vector<float> to_v0_W, to_v0_b, to_v1_W, to_v1_b;
    std::vector<float> to_coeff0_W, to_coeff0_b, to_coeff2_W, to_coeff2_b;
    std::vector<float> freqs;
    static DecoderWeights load(const std::string& path);
};

// ---- 隐式二叉树码本：结构码 <-> 叶子划分（与 Python enumerate_partitions 一致）----
class Codebook {
public:
    explicit Codebook(int max_depth);
    int n_codes() const { return static_cast<int>(partitions_.size()); }
    // 给定结构码 + 块边界，返回各叶子 [start,end) 绝对边界。
    void leaf_bounds(int code, int block_start, int block_len,
                     std::vector<std::pair<int, int>>& out) const;
    int num_leaves(int code) const {
        return static_cast<int>(partitions_[code].size());
    }
    // O(叶子数) 定位 t 落在第几个叶子（叶子数 <= 2^depth，常数）。
    int locate_leaf(int code, int block_start, int block_len, int t) const;
private:
    std::vector<std::vector<float>> partitions_;  // 各划分的叶子长度比例
};

// ---- 主 accessor ----
class BlockAccessor {
public:
    explicit BlockAccessor(const std::string& export_dir);

    int32_t T() const { return meta_.T; }
    int32_t num_blocks() const { return meta_.num_blocks; }

    // base/patch 任务级并行开关（默认关 = 串行，主对比表用串行保持单一加速）。
    // 开启后用 2 个线程分别承载 base 流与 patch 流（二者无数据依赖），
    // 最后串行合并。讲的是"统一系数表示使 base/patch 解耦、天然可并行"这一
    // 结构性质，非堆核；分析用，不进主对比表。
    void set_parallel(bool on) const { parallel_ = on; }
    bool parallel() const { return parallel_; }

    // 单点（先去重块，内部走批量）
    float query_point(int32_t t) const;

    // 随机点批量：N 个点 → 去重块 → 分桶 GEMM → 散射。out 长度 N。
    std::vector<float> query_batch(const int32_t* times, int n) const;
    std::vector<float> query_batch(const std::vector<int32_t>& times) const {
        return query_batch(times.data(), (int)times.size());
    }

    // 随机段：[t_start,t_end) 跨到的块整体分桶 GEMM → 拼接裁剪。
    std::vector<float> query_range(int32_t t_start, int32_t t_end) const;

    // 全量解压。
    std::vector<float> decompress_all() const;

    // ---- profile 版（额外传 AccessProfile* 累计各阶段耗时；prof=nullptr 时零开销）----
    std::vector<float> query_batch(const std::vector<int32_t>& times, AccessProfile* prof) const;
    std::vector<float> query_range(int32_t t_start, int32_t t_end, AccessProfile* prof) const;
    std::vector<float> decompress_all(AccessProfile* prof) const;

private:
    // 批量重建给定块的完整波形（base 分桶GEMM + patch 分桶GEMM）。
    // 输出写入 out_blocks（每块连续 block_len，按 block_id 升序排布在 row_of[bid]）。
    void reconstruct_blocks(const std::vector<int32_t>& block_ids,
                            std::unordered_map<int32_t, std::vector<float>>& out,
                            AccessProfile* prof = nullptr) const;

    // base 流：z → decoder → base 波形（写入 out，覆盖式）。与 patch 流无数据依赖。
    void compute_base(const std::vector<int32_t>& bids,
                      std::unordered_map<int32_t, std::vector<float>>& out,
                      AccessProfile* prof) const;
    // patch 流：δ → Φ → patch 修正（写入独立的 patch_buf，不碰 base）。
    // 与 base 流无数据依赖，可与 compute_base 并行执行。
    void compute_patch(const std::vector<int32_t>& bids,
                       std::unordered_map<int32_t, std::vector<float>>& patch_buf,
                       AccessProfile* prof) const;
    // 残差流：解 EDWB 残差加到 out（误差精确重建）。has_residual_=false 时跳过。
    void apply_residual(const std::vector<int32_t>& bids,
                        std::unordered_map<int32_t, std::vector<float>>& out,
                        AccessProfile* prof) const;

    const float* grid_row(int32_t node_id) const {
        return grid_.data() + static_cast<std::size_t>(node_id) * meta_.feature_dim;
    }

    Meta meta_{};
    DecoderWeights dec_;
    std::vector<float> grid_;            // [num_nodes, feature_dim]
    std::vector<int32_t> struct_code_;   // [num_blocks]
    std::vector<int32_t> block_offset_;  // [num_blocks+1]
    std::vector<int32_t> block_meta_;    // [num_blocks*3] (start,left,right)
    std::vector<float> coeff_pool_;      // [num_leaves, K_fixed+2]
    Codebook codebook_;
    mutable bool parallel_ = false;      // base/patch 任务级并行开关（默认关）

    // 残差（EDWB，方案B）：per-leaf 位宽 + r_min + 比特起点前缀和 + 位流
    bool has_residual_ = false;
    float res_step_ = 0.0f;              // 量化步长 = 2ε
    std::vector<uint8_t> res_bits_;      // [num_leaves] 每叶子位宽
    std::vector<float> res_rmin_;        // [num_leaves] 每叶子 r_min
    std::vector<int32_t> res_len_;       // [num_leaves] 段长
    std::vector<int64_t> res_bitpos_;    // [num_leaves+1] 比特起点前缀和
    std::vector<uint8_t> res_stream_;    // 位流
    // 解出第 row 个叶子的残差，写入 dst[0..len)
    void decode_residual_leaf(int row, float* dst) const;

    // 每块长度 / 每段长度的 sin_basis 缓存（base 用 [block_len,F]，patch 用 [seg_len,K]）
    mutable std::unordered_map<int32_t, std::vector<float>> sin_cache_base_;
    mutable std::unordered_map<int32_t, std::vector<float>> sin_cache_patch_;
    const std::vector<float>& sin_basis(int len, int F, const float* freqs,
                                        std::unordered_map<int32_t, std::vector<float>>& cache) const;
};

}  // namespace bc
