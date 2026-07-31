// BlockCodec patch 版 meta.bin（68 字节固定头）。
// 与 export_blockcodec_cpp.py 的 BINARY FORMAT 完全对应；改动须同步 format_version。
#pragma once
#include <cstdint>
#include <string>

namespace bc {

constexpr int32_t kFormatVersion = 3;
constexpr int kMetaBinSize = 80;

#pragma pack(push, 1)
struct Meta {
    int32_t format_version;   // 0
    int32_t T;                // 4
    int32_t base_block_size;  // 8
    int32_t min_resolution;   // 12
    int32_t max_depth;        // 16
    int32_t n_codes;          // 20
    int32_t code_bytes;       // 24  (1 或 2)
    int32_t num_blocks;       // 28
    int32_t num_leaves;       // 32
    int32_t num_nodes;        // 36
    int32_t feature_dim;      // 40
    int32_t K_fixed;          // 44
    int32_t dec_in_dim;       // 48
    int32_t dec_hidden_dim;   // 52
    int32_t dec_num_freqs;    // 56
    float   scaler_mean;      // 60
    float   scaler_std;       // 64
    int32_t has_residual;     // 68  (1=有残差文件)
    int32_t res_stream_bytes; // 72  (residual_stream.bin 字节数)
    float   res_step;         // 76  (量化步长 = 2ε)
};
#pragma pack(pop)

static_assert(sizeof(Meta) == kMetaBinSize, "Meta must be 80 bytes");

Meta load_meta(const std::string& path);

}  // namespace bc
