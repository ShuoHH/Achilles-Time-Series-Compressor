// BlockCodec patch 版 C++ accessor —— 实现。
// 加速策略：仅一个核心加速 —— 分桶 GEMM（base 与 patch 各一次 cblas_sgemm），
// 配 sin_basis 缓存（GEMM 的必要配套）。无 OpenMP、无 Sleef，与原版核心一致。
#include "bc/blockcodec.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <set>
#include <stdexcept>
#include <tuple>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef BC_HAS_BLAS
  #include <cblas.h>
#endif

#include <chrono>
#include <thread>

namespace bc {

namespace {
using ProfClock = std::chrono::steady_clock;
inline double prof_sec(ProfClock::time_point t0) {
    return std::chrono::duration<double>(ProfClock::now() - t0).count();
}
}  // namespace

namespace {

std::vector<char> read_all(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<char> buf(static_cast<std::size_t>(n));
    f.read(buf.data(), n);
    return buf;
}

template <typename T>
std::vector<T> read_vec(const std::string& path) {
    auto raw = read_all(path);
    std::vector<T> out(raw.size() / sizeof(T));
    std::memcpy(out.data(), raw.data(), out.size() * sizeof(T));
    return out;
}

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

// Y[K,M] = X[K,N] · W^T[N,M] + b。BLAS 时一次 sgemm；否则三重循环（无 OpenMP）。
void linear_batched(const float* X, const float* W, const float* b,
                    int K, int N, int M, float* Y) {
#ifdef BC_HAS_BLAS
    for (int k = 0; k < K; ++k)
        std::memcpy(Y + (std::size_t)k * M, b, M * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                K, M, N, 1.0f, X, N, W, N, 1.0f, Y, M);
#else
    for (int k = 0; k < K; ++k) {
        const float* xk = X + (std::size_t)k * N;
        float* yk = Y + (std::size_t)k * M;
        for (int m = 0; m < M; ++m) {
            float acc = b[m];
            const float* wr = W + (std::size_t)m * N;
            for (int n = 0; n < N; ++n) acc += wr[n] * xk[n];
            yk[m] = acc;
        }
    }
#endif
}

std::vector<std::vector<float>> enumerate_partitions(int depth) {
    std::vector<std::vector<std::vector<float>>> cache(depth + 1);
    std::vector<bool> done(depth + 1, false);
    std::function<std::vector<std::vector<float>>(int)> rec =
        [&](int d) -> std::vector<std::vector<float>> {
        if (done[d]) return cache[d];
        std::vector<std::vector<float>> result;
        result.push_back({1.0f});
        if (d > 0) {
            auto sub = rec(d - 1);
            for (auto& L : sub)
                for (auto& R : sub) {
                    std::vector<float> merged;
                    for (float x : L) merged.push_back(x * 0.5f);
                    for (float x : R) merged.push_back(x * 0.5f);
                    result.push_back(std::move(merged));
                }
        }
        std::vector<std::vector<float>> uniq;
        std::set<std::vector<float>> seen;
        for (auto& p : result)
            if (seen.insert(p).second) uniq.push_back(p);
        cache[d] = uniq; done[d] = true;
        return uniq;
    };
    return rec(depth);
}

}  // namespace

Meta load_meta(const std::string& path) {
    auto raw = read_all(path);
    if (raw.size() != (std::size_t)kMetaBinSize)
        throw std::runtime_error("meta.bin size != 68");
    Meta m;
    std::memcpy(&m, raw.data(), kMetaBinSize);
    if (m.format_version != kFormatVersion)
        throw std::runtime_error("meta.bin format_version mismatch");
    return m;
}

DecoderWeights DecoderWeights::load(const std::string& path) {
    auto raw = read_all(path);
    const char* p = raw.data();
    DecoderWeights w;
    int32_t hdr[4];
    std::memcpy(hdr, p, 16); p += 16;
    w.in_dim = hdr[0]; w.hidden_dim = hdr[1]; w.num_freqs = hdr[2];
    auto take = [&](std::vector<float>& v, int n) {
        v.resize(n);
        std::memcpy(v.data(), p, (std::size_t)n * sizeof(float));
        p += (std::size_t)n * sizeof(float);
    };
    const int D = w.in_dim, H = w.hidden_dim, F = w.num_freqs;
    take(w.to_v0_W, D); take(w.to_v0_b, 1);
    take(w.to_v1_W, D); take(w.to_v1_b, 1);
    take(w.to_coeff0_W, H * D); take(w.to_coeff0_b, H);
    take(w.to_coeff2_W, F * H); take(w.to_coeff2_b, F);
    take(w.freqs, F);
    return w;
}

Codebook::Codebook(int max_depth) {
    partitions_ = enumerate_partitions(max_depth);
}

void Codebook::leaf_bounds(int code, int block_start, int block_len,
                           std::vector<std::pair<int, int>>& out) const {
    out.clear();
    const auto& fr = partitions_[code];
    int t = block_start;
    for (std::size_t i = 0; i < fr.size(); ++i) {
        int seg = (int)std::lround(fr[i] * block_len);
        out.emplace_back(t, t + seg);
        t += seg;
    }
    if (!out.empty()) out.back().second = block_start + block_len;
}

int Codebook::locate_leaf(int code, int block_start, int block_len, int t) const {
    const auto& fr = partitions_[code];
    int rel = t - block_start, acc = 0;
    for (std::size_t i = 0; i < fr.size(); ++i) {
        int seg = (int)std::lround(fr[i] * block_len);
        if (rel < acc + seg) return (int)i;
        acc += seg;
    }
    return (int)fr.size() - 1;
}

BlockAccessor::BlockAccessor(const std::string& dir)
    : meta_(load_meta(dir + "/meta.bin")),
      dec_(DecoderWeights::load(dir + "/decoder.bin")),
      codebook_(meta_.max_depth) {
    grid_ = read_vec<float>(dir + "/grid.bin");
    block_offset_ = read_vec<int32_t>(dir + "/block_offset.bin");
    block_meta_ = read_vec<int32_t>(dir + "/block_meta.bin");
    coeff_pool_ = read_vec<float>(dir + "/coeff_pool.bin");
    if (meta_.code_bytes == 2) {
        auto raw = read_vec<uint16_t>(dir + "/struct_code.bin");
        struct_code_.assign(raw.begin(), raw.end());
    } else {
        auto raw = read_vec<uint8_t>(dir + "/struct_code.bin");
        struct_code_.assign(raw.begin(), raw.end());
    }
    // 残差文件（format v3，可选）
    has_residual_ = (meta_.has_residual != 0);
    if (has_residual_) {
        res_step_ = meta_.res_step;
        res_bits_ = read_vec<uint8_t>(dir + "/residual_bits.bin");
        res_rmin_ = read_vec<float>(dir + "/residual_rmin.bin");
        res_len_ = read_vec<int32_t>(dir + "/residual_len.bin");
        res_bitpos_ = read_vec<int64_t>(dir + "/residual_bitpos.bin");
        res_stream_ = read_vec<uint8_t>(dir + "/residual_stream.bin");
    }
}

// 解出第 row 个叶子残差（MSB-first，与 numpy packbits 一致），写入 dst[0..len)。
void BlockAccessor::decode_residual_leaf(int row, float* dst) const {
    int L = res_len_[row];
    float rmn = res_rmin_[row];
    int b = res_bits_[row];
    if (b == 0) {
        for (int j = 0; j < L; ++j) dst[j] = rmn;
        return;
    }
    // 向量化取码：每个码用"整字节聚合 + 移位掩码"一次取出 b 比特，
    // 替代逐比特内层循环（b<=16，故最多聚合 3 字节到 uint32）。MSB-first 语义不变。
    const int64_t base = res_bitpos_[row];
    const uint32_t mask = (1u << b) - 1u;
    const uint8_t* S = res_stream_.data();
    for (int j = 0; j < L; ++j) {
        int64_t bitpos = base + (int64_t)j * b;
        int64_t byte0 = bitpos >> 3;
        int total = (int)(bitpos & 7) + b;          // 窗口内有效位数 <= 7+16=23
        int nbytes = (total + 7) >> 3;              // 跨越 1..3 字节
        uint32_t acc = 0;
        for (int i = 0; i < nbytes; ++i)
            acc = (acc << 8) | (uint32_t)S[byte0 + i];
        uint32_t q = (acc >> (nbytes * 8 - total)) & mask;   // 右对齐取 b 位
        dst[j] = rmn + (float)q * res_step_;
    }
}

const std::vector<float>& BlockAccessor::sin_basis(
    int len, int F, const float* freqs,
    std::unordered_map<int32_t, std::vector<float>>& cache) const {
    auto it = cache.find(len);
    if (it != cache.end()) return it->second;
    std::vector<float> sb((std::size_t)len * F);
    for (int i = 0; i < len; ++i) {
        float t = (len == 1) ? 0.0f : (float)i / (len - 1);
        float* row = sb.data() + (std::size_t)i * F;
        for (int f = 0; f < F; ++f) row[f] = std::sin(freqs[f] * t);
    }
    auto res = cache.emplace(len, std::move(sb));
    return res.first->second;
}

void BlockAccessor::reconstruct_blocks(
    const std::vector<int32_t>& block_ids_in,
    std::unordered_map<int32_t, std::vector<float>>& out,
    AccessProfile* prof) const {
    ProfClock::time_point _ts;
    if (prof) _ts = ProfClock::now();
    std::vector<int32_t> bids;
    {
        std::set<int32_t> seen;
        for (auto b : block_ids_in)
            if (seen.insert(b).second) bids.push_back(b);
    }
    if (prof) { prof->t_dedup += prof_sec(_ts); prof->n_unique_blocks += (long)bids.size(); }

    if (!parallel_) {
        // ---- 串行：base 先建 out，patch 直接叠加到 out（默认路径，主对比表用）----
        compute_base(bids, out, prof);
        compute_patch(bids, out, prof);   // 串行模式下 patch_buf 即 out，直接 +=
        apply_residual(bids, out, prof);  // 残差叠加（误差精确重建，误差≤ε）
        return;
    }

    // ---- 并行：base 流与 patch 流无数据依赖，分两线程同时跑 ----
    // base 写 out；patch 写独立 patch_buf（不碰 base 值）；最后串行合并。
    std::unordered_map<int32_t, std::vector<float>> patch_buf;
    AccessProfile pb, pp;   // 两线程各自的子计时（避免数据竞争）
    AccessProfile* pbp = prof ? &pb : nullptr;
    AccessProfile* ppp = prof ? &pp : nullptr;

    ProfClock::time_point _tp;
    if (prof) _tp = ProfClock::now();
    std::thread th_base([&] { compute_base(bids, out, pbp); });
    compute_patch(bids, patch_buf, ppp);   // 当前线程承载 patch 流
    th_base.join();
    double wall = prof ? prof_sec(_tp) : 0.0;

    // 合并：out[bid] += patch_buf[bid]（廉价，O(touched points)）
    if (prof) _ts = ProfClock::now();
    for (auto& kv : patch_buf) {
        auto it = out.find(kv.first);
        if (it == out.end()) continue;
        float* w = it->second.data();
        const float* r = kv.second.data();
        int n = (int)std::min(it->second.size(), kv.second.size());
        for (int t = 0; t < n; ++t) w[t] += r[t];
    }
    if (prof) {
        // 把两线程的子计时累加进总 profile（用于"假如串行各阶段耗时"对照），
        // 同时记录真实并行墙钟 t_parallel_recon。
        prof->t_base_mlp += pb.t_base_mlp;
        prof->t_base_gemm += pb.t_base_gemm;
        prof->t_patch_gemm += pp.t_patch_gemm;
        prof->n_leaves_touched = pp.n_leaves_touched;
        prof->t_parallel_recon += wall + prof_sec(_ts);  // 含合并开销
        prof->was_parallel = true;
    }
    apply_residual(bids, out, prof);   // 残差叠加（误差精确重建，误差≤ε）
}

// 残差叠加：对每个块的每个叶子，解出 EDWB 残差加到 out[bid] 对应区间。
// 按位宽分桶解包（保持批处理友好）；has_residual_=false 时直接跳过（base+patch 近似）。
void BlockAccessor::apply_residual(
    const std::vector<int32_t>& bids,
    std::unordered_map<int32_t, std::vector<float>>& out,
    AccessProfile* prof) const {
    if (!has_residual_) return;
    ProfClock::time_point _ts;
    if (prof) _ts = ProfClock::now();
    std::vector<float> buf;
    std::vector<std::pair<int, int>> bounds;
    for (auto bid : bids) {
        int code = struct_code_[bid];
        int bstart = block_meta_[bid * 3 + 0];
        int blen = std::min(meta_.base_block_size, meta_.T - bstart);
        codebook_.leaf_bounds(code, bstart, blen, bounds);
        int lo_row = block_offset_[bid];
        // 该块真实叶子数（前缀和差），防止 leaf_bounds 与存储叶子数不一致时越界
        int n_leaves = block_offset_[bid + 1] - lo_row;
        auto it = out.find(bid);
        if (it == out.end()) continue;
        float* w = it->second.data();
        int n_iter = std::min((int)bounds.size(), n_leaves);
        for (int i = 0; i < n_iter; ++i) {
            int row = lo_row + i;
            int L = res_len_[row];
            if (L <= 0) continue;
            buf.resize(L);
            decode_residual_leaf(row, buf.data());
            int lo = bounds[i].first - bstart;
            for (int t = 0; t < L && lo + t < blen; ++t) w[lo + t] += buf[t];
        }
    }
    if (prof) prof->t_residual += prof_sec(_ts);  // 残差解包单独计时（诚实计时）
}

// base 流：z → decoder → base 波形（覆盖写 out）。
void BlockAccessor::compute_base(
    const std::vector<int32_t>& bids,
    std::unordered_map<int32_t, std::vector<float>>& out,
    AccessProfile* prof) const {
    ProfClock::time_point _ts;
    const int D = meta_.feature_dim, H = dec_.hidden_dim, F = dec_.num_freqs;

    std::unordered_map<int, std::vector<int32_t>> len_bucket;
    for (auto bid : bids) {
        int bstart = block_meta_[bid * 3 + 0];
        int blen = std::min(meta_.base_block_size, meta_.T - bstart);
        len_bucket[blen].push_back(bid);
    }
    for (auto& kv : len_bucket) {
        int blen = kv.first;
        auto& group = kv.second;
        int K = (int)group.size();
        std::vector<float> zs((std::size_t)K * D);
        for (int k = 0; k < K; ++k) {
            int left = block_meta_[group[k] * 3 + 1];
            std::memcpy(zs.data() + (std::size_t)k * D, grid_row(left), D * sizeof(float));
        }
        if (prof) _ts = ProfClock::now();
        std::vector<float> V0(K), V1(K);
        for (int k = 0; k < K; ++k) {
            const float* z = zs.data() + (std::size_t)k * D;
            float v0 = dec_.to_v0_b[0], v1 = dec_.to_v1_b[0];
            for (int j = 0; j < D; ++j) { v0 += dec_.to_v0_W[j] * z[j]; v1 += dec_.to_v1_W[j] * z[j]; }
            V0[k] = v0; V1[k] = v1;
        }
        std::vector<float> Hm((std::size_t)K * H);
        linear_batched(zs.data(), dec_.to_coeff0_W.data(), dec_.to_coeff0_b.data(), K, D, H, Hm.data());
        for (auto& v : Hm) v = silu(v);
        std::vector<float> A((std::size_t)K * F);
        linear_batched(Hm.data(), dec_.to_coeff2_W.data(), dec_.to_coeff2_b.data(), K, H, F, A.data());
        if (prof) prof->t_base_mlp += prof_sec(_ts);

        if (prof) _ts = ProfClock::now();
        const auto& sinB = sin_basis(blen, F, dec_.freqs.data(), sin_cache_base_);
        std::vector<float> osc((std::size_t)K * blen);
#ifdef BC_HAS_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    K, blen, F, 1.0f, A.data(), F, sinB.data(), F, 0.0f, osc.data(), blen);
#else
        for (int k = 0; k < K; ++k)
            for (int t = 0; t < blen; ++t) {
                const float* a = A.data() + (std::size_t)k * F;
                const float* sb = sinB.data() + (std::size_t)t * F;
                float s = 0; for (int f = 0; f < F; ++f) s += a[f] * sb[f];
                osc[(std::size_t)k * blen + t] = s;
            }
#endif
        for (int k = 0; k < K; ++k) {
            std::vector<float> wave(blen);
            for (int t = 0; t < blen; ++t) {
                float tv = (blen == 1) ? 0.0f : (float)t / (blen - 1);
                wave[t] = (1.0f - tv) * V0[k] + tv * V1[k] + osc[(std::size_t)k * blen + t];
            }
            out[group[k]] = std::move(wave);
        }
        if (prof) prof->t_base_gemm += prof_sec(_ts);
    }
}

// patch 流：δ → Φ → patch 修正。
//   串行模式（patch_buf 传入 = out）：直接 w[t] += r[t] 叠加到 base。
//   并行模式（patch_buf 独立）：先为每个块零初始化缓冲，再 += 修正，合并在外层做。
void BlockAccessor::compute_patch(
    const std::vector<int32_t>& bids,
    std::unordered_map<int32_t, std::vector<float>>& patch_buf,
    AccessProfile* prof) const {
    ProfClock::time_point _ts;
    if (prof) _ts = ProfClock::now();
    const int Kp = meta_.K_fixed + 2;
    const bool independent = parallel_;  // 并行时 patch_buf 与 out 分离，需自建缓冲

    std::unordered_map<int, std::vector<std::tuple<int32_t, int, int>>> seg_bucket;
    std::vector<std::pair<int, int>> bounds;
    for (auto bid : bids) {
        int code = struct_code_[bid];
        int bstart = block_meta_[bid * 3 + 0];
        int blen = std::min(meta_.base_block_size, meta_.T - bstart);
        codebook_.leaf_bounds(code, bstart, blen, bounds);
        int lo_row = block_offset_[bid];
        for (std::size_t i = 0; i < bounds.size(); ++i) {
            int row = lo_row + (int)i;
            const float* d = coeff_pool_.data() + (std::size_t)row * Kp;
            bool nz = false;
            for (int j = 0; j < Kp; ++j) if (d[j] != 0.0f) { nz = true; break; }
            if (!nz) continue;
            int seg = bounds[i].second - bounds[i].first;
            // 并行模式：为该块分配独立 patch 缓冲（长 blen，零初始化）
            if (independent && patch_buf.find(bid) == patch_buf.end())
                patch_buf.emplace(bid, std::vector<float>(blen, 0.0f));
            seg_bucket[seg].push_back(std::make_tuple(bid, bounds[i].first - bstart, row));
        }
    }
    for (auto& kv : seg_bucket) {
        int seg = kv.first;
        auto& items = kv.second;
        int M = (int)items.size();
        if (prof) prof->n_leaves_touched += M;
        std::vector<float> Phi((std::size_t)seg * Kp);
        for (int i = 0; i < seg; ++i) {
            float t = (seg == 1) ? 0.0f : (float)i / (seg - 1);
            float* row = Phi.data() + (std::size_t)i * Kp;
            row[0] = 1.0f - t; row[1] = t;
            for (int k = 1; k <= meta_.K_fixed; ++k) row[1 + k] = std::sin((float)M_PI * k * t);
        }
        std::vector<float> dmat((std::size_t)M * Kp);
        for (int m = 0; m < M; ++m)
            std::memcpy(dmat.data() + (std::size_t)m * Kp,
                        coeff_pool_.data() + (std::size_t)std::get<2>(items[m]) * Kp,
                        Kp * sizeof(float));
        std::vector<float> recon((std::size_t)M * seg);
#ifdef BC_HAS_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    M, seg, Kp, 1.0f, dmat.data(), Kp, Phi.data(), Kp, 0.0f, recon.data(), seg);
#else
        for (int m = 0; m < M; ++m)
            for (int t = 0; t < seg; ++t) {
                const float* dd = dmat.data() + (std::size_t)m * Kp;
                const float* pp = Phi.data() + (std::size_t)t * Kp;
                float s = 0; for (int j = 0; j < Kp; ++j) s += dd[j] * pp[j];
                recon[(std::size_t)m * seg + t] = s;
            }
#endif
        for (int m = 0; m < M; ++m) {
            int bid = std::get<0>(items[m]);
            int lo = std::get<1>(items[m]);
            float* w = patch_buf[bid].data() + lo;   // 串行=out[bid]，并行=独立缓冲
            const float* r = recon.data() + (std::size_t)m * seg;
            for (int t = 0; t < seg; ++t) w[t] += r[t];
        }
    }
    if (prof) prof->t_patch_gemm += prof_sec(_ts);
}

float BlockAccessor::query_point(int32_t t) const {
    int bid = t / meta_.base_block_size;
    std::unordered_map<int32_t, std::vector<float>> blocks;
    reconstruct_blocks({bid}, blocks);
    int bstart = block_meta_[bid * 3 + 0];
    return blocks[bid][t - bstart];
}

std::vector<float> BlockAccessor::query_batch(const int32_t* times, int n) const {
    std::vector<int32_t> bids(n);
    for (int i = 0; i < n; ++i) bids[i] = times[i] / meta_.base_block_size;
    std::unordered_map<int32_t, std::vector<float>> blocks;
    reconstruct_blocks(bids, blocks);
    std::vector<float> out(n);
    for (int i = 0; i < n; ++i) {
        int bid = bids[i];
        int bstart = block_meta_[bid * 3 + 0];
        out[i] = blocks[bid][times[i] - bstart];
    }
    return out;
}

std::vector<float> BlockAccessor::query_range(int32_t ts, int32_t te) const {
    if (te <= ts) return {};
    int b0 = ts / meta_.base_block_size;
    int b1 = (te - 1) / meta_.base_block_size;
    std::vector<int32_t> bids;
    for (int b = b0; b <= b1; ++b) bids.push_back(b);
    std::unordered_map<int32_t, std::vector<float>> blocks;
    reconstruct_blocks(bids, blocks);
    std::vector<float> out(te - ts);
    for (int b = b0; b <= b1; ++b) {
        int bstart = block_meta_[b * 3 + 0];
        int blen = (int)blocks[b].size();
        int a = std::max(bstart, ts), e = std::min(bstart + blen, te);
        for (int t = a; t < e; ++t) out[t - ts] = blocks[b][t - bstart];
    }
    return out;
}

std::vector<float> BlockAccessor::decompress_all() const {
    std::vector<int32_t> bids(meta_.num_blocks);
    for (int b = 0; b < meta_.num_blocks; ++b) bids[b] = b;
    std::unordered_map<int32_t, std::vector<float>> blocks;
    reconstruct_blocks(bids, blocks);
    std::vector<float> out(meta_.T, 0.0f);
    for (int b = 0; b < meta_.num_blocks; ++b) {
        int bstart = block_meta_[b * 3 + 0];
        auto& w = blocks[b];
        for (int t = 0; t < (int)w.size() && bstart + t < meta_.T; ++t)
            out[bstart + t] = w[t];
    }
    return out;
}

// ===================== profile 版（分阶段计时） =====================

std::vector<float> BlockAccessor::query_batch(
    const std::vector<int32_t>& times, AccessProfile* prof) const {
    int n = (int)times.size();
    std::vector<int32_t> bids(n);
    for (int i = 0; i < n; ++i) bids[i] = times[i] / meta_.base_block_size;
    std::unordered_map<int32_t, std::vector<float>> blocks;
    reconstruct_blocks(bids, blocks, prof);
    ProfClock::time_point _ts;
    if (prof) _ts = ProfClock::now();
    std::vector<float> out(n);
    for (int i = 0; i < n; ++i) {
        int bid = bids[i];
        int bstart = block_meta_[bid * 3 + 0];
        out[i] = blocks[bid][times[i] - bstart];
    }
    if (prof) { prof->t_scatter += prof_sec(_ts); prof->n_queries += n; }
    return out;
}

std::vector<float> BlockAccessor::query_range(
    int32_t ts, int32_t te, AccessProfile* prof) const {
    if (te <= ts) return {};
    int b0 = ts / meta_.base_block_size;
    int b1 = (te - 1) / meta_.base_block_size;
    std::vector<int32_t> bids;
    for (int b = b0; b <= b1; ++b) bids.push_back(b);
    std::unordered_map<int32_t, std::vector<float>> blocks;
    reconstruct_blocks(bids, blocks, prof);
    ProfClock::time_point _ts;
    if (prof) _ts = ProfClock::now();
    std::vector<float> out(te - ts);
    for (int b = b0; b <= b1; ++b) {
        int bstart = block_meta_[b * 3 + 0];
        int blen = (int)blocks[b].size();
        int a = std::max(bstart, ts), e = std::min(bstart + blen, te);
        for (int t = a; t < e; ++t) out[t - ts] = blocks[b][t - bstart];
    }
    if (prof) { prof->t_scatter += prof_sec(_ts); prof->n_queries += (te - ts); }
    return out;
}

std::vector<float> BlockAccessor::decompress_all(AccessProfile* prof) const {
    std::vector<int32_t> bids(meta_.num_blocks);
    for (int b = 0; b < meta_.num_blocks; ++b) bids[b] = b;
    std::unordered_map<int32_t, std::vector<float>> blocks;
    reconstruct_blocks(bids, blocks, prof);
    ProfClock::time_point _ts;
    if (prof) _ts = ProfClock::now();
    std::vector<float> out(meta_.T, 0.0f);
    for (int b = 0; b < meta_.num_blocks; ++b) {
        int bstart = block_meta_[b * 3 + 0];
        auto& w = blocks[b];
        for (int t = 0; t < (int)w.size() && bstart + t < meta_.T; ++t)
            out[bstart + t] = w[t];
    }
    if (prof) { prof->t_scatter += prof_sec(_ts); prof->n_queries += meta_.T; }
    return out;
}

}  // namespace bc
