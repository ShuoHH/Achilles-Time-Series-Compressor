// BlockCodec patch 版 C++ —— 随机访问时间消耗分布分析（profile）。
//
// 用法: ./profile_blockcodec <export_dir>
//
// 把随机访问的总时间拆成各阶段，打印 ASCII 柱状分布图，定位瓶颈：
//   dedup     : 块去重 / locate（block_id = t / base，整型运算）
//   base_mlp  : base 系数 MLP（to_v0/v1 + to_coeff 两次 linear）
//   base_gemm : base 振荡 A·sinB^T 分桶 GEMM + ramp 组装
//   patch_gemm: patch δ·Φ^T 分桶 GEMM + 叠加
//   scatter   : 从重建块按查询点/段散射取值
//
// 计时只在传入 AccessProfile* 时启用；正常 benchmark 路径（bench_main）零开销，
// 公平性不受影响。本程序仅用于"我们方法的内部开销构成"分析，不进对比表。
#include "bc/blockcodec.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace bc;
using Clock = std::chrono::steady_clock;

static double sec_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// 画一行 ASCII 柱：标签 | 绝对耗时 ns | 百分比条 | 百分比
static void bar_row(const char* label, double sec_stage, double sec_total,
                    long n_queries, int width = 40) {
    double frac = (sec_total > 0) ? (sec_stage / sec_total) : 0.0;
    int k = (int)(frac * width + 0.5);
    if (k > width) k = width;
    double ns_per_q = (n_queries > 0) ? (sec_stage / n_queries * 1e9) : 0.0;
    std::string bar(k, '#');
    bar.append(width - k, '.');
    std::printf("    %-11s %10.1f ns | %s | %5.1f%%\n",
                label, ns_per_q, bar.c_str(), frac * 100.0);
}

static void print_breakdown(const char* title, const AccessProfile& p) {
    double rec = p.total_recon();      // 重建阶段（dedup+mlp+base_gemm+patch_gemm）
    double tot = p.total();            // 含 scatter
    std::printf("\n  %s\n", title);
    std::printf("    unique_blocks=%ld, queries=%ld, leaves_touched=%ld\n",
                p.n_unique_blocks, p.n_queries, p.n_leaves_touched);
    std::printf("    total = %.3f ms  (%.1f ns/query)\n",
                tot * 1e3, p.n_queries ? tot / p.n_queries * 1e9 : 0.0);
    bar_row("dedup/loc",  p.t_dedup,      tot, p.n_queries);
    bar_row("base_mlp",   p.t_base_mlp,   tot, p.n_queries);
    bar_row("base_gemm",  p.t_base_gemm,  tot, p.n_queries);
    bar_row("patch_gemm", p.t_patch_gemm, tot, p.n_queries);
    bar_row("residual",   p.t_residual,   tot, p.n_queries);
    bar_row("scatter",    p.t_scatter,    tot, p.n_queries);
    std::printf("    [recon %.1f%% | scatter %.1f%%]\n",
                tot > 0 ? rec / tot * 100 : 0.0,
                tot > 0 ? p.t_scatter / tot * 100 : 0.0);
}

// 累加单次 profile 到汇总（保留计数为最后一次的去重值）
static void accumulate(AccessProfile& dst, const AccessProfile& one) {
    dst.t_dedup += one.t_dedup; dst.t_base_mlp += one.t_base_mlp;
    dst.t_base_gemm += one.t_base_gemm; dst.t_patch_gemm += one.t_patch_gemm;
    dst.t_residual += one.t_residual;
    dst.t_scatter += one.t_scatter; dst.t_parallel_recon += one.t_parallel_recon;
    dst.n_unique_blocks = one.n_unique_blocks;
    dst.n_queries += one.n_queries;
    dst.n_leaves_touched = one.n_leaves_touched;
    dst.was_parallel = one.was_parallel;
}

// base/patch 并行收益分析：串行各阶段 vs 并行墙钟。
static void print_parallel_gain(const AccessProfile& serial) {
    double t_base = serial.t_base_mlp + serial.t_base_gemm;
    double t_patch = serial.t_patch_gemm;
    double t_sum = t_base + t_patch;                       // 串行重建(base+patch)
    double t_ideal = std::max(t_base, t_patch);            // 理想并行下限
    double upper_gain = (t_sum > 0) ? (1.0 - t_ideal / t_sum) * 100 : 0.0;
    std::printf("\n  ── base/patch 并行收益分析（基于串行各阶段实测）──\n");
    std::printf("    base 流 (mlp+gemm) : %.3f ms\n", t_base * 1e3);
    std::printf("    patch 流 (gemm)    : %.3f ms\n", t_patch * 1e3);
    std::printf("    串行 base+patch    : %.3f ms\n", t_sum * 1e3);
    std::printf("    理想并行 max(.)    : %.3f ms  → 理论上限收益 %.1f%%\n",
                t_ideal * 1e3, upper_gain);
    std::printf("    (理论上限 = 较小流 / 总和；实测见下方 parallel 墙钟)\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <export_dir>\n", argv[0]);
        return 1;
    }
    std::string dir = argv[1];
    BlockAccessor acc(dir);
    const int T = acc.T();
    std::printf("Loaded BlockCodec: T=%d, num_blocks=%d\n", T, acc.num_blocks());
    std::printf("=== RANDOM-ACCESS TIME BREAKDOWN (profile only, not a benchmark) ===\n");

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> uni(0, T - 1);

    // ---------- 随机点访问：各 N 的阶段分布 + 并行收益分析 ----------
    std::printf("\n############ POINT ACCESS ############");
    for (int N : {100, 1000, 10000, 100000}) {
        std::vector<int32_t> times(N);
        for (int i = 0; i < N; ++i) times[i] = uni(rng);
        acc.set_parallel(false);
        acc.query_batch(times);  // warmup
        int rounds = (N <= 1000) ? 20 : 5;
        AccessProfile prof;
        for (int r = 0; r < rounds; ++r) {
            AccessProfile one;
            acc.query_batch(times, &one);
            accumulate(prof, one);
        }
        char title[64];
        std::snprintf(title, sizeof(title), "[N=%d点, %d rounds, serial]", N, rounds);
        print_breakdown(title, prof);
        print_parallel_gain(prof);

        // 实测串行 vs 并行墙钟（warmup 后，去掉 profile 打点开销，纯墙钟）
        acc.set_parallel(false);
        auto t0 = Clock::now();
        for (int r = 0; r < rounds; ++r) acc.query_batch(times);
        double s_ser = sec_since(t0) / rounds;
        acc.set_parallel(true);
        acc.query_batch(times);  // 并行 warmup
        t0 = Clock::now();
        for (int r = 0; r < rounds; ++r) acc.query_batch(times);
        double s_par = sec_since(t0) / rounds;
        acc.set_parallel(false);
        std::printf("    [wall] serial %.3f ms | parallel %.3f ms | speedup %.2fx (省 %.1f%%)\n",
                    s_ser * 1e3, s_par * 1e3, s_ser / s_par,
                    (1.0 - s_par / s_ser) * 100);
    }

    // ---------- 随机段访问：各宽度的阶段分布 ----------
    std::printf("\n############ RANGE SCAN ############");
    for (int W : {100, 1000, 10000, 100000}) {
        if (W >= T) continue;
        std::uniform_int_distribution<int> us(0, T - W - 1);
        std::vector<int> starts;
        for (int i = 0; i < 5; ++i) starts.push_back(us(rng));
        acc.set_parallel(false);
        acc.query_range(starts[0], starts[0] + W);  // warmup
        int rounds = 5;
        AccessProfile prof;
        for (int r = 0; r < rounds; ++r)
            for (int s : starts) {
                AccessProfile one;
                acc.query_range(s, s + W, &one);
                accumulate(prof, one);
            }
        char title[80];
        std::snprintf(title, sizeof(title), "[W=%d段, %d starts x %d rounds, serial]",
                      W, (int)starts.size(), rounds);
        print_breakdown(title, prof);
        print_parallel_gain(prof);
    }

    // ---------- 全量解压：阶段分布 + 串行/并行墙钟 ----------
    std::printf("\n############ DECOMPRESSION ############");
    acc.set_parallel(false);
    acc.decompress_all();  // warmup
    {
        AccessProfile prof;
        int rounds = 3;
        for (int r = 0; r < rounds; ++r) {
            AccessProfile one;
            acc.decompress_all(&one);
            accumulate(prof, one);
        }
        print_breakdown("[full decompress, 3 rounds, serial]", prof);
        print_parallel_gain(prof);

        acc.set_parallel(false);
        auto t0 = Clock::now();
        for (int r = 0; r < rounds; ++r) acc.decompress_all();
        double s_ser = sec_since(t0) / rounds;
        acc.set_parallel(true);
        acc.decompress_all();  // warmup
        t0 = Clock::now();
        for (int r = 0; r < rounds; ++r) acc.decompress_all();
        double s_par = sec_since(t0) / rounds;
        acc.set_parallel(false);
        std::printf("    [wall] serial %.2f ms | parallel %.2f ms | speedup %.2fx (省 %.1f%%)\n",
                    s_ser * 1e3, s_par * 1e3, s_ser / s_par,
                    (1.0 - s_par / s_ser) * 100);
    }

    std::printf("\n说明：base_gemm/patch_gemm 占比高 → 瓶颈在 GEMM（计算密集，符合预期）；\n");
    std::printf("      dedup/scatter 占比高 → 瓶颈在 O(N) 整型/散射（访存密集）。\n");
    std::printf("      并行收益 = base/patch 两条无依赖流分 2 线程；理论上限=较小流占比，\n");
    std::printf("      讲的是统一系数表示带来的结构解耦，非堆核加速，默认关闭、不进主对比表。\n");
    return 0;
}
