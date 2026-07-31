// BlockCodec patch 版 C++ benchmark。
// 用法: ./bench_blockcodec <export_dir>
// 输出: 随机点 ns/query、随机段 MB/s、解压 MB/s（对齐论文表格）。
#include "bc/blockcodec.hpp"

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using namespace bc;
using Clock = std::chrono::steady_clock;

static double sec_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
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

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> uni(0, T - 1);

    // ---- 随机点 ----
    std::printf("\n=== POINT ACCESS (ns/query) ===\n");
    for (int N : {1, 100, 1000, 10000, 100000}) {
        std::vector<int32_t> times(N);
        for (int i = 0; i < N; ++i) times[i] = uni(rng);
        // warmup
        acc.query_batch(times);
        int rounds = (N <= 1000) ? 50 : 5;
        auto t0 = Clock::now();
        for (int r = 0; r < rounds; ++r) acc.query_batch(times);
        double sec = sec_since(t0) / rounds;
        std::printf("  N=%7d : %10.1f ns/query  (%.3f ms total)\n",
                    N, sec / N * 1e9, sec * 1e3);
    }

    // ---- 随机段 ----
    std::printf("\n=== RANGE SCAN (MB/s) ===\n");
    for (int W : {10, 100, 1000, 10000, 100000}) {
        if (W >= T) continue;
        std::uniform_int_distribution<int> us(0, T - W - 1);
        std::vector<int> starts;
        for (int i = 0; i < 5; ++i) starts.push_back(us(rng));
        acc.query_range(starts[0], starts[0] + W);  // warmup
        auto t0 = Clock::now();
        int rounds = 3;
        for (int r = 0; r < rounds; ++r)
            for (int s : starts) acc.query_range(s, s + W);
        double sec = sec_since(t0) / (rounds * (int)starts.size());
        double mbps = (double)W * 4 / sec / 1e6;
        std::printf("  W=%7d : %10.2f MB/s  (%.4f ms/query)\n", W, mbps, sec * 1e3);
    }

    // ---- 解压 ----
    std::printf("\n=== DECOMPRESSION (MB/s) ===\n");
    acc.decompress_all();  // warmup
    auto t0 = Clock::now();
    int rounds = 3;
    for (int r = 0; r < rounds; ++r) acc.decompress_all();
    double sec = sec_since(t0) / rounds;
    std::printf("  T=%d : %.2f MB/s  (%.2f ms)\n",
                T, (double)T * 4 / sec / 1e6, sec * 1e3);
    return 0;
}
