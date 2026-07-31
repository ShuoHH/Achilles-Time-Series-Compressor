// 把 C++ decompress_all 结果写成 float32 .bin，供 Python 比对数值一致性。
// 用法: ./dump_blockcodec <export_dir> <out.bin>
#include "bc/blockcodec.hpp"
#include <cstdio>
#include <fstream>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <export_dir> <out.bin>\n", argv[0]);
        return 1;
    }
    bc::BlockAccessor acc(argv[1]);
    auto full = acc.decompress_all();
    std::ofstream f(argv[2], std::ios::binary);
    f.write(reinterpret_cast<const char*>(full.data()),
            (std::streamsize)full.size() * sizeof(float));
    std::printf("wrote %zu floats to %s\n", full.size(), argv[2]);
    return 0;
}
