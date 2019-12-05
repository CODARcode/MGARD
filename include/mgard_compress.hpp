#include <cstddef>
#include <cstdint>

#include <string>
#include <vector>

namespace mgard {

void compress_memory_z(
    void *in_data, size_t in_data_size, std::vector<uint8_t> &out_data
);

void decompress_memory_z(const void *src, int srcLen, int *dst, int dstLen);

}
