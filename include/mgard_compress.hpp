#include <cstddef>
#include <cstdint>

#include <vector>

namespace mgard {

//! Compress an array of data using `zlib`.
//!
//!\param in_data Pointer to data to be compressed.
//!\param in_data_size Size in bytes of the data to be compressed.
//!\param out_data Vector used to store compressed data.
void compress_memory_z(void *const in_data, const std::size_t in_data_size,
                       std::vector<std::uint8_t> &out_data);

//! Decompress an array of data using `zlib`.
//!
//!\param src Pointer to data to be decompressed.
//!\param srcLen Size in bytes of the data to be decompressed.
//!\param dst Pointer to buffer used to store decompressed data.
//!\param dstLen Size in bytes of the decompressed data.
void decompress_memory_z(void *const src, const int srcLen, int *const dst,
                         const int dstLen);

} // namespace mgard
