#include <cstddef>
#include <cstdint>

#include <vector>

namespace mgard {
void huffman_encoding(int *const in_data, const std::size_t in_data_size,
                      unsigned char **out_data_hit, size_t * out_data_hit_size,
		      unsigned char **out_data_miss, size_t * out_data_miss_size,
		      unsigned char ** out_tree, size_t * out_tree_size);

void huffman_decoding(int *const in_data, const std::size_t in_data_size,
                      unsigned char *out_data_hit, size_t out_data_hit_size,
                      unsigned char *out_data_miss, size_t out_data_miss_size,
                      unsigned char * out_tree, size_t out_tree_size);

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

void decompress_memory_z_huffman(void *const src, const int srcLen, 
		                 unsigned char *const dst, const int dstLen);

} // namespace mgard
