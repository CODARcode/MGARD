#include <cstddef>
#include <cstdint>

#include <vector>

namespace mgard {
unsigned char *compress_memory_huffman(const std::vector<long int> &qv,
                                       std::vector<unsigned char> &out_data,
                                       int &outsize);
void decompress_memory_huffman(unsigned char *data, int data_len,
                               long int *out_data, int outsize);
void huffman_encoding(long int *in_data, const std::size_t in_data_size,
                      unsigned char **out_data_hit, size_t *out_data_hit_size,
                      unsigned char **out_data_miss, size_t *out_data_miss_size,
                      unsigned char **out_tree, size_t *out_tree_size);

void huffman_decoding(long int *const in_data, const std::size_t in_data_size,
                      unsigned char *out_data_hit, size_t out_data_hit_size,
                      unsigned char *out_data_miss, size_t out_data_miss_size,
                      unsigned char *out_tree, size_t out_tree_size);
#ifdef MGARD_ZSTD
//! Compress an array of data using `zstd`.
void compress_memory_zstd(void *const in_data, const std::size_t in_data_size,
                          std::vector<std::uint8_t> &out_data);
#endif
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

#ifdef MGARD_ZSTD
void decompress_memory_zstd_huffman(void *const src, const int srcLen,
                                    unsigned char *const dst, const int dstLen);
#endif
} // namespace mgard
