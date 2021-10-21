#ifndef COMPRESSORS_HPP
#define COMPRESSORS_HPP
//!\file
//!\brief Lossless compressors for quantized multilevel coefficients.

#include <cstddef>
#include <cstdint>

#include <vector>

namespace mgard {

//! Compress an array using a Huffman tree.
//!
//!\param[in] qv Array to be compressed.
//!\param[out] outsize Size in bytes of the compressed array.
//!
//!\return Compressed array.
unsigned char *compress_memory_huffman(const std::vector<long int> &qv,
                                       std::size_t &outsize);

//! Decompress an array compressed with `compress_memory_huffman`.
//!
//!\param[in] data Compressed array.
//!\param[in] data_len Size in bytes of the compressed array.
//!\param[out] out_data Decompressed array.
//!\param[in] outsize Size in bytes of the decompressed array.
void decompress_memory_huffman(unsigned char *data, const std::size_t data_len,
                               long int *out_data, const std::size_t outsize);

#ifdef MGARD_ZSTD
//! Compress an array using `zstd`.
//!
//!\param[in] in_data Array to be compressed.
//!\param[in] in_data_size Size in bytes of the array to be compressed.
//!\param[out] out_data Compressed array.
void compress_memory_zstd(void *const in_data, const std::size_t in_data_size,
                          std::vector<std::uint8_t> &out_data);

//! Decompress an array compressed with `compress_memory_zstd`.
//!
//!\param[in] src Compressed array.
//!\param[in] srcLen Size in bytes of the compressed array.
//!\param[out] dst Decompressed array.
//!\param[in] dstLen Size in bytes of the decompressed array.
void decompress_memory_zstd(void *const src, const std::size_t srcLen,
                            unsigned char *const dst, const std::size_t dstLen);
#endif

//! Compress an array using `zlib`.
//!
//!\param in_data Array to be compressed.
//!\param in_data_size Size in bytes of the array to be compressed.
//!\param out_data Compressed array.
void compress_memory_z(void *const in_data, const std::size_t in_data_size,
                       std::vector<std::uint8_t> &out_data);

//! Decompress an array with `compress_memory_z`.
//!
//!\param src Compressed array.
//!\param srcLen Size in bytes of the compressed array data
//!\param dst Decompressed array.
//!\param dstLen Size in bytes of the decompressed array.
void decompress_memory_z(void *const src, const std::size_t srcLen,
                         unsigned char *const dst, const std::size_t dstLen);

} // namespace mgard

#endif
