#ifndef COMPRESSORS_HPP
#define COMPRESSORS_HPP
//!\file
//!\brief Lossless compressors for quantized multilevel coefficients.

#include <cstddef>

// For `z_const`.
#include <zlib.h>

#include <memory>

#include "proto/mgard.pb.h"

#include "utilities.hpp"

namespace mgard {

//! Compress an array using a Huffman tree.
//!
//!\param[in] src Array to be compressed.
//!\param[in] srcLen Size of array (number of elements) to be compressed.
MemoryBuffer<unsigned char> compress_memory_huffman(long int *const src,
                                                    const std::size_t srcLen);

//! Decompress an array compressed with `compress_memory_huffman`.
//!
//!\param[in] src Compressed array.
//!\param[in] srcLen Size in bytes of the compressed array.
//!\param[out] dst Decompressed array.
//!\param[in] dstLen Size in bytes of the decompressed array.
void decompress_memory_huffman(unsigned char *const src,
                               const std::size_t srcLen, long int *const dst,
                               const std::size_t dstLen);

#ifdef MGARD_ZSTD
//! Compress an array using `zstd`.
//!
//!\param[in] src Array to be compressed.
//!\param[in] srcLen Size in bytes of the array to be compressed.
MemoryBuffer<unsigned char> compress_memory_zstd(void const *const src,
                                                 const std::size_t srcLen);

//! Decompress an array compressed with `compress_memory_zstd`.
//!
//!\param[in] src Compressed array.
//!\param[in] srcLen Size in bytes of the compressed array.
//!\param[out] dst Decompressed array.
//!\param[in] dstLen Size in bytes of the decompressed array.
void decompress_memory_zstd(void const *const src, const std::size_t srcLen,
                            unsigned char *const dst, const std::size_t dstLen);
#endif

//! Compress an array using `zlib`.
//!
//!\param src Array to be compressed.
//!\param srcLen Size in bytes of the array to be compressed.
MemoryBuffer<unsigned char> compress_memory_z(void z_const *const src,
                                              const std::size_t srcLen);

//! Decompress an array with `compress_memory_z`.
//!
//!\param src Compressed array.
//!\param srcLen Size in bytes of the compressed array data
//!\param dst Decompressed array.
//!\param dstLen Size in bytes of the decompressed array.
void decompress_memory_z(void z_const *const src, const std::size_t srcLen,
                         unsigned char *const dst, const std::size_t dstLen);

//! Compress an array of quantized multilevel coefficients.
//!
//! `src` must have the correct alignment for the quantization type.
//!
//!\param[in] src Array of quantized multilevel coefficients.
//!\param[in] srcLen Size in bytes of the input array.
//!\param[in] header Header for the self-describing buffer.
MemoryBuffer<unsigned char> compress(void *const src, const std::size_t srcLen,
                                     const pb::Header &header);

//! Decompress an array of quantized multilevel coefficients.
//!
//! `dst` must have the correct alignment for the quantization type.
//!
//!\param[in] src Compressed array of quantized multilevel coefficients.
//!\param[in] srcLen Size in bytes of the compressed array.
//!\param[out] dst Decompressed array.
//!\param[in] dstLen Size in bytes of the decompressed array.
//!\param[in] header Header parsed from the original self-describing buffer.
void decompress(void *const src, const std::size_t srcLen, void *const dst,
                const std::size_t dstLen, const pb::Header &header);

} // namespace mgard

#endif
