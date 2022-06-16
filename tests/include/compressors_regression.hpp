#ifndef TESTING_COMPRESSORS_REGRESSION_HPP
#define TESTING_COMPRESSORS_REGRESSION_HPP
//!\file
//!\brief Huffman compression and decompression functions for regression tests.

#include <cstddef>

#include "format.hpp"
#include "utilities.hpp"

namespace mgard {

namespace regression {

//! Compress an array using a Huffman tree.
//!
//!\param[in] header Header for the self-describing buffer.
//!\param[in] src Array to be compressed.
//!\param[in] srcLen Size of array (number of elements) to be compressed.
MemoryBuffer<unsigned char> compress_memory_huffman(const pb::Header &header,
                                                    long int const *const src,
                                                    const std::size_t srcLen);

//! Decompress an array compressed with `compress_memory_huffman`.
//!
//!\param[in] header Header parsed from the original self-describing buffer.
//!\param[in] src Compressed array.
//!\param[in] srcLen Size in bytes of the compressed array.
//!\param[out] dst Decompressed array.
//!\param[in] dstLen Size in bytes of the decompressed array.
void decompress_memory_huffman(const pb::Header &header,
                               unsigned char const *const src,
                               const std::size_t srcLen, long int *const dst,
                               const std::size_t dstLen);

} // namespace regression

} // namespace mgard

#endif
