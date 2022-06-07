#ifndef TESTING_COMPRESSORS_REGRESSION_HPP
#define TESTING_COMPRESSORS_REGRESSION_HPP
//!\file
//!\brief Huffman compression and decompression functions for regression tests.

#include <cstddef>

#include "utilities.hpp"

namespace mgard {

//! Compress an array using a Huffman tree.
//!
//!\param[in] src Array to be compressed.
//!\param[in] srcLen Size of array (number of elements) to be compressed.
MemoryBuffer<unsigned char> compress_memory_huffman(long int *const src,
                                                    const std::size_t srcLen);

} // namespace mgard

#endif
