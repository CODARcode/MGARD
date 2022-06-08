#ifndef TESTING_HUFFMAN_REGRESSION_HPP
#define TESTING_HUFFMAN_REGRESSION_HPP
//!\file
//!\brief Huffman encoding and decoding functions for regression tests.

#include <cstddef>

#include "huffman.hpp"

namespace mgard {

//! Encode quantized coefficients using a Huffman code.
//!
//!\param[in, out] quantized_data Input buffer (quantized coefficients). This
//! buffer will be changed by the encoding process.
//!\param[in] n Number of symbols (`long int` quantized coefficients) in the
//! input buffer.
HuffmanEncodedStream huffman_encoding(long int *const quantized_data,
                                      const std::size_t n);

//! Decode a stream encoded using a Huffman code.
//!
//!\param[in] encoded Input buffer (Huffman-encoded stream).
MemoryBuffer<long int> huffman_decoding(const HuffmanEncodedStream &encoded);

} // namespace mgard

#endif
