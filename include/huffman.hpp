#ifndef HUFFMAN_HPP
#define HUFFMAN_HPP
//!\file
//!\brief Huffman trees for quantized multilevel coefficients.

#include <cstddef>

#include <memory>
#include <vector>

#include "utilities.hpp"

namespace mgard {

//! A stream compressed using a Huffman code.
struct HuffmanEncodedStream {
  //! Constructor.
  //!
  //!\param nbits Length in bits of the compressed stream.
  //!\param ncompressed Length in bytes of the compressed stream.
  //!\param nmissed Length in bytes of the missed array.
  //!\param ntable Length in bytes of the frequency table.
  HuffmanEncodedStream(const std::size_t nbits, const std::size_t ncompressed,
                       const std::size_t nmissed, const std::size_t ntable);

  //! Length in bits of the compressed stream.
  std::size_t nbits;

  //! Compressed stream.
  MemoryBuffer<unsigned char> hit;

  //! Missed array.
  MemoryBuffer<unsigned char> missed;

  //! Frequency table.
  MemoryBuffer<unsigned char> frequencies;
};

//! Encode quantized coefficients using a Huffman code.
//!
//!\param[in, out] quantized_data Input buffer (quantized coefficients). This
//! buffer will be changed by the encoding process.
//!\param[in] n Number of symbols (`long int` quantized coefficients) in the
//! input buffer.
HuffmanEncodedStream huffman_encoding(long int *const quantized_data,
                                      const std::size_t n);

//! Encode quantized coefficients using a Huffman code.
//!
//!\param[in] quantized_data Input buffer (quantized coefficients).
//!\param[in] n Number of symbols (`long int` quantized coefficients) in the
//! input buffer.
HuffmanEncodedStream
huffman_encoding_rewritten(long int const *const quantized_data,
                           const std::size_t n);

//! Decode a stream encoded using a Huffman code.
//!
//!\param[in] encoded Input buffer (Huffman-encoded stream).
MemoryBuffer<long int> huffman_decoding(const HuffmanEncodedStream &encoded);

//! Decode a stream encoded using a Huffman code.
//!
//!\param[in] encoded Input buffer (Huffman-encoded stream).
MemoryBuffer<long int>
huffman_decoding_rewritten(const HuffmanEncodedStream &encoded);

//! Codeword (in progress) associated to a node in a Huffman code creation tree.
struct HuffmanCodeword {
  //! Bytes containing the bits of the codeword.
  std::vector<unsigned char> bytes = {};

  //! Length in bits of the codeword.
  std::size_t length = 0;

  //! Append a bit to the codeword.
  void push_back(const bool bit);

  //! Generate the codeword associated to the left child in the tree.
  HuffmanCodeword left() const;

  //! Generate the codeword associated to the right child in the tree.
  HuffmanCodeword right() const;
};

//! Node in a Huffman code creation tree.
struct CodeCreationTreeNode {
  //! Constructor.
  //!
  //! Create a leaf node.
  //!
  //!\param codeword Associated codeword.
  //!\param count Frequency of the associated symbol.
  CodeCreationTreeNode(HuffmanCodeword *const codeword,
                       const std::size_t count);

  //! Constructor.
  //!
  //! Create an inner (parent) node.
  //!
  //!\param left Left child of the node to be created.
  //!\param right Right child of the node to be created.
  CodeCreationTreeNode(const std::shared_ptr<CodeCreationTreeNode> &left,
                       const std::shared_ptr<CodeCreationTreeNode> &right);

  //! Associated codeword (if this node is a leaf).
  HuffmanCodeword *codeword = nullptr;

  //! Sum of frequencies of symbols associated to leaves descending from this
  //! node.
  std::size_t count;

  //! Left child of this node.
  std::shared_ptr<CodeCreationTreeNode> left;

  //! Right child of this node.
  std::shared_ptr<CodeCreationTreeNode> right;
};

//! Huffman code generated from/for an input stream.
template <typename Symbol> class HuffmanCode {
public:
  //! Constructor.
  //!
  //!\param ncodewords Number of symbols that will be assigned codewords.
  //!\param begin Beginning of input stream.
  //!\param end End of output stream.
  HuffmanCode(const std::size_t ncodewords, Symbol const *const begin,
              Symbol const *const end);

  //! Number of symbols that will be assigned codewords.
  std::size_t ncodewords;

  //! Frequencies of the symbols in the input stream.
  std::vector<std::size_t> frequencies;

  //! Codewords associated to the symbols.
  std::vector<HuffmanCodeword> codewords;

  //! Report the number of out-of-range symbols encountered in the stream.
  std::size_t nmissed() const;

  //! Check whether a symbol is eligible for a codeword.
  bool out_of_range(const Symbol symbol) const;

  //! Determine the codeword index for a symbol.
  std::size_t index(const Symbol symbol) const;

private:
  //! Smallest symbol (inclusive) to receive a codeword.
  Symbol min_symbol;

  //! Largest symbol (inclusive) to receive a codeword.
  Symbol max_symbol;

  // TODO: Check that frequency count ties aren't going to hurt us here. Stable
  // sorting algorithm in `priority_queue`?

  //! Set codewords for given node and descendants.
  void
  recursively_set_codewords(const std::shared_ptr<CodeCreationTreeNode> &node,
                            const HuffmanCodeword codeword);
};

} // namespace mgard

#include "huffman.tpp"
#endif
