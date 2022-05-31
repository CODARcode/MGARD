#ifndef HUFFMAN_HPP
#define HUFFMAN_HPP
//!\file
//!\brief Huffman trees for quantized multilevel coefficients.

#include <cstddef>

#include <memory>
#include <vector>

namespace mgard {

//! Encode quantized coefficients using a Huffman code.
//!
//!\param[in, out] quantized_data Input buffer (quantized coefficients). This
//! buffer will be changed by the encoding process.
//!\param[in] n Number of symbols (`long int` quantized coefficients) in the
//! input buffer.
//!\param[out] out_data_hit Pointer to compressed buffer.
//!\param[out] out_data_hit_size Size *in bits* of compressed buffer.
//!\param[out] out_data_miss Pointer to 'missed' buffer (input symbols not
//! assigned codes).
//!\param[out] out_data_miss_size Size *in bytes* of 'missed'
//! buffer.
//!\param[out] out_tree Frequency table for input buffer.
//!\param[out] out_tree_size Size *in bytes* of the frequency table.
void huffman_encoding(long int *const quantized_data, const std::size_t n,
                      unsigned char *&out_data_hit, size_t &out_data_hit_size,
                      unsigned char *&out_data_miss, size_t &out_data_miss_size,
                      unsigned char *&out_tree, size_t &out_tree_size);

//! Encode quantized coefficients using a Huffman code.
//!
//!\param[in, out] quantized_data Input buffer (quantized coefficients). This
//! buffer will be changed by the encoding process.
//!\param[in] n Number of symbols (`long int` quantized coefficients) in the
//! input buffer.
//!\param[out] out_data_hit Pointer to compressed buffer.
//!\param[out] out_data_hit_size Size *in bits* of compressed buffer.
//!\param[out] out_data_miss Pointer to 'missed' buffer (input symbols not
//! assigned codes).
//!\param[out] out_data_miss_size Size *in bytes* of 'missed'
//! buffer.
//!\param[out] out_tree Frequency table for input buffer.
//!\param[out] out_tree_size Size *in bytes* of the frequency table.
void huffman_encoding_rewritten(
    long int const *const quantized_data, const std::size_t n,
    unsigned char *&out_data_hit, std::size_t &out_data_hit_size,
    unsigned char *&out_data_miss, std::size_t &out_data_miss_size,
    unsigned char *&out_tree, std::size_t &out_tree_size);

//! Decode a stream encoded using a Huffman code.
//!
//!\param[out] quantized_data Output buffer (quantized coefficients).
//!\param[in] quantized_data_size Size *in bytes* of output buffer.
//!\param[in] out_data_hit Compressed buffer.
//!\param[in] out_data_hit_size Size *in bits* of compressed buffer.
//!\param[in] out_data_miss 'Missed' buffer (input symbols not assigned codes).
//!\param[in] out_data_miss_size Size *in bytes* of 'missed' buffer.
//!\param[in] out_tree Frequency table for input buffer.
//!\param[in] out_tree_size Size *in bytes* of the frequency table.
void huffman_decoding(
    long int *const quantized_data, const std::size_t quantized_data_size,
    unsigned char const *const out_data_hit, const size_t out_data_hit_size,
    unsigned char const *const out_data_miss, const size_t out_data_miss_size,
    unsigned char const *const out_tree, const size_t out_tree_size);

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
