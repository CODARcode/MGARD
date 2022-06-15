#ifndef HUFFMAN_HPP
#define HUFFMAN_HPP
//!\file
//!\brief Huffman trees for quantized multilevel coefficients.

#include <cstddef>

#include <memory>
#include <queue>
#include <type_traits>
#include <vector>

#include "utilities.hpp"

namespace mgard {

//! One more than the number of symbols assigned codewords in the deprecated
//! Huffman encoding and decoding functions.
//!
//!\deprecated
inline constexpr std::size_t nql = 1 << 17;

//! A stream compressed using a Huffman code.
struct HuffmanEncodedStream {
  //! Constructor.
  //!
  //!\param nbits Length in bits of the compressed stream.
  //!\param nmissed Length in bytes of the missed array.
  //!\param ntable Length in bytes of the frequency table.
  HuffmanEncodedStream(const std::size_t nbits, const std::size_t nmissed,
                       const std::size_t ntable);

  //! Length in bits of the compressed stream.
  std::size_t nbits;

  //! Compressed stream.
  MemoryBuffer<unsigned char> hit;

  //! Missed array.
  MemoryBuffer<unsigned char> missed;

  //! Frequency table.
  MemoryBuffer<unsigned char> frequencies;
};

//! Serialize a Huffman-encoded stream and then compress.
//!
//!\deprecated
//!
//! The serialized stream will be compressed with ZSTD if `MGARD_ZSTD` is
//! defined and with `zlib` otherwise.
//!
//!\param encoded Huffman-encoded stream to serialize and compress.
MemoryBuffer<unsigned char>
serialize_compress(const HuffmanEncodedStream &encoded);

//! Decompress and then deserialize a Huffman-encoded stream.
//!
//!\deprecated
//!
//! The buffer will be decompressed with ZSTD if `MGARD_ZSTD` if defined and
//! with `zlib` otherwise.
//!
//!\param src Buffer containing serialized and compressed Huffman-encoded
//! stream.
//!\param srcLen Size in bytes of the buffer.
HuffmanEncodedStream decompress_deserialize(unsigned char const *const src,
                                            const std::size_t srcLen);

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
//!
//!\note The construction of this class is a little convoluted.
template <typename Symbol> class HuffmanCode {
public:
  static_assert(std::is_integral<Symbol>::value and
                    std::is_signed<Symbol>::value,
                "symbol type must be signed and integral");

  //! Shared pointer to node in Huffman code creation tree.
  using Node = std::shared_ptr<CodeCreationTreeNode>;

  //! Constructor.
  //!
  //!\param endpoints Smallest and largest symbols (inclusive) to receive
  //! codewords.
  //!\param begin Beginning of input stream.
  //!\param end End of output stream.
  HuffmanCode(const std::pair<Symbol, Symbol> &endpoints,
              Symbol const *const begin, Symbol const *const end);

  //! Constructor.
  //!
  //! The endpoints will be set to `default_endpoints`.
  //!
  //!\param begin Beginning of input stream.
  //!\param end End of output stream.
  HuffmanCode(Symbol const *const begin, Symbol const *const end);

  //! Constructor.
  //!
  //! `It::value_type` should be (convertible to)
  //! `std::pair<std::size_t, std::size_t>`.
  //!
  //!\param endpoints Smallest and largest symbols (inclusive) to receive
  //! codewords.
  //!\param begin Beginning of index–frequency pair range for frequency table.
  //!\param end Beginning of index–frequency pair range for frequency table.
  template <typename It>
  HuffmanCode(const std::pair<Symbol, Symbol> &endpoints, const It begin,
              const It end);

  //! Smallest and largest symbols (inclusive) to receive codewords.
  std::pair<Symbol, Symbol> endpoints;

  //! Number of symbols that will be assigned codewords (including one for the
  //! 'missed' symbol).
  std::size_t ncodewords;

  //! Frequencies of the symbols in the input stream.
  std::vector<std::size_t> frequencies;

  //! Codewords associated to the symbols.
  std::vector<HuffmanCodeword> codewords;

  //! Report the number of out-of-range symbols encountered in the stream or
  //! given in the frequency table pairs.
  std::size_t nmissed() const;

  //! Check whether a symbol is eligible for a codeword.
  bool out_of_range(const Symbol symbol) const;

  //! Determine the codeword index for a symbol.
  std::size_t index(const Symbol symbol) const;

private:
  //! Function object used to compare code creation tree nodes.
  struct HeldCountGreater {
    bool operator()(const Node &a, const Node &b) const;
  };

public:
  //! Huffman code creation tree.
  std::priority_queue<Node, std::vector<Node>, HeldCountGreater> queue;

  //! Decode a codeword (identified by associated leaf) to a symbol.
  //!
  //!\pre `leaf` must be a leaf (rather than an interior node) of the code
  //! creation tree.
  //!
  //!\param leaf Leaf (associated to a codeword) to decode.
  //!\return A boolean indicating whether the original symbol was 'hit' and the
  //! symbol itself (junk if the original symbol was 'missed').
  std::pair<bool, Symbol> decode(const Node &leaf) const;

private:
  //! Default symbol range.
  const static std::pair<Symbol, Symbol> default_endpoints;

  //! Populate the frequency table using a stream of symbols.
  //!
  //!\pre `frequencies` should have length `ncodewords` and all entries should
  //! be zero.
  //!
  //!\param begin Beginning of stream of symbols.
  //!\param end End of stream of symbols.
  void populate_frequencies(Symbol const *const begin, Symbol const *const end);

  //! Populate the frequency table from a collection of index–frequency pairs.
  //!
  //!\param begin Beginning of index–frequency pair range.
  //!\param end End of index–frequency pair range.
  template <typename It>
  void populate_frequencies(const It begin, const It end);

  //! Create the Huffman code creation tree.
  //!
  //!\note This function depends on `frequencies`.
  void create_code_creation_tree();

  // TODO: Check that frequency count ties aren't going to hurt us here. Stable
  // sorting algorithm in `priority_queue`?

  //! Set codewords for given node and descendants.
  void
  recursively_set_codewords(const std::shared_ptr<CodeCreationTreeNode> &node,
                            const HuffmanCodeword codeword);
};

//! Encode quantized coefficients using a Huffman code.
//!
//!\deprecated
//!
//!\param[in] quantized_data Input buffer (quantized coefficients).
//!\param[in] n Number of symbols (`long int` quantized coefficients) in the
//! input buffer.
HuffmanEncodedStream huffman_encoding(long int const *const quantized_data,
                                      const std::size_t n);

//! Decode a stream encoded using a Huffman code.
//!
//!\deprecated
//!
//!\param[in] encoded Input buffer (Huffman-encoded stream).
MemoryBuffer<long int> huffman_decoding(const HuffmanEncodedStream &encoded);

//! Encode quantized coefficients using a Huffman code.
//!
//!\param begin Input buffer (quantized coefficients).
//!\param n Number of symbols in the input buffer.
template <typename Symbol>
MemoryBuffer<unsigned char> huffman_encode(Symbol const *const begin,
                                           const std::size_t n);

//! Decode a stream encoded using a Huffman code.
//!
//!\param buffer Input buffer (Huffman-encoded stream).
template <typename Symbol>
MemoryBuffer<Symbol> huffman_decode(const MemoryBuffer<unsigned char> &buffer);

} // namespace mgard

#include "huffman.tpp"
#endif
