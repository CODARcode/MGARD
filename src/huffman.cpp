#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <array>
#include <numeric>
#include <queue>
#include <vector>

#include <iostream>

#include "huffman.hpp"

namespace mgard {

HuffmanEncodedStream::HuffmanEncodedStream(const std::size_t nbits,
                                           const std::size_t ncompressed,
                                           const std::size_t nmissed,
                                           const std::size_t nfrequencies)
    : nbits(nbits), hit(ncompressed), missed(nmissed),
      frequencies(nfrequencies) {}

void HuffmanCodeword::push_back(const bool bit) {
  const unsigned char offset = length % CHAR_BIT;
  if (not offset) {
    bytes.push_back(0);
  }
  bytes.back() |= static_cast<unsigned char>(bit) << (CHAR_BIT - 1 - offset);
  ++length;
}

HuffmanCodeword HuffmanCodeword::left() const {
  HuffmanCodeword tmp = *this;
  tmp.push_back(false);
  return tmp;
}

HuffmanCodeword HuffmanCodeword::right() const {
  HuffmanCodeword tmp = *this;
  tmp.push_back(true);
  return tmp;
}

CodeCreationTreeNode::CodeCreationTreeNode(HuffmanCodeword *const codeword,
                                           const std::size_t count)
    : codeword(codeword), count(count) {}

CodeCreationTreeNode::CodeCreationTreeNode(
    const std::shared_ptr<CodeCreationTreeNode> &left,
    const std::shared_ptr<CodeCreationTreeNode> &right)
    : count(left->count + right->count), left(left), right(right) {}

namespace {

void endianness_shuffle(unsigned char *const buffer, const std::size_t nbytes) {
  if (nbytes % sizeof(unsigned int)) {
    throw std::runtime_error(
        "buffer size not a multiple of `sizeof(unsigned int)`");
  }
  const unsigned int one{1};
  const bool little_endian = *reinterpret_cast<unsigned char const *>(&one);
  if (little_endian) {
    for (std::size_t i = 0; i < nbytes; i += sizeof(unsigned int)) {
      unsigned char *a = buffer + i;
      unsigned char *b = a + sizeof(unsigned int) - 1;
      for (std::size_t j = 0; j < sizeof(unsigned int) / 2; ++j) {
        std::swap(*a++, *b--);
      }
    }
  }
}

} // namespace
namespace {

void check_type_sizes() {
  static_assert(CHAR_BIT == 8,
                "code written with assumption that `CHAR_BIT == 8`");
  static_assert(
      sizeof(unsigned int) == 4,
      "code written with assumption that `sizeof(unsigned int) == 4`");
  static_assert(sizeof(int) == 4,
                "code written with assumption that `sizeof(int) == 4`");
  static_assert(
      sizeof(std::size_t) == 8,
      "code written with assumption that `sizeof(unsigned int) == 8`");
}

} // namespace

namespace {

const std::pair<long int, long int> nql_endpoints{
    -static_cast<long int>((nql - 1) / 2), nql / 2 - 1};
}

HuffmanEncodedStream huffman_encoding(long int const *const quantized_data,
                                      const std::size_t n) {
  check_type_sizes();

  using Symbol = long int;
  using MissedSymbol = int;

  const HuffmanCode<Symbol> code(nql_endpoints, quantized_data,
                                 quantized_data + n);

  std::vector<std::size_t> lengths;
  for (const HuffmanCodeword &codeword : code.codewords) {
    lengths.push_back(codeword.length);
  }
  const std::size_t nbits =
      std::inner_product(code.frequencies.begin(), code.frequencies.end(),
                         lengths.begin(), static_cast<std::size_t>(0));
  const std::size_t nbytes =
      sizeof(unsigned int) * ((nbits + CHAR_BIT * sizeof(unsigned int) - 1) /
                              (CHAR_BIT * sizeof(unsigned int)));
  if (nbytes % sizeof(unsigned int)) {
    throw std::runtime_error(
        "`nbytes` not bumped up to nearest multiple of `unsigned int` size");
  }

  const std::size_t nnz =
      code.ncodewords -
      std::count(code.frequencies.begin(), code.frequencies.end(), 0);

  HuffmanEncodedStream out(nbits, nbytes, code.nmissed() * sizeof(MissedSymbol),
                           2 * nnz * sizeof(std::size_t));

  // Write frequency table.
  {
    std::size_t *p =
        reinterpret_cast<std::size_t *>(out.frequencies.data.get());
    const std::vector<std::size_t> &frequencies = code.frequencies;
    for (std::size_t i = 0; i < code.ncodewords; ++i) {
      const std::size_t frequency = frequencies.at(i);
      if (frequency) {
        *p++ = i;
        *p++ = frequency;
      }
    }
  }

  unsigned char *const buffer = out.hit.data.get();
  {
    unsigned char *const p = out.hit.data.get();
    std::fill(p, p + out.hit.size, 0);
  }
  unsigned char *hit = buffer;

  MissedSymbol *missed =
      reinterpret_cast<MissedSymbol *>(out.missed.data.get());

  unsigned char offset = 0;
  for (const Symbol q : PseudoArray(quantized_data, n)) {
    if (code.out_of_range(q)) {
      // Remember that `missed` is an `int` rather than a `long int`.
      *missed++ = q + nql / 2;
    }

    const HuffmanCodeword codeword = code.codewords.at(code.index(q));
    std::size_t NREMAINING = codeword.length;
    for (unsigned char byte : codeword.bytes) {
      // Number of bits of `byte` left to write.
      unsigned char nremaining =
          std::min(static_cast<std::size_t>(CHAR_BIT), NREMAINING);
      // Premature, but this will hold when we're done with `byte`.
      NREMAINING -= nremaining;

      while (nremaining) {
        *hit |= byte >> offset;
        // Number of bits of `byte` just written (not cumulative).
        const unsigned char nwritten = std::min(
            nremaining, static_cast<unsigned char>(
                            static_cast<unsigned char>(CHAR_BIT) - offset));
        offset += nwritten;
        hit += offset / CHAR_BIT;
        offset %= CHAR_BIT;
        nremaining -= nwritten;
        byte <<= nwritten;
      }
    }
  }

  endianness_shuffle(buffer, nbytes);
  return out;
}

namespace {

//! Decode a codeword (identified by associated leaf) to a symbol and shift.
//!
//!\pre `leaf` must be a leaf (rather than an interior node) of the code
//! creation tree.
//!
//!\deprecated
//!
//!\param code Code containing the code creation tree.
//!\param leaf Leaf (associated to a codeword) to decode.
//!\param missed Pointer to next out-of-range symbol. If `leaf` is associated
//! to the out-of-range codeword, this pointer will be dereferenced and
//! incremented.
long int decode_and_shift(const HuffmanCode<long int> &code,
                          const typename HuffmanCode<long int>::Node &leaf,
                          long int const *&missed) {
  long int const *const start = missed;
  long int decoded = code.decode(leaf, missed);
  if (missed != start) {
    decoded -= nql / 2;
  }
  return decoded;
}

} // namespace

MemoryBuffer<long int> huffman_decoding(const HuffmanEncodedStream &encoded) {
  check_type_sizes();

  using Symbol = long int;
  using MissedSymbol = int;

  const std::size_t nnz = encoded.frequencies.size / (2 * sizeof(std::size_t));
  std::vector<std::pair<std::size_t, std::size_t>> pairs(nnz);
  std::size_t nquantized = 0;
  {
    std::size_t const *p =
        reinterpret_cast<std::size_t const *>(encoded.frequencies.data.get());
    for (std::pair<std::size_t, std::size_t> &pair : pairs) {
      const std::size_t index = *p++;
      const std::size_t frequency = *p++;
      pair = {index, frequency};
      nquantized += frequency;
    }
  }

  HuffmanCode<Symbol> code(nql_endpoints, pairs);

  MemoryBuffer<Symbol> out(nquantized);
  Symbol *q = out.data.get();

  assert(not(encoded.missed.size % sizeof(MissedSymbol)));
  const std::size_t nmissed = encoded.missed.size / sizeof(MissedSymbol);
  Symbol *const missed = new Symbol[nmissed];
  {
    MissedSymbol const *const p =
        reinterpret_cast<MissedSymbol const *>(encoded.missed.data.get());
    std::copy(p, p + nmissed, missed);
  }
  Symbol const *p_missed = missed;

  const std::size_t nbytes = encoded.hit.size;
  unsigned char *const buffer = new unsigned char[nbytes];
  {
    unsigned char const *const p = encoded.hit.data.get();
    std::copy(p, p + nbytes, buffer);
  }
  endianness_shuffle(buffer, nbytes);
  const Bits bits(buffer, buffer + encoded.nbits / CHAR_BIT,
                  encoded.nbits % CHAR_BIT);

  std::size_t nbits = 0;
  const HuffmanCode<Symbol>::Node root = code.queue.top();
  assert(root);
  Bits::iterator b = bits.begin();
  for (std::size_t i = 0; i < nquantized; ++i) {
    HuffmanCode<Symbol>::Node node;
    for (node = root; node->left;
         node = *b++ ? node->right : node->left, ++nbits)
      ;
    *q++ = decode_and_shift(code, node, p_missed);
  }
  assert(nbits == encoded.nbits);
  assert(sizeof(MissedSymbol) * (p_missed - missed) == encoded.missed.size);

  delete[] missed;
  delete[] buffer;

  return out;
}

} // namespace mgard
