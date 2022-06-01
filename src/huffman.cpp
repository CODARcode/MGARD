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

#include "huffman.hpp"

namespace mgard {

const int nql = 32768 * 4;

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

//! Node in the Huffman code creation tree.
struct htree_node {
  //! Constructor.
  //!
  //!\param q (Transformed) symbol.
  //!\param cnt Number of occurences of the (transformed) symbol in the source.
  htree_node(const int q, const std::size_t cnt)
      : q(q), cnt(cnt), code(0), len(0), left(nullptr), right(nullptr) {}

  //! (Transformed) symbol.
  int q;

  //! Number of occurences of the (transformed) symbol in the source.
  std::size_t cnt;

  //! Codeword associated to the (transformed) symbol.
  unsigned int code;

  //! Length in bits of the codeword.
  std::size_t len;

  //! Left child in the code creation tree.
  htree_node *left;

  //! Right child in the code creation tree.
  htree_node *right;
};

//! Input symbol–Huffman code pair.
struct huffman_codec {
  //! (Transformed) symbol.
  int q;

  //! Codeword associated to the (transformed) symbol.
  unsigned int code;

  //! Length in bits of the codeword.
  std::size_t len;
};

//! Frequency table and symbol–code mappings for encoding source.
template <std::size_t NQL> struct HuffmanCodec {
  // The arrays are value-initialized, which leads to each of their elements
  // being value-initialized (ultimately zero-initialized).

  //! Input symbol–Huffman code pairs.
  std::array<huffman_codec, NQL> codec{};

  //! Frequency table for encoding source.
  std::array<std::size_t, NQL> frequency_table{};
};

//! Function object for comparing Huffman code creation nodes.
struct LessThanByCnt {
  //! Return whether the first node has a larger count than the second.
  //!
  //!\param lhs First node.
  //!\param rhs Second node.
  bool operator()(htree_node const *const lhs,
                  htree_node const *const rhs) const {
    return lhs->cnt > rhs->cnt;
  }
};

template <class T>
using my_priority_queue =
    std::priority_queue<T *, std::vector<T *>, LessThanByCnt>;

void initialize_codec(HuffmanCodec<nql> &codec, htree_node *const root,
                      const unsigned int code, const std::size_t len) {
  std::array<huffman_codec, nql> &codewords = codec.codec;

  root->code = code;
  root->len = len;

  if (!root->left && !root->right) {
    const std::size_t index = root->q;
    codewords.at(index) = {root->q, code, len};
  }

  if (root->left) {
    initialize_codec(codec, root->left, code << 1, len + 1);
  }

  if (root->right) {
    initialize_codec(codec, root->right, code << 1 | 0x1, len + 1);
  }
}

my_priority_queue<htree_node> *build_tree(std::size_t const *const cnt) {
  my_priority_queue<htree_node> *const phtree =
      new my_priority_queue<htree_node>;
  for (int i = 0; i < nql; i++) {
    if (cnt[i] != 0) {
      htree_node *const new_node = new htree_node(i, cnt[i]);
      phtree->push(new_node);
    }
  }

  while (phtree->size() > 1) {
    htree_node *const top_node1 = phtree->top();
    phtree->pop();
    htree_node *const top_node2 = phtree->top();
    phtree->pop();

    htree_node *const new_node =
        new htree_node(-1, top_node1->cnt + top_node2->cnt);
    new_node->left = top_node1;
    new_node->right = top_node2;
    phtree->push(new_node);
  }
  return phtree;
}

void free_htree_node(htree_node *const node) {
  if (node->left) {
    free_htree_node(node->left);
    node->left = nullptr;
  }

  if (node->right) {
    free_htree_node(node->right);
    node->right = nullptr;
  }

  delete node;
}

void free_tree(my_priority_queue<htree_node> *const phtree) {
  if (phtree) {
    free_htree_node(phtree->top());

    phtree->pop();

    delete phtree;
  }
}

//! Populate the frequency table of a `HuffmanCodec`.
//!
//!\note This function will change the quantized data.
//!
//!\param[in, out] quantized_data Input buffer (quantized coefficients). This
//! buffer will be changed by the codec-building process.
//\param[in] n Number of symbols (`long int` quantized coefficients) in the
//! input buffer.
void initialize_frequency_table(HuffmanCodec<nql> &codec,
                                long int *const quantized_data,
                                const std::size_t n) {
  assert(*std::max_element(codec.frequency_table.begin(),
                           codec.frequency_table.end()) == 0);

  for (std::size_t i = 0; i < n; i++) {
    // Convert quantization level to positive so that counting freq can be
    // easily done. Level 0 is reserved a out-of-range flag.
    quantized_data[i] = quantized_data[i] + nql / 2;
    ++codec.frequency_table[quantized_data[i] > 0 &&
                                    quantized_data[i] <
                                        static_cast<long int>(nql)
                                ? quantized_data[i]
                                : 0];
  }
}

//! Build a Huffman codec for an input buffer.
//!
//!\param[in, out] quantized_data Input buffer (quantized coefficients). This
//! buffer will be changed by the codec-building process.
//\param[in] n Number of symbols (`long int` quantized coefficients) in the
//! input buffer.
template <std::size_t N>
HuffmanCodec<N> build_huffman_codec(long int *const quantized_data,
                                    const std::size_t n) {
  HuffmanCodec<N> codec;
  initialize_frequency_table(codec, quantized_data, n);

  my_priority_queue<htree_node> *const phtree =
      build_tree(codec.frequency_table.data());

  initialize_codec(codec, phtree->top(), 0, 0);

  free_tree(phtree);

  return codec;
}

HuffmanEncodedStream huffman_encoding(long int *const quantized_data,
                                      const std::size_t n) {
  const HuffmanCodec<nql> codec = build_huffman_codec<nql>(quantized_data, n);
  const std::size_t num_miss = codec.frequency_table[0];

  assert(n >= num_miss);

  /* For those miss points, we still need to maintain a flag (q = 0),
   * and therefore we need to allocate space for n numbers.
   */
  // The elements of the array are value-initialized (here, zero-initialized).
  unsigned int *const p_hit = new unsigned int[n]();

  int *p_miss = nullptr;
  if (num_miss > 0) {
    // The elements of the array are value-initialized (here, zero-initialized).
    p_miss = new int[num_miss]();
  }

  unsigned char const *const out_data_hit =
      reinterpret_cast<unsigned char *>(p_hit);
  unsigned char const *const out_data_miss =
      reinterpret_cast<unsigned char *>(p_miss);
  std::size_t out_data_hit_size = 0;
  std::size_t out_data_miss_size = 0;

  std::size_t start_bit = 0;
  unsigned int *cur = p_hit;
  for (std::size_t i = 0; i < n; i++) {
    const int q = quantized_data[i];
    unsigned int code;
    std::size_t len;

    if (q > 0 && q < nql) {
      // for those that are within the range
      code = codec.codec[q].code;
      len = codec.codec[q].len;
    } else {
      // for those that are out of the range, q is set to 0
      code = codec.codec[0].code;
      len = codec.codec[0].len;

      *p_miss = q;
      p_miss++;
    }

    // Note that if len == 0, then that means that either the data is all the
    // same number or (more likely) all data are outside the quantization
    // range. Either way, the code contains no information and is therefore 0
    // bits.

    if (32 - start_bit % 32 < len) {
      // current unsigned int cannot hold the code
      // copy 32 - start_bit % 32 bits to the current int
      // and copy  the rest len - (32 - start_bit % 32) to the next int
      const std::size_t rshift = len - (32 - start_bit % 32);
      const std::size_t lshift = 32 - rshift;
      *(cur + start_bit / 32) = (*(cur + start_bit / 32)) | (code >> rshift);
      *(cur + start_bit / 32 + 1) =
          (*(cur + start_bit / 32 + 1)) | (code << lshift);
      start_bit += len;
    } else if (len > 0) {
      code = code << (32 - start_bit % 32 - len);
      *(cur + start_bit / 32) = (*(cur + start_bit / 32)) | code;
      start_bit += len;
    } else {
      // Sequence is empty (everything must be the same). Do nothing.
    }
  }

  // Note: hit size is in bits, while miss size is in bytes.
  out_data_hit_size = start_bit;
  out_data_miss_size = num_miss * sizeof(int);

  // write frequency table to buffer
  int nonZeros = 0;
  for (int i = 0; i < nql; i++) {
    if (codec.frequency_table[i] > 0) {
      nonZeros++;
    }
  }

  std::size_t *const cft = new std::size_t[2 * nonZeros];
  int off = 0;
  for (int i = 0; i < nql; i++) {
    if (codec.frequency_table[i] > 0) {
      cft[2 * off] = i;
      cft[2 * off + 1] = codec.frequency_table[i];
      off++;
    }
  }

  unsigned char const *const out_tree = (unsigned char *)cft;
  const std::size_t out_tree_size = 2 * nonZeros * sizeof(std::size_t);

  const std::size_t nbytes =
      sizeof(unsigned int) *
      ((out_data_hit_size + CHAR_BIT * sizeof(unsigned int) - 1) /
       (CHAR_BIT * sizeof(unsigned int)));
  HuffmanEncodedStream out(out_data_hit_size, nbytes, out_data_miss_size,
                           out_tree_size);
  std::copy(out_data_hit, out_data_hit + nbytes, out.hit.data.get());
  std::copy(out_data_miss, out_data_miss + out_data_miss_size,
            out.missed.data.get());
  std::copy(out_tree, out_tree + out_tree_size, out.frequencies.data.get());
  return out;
}

HuffmanEncodedStream
huffman_encoding_rewritten(long int const *const quantized_data,
                           const std::size_t n) {
  const std::size_t ncodewords = nql - 1;
  const HuffmanCode<long int> code(ncodewords, quantized_data,
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

  const std::size_t nnz = ncodewords - std::count(code.frequencies.begin(),
                                                  code.frequencies.end(), 0);

  HuffmanEncodedStream out(nbits, nbytes, code.nmissed() * sizeof(int),
                           2 * nnz * sizeof(std::size_t));

  // Write frequency table.
  {
    std::size_t *p =
        reinterpret_cast<std::size_t *>(out.frequencies.data.get());
    const std::vector<std::size_t> &frequencies = code.frequencies;
    for (std::size_t i = 0; i < ncodewords; ++i) {
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

  int *missed = reinterpret_cast<int *>(out.missed.data.get());

  unsigned char offset = 0;
  for (const long int q : PseudoArray(quantized_data, n)) {
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

  {
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

  return out;
}

void huffman_decoding(long int *const quantized_data,
                      const std::size_t quantized_data_size,
                      unsigned char const *const out_data_hit,
                      const std::size_t out_data_hit_size,
                      unsigned char const *const out_data_miss,
                      const std::size_t out_data_miss_size,
                      unsigned char const *const out_tree,
                      const std::size_t out_tree_size) {
  std::size_t const *const cft = (std::size_t const *)out_tree;
  const int nonZeros = out_tree_size / (2 * sizeof(std::size_t));
  // The elements of the array are value-initialized (here, zero-initialized).
  std::size_t *const ft = new std::size_t[nql]();

  for (int j = 0; j < nonZeros; j++) {
    ft[cft[2 * j]] = cft[2 * j + 1];
  }

  my_priority_queue<htree_node> *const phtree = build_tree(ft);
  delete[] ft;

  unsigned int const *const buf = (unsigned int const *)out_data_hit;

  // The out_data_miss may not be aligned. Therefore, the code
  // here makes a new buffer.
  assert(not(out_data_miss_size % sizeof(int)));
  int *const miss_buf = new int[out_data_miss_size / sizeof(int)];
  if (out_data_miss_size) {
    std::memcpy(miss_buf, out_data_miss, out_data_miss_size);
  }

  int const *miss_bufp = miss_buf;

  std::size_t start_bit = 0;
  unsigned int mask = 0x80000000;

  long int *q = quantized_data;
  std::size_t i = 0;
  std::size_t num_missed = 0;
  while (q < (quantized_data + (quantized_data_size / sizeof(*q)))) {
    htree_node const *root = phtree->top();
    assert(root);

    std::size_t len = 0;
    int offset = 0;
    while (root->left) {
      int flag = *(buf + start_bit / 32 + offset) & mask;
      if (!flag) {
        root = root->left;
      } else {
        root = root->right;
      }

      len++;

      mask >>= 1;
      if (!mask) {
        mask = 0x80000000;
        offset = 1;
      } else {
        //        offset = 0;
      }
    }

    if (root->q != 0) {
      *q = root->q - nql / 2;

    } else {
      *q = *miss_bufp - nql / 2;

      miss_bufp++;
      num_missed++;
    }

    q++;
    i++;

    start_bit += len;
  }

  assert(start_bit == out_data_hit_size);
  assert(sizeof(int) * num_missed == out_data_miss_size);

  // Avoid unused argument warning. If NDEBUG is defined, then the assert
  // becomes empty and out_data_hit_size is unused. Tell the compiler that
  // is OK and expected.
  (void)out_data_hit_size;

  delete[] miss_buf;
  free_tree(phtree);
}

} // namespace mgard
