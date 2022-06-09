#include "utilities.hpp"

#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>

#include <limits>
#include <numeric>
#include <stdexcept>

#include "format.hpp"

#include "proto/mgard.pb.h"

namespace mgard {

// Aliases for compound message field types.
namespace {

using Endpoints = google::protobuf::RepeatedField<google::protobuf::int64>;
using Missed = google::protobuf::RepeatedField<google::protobuf::int64>;
using Frequencies =
    google::protobuf::Map<google::protobuf::uint64, google::protobuf::uint64>;

} // namespace

template <typename Symbol>
bool HuffmanCode<Symbol>::HeldCountGreater::
operator()(const typename HuffmanCode<Symbol>::Node &a,
           const typename HuffmanCode<Symbol>::Node &b) const {
  return a->count > b->count;
}

template <typename Symbol>
void HuffmanCode<Symbol>::create_code_creation_tree() {
  // We can't quite use a `ZippedRange` here, I think, because
  // `ZippedRange::iterator` doesn't expose the underlying iterators and we want
  // a pointer to the codeword.
  typename std::vector<std::size_t>::const_iterator p = frequencies.cbegin();
  HuffmanCodeword *q = codewords.data();
  for (std::size_t i = 0; i < ncodewords; ++i) {
    const std::size_t count = *p;
    if (count) {
      queue.push(std::make_shared<CodeCreationTreeNode>(q, count));
    }
    ++p;
    ++q;
  }
  while (queue.size() > 1) {
    const std::shared_ptr<CodeCreationTreeNode> a = queue.top();
    queue.pop();
    const std::shared_ptr<CodeCreationTreeNode> b = queue.top();
    queue.pop();

    queue.push(std::make_shared<CodeCreationTreeNode>(a, b));
  }
}

// This default will be used for `std::int{8,16}_t` We'll specialize the default
// for `std::int{32,64}_t` in the implementation file.
template <typename Symbol>
const std::pair<Symbol, Symbol> HuffmanCode<Symbol>::default_endpoints = {
    std::numeric_limits<Symbol>::min(), std::numeric_limits<Symbol>::max()};

// I believe these are called 'template specialization declarations.'
template <>
const std::pair<std::int32_t, std::int32_t>
    HuffmanCode<std::int32_t>::default_endpoints;

template <>
const std::pair<std::int64_t, std::int64_t>
    HuffmanCode<std::int64_t>::default_endpoints;

template <typename Symbol>
void HuffmanCode<Symbol>::populate_frequencies(Symbol const *const begin,
                                               Symbol const *const end) {
  for (const Symbol symbol :
       RangeSlice<Symbol const *const>{.begin_ = begin, .end_ = end}) {
    ++frequencies.at(index(symbol));
  }
}

template <typename Symbol>
template <typename It>
Symbol
HuffmanCode<Symbol>::decode(const typename HuffmanCode<Symbol>::Node &leaf,
                            It &missed) const {
  const std::ptrdiff_t offset = leaf->codeword - codewords.data();
  // If `offset == 0`, this is the leaf corresponding to out-of-range symbols.
  assert(offset >= 0);
  return offset ? endpoints.first + (offset - 1) : *missed++;
}

template <typename Symbol>
void HuffmanCode<Symbol>::populate_frequencies(
    const std::vector<std::pair<std::size_t, std::size_t>> &pairs) {
  for (auto [index, frequency] : pairs) {
    frequencies.at(index) = frequency;
  }
}

namespace {

template <typename Symbol>
std::size_t
ncodewords_from_endpoints(const std::pair<Symbol, Symbol> &endpoints) {
  if (endpoints.first > endpoints.second) {
    throw std::invalid_argument(
        "maximum symbol must be greater than or equal to minimum symbol");
  }
  // One for the 'missed' symbol, and the endpoints are inclusive.
  // Overflow is possible in the subtraction `endpoints.second -
  // endpoints.first` (suppose `Symbol` is `char` and `endpoints` is `{CHAR_MIN,
  // CHAR_MAX}`. Casting to `std::int64_t` should avoid the problem in all
  // practical cases.
  const std::size_t ncodewords = 1 +
                                 static_cast<std::int64_t>(endpoints.second) -
                                 static_cast<std::int64_t>(endpoints.first) + 1;
  return ncodewords;
}

} // namespace

template <typename Symbol>
HuffmanCode<Symbol>::HuffmanCode(const std::pair<Symbol, Symbol> &endpoints,
                                 Symbol const *const begin,
                                 Symbol const *const end)
    : endpoints(endpoints), ncodewords(ncodewords_from_endpoints(endpoints)),
      frequencies(ncodewords), codewords(ncodewords) {
  populate_frequencies(begin, end);
  create_code_creation_tree();
  recursively_set_codewords(queue.top(), {});
}

template <typename Symbol>
HuffmanCode<Symbol>::HuffmanCode(Symbol const *const begin,
                                 Symbol const *const end)
    : HuffmanCode(default_endpoints, begin, end) {}

template <typename Symbol>
HuffmanCode<Symbol>::HuffmanCode(
    const std::pair<Symbol, Symbol> &endpoints,
    const std::vector<std::pair<std::size_t, std::size_t>> &pairs)
    : endpoints(endpoints), ncodewords(ncodewords_from_endpoints(endpoints)),
      frequencies(ncodewords), codewords(ncodewords) {
  populate_frequencies(pairs);
  create_code_creation_tree();
  recursively_set_codewords(queue.top(), {});
}

template <typename Symbol> std::size_t HuffmanCode<Symbol>::nmissed() const {
  return frequencies.at(0);
}

template <typename Symbol>
bool HuffmanCode<Symbol>::out_of_range(const Symbol symbol) const {
  return symbol < endpoints.first or symbol > endpoints.second;
}

template <typename Symbol>
std::size_t HuffmanCode<Symbol>::index(const Symbol symbol) const {
  return out_of_range(symbol) ? 0 : 1 + symbol - endpoints.first;
}

template <typename Symbol>
void HuffmanCode<Symbol>::recursively_set_codewords(
    const std::shared_ptr<CodeCreationTreeNode> &node,
    const HuffmanCodeword codeword) {
  const bool children = node->left;
  assert(children == static_cast<bool>(node->right));
  if (children) {
    recursively_set_codewords(node->left, codeword.left());
    recursively_set_codewords(node->right, codeword.right());
  } else {
    *node->codeword = codeword;
  }
}

template <typename Symbol>
MemoryBuffer<unsigned char> huffman_encode(Symbol const *const begin,
                                           const std::size_t n) {
  const HuffmanCode<Symbol> code(begin, begin + n);

  std::vector<std::size_t> lengths;
  for (const HuffmanCodeword &codeword : code.codewords) {
    lengths.push_back(codeword.length);
  }
  const std::size_t nbits =
      std::inner_product(code.frequencies.begin(), code.frequencies.end(),
                         lengths.begin(), static_cast<std::size_t>(0));
  const std::size_t nbytes = (nbits + CHAR_BIT - 1) / CHAR_BIT;

  pb::HuffmanHeader header;
  header.set_index_mapping(pb::HuffmanHeader::INCLUSIVE_RANGE);
  header.set_codeword_mapping(pb::HuffmanHeader::INDEX_FREQUENCY_PAIRS);
  header.set_missed_encoding(pb::HuffmanHeader::LITERAL);
  header.set_hit_encoding(pb::HuffmanHeader::RUN_TOGETHER);

  header.add_endpoints(code.endpoints.first);
  header.add_endpoints(code.endpoints.second);
  header.set_nbits(nbits);

  Frequencies &frequencies = *header.mutable_frequencies();
  {
    std::size_t i = 0;
    for (const std::size_t frequency : code.frequencies) {
      if (frequency) {
        frequencies.insert({i, frequency});
      }
      ++i;
    }
  }

  Missed &missed_ = *header.mutable_missed();
  missed_.Resize(code.nmissed(), 0);
  Missed::iterator missed = missed_.begin();

  // Zero-initialize the bytes.
  unsigned char *const hit_ = new unsigned char[nbytes]();
  unsigned char *hit = hit_;

  unsigned char offset = 0;
  for (const Symbol q : PseudoArray(begin, n)) {
    if (code.out_of_range(q)) {
      *missed++ = q;
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

  const std::uint_least64_t nheader = header.ByteSize();
  MemoryBuffer<unsigned char> out(HEADER_SIZE_SIZE + nheader + nbytes);
  {
    unsigned char *p = out.data.get();
    const std::array<unsigned char, HEADER_SIZE_SIZE> nheader_ =
        serialize_header_size(nheader);
    std::copy(nheader_.begin(), nheader_.end(), p);
    p += HEADER_SIZE_SIZE;

    header.SerializeToArray(p, nheader);
    p += nheader;

    std::copy(hit_, hit_ + nbytes, p);
    p += nbytes;
  }

  delete[] hit_;

  return out;
}

template <typename Symbol>
MemoryBuffer<Symbol> huffman_decode(const MemoryBuffer<unsigned char> &buffer) {
  BufferWindow window(buffer.data.get(), buffer.size);
  const std::uint_least64_t nheader = read_header_size(window);
  pb::HuffmanHeader header = read_message<pb::HuffmanHeader>(window, nheader);

  if (header.index_mapping() != pb::HuffmanHeader::INCLUSIVE_RANGE) {
    throw std::runtime_error("unrecognized Huffman index mapping");
  }
  const Endpoints &endpoints_ = header.endpoints();
  if (endpoints_.size() != 2) {
    throw std::runtime_error("received an unexpected number of endpoints");
  }
  const std::pair<std::size_t, std::size_t> endpoints(endpoints_.Get(0),
                                                      endpoints_.Get(1));

  if (header.codeword_mapping() != pb::HuffmanHeader::INDEX_FREQUENCY_PAIRS) {
    throw std::runtime_error("unrecognized Huffman codeword mapping");
  }
  const Frequencies &frequencies_ = header.frequencies();
  // TODO: Change `HuffmanCode` constructor so it can take a pair of iterators
  // dereferencing to (something convertible to)
  // `std::pair<std::size_t, std::size_t>`s directly.
  const std::vector<std::pair<std::size_t, std::size_t>> pairs(
      frequencies_.begin(), frequencies_.end());

  if (header.missed_encoding() != pb::HuffmanHeader::LITERAL) {
    throw std::runtime_error("unrecognized Huffman missed buffer encoding");
  }
  const Missed &missed_ = header.missed();
  Missed::const_iterator missed = missed_.cbegin();

  if (header.hit_encoding() != pb::HuffmanHeader::RUN_TOGETHER) {
    throw std::runtime_error("unrecognized Huffman hit buffer encoding");
  }

  const std::size_t nbits = header.nbits();
  const std::size_t nbytes = (nbits + CHAR_BIT - 1) / CHAR_BIT;
  if (window.current + nbytes != window.end) {
    throw std::runtime_error("number of bits in hit buffer inconsistent with "
                             "number of bytes in hit buffer");
  }

  const HuffmanCode<Symbol> code(endpoints, pairs);
  // TODO: Maybe add a member function for this.
  const std::size_t nout =
      std::accumulate(code.frequencies.begin(), code.frequencies.end(),
                      static_cast<std::size_t>(0));
  MemoryBuffer<Symbol> out(nout);
  Symbol *q = out.data.get();

  const Bits bits(window.current, window.current + nbits / CHAR_BIT,
                  nbits % CHAR_BIT);
  std::size_t nbits_read = 0;
  const typename HuffmanCode<Symbol>::Node root = code.queue.top();
  assert(root);
  Bits::iterator b = bits.begin();
  for (std::size_t i = 0; i < nout; ++i) {
    typename HuffmanCode<Symbol>::Node node;
    for (node = root; node->left;
         node = *b++ ? node->right : node->left, ++nbits_read)
      ;
    // TODO: Make sure `HuffmanCode::decode` can properly take `missed` (not
    // relying on `google::protobuf::uint64` being the same as `std::size_t` or
    // anything).
    const Symbol decoded = code.decode(node, missed);
    *q++ = decoded;
  }
  assert(nbits_read == nbits);
  assert(missed == missed_.cend());

  return out;
}

} // namespace mgard
