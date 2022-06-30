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
using SubtableSizes = google::protobuf::RepeatedField<google::protobuf::uint64>;

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
std::pair<bool, Symbol> HuffmanCode<Symbol>::decode(
    const typename HuffmanCode<Symbol>::Node &leaf) const {
  const std::ptrdiff_t offset = leaf->codeword - codewords.data();
  // If `offset == 0`, this is the leaf corresponding to out-of-range symbols.
  assert(offset >= 0);
  return offset ? std::pair<bool, Symbol>(true, endpoints.first + (offset - 1))
                : std::pair<bool, Symbol>(false, {});
}

template <typename Symbol>
template <typename It>
void HuffmanCode<Symbol>::populate_frequencies(const It begin, const It end) {
  for (auto [index, frequency] : RangeSlice<It>{.begin_ = begin, .end_ = end}) {
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
template <typename It>
HuffmanCode<Symbol>::HuffmanCode(const std::pair<Symbol, Symbol> &endpoints,
                                 const It begin, const It end)
    : endpoints(endpoints), ncodewords(ncodewords_from_endpoints(endpoints)),
      frequencies(ncodewords), codewords(ncodewords) {
  populate_frequencies(begin, end);
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

namespace {

//! Maximum number of elements per frequency/missed subtable.
inline constexpr std::size_t SUBTABLE_MAX_SIZE = 1 << 20;

//! A logical table split into one or more subtables of moderate size.
//!
//! The logical table can be read by chaining the subtables.
template <typename Message, typename It> struct Supertable {
  // The beginning and size of a subtable.
  using Segment = std::pair<It, std::size_t>;

  //! Constructor.
  //!
  //! Construct an 'empty' `Supertable`. The data members will be given the
  //! right sizes, but for the most part they will not populated. That is left
  //! to derived class constructors or callers.
  //!
  //!\param nelements Total number of subtable entries.
  //!\param nbytes_subtables Sizes in bytes of the subtables (field in
  //! `pb::HuffmanHeader`). This field will be written to.
  Supertable(const std::size_t nelements, SubtableSizes &nbytes_subtables)
      : nsubtables((nelements + SUBTABLE_MAX_SIZE - 1) / SUBTABLE_MAX_SIZE),
        subtables(nsubtables), segments(nsubtables),
        nbytes_subtables(nbytes_subtables) {
    nbytes_subtables.Resize(nsubtables, 0);

    for (std::size_t i = 0; i + 1 < nsubtables; ++i) {
      segments.at(i).second = SUBTABLE_MAX_SIZE;
    }
    if (nsubtables) {
      // If `nelements` is an exact multiple of `SUBTABLE_MAX_SIZE` and not
      // zero, we need this last size to be `SUBTABLE_MAX_SIZE`, not `0`. If
      // `nelements` is zero, we won't be executing this statement.
      segments.back().second = nelements % SUBTABLE_MAX_SIZE
                                   ? nelements % SUBTABLE_MAX_SIZE
                                   : SUBTABLE_MAX_SIZE;
    }
  }

  //! Constructor.
  //!
  //! Construct a `Supertable` from a collection of parsed messages. This
  //! constructor leaves `segments` uninitialized. This is because `Supertable`
  //! doesn't know which field of `Message` is the subtable.
  //!
  //!\param nbytes_subtables Sizes in bytes of the subtables (field in
  //! `pb::HuffmanHeader`).
  //!\param window Window into buffer containing messages to be parsed.
  Supertable(SubtableSizes &nbytes_subtables, BufferWindow &window)
      : nsubtables(nbytes_subtables.size()), subtables(nsubtables),
        segments(nsubtables), nbytes_subtables(nbytes_subtables) {
    for (std::size_t i = 0; i < nsubtables; ++i) {
      subtables.at(i) = read_message<Message>(window, nbytes_subtables.Get(i));
    }
  }

  //! Calculate and store the sizes in bytes of the subtables.
  //!
  //! This function should be called once the subtables are populated.
  //! `nbytes_subtables` (a field in some `pb::HuffmanHeader`) will be modified.
  //! Subsequent changes to the subtables will invalidate the sizes.
  void calculate_nbytes_subtables() {
    for (std::size_t i = 0; i < nsubtables; ++i) {
      nbytes_subtables.Set(i, subtables.at(i).ByteSize());
    }
  }

  //! Calculate the total size in bytes of the subtables.
  //!
  //! This function assumes no changes have been made to the subtables since the
  //! last call to `calculate_nbytes_subtables`.
  std::size_t ByteSize() const {
    return std::accumulate(nbytes_subtables.begin(), nbytes_subtables.end(),
                           static_cast<std::size_t>(0));
  }

  //! Write the subtables out to a buffer.
  //!
  //!\param p Buffer to which to serialize the subtables.
  //!\param n Expected number of bytes that will be written.
  void SerializeToArray(void *const p, const std::size_t n) const {
    unsigned char *const p_ = reinterpret_cast<unsigned char *>(p);
    std::size_t total = 0;
    for (std::size_t i = 0; i < nsubtables; ++i) {
      const Message &subtable = subtables.at(i);
      const google::protobuf::uint64 nbytes_subtable = nbytes_subtables.Get(i);

      subtable.SerializeToArray(p_ + total, nbytes_subtable);
      total += nbytes_subtable;
    }
    if (total != n) {
      throw std::invalid_argument("serialization buffer size incorrect");
    }
  }

  //! Number of subtables.
  std::size_t nsubtables;

  //! Subtables.
  //!
  //! It might be better to name this member 'messages.' Elsewhere we use
  //! 'subtable' to refer to the fields of the messages containing the
  //! supertable elements. Using that vocabulary, a `pb::FrequencySubtable`
  //! would be a message while its `frequencies` field would be the subtable.
  std::vector<Message> subtables;

  //! Segments for a concatenated subtable chain.
  //!
  //! A `Chain<std::vector<Segment>::iterator>` can be constructed from this.
  std::vector<Segment> segments;

  //! Sizes in bytes of the subtables.
  SubtableSizes &nbytes_subtables;
};

//! A logical frequency table split into one or more subtables of moderate size.
struct FrequencySupertable
    : Supertable<pb::FrequencySubtable, Frequencies::iterator> {
  //! Constructor.
  //!
  //! Construct and populate a `FrequencySupertable` from a vector of symbol
  //! frequencies.
  //!
  //!\param frequencies Symbol frequencies to store in the subtables.
  //!\param nbytes_subtables Sizes in bytes of the subtables (field in
  //! `pb::HuffmanHeader`). This field will be written to.
  FrequencySupertable(const std::vector<std::size_t> &frequencies,
                      SubtableSizes &nbytes_subtables)
      : Supertable(std::count_if(frequencies.begin(), frequencies.end(),
                                 [](const std::size_t frequency) -> bool {
                                   return frequency;
                                 }),
                   nbytes_subtables) {
    // `i` is the index of the subtable we're inserting into. (Technically
    // we're inserting into the subtable's frequency map field rather than
    // the subtable itself.) `j` is the number of entries we've inserted
    // into subtable `i`. `k` is the index in the vector of frequencies
    // passed to the constructor.
    std::size_t k = 0;
    for (std::size_t i = 0; i < nsubtables; ++i) {
      Frequencies &frequencies_ = *subtables.at(i).mutable_frequencies();
      Segment &segment = segments.at(i);
      // How big `frequencies_` should be when we're done.
      const std::size_t nfrequencies_ = segment.second;
      for (std::size_t j = 0; j < nfrequencies_; ++k) {
        const std::size_t frequency = frequencies.at(k);
        if (frequency) {
          frequencies_.insert({k, frequency});
          ++j;
        }
      }
      segment.first = frequencies_.begin();
    }

    calculate_nbytes_subtables();
  }

  //! Constructor.
  //!
  //! Construct a `FrequencySubtable` from a collection of parsed messages.
  //!
  //!\param nbytes_subtables Sizes in bytes of the subtables (field in
  //! `pb::HuffmanHeader`).
  //!\param window Window into buffer containing messages to be parsed.
  FrequencySupertable(SubtableSizes &nbytes_subtables, BufferWindow &window)
      : Supertable(nbytes_subtables, window) {
    for (std::size_t i = 0; i < nsubtables; ++i) {
      Segment &segment = segments.at(i);
      Frequencies &frequencies = *subtables.at(i).mutable_frequencies();

      segment.first = frequencies.begin();
      segment.second = frequencies.size();
    }
  }
};

//! A logical 'missed' table split into one or more subtables of moderate size.
struct MissedSupertable : Supertable<pb::MissedSubtable, Missed::iterator> {
  //! Constructor.
  //!
  //! Construct an 'empty' `MissedSupertable`. It is expected that the caller
  //! will subsequently write to the subtables using `Chain`.
  //!
  //!\param nmissed Number of missed symbols.
  //!\param nbytes_subtables Sizes in bytes of the subtables (field in
  //! `pb::HuffmanHeader`). This field will be written to.
  MissedSupertable(const std::size_t nmissed, SubtableSizes &nbytes_subtables)
      : Supertable(nmissed, nbytes_subtables) {
    for (std::size_t i = 0; i < nsubtables; ++i) {
      Missed &missed = *subtables.at(i).mutable_missed();
      Segment &segment = segments.at(i);
      // How big `missed` should be when we're done.
      const std::size_t nmissed = segment.second;

      missed.Resize(nmissed, 0);
      segment.first = missed.begin();
    }
  }

  //! Constructor.
  //!
  //! Construct a `MissedSubtable` from a collection of parsed messages.
  //!
  //!\param nbytes_subtables Sizes in bytes of the subtables (field in
  //! `pb::HuffmanHeader`).
  //!\param window Window into buffer containing messages to be parsed.
  MissedSupertable(SubtableSizes &nbytes_subtables, BufferWindow &window)
      : Supertable(nbytes_subtables, window) {
    for (std::size_t i = 0; i < nsubtables; ++i) {
      Segment &segment = segments.at(i);
      Missed &missed = *subtables.at(i).mutable_missed();

      segment.first = missed.begin();
      segment.second = missed.size();
    }
  }
};

} // namespace

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
  const std::size_t nbytes_hit = (nbits + CHAR_BIT - 1) / CHAR_BIT;

  pb::HuffmanHeader header;
  header.set_index_mapping(pb::HuffmanHeader::INCLUSIVE_RANGE);
  header.set_codeword_mapping(pb::HuffmanHeader::INDEX_FREQUENCY_PAIRS);
  header.set_missed_encoding(pb::HuffmanHeader::LITERAL);
  header.set_hit_encoding(pb::HuffmanHeader::RUN_TOGETHER);

  header.add_endpoints(code.endpoints.first);
  header.add_endpoints(code.endpoints.second);
  header.set_nbits(nbits);

  // Originally, `pb::HuffmanHeader` had a field each for the frequency and
  // 'missed' tables. Unfortunately, these tables can get very big. In
  // particular, if the error tolerance is very low, the quantized coefficients
  // will be very large, and many of them will be missed. This could result in
  // the size of the 'missed' table exceeding the (default) limit imposed by
  // `google::protobuf::CodedInputStream`. See <https://developers.google.com/
  // protocol-buffers/docs/reference/csharp/class/google/protobuf/
  // coded-input-stream#sizelimit>. As a workaround, we are splitting the
  // 'missed' table (and, for good measure, the frequency table, too) into a
  // sequence of subtables of moderate size.

  // This `FrequencySupertable` creates and populates the frequency subtables.
  FrequencySupertable frequency_supertable(
      code.frequencies, *header.mutable_nbytes_frequency_subtables());
  // This `MissedSupertable` creates but does not populate the 'missed'
  // subtables. We'll populate the subtables below, as we encode the stream.
  MissedSupertable missed_supertable(code.nmissed(),
                                     *header.mutable_nbytes_missed_subtables());

  // This `Chain` lets us treat the 'missed' subtables as a single logical
  // table. It frees us from manually keeping track of when we need to advance
  // from one subtable to the next.
  Chain<Missed::iterator> chained_missed_supertable(missed_supertable.segments);
  Chain<Missed::iterator>::iterator missed = chained_missed_supertable.begin();
  // Now we're ready to populate the 'missed' subtables in the course of
  // populating the 'hit' buffer.

  // Zero-initialize the bytes.
  unsigned char *const hit_ = new unsigned char[nbytes_hit]();
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

  // We're done writing to the 'missed' subtables, so we can now calculate their
  // serialized sizes. We need to do this before calling
  // `missed_supertable.ByteSize`.
  missed_supertable.calculate_nbytes_subtables();

  const std::uint_least64_t nheader = header.ByteSize();
  const std::size_t nbytes_frequency_supertable =
      frequency_supertable.ByteSize();
  const std::size_t nbytes_missed_supertable = missed_supertable.ByteSize();
  MemoryBuffer<unsigned char> out(HEADER_SIZE_SIZE + nheader +
                                  nbytes_frequency_supertable +
                                  nbytes_missed_supertable + nbytes_hit);
  {
    unsigned char *p = out.data.get();
    const std::array<unsigned char, HEADER_SIZE_SIZE> nheader_ =
        serialize_header_size(nheader);
    std::copy(nheader_.begin(), nheader_.end(), p);
    p += HEADER_SIZE_SIZE;

    header.SerializeToArray(p, nheader);
    p += nheader;

    frequency_supertable.SerializeToArray(p, nbytes_frequency_supertable);
    p += nbytes_frequency_supertable;

    missed_supertable.SerializeToArray(p, nbytes_missed_supertable);
    p += nbytes_missed_supertable;

    std::copy(hit_, hit_ + nbytes_hit, p);
    p += nbytes_hit;
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
  const std::pair<Symbol, Symbol> endpoints(endpoints_.Get(0),
                                            endpoints_.Get(1));

  if (header.codeword_mapping() != pb::HuffmanHeader::INDEX_FREQUENCY_PAIRS) {
    throw std::runtime_error("unrecognized Huffman codeword mapping");
  }
  // See the comments in `huffman_encode` for an explanation of why we use these
  // `Supertable`s and `Chain`s.
  FrequencySupertable frequency_supertable(
      *header.mutable_nbytes_frequency_subtables(), window);
  Chain<Frequencies::iterator> chained_frequency_supertable(
      frequency_supertable.segments);

  if (header.missed_encoding() != pb::HuffmanHeader::LITERAL) {
    throw std::runtime_error("unrecognized Huffman missed buffer encoding");
  }
  MissedSupertable missed_supertable(*header.mutable_nbytes_missed_subtables(),
                                     window);
  Chain<Missed::iterator> chained_missed_supertable(missed_supertable.segments);
  Chain<Missed::iterator>::iterator missed = chained_missed_supertable.begin();

  if (header.hit_encoding() != pb::HuffmanHeader::RUN_TOGETHER) {
    throw std::runtime_error("unrecognized Huffman hit buffer encoding");
  }

  const std::size_t nbits = header.nbits();
  const std::size_t nbytes = (nbits + CHAR_BIT - 1) / CHAR_BIT;
  if (window.current + nbytes != window.end) {
    throw std::runtime_error("number of bits in hit buffer inconsistent with "
                             "number of bytes in hit buffer");
  }

  const HuffmanCode<Symbol> code(endpoints,
                                 chained_frequency_supertable.begin(),
                                 chained_frequency_supertable.end());
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
    const std::pair<bool, Symbol> decoded = code.decode(node);
    *q++ = decoded.first ? decoded.second : *missed++;
  }
  assert(nbits_read == nbits);
  assert(missed == chained_missed_supertable.end());

  return out;
}

} // namespace mgard
