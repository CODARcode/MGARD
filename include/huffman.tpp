#include "utilities.hpp"

#include <cassert>
#include <cstddef>

#include <stdexcept>

namespace mgard {

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

template <typename Symbol>
void HuffmanCode<Symbol>::populate_frequencies(Symbol const *const begin,
                                               Symbol const *const end) {
  for (const Symbol symbol :
       RangeSlice<Symbol const *const>{.begin_ = begin, .end_ = end}) {
    ++frequencies.at(index(symbol));
  }
}

template <typename Symbol>
Symbol
HuffmanCode<Symbol>::decode(const typename HuffmanCode<Symbol>::Node &leaf,
                            Symbol const *&missed) const {
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

} // namespace mgard
