#include "utilities.hpp"

#include <cassert>
#include <cstddef>

#include <limits>
#include <stdexcept>

namespace mgard {

template <typename Symbol>
bool HuffmanCode<Symbol>::HeldCountGreater::
operator()(const typename HuffmanCode<Symbol>::Node &a,
           const typename HuffmanCode<Symbol>::Node &b) const {
  return a->count > b->count;
}

template <typename Symbol> void HuffmanCode<Symbol>::set_endpoints() {
  // Haven't carefully checked what the minimum acceptable value is.
  if (not ncodewords) {
    throw std::invalid_argument("`ncodewords` must be positive.");
  }
  const Symbol SYMBOL_MAX = std::numeric_limits<Symbol>::max();
  const Symbol SYMBOL_MIN = std::numeric_limits<Symbol>::min();

  const std::size_t max_symbol_ = (ncodewords + 1) / 2 - 1;
  const std::size_t opp_min_symbol_ = ncodewords / 2;

  // There is surely a better way of doing this. Lots of potential issues with
  // directly comparing `opp_min_symbol_` and `-SYMBOL_MIN`. `-SYMBOL_MIN`
  // can't necessarily be represented as a `Symbol`, for example. Trying to
  // avoid overflows.
  std::size_t a = opp_min_symbol_;
  Symbol b = SYMBOL_MIN;
  while (a) {
    a /= 2;
    b /= 2;
  }
  if (not b) {
    // Only a "risk" because we haven't actually established that
    // `opp_min_symbol_` is greater in magnitude than `SYMBOL_MIN`.
    throw std::overflow_error(
        "risk that minimum symbol cannot be represented in symbol type");
  } else if (opp_min_symbol_ > SYMBOL_MAX) {
    throw std::overflow_error(
        "opposite of minimum symbol canont be represented in symbol type");
  } else {
    endpoints.first = -static_cast<Symbol>(opp_min_symbol_);
  }

  // `opp_min_symbol_` is either equal to or one greater than `max_symbol_`,
  // and we checked above that `opp_min_symbol <= SYMBOL_MAX`. So, we know
  // that `max_symbol_ <= SYMBOL_MAX` here.
  endpoints.second = max_symbol_;
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

template <typename Symbol>
HuffmanCode<Symbol>::HuffmanCode(const std::size_t ncodewords,
                                 Symbol const *const begin,
                                 Symbol const *const end)
    : ncodewords(ncodewords), frequencies(ncodewords), codewords(ncodewords) {
  set_endpoints();
  populate_frequencies(begin, end);
  create_code_creation_tree();
  recursively_set_codewords(queue.top(), {});
}

template <typename Symbol>
HuffmanCode<Symbol>::HuffmanCode(
    const std::size_t ncodewords,
    const std::vector<std::pair<std::size_t, std::size_t>> &pairs)
    : ncodewords(ncodewords), frequencies(ncodewords), codewords(ncodewords) {
  set_endpoints();
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
