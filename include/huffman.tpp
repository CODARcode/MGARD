#include "utilities.hpp"

#include <cassert>

#include <limits>
#include <queue>
#include <stdexcept>
#include <type_traits>

namespace mgard {

//! This is used in the instantization of `std::priority_queue`.
template <typename T> struct HeldCountGreater {
  bool operator()(const T &a, const T &b) const { return a->count > b->count; }
};

template <typename Symbol>
HuffmanCode<Symbol>::HuffmanCode(const std::size_t ncodewords,
                                 Symbol const *const begin,
                                 Symbol const *const end)
    : ncodewords(ncodewords), frequencies(ncodewords), codewords(ncodewords) {
  static_assert(std::is_integral<Symbol>::value and
                    std::is_signed<Symbol>::value,
                "symbol type must be signed and integral");
  // Haven't carefully checked what the minimum acceptable value is.
  if (not ncodewords) {
    throw std::invalid_argument("`ncodewords` must be positive.");
  }
  {
    const Symbol SYMBOL_MAX = std::numeric_limits<Symbol>::max();
    const Symbol SYMBOL_MIN = std::numeric_limits<Symbol>::min();

    const std::size_t max_symbol_ = (ncodewords + 1) / 2 - 1;
    const std::size_t opp_min_symbol_ = ncodewords / 2;

    // TODO: There is surely a better way of doing this. Lots of potential
    // issues with directly comparing `opp_min_symbol_` and `-SYMBOL_MIN`.
    // `-SYMBOL_MIN` can't necessarily be represented as a `Symbol`, for
    // example. Trying to avoid overflows.
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
      min_symbol = -static_cast<Symbol>(opp_min_symbol_);
    }

    // `opp_min_symbol_` is either equal to or one greater than `max_symbol_`,
    // and we checked above that `opp_min_symbol <= SYMBOL_MAX`. So, we know
    // that `max_symbol_ <= SYMBOL_MAX` here.
    max_symbol = max_symbol_;
  }
  for (const Symbol symbol :
       RangeSlice<Symbol const *const>{.begin_ = begin, .end_ = end}) {
    ++frequencies.at(index(symbol));
  }

  using T = std::shared_ptr<CodeCreationTreeNode>;
  std::priority_queue<T, std::vector<T>, HeldCountGreater<T>> queue;

  // We can't quite use a `ZippedRange` here, I think, because
  // `ZippedRange::iterator` doesn't expose the underlying iterators and
  // we want a pointer to the codeword.
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

  recursively_set_codewords(queue.top(), {});
}

template <typename Symbol> std::size_t HuffmanCode<Symbol>::nmissed() const {
  return frequencies.at(0);
}

template <typename Symbol>
bool HuffmanCode<Symbol>::out_of_range(const Symbol symbol) const {
  return symbol < min_symbol or symbol > max_symbol;
}

template <typename Symbol>
std::size_t HuffmanCode<Symbol>::index(const Symbol symbol) const {
  return out_of_range(symbol) ? 0 : 1 + symbol - min_symbol;
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
