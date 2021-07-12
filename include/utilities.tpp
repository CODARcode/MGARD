#include <algorithm>
#include <stdexcept>

namespace mgard {

template <typename T>
PseudoArray<T>::PseudoArray(T *const data, const std::size_t size)
    : data(data), size(size) {}

template <typename T>
PseudoArray<T>::PseudoArray(T *const data, const int _size)
    : PseudoArray<T>::PseudoArray(data, static_cast<std::size_t>(_size)) {
  if (_size < 0) {
    throw std::invalid_argument("size must be nonnegative");
  }
}

template <typename T> T *PseudoArray<T>::begin() const { return data; }

template <typename T> T *PseudoArray<T>::end() const { return data + size; }

template <typename T> T PseudoArray<T>::operator[](const std::size_t i) const {
  if (i >= size) {
    throw std::out_of_range("index too large");
  }
  return data[i];
}

template <typename It>
Enumeration<It>::Enumeration(const It begin, const It end)
    : begin_(begin), end_(end) {}

template <typename It>
template <typename T>
Enumeration<It>::Enumeration(const T &container)
    : begin_(container.begin()), end_(container.end()) {}

template <typename It>
typename Enumeration<It>::iterator Enumeration<It>::begin() const {
  return iterator(*this, 0, begin_);
}

template <typename It>
typename Enumeration<It>::iterator Enumeration<It>::end() const {
  return iterator(*this, end_ - begin_, end_);
}

template <typename It>
bool operator==(const Enumeration<It> &a, const Enumeration<It> &b) {
  return a.begin_ == b.begin_ && a.end_ == b.end_;
}

template <typename It>
bool operator!=(const Enumeration<It> &a, const Enumeration<It> &b) {
  return !operator==(a, b);
}

template <typename It>
Enumeration<It>::iterator::iterator(const Enumeration<It> &iterable,
                                    const std::size_t index, const It inner)
    : iterable(iterable), index(index), inner(inner) {}

template <typename It>
bool Enumeration<It>::iterator::
operator==(const Enumeration<It>::iterator &other) const {
  return (&iterable == &other.iterable || iterable == other.iterable) &&
         index == other.index && inner == other.inner;
}

template <typename It>
bool Enumeration<It>::iterator::
operator!=(const Enumeration<It>::iterator &other) const {
  return !operator==(other);
}

template <typename It>
typename Enumeration<It>::iterator &Enumeration<It>::iterator::operator++() {
  ++index;
  ++inner;
  return *this;
}

template <typename It>
typename Enumeration<It>::iterator Enumeration<It>::iterator::operator++(int) {
  const iterator tmp = *this;
  operator++();
  return tmp;
}

template <typename It>
typename Enumeration<It>::iterator::reference Enumeration<It>::iterator::
operator*() const {
  return {index, *inner};
}

template <typename It, typename Jt>
ZippedRange<It, Jt>::ZippedRange(const It begin_first, const It end_first,
                                 const Jt begin_second, const Jt end_second)
    : begin_first(begin_first), end_first(end_first),
      begin_second(begin_second), end_second(end_second) {}

template <typename It, typename Jt>
template <typename T, typename U>
ZippedRange<It, Jt>::ZippedRange(const T &container_first,
                                 const U &container_second)
    : begin_first(container_first.begin()), end_first(container_first.end()),
      begin_second(container_second.begin()),
      end_second(container_second.end()) {}

template <typename It, typename Jt>
typename ZippedRange<It, Jt>::iterator ZippedRange<It, Jt>::begin() const {
  return iterator(*this, begin_first, begin_second);
}

template <typename It, typename Jt>
typename ZippedRange<It, Jt>::iterator ZippedRange<It, Jt>::end() const {
  return iterator(*this, end_first, end_second);
}

template <typename It, typename Jt>
bool operator==(const ZippedRange<It, Jt> &a, const ZippedRange<It, Jt> &b) {
  return (a.begin_first == b.begin_first && a.end_first == b.end_first &&
          a.begin_second == b.begin_second && a.end_second == b.end_second);
}

template <typename It, typename Jt>
bool operator!=(const ZippedRange<It, Jt> &a, const ZippedRange<It, Jt> &b) {
  return !operator==(a, b);
}

template <typename It, typename Jt>
ZippedRange<It, Jt>::iterator::iterator(const ZippedRange<It, Jt> &iterable,
                                        const It inner_first,
                                        const Jt inner_second)
    : iterable(iterable), inner_first(inner_first), inner_second(inner_second) {
}

template <typename It, typename Jt>
typename ZippedRange<It, Jt>::iterator &ZippedRange<It, Jt>::iterator::
operator=(const ZippedRange<It, Jt>::iterator &other) {
  if (iterable != other.iterable) {
    throw std::domain_error(
        "can only assign to iterators to the same iterable");
  }
  inner_first = other.inner_first;
  inner_second = other.inner_second;
  return *this;
}

// Iteration won't stop when only one of the iterators reaches its end.
template <typename It, typename Jt>
bool ZippedRange<It, Jt>::iterator::
operator==(const ZippedRange<It, Jt>::iterator &other) const {
  return (&iterable == &other.iterable || iterable == other.iterable) &&
         inner_first == other.inner_first && inner_second == other.inner_second;
}

template <typename It, typename Jt>
bool ZippedRange<It, Jt>::iterator::
operator!=(const ZippedRange<It, Jt>::iterator &other) const {
  return !operator==(other);
}

template <typename It, typename Jt>
typename ZippedRange<It, Jt>::iterator &ZippedRange<It, Jt>::iterator::
operator++() {
  ++inner_first;
  ++inner_second;
  return *this;
}

template <typename It, typename Jt>
typename ZippedRange<It, Jt>::iterator ZippedRange<It, Jt>::iterator::
operator++(int) {
  const ZippedRange<It, Jt>::iterator tmp = *this;
  operator++();
  return tmp;
}

template <typename It, typename Jt>
typename ZippedRange<It, Jt>::iterator::reference
    ZippedRange<It, Jt>::iterator::operator*() const {
  return {*inner_first, *inner_second};
}

template <typename It> It RangeSlice<It>::begin() const { return begin_; }

template <typename It> It RangeSlice<It>::end() const { return end_; }

template <typename It> std::size_t RangeSlice<It>::size() const {
  return end_ - begin_;
}

template <typename It>
bool operator==(const RangeSlice<It> &a, const RangeSlice<It> &b) {
  return a.begin_ == b.begin_ && a.end_ == b.end_;
}

template <typename It>
bool operator!=(const RangeSlice<It> &a, const RangeSlice<It> &b) {
  return !operator==(a, b);
}

template <typename T, std::size_t N>
CartesianProduct<T, N>::CartesianProduct(const std::array<T, N> factors)
    : factors(factors) {
  if (std::any_of(factors.begin(), factors.end(), [](const T &factor) -> bool {
        return factor.begin() == factor.end();
      })) {
    throw std::invalid_argument("none of the factors may be empty");
  }
}

template <typename T, std::size_t N>
bool operator==(const CartesianProduct<T, N> &a,
                const CartesianProduct<T, N> &b) {
  return a.factors == b.factors;
}

template <typename T, std::size_t N>
bool operator!=(const CartesianProduct<T, N> &a,
                const CartesianProduct<T, N> &b) {
  return !operator==(a, b);
}

template <typename T, std::size_t N>
typename CartesianProduct<T, N>::iterator
CartesianProduct<T, N>::begin() const {
  std::array<typename iterator::T_iterator, N> inner;
  for (std::size_t i = 0; i < N; ++i) {
    inner.at(i) = factors.at(i).begin();
  }
  return iterator(*this, inner);
}

template <typename T, std::size_t N>
typename CartesianProduct<T, N>::iterator CartesianProduct<T, N>::end() const {
  std::array<typename iterator::T_iterator, N> inner;
  inner.at(0) = factors.at(0).end();
  for (std::size_t i = 1; i < N; ++i) {
    inner.at(i) = factors.at(i).begin();
  }
  return iterator(*this, inner);
}

template <typename T, std::size_t N>
CartesianProduct<T, N>::iterator::iterator(
    const CartesianProduct<T, N> &iterable,
    const std::array<typename CartesianProduct<T, N>::iterator::T_iterator, N>
        inner)
    : iterable(&iterable), inner(inner) {}

template <typename T, std::size_t N>
bool CartesianProduct<T, N>::iterator::
operator==(const CartesianProduct<T, N>::iterator &other) const {
  return (iterable == other.iterable || *iterable == *(other.iterable)) &&
         inner == other.inner;
}

template <typename T, std::size_t N>
bool CartesianProduct<T, N>::iterator::
operator!=(const CartesianProduct<T, N>::iterator &other) const {
  return !operator==(other);
}

template <typename T, std::size_t N>
typename CartesianProduct<T, N>::iterator &CartesianProduct<T, N>::iterator::
operator++() {
  for (std::size_t i = N; i != 0; --i) {
    const std::size_t j = i - 1;
    const T &factor = iterable->factors.at(j);
    T_iterator &it = inner.at(j);
    if (++it != factor.end()) {
      break;
    } else if (j) {
      // Unless we've hit the end of the product, reset this dimension's
      // iterator and repeat the loop with the 'next' dimension.
      it = factor.begin();
      // Otherwise, we leave this dimension's iterator at the end so we can
      // distinguish the end of the product from the beginning of the product.
    }
  }
  return *this;
}

template <typename T, std::size_t N>
typename CartesianProduct<T, N>::iterator CartesianProduct<T, N>::iterator::
operator++(int) {
  const iterator tmp = *this;
  operator++();
  return tmp;
}

template <typename T, std::size_t N>
typename CartesianProduct<T, N>::iterator
CartesianProduct<T, N>::iterator::predecessor(const std::size_t i) const {
  check_dimension_index_bounds<N>(i);
  std::array<T_iterator, N> inner_predecessor = inner;
  T_iterator &p = inner_predecessor.at(i);
  if (p != iterable->factors.at(i).begin()) {
    --p;
  }
  return iterator(*iterable, inner_predecessor);
}

template <typename T, std::size_t N>
typename CartesianProduct<T, N>::iterator
CartesianProduct<T, N>::iterator::successor(const std::size_t i) const {
  check_dimension_index_bounds<N>(i);
  std::array<T_iterator, N> inner_successor = inner;
  T_iterator &p = inner_successor.at(i);
  if (++p == iterable->factors.at(i).end()) {
    --p;
  }
  return iterator(*iterable, inner_successor);
}

template <typename T, std::size_t N>
typename CartesianProduct<T, N>::iterator::reference
    CartesianProduct<T, N>::iterator::operator*() const {
  reference value;
  for (std::size_t i = 0; i < N; ++i) {
    value.at(i) = *inner.at(i);
  }
  return value;
}

template <std::size_t N>
void check_dimension_index_bounds(const std::size_t dimension) {
  if (dimension >= N) {
    throw std::out_of_range("dimension index out of range encountered");
  }
}

} // namespace mgard
