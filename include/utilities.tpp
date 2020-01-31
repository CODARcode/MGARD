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
  return (iterable == other.iterable && index == other.index &&
          inner == other.inner);
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
std::pair<std::size_t, typename std::iterator_traits<It>::value_type>
    Enumeration<It>::iterator::operator*() const {
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

// Iteration won't stop when only one of the iterators reaches its end.
template <typename It, typename Jt>
bool ZippedRange<It, Jt>::iterator::
operator==(const ZippedRange<It, Jt>::iterator &other) const {
  return (iterable == other.iterable && inner_first == other.inner_first &&
          inner_second == other.inner_second);
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
std::pair<typename std::iterator_traits<It>::value_type,
          typename std::iterator_traits<Jt>::value_type>
    ZippedRange<It, Jt>::iterator::operator*() const {
  return {*inner_first, *inner_second};
}

template <typename It>
RangeSlice<It>::RangeSlice(const It begin, const It end)
    : begin_(begin), end_(end) {}

template <typename It> It RangeSlice<It>::begin() const { return begin_; }

template <typename It> It RangeSlice<It>::end() const { return end_; }

} // namespace mgard
