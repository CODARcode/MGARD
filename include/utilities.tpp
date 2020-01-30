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

template <typename T, typename U>
ZippedRange<T, U>::ZippedRange(
  const T &container_first, const U &container_second
):
  container_first(container_first),
  container_second(container_second)
{
}

template <typename T, typename U>
typename ZippedRange<T, U>::iterator ZippedRange<T, U>::begin() const {
  return iterator(*this, container_first.begin(), container_second.begin());
}

template <typename T, typename U>
typename ZippedRange<T, U>::iterator ZippedRange<T, U>::end() const {
  return iterator(*this, container_first.end(), container_second.end());
}

template <typename T, typename U>
bool operator==(const ZippedRange<T, U> &a, const ZippedRange<T, U> &b) {
  return (
    a.container_first == b.container_first &&
    a.container_second == b.container_second
  );
}

template <typename T, typename U>
bool operator!=(const ZippedRange<T, U> &a, const ZippedRange<T, U> &b) {
  return !operator==(a, b);
}

template <typename T, typename U>
ZippedRange<T, U>::iterator::iterator(
  const ZippedRange<T, U> &iterable,
  const typename T::const_iterator inner_first,
  const typename U::const_iterator inner_second
):
  iterable(iterable),
  inner_first(inner_first),
  inner_second(inner_second)
{
}

//Iteration won't stop when only one of the iterators reaches its end.
template <typename T, typename U>
bool ZippedRange<T, U>::iterator::operator==(
  const ZippedRange<T, U>::iterator &other
) const {
  return (
    iterable == other.iterable && inner_first == other.inner_first &&
    inner_second == other.inner_second
  );
}

template <typename T, typename U>
bool ZippedRange<T, U>::iterator::operator!=(
  const ZippedRange<T, U>::iterator &other
) const {
  return !operator==(other);
}

template <typename T, typename U>
typename ZippedRange<T, U>::iterator &
ZippedRange<T, U>::iterator::operator++() {
  ++inner_first;
  ++inner_second;
  return *this;
}

template <typename T, typename U>
typename ZippedRange<T, U>::iterator
ZippedRange<T, U>::iterator::operator++(int) {
  const ZippedRange<T, U>::iterator tmp = *this;
  operator++();
  return tmp;
}

template <typename T, typename U>
std::pair<typename T::value_type, typename U::value_type>
ZippedRange<T, U>::iterator::operator*() const {
  return {*inner_first, *inner_second};
}

} // namespace mgard
