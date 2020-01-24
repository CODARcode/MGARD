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

template <typename T>
Enumeration<T>::Enumeration(const T &container) : container(container) {}

template <typename T>
typename Enumeration<T>::iterator Enumeration<T>::begin() const {
  return iterator(*this, 0, container.begin());
}

template <typename T>
typename Enumeration<T>::iterator Enumeration<T>::end() const {
  return iterator(*this, container.size(), container.end());
}

template <typename T>
bool operator==(const Enumeration<T> &a, const Enumeration<T> &b) {
  return a.container == b.container;
}

template <typename T>
bool operator!=(const Enumeration<T> &a, const Enumeration<T> &b) {
  return !operator==(a, b);
}

template <typename T>
Enumeration<T>::iterator::iterator(const Enumeration<T> &iterable,
                                   const typename T::size_type index,
                                   const typename T::const_iterator inner)
    : iterable(iterable), index(index), inner(inner) {}

template <typename T>
bool Enumeration<T>::iterator::
operator==(const Enumeration<T>::iterator &other) const {
  return (iterable == other.iterable && index == other.index &&
          inner == other.inner);
}

template <typename T>
bool Enumeration<T>::iterator::
operator!=(const Enumeration<T>::iterator &other) const {
  return !operator==(other);
}

template <typename T>
typename Enumeration<T>::iterator &Enumeration<T>::iterator::operator++() {
  ++index;
  ++inner;
  return *this;
}

template <typename T>
typename Enumeration<T>::iterator Enumeration<T>::iterator::operator++(int) {
  const iterator tmp = *this;
  operator++();
  return tmp;
}

template <typename T>
std::pair<typename T::size_type, typename T::value_type>
    Enumeration<T>::iterator::operator*() const {
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
