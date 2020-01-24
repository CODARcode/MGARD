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

} // namespace mgard
