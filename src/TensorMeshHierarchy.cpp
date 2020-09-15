#include "TensorMeshHierarchy.hpp"

#include <stdexcept>

namespace mgard {

TensorIndexRange TensorIndexRange::singleton() {
  TensorIndexRange range;
  range.size_coarse = range.size_finest = 1;
  return range;
}

bool operator==(const TensorIndexRange &a, const TensorIndexRange &b) {
  return a.size_finest == b.size_finest && a.size_coarse == b.size_coarse;
}

bool operator!=(const TensorIndexRange &a, const TensorIndexRange &b) {
  return !operator==(a, b);
}

std::size_t TensorIndexRange::size() const { return size_coarse; }

TensorIndexRange::iterator TensorIndexRange::begin() const {
  return iterator(*this, 0);
}

TensorIndexRange::iterator TensorIndexRange::end() const {
  return iterator(*this, size_coarse);
}

TensorIndexRange::iterator::iterator(const TensorIndexRange &iterable,
                                     const std::size_t inner)
    : iterable(&iterable), inner(inner) {
  // `inner == iterable.size_coarse` is allowed for the iterator to the end.
  if (inner > iterable.size_coarse) {
    throw std::invalid_argument("index position is too large");
  }
}

bool TensorIndexRange::iterator::
operator==(const TensorIndexRange::iterator &other) const {
  return (iterable == other.iterable || *iterable == *other.iterable) &&
         inner == other.inner;
}

bool TensorIndexRange::iterator::
operator!=(const TensorIndexRange::iterator &other) const {
  return !operator==(other);
}

TensorIndexRange::iterator &TensorIndexRange::iterator::operator++() {
  ++inner;
  return *this;
}

TensorIndexRange::iterator TensorIndexRange::iterator::operator++(int) {
  const iterator tmp = *this;
  operator++();
  return tmp;
}

TensorIndexRange::iterator &TensorIndexRange::iterator::operator--() {
  if (!inner) {
    throw std::range_error("attempt to decrement iterator past beginning");
  }
  --inner;
  return *this;
}

TensorIndexRange::iterator TensorIndexRange::iterator::operator--(int) {
  const iterator tmp = *this;
  operator--();
  return tmp;
}

// TODO: Look into making this test at construction.
std::size_t TensorIndexRange::iterator::operator*() const {
  return iterable->size_coarse > 1 ? (inner * (iterable->size_finest - 1)) /
                                         (iterable->size_coarse - 1)
                                   : 0;
}

} // namespace mgard
