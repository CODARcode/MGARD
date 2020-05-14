#include <algorithm>
#include <limits>
#include <stdexcept>

namespace mgard {

template <std::size_t N>
Dimensions2kPlus1<N>::Dimensions2kPlus1(
    const std::array<std::size_t, N> input_) {
  nlevel = std::numeric_limits<std::size_t>::max();
  bool nlevel_never_set = true;
  for (std::size_t i = 0; i < N; ++i) {
    const std::size_t size = input.at(i) = input_.at(i);
    if (size == 0) {
      throw std::domain_error(
          "dataset must have size larger than 0 in every dimension");
    } else if (size == 1) {
      rnded.at(i) = size;
    } else {
      const std::size_t exp = nlevel_from_size(size);
      rnded.at(i) = size_from_nlevel(exp);
      nlevel = std::min(nlevel, exp);
      nlevel_never_set = false;
    }
  }
  if (nlevel_never_set) {
    throw std::domain_error(
        "dataset must have size larger than 1 in some dimension");
  }
}

template <std::size_t N> bool Dimensions2kPlus1<N>::is_2kplus1() const {
  for (const std::size_t n : input) {
    if (!(n == 1 || n == size_from_nlevel(nlevel_from_size(n)))) {
      return false;
    }
  }
  return true;
}

template <std::size_t N>
template <typename Real>
LevelValues<N, Real>
Dimensions2kPlus1<N>::on_nodes(Real *const coefficients,
                               const std::size_t index_difference) const {
  return LevelValues<N, Real>(*this, coefficients, index_difference);
}

template <std::size_t N>
bool operator==(const Dimensions2kPlus1<N> &a, const Dimensions2kPlus1<N> &b) {
  return a.input == b.input;
}

template <std::size_t N>
bool operator!=(const Dimensions2kPlus1<N> &a, const Dimensions2kPlus1<N> &b) {
  return !operator==(a, b);
}

template <std::size_t N, typename Real>
LevelValues<N, Real>::LevelValues(const Dimensions2kPlus1<N> &dimensions,
                                  Real *const coefficients,
                                  const std::size_t index_difference)
    : dimensions(dimensions), coefficients(coefficients),
      stride(stride_from_index_difference(index_difference)),
      rectangle(dimensions.input) {}

template <std::size_t N, typename Real>
bool operator==(const LevelValues<N, Real> &a, const LevelValues<N, Real> &b) {
  return a.dimensions == b.dimensions && a.coefficients == b.coefficients &&
         a.stride == b.stride;
}

template <std::size_t N, typename Real>
bool operator!=(const LevelValues<N, Real> &a, const LevelValues<N, Real> &b) {
  return !operator==(a, b);
}

template <std::size_t N, typename Real>
typename LevelValues<N, Real>::iterator LevelValues<N, Real>::begin() const {
  return iterator(*this, rectangle.begin(stride));
}

template <std::size_t N, typename Real>
typename LevelValues<N, Real>::iterator LevelValues<N, Real>::end() const {
  return iterator(*this, rectangle.end(stride));
}

template <std::size_t N, typename Real>
LevelValues<N, Real>::iterator::iterator(
    const LevelValues<N, Real> &iterable,
    const typename MultiindexRectangle<N>::iterator &inner)
    : iterable(iterable), inner(inner) {
  if (inner.rectangle != iterable.rectangle) {
    throw std::domain_error(
        "index iterable cannot be associated to a different multiindex set");
  }
  if (inner.stride != iterable.stride) {
    throw std::domain_error("index iterable cannot have a different stride");
  }
}

template <std::size_t N, typename Real>
bool LevelValues<N, Real>::iterator::
operator==(const LevelValues<N, Real>::iterator &other) const {
  return iterable == other.iterable && inner == other.inner;
}

template <std::size_t N, typename Real>
bool LevelValues<N, Real>::iterator::
operator!=(const LevelValues<N, Real>::iterator &other) const {
  return !operator==(other);
}

template <std::size_t N, typename Real>
typename LevelValues<N, Real>::iterator &LevelValues<N, Real>::iterator::
operator++() {
  ++inner;
  return *this;
}

template <std::size_t N, typename Real>
typename LevelValues<N, Real>::iterator LevelValues<N, Real>::iterator::
operator++(int) {
  const iterator tmp = *this;
  operator++();
  return tmp;
}

template <std::size_t N, typename Real>
Real &LevelValues<N, Real>::iterator::operator*() const {
  const std::array<std::size_t, N> indices = *inner;
  static_assert(N, "`N` must be nonzero to dereference");
  std::size_t index = indices.at(0);
  for (std::size_t i = 1; i < N; ++i) {
    index *= iterable.dimensions.input.at(i);
    index += indices.at(i);
  }
  return iterable.coefficients[index];
}

} // namespace mgard
