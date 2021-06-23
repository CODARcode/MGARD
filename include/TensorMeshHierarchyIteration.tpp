#include <stdexcept>

namespace mgard {

template <std::size_t N, typename Real>
TensorIndexRange::TensorIndexRange(
    const TensorMeshHierarchy<N, Real> &hierarchy, const std::size_t l,
    const std::size_t dimension)
    : size_finest(hierarchy.shapes.at(hierarchy.L).at(dimension)),
      size_coarse(hierarchy.shapes.at(l).at(dimension)) {
  if (size_coarse > size_finest) {
    throw std::invalid_argument(
        "coarse size cannot be larger than finest size");
  }
  if (!(size_finest && size_coarse)) {
    throw std::invalid_argument("sizes must be nonzero");
  }
}

template <std::size_t N>
TensorNode<N>::TensorNode(
    const typename CartesianProduct<TensorIndexRange, N>::iterator inner)
    : multiindex(*inner), inner(inner) {}

template <std::size_t N>
TensorNode<N> TensorNode<N>::predecessor(const std::size_t i) const {
  check_dimension_index_bounds<N>(i);
  return TensorNode(inner.predecessor(i));
}

template <std::size_t N>
TensorNode<N> TensorNode<N>::successor(const std::size_t i) const {
  check_dimension_index_bounds<N>(i);
  return TensorNode(inner.successor(i));
}

namespace {

template <std::size_t N, typename Real>
std::array<TensorIndexRange, N>
make_factors(const TensorMeshHierarchy<N, Real> &hierarchy,
             const std::size_t l) {
  std::array<TensorIndexRange, N> factors;
  for (std::size_t i = 0; i < N; ++i) {
    factors.at(i) = hierarchy.indices(l, i);
  }
  return factors;
}

} // namespace

template <std::size_t N, typename Real>
UnshuffledTensorNodeRange<N, Real>::UnshuffledTensorNodeRange(
    const TensorMeshHierarchy<N, Real> &hierarchy, const std::size_t l)
    : hierarchy(hierarchy), l(l), multiindices(make_factors(hierarchy, l)) {}

template <std::size_t N, typename Real>
bool operator==(const UnshuffledTensorNodeRange<N, Real> &a,
                const UnshuffledTensorNodeRange<N, Real> &b) {
  return a.l == b.l && a.hierarchy == b.hierarchy;
}

template <std::size_t N, typename Real>
bool operator!=(const UnshuffledTensorNodeRange<N, Real> &a,
                const UnshuffledTensorNodeRange<N, Real> &b) {
  return !operator==(a, b);
}

template <std::size_t N, typename Real>
typename UnshuffledTensorNodeRange<N, Real>::iterator
UnshuffledTensorNodeRange<N, Real>::begin() const {
  return iterator(*this, multiindices.begin());
}

template <std::size_t N, typename Real>
typename UnshuffledTensorNodeRange<N, Real>::iterator
UnshuffledTensorNodeRange<N, Real>::end() const {
  return iterator(*this, multiindices.end());
}

template <std::size_t N, typename Real>
UnshuffledTensorNodeRange<N, Real>::iterator::iterator(
    const UnshuffledTensorNodeRange<N, Real> &iterable,
    const typename CartesianProduct<TensorIndexRange, N>::iterator &inner)
    : iterable(&iterable), inner(inner) {}

template <std::size_t N, typename Real>
bool UnshuffledTensorNodeRange<N, Real>::iterator::
operator==(const UnshuffledTensorNodeRange<N, Real>::iterator &other) const {
  return (iterable == other.iterable || *iterable == *(other.iterable)) &&
         inner == other.inner;
}

template <std::size_t N, typename Real>
bool UnshuffledTensorNodeRange<N, Real>::iterator::
operator!=(const UnshuffledTensorNodeRange<N, Real>::iterator &other) const {
  return !operator==(other);
}

template <std::size_t N, typename Real>
typename UnshuffledTensorNodeRange<N, Real>::iterator &
UnshuffledTensorNodeRange<N, Real>::iterator::operator++() {
  ++inner;
  return *this;
}

template <std::size_t N, typename Real>
typename UnshuffledTensorNodeRange<N, Real>::iterator
UnshuffledTensorNodeRange<N, Real>::iterator::operator++(int) {
  const iterator tmp = *this;
  operator++();
  return tmp;
}

template <std::size_t N, typename Real>
TensorNode<N> UnshuffledTensorNodeRange<N, Real>::iterator::operator*() const {
  return TensorNode<N>(inner);
}

namespace {

template <std::size_t N, typename Real>
std::vector<UnshuffledTensorNodeRange<N, Real>>
make_ranges(const TensorMeshHierarchy<N, Real> &hierarchy,
            const std::size_t l) {
  std::vector<UnshuffledTensorNodeRange<N, Real>> ranges;
  ranges.reserve(l + 1);
  for (std::size_t ell = 0; ell <= l; ++ell) {
    ranges.push_back(UnshuffledTensorNodeRange<N, Real>(hierarchy, ell));
  }
  return ranges;
}

template <std::size_t N, typename Real>
std::vector<
    std::array<typename UnshuffledTensorNodeRange<N, Real>::iterator, 2>>
make_range_endpoints(
    const std::vector<UnshuffledTensorNodeRange<N, Real>> &ranges,
    const std::size_t l) {
  std::vector<
      std::array<typename UnshuffledTensorNodeRange<N, Real>::iterator, 2>>
      range_endpoints;
  range_endpoints.reserve(l + 1);
  for (const UnshuffledTensorNodeRange<N, Real> &range : ranges) {
    range_endpoints.push_back({range.begin(), range.end()});
  }
  return range_endpoints;
}

} // namespace

template <std::size_t N, typename Real>
ShuffledTensorNodeRange<N, Real>::ShuffledTensorNodeRange(
    const TensorMeshHierarchy<N, Real> &hierarchy, const std::size_t l)
    : hierarchy(hierarchy), ranges(make_ranges(hierarchy, l)), l(l),
      range_endpoints(make_range_endpoints(ranges, l)) {}

template <std::size_t N, typename Real>
bool operator==(const ShuffledTensorNodeRange<N, Real> &a,
                const ShuffledTensorNodeRange<N, Real> &b) {
  return a.l == b.l && a.hierarchy == b.hierarchy;
}

template <std::size_t N, typename Real>
bool operator!=(const ShuffledTensorNodeRange<N, Real> &a,
                const ShuffledTensorNodeRange<N, Real> &b) {
  return !operator==(a, b);
}

template <std::size_t N, typename Real>
typename ShuffledTensorNodeRange<N, Real>::iterator
ShuffledTensorNodeRange<N, Real>::begin() const {
  return iterator(*this, 0, range_endpoints.front().front(), 0);
}

template <std::size_t N, typename Real>
typename ShuffledTensorNodeRange<N, Real>::iterator
ShuffledTensorNodeRange<N, Real>::end() const {
  return iterator(*this, l, range_endpoints.back().back(), hierarchy.ndof(l));
}

template <std::size_t N, typename Real>
ShuffledTensorNodeRange<N, Real>::iterator::iterator(
    const ShuffledTensorNodeRange<N, Real> &iterable, const std::size_t ell,
    const typename UnshuffledTensorNodeRange<N, Real>::iterator &inner)
    : iterable(iterable), ell(ell), inner(inner) {}

template <std::size_t N, typename Real>
bool ShuffledTensorNodeRange<N, Real>::iterator::
operator==(const ShuffledTensorNodeRange<N, Real>::iterator &other) const {
  return (&iterable == &other.iterable || iterable == other.iterable) &&
         inner == other.inner;
}

template <std::size_t N, typename Real>
bool ShuffledTensorNodeRange<N, Real>::iterator::
operator!=(const ShuffledTensorNodeRange<N, Real>::iterator &other) const {
  return !operator==(other);
}

template <std::size_t N, typename Real>
typename ShuffledTensorNodeRange<N, Real>::iterator &
ShuffledTensorNodeRange<N, Real>::iterator::operator++() {
  while (true) {
    if (++inner == iterable.range_endpoints.at(ell).back()) {
      // Will likely want to make `iterable.l` public. Keeping it simple for
      // now.
      if (ell + 1 == iterable.ranges.size()) {
        break;
      }
      ++ell;
      inner = iterable.range_endpoints.at(ell).front();
      // `inner` now dereferences to a node in the coarsest mesh, but it feels
      // like skipping the next check would be asking for trouble.
    }
    if (iterable.hierarchy.date_of_birth((*inner).multiindex) == ell) {
      break;
    }
  }
  return *this;
}

template <std::size_t N, typename Real>
typename ShuffledTensorNodeRange<N, Real>::iterator
ShuffledTensorNodeRange<N, Real>::iterator::operator++(int) {
  const ShuffledTensorNodeRange<N, Real>::iterator tmp = *this;
  operator++();
  return tmp;
}

template <std::size_t N, typename Real>
TensorNode<N> ShuffledTensorNodeRange<N, Real>::iterator::operator*() const {
  return *inner;
}

} // namespace mgard
