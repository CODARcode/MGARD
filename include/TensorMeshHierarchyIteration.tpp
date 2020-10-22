#include <stdexcept>

namespace mgard {

template <std::size_t N, typename Real>
TensorIndexRange::TensorIndexRange(
    const TensorMeshHierarchy<N, Real> &hierarchy, const std::size_t l,
    const std::size_t dimension)
    : size_finest(hierarchy.meshes.at(hierarchy.L).shape.at(dimension)),
      size_coarse(hierarchy.meshes.at(l).shape.at(dimension)) {
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
TensorNodeRange<N, Real>::TensorNodeRange(
    const TensorMeshHierarchy<N, Real> &hierarchy, const std::size_t l)
    : hierarchy(hierarchy), l(l), multiindices(make_factors(hierarchy, l)) {}

template <std::size_t N, typename Real>
bool TensorNodeRange<N, Real>::
operator==(const TensorNodeRange<N, Real> &other) const {
  return hierarchy == other.hierarchy && l == other.l;
}

template <std::size_t N, typename Real>
bool TensorNodeRange<N, Real>::
operator!=(const TensorNodeRange<N, Real> &other) const {
  return !operator==(other);
}

template <std::size_t N, typename Real>
typename TensorNodeRange<N, Real>::iterator
TensorNodeRange<N, Real>::begin() const {
  return iterator(*this, multiindices.begin());
}

template <std::size_t N, typename Real>
typename TensorNodeRange<N, Real>::iterator
TensorNodeRange<N, Real>::end() const {
  return iterator(*this, multiindices.end());
}

template <std::size_t N, typename Real>
TensorNodeRange<N, Real>::iterator::iterator(
    const TensorNodeRange<N, Real> &iterable,
    const typename CartesianProduct<TensorIndexRange, N>::iterator &inner)
    : iterable(&iterable), inner(inner) {}

template <std::size_t N, typename Real>
bool TensorNodeRange<N, Real>::iterator::
operator==(const TensorNodeRange<N, Real>::iterator &other) const {
  return (iterable == other.iterable || *iterable == *(other.iterable)) &&
         inner == other.inner;
}

template <std::size_t N, typename Real>
bool TensorNodeRange<N, Real>::iterator::
operator!=(const TensorNodeRange<N, Real>::iterator &other) const {
  return !operator==(other);
}

template <std::size_t N, typename Real>
typename TensorNodeRange<N, Real>::iterator &
TensorNodeRange<N, Real>::iterator::operator++() {
  ++inner;
  return *this;
}

template <std::size_t N, typename Real>
typename TensorNodeRange<N, Real>::iterator TensorNodeRange<N, Real>::iterator::
operator++(int) {
  const iterator tmp = *this;
  operator++();
  return tmp;
}

template <std::size_t N, typename Real>
TensorNode<N> TensorNodeRange<N, Real>::iterator::operator*() const {
  return TensorNode<N>(inner);
}

namespace {

template <std::size_t N, typename Real>
std::vector<TensorNodeRange<N, Real>>
make_ranges(const TensorMeshHierarchy<N, Real> &hierarchy,
            const std::size_t l) {
  std::vector<TensorNodeRange<N, Real>> ranges;
  ranges.reserve(l + 1);
  for (std::size_t ell = 0; ell <= l; ++ell) {
    ranges.push_back(TensorNodeRange<N, Real>(hierarchy, ell));
  }
  return ranges;
}

} // namespace

template <std::size_t N, typename Real>
TensorReservedNodeRange<N, Real>::TensorReservedNodeRange(
    const TensorMeshHierarchy<N, Real> &hierarchy, const std::size_t l)
    : hierarchy(hierarchy), l(l), ranges(make_ranges(hierarchy, l)) {}

template <std::size_t N, typename Real>
bool TensorReservedNodeRange<N, Real>::
operator==(const TensorReservedNodeRange<N, Real> &other) const {
  return hierarchy == other.hierarchy && l == other.l;
}

template <std::size_t N, typename Real>
bool TensorReservedNodeRange<N, Real>::
operator!=(const TensorReservedNodeRange<N, Real> &other) const {
  return !operator==(other);
}

template <std::size_t N, typename Real>
typename TensorReservedNodeRange<N, Real>::iterator
TensorReservedNodeRange<N, Real>::begin() const {
  std::vector<typename TensorNodeRange<N, Real>::iterator> inners;
  inners.reserve(l + 1);
  for (const TensorNodeRange<N, Real> &range : ranges) {
    inners.push_back(range.begin());
  }
  return iterator(*this, inners);
}

template <std::size_t N, typename Real>
typename TensorReservedNodeRange<N, Real>::iterator
TensorReservedNodeRange<N, Real>::end() const {
  std::vector<typename TensorNodeRange<N, Real>::iterator> inners;
  inners.reserve(l + 1);
  for (const TensorNodeRange<N, Real> &range : ranges) {
    inners.push_back(range.end());
  }
  return iterator(*this, inners);
}

template <std::size_t N, typename Real>
TensorReservedNodeRange<N, Real>::iterator::iterator(
    const TensorReservedNodeRange<N, Real> &iterable,
    const std::vector<typename TensorNodeRange<N, Real>::iterator> inners)
    : iterable(iterable), inners(inners), inner_finest(inners.back()) {}

template <std::size_t N, typename Real>
bool TensorReservedNodeRange<N, Real>::iterator::
operator==(const TensorReservedNodeRange<N, Real>::iterator &other) const {
  return (&iterable == &other.iterable || iterable == other.iterable) &&
         inner_finest == other.inner_finest;
}

template <std::size_t N, typename Real>
bool TensorReservedNodeRange<N, Real>::iterator::
operator!=(const TensorReservedNodeRange<N, Real>::iterator &other) const {
  return !operator==(other);
}

template <std::size_t N, typename Real>
typename TensorReservedNodeRange<N, Real>::iterator &
TensorReservedNodeRange<N, Real>::iterator::operator++() {
  ++inner_finest;
  return *this;
}

template <std::size_t N, typename Real>
typename TensorReservedNodeRange<N, Real>::iterator
TensorReservedNodeRange<N, Real>::iterator::operator++(int) {
  const iterator tmp = *this;
  operator++();
  return tmp;
}

template <std::size_t N, typename Real>
TensorNode<N> TensorReservedNodeRange<N, Real>::iterator::operator*() const {
  const std::array<std::size_t, N> multiindex = (*inner_finest).multiindex;
  // Find the iterator on the coarsest mesh containing this node (the mesh which
  // introduced this node).
  const std::size_t ell =
      iterable.ranges.back().hierarchy.date_of_birth(multiindex);
  typename TensorNodeRange<N, Real>::iterator &inner_coarsest = inners.at(ell);
  while (true) {
    const TensorNode<N> node = *inner_coarsest;
    if (node.multiindex == multiindex) {
      return node;
    } else {
      ++inner_coarsest;
    }
  }
}

template <std::size_t N, typename Real>
ShuffledTensorNodeRange<N, Real>::ShuffledTensorNodeRange(
    const TensorMeshHierarchy<N, Real> &hierarchy, const std::size_t l)
    : hierarchy(hierarchy), ranges(make_ranges(hierarchy, l)), l(l) {}

template <std::size_t N, typename Real>
bool ShuffledTensorNodeRange<N, Real>::
operator==(const ShuffledTensorNodeRange<N, Real> &other) const {
  return hierarchy == other.hierarchy && l == other.l;
}

template <std::size_t N, typename Real>
bool ShuffledTensorNodeRange<N, Real>::
operator!=(const ShuffledTensorNodeRange<N, Real> &other) const {
  return !operator==(other);
}

template <std::size_t N, typename Real>
typename ShuffledTensorNodeRange<N, Real>::iterator
ShuffledTensorNodeRange<N, Real>::begin() const {
  return iterator(*this, 0, ranges.front().begin());
}

template <std::size_t N, typename Real>
typename ShuffledTensorNodeRange<N, Real>::iterator
ShuffledTensorNodeRange<N, Real>::end() const {
  return iterator(*this, hierarchy.L, ranges.back().end());
}

template <std::size_t N, typename Real>
ShuffledTensorNodeRange<N, Real>::iterator::iterator(
    const ShuffledTensorNodeRange<N, Real> &iterable, const std::size_t ell,
    const typename TensorNodeRange<N, Real>::iterator inner)
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
    // Will likely want to save the end iterator and to make `iterable.l`
    // public. Keeping it simple for now.
    if (++inner == iterable.ranges.at(ell).end()) {
      if (ell + 1 == iterable.ranges.size()) {
        break;
      }
      ++ell;
      inner = iterable.ranges.at(ell).begin();
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
