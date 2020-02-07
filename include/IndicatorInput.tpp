#include <type_traits>

namespace mgard {

template <typename Real>
IndicatorInputRange<Real>::IndicatorInputRange(
    const MeshHierarchy &hierarchy, const MultilevelCoefficients<Real> u)
    : hierarchy(hierarchy), u(u), indexed_meshes(hierarchy.meshes),
      indexed_meshes_end(indexed_meshes.end()) {}

template <typename Real>
typename IndicatorInputRange<Real>::iterator
IndicatorInputRange<Real>::begin() const {
  return iterator(*this, indexed_meshes.begin());
}

template <typename Real>
typename IndicatorInputRange<Real>::iterator
IndicatorInputRange<Real>::end() const {
  return iterator(*this, indexed_meshes_end);
}

template <typename Real>
bool operator==(const IndicatorInputRange<Real> &a,
                const IndicatorInputRange<Real> &b) {
  return a.hierarchy == b.hierarchy && a.u == b.u;
}

template <typename Real>
bool operator!=(const IndicatorInputRange<Real> &a,
                const IndicatorInputRange<Real> &b) {
  return !operator==(a, b);
}

template <typename Real>
IndicatorInputRange<Real>::iterator::iterator(
    const IndicatorInputRange<Real> &iterable,
    const Enumeration<std::vector<MeshLevel>::const_iterator>::iterator
        inner_mesh)
    : iterable(iterable), inner_mesh(inner_mesh) {
  reset_nc_iterator_pair();
}

template <typename Real>
bool IndicatorInputRange<Real>::iterator::
operator==(const IndicatorInputRange<Real>::iterator &other) const {
  return (iterable == other.iterable && inner_mesh == other.inner_mesh &&
          inner_node == other.inner_node);
}

template <typename Real>
bool IndicatorInputRange<Real>::iterator::
operator!=(const IndicatorInputRange<Real>::iterator &other) const {
  return !operator==(other);
}

template <typename Real>
typename IndicatorInputRange<Real>::iterator &
IndicatorInputRange<Real>::iterator::operator++() {
  // First entry is the current position. Second entry is the end.
  std::array<NCIterator, 2> &iterators = inner_node.value();
  if (++iterators[0] == iterators[1]) {
    // Must increment the mesh iterator before resetting. Otherwise, we'll get
    // sent back to the beginning of the *current* node–coefficient range.
    ++inner_mesh;
    reset_nc_iterator_pair();
  }
  return *this;
}

template <typename Real>
typename IndicatorInputRange<Real>::iterator
IndicatorInputRange<Real>::iterator::operator++(int) {
  const IndicatorInputRange<Real>::iterator tmp = *this;
  operator++();
  return tmp;
}

template <typename Real>
IndicatorInput<Real> IndicatorInputRange<Real>::iterator::operator*() const {
  // '&' to prevent a copy of the mesh being made. This is important, since the
  // returned `IndicatorInput<Real>` will contain a reference to `mesh`.
  const auto &[l, mesh] = *inner_mesh;
  const auto [node, coefficient] = *inner_node.value()[0];
  return {.l = l, .mesh = mesh, .node = node, .coefficient = coefficient};
}

template <typename Real>
void IndicatorInputRange<Real>::iterator::reset_nc_iterator_pair() {
  // We want to reset even if we're later going to reassign. This prevents us
  // from assigning iterators to the new node–coefficient range to lvalues
  // containing iterators to the old node–coefficient range (which isn't allowed
  // because the range is stored as a reference in the iterable).
  inner_node.reset();
  if (inner_mesh == iterable.indexed_meshes_end) {
    return;
  }
  const std::size_t l = (*inner_mesh).index;
  const SituatedCoefficientRange<Real> scr(iterable.hierarchy, iterable.u, l);
  inner_node = {scr.begin(), scr.end()};
}

} // namespace mgard
