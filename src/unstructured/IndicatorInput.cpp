#include "unstructured/IndicatorInput.hpp"

namespace mgard {

IndicatorInputRange::IndicatorInputRange(const MeshHierarchy &hierarchy)
    : hierarchy(hierarchy), indexed_meshes(hierarchy.meshes),
      indexed_meshes_end(indexed_meshes.end()) {}

typename IndicatorInputRange::iterator IndicatorInputRange::begin() const {
  return iterator(*this, indexed_meshes.begin());
}

typename IndicatorInputRange::iterator IndicatorInputRange::end() const {
  return iterator(*this, indexed_meshes_end);
}

bool operator==(const IndicatorInputRange &a, const IndicatorInputRange &b) {
  return a.hierarchy == b.hierarchy;
}

bool operator!=(const IndicatorInputRange &a, const IndicatorInputRange &b) {
  return !operator==(a, b);
}

IndicatorInputRange::iterator::iterator(
    const IndicatorInputRange &iterable,
    const Enumeration<std::vector<MeshLevel>::const_iterator>::iterator
        inner_mesh)
    : iterable(iterable), inner_mesh(inner_mesh) {
  reset_node_iterator_pair();
}

bool IndicatorInputRange::iterator::operator==(
    const IndicatorInputRange::iterator &other) const {
  return (&iterable == &other.iterable || iterable == other.iterable) &&
         inner_mesh == other.inner_mesh && inner_node == other.inner_node;
}

bool IndicatorInputRange::iterator::operator!=(
    const IndicatorInputRange::iterator &other) const {
  return !operator==(other);
}

typename IndicatorInputRange::iterator &
IndicatorInputRange::iterator::operator++() {
  // First entry is the current position. Second entry is the end.
  std::array<moab::Range::const_iterator, 2> &iterators = inner_node.value();
  if (++iterators[0] == iterators[1]) {
    // Must increment the mesh iterator before resetting. Otherwise, we'll get
    // sent back to the beginning of the *current* node–coefficient range.
    ++inner_mesh;
    reset_node_iterator_pair();
  }
  return *this;
}

typename IndicatorInputRange::iterator
IndicatorInputRange::iterator::operator++(int) {
  const IndicatorInputRange::iterator tmp = *this;
  operator++();
  return tmp;
}

IndicatorInput IndicatorInputRange::iterator::operator*() const {
  // '&' to prevent a copy of the mesh being made. This is important, since the
  // returned `IndicatorInput` will contain a reference to `mesh`.
  const auto &[l, mesh] = *inner_mesh;
  const moab::EntityHandle node = *inner_node.value()[0];
  return {.l = l, .mesh = mesh, .node = node};
}

void IndicatorInputRange::iterator::reset_node_iterator_pair() {
  // We want to reset even if we're later going to reassign. This prevents us
  // from assigning iterators to the new node–coefficient range to lvalues
  // containing iterators to the old node–coefficient range (which isn't allowed
  // because the range is stored as a reference in the iterable).
  inner_node.reset();
  if (inner_mesh == iterable.indexed_meshes_end) {
    return;
  }
  const std::size_t l = (*inner_mesh).index;
  const RangeSlice<moab::Range::const_iterator> slice =
      iterable.hierarchy.new_nodes(l);
  inner_node = {slice.begin(), slice.end()};
}

} // namespace mgard
