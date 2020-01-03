#include <cassert>

namespace mgard {

template <typename T>
EdgeFamily::EdgeFamily(const EdgeFamilyIterable<T> &iterable,
                       const moab::EntityHandle edge)
    : edge(edge) {
  const MeshLevel &mesh = iterable.mesh;
  const MeshLevel &MESH = iterable.MESH;

  std::array<moab::EntityHandle, 2>::iterator p = endpoints.begin();
  for (const moab::EntityHandle node : mesh.connectivity(edge)) {
    *p++ = node;
  }

  midpoint = MESH.entities[moab::MBVERTEX][mesh.ndof() + mesh.index(edge)];
}

template <typename T>
EdgeFamilyIterable<T>::EdgeFamilyIterable(const MeshLevel &mesh,
                                          const MeshLevel &MESH, const T begin,
                                          const T end)
    : mesh(mesh), MESH(MESH), begin_(begin), end_(end) {
  //`MESH` should be produced by refining `mesh`. Sanity checks here.
  const moab::Range &edges = mesh.entities[moab::MBEDGE];
  const moab::Range &elements = mesh.entities[mesh.element_type];
  const moab::Range &EDGES = MESH.entities[moab::MBEDGE];
  const moab::Range &ELEMENTS = MESH.entities[MESH.element_type];
  assert(MESH.ndof() == mesh.ndof() + edges.size());
  assert(EDGES.size() == 2 * edges.size() + 3 * elements.size());
  assert(mesh.element_type == MESH.element_type);
  assert(ELEMENTS.size() ==
         elements.size() * (1 << mesh.topological_dimension));
}

template <typename T>
EdgeFamilyIterable<T>::iterator::iterator(const EdgeFamilyIterable<T> &iterable,
                                          const T inner)
    : iterable(iterable), inner(inner) {}

// Could compare the `iterable`s here, but then we'd want to compare the meshes
// in the `iterable`s, and I don't want to define `MeshLevel::operator==`. At
// least as of this writing, edges are unique to the mesh, so it should be OK to
// just compare the iterators.
template <typename T>
bool EdgeFamilyIterable<T>::iterator::
operator==(const EdgeFamilyIterable<T>::iterator &other) const {
  return inner == other.inner;
}

template <typename T>
bool EdgeFamilyIterable<T>::iterator::
operator!=(const EdgeFamilyIterable<T>::iterator &other) const {
  return !(this->operator==(other));
}

template <typename T>
typename EdgeFamilyIterable<T>::iterator &EdgeFamilyIterable<T>::iterator::
operator++() {
  ++inner;
  return *this;
}

template <typename T>
typename EdgeFamilyIterable<T>::iterator EdgeFamilyIterable<T>::iterator::
operator++(int) {
  const EdgeFamilyIterable<T>::iterator tmp = *this;
  this->operator++();
  return tmp;
}

template <typename T>
EdgeFamily EdgeFamilyIterable<T>::iterator::operator*() const {
  return EdgeFamily(iterable, *inner);
}

template <typename T>
typename EdgeFamilyIterable<T>::iterator EdgeFamilyIterable<T>::begin() const {
  return EdgeFamilyIterable<T>::iterator(*this, begin_);
}

template <typename T>
typename EdgeFamilyIterable<T>::iterator EdgeFamilyIterable<T>::end() const {
  return EdgeFamilyIterable<T>::iterator(*this, end_);
}

} // namespace mgard
