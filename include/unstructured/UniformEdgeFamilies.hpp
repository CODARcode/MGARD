#ifndef UNIFORMEDGEFAMILIES_HPP
#define UNIFORMEDGEFAMILIES_HPP
//!\file
//!\brief Edges, their endpoints, and their midpoints associated with uniform
//! mesh hierarchies.

#include <cstddef>

#include <array>
#include <iterator>

#include "moab/EntityHandle.hpp"

#include "unstructured/MeshLevel.hpp"

namespace mgard {

// Forward declaration.
template <typename T> class EdgeFamilyIterable;

//! An edge in a coarse mesh, its endpoints in the coarse mesh, and its midpoint
//! in the fine mesh.
struct EdgeFamily {
  //! Constructor.
  //!
  //!\param iterable Group of edges to which this family is associated.
  //!\param edge Edge in the coarse mesh of the iterable.
  template <typename T>
  EdgeFamily(const EdgeFamilyIterable<T> &iterable,
             const moab::EntityHandle edge);

  //! Edge being refined.
  moab::EntityHandle edge;

  //! Endpoints (in the coarse mesh) of the edge.
  std::array<moab::EntityHandle, 2> endpoints;

  //! Midpoint (in the fine mesh) of the edge.
  moab::EntityHandle midpoint;
};

//! Object allowing a user to iterate over a group of edges. `T` should be an
//! iterator that dereferences to `moab::EntityHandle` (an edge).
template <typename T> class EdgeFamilyIterable {
public:
  //! Constructor.
  //!
  //!\param mesh Mesh containing the edges in question.
  //!\param MESH Mesh produced by refining the first mesh.
  //!\param begin Beginning of edge range.
  //!\param end End of edge range.
  EdgeFamilyIterable(const MeshLevel &mesh, const MeshLevel &MESH,
                     const T begin, const T end);

  //! Iterator over a group of edges.
  class iterator {
  public:
    //! Category of the iterator.
    using iterator_category = std::input_iterator_tag;
    //! Type iterated over.
    using value_type = EdgeFamily;
    //! Type for distance between iterators.
    using difference_type = std::ptrdiff_t;
    //! Pointer to `value_type`.
    using pointer = value_type *;
    //! Type returned by the dereference operator.
    using reference = value_type;

    //! Constructor.
    //!
    //!\param iterable Group of edges to which this iterator is
    //! associated.
    //!\param inner Iterator that dereferences to edges.
    iterator(const EdgeFamilyIterable &iterable, const T inner);

    //! Equality comparison.
    //!
    //!\param other Iterator to compare with.
    bool operator==(const iterator &other) const;

    //! Inequality comparison.
    //!
    //!\param other Iterator to compare with.
    bool operator!=(const iterator &other) const;

    //! Preincrement.
    iterator &operator++();

    //! Postincrement.
    iterator operator++(int);

    //! Dereference.
    reference operator*() const;

  private:
    const EdgeFamilyIterable &iterable;
    T inner;
  };

  //! Return an iterator to the beginning of the iterable.
  iterator begin() const;

  //! Return an iterator to the end of the iterable.
  iterator end() const;

  //! Mesh containing the edges in question.
  const MeshLevel &mesh;

  //! Mesh produced by refining the first mesh.
  const MeshLevel &MESH;

private:
  //! Beginning of edge range.
  const T begin_;

  //! End of edge range.
  const T end_;
};

} // namespace mgard

#include "unstructured/UniformEdgeFamilies.tpp"
#endif
