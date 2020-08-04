#ifndef TENSORMESHHIERARCHY_HPP
#define TENSORMESHHIERARCHY_HPP
//!\file
//!\brief Increasing hierarchy of tensor meshes.

#include <cstddef>

#include <array>
#include <type_traits>
#include <vector>

#include "TensorMeshLevel.hpp"
#include "utilities.hpp"

namespace mgard {

// Forward declaration.
template <std::size_t N, typename Real> class TensorNodeRange;

//! Hierarchy of meshes produced by subsampling an initial mesh.
template <std::size_t N, typename Real> class TensorMeshHierarchy {
public:
  //! Constructor.
  //!
  //!\param mesh Initial, finest mesh to sit atop the hierarchy.
  TensorMeshHierarchy(const TensorMeshLevel<N, Real> &mesh);

  //! Constructor.
  //!
  //!\param mesh Initial, finest mesh to sit atop the hierarchy.
  //!\param coordinates Coordinates of the nodes in the finest mesh.
  TensorMeshHierarchy(const TensorMeshLevel<N, Real> &mesh,
                      const std::array<std::vector<Real>, N> &coordinates);

  // TODO: We may want to remove these. Using it refactoring.
  // TODO: Instead, we may want to remove the previous constructors. Check
  // whether `TensorMeshLevel` is needed anywhere.

  //! Constructor.
  //!
  //!\param shape Shape of the initial, finest mesh to sit atop the hiearachy.
  TensorMeshHierarchy(const std::array<std::size_t, N> &shape);

  //! Constructor.
  //!
  //!\param shape Shape of the initial, finest mesh to sit atop the hiearachy.
  //!\param coordinates Coordinates of the nodes in the finest mesh.
  TensorMeshHierarchy(const std::array<std::size_t, N> &shape,
                      const std::array<std::vector<Real>, N> &coordinates);

  //! Report the number of degrees of freedom in the finest TensorMeshLevel.
  std::size_t ndof() const;

  //! Report the number of degrees of freedom in a TensorMeshLevel.
  //!
  //!\param l Index of the TensorMeshLevel.
  std::size_t ndof(const std::size_t l) const;

  //! Calculate the stride between entries in a 1D slice on some level.
  //!
  //!\param l Index of the TensorMeshLevel.
  //!\param dimension Index of the dimension.
  std::size_t stride(const std::size_t l, const std::size_t dimension) const;

  // TODO: Temporary member function to be removed once indices rather than
  // index differences are used everywhere.
  std::size_t l(const std::size_t index_difference) const;

  //! Generate the indices (in a particular dimension) of a mesh level.
  //!
  //!\param l Mesh index.
  //!\param dimension Dimension index.
  std::vector<std::size_t> indices(const std::size_t l,
                                   const std::size_t dimension) const;

  //! Compute the offset of the value associated to a node.
  //!
  //! The offset is the distance in a contiguous dataset defined on the finest
  //! mesh in the hierarchy from the value associated to the zeroth node to
  //! the value associated to the given node.
  //!
  //!\param multiindex Multiindex of the node.
  std::size_t offset(const std::array<std::size_t, N> multiindex) const;

  //! Find the index of the level which introduced a node.
  //!
  //!\param multiindex Multiindex of the node.
  std::size_t date_of_birth(const std::array<std::size_t, N> multiindex) const;

  //! Access the value associated to a particular node.
  //!
  //!\param v Dataset defined on the hierarchy.
  //!\param multiindex Multiindex of the node.
  Real &at(Real *const v, const std::array<std::size_t, N> multiindex) const;

  //!\overload
  const Real &at(Real const *const u,
                 const std::array<std::size_t, N> multiindex) const;

  //! Access the nodes of a level in the hierarchy.
  //!
  //!\param l Index of the mesh level to be iterated over.
  TensorNodeRange<N, Real> nodes(const std::size_t l) const;

  //! Meshes composing the hierarchy, in 'increasing' order.
  std::vector<TensorMeshLevel<N, Real>> meshes;

  //! Coordinates of the nodes in the finest mesh.
  std::array<std::vector<Real>, N> coordinates;

  //! Index of finest TensorMeshLevel.
  std::size_t L;

  //! For each dimension, for each node in the finest level, the index of the
  //! level which introduced that node (its 'date of birth').
  std::array<std::vector<std::size_t>, N> dates_of_birth;

protected:
  //! Check that a mesh index is in bounds.
  //!
  //!\param l Mesh index.
  void check_mesh_index_bounds(const std::size_t l) const;

  //! Check that a pair of mesh indices are nondecreasing.
  //!
  //!\param l Smaller (nonlarger) mesh index.
  //!\param m Larger (nonsmaller) mesh index.
  void check_mesh_indices_nondecreasing(const std::size_t l,
                                        const std::size_t m) const;

  //! Check that a mesh index is nonzero.
  //!
  //!\param l Mesh index.
  void check_mesh_index_nonzero(const std::size_t l) const;

  //! Check that a dimension index is in bounds.
  //!
  //!\param dimension Dimension index.
  void check_dimension_index_bounds(const std::size_t dimension) const;
};

//! Equality comparison.
template <std::size_t N, typename Real>
bool operator==(const TensorMeshHierarchy<N, Real> &a,
                const TensorMeshHierarchy<N, Real> &b);

//! Inequality comparison.
template <std::size_t N, typename Real>
bool operator==(const TensorMeshHierarchy<N, Real> &a,
                const TensorMeshHierarchy<N, Real> &b);

//! Nodes of a particular level in a mesh hierarchy.
template <std::size_t N, typename Real> class TensorNodeRange {
public:
  //! Constructor.
  //!
  //!\param hierarchy Associated mesh hierarchy.
  //!\param l Index of the mesh level to be iterated over.
  TensorNodeRange(const TensorMeshHierarchy<N, Real> &hierarchy,
                  const std::size_t l);

  // Forward declaration.
  class iterator;

  //! Return an iterator to the beginning of the nodes.
  iterator begin() const;

  //! Return an iterator to the end of the nodes.
  iterator end() const;

  //! Equality comparison.
  bool operator==(const TensorNodeRange &other) const;

  //! Inequality comparison.
  bool operator!=(const TensorNodeRange &other) const;

  //! Associated mesh hierarchy.
  const TensorMeshHierarchy<N, Real> &hierarchy;

private:
  //! Index of the level being iterated over.
  //!
  //! This is only stored so we can avoid comparing `factors` and `multiindices`
  //! in the (in)equality comparison operators.
  const std::size_t l;

  // The indices whose Cartesian product will give the multiindices of the nodes
  // on the level being iterated over.
  const std::array<std::vector<std::size_t>, N> factors;

  //! Multiindices of the nodes on the level being iterated over.
  const CartesianProduct<std::size_t, N> multiindices;
};

//! A node (and auxiliary data) in a mesh in a mesh hierarchy.
template <std::size_t N, typename Real> struct TensorNode {
  //! Index of the mesh level which introduced the node.
  std::size_t l;

  //! Multiindex of the node.
  std::array<std::size_t, N> multiindex;

  //! Coordinates of the node.
  std::array<Real, N> coordinates;
};

//! Iterator over the nodes of a mesh in a mesh hierarchy.
template <std::size_t N, typename Real>
class TensorNodeRange<N, Real>::iterator {
public:
  using iterator_category = std::input_iterator_tag;
  using value_type = TensorNode<N, Real>;
  using difference_type = std::ptrdiff_t;
  using pointer = value_type *;
  using reference = value_type &;

  //! Constructor.
  //!
  //!\param iterable View of nodes to be iterated over.
  //!\param inner Underlying multiindex iterator.
  iterator(const TensorNodeRange &iterable,
           const typename CartesianProduct<std::size_t, N>::iterator &inner);

  //! Equality comparison.
  bool operator==(const iterator &other) const;

  //! Inequality comparison.
  bool operator!=(const iterator &other) const;

  //! Preincrement.
  iterator &operator++();

  //! Postincrement.
  iterator operator++(int);

  //! Dereference.
  value_type operator*() const;

  //! View of nodes being iterated over.
  const TensorNodeRange &iterable;

private:
  //! Underlying multiindex iterator.
  typename CartesianProduct<std::size_t, N>::iterator inner;
};

} // namespace mgard

#include "TensorMeshHierarchy.tpp"
#endif
