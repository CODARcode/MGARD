#ifndef TENSORMESHHIERARCHY_HPP
#define TENSORMESHHIERARCHY_HPP
//!\file
//!\brief Increasing hierarchy of tensor meshes.

#include <cstddef>

#include <array>
#include <iterator>
#include <type_traits>
#include <vector>

#include "TensorMeshLevel.hpp"
#include "utilities.hpp"

namespace mgard {

// Forward declarations.
template <std::size_t N, typename Real> class TensorNodeRange;

class TensorIndexRange;

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

  //! Calculate a mesh index from an index difference.
  //!
  //!\deprecated Temporary member function to be removed once indices rather
  //! than index differences are used everywhere.
  //!
  //!\param index_difference Difference between the index of the finest mesh and
  //! the index of the mesh in question.
  std::size_t l(const std::size_t index_difference) const;

  //! Generate the indices (in a particular dimension) of a mesh level.
  //!
  //!\param l Mesh index.
  //!\param dimension Dimension index.
  TensorIndexRange indices(const std::size_t l,
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
bool operator!=(const TensorMeshHierarchy<N, Real> &a,
                const TensorMeshHierarchy<N, Real> &b);

//! Indices in a particular dimension of nodes of a particular level in a mesh
//! hierarchy.
class TensorIndexRange {
public:
  //! Constructor.
  //!
  //! We define this constructor for use in `singleton` and so that objects with
  //! data members of type `TensorIndexRange` (for example,
  //! `ConstituentRestriction`) may be default constructed.
  TensorIndexRange() = default;

  //! Constructor.
  //
  //!\param hierarchy Associated mesh hierarchy.
  //!\param l Mesh index.
  //!\param dimension Dimension index.
  template <std::size_t N, typename Real>
  TensorIndexRange(const TensorMeshHierarchy<N, Real> &hierarchy,
                   const std::size_t l, const std::size_t dimension);

  //! Factory member function.
  //!
  //! We define this function so that we can create ranges which yield the
  //! single value `0` when iterated over. This is convenient for
  //! `TensorLinearOperator`.
  static TensorIndexRange singleton();

  //! Return the size of the range.
  std::size_t size() const;

  // Forward declaration.
  class iterator;

  //! Return an iterator to the beginning of the indices.
  iterator begin() const;

  //! Return an iterator to the end of the indices.
  iterator end() const;

  // `size_finest` and `size_coarse` aren't `const` so that the defaulted copy
  // assignment operator won't be deleted.

  //! Size in the particular dimension of the finest mesh in the hierarchy.
  std::size_t size_finest;

  //! Size in the particular dimension of the mesh in question.
  std::size_t size_coarse;
};

//! Equality comparison.
bool operator==(const TensorIndexRange &a, const TensorIndexRange &b);

//! Inequality comparison.
bool operator!=(const TensorIndexRange &a, const TensorIndexRange &b);

//! Iterator over the indices in a particular dimension of nodes of a particular
//! level in a mesh hierarchy.
//!
//! This iterator does *not* satisfy all the requirements of a forward iterator.
//! See <https://en.cppreference.com/w/cpp/named_req/ForwardIterator>. Like a
//! forward iterator, though, a `TensorIndexRange::iterator` can be used to
//! iterate over a `TensorIndexRange` repeatedly, with the same values obtained
//! each time.
class TensorIndexRange::iterator {
public:
  // See note above.
  //! Category of the iterator.
  using iterator_category = std::input_iterator_tag;
  //! Type iterated over.
  using value_type = std::size_t;
  //! Type for distance between iterators.
  using difference_type = std::ptrdiff_t;
  //! Pointer to `value_type`.
  using pointer = value_type *;
  //! Type returned by the dereference operator.
  using reference = value_type;

  //! Constructor.
  //!
  //! This constructor is provided so that arrays of iterators may be formed.
  //! A default-constructed iterator must be assigned to before being used.
  iterator() = default;

  //! Constructor.
  //!
  //!\param iterable View of indices to be iterated over.
  //!\param inner Position in the index range.
  iterator(const TensorIndexRange &iterable, const std::size_t inner);

  //! Equality comparison.
  bool operator==(const iterator &other) const;

  //! Inequality comparison.
  bool operator!=(const iterator &other) const;

  //! Preincrement.
  iterator &operator++();

  //! Postincrement.
  iterator operator++(int);

  //! Predecrement.
  iterator &operator--();

  //! Postdecrement.
  iterator operator--(int);

  //! Dereference.
  reference operator*() const;

  //! View of indices being iterated over.
  TensorIndexRange const *iterable;

private:
  //! Position in the index range.
  std::size_t inner;
};

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
  //! This is only stored so we can avoid comparing `multiindices` in the
  //! (in)equality comparison operators.
  const std::size_t l;

  //! Multiindices of the nodes on the level being iterated over.
  const CartesianProduct<TensorIndexRange, N> multiindices;
};

//! A node (and auxiliary data) in a mesh in a mesh hierarchy.
template <std::size_t N, typename Real> struct TensorNode {
  //! Multiindex of the node.
  std::array<std::size_t, N> multiindex;
};

//! Iterator over the nodes of a mesh in a mesh hierarchy.
template <std::size_t N, typename Real>
class TensorNodeRange<N, Real>::iterator {
public:
  //! Category of the iterator.
  using iterator_category = std::input_iterator_tag;
  //! Type iterated over.
  using value_type = TensorNode<N, Real>;
  //! Type for distance between iterators.
  using difference_type = std::ptrdiff_t;
  //! Pointer to `value_type`.
  using pointer = value_type *;
  //! Type returned by the dereference operator.
  using reference = value_type;

  //! Constructor.
  //!
  //!\param iterable View of nodes to be iterated over.
  //!\param inner Underlying multiindex iterator.
  iterator(
      const TensorNodeRange &iterable,
      const typename CartesianProduct<TensorIndexRange, N>::iterator &inner);

  //! Equality comparison.
  bool operator==(const iterator &other) const;

  //! Inequality comparison.
  bool operator!=(const iterator &other) const;

  //! Preincrement.
  iterator &operator++();

  //! Postincrement.
  iterator operator++(int);

  //! Dereference.
  reference operator*() const;

  //! View of nodes being iterated over.
  const TensorNodeRange &iterable;

private:
  //! Underlying multiindex iterator.
  typename CartesianProduct<TensorIndexRange, N>::iterator inner;
};

} // namespace mgard

#include "TensorMeshHierarchy.tpp"
#endif
