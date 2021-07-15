#ifndef TENSORMESHHIERARCHYITERATION_HPP
#define TENSORMESHHIERARCHYITERATION_HPP
//!\file
//!\brief Iteration over `TensorMeshHierarchy`s.

#include <cstddef>

#include <array>
#include <vector>

#include "utilities.hpp"

namespace mgard {

//! Forward declaration.
template <std::size_t N, typename Real> class TensorMeshHierarchy;

//! Indices in a particular dimension of nodes of a particular level in a mesh
//! hierarchy.
using TensorIndexRange = RangeSlice<std::size_t const *>;

//! Node in a tensor product mesh.
template <std::size_t N> class TensorNode {
public:
  //! Constructor.
  //!
  //!\param inner Underlying multiindex iterator.
  explicit TensorNode(
      const typename CartesianProduct<TensorIndexRange, N>::iterator inner);

  //! Multiindex of the node.
  std::array<std::size_t, N> multiindex;

  //! Return the node to the left in a given dimension *in the mesh currently
  //! being iterated over* (determined by `inner`).
  //!
  //! If this node is at the lefthand boundary of the domain, this node will
  //! be returned.
  //!
  //!\param i Index of the dimension.
  TensorNode predecessor(const std::size_t i) const;

  //! Return the node to the right in a given dimension *in the mesh currently
  //! being iterated over* (determined by `inner`).
  //!
  //! If this node is at the righthand boundary of the domain, this node will
  //! be returned.
  //!
  //!\param i Index of the dimension.
  TensorNode successor(const std::size_t i) const;

private:
  //! Underlying multiindex iterator.
  const typename CartesianProduct<TensorIndexRange, N>::iterator inner;
};

//! Nodes of a particular level in a mesh hierarchy.
//!
//! This range yields the nodes ordered by physical location (`{0, 0, 0}`,
//! `{0, 0, 1}`, `{0, 0, 2}`, etc.).
template <std::size_t N, typename Real> class UnshuffledTensorNodeRange {
public:
  //! Constructor.
  //!
  //!\param hierarchy Associated mesh hierarchy.
  //!\param l Index of the mesh level to be iterated over.
  UnshuffledTensorNodeRange(const TensorMeshHierarchy<N, Real> &hierarchy,
                            const std::size_t l);

  // Forward declaration.
  class iterator;

  //! Return an iterator to the beginning of the nodes.
  iterator begin() const;

  //! Return an iterator to the end of the nodes.
  iterator end() const;

  //! Associated mesh hierarchy.
  const TensorMeshHierarchy<N, Real> &hierarchy;

  //! Index of the level being iterated over.
  const std::size_t l;

private:
  //! Multiindices of the nodes on the level being iterated over.
  const CartesianProduct<TensorIndexRange, N> multiindices;
};

//! Equality comparison.
template <std::size_t N, typename Real>
bool operator==(const UnshuffledTensorNodeRange<N, Real> &a,
                UnshuffledTensorNodeRange<N, Real> &b);

//! Inequality comparison.
template <std::size_t N, typename Real>
bool operator!=(const UnshuffledTensorNodeRange<N, Real> &a,
                UnshuffledTensorNodeRange<N, Real> &b);

//! Iterator over the nodes of a mesh in a mesh hierarchy.
template <std::size_t N, typename Real>
class UnshuffledTensorNodeRange<N, Real>::iterator {
public:
  //! Category of the iterator.
  using iterator_category = std::input_iterator_tag;
  //! Type iterated over.
  using value_type = TensorNode<N>;
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
  //!\param index Position in the unshuffled range.
  iterator(
      const UnshuffledTensorNodeRange &iterable,
      const typename CartesianProduct<TensorIndexRange, N>::iterator &inner,
      const std::size_t index);

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
  //!
  //! A pointer rather than a reference so we can assign.
  UnshuffledTensorNodeRange const *iterable;

private:
  //! Underlying multiindex iterator.
  typename CartesianProduct<TensorIndexRange, N>::iterator inner;

  //! Position in the `UnshuffledTensorNodeRange`.
  //!
  //! This is only stored so we can avoid comparing `inner` in the
  //! (in)equality comparison operators.
  std::size_t index;
};

//! Nodes of a particular level in a mesh hierarchy.
//!
//! This range yields the nodes ordered by level, matching the coefficient
//! ordering produced by `shuffle`.
template <std::size_t N, typename Real> class ShuffledTensorNodeRange {
public:
  //! Constructor.
  //!
  //!\param hierarchy Associated mesh hierarchy.
  //!\param l Index of the mesh level to be iterated over.
  ShuffledTensorNodeRange(const TensorMeshHierarchy<N, Real> &hierarchy,
                          const std::size_t l);

  // Forward declaration.
  class iterator;

  //! Return an iterator to the beginning of the nodes.
  iterator begin() const;

  //! Return an iterator to the end of the nodes.
  iterator end() const;

  //! Associated mesh hierarchy.
  const TensorMeshHierarchy<N, Real> &hierarchy;

  //! Node ranges from the coarsest level up to the level being iterated over.
  const std::vector<UnshuffledTensorNodeRange<N, Real>> ranges;

  //! Index of the level being iterated over.
  const std::size_t l;

private:
  //! Endpoints of the node ranges.
  const std::vector<
      std::array<typename UnshuffledTensorNodeRange<N, Real>::iterator, 2>>
      range_endpoints;
};

//! Equality comparison.
template <std::size_t N, typename Real>
bool operator==(const ShuffledTensorNodeRange<N, Real> &a,
                ShuffledTensorNodeRange<N, Real> &b);

//! Inequality comparison.
template <std::size_t N, typename Real>
bool operator!=(const ShuffledTensorNodeRange<N, Real> &a,
                ShuffledTensorNodeRange<N, Real> &b);

//! Iterator over the nodes of a mesh in a mesh hierarchy.
template <std::size_t N, typename Real>
class ShuffledTensorNodeRange<N, Real>::iterator {
public:
  //! Category of the iterator.
  using iterator_category = std::input_iterator_tag;
  //! Type iterated over.
  using value_type = TensorNode<N>;
  //! Type for distance between iterators.
  using difference_type = std::ptrdiff_t;
  //! Pointer to `value_type`.
  using pointer = value_type *;
  //! Type returned by the dereference operator.
  using reference = value_type;

  //! Constructor.
  //!
  //!\param iterable View of nodes to be iterated over.
  //!\param ell Index of mesh currently being iterated over.
  //!\param inner Underlying range iterator.
  //!\param index Position in the shuffled range.
  iterator(const ShuffledTensorNodeRange &iterable, const std::size_t ell,
           const typename UnshuffledTensorNodeRange<N, Real>::iterator &inner,
           const std::size_t index);

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
  const ShuffledTensorNodeRange &iterable;

private:
  //! Index of mesh currently being iterated over.
  std::size_t ell;

  //! Underlying range iterator.
  typename UnshuffledTensorNodeRange<N, Real>::iterator inner;

  //! Position in the `ShuffledTensorNodeRange`.
  //!
  //! This is only stored so we can avoid comparing `inner` in the
  //! (in)equality comparison operators.
  std::size_t index;
};

} // namespace mgard

#include "TensorMeshHierarchyIteration.tpp"
#endif
