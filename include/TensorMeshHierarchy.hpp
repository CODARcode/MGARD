#ifndef TENSORMESHHIERARCHY_HPP
#define TENSORMESHHIERARCHY_HPP
//!\file
//!\brief Increasing hierarchy of tensor meshes.

#include <cstddef>

#include <array>
#include <vector>

#include "TensorMeshLevel.hpp"

namespace mgard {

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

  //! Access the value associated to a particular node.
  //!
  //!\param v Dataset defined on the hierarchy.
  //!\param multiindex Multiindex of the node.
  Real &at(Real *const v, const std::array<std::size_t, N> multiindex) const;

  //! Meshes composing the hierarchy, in 'increasing' order.
  std::vector<TensorMeshLevel<N, Real>> meshes;

  //! Coordinates of the nodes in the finest mesh.
  std::array<std::vector<Real>, N> coordinates;

  //! Index of finest TensorMeshLevel.
  std::size_t L;

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

} // namespace mgard

#include "TensorMeshHierarchy.tpp"
#endif
