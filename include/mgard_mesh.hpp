#ifndef MGARD_MESH_HPP
#define MGARD_MESH_HPP

#include <cstddef>

#include <array>
#include <iterator>

#include "utilities.hpp"

namespace mgard {

//! Dimensions for a tensor product mesh on a Cartesian product of intervals.
template <std::size_t N> struct Dimensions2kPlus1 {
  //! Constructor.
  //!
  //!\param input_ Numbers of nodes in each dimension.
  Dimensions2kPlus1(const std::array<std::size_t, N> input_);

  //! Mesh dimensions (number of nodes in each dimension) originally input.
  std::array<std::size_t, N> input;

  //! Mesh dimensions (number of nodes in each dimension) rounded to a power of
  //! two plus one.
  std::array<std::size_t, N> rnded;

  //! Overall largest index in the mesh hierarchy. The largest index is one less
  //! than the number of levels. Any dimension of size `1` is ignored.
  std::size_t nlevel;

  //! Determine whether all dimensions are either equal to `1` or of the form
  //! `2^k + 1`.
  bool is_2kplus1() const;
};

//! Equality comparison.
template <std::size_t N>
bool operator==(const Dimensions2kPlus1<N> &a, const Dimensions2kPlus1<N> &b);

//! Inequality comparison.
template <std::size_t N>
bool operator!=(const Dimensions2kPlus1<N> &a, const Dimensions2kPlus1<N> &b);

// As of this writing, `nlevel_from_size` and `size_from_nlevel`are only needed
// in the implementation of the `Dimensions2kPlus1` constructor.

//! Compute `log2(n - 1)`.
//!
//!\param n Size of the mesh in a particular dimension.
std::size_t nlevel_from_size(const std::size_t n);

//! Compute `2^n + 1`.
//!
//!\param n Level index in a particular dimension (assuming a dyadic grid).
std::size_t size_from_nlevel(const std::size_t n);

//! Translate a 2D to a 1D index.
//!
//!\deprecated Use `TensorMeshHierarchy::at` instead.
//!
//!\param ncol Number of columns in the 2D dataset.
//!\param i Row index in the 2D dataset.
//!\param j Column index in the 2D dataset.
int get_index(const int ncol, const int i, const int j);

//! Translate a 3D to a 1D index.
//!
//!\deprecated Use `TensorMeshHierarchy::at` instead.
//!
//!\param ncol Number of columns in the 3D dataset.
//!\param nfib Number of fibers in the 3D dataset.
//!\param i Row index in the 3D dataset.
//!\param j Column index in the 3D dataset.
//!\param k Fiber index in the 3D dataset.
int get_index3(const int ncol, const int nfib, const int i, const int j,
               const int k);

//! Translate a 1D index to an index in the largest contained dyadic grid.
//!
//!\deprecated Use `TensorMeshHierarchy::indices` instead.
//!
//!\param n Size of the largest contained dyadic grid.
//!\param no Size of the 1D grid (greater than or equal to `n`).
//!\param i Row index in the 1D dataset.
int get_lindex(const int n, const int no, const int i);

//! Compute the stride for a mesh level.
//!
//!\param index_difference Difference between the index of the finest mesh level
//! and the index of the mesh level in question.
std::size_t stride_from_index_difference(const std::size_t index_difference);

} // namespace mgard

#include "mgard_mesh.tpp"
#endif
