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

} // namespace mgard

#include "mgard_mesh.tpp"
#endif
