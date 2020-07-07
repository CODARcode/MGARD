#ifndef MGARD_MESH_HPP
#define MGARD_MESH_HPP

#include <cstddef>

#include <array>

namespace mgard {

//! Dimensions for a tensor product mesh on a Cartesian product of intervals.
template <std::size_t N> struct Dimensions2kPlus1 {
  //! Constructor.
  //!
  //!\param input_ Numbers of nodes in each dimension.
  Dimensions2kPlus1(const std::array<int, N> input_);

  //! Mesh dimensions (number of nodes in each dimension) originally input.
  std::array<int, N> input;

  //! Mesh dimensions (number of nodes in each dimension) rounded to a power of
  //! two plus one.
  std::array<int, N> rnded;

  //! Hypothetical largest index in the mesh hierarchies (considering each
  //! dimension separately). The largest index is one less than the number of
  //! levels.
  std::array<int, N> nlevels;

  //! Overall largest index in the mesh hierarchy. The largest index is one less
  //! than the number of levels.
  int nlevel;
};

template struct Dimensions2kPlus1<1>;

template struct Dimensions2kPlus1<2>;

template struct Dimensions2kPlus1<3>;

// As of this writing, these are only needed in the implementations of the
// `Dimensions2kPlus1` constructor and `is_2kplus1`.
int nlevel_from_size(const int n);

int size_from_nlevel(const int n);

bool is_2kplus1(const int n);

// These were originally `inline`.
int get_index(const int ncol, const int i, const int j);

int get_lindex(const int n, const int no, const int i);

int get_index3(const int ncol, const int nfib, const int i, const int j,
               const int k);

} // namespace mgard

#endif
