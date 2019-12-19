#ifndef MGARD_MESH_HPP
#define MGARD_MESH_HPP

#include <cstddef>

#include <array>

namespace mgard {

template <std::size_t N> struct Dimensions2kPlus1 {
  Dimensions2kPlus1(const std::array<int, N> input_);

  std::array<int, N> input;
  std::array<int, N> rnded;
  std::array<int, N> nlevels;
  int nlevel;
};

template
struct Dimensions2kPlus1<1>;

template
struct Dimensions2kPlus1<2>;

template
struct Dimensions2kPlus1<3>;

// As of this writing, these are only needed in the implementations of the
//`Dimensions2kPlus1` constructor and `is_2kplus1`.
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
