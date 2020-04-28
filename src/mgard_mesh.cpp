#include "mgard_mesh.hpp"

#include <cmath>

#include <stdexcept>

namespace mgard {

static int log2(int n) {
  if (n <= 0) {
    throw std::domain_error("can only take logarithm of positive numbers");
  }
  int exp;
  for (exp = -1; n; ++exp, n >>= 1)
    ;
  return exp;
}

int nlevel_from_size(const int n) { return log2(n - 1); }

int size_from_nlevel(const int n) { return (1 << n) + 1; }

int get_index(const int ncol, const int i, const int j) { return ncol * i + j; }

int get_index3(const int ncol, const int nfib, const int i, const int j,
               const int k) {
  return (ncol * i + j) * nfib + k;
}

int get_lindex(const int n, const int no, const int i) {
  // no: original number of points
  // n : number of points at next coarser level (L-1) with  2^k+1 nodes
  return (i != n - 1 ? std::floor(i * static_cast<double>(no - 2) / (n - 2))
                     : no - 1);
}

std::size_t stride_from_index_difference(const std::size_t index_difference) {
  return 1 << index_difference;
}

} // namespace mgard
