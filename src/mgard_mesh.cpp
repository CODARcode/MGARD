#include "mgard_mesh.hpp"
#include "mgard_mesh.tpp"

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

// TODO: wait for response to GitHub issue to see what this is supposed to be.
bool is_2kplus1(const int n) {
  return n == 1 || n == size_from_nlevel(nlevel_from_size(n));
}

int get_index(const int ncol, const int i, const int j) { return ncol * i + j; }

int get_index3(const int ncol, const int nfib, const int i, const int j,
               const int k) {
  return (ncol * i + j) * nfib + k;
}

int get_lindex(const int n, const int no, const int i) {
  // no: original number of points
  // n : number of points at next coarser level (L-1) with  2^k+1 nodes
  return (i != n - 1 ? std::floor(i * static_cast<float>(no - 2) / (n - 2))
                     : no - 1);
}

} // namespace mgard
