#include "mgard_mesh.hpp"

#include <cmath>

#include <stdexcept>

namespace mgard {

static std::size_t log2(std::size_t n) {
  if (!n) {
    throw std::domain_error("can only take logarithm of positive numbers");
  }
  int exp;
  for (exp = -1; n; ++exp, n >>= 1)
    ;
  return static_cast<std::size_t>(exp);
}

std::size_t nlevel_from_size(const std::size_t n) {
  if (n == 0) {
    throw std::domain_error("size must be nonzero");
  }
  return log2(n - 1);
}

std::size_t size_from_nlevel(const std::size_t n) { return (1 << n) + 1; }

} // namespace mgard
