#include "LinearOperator.hpp"

namespace mgard {

LinearOperator::LinearOperator(const std::size_t N, const std::size_t M)
    : domain_dimension(N), range_dimension(M) {}

LinearOperator::LinearOperator(const std::size_t N) : LinearOperator(N, N) {}

std::pair<std::size_t, std::size_t> LinearOperator::dimensions() const {
  return {domain_dimension, range_dimension};
}

bool LinearOperator::is_square() const {
  return domain_dimension == range_dimension;
}

void LinearOperator::operator()(double const *const x, double *const b) const {
  return do_operator_parentheses(x, b);
}

} // namespace mgard
