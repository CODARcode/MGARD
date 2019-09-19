#include "LinearOperator.hpp"

namespace helpers {

LinearOperator::LinearOperator(const std::size_t N, const std::size_t M):
    domain_dimension(N),
    range_dimension(M)
{
}

LinearOperator::LinearOperator(const std::size_t N):
    LinearOperator(N, N)
{
}

bool LinearOperator::is_square() const {
    return domain_dimension == range_dimension;
}

void LinearOperator::operator()(
    double const * const x, double * const b
) const {
    return do_operator_parentheses(x, b);
}

}
