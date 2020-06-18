#include <stdexcept>

#include "utilities.hpp"

namespace mgard {

template <std::size_t N, typename Real>
ConstituentLinearOperator<N, Real>::ConstituentLinearOperator(
    const TensorMeshHierarchy<N, Real> &hierarchy, const std::size_t l,
    const std::size_t dimension)
    : hierarchy(hierarchy), dimension_(dimension),
      indices(hierarchy.indices(l, dimension)) {}

template <std::size_t N, typename Real>
std::size_t ConstituentLinearOperator<N, Real>::dimension() const {
  return indices.size();
}

template <std::size_t N, typename Real>
void ConstituentLinearOperator<N, Real>::
operator()(const std::array<std::size_t, N> multiindex, Real *const v) const {
  // TODO: Could be good to check that `multiindex` corresponds to a 'spear' in
  // the level.
  if (multiindex.at(dimension_)) {
    throw std::invalid_argument(
        "'spear' must start at a lower boundary of the domain");
  }
  return do_operator_parentheses(multiindex, v);
}

} // namespace mgard
