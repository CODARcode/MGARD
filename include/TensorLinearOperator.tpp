#include <stdexcept>

#include "utilities.hpp"

namespace mgard {

template <std::size_t N, typename Real>
ConstituentLinearOperator<N, Real>::ConstituentLinearOperator(
    const TensorMeshHierarchy<N, Real> &hierarchy, const std::size_t l,
    const std::size_t dimension)
    : hierarchy(&hierarchy), dimension_(dimension),
      indices(hierarchy.indices(l, dimension)) {}

template <std::size_t N, typename Real>
std::size_t ConstituentLinearOperator<N, Real>::dimension() const {
  return indices.size();
}

template <std::size_t N, typename Real>
void ConstituentLinearOperator<N, Real>::
operator()(const std::array<std::size_t, N> multiindex, Real *const v) const {
  // TODO: Could be good to check that `multiindex` corresponds to a 'spear' in
  // the level. For this we'll need to have the indices in every dimension.
  if (multiindex.at(dimension_)) {
    throw std::invalid_argument(
        "'spear' must start at a lower boundary of the domain");
  }
  return do_operator_parentheses(multiindex, v);
}

namespace {

template <std::size_t N, typename Real>
std::array<std::vector<std::size_t>, N>
level_multiindex_components(const TensorMeshHierarchy<N, Real> &hierarchy,
                            const std::size_t l) {
  std::array<std::vector<std::size_t>, N> multiindex_components;
  for (std::size_t i = 0; i < N; ++i) {
    multiindex_components.at(i) = hierarchy.indices(l, i);
  }
  return multiindex_components;
}

} // namespace

template <std::size_t N, typename Real>
TensorLinearOperator<N, Real>::TensorLinearOperator(
    const TensorMeshHierarchy<N, Real> &hierarchy, const std::size_t l,
    const std::array<ConstituentLinearOperator<N, Real> const *, N> operators)
    : hierarchy(hierarchy), operators(operators),
      multiindex_components(level_multiindex_components(hierarchy, l)) {
  for (std::size_t i = 0; i < N; ++i) {
    // TODO: This will cause a problem when `operators.at(i)` is allowed to be
    // `nullptr` (assuming we go that route).
    // TODO: Maybe call something like `initialize_operators` and then have this
    // check (but this will get called before subclass constructors).
    if (multiindex_components.at(i).size() != operators.at(i)->dimension()) {
      throw std::invalid_argument(
          "mesh dimension does not match operator dimension");
    }
  }
}

template <std::size_t N, typename Real>
void TensorLinearOperator<N, Real>::operator()(Real *const v) const {
  std::array<std::vector<std::size_t>, N> multiindex_components_ =
      multiindex_components;
  for (std::size_t i = 0; i < N; ++i) {
    ConstituentLinearOperator<N, Real> const *const A = operators.at(i);
    multiindex_components_.at(i) = {0};
    for (const std::array<std::size_t, N> multiindex :
         CartesianProduct<std::size_t, N>(multiindex_components_)) {
      A->operator()(multiindex, v);
    }
    // Reinstate this dimension's indices for the next iteration.
    multiindex_components_.at(i) = multiindex_components.at(i);
  }
}

} // namespace mgard
