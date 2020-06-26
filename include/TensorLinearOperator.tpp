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
      multiindex_components(level_multiindex_components(hierarchy, l)) {}

template <std::size_t N, typename Real>
TensorLinearOperator<N, Real>::TensorLinearOperator(
    const TensorMeshHierarchy<N, Real> &hierarchy, const std::size_t l)
    : TensorLinearOperator(hierarchy, l, {}) {
  operators.fill(nullptr);
}

template <std::size_t N, typename Real>
void TensorLinearOperator<N, Real>::operator()(Real *const v) const {
  std::array<std::vector<std::size_t>, N> multiindex_components_ =
      multiindex_components;
  for (std::size_t i = 0; i < N; ++i) {
    ConstituentLinearOperator<N, Real> const *const A = operators.at(i);
    // We can't check these preconditions in the constructor because the
    // operators won't be valid at that point in derived class constructors. It
    // shouldn't be very expensive to run the tests each time this operator is
    // called. Possibly we could put them in some sort of setter method for
    // `operators`.
    if (A == nullptr) {
      throw std::logic_error("operator has not been initialized");
    }
    if (A->dimension() != multiindex_components.at(i).size()) {
      throw std::invalid_argument(
          "operator dimension does not match mesh dimension");
    }
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
