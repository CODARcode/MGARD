#include <stdexcept>
#include <vector>

#include <omp.h>

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
std::array<TensorIndexRange, N>
level_multiindex_components(const TensorMeshHierarchy<N, Real> &hierarchy,
                            const std::size_t l) {
  std::array<TensorIndexRange, N> multiindex_components;
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
  const std::array<std::size_t, N> &SHAPE = hierarchy.shapes.back();
  for (std::size_t i = 0; i < N; ++i) {
    if (SHAPE.at(i) == 1 && operators.at(i) != nullptr) {
      throw std::invalid_argument("the component operator corresponding to any "
                                  "dimension of size 1 must be the identity");
    }
  }
}

template <std::size_t N, typename Real>
TensorLinearOperator<N, Real>::TensorLinearOperator(
    const TensorMeshHierarchy<N, Real> &hierarchy, const std::size_t l)
    : TensorLinearOperator(hierarchy, l, {}) {}

template <std::size_t N, typename Real>
void TensorLinearOperator<N, Real>::operator()(Real *const v) const {
  std::array<TensorIndexRange, N> multiindex_components_ =
      multiindex_components;
  const std::array<std::size_t, N> &SHAPE = hierarchy.shapes.back();
  for (std::size_t i = 0; i < N; ++i) {
    if (SHAPE.at(i) == 1) {
      continue;
    }
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
    // Range which will yield `0` once.
    multiindex_components_.at(i) = TensorIndexRange::singleton();

    const CartesianProduct<TensorIndexRange, N> product(multiindex_components_);
    const std::vector<std::array<std::size_t, N>> multiindices(product.begin(),
                                                               product.end());
    const std::size_t M = multiindices.size();

#pragma omp parallel for
    for (std::size_t j = 0; j < M; ++j) {
      A->operator()(multiindices.at(j), v);
    }

    // Reinstate this dimension's indices for the next iteration.
    multiindex_components_.at(i) = multiindex_components.at(i);
  }
}

} // namespace mgard
