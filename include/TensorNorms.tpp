#include <cmath>
#include <cstddef>
#include <cstdlib>

#include <algorithm>
#include <array>

#include "TensorMassMatrix.hpp"
#include "TensorRestriction.hpp"
#include "blas.hpp"
#include "utilities.hpp"

namespace mgard {

namespace {

template <std::size_t N, typename Real>
Real L_infinity_norm(const TensorMeshHierarchy<N, Real> &hierarchy,
                     Real const *const u) {
  Real maximum = 0;
  for (const Real x : PseudoArray<const Real>(u, hierarchy.ndof())) {
    maximum = std::max(maximum, std::abs(x));
  }
  return maximum;
}

template <std::size_t N, typename Real>
Real L_2_norm(const TensorMeshHierarchy<N, Real> &hierarchy,
              Real const *const u) {
  const std::size_t ndof = hierarchy.ndof();
  // TODO: Allow memory buffer to be passed in.
  Real *const product = static_cast<Real *>(std::malloc(sizeof(Real) * ndof));
  {
    std::copy(u, u + ndof, product);
    const TensorMassMatrix<N, Real> M(hierarchy, hierarchy.L);
    M(product);
  }
  const Real norm = std::sqrt(blas::dotu(ndof, u, product));
  std::free(product);
  return norm;
}

} // namespace

template <std::size_t N, typename Real>
std::vector<Real>
orthogonal_component_square_norms(const TensorMeshHierarchy<N, Real> &hierarchy,
                                  Real const *const u, Real *const f) {
  // Square `L^2` norms of the `L^2` projections of `u` onto the levels in the
  // hierarchy, ordered from coarsest to finest.
  std::vector<Real> square_norms(hierarchy.L + 1);

  // For the finest level, we don't need to compute the projection.
  square_norms.at(hierarchy.L) = blas::dotu(hierarchy.ndof(), u, f);

  // Getting away with allocating `hierarchy.ndof(hierarchy.L - 1)` instead of
  // `hierarchy.ndof(L)` `Real`s because the coefficients corresponding to the
  // `hierarchy.L - 1`th level are located at the front of a shuffled array.
  Real *const projection =
      hierarchy.L ? static_cast<Real *>(std::malloc(
                        hierarchy.ndof(hierarchy.L - 1) * sizeof(Real)))
                  : nullptr;
  // Shuffled arrays hold the coefficients associated to any of the levels in
  // the hierarchy in a contiguous block at the front. This function relies on
  // this property, although the calls to `hierarchy.on_nodes` might give the
  // impression that it's agnostic to it. The functions at the top of
  // `mgard.tpp` more nicely mix statements that might work for arrays shuffled
  // differently and statements that rely on the current shuffling algorithm.
  for (std::size_t i = 1; i <= hierarchy.L; ++i) {
    const std::size_t l = hierarchy.L - i;

    const PseudoArray<Real> f_on_finer = hierarchy.on_nodes(f, l + 1);
    const TensorRestriction<N, Real> R(hierarchy, l + 1);
    R(f_on_finer.data);
    const PseudoArray<const Real> f_on_l =
        hierarchy.on_nodes(static_cast<Real const *>(f), l);

    const PseudoArray<Real> projection_on_l = hierarchy.on_nodes(projection, l);
    std::copy(f_on_l.begin(), f_on_l.end(), projection_on_l.begin());

    const TensorMassMatrixInverse<N, Real> m_inv(hierarchy, l);
    m_inv(projection_on_l.data);

    square_norms.at(l) =
        blas::dotu(projection_on_l.size, projection_on_l.data, f_on_l.data);
  }
  std::free(projection);

  for (std::size_t i = 1; i <= hierarchy.L; ++i) {
    const std::size_t l = hierarchy.L - i;

    // In the Python implementation, I found I could get negative differences
    // when an orthogonal component was almost zero.
    square_norms.at(l + 1) = std::max(
        static_cast<Real>(0), square_norms.at(l + 1) - square_norms.at(l));
  }
  return square_norms;
}

namespace {

template <std::size_t N, typename Real>
Real s_norm(const TensorMeshHierarchy<N, Real> &hierarchy, Real const *const u,
            const Real s) {
  const std::size_t ndof = hierarchy.ndof();

  Real *const f = static_cast<Real *>(std::malloc(sizeof(Real) * ndof));
  {
    std::copy(u, u + ndof, f);
    const TensorMassMatrix<N, Real> M(hierarchy, hierarchy.L);
    M(f);
  }
  const std::vector<Real> squares_for_norm =
      orthogonal_component_square_norms<N, Real>(hierarchy, u, f);
  std::free(f);

  Real square_norm = 0;
  for (std::size_t l = 0; l <= hierarchy.L; ++l) {
    square_norm += std::exp2(2 * s * l) * squares_for_norm.at(l);
  }
  return std::sqrt(square_norm);
}

} // namespace

template <std::size_t N, typename Real>
Real norm(const TensorMeshHierarchy<N, Real> &hierarchy, Real const *const u,
          const Real s) {
  if (s == std::numeric_limits<Real>::infinity()) {
    return L_infinity_norm(hierarchy, u);
  } else if (s == 0) {
    return L_2_norm(hierarchy, u);
  } else {
    return s_norm(hierarchy, u, s);
  }
}

} // namespace mgard
