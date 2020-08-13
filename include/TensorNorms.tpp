#include <cmath>

#include <algorithm>
#include <vector>

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
  for (const Real x : PseudoArray(u, hierarchy.ndof())) {
    maximum = std::max(maximum, std::abs(x));
  }
  return maximum;
}

template <std::size_t N, typename Real>
Real L_2_norm(const TensorMeshHierarchy<N, Real> &hierarchy,
              Real const *const u) {
  // TODO: Allow memory buffer to be passed in.
  const std::size_t ndof = hierarchy.ndof();
  std::vector<Real> product_(ndof);
  Real *const product = product_.data();
  std::copy(u, u + ndof, product);
  const TensorMassMatrix<N, Real> M(hierarchy, hierarchy.L);
  M(product);
  return std::sqrt(blas::dotu(ndof, u, product));
}

template <std::size_t N, typename Real>
Real s_norm(const TensorMeshHierarchy<N, Real> &hierarchy, Real const *const u,
            const Real s) {
  const std::size_t ndof = hierarchy.ndof();

  // TODO: Allow memory buffers to be passed in
  std::vector<Real> product_(ndof);
  Real *const product = product_.data();

  // In theory this only needs to be as big as the second-finest mesh. But then
  // the indices would all change, so we'd have to form another
  // `TensorMeshHierarchy`.
  std::vector<Real> projection_(ndof);
  Real *const projection = projection_.data();

  // Square `L^2` norms of the `L^2` projections of `u` onto the levels in the
  // hierarchy, ordered from coarsest to finest.
  std::vector<Real> squares_for_norm(hierarchy.L + 1);

  // This is the only time `product` will contain something other than the
  // product of a mass matrix and a projection of `u`.
  std::copy(u, u + ndof, product);

  // For the finest level, we don't need to compute the projection.
  {
    const TensorMassMatrix<N, Real> M(hierarchy, hierarchy.L);
    M(product);
    squares_for_norm.at(hierarchy.L) = blas::dotu(ndof, u, product);
  }

  for (std::size_t i = 1; i <= hierarchy.L; ++i) {
    const std::size_t l = hierarchy.L - i;
    const TensorNodeRange<N, Real> nodes = hierarchy.nodes(l);

    const TensorRestriction<N, Real> R(hierarchy, l + 1);
    R(product);

    for (const TensorNode<N, Real> node : nodes) {
      const std::array<std::size_t, N> &multiindex = node.multiindex;
      hierarchy.at(projection, multiindex) = hierarchy.at(product, multiindex);
    }
    const TensorMassMatrixInverse<N, Real> m_inv(hierarchy, l);
    m_inv(projection);

    Real projection_square_norm = 0;
    for (const TensorNode<N, Real> node : nodes) {
      const std::array<std::size_t, N> &multiindex = node.multiindex;
      projection_square_norm += hierarchy.at(projection, multiindex) *
                                hierarchy.at(product, multiindex);
    }
    squares_for_norm.at(l) = projection_square_norm;
  }

  // Could have accumulated this as we went.
  Real square_norm = 0;
  for (std::size_t i = 0; i <= hierarchy.L; ++i) {
    const std::size_t l = hierarchy.L - i;
    // In the Python implementation, I found I could get negative differences
    // when an orthogonal component was almost zero.
    const Real difference_of_squares =
        std::max(static_cast<Real>(0),
                 squares_for_norm.at(l) - (l ? squares_for_norm.at(l - 1) : 0));
    square_norm += std::pow(2, 2 * s * l) * difference_of_squares;
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
