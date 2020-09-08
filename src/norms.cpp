#include "norms.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>

#include <algorithm>
#include <limits>
#include <vector>

#include "blas.hpp"

#include "MassMatrix.hpp"
#include "UniformRestriction.hpp"
#include "pcg.hpp"
#include "utilities.hpp"

namespace mgard {

static double L_infinity_norm(const NodalCoefficients<double> u,
                              const MeshHierarchy &hierarchy) {
  // This could be done with something like `std::transform_reduce` once it's
  // available. Could also use `blas::iamax`.
  double maximum = 0;
  for (const double x : PseudoArray(u.data, hierarchy.ndof())) {
    maximum = std::max(maximum, std::abs(x));
  };
  return maximum;
}

static double L_2_norm(const NodalCoefficients<double> u,
                       const MeshHierarchy &hierarchy) {
  const std::size_t N = hierarchy.ndof();
  const MassMatrix M(hierarchy.meshes.back());
  // TODO: Allow memory buffer to be passed in.
  std::vector<double> scratch(N);
  // This variable name comes from `s_norm`. Not solving any systems here.
  double *const rhs = scratch.data();
  M(u.data, rhs);
  return std::sqrt(blas::dotu(N, u.data, rhs));
}

static double s_norm(const NodalCoefficients<double> u,
                     const MeshHierarchy &hierarchy, const double s) {
  // Square `L^2` norms of the `L^2` projections of `u` onto the levels in the
  // hierarchy, ordered from coarsest to finest.
  std::vector<double> squares_for_norm(hierarchy.L + 1);
  // TODO: allow memory buffer to be passed in.
  std::size_t scratch_sizes[2];
  {
    const std::size_t N = hierarchy.ndof();
    const std::size_t n = (hierarchy.L ? hierarchy.ndof(hierarchy.L - 1) : 0);
    scratch_sizes[0] = std::max(4 * n + n, N);
    scratch_sizes[1] = n;
  }
  std::vector<double> scratch(scratch_sizes[0] + scratch_sizes[1]);
  // Product of mass matrix and projection of `u` to the fine mesh.
  double *RHS = NULL;
  // Product of mass matrix and projection of `u` to the coarse mesh.
  double *rhs = NULL;
  // Projection of `u` to the coarse mesh.
  double *projection = NULL;
  // Buffer needed for PCG.
  double *pcg_buffer = NULL;
  // For the finest level, we get the righthand side by applying the mass
  // matrix to the projection (here, `u`).
  {
    std::size_t l = hierarchy.L;
    const MeshLevel &mesh = hierarchy.meshes.at(l);
    const std::size_t n = mesh.ndof();
    const MassMatrix M(mesh);

    double *_buffer = scratch.data();
    rhs = _buffer;
    _buffer += scratch_sizes[0];
    // TODO: currently inaccurate. Try converting.
    // I would assign `projection` to `u.data` here for consistency, but `u`
    // points to `const double` while `projection` points to `double`.
    M(u.data, rhs);
    squares_for_norm.at(l) = blas::dotu(n, u.data, rhs);

    RHS = rhs;
    rhs = _buffer;
    _buffer += scratch_sizes[1];
  }
  // For the coarser levels, we get the righthand side by restricting the
  // previous righthand side to the latest coarse level.
  for (std::size_t i = 1; i <= hierarchy.L; ++i) {
    const std::size_t l = hierarchy.L - i;
    const MeshLevel &MESH = hierarchy.meshes.at(l + 1);
    const MeshLevel &mesh = hierarchy.meshes.at(l);
    const std::size_t n = mesh.ndof();
    const MassMatrix M(mesh);
    const MassMatrixPreconditioner P(mesh);
    const UniformRestriction R = UniformRestriction(MESH, mesh);
    R(RHS, rhs);
    // In the first iteration, we split `RHS` into `pcg_buffer` and
    //`projection`. In the rest of the iterations, `pcg_buffer` stays in
    // place while `projection` reclaims `RHS`. (Here both arrays use only
    // part of the available space.)
    if (pcg_buffer == NULL) {
      double *_buffer = RHS;
      pcg_buffer = _buffer;
      // Could possibly incorporate `4 * n` into `scratch_sizes` somehow.
      _buffer += 4 * n;
      projection = _buffer;
      _buffer += n;
    } else {
      projection = RHS;
    }
    RHS = NULL;
    std::fill(projection, projection + n, 0);
    const pcg::Diagnostics diagnostics =
        pcg::pcg(M, rhs, P, projection, pcg_buffer);
    assert(diagnostics.converged);
    squares_for_norm.at(l) = blas::dotu(n, projection, rhs);

    RHS = rhs;
    rhs = projection;
    projection = NULL;
  }

  // Could have accumulated this as we went.
  double square_norm = 0;
  for (std::size_t i = 0; i <= hierarchy.L; ++i) {
    const std::size_t l = hierarchy.L - i;
    // In the Python implementation, I found I could get negative differences
    // when an orthogonal component was almost zero.
    const double difference_of_squares = std::max(
        0.0, squares_for_norm.at(l) - (l ? squares_for_norm.at(l - 1) : 0));
    square_norm += std::exp2(2 * s * l) * difference_of_squares;
  }
  return std::sqrt(square_norm);
}

double norm(const NodalCoefficients<double> u, const MeshHierarchy &hierarchy,
            const double s) {
  if (s == std::numeric_limits<double>::infinity()) {
    return L_infinity_norm(u, hierarchy);
  } else if (s == 0) {
    return L_2_norm(u, hierarchy);
  } else {
    return s_norm(u, hierarchy, s);
  }
}

} // namespace mgard
