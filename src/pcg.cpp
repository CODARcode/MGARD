#include "pcg.hpp"

#include <cassert>

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "blas.hpp"

//! Calculate the residual 'from scratch.'
//!
//!\param [in] N Size of the system.
//!\param [in] A Symmetric, positive definite matrix.
//!\param [in] b Righthand side of the system.
//!\param [in] x Candidate solution to the system.
//!\param [out] residual Buffer in which residual will be saved.
static inline void calculate_residual(const std::size_t N,
                                      const mgard::LinearOperator &A,
                                      double const *const b,
                                      double const *const x,
                                      double *const residual) {
  // Could use `std::invoke` here (and elsewhere).
  A(x, residual);
  blas::axpy(N, -1.0, b, residual);
  blas::scal(N, -1.0, residual);
}

namespace mgard {

namespace pcg {

StoppingCriteria::StoppingCriteria(const double relative, const double absolute,
                                   const std::size_t max_iterations)
    : relative(relative), absolute(absolute), max_iterations(max_iterations) {
  if (relative <= 0 && absolute <= 0) {
    throw std::invalid_argument("must provide a positive tolerance");
  }
  if (relative > 1) {
    throw std::invalid_argument("relative tolerance cannot be greater than 1");
  }
}

double StoppingCriteria::tolerance(const double rhs_norm) const {
  assert(rhs_norm >= 0);
  return std::max(absolute, relative * rhs_norm);
}

Diagnostics pcg(const LinearOperator &A, double const *const b,
                const LinearOperator &P, double *const x, double *const buffer,
                const StoppingCriteria criteria) {
  if (!(A.is_square() and P.is_square())) {
    throw std::invalid_argument(
        "system matrix and preconditioner must both be square");
  }
  const std::pair<std::size_t, std::size_t> A_dimensions = A.dimensions();
  // Not claiming that I could generalize this to nonsquare systems.
  if (A_dimensions.second != P.dimensions().first) {
    throw std::invalid_argument(
        "system matrix and preconditioner must have the same shape");
  }
  const std::size_t N = A_dimensions.first;
  const std::size_t RESIDUAL_RECALCULATION_ITERVAL = 1 << 5;
  double *_buffer = buffer;
  double *const residual = _buffer;
  _buffer += N;
  // We'll swap these below.
  double *pc_residual = _buffer;
  _buffer += N;
  double *direction = _buffer;
  _buffer += N;
  double *const Adirection = _buffer;
  _buffer += N;

  // These variables are updated both inside and outside the loop. Declaring/
  // defining them here will make the assignments look the same (I always
  // momentarily wonder about constness).
  double residual_norm;
  double beta_numerator;

  const double b_norm = blas::nrm2(N, b);
  //`Criteria::tolerance` expects a nonzero righthand side norm. We can just
  // return immediately in this case.
  if (b_norm == 0) {
    std::fill(x, x + N, 0);
    return {true, 0, 0};
  }
  const double atol = criteria.tolerance(b_norm);
  calculate_residual(N, A, b, x, residual);
  residual_norm = blas::nrm2(N, residual);
  if (residual_norm <= atol) {
    return {true, residual_norm, 0};
  }
  P(residual, pc_residual);
  // Presumably `dot` in general.
  beta_numerator = blas::dotu(N, residual, pc_residual);
  blas::copy(N, pc_residual, direction);
  for (std::size_t num_iter = 0; num_iter < criteria.max_iterations;
       ++num_iter) {
    const double alpha_numerator = beta_numerator;
    A(direction, Adirection);
    const double alpha_denominator = blas::dotu(N, Adirection, direction);
    //`A` is symmetric positive definite, so this should only happen when
    //`direction` is the zero vector. In that case, further iterations will
    // have no effect and we might as well exit now.
    if (alpha_denominator == 0) {
      //`num_iter` rather than `num_iter + 1` because we haven't updated
      //`x` yet.
      return {false, residual_norm, num_iter};
    }
    const double alpha = alpha_numerator / alpha_denominator;
    blas::axpy(N, alpha, direction, x);
    if ((num_iter + 1) % RESIDUAL_RECALCULATION_ITERVAL) {
      // Recalculate the residual.
      calculate_residual(N, A, b, x, residual);
    } else {
      // Update the residual.
      blas::axpy(N, -alpha, Adirection, residual);
    }
    residual_norm = blas::nrm2(N, residual);
    if (residual_norm <= atol) {
      return {true, residual_norm, num_iter + 1};
    }
    P(residual, pc_residual);
    beta_numerator = blas::dotu(N, pc_residual, residual);
    const double beta = beta_numerator / alpha_numerator;
    // This is an attempt to accomplish
    //    direction = beta * direction + pc_residual
    // efficiently. Note that `pc_residual` is recalculated before being read
    // from again. Imagine starting `direction` (the new values, to be
    // assigned) as zero. First we add `pc_residual`.
    std::swap(direction, pc_residual);
    // Then we add `beta * direction`, only `direction` had its name changed
    // to `pc_residual` in the last line.
    blas::axpy(N, beta, pc_residual, direction);
  }
  return {false, residual_norm, criteria.max_iterations};
}

} // namespace pcg

} // namespace mgard
