#include "pcg.hpp"

#include <stdexcept>
#include <utility>

#include "blaspp/blas.hh"

namespace helpers {

//!Calculate the residual 'from scratch.'
//!
//!\param [in] N Size of the system.
//!\param [in] A Symmetric, positive definite matrix.
//!\param [in] b Righthand side of the system.
//!\param [in] x Candidate solution to the system.
//!\param [out] residual Buffer in which residual will be saved.
static inline void calculate_residual(
    const std::size_t N,
    const LinearOperator &A,
    double const * const b,
    double const * const x,
    double * const residual
) {
    //Could use `std::invoke` here (and elsewhere).
    A(x, residual);
    blas::axpy(N, -1, b, 1, residual, 1);
    blas::scal(N, -1, residual, 1);
}

double pcg(
    const LinearOperator &A,
    double const * const b,
    const LinearOperator &P,
    double * const x,
    double * const buffer,
    const double rtol,
    const std::size_t max_iterations
) {
    if (!(A.is_square() and P.is_square())) {
        throw std::invalid_argument(
            "system matrix and preconditioner must both be square"
        );
    }
    const std::pair<std::size_t, std::size_t> A_dimensions = A.dimensions();
    //Not claiming that I could generalize this to nonsquare systems.
    if (A_dimensions.second != P.dimensions().first) {
        throw std::invalid_argument(
            "system matrix and preconditioner must have the same shape"
        );
    }
    const std::size_t N = A_dimensions.first;
    const std::size_t RESIDUAL_RECALCULATION_ITERVAL = 1 << 5;
    double *_buffer = buffer;
    double * const residual = _buffer;
    _buffer += N;
    //We'll swap these below.
    double *pc_residual = _buffer;
    _buffer += N;
    double *direction = _buffer;
    _buffer += N;
    double * const Adirection = _buffer;
    _buffer += N;

    if (!(0 <= rtol && rtol <= 1)) {
        throw std::invalid_argument("`rtol` must be in [0, 1].");
    }
    const double b_norm = blas::nrm2(N, b, 1);
    //If this quantity is zero, `atol` will be set to zero and our
    //iteration will never 'naturally' end. (The check on
    //`alpha_denominator` or the check on the iteration count should trigger
    //eventually.) Better to immediately return.
    if (!b_norm) {
        for (double *p = x; p != x + N; ++p) {
            *p = 0;
        }
        return 0;
    }
    const double atol = rtol * b_norm;
    calculate_residual(N, A, b, x, residual);
    double residual_norm = blas::nrm2(N, residual, 1);
    if (residual_norm <= atol) {
        return residual_norm / b_norm;
    }
    P(residual, pc_residual);
    blas::copy(N, pc_residual, 1, direction, 1);
    //This should be the inner product of `direction` and `pc_residual`. Here,
    //since they're the same, we can just find the norm of one and square it.
    double beta_numerator = blas::nrm2(N, pc_residual, 1);;
    beta_numerator *= beta_numerator;
    for (std::size_t num_iter = 0; num_iter < max_iterations; ++num_iter) {
        double alpha_numerator = beta_numerator;
        A(direction, Adirection);
        const double alpha_denominator = blas::dot(
            N, Adirection, 1, direction, 1
            );
        //`A` is symmetric positive definite, so this should only happen when
        //`direction` is the zero vector. In that case, further iterations will
        //have no effect and we might as well exit now.
        if (!alpha_denominator) {
            return residual_norm / b_norm;
        }
        const double alpha = alpha_numerator / alpha_denominator;
        blas::axpy(N, alpha, direction, 1, x, 1);
        if ((num_iter + 1) % RESIDUAL_RECALCULATION_ITERVAL) {
            //Recalculate the residual.
            calculate_residual(N, A, b, x, residual);
        } else {
            //Update the residual.
            blas::axpy(N, -alpha, Adirection, 1, residual, 1);
        }
        residual_norm = blas::nrm2(N, residual, 1);
        if (residual_norm <= atol) {
            return residual_norm / b_norm;
        }
        P(residual, pc_residual);
        beta_numerator = blas::dot(N, pc_residual, 1, residual, 1);
        const double beta = beta_numerator / alpha_numerator;
        //This is an attempt to accomplish
        //    direction = beta * direction + pc_residual
        //efficiently. Note that `pc_residual` is recalculated before being read
        //from again. Imagine starting `direction` (the new values, to be
        //assigned) as zero. First we add `pc_residual`.
        std::swap(direction, pc_residual);
        //Then we add `beta * direction`, only `direction` had its name changed
        //to `pc_residual` in the last line.
        blas::axpy(N, beta, pc_residual, 1, direction, 1);
    }
    return residual_norm / b_norm;
}

}
