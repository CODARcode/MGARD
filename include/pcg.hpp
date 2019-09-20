#ifndef INCLUDE_HPP
#define INCLUDE_HPP
//!\file
//!\brief Implementation of the preconditioned conjugate gradient method for
//!solving linear systems.

#include <cstddef>

#include "LinearOperator.hpp"

namespace helpers {

//!Use the preconditioned conjugate method to solve `Ax = b` for `x`.
//!
//!\param [in] A Symmetric, positive definite matrix.
//!\param [in] b Righthand side of the system.
//!\param [in] preconditioner Symmetric, positive definite matrix approximating
//!the inverse of `A`.
//!\param [in, out] x Starting point for the iteration.
//!\param [in] buffer Buffer of size `4 * N` for use in the algorithm.
//!\param [in] rtol Relative error threshold for stopping iteration. If the l²
//!norm of the residual falls below `rtol` times the l² norm of the righthand
//!side, the algorithm will stop.
//!\param [in] max_iterations Condition for stopping iteration. After
//!`max_iterations` iterations, the algorithm will stop even if the error exceeds
//!the tolerance.
//!
//!\return The ratio of the norm of the residual to the norm of the righthand
//!side. Zero if the righthand side is zero.
double pcg(
    const LinearOperator &A,
    double const * const b,
    const LinearOperator &P,
    double * const x,
    double * const buffer,
    const double rtol = 1e-9,
    const std::size_t max_iterations = 1 << 10
);

}

#endif
