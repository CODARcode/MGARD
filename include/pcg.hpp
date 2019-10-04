#ifndef INCLUDE_HPP
#define INCLUDE_HPP
//!\file
//!\brief Implementation of the preconditioned conjugate gradient method for
//!solving linear systems.

#include <cstddef>

#include "LinearOperator.hpp"

namespace helpers {

//!Stopping criteria for PCG algorithm.
struct PCGStoppingCriteria {
    //!Constructor.
    //!
    //!\param relative Relative tolerance.
    //!\param absolute Absolute tolerance.
    //!\param max_iterations Maximum number of PCG iterations.
    //!
    //!`relative` must be no more than 1. To indicate that a tolerance should
    //!be ignored, pass a nonpositive value. At least one tolerance must be
    //!positive.
    PCGStoppingCriteria(
        const double relative = 1e-9,
        const double absolute = 1e-12,
        const std::size_t max_iterations = 1 << 15
    );

    //!Relative tolerance;
    double relative;

    //!Absolute tolerance;
    double absolute;

    //!Maximum number of PCG iterations.
    std::size_t max_iterations;

    //!Compute the overall absolute tolerance to use when solving a system.
    //!
    //!\param N rhs_norm Norm of the righthand side.
    //!
    //!\return Absolute tolerance to use when solving the system.
    double tolerance(const double rhs_norm) const;
};

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
    const PCGStoppingCriteria criteria = PCGStoppingCriteria()
);

}

#endif
