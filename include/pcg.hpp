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
    //!\param rhs_norm Norm of the righthand side.
    //!
    //!\return Absolute tolerance to use when solving the system.
    double tolerance(const double rhs_norm) const;
};

//!Diagnostics for PCG run.
struct PCGDiagnostics {
    //!Whether the iteration converged or was halted for some other reason.
    bool converged;

    //!Norm of the residual `b - Ax`.
    double residual_norm;

    //!Number of iterations performed.
    std::size_t num_iterations;
};

//!Use the preconditioned conjugate method to solve `Ax = b` for `x`.
//!
//!\param [in] A Symmetric, positive definite matrix.
//!\param [in] b Righthand side of the system.
//!\param [in] P Symmetric, positive definite matrix approximating
//!the inverse of `A`.
//!\param [in, out] x Starting point for the iteration.
//!\param [in] buffer Buffer of size `4 * N` for use in the algorithm.
//!\param [in] criteria Stopping criteria for the iteration.
//!
//!\return Diagnostics of the PCG run.
PCGDiagnostics pcg(
    const LinearOperator &A,
    double const * const b,
    const LinearOperator &P,
    double * const x,
    double * const buffer,
    const PCGStoppingCriteria criteria = PCGStoppingCriteria()
);

}

#endif
