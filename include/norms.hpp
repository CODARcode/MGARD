#ifndef NORMS_HPP
#define NORMS_HPP
//!\file
//!\brief Function space norms.

#include "MeshHierarchy.hpp"
#include "data.hpp"

namespace mgard {

//! Compute the norm of a function on a mesh hierarchy.
//!
//!\param [in] u Nodal values of the function.
//!\param [in] hierarchy Mesh hierarchy on which the function is defined.
//!\param [in] s Smoothness parameter for the norm.
//!
//!\return Norm of the function.
//!
//! If `s` is `+inf`, the `L^inf` norm (supremum norm) is calculated. Otherwise,
//! the '`s` norm' is calculated. When `s` is zero, this norm is equal to the
//!`L^2` norm.
double norm(const NodalCoefficients<double> u, const MeshHierarchy &hierarchy,
            const double s);

} // namespace mgard

#endif
