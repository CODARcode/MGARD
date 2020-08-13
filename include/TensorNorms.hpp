#ifndef TENSORNORMS_HPP
#define TENSORNORMS_HPP
//!\file
//!\brief Tensor product mesh function space norms.

#include "TensorMeshHierarchy.hpp"

namespace mgard {

template <std::size_t N, typename Real>
//! Compute the norm of a function on a mesh hierarchy.
//!
//!\param [in] hierarchy Mesh hierarchy on which the function is defined.
//!\param [in] u Nodal values of the function.
//!\param [in] s Smoothness parameter for the norm.
//!
//! If `s` is `+inf`, the `L^inf` norm (supremum norm) is calculated. Otherwise,
//! the '`s` norm' is calculated. When `s` is zero, this norm is equal to the
//!`L^2` norm.
Real norm(const TensorMeshHierarchy<N, Real> &hierarchy, Real const *const u,
          const Real s);

} // namespace mgard

#include "TensorNorms.tpp"
#endif
