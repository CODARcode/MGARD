#ifndef TENSORNORMS_HPP
#define TENSORNORMS_HPP
//!\file
//!\brief Tensor product mesh function space norms.

#include "TensorMeshHierarchy.hpp"

namespace mgard {

//! Compute the norm of a function on a mesh hierarchy.
//!
//! IMPORTANT: The input must be shuffled.
//!
//!\param [in] hierarchy Mesh hierarchy on which the function is defined.
//!\param [in] u Nodal values of the function.
//!\param [in] s Smoothness parameter for the norm.
//!
//! If `s` is `+inf`, the `L^inf` norm (supremum norm) is calculated. Otherwise,
//! the '`s` norm' is calculated. When `s` is zero, this norm is equal to the
//!`L^2` norm.
template <std::size_t N, typename Real>
Real norm(const TensorMeshHierarchy<N, Real> &hierarchy, Real const *const u,
          const Real s);

//! Compute the norm of a functional on a function space.
//!
//!\deprecated We expect this to be replaced by a function taking a more general
//! quantity of interest callable object.
//!
//!\param[in] nrow Size of the domain grid in the first dimension.
//!\param[in] ncol Size of the domain grid in the second dimension.
//!\param[in] nfib Size of the domain grid in the third dimension.
//!\param[in] qoi Quantity of interest whose norm is to be computed.
//!\param[in] s Smoothness parameter. The norm of the Riesz representative of
//! the functional will be computed using the `s` norm.
//!
//!\return Norm of the functional as an element of `(H^-s)*`.
template <typename Real>
Real norm(const int nrow, const int ncol, const int nfib,
          Real (*const qoi)(int, int, int, Real *), const Real s);

//! Compute the norm of a functional on a function space.
//!
//!\overload
//!\deprecated We expect this to be replaced by a function taking a more general
//! quantity of interest callable object.
//!
//!\param[in] nrow Size of the domain grid in the first dimension.
//!\param[in] ncol Size of the domain grid in the second dimension.
//!\param[in] nfib Size of the domain grid in the third dimension.
//!\param[in] qoi Quantity of interest whose norm is to be computed.
//!\param[in] s Smoothness parameter. The norm of the Riesz representative of
//! the functional will be computed using the `s` norm.
//!\param[in] data Data to be passed to the quantity of interest.
//!
//!\return Norm of the functional as an element of `(H^-s)*`.
template <typename Real>
Real norm(const int nrow, const int ncol, const int nfib,
          Real (*const qoi)(int, int, int, Real *, void *), const Real s,
          void *const data);

} // namespace mgard

#include "TensorNorms.tpp"
#endif
