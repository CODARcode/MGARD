#ifndef MGARD_NORMS_HPP
#define MGARD_NORMS_HPP
//!\file
//!\brief Compute function(al) norms needed for the structured implementation.

#include <vector>

namespace mgard {

//! Compute the norm of a functional on a function space.
//!
//!\param[in] nrow Size of the domain grid in the first dimension.
//!\param[in] ncol Size of the domain grid in the second dimension.
//!\param[in] nfib Size of the domain grid in the third dimension.
//!\param[in] coords_x First coordinates of the nodes of the grid.
//!\param[in] coords_y Second coordinates of the nodes of the grid.
//!\param[in] coords_z Third coordinates of the nodes of the grid.
//!\param[in] qoi Quantity of interest whose norm is to be computed.
//!\param[in] s Smoothness parameter. The norm of the Riesz representative of
//! the functional will be computed using the `s` norm.
//!
//!\return Norm of the functional as an element of `(H^-s)*`.
template <typename Real>
Real qoi_norm(const int nrow, const int ncol, const int nfib,
              const std::vector<Real> &coords_x,
              const std::vector<Real> &coords_y,
              const std::vector<Real> &coords_z,
              Real (*const qoi)(int, int, int, std::vector<Real>),
              const Real s);

//! Compute the norm of a functional on a function space.
//!
//!\note This is a C-compatible overload of the above function, differing only
//! in the type of `qoi`.
//!
//!\param[in] nrow Size of the domain grid in the first dimension.
//!\param[in] ncol Size of the domain grid in the second dimension.
//!\param[in] nfib Size of the domain grid in the third dimension.
//!\param[in] coords_x First coordinates of the nodes of the grid.
//!\param[in] coords_y Second coordinates of the nodes of the grid.
//!\param[in] coords_z Third coordinates of the nodes of the grid.
//!\param[in] qoi Quantity of interest whose norm is to be computed.
//!\param[in] s Smoothness parameter. The norm of the Riesz representative of
//! the functional will be computed using the `s` norm.
//!
//!\return Norm of the functional as an element of `(H^-s)*`.
template <typename Real>
Real qoi_norm(const int nrow, const int ncol, const int nfib,
              const std::vector<Real> &coords_x,
              const std::vector<Real> &coords_y,
              const std::vector<Real> &coords_z,
              Real (*const qoi)(int, int, int, Real *), const Real s);

} // namespace mgard

#include "mgard_norms.tpp"
#endif
