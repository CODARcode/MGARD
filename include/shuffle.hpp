#ifndef SHUFFLE_HPP
#define SHUFFLE_HPP
//!\file
//!\brief Reorder nodal coefficients by level and physical location.

#include "TensorMeshHierarchy.hpp"

namespace mgard {

//! Reorder nodal coefficients by level.
//!
//!\param[in] hierarchy Mesh hierarchy on which the coefficients are defined.
//!\param[in] src Input coefficients, ordered by physical location.
//!\param[out] dst Output coefficients, ordered by level.
template <std::size_t N, typename Real>
void shuffle(const TensorMeshHierarchy<N, Real> &hierarchy,
             Real const *const src, Real *const dst);

//! Reorder nodal coefficients by physical location.
//!
//!\param[in] hierarchy Mesh hierarchy on which the coefficients are defined.
//!\param[in] src Input coefficients, ordered by level.
//!\param[out] dst Output coefficients, ordered by physical location.
template <std::size_t N, typename Real>
void unshuffle(const TensorMeshHierarchy<N, Real> &hierarchy,
               Real const *const src, Real *const dst);

} // namespace mgard

#include "shuffle.tpp"
#endif
