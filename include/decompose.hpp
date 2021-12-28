// Copyright 2017, Brown University, Providence, RI.
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney, Qing Liu
// Corresponding Author: Ben Whitney, Qing Liu
//
// See LICENSE for details.
#ifndef DECOMPOSE_HPP
#define DECOMPOSE_HPP
//!\file
//!\brief Decomposition and recomposition to and from multilevel coefficients.

#include <cstddef>

#include "TensorMeshHierarchy.hpp"

#ifdef MGARD_PROTOBUF
#include "proto/mgard.pb.h"
#endif

namespace mgard {

//! Transform nodal coefficients into multilevel coefficients.
//!
//!\param[in] hierarchy Mesh hierarchy on which the input function is defined.
//!\param[in, out] v Nodal coefficients of the input function on the finest mesh
//! in the hierarchy.
template <std::size_t N, typename Real>
void decompose(const TensorMeshHierarchy<N, Real> &hierarchy, Real *const v);

#ifdef MGARD_PROTOBUF
//! Transform nodal coefficients into multilevel coefficients.
//!
//!\overload
//!
//! The relevant fields of `header` will be populated.
//!
//!\param[in] hierarchy Mesh hierarchy on which the input function is defined.
//!\param[in, out] v Nodal coefficients of the input function on the finest mesh
//!\param[in, out] v Header to be populated.
template <std::size_t N, typename Real>
void decompose(const TensorMeshHierarchy<N, Real> &hierarchy, Real *const v,
               pb::Header &header);
#endif

//! Transform multilevel coefficients into nodal coefficients.
//!
//!\param[in] hierarchy Mesh hierarchy on which the output function is defined.
//!\param[in, out] v Multilevel coefficients of the output function on the
//! finest mesh in the hierarchy.
template <std::size_t N, typename Real>
void recompose(const TensorMeshHierarchy<N, Real> &hierarchy, Real *const v);

#ifdef MGARD_PROTOBUF
//! Transform multilevel coefficients into nodal coefficients.
//!
//!\overload
//!
//!\param[in] hierarchy Mesh hierarchy on which the output function is defined.
//!\param[in, out] v Multilevel coefficients of the output function on the
//! finest mesh in the hierarchy.
//!\param[in] header Header parsed from the original self-describing buffer.
template <std::size_t N, typename Real>
void recompose(const TensorMeshHierarchy<N, Real> &hierarchy, Real *const v,
               const pb::Header &header);
#endif

} // namespace mgard

#include "decompose.tpp"
#endif
