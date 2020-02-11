#ifndef INDICATORS_HPP
#define INDICATORS_HPP
//!\file
//!\brief Function space norm estimator indicators.

#include "IndicatorInput.hpp"
#include "MeshHierarchy.hpp"
#include "estimators.hpp"

namespace mgard {

//! Compute the factors by which to scale the square `s` indicator to bound the
//! square `s` estimator below and above.
//!
//!\param hierarchy Mesh hierarchy on which the indicators and estimators are
//! computed.
RatioBounds s_square_indicator_bounds(const MeshHierarchy &hierarchy);

//! Compute the factor used to quantize a (square) multilevel coefficient.
//!
//! This factor includes the reliability constants for the square estimator
//! and indicator, so that scaling the square multilevel coefficients by these
//! factors and summing will produce an upper bound for the square norm.
//!
//!\param input Mesh index, mesh, node, and multilevel coefficient needed to
//! compute the square indicator coefficient.
//\param s Smoothness parameter.
template <typename Real>
Real square_indicator_factor(const IndicatorInput<Real> input, const float s);

} // namespace mgard

#include "indicators.tpp"
#endif
