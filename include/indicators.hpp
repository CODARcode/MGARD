#ifndef INDICATORS_HPP
#define INDICATORS_HPP
//!\file
//!\brief Function space norm estimator indicators.

#include "IndicatorInput.hpp"
#include "estimators.hpp"

namespace mgard {

//! Compute the factor used to quantize a (square) multilevel coefficient.
//!
//!\param input Mesh index, mesh, node, and multilevel coefficient needed to
//! compute the square indicator coefficient.
//\param s Smoothness parameter.
template <typename Real>
SandwichBounds square_indicator_coefficient(const IndicatorInput<Real> input,
                                            const float s);

} // namespace mgard

#include "indicators.tpp"
#endif
