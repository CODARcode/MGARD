#ifndef INTERPOLATION_HPP
#define INTERPOLATION_HPP
//!\file
//!\brief Interpolate multilinear functions on rectangular cells.

namespace mgard {

//! Interpolate a linear polynomial.
//!
//!\param q0 Function value at the left point.
//!\param q1 Function value at the right point.
//!\param x0 Coordinate of the left point.
//!\param x1 Coordinate of the right point.
//!\param x Coordinate of the point at which to interpolate the function.
//!
//!\return Function value at the given point.
template <typename Real>
Real interpolate(const Real q0, const Real q1, const Real x0, const Real x1,
                 const Real x);

//! Interpolate a bilinear polynomial.
//!
//!\param q00 Function value at the front left point.
//!\param q01 Function value at the back left point.
//!\param q10 Function value at the front right point.
//!\param q11 Function value at the back right point.
//!\param x0 `x` coordinate of the left points.
//!\param x1 `x` coordinate of the right points.
//!\param y0 `y` coordinate of the front points.
//!\param y1 `y` coordinate of the back points.
//!\param x `x` coordinate of the point at which to interpolate the function.
//!\param y `y` coordinate of the point at which to interpolate the function.
//!
//!\return Function value at the given point.
template <typename Real>
Real interpolate(const Real q00, const Real q01, const Real q10, const Real q11,
                 const Real x0, const Real x1, const Real y0, const Real y1,
                 const Real x, const Real y);

//! Interpolate a trilinear polynomial.
//!
//!\param q000 Function value at the bottom front left point.
//!\param q001 Function value at the top front left point.
//!\param q010 Function value at the bottom back left point.
//!\param q011 Function value at the top back left point.
//!\param q100 Function value at the bottom front right point.
//!\param q101 Function value at the top front right point.
//!\param q110 Function value at the bottom back right point.
//!\param q111 Function value at the top back right point.
//!\param x0 `x` coordinate of the left points.
//!\param x1 `x` coordinate of the right points.
//!\param y0 `y` coordinate of the front points.
//!\param y1 `y` coordinate of the back points.
//!\param z0 `z` coordinate of the bottom points.
//!\param z1 `z` coordinate of the top points.
//!\param x `x` coordinate of the point at which to interpolate the function.
//!\param y `y` coordinate of the point at which to interpolate the function.
//!\param z `z` coordinate of the point at which to interpolate the function.
//!
//!\return Function value at the given point.
template <typename Real>
Real interpolate(const Real q000, const Real q001, const Real q010,
                 const Real q011, const Real q100, const Real q101,
                 const Real q110, const Real q111, const Real x0, const Real x1,
                 const Real y0, const Real y1, const Real z0, const Real z1,
                 const Real x, const Real y, const Real z);

} // namespace mgard

#endif
