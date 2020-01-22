#ifndef ESTIMATORS_HPP
#define ESTIMATORS_HPP
//!\file
//!\brief Function space norm estimators.

#include "MeshHierarchy.hpp"
#include "data.hpp"

namespace mgard {

//! Bounds relating an estimator (or indicator norm) to the corresponding norm
//!(or estimator). If `estimate` is the estimate and `norm` the corresponding
//! norm, we will have
//!    realism * estimate <= norm <= reliability * estimate
//!
//! Note: I don't expect the user to ever instantiate this.
struct RatioBounds {
  //! Realism constant, controlling how much smaller the norm can be.
  double realism;

  //! Reliability constant, controlling how much larger the norm can be.
  double reliability;
};

//! Upper and lower bounds on a norm.
//!
//! Note: I don't expect the user to ever instantiate this.
struct SandwichBounds {
  //! Constructor.
  //!
  //!\param [in] bounds Bounds for the 'bread.'
  //!\param [in] unscaled Estimate for the 'meat.'
  SandwichBounds(const RatioBounds bounds, const double unscaled);

  //! Lower bound for the norm.
  double lower;

  //! Unscaled estimate.
  double unscaled;

  //! Upper bound for the norm.
  double upper;
};

//! Compute the estimator of a norm of a function on a mesh hierarchy.
//!
//!\param [in] u Multilevel coefficients of the function.
//!\param [in] hierarchy Mesh hierarchy on which the function is defined.
//!\param [in] s Smoothness parameter for the norm estimator.
//!
//!\return Estimator for the norm of the function.
//!
//! If `s` is `+inf`, the estimator for the `L^inf` norm (supremum norm) is
//! calculated. Otherwise, the estimator for the '`s` norm' is calculated.
SandwichBounds estimator(const MultilevelCoefficients<double> u,
                         const MeshHierarchy &hierarchy, const double s);

} // namespace mgard

#endif
