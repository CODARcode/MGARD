#ifndef TESTING_RANDOM_HPP
#define TESTING_RANDOM_HPP

#include <array>
#include <random>

#include "TensorMeshHierarchy.hpp"

//! Random polynomial with the exponent of each variable in each term being
//! either zero or one.
template <typename Real, std::size_t N> class MultilinearPolynomial {
public:
  //! Constructor.
  //!
  //!\param generator Generator to use in generating the coefficients.
  //!\param distribution Distribution to use in generating the coefficients.
  MultilinearPolynomial(std::default_random_engine &generator,
                        std::uniform_real_distribution<Real> &distribution);

  //! Evaluate the polynomial at a point.
  Real operator()(const std::array<Real, N> &coordinates) const;

private:
  //! Coefficients of the constituent monomials. See `operator()` for the
  //! ordering.
  std::array<Real, 1 << N> coefficients;
};

//! Generate a mesh hierarchy with the given shape and random node positions.
//!
//!\param generator Generator to use in generating the node positions.
//!\param distribution Distribution to use in generating the node positions.
//!\param shape Desired shape of the finest level in the mesh hierarchy.
template <std::size_t N, typename Real>
mgard::TensorMeshHierarchy<N, Real> hierarchy_with_random_spacing(
    std::default_random_engine &generator,
    std::uniform_real_distribution<Real> &distribution,
    const std::array<std::size_t, N> shape);

//! Generate a function in a Sobolev space.
//!
//! See note in implementation file. Not to be used in important tasks.
//!
//!\param[in] hierarchy Mesh hierarchy on which the function will be defined.
//!\param[in] s Smoothness parameter to use in generating the function.
//!\param[in] generator Generator to use in generating the function.
//!\param[out] Buffer in which to store nodal values of generated function.
template <std::size_t N, typename Real>
void generate_reasonable_function(
    const mgard::TensorMeshHierarchy<N, Real> &hierarchy, const Real s,
    std::default_random_engine &generator, Real *const u);

#include "testing_random.tpp"
#endif
