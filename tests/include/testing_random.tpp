#include <cmath>

#include <stdexcept>
#include <utility>
#include <vector>

#include "blas.hpp"

#include "testing_utilities.hpp"

template <typename Real, std::size_t N>
MultilinearPolynomial<Real, N>::MultilinearPolynomial(
    std::default_random_engine &generator,
    std::uniform_real_distribution<Real> &distribution) {
  for (Real &coefficient : coefficients) {
    coefficient = distribution(generator);
  }
}

template <typename Real, std::size_t N>
Real MultilinearPolynomial<Real, N>::
operator()(const std::array<Real, N> &coordinates) const {
  Real value = 0;
  //! We use `exponents` to iterate over the monomials. Its bits determine the
  //! exponents of the variables (and so the monomial).
  std::size_t exponents = 0;
  for (const Real coefficient : coefficients) {
    Real term = coefficient;
    // Could check that `std::size_t` is sufficiently wide.
    for (std::size_t i = 0; i < N; ++i) {
      if (exponents & (1 << i)) {
        term *= coordinates.at(i);
      }
      value += term;
    }
    ++exponents;
  }
  return value;
}

namespace {

//! Random element of a Sobolev space.
//
// This is just meant to be something better than randomly generating nodal
// values. No guarantees that it's perfectly correct.
template <typename Real, std::size_t N> class SobolevFunction {
public:
  //! Constructor.
  //!
  //!\param s Smoothness parameter. Must be finite and nonnegative.
  //!\param Generator to use in generating the function.
  SobolevFunction(const Real s, std::default_random_engine &generator);

  //! Evaluate the function at a point.
  Real operator()(const std::array<Real, N> &coordinates) const;

private:
  //! Frequency and coefficient for each sinusoid that makes up the function.
  std::vector<std::pair<std::array<Real, N>, Real>> modes;
};

template <typename Real, std::size_t N>
SobolevFunction<Real, N>::SobolevFunction(
    const Real s, std::default_random_engine &generator) {
  if (s < 0 || s == std::numeric_limits<Real>::infinity()) {
    throw std::invalid_argument(
        "smoothness parameter must be finite and nonnegative");
  }
  std::uniform_real_distribution<Real> coefficient_distribution(-1, 1);

  // DC component.
  {
    std::array<Real, N> frequency;
    frequency.fill(0);
    modes.push_back({frequency, coefficient_distribution(generator)});
  }

  static_assert(N, "dimension cannot be zero");
  std::uniform_int_distribution<std::size_t> index_distribution(0, N - 1);
  for (std::size_t n = 1;; ++n) {
    std::uniform_int_distribution<int> frequency_distribution(
        -static_cast<int>(n), n);
    bool maximum_reached = false;
    std::array<Real, N> frequency;
    for (std::size_t i = 0; i < N; ++i) {
      maximum_reached =
          std::abs(frequency.at(i) = frequency_distribution(generator)) == n ||
          maximum_reached;
    }
    //! In theory it might be better to not exclude the possibility of multiple
    //! entries being `n`.
    if (!maximum_reached) {
      frequency.at(index_distribution(generator)) =
          coefficient_distribution(generator) > 0 ? n : -static_cast<Real>(n);
    }

    const Real frequency_square_norm = blas::nrm2(N, frequency.data());
    const Real coefficient = std::exp2(-0.5 * n) *
                             std::pow(frequency_square_norm, -0.5 * s) *
                             coefficient_distribution(generator);
    modes.push_back({frequency, coefficient});

    // The mode contributes to the square norm according to the *square* of
    // the coefficient, so this cutoff doesn't need to be terribly small.
    if (std::abs(coefficient) < 1e-4) {
      break;
    }
  }
}

template <typename Real, std::size_t N>
Real SobolevFunction<Real, N>::
operator()(const std::array<Real, N> &coordinates) const {
  Real value = 0;
  for (const auto [frequency, coefficient] : modes) {
    value += coefficient *
             std::sin(blas::dotu(N, frequency.data(), coordinates.data()));
  }
  return value;
}

} // namespace

template <std::size_t N, typename Real>
mgard::TensorMeshHierarchy<N, Real> hierarchy_with_random_spacing(
    std::default_random_engine &generator,
    std::uniform_real_distribution<Real> &distribution,
    const std::array<std::size_t, N> shape) {
  if (distribution.a() <= 0) {
    throw std::invalid_argument(
        "node spacing should be bounded away from zero");
  }
  std::array<std::vector<Real>, N> coordinates;
  for (std::size_t i = 0; i < N; ++i) {
    const std::size_t n = shape.at(i);
    std::vector<Real> &xs = coordinates.at(i);
    xs.resize(n);
    Real previous;
    previous = xs.at(0) = 0;
    for (std::size_t j = 1; j < n; ++j) {
      previous = xs.at(j) = previous + distribution(generator);
    }
  }
  return mgard::TensorMeshHierarchy<N, Real>(shape, coordinates);
}

template <std::size_t N, typename Real>
void generate_reasonable_function(
    const mgard::TensorMeshHierarchy<N, Real> &hierarchy, const Real s,
    std::default_random_engine &generator, Real *const u) {
  const SobolevFunction<Real, N> f(s, generator);
  for (const mgard::TensorNode<N, Real> node : hierarchy.nodes(hierarchy.L)) {
    hierarchy.at(u, node.multiindex) = f(coordinates(hierarchy, node));
  }
}
