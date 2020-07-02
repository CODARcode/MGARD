#include <stdexcept>
#include <vector>

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
