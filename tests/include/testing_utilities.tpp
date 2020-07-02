#include "catch2/catch.hpp"

#include <cassert>

template <typename T, typename U, typename SizeType>
void require_vector_equality(T p, U q, const SizeType N, const double margin) {
  bool all_close = true;
  for (SizeType i = 0; i < N; ++i) {
    all_close = all_close && *p++ == Approx(*q++).margin(margin);
  }
  REQUIRE(all_close);
}

template <typename T, typename U>
void require_vector_equality(const T &t, const U &u, const double margin) {
  const typename T::size_type N = t.size();
  const typename U::size_type M = u.size();
  assert(N == M);
  require_vector_equality(t.begin(), u.begin(), N, margin);
}

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
