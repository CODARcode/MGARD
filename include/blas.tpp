#include <cmath>

namespace blas {

template <typename Real>
Real dotu(const std::size_t N, Real const *p, Real const *q) {
  Real product = 0;
  for (std::size_t i = 0; i < N; ++i) {
    product += *p++ * *q++;
  }
  return product;
}

template <typename Real> Real nrm2(const std::size_t N, Real const *p) {
  Real square_norm = 0;
  for (std::size_t i = 0; i < N; ++i) {
    const double value = *p++;
    square_norm += value * value;
  }
  return std::sqrt(square_norm);
}

template <typename Real>
void axpy(const std::size_t N, const Real alpha, Real const *p, Real *q) {
  for (std::size_t i = 0; i < N; ++i) {
    *q++ += alpha * *p++;
  }
}

template <typename Real>
void copy(const std::size_t N, Real const *p, Real *q) {
  for (std::size_t i = 0; i < N; ++i) {
    *q++ = *p++;
  }
}

template <typename Real>
void scal(const std::size_t N, Real const alpha, Real *p) {
  for (std::size_t i = 0; i < N; ++i) {
    *p++ *= alpha;
  }
}

} // namespace blas
