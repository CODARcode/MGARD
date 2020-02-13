#include <cmath>

#include <limits>
#include <stdexcept>

namespace mgard {

template <typename Real, typename Int>
LinearQuantizer<Real, Int>::LinearQuantizer(const Real quantum)
    : quantum(quantum),
      // We assume no overflows here.
      minimum(quantum * (std::numeric_limits<Int>::min() - 0.5)),
      maximum(quantum * (std::numeric_limits<Int>::max() + 0.5)) {
  if (quantum <= 0) {
    throw std::invalid_argument("quantum must be positive");
  }
}

template <typename Real, typename Int>
Int LinearQuantizer<Real, Int>::operator()(const Real x) const {
  if (x <= minimum || x >= maximum) {
    throw std::domain_error("number too large to be quantized");
  }
  // See <https://www.cs.cmu.edu/~rbd/papers/cmj-float-to-int.html>.
  return std::copysign(0.5 + std::abs(x / quantum), x);
}

template <typename Real, typename Int>
bool operator==(const LinearQuantizer<Real, Int> &a,
                const LinearQuantizer<Real, Int> &b) {
  return a.quantum == b.quantum;
}

template <typename Real, typename Int>
bool operator!=(const LinearQuantizer<Real, Int> &a,
                const LinearQuantizer<Real, Int> &b) {
  return !operator==(a, b);
}

template <typename Int, typename Real>
LinearDequantizer<Int, Real>::LinearDequantizer(const Real quantum)
    : quantum(quantum) {
  if (quantum <= 0) {
    throw std::invalid_argument("quantum must be positive");
  }
}

template <typename Int, typename Real>
Real LinearDequantizer<Int, Real>::operator()(const Int n) const {
  // We assume that all numbers of the form `quantum * n` are representable by
  //`Real`s.
  return quantum * n;
}

template <typename Int, typename Real>
bool operator==(const LinearDequantizer<Int, Real> &a,
                const LinearDequantizer<Int, Real> &b) {
  return a.quantum == b.quantum;
}

template <typename Int, typename Real>
bool operator!=(const LinearDequantizer<Int, Real> &a,
                const LinearDequantizer<Int, Real> &b) {
  return !operator==(a, b);
}

} // namespace mgard
