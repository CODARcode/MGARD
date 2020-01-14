#include <cmath>

#include <limits>
#include <stdexcept>

namespace mgard {

template <typename Real, typename Int>
Quantizer<Real, Int>::Quantizer(const Real quantum)
    : quantum(quantum),
      // We assume no overflows here.
      minimum(quantum * (std::numeric_limits<Int>::min() - 0.5)),
      maximum(quantum * (std::numeric_limits<Int>::max() + 0.5)) {
  if (quantum <= 0) {
    throw std::invalid_argument("quantum must be positive");
  }
}

template <typename Real, typename Int>
Int Quantizer<Real, Int>::quantize(const Real x) const {
  if (x <= minimum || x >= maximum) {
    throw std::domain_error("number too large to be quantized");
  }
  // See <https://www.cs.cmu.edu/~rbd/papers/cmj-float-to-int.html>.
  return std::copysign(0.5 + std::abs(x / quantum), x);
}

template <typename Real, typename Int>
template <typename It>
QuantizedRange<Real, Int, It>
Quantizer<Real, Int>::quantize(const It begin, const It end) const {
  return QuantizedRange<Real, Int, It>(*this, begin, end);
}

template <typename Real, typename Int>
Real Quantizer<Real, Int>::dequantize(const Int n) const {
  // We assume that all numbers of the form `quantum * n` are representable by
  //`Real`s.
  return quantum * n;
}

template <typename Real, typename Int>
template <typename It>
DequantizedRange<Real, Int, It>
Quantizer<Real, Int>::dequantize(const It begin, const It end) const {
  return DequantizedRange<Real, Int, It>(*this, begin, end);
}

template <typename Real, typename Int>
bool operator==(const Quantizer<Real, Int> &a, const Quantizer<Real, Int> &b) {
  return a.quantum == b.quantum;
}

template <typename Real, typename Int>
bool operator!=(const Quantizer<Real, Int> &a, const Quantizer<Real, Int> &b) {
  return !operator==(a, b);
}

} // namespace mgard
