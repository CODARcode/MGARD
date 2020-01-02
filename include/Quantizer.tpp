#include <cmath>

#include <limits>
#include <stdexcept>

namespace mgard {

template <typename Real, typename Int>
Quantizer<Real, Int>::Quantizer(const Real quantum):
    quantum(quantum),
    //We assume no overflows here.
    minimum(quantum * (std::numeric_limits<Int>::min() - 0.5)),
    maximum(quantum * (std::numeric_limits<Int>::max() + 0.5))
{
    if (quantum <= 0) {
        throw std::invalid_argument("quantum must be positive");
    }
}

template <typename Real, typename Int>
Int Quantizer<Real, Int>::quantize(const Real x) const {
    if (x <= minimum || x >= maximum) {
        throw std::domain_error("number too large to be quantized");
    }
    return do_quantize(x);
}

template <typename Real, typename Int>
Real Quantizer<Real, Int>::dequantize(const Int n) const {
    //We assume that all numbers of the form `quantum * n` are representable by
    //`Real`s.
    return do_dequantize(n);
}

template <typename Real, typename Int>
Int Quantizer<Real, Int>::do_quantize(const Real x) const {
    //See <https://www.cs.cmu.edu/~rbd/papers/cmj-float-to-int.html>.
    return std::copysign(0.5 + std::abs(x / quantum), x);
}

template <typename Real, typename Int>
Real Quantizer<Real, Int>::do_dequantize(const Int n) const {
    return quantum * n;
}

}
