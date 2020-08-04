#include <cmath>
#include <limits>
#include <stdexcept>

namespace mgard {

#define Qntzr TensorMultilevelCoefficientQuantizer

namespace {

template <std::size_t N, typename Real>
Real quantum(const TensorMeshHierarchy<N, Real> &hierarchy, const Real s,
             const Real tolerance) {
  if (s == std::numeric_limits<Real>::infinity()) {
    return (2 * tolerance) / ((hierarchy.L + 1) * (1 + std::pow(3, N)));
  } else {
    throw std::invalid_argument("only supremum norm thresholding implemented");
  }
}

} // namespace

template <std::size_t N, typename Real, typename Int>
Qntzr<N, Real, Int>::Qntzr(const TensorMeshHierarchy<N, Real> &hierarchy,
                           const Real s, const Real tolerance)
    : hierarchy(hierarchy), s(s), tolerance(tolerance),
      supremum_quantizer(quantum(hierarchy, s, tolerance)) {}

template <std::size_t N, typename Real, typename Int>
Int Qntzr<N, Real, Int>::
operator()(const SituatedCoefficient<N, Real> coefficient) const {
  if (s == std::numeric_limits<Real>::infinity()) {
    return supremum_quantizer(*coefficient.value);
  } else {
    throw std::invalid_argument("only supremum norm thresholding implemented");
  }
}

template <std::size_t N, typename Real, typename Int>
TensorQuantizedRange<N, Real, Int> Qntzr<N, Real, Int>::
operator()(Real *const u) const {
  const TensorLevelValues<N, Real> values = hierarchy.on_nodes(u, hierarchy.L);
  return TensorQuantizedRange<N, Real, Int>(*this, u);
}

template <std::size_t N, typename Real, typename Int>
bool operator==(const Qntzr<N, Real, Int> &a, const Qntzr<N, Real, Int> &b) {
  return a.hierarchy == b.hierarchy && a.s == b.s && a.tolerance == b.tolerance;
}

template <std::size_t N, typename Real, typename Int>
bool operator!=(const Qntzr<N, Real, Int> &a, const Qntzr<N, Real, Int> &b) {
  return !operator==(a, b);
}

template <std::size_t N, typename Real, typename Int>
TensorQuantizedRange<N, Real, Int>::TensorQuantizedRange(
    const Qntzr<N, Real, Int> &quantizer, Real *const u)
    : values(quantizer.hierarchy.on_nodes(u, quantizer.hierarchy.L)),
      begin_(quantizer, values.begin()), end_(quantizer, values.end()) {}

template <std::size_t N, typename Real, typename Int>
typename TensorMultilevelCoefficientQuantizer<N, Real, Int>::iterator
TensorQuantizedRange<N, Real, Int>::begin() const {
  return begin_;
}

template <std::size_t N, typename Real, typename Int>
typename TensorMultilevelCoefficientQuantizer<N, Real, Int>::iterator
TensorQuantizedRange<N, Real, Int>::end() const {
  return end_;
}

template <std::size_t N, typename Real, typename Int>
Qntzr<N, Real, Int>::iterator::iterator(
    const Qntzr &quantizer,
    const typename TensorLevelValues<N, Real>::iterator inner)
    : quantizer(quantizer), inner(inner) {}

template <std::size_t N, typename Real, typename Int>
bool Qntzr<N, Real, Int>::iterator::
operator==(const typename Qntzr<N, Real, Int>::iterator &other) const {
  return quantizer == other.quantizer && inner == other.inner;
}

template <std::size_t N, typename Real, typename Int>
bool Qntzr<N, Real, Int>::iterator::
operator!=(const typename Qntzr<N, Real, Int>::iterator &other) const {
  return !operator==(other);
}

template <std::size_t N, typename Real, typename Int>
typename Qntzr<N, Real, Int>::iterator &Qntzr<N, Real, Int>::iterator::
operator++() {
  ++inner;
  return *this;
}

template <std::size_t N, typename Real, typename Int>
typename Qntzr<N, Real, Int>::iterator Qntzr<N, Real, Int>::iterator::
operator++(int) {
  const iterator tmp = *this;
  operator++();
  return tmp;
}

template <std::size_t N, typename Real, typename Int>
typename Qntzr<N, Real, Int>::iterator::value_type
    Qntzr<N, Real, Int>::iterator::operator*() const {
  return quantizer(*inner);
}

#undef Qntzer

} // namespace mgard
