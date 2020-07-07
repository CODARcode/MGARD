#include <cmath>

#include "LinearQuantizer.hpp"
#include "indicators.hpp"

namespace mgard {

static float s_quantum(const MeshHierarchy &hierarchy, const float s,
                       const float tolerance, const IndicatorInput input) {
  const float square_factor = s_square_estimator_bounds(hierarchy).reliability *
                              s_square_indicator_bounds(hierarchy).reliability *
                              square_indicator_factor(input, s) *
                              hierarchy.ndof();
  // The maximum quantization error is half the quantum. Note that (assuming a
  // uniform distribution) the *expected* quantization error is a quarter of the
  // quantum, so we'll likely get half the error we're prepared to accept.
  return 2 * tolerance / std::sqrt(square_factor);
}

#define Qntzr MultilevelCoefficientQuantizer
#define Dqntzr MultilevelCoefficientDequantizer

template <typename Real, typename Int>
Qntzr<Real, Int>::Qntzr(const MeshHierarchy &hierarchy, const float s,
                        const float tolerance)
    : hierarchy(hierarchy), s(s), tolerance(tolerance),
      indicator_input_range(hierarchy) {}

template <typename Real, typename Int>
bool operator==(const Qntzr<Real, Int> &a, const Qntzr<Real, Int> &b) {
  return a.hierarchy == b.hierarchy && a.s == b.s && a.tolerance == b.tolerance;
}

template <typename Real, typename Int>
bool operator!=(const Qntzr<Real, Int> &a, const Qntzr<Real, Int> &b) {
  return !operator==(a, b);
}

template <typename Real, typename Int>
Int Qntzr<Real, Int>::operator()(const IndicatorInput input,
                                 const Real x) const {
  const float quantum = s_quantum(hierarchy, s, tolerance, input);
  return LinearQuantizer<Real, Int>(static_cast<Real>(quantum))(x);
}

template <typename Real, typename Int>
RangeSlice<typename Qntzr<Real, Int>::iterator> Qntzr<Real, Int>::
operator()(const MultilevelCoefficients<Real> u) const {
  Real const *const begin_mc = u.data;
  Real const *const end_mc = begin_mc + hierarchy.ndof();
  return {iterator(*this, indicator_input_range.begin(), begin_mc),
          iterator(*this, indicator_input_range.end(), end_mc)};
}

template <typename Real, typename Int>
Qntzr<Real, Int>::iterator::iterator(
    const Qntzr<Real, Int> &quantizer,
    const IndicatorInputRange::iterator inner_input, const Real *const inner_mc)
    : quantizer(quantizer), inner_input(inner_input), inner_mc(inner_mc) {}

template <typename Real, typename Int>
bool Qntzr<Real, Int>::iterator::
operator==(const Qntzr<Real, Int>::iterator &other) const {
  return quantizer == other.quantizer && inner_input == other.inner_input &&
         inner_mc == other.inner_mc;
}

template <typename Real, typename Int>
bool Qntzr<Real, Int>::iterator::
operator!=(const Qntzr<Real, Int>::iterator &other) const {
  return !operator==(other);
}

template <typename Real, typename Int>
typename Qntzr<Real, Int>::iterator &Qntzr<Real, Int>::iterator::operator++() {
  ++inner_input;
  ++inner_mc;
  return *this;
}

template <typename Real, typename Int>
typename Qntzr<Real, Int>::iterator Qntzr<Real, Int>::iterator::
operator++(int) {
  const iterator tmp = *this;
  operator++();
  return tmp;
}

template <typename Real, typename Int>
Int Qntzr<Real, Int>::iterator::operator*() const {
  return quantizer(*inner_input, *inner_mc);
}

template <typename Int, typename Real>
Dqntzr<Int, Real>::Dqntzr(const MeshHierarchy &hierarchy, const float s,
                          const float tolerance)
    : hierarchy(hierarchy), s(s), tolerance(tolerance),
      indicator_input_range(hierarchy) {}

template <typename Int, typename Real>
bool operator==(const Dqntzr<Int, Real> &a, const Dqntzr<Int, Real> &b) {
  return a.hierarchy == b.hierarchy && a.s == b.s && a.tolerance == b.tolerance;
}

template <typename Int, typename Real>
bool operator!=(const Dqntzr<Int, Real> &a, const Dqntzr<Int, Real> &b) {
  return !operator==(a, b);
}

template <typename Int, typename Real>
Real Dqntzr<Int, Real>::operator()(const IndicatorInput input,
                                   const Int n) const {
  const float quantum = s_quantum(hierarchy, s, tolerance, input);
  return LinearDequantizer<Int, Real>(static_cast<Real>(quantum))(n);
}

template <typename Int, typename Real>
template <typename It>
RangeSlice<typename Dqntzr<Int, Real>::template iterator<It>>
Dqntzr<Int, Real>::operator()(const It begin, const It end) const {
  return {iterator<It>(*this, indicator_input_range.begin(), begin),
          iterator<It>(*this, indicator_input_range.end(), end)};
}

template <typename Int, typename Real>
template <typename It>
Dqntzr<Int, Real>::iterator<It>::iterator(
    const Dqntzr<Int, Real> &dequantizer,
    const IndicatorInputRange::iterator inner_input, const It inner_qc)
    : dequantizer(dequantizer), inner_input(inner_input), inner_qc(inner_qc) {}

template <typename Int, typename Real>
template <typename It>
bool Dqntzr<Int, Real>::iterator<It>::
operator==(const Dqntzr<Int, Real>::iterator<It> &other) const {
  return dequantizer == other.dequantizer && inner_input == other.inner_input &&
         inner_qc == other.inner_qc;
}

template <typename Int, typename Real>
template <typename It>
bool Dqntzr<Int, Real>::iterator<It>::
operator!=(const Dqntzr<Int, Real>::iterator<It> &other) const {
  return !operator==(other);
}

template <typename Int, typename Real>
template <typename It>
typename Dqntzr<Int, Real>::template iterator<It> &
Dqntzr<Int, Real>::iterator<It>::operator++() {
  ++inner_input;
  ++inner_qc;
  return *this;
}

template <typename Int, typename Real>
template <typename It>
typename Dqntzr<Int, Real>::template iterator<It>
Dqntzr<Int, Real>::iterator<It>::operator++(int) {
  const iterator tmp = *this;
  operator++();
  return tmp;
}

template <typename Int, typename Real>
template <typename It>
Real Dqntzr<Int, Real>::iterator<It>::operator*() const {
  return dequantizer(*inner_input, *inner_qc);
}

#undef Qntzr
#undef Dqntzr

} // namespace mgard
