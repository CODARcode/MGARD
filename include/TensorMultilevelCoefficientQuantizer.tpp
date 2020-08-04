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
    // The maximum error is half the quantizer.
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
      nodes(hierarchy.nodes(hierarchy.L)),
      supremum_quantizer(quantum(hierarchy, s, tolerance)) {}

template <std::size_t N, typename Real, typename Int>
Int Qntzr<N, Real, Int>::operator()(const TensorNode<N, Real>,
                                    const Real coefficient) const {
  // TODO: Look into moving this test outside of the operator.
  if (s == std::numeric_limits<Real>::infinity()) {
    return supremum_quantizer(coefficient);
  } else {
    throw std::invalid_argument("only supremum norm thresholding implemented");
  }
}

template <std::size_t N, typename Real, typename Int>
RangeSlice<typename Qntzr<N, Real, Int>::iterator> Qntzr<N, Real, Int>::
operator()(Real *const u) const {
  return {.begin_ = iterator(*this, nodes.begin(), u),
          .end_ = iterator(*this, nodes.end(), u + hierarchy.ndof())};
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
Qntzr<N, Real, Int>::iterator::iterator(
    const Qntzr &quantizer,
    const typename TensorNodeRange<N, Real>::iterator inner_node,
    Real const *const inner_coeff)
    : quantizer(quantizer), inner_node(inner_node), inner_coeff(inner_coeff) {}

template <std::size_t N, typename Real, typename Int>
bool Qntzr<N, Real, Int>::iterator::
operator==(const typename Qntzr<N, Real, Int>::iterator &other) const {
  return quantizer == other.quantizer && inner_node == other.inner_node &&
         inner_coeff == other.inner_coeff;
}

template <std::size_t N, typename Real, typename Int>
bool Qntzr<N, Real, Int>::iterator::
operator!=(const typename Qntzr<N, Real, Int>::iterator &other) const {
  return !operator==(other);
}

template <std::size_t N, typename Real, typename Int>
typename Qntzr<N, Real, Int>::iterator &Qntzr<N, Real, Int>::iterator::
operator++() {
  ++inner_node;
  ++inner_coeff;
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
  return quantizer(*inner_node, *inner_coeff);
}

#undef Qntzer

} // namespace mgard
