#include <cmath>

#include <algorithm>
#include <limits>

namespace mgard {

#define Qntzr TensorMultilevelCoefficientQuantizer

namespace {

template <std::size_t N, typename Real>
Real supremum_quantum(const TensorMeshHierarchy<N, Real> &hierarchy,
                      const Real tolerance) {
  // The maximum error is half the quantizer.
  return (2 * tolerance) / ((hierarchy.L + 1) * (1 + std::pow(3, N)));
}

template <std::size_t N, typename Real>
Real s_quantum(const TensorMeshHierarchy<N, Real> &hierarchy, const Real s,
               const Real tolerance, const TensorNode<N, Real> node) {
  Real volume_factor = 1;
  for (std::size_t i = 0; i < N; ++i) {
    const TensorIndexRange indices = hierarchy.indices(node.l, i);
    // `TensorNode` doesn't immediately give us a way of accessing the
    // neighboring nodes in the level, so we have to search.
    const TensorIndexRange::iterator p =
        std::find(indices.begin(), indices.end(), node.multiindex.at(i));
    const std::vector<Real> &coordinates = hierarchy.coordinates.at(i);
    const Real x = node.coordinates.at(i);

    // Temporary copy of `p` that can be incremented and decremented.
    TensorIndexRange::iterator q;

    q = p;
    const Real h_left = q == indices.begin() ? 0 : x - coordinates.at(*--q);

    q = p;
    const Real h_right = ++q == indices.end() ? 0 : coordinates.at(*q) - x;

    volume_factor *= (h_left + h_right) / 2;
  }
  // The maximum error is half the quantizer.
  return (2 * tolerance) /
         (std::exp2(s * node.l) * std::sqrt(hierarchy.ndof() * volume_factor));
}

} // namespace

template <std::size_t N, typename Real, typename Int>
Qntzr<N, Real, Int>::Qntzr(const TensorMeshHierarchy<N, Real> &hierarchy,
                           const Real s, const Real tolerance)
    : hierarchy(hierarchy), s(s), tolerance(tolerance),
      nodes(hierarchy.nodes(hierarchy.L)),
      supremum_quantizer(supremum_quantum(hierarchy, tolerance)) {}

template <std::size_t N, typename Real, typename Int>
Int Qntzr<N, Real, Int>::operator()(const TensorNode<N, Real> node,
                                    const Real coefficient) const {
  // TODO: Look into moving this test outside of the operator.
  if (s == std::numeric_limits<Real>::infinity()) {
    return supremum_quantizer(coefficient);
  } else {
    const LinearQuantizer<Real, Int> quantizer(
        s_quantum(hierarchy, s, tolerance, node));
    return quantizer(coefficient);
  }
}

template <std::size_t N, typename Real, typename Int>
RangeSlice<typename Qntzr<N, Real, Int>::iterator> Qntzr<N, Real, Int>::
operator()(Real const *const u) const {
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
  return (&quantizer == &other.quantizer || quantizer == other.quantizer) &&
         inner_node == other.inner_node && inner_coeff == other.inner_coeff;
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
Int Qntzr<N, Real, Int>::iterator::operator*() const {
  return quantizer(*inner_node, *inner_coeff);
}

#undef Qntzer
#define Dqntzr TensorMultilevelCoefficientDequantizer

template <std::size_t N, typename Int, typename Real>
Dqntzr<N, Int, Real>::Dqntzr(const TensorMeshHierarchy<N, Real> &hierarchy,
                             const Real s, const Real tolerance)
    : hierarchy(hierarchy), s(s), tolerance(tolerance),
      nodes(hierarchy.nodes(hierarchy.L)),
      supremum_dequantizer(supremum_quantum(hierarchy, tolerance)) {}

template <std::size_t N, typename Int, typename Real>
Real Dqntzr<N, Int, Real>::operator()(const TensorNode<N, Real> node,
                                      const Int n) const {
  // TODO: Look into moving this test outside of the operator.
  if (s == std::numeric_limits<Real>::infinity()) {
    return supremum_dequantizer(n);
  } else {
    const LinearDequantizer<Int, Real> dequantizer(
        s_quantum(hierarchy, s, tolerance, node));
    return dequantizer(n);
  }
}

template <std::size_t N, typename Int, typename Real>
template <typename It>
RangeSlice<typename Dqntzr<N, Int, Real>::template iterator<It>>
Dqntzr<N, Int, Real>::operator()(const It begin, const It end) const {
  return {.begin_ = iterator(*this, nodes.begin(), begin),
          .end_ = iterator(*this, nodes.end(), end)};
}

template <std::size_t N, typename Int, typename Real>
bool operator==(const Dqntzr<N, Int, Real> &a, const Dqntzr<N, Int, Real> &b) {
  return a.hierarchy == b.hierarchy && a.s == b.s && a.tolerance == b.tolerance;
}

template <std::size_t N, typename Int, typename Real>
bool operator!=(const Dqntzr<N, Int, Real> &a, const Dqntzr<N, Int, Real> &b) {
  return !operator==(a, b);
}

template <std::size_t N, typename Int, typename Real>
template <typename It>
Dqntzr<N, Int, Real>::iterator<It>::iterator(
    const Dqntzr<N, Int, Real> &dequantizer,
    const typename TensorNodeRange<N, Real>::iterator inner_node,
    const It inner_coeff)
    : dequantizer(dequantizer), inner_node(inner_node),
      inner_coeff(inner_coeff) {}

template <std::size_t N, typename Int, typename Real>
template <typename It>
bool Dqntzr<N, Int, Real>::iterator<It>::iterator::operator==(
    const typename Dqntzr<N, Int, Real>::template iterator<It>::iterator &other)
    const {
  return (&dequantizer == &other.dequantizer ||
          dequantizer == other.dequantizer) &&
         inner_node == other.inner_node && inner_coeff == other.inner_coeff;
}

template <std::size_t N, typename Int, typename Real>
template <typename It>
bool Dqntzr<N, Int, Real>::iterator<It>::iterator::operator!=(
    const typename Dqntzr<N, Int, Real>::template iterator<It>::iterator &other)
    const {
  return !operator==(other);
}

template <std::size_t N, typename Int, typename Real>
template <typename It>
typename Dqntzr<N, Int, Real>::template iterator<It>::iterator &
Dqntzr<N, Int, Real>::iterator<It>::iterator::operator++() {
  ++inner_node;
  ++inner_coeff;
  return *this;
}

template <std::size_t N, typename Int, typename Real>
template <typename It>
typename Dqntzr<N, Int, Real>::template iterator<It>::iterator
Dqntzr<N, Int, Real>::iterator<It>::iterator::operator++(int) {
  const iterator tmp = *this;
  operator++();
  return tmp;
}

template <std::size_t N, typename Int, typename Real>
template <typename It>
Real Dqntzr<N, Int, Real>::iterator<It>::iterator::operator*() const {
  return dequantizer(*inner_node, *inner_coeff);
}

#undef Dqntzr

} // namespace mgard
