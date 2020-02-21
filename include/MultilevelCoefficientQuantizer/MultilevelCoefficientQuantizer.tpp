namespace mgard {

template <typename Real, typename Int>
MultilevelCoefficientQuantizer<Real, Int>::MultilevelCoefficientQuantizer(
    const MeshHierarchy &hierarchy, const float s, const float tolerance)
    : hierarchy(hierarchy), s(s), tolerance(tolerance) {}

template <typename Real, typename Int>
bool operator==(const MultilevelCoefficientQuantizer<Real, Int> &a,
                const MultilevelCoefficientQuantizer<Real, Int> &b) {
  return a.hierarchy == b.hierarchy && a.s == b.s && a.tolerance == b.tolerance;
}

template <typename Real, typename Int>
bool operator!=(const MultilevelCoefficientQuantizer<Real, Int> &a,
                const MultilevelCoefficientQuantizer<Real, Int> &b) {
  return !operator==(a, b);
}

template <typename Real, typename Int>
MultilevelCoefficientQuantizedRange<Real, Int>
MultilevelCoefficientQuantizer<Real, Int>(
    const MultilevelCoefficients<Real> u) const {
  return quantize(begin(hierarchy, u), end(hierarchy, u));
}

// TODO: provide definitions for `quantize` and `dequantize`.

} // namespace mgard
