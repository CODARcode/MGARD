namespace mgard {

template <typename Int>
MultilevelCoefficientQuantizer<Int>::MultilevelCoefficientQuantizer(
    const MeshHierarchy &hierarchy, const double s, const double tolerance)
    : hierarchy(hierarchy), s(s), tolerance(tolerance) {}

template <typename Int>
bool operator==(const MultilevelCoefficientQuantizer<Int> &a,
                const MultilevelCoefficientQuantizer<Int> &b) {
  return a.hierarchy == b.hierarchy && a.s == b.s && a.tolerance == b.tolerance;
}

template <typename Int>
bool operator!=(const MultilevelCoefficientQuantizer<Int> &a,
                const MultilevelCoefficientQuantizer<Int> &b) {
  return !operator==(a, b);
}

template <typename Int>
MultilevelCoefficientQuantizedRange<Int> MultilevelCoefficientQuantizer<Int>(
    const MultilevelCoefficients<double> u) const {
  return quantize(u.data, u.data + hierarchy.ndof());
}

// TODO: provide definitions for `quantize` and `dequantize`.

} // namespace mgard
