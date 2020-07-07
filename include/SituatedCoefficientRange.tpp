namespace mgard {

template <typename Real>
SituatedCoefficientRange<Real>::SituatedCoefficientRange(
    const MeshHierarchy &hierarchy, const MultilevelCoefficients<Real> &u,
    const std::size_t l)
    : ZippedRange<moab::Range::const_iterator, Real const *>(
          hierarchy.new_nodes(l), hierarchy.on_new_nodes(u, l)) {}

} // namespace mgard
