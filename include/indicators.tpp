#include "moab/EntityType.hpp"

namespace mgard {

// The estimator squared is bounded above and below by these factors times the
// sum of the square indicator coefficients.
static RatioBounds s_square_indicator_bounds(const MeshLevel &mesh) {
  const std::size_t d = mesh.topological_dimension;
  return {.realism = 1 / static_cast<double>((d + 1) * (d + 2)),
          .reliability = 1 / static_cast<double>(d + 1)};
}

template <typename Real>
SandwichBounds square_indicator_coefficient(const IndicatorInput<Real> input,
                                            const float s) {
  const double square_unscaled =
      std::pow(2, 2 * s * input.l) *
      input.mesh.containing_elements_measure(input.node) * input.coefficient *
      input.coefficient;
  return SandwichBounds(s_square_indicator_bounds(input.mesh), square_unscaled);
}

} // namespace mgard
