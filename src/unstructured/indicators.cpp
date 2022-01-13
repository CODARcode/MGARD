#include "unstructured/indicators.hpp"

#include <cmath>

namespace mgard {

RatioBounds s_square_indicator_bounds(const MeshHierarchy &hierarchy) {
  const std::size_t d = hierarchy.meshes.front().topological_dimension;
  return {.realism = 1.f / ((d + 1) * (d + 2)), .reliability = 1.f / (d + 1)};
}

float square_indicator_factor(const IndicatorInput input, const float s) {
  return std::exp2(2 * s * input.l) *
         input.mesh.containing_elements_measure(input.node);
}

} // namespace mgard
