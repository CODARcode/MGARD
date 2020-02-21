#include "indicators.hpp"

namespace mgard {

RatioBounds s_square_indicator_bounds(const MeshHierarchy &hierarchy) {
  const std::size_t d = hierarchy.meshes.front().topological_dimension;
  return {.realism = 1.f / ((d + 1) * (d + 2)), .reliability = 1.f / (d + 1)};
}

} // namespace mgard
