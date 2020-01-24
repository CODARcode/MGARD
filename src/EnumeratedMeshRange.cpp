#include "EnumeratedMeshRange.hpp"

namespace mgard {

EnumeratedMeshRange::EnumeratedMeshRange(const MeshHierarchy &hierarchy)
    : Enumeration(hierarchy.meshes) {}

} // namespace mgard
