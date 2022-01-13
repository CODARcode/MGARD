#ifndef ENUMERATEDMESHRANGE_HPP
#define ENUMERATEDMESHRANGE_HPP
//!\file
//!\brief Indexed meshes from a `mgard::MeshHierarchy`.

#include <vector>

#include "utilities.hpp"

#include "unstructured/MeshHierarchy.hpp"
#include "unstructured/MeshLevel.hpp"

namespace mgard {

//! Indexed mesh levels from a mesh hierarchy.
class EnumeratedMeshRange
    : public Enumeration<std::vector<MeshLevel>::const_iterator> {
public:
  //! Constructor.
  //!
  //!\param hierarchy Mesh hierarchy whose levels are to be iterated over.
  explicit EnumeratedMeshRange(const MeshHierarchy &hierarchy);
};

} // namespace mgard

#endif
