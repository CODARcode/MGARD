#ifndef ENUMERATEDMESHRANGE_HPP
#define ENUMERATEDMESHRANGE_HPP
//!\file
//!\brief Indexed meshes from a `MeshHierarchy`.

#include <vector>

#include "MeshHierarchy.hpp"
#include "MeshLevel.hpp"
#include "utilities.hpp"

namespace mgard {

//! Indexed mesh levels from a mesh hierarchy.
class EnumeratedMeshRange
    : public Enumeration<std::vector<MeshLevel>::const_iterator> {
public:
  //! Constructor.
  //!
  //!\param hierarchy Mesh hierarchy whose levels are to be iterated over.
  EnumeratedMeshRange(const MeshHierarchy &hierarchy);
};

} // namespace mgard

#endif
