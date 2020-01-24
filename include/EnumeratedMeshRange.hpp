#ifndef ENUMERATEDMESHRANGE_HPP
#define ENUMERATEDMESHRANGE_HPP
//!\file
//!\brief Indexed meshes from a `MeshHierarchy`.

#include <vector>

#include "MeshHierarchy.hpp"
#include "MeshLevel.hpp"
#include "utilities.hpp"

namespace mgard {

class EnumeratedMeshRange : public Enumeration<std::vector<MeshLevel>> {
public:
  //! Constructor.
  //!
  //!\param hierarchy Mesh hierarchy whose levels are to be iterated over.
  EnumeratedMeshRange(const MeshHierarchy &hierarchy);
};

} // namespace mgard

#endif
