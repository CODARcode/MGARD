#include "unstructured/UniformRestriction.hpp"

#include "unstructured/UniformEdgeFamilies.hpp"

namespace mgard {

UniformRestriction::UniformRestriction(const MeshLevel &MESH,
                                       const MeshLevel &mesh)
    : LinearOperator(MESH.ndof(), mesh.ndof()), MESH(MESH), mesh(mesh) {}

void UniformRestriction::do_operator_parentheses(double const *const F,
                                                 double *const f) const {
  const moab::Range &nodes = mesh.entities[moab::MBVERTEX];
  const moab::Range &edges = mesh.entities[moab::MBEDGE];
  for (const moab::EntityHandle node : nodes) {
    f[mesh.index(node)] = F[MESH.index(node)];
  }
  // Could use `UniformMeshHierarchy::edge_families` here.
  const EdgeFamilyIterable<moab::Range::iterator> families =
      (EdgeFamilyIterable(mesh, MESH, edges.begin(), edges.end()));
  for (const EdgeFamily family : families) {
    const double contribution = 0.5 * F[MESH.index(family.midpoint)];
    for (const moab::EntityHandle endpoint : family.endpoints) {
      f[mesh.index(endpoint)] += contribution;
    }
  }
}

} // namespace mgard
