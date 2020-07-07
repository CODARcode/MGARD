#ifndef UNIFORMMESHREFINER_HPP
#define UNIFORMMESHREFINER_HPP
//!\file
//!\brief Function object which uniformly refines meshes.

#include "moab/Range.hpp"
#include "moab/Types.hpp"

#include "MeshLevel.hpp"
#include "MeshRefiner.hpp"

namespace mgard {

//! Function object that uniformly refines meshes.
class UniformMeshRefiner : public MeshRefiner {
private:
  virtual MeshLevel do_operator_parentheses(const MeshLevel &mesh) override;

  //! Bisect the edges of the old mesh, producing for each old mesh one new
  //! node and two new edges.
  //!
  //!\param [in] mesh Mesh being refined.
  //!\param [out] NODES Nodes of new mesh.
  //!\param [out] EDGES Edges of new mesh.
  //!
  //!`NODES` and `EDGES` should initially be empty.
  moab::ErrorCode bisect_edges(const MeshLevel &mesh, moab::Range &NODES,
                               moab::Range &EDGES);

  //! Quadrisect the triangles of the old mesh.
  //!
  //!\param [in] mesh Mesh being refined.
  //!\param [in] NODES Nodes of new mesh.
  //!\param [in, out] EDGES Edges of new mesh.
  //!\param [out] ELEMENTS Elements of new mesh.
  //!
  //! Three edges will be added for each 'interior' triangle.
  moab::ErrorCode quadrisect_triangles(const MeshLevel &mesh,
                                       const moab::Range &NODES,
                                       moab::Range &EDGES,
                                       moab::Range &ELEMENTS);

  //! Octasect the tetrahedra of the old mesh using the technique given in
  //![Ong's 1994 paper][Ong3D].
  //!
  //![Ong3D]: https://doi.org/10.1137/0915070
  //!
  //!\param [in] mesh Mesh being refined.
  //!\param [in] NODES Nodes of new mesh.
  //!\param [in, out] EDGES Edges of new mesh.
  //!\param [out] ELEMENTS Elements of new mesh.
  //!
  //! Thirteen edges will be added for each 'interior' octahedron.
  moab::ErrorCode octasect_tetrahedra(const MeshLevel &mesh,
                                      const moab::Range &NODES,
                                      moab::Range &EDGES,
                                      moab::Range &ELEMENTS);

  // Could alternatively use `\cite` there.
};

} // namespace mgard

#endif
