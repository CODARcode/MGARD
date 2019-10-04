#ifndef UNIFORMMESHREFINER_HPP
#define UNIFORMMESHREFINER_HPP
//!\file
//!\brief Function object which uniformly refines meshes.

#include "moab/Types.hpp"
#include "moab/Range.hpp"

#include "MeshRefiner.hpp"
#include "MeshLevel.hpp"

namespace mgard {

//!Function object that uniformly refines meshes.
class UniformMeshRefiner: public MeshRefiner {
    private:
        virtual MeshLevel do_operator_parentheses(
            const MeshLevel &mesh
        ) override;

        //!Bisect the edges of the old mesh, producing for each old mesh one new
        //!node and two new edges.
        //!
        //!\param [in] mesh Mesh being refined.
        //!\param [out] NODES Nodes of new mesh.
        //!\param [out] EDGES Edges of new mesh.
        //!
        //!`NODES` and `EDGES` should initially be empty.
        moab::ErrorCode bisect_edges(
            const MeshLevel &mesh,
            moab::Range &NODES,
            moab::Range &EDGES
        );

        //!Quadrisect the triangles of the old mesh.
        //!
        //\param [in] mesh Mesh being refined.
        //!\param [in] NODES Nodes of new mesh.
        //!\param [in, out] EDGES Edges of new mesh.
        //!\param [out] ELEMENTS Elements of new mesh.
        //!
        //!Three edges will be added for each 'interior' triangle.
        moab::ErrorCode quadrisect_triangles(
            const MeshLevel &mesh,
            const moab::Range &NODES,
            moab::Range &EDGES,
            moab::Range &ELEMENTS
        );
};

}

#endif
