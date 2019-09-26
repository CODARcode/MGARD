#ifndef MESHREFINER_HPP
#define MESHREFINER_HPP
//!\file
//!\brief Function object which refines meshes.

#include "MeshLevel.hpp"

namespace mgard {

class MeshRefiner {
    public:
        //!Refine a mesh.
        //!
        //!\param mesh Mesh to be refined.
        //!
        //!\return Mesh obtained by refining the input mesh.
        MeshLevel operator()(const MeshLevel &mesh);

    private:
        //!Refine a mesh.
        //!
        //!\param mesh Mesh to be refined.
        //!
        //!\return Mesh obtained by refining the input mesh.
        virtual MeshLevel do_operator_parentheses(const MeshLevel &mesh) = 0;
};

}

#endif
