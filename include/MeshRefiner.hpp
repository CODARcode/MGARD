#ifndef MESHREFINER_HPP
#define MESHREFINER_HPP
//!\file
//!\brief Function object which refines meshes.

#include "MeshLevel.hpp"

namespace mgard {

class MeshRefiner {
    public:
        MeshLevel operator()(const MeshLevel &mesh);

    private:
        virtual MeshLevel do_operator_parentheses(const MeshLevel &mesh) = 0;
};

}

#endif
