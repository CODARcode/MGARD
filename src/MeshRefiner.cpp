#include "MeshRefiner.hpp"

namespace mgard {

MeshLevel MeshRefiner::operator()(const MeshLevel &mesh) {
    return do_operator_parentheses(mesh);
}

}
