#include <cstddef>

#include "moab/Interface.hpp"

#include "MeshLevel.hpp"

mgard::MeshLevel make_mesh_level(
    moab::Interface &mbcore,
    const std::size_t num_nodes,
    const std::size_t num_edges,
    const std::size_t num_elements,
    const std::size_t dimension,
    double const * const coordinates,
    //The elements of both this and `element_connectivity` are just offsets, to
    //be added to the `moab::EntityHandle` of the first vertex.
    std::size_t  const * const edge_connectivity,
    std::size_t const * const element_connectivity
);

void require_moab_success(const moab::ErrorCode ecode);
