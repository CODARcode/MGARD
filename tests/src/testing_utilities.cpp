#include "testing_utilities.hpp"

#include <cassert>

#include <stdexcept>

#include "moab/EntityType.hpp"

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
) {
    moab::ErrorCode ecode;

    moab::Range nodes;
    ecode = mbcore.create_vertices(coordinates, num_nodes, nodes);
    require_moab_success(ecode);
    assert(nodes.psize() == 1);

    moab::Range edges;
    for (std::size_t i = 0; i < num_edges; ++i) {
        //Local connectivity vector.
        std::vector<moab::EntityHandle> connectivity(2, nodes.front());
        for (std::size_t j = 0; j < 2; ++j) {
            connectivity.at(j) += edge_connectivity[2 * i + j];
        }
        moab::EntityHandle handle;
        ecode = mbcore.create_element(
            moab::MBEDGE, connectivity.data(), connectivity.size(), handle
        );
        require_moab_success(ecode);
        edges.insert(handle);
    }
    assert(edges.psize() == 1);

    moab::Range elements;
    //Nodes per element.
    std::size_t n;
    moab::EntityType element_type;
    switch (dimension) {
        case 2:
            n = 3;
            element_type = moab::MBTRI;
            break;
        case 3:
            n = 4;
            element_type = moab::MBTET;
            break;
        default:
            throw std::invalid_argument("`dimension` must be 2 or 3.");
    }
    for (std::size_t i = 0; i < num_elements; ++i) {
        //Local connectivity vector.
        std::vector<moab::EntityHandle> connectivity(n, nodes.front());
        for (std::size_t j = 0; j < n; ++j) {
            connectivity.at(j) += element_connectivity[n * i + j];
        }
        moab::EntityHandle handle;
        ecode = mbcore.create_element(
            element_type, connectivity.data(), connectivity.size(), handle
        );
        require_moab_success(ecode);
        elements.insert(handle);
    }
    assert(elements.psize() == 1);
    return mgard::MeshLevel(&mbcore, nodes, edges, elements);
}

void require_moab_success(const moab::ErrorCode ecode) {
    if (ecode != moab::MB_SUCCESS) {
        throw std::runtime_error("MOAB error encountered");
    }
}
