#include "UniformMeshRefiner.hpp"

#include <cstddef>

#include <unordered_map>

#include "blaspp/blas.hh"

#include "utilities.hpp"

//!Find the node of a triangle not included in an edge.
//!
//!\param triangle_connectivity Nodes of the triangle. Must have length 3.
//!\param edge_connectivity Nodes of the edge. Must have length 2.
//!
//!\return Node opposite the edge.
static moab::EntityHandle find_node_opposite_edge(
    const helpers::PseudoArray<const moab::EntityHandle> triangle_connectivity,
    const helpers::PseudoArray<const moab::EntityHandle> edge_connectivity
) {
    moab::EntityHandle const * const _begin = edge_connectivity.begin();
    moab::EntityHandle const * const _end = edge_connectivity.end();
    for (moab::EntityHandle x : triangle_connectivity) {
        //Look for this node in the edge.
        moab::EntityHandle const * const p = std::find(_begin, _end, x);
        //If we don't find it, the node is opposite the edge.
        if (p == _end) {
            return x;
        }
    }
    throw std::invalid_argument(
        "triangle nodes somehow contained in edge nodes"
    );
}

namespace mgard {

MeshLevel UniformMeshRefiner::do_operator_parentheses(const MeshLevel &mesh) {
    moab::ErrorCode ecode;

    moab::Range NODES;
    moab::Range EDGES;
    ecode = bisect_edges(mesh, NODES, EDGES);
    if (ecode != moab::MB_SUCCESS) {
        throw std::runtime_error("failed to bisect edges");
    }

    moab::Range ELEMENTS;
    switch (mesh.topological_dimension) {
        case 2:
            ecode = quadrisect_triangles(mesh, NODES, EDGES, ELEMENTS);
            if (ecode != moab::MB_SUCCESS) {
                throw std::runtime_error("failed to quadrisect triangles");
            }
            break;
        case 3:
            throw std::logic_error("tetrahedron refinement not implemented");
            break;
        default:
            throw std::logic_error("unsupported topological dimension");
    }
    return MeshLevel(mesh.impl, NODES, EDGES, ELEMENTS);
}

moab::ErrorCode UniformMeshRefiner::bisect_edges(
    const MeshLevel &mesh, moab::Range &NODES, moab::Range &EDGES
) {
    moab::ErrorCode ecode;
    assert(NODES.empty());
    assert(EDGES.empty());
    const moab::Range &nodes = mesh.entities[moab::MBVERTEX];
    const moab::Range &edges = mesh.entities[moab::MBEDGE];
    const std::size_t nnodes = mesh.ndof();
    const std::size_t nedges = edges.size();
    //This could be made cleaner by using something like
    //`std::vector<double [3]>`, but I don't want to spend a bunch of time
    //looking into initialization and contiguity now.
    //TODO: I believe it will be automatically initalized to zero. Check.
    std::vector<double> midpoint_coordinates(3 * nedges, 0);
    //Similarly.
    std::vector<double> endpoint_coordinates(3 * nnodes);
    ecode = mesh.impl.get_coords(nodes, endpoint_coordinates.data());

    moab::Range NEW_NODES;
    //We'll reset the coordinates below. Creating the nodes now lets us iterate
    //over the edges just once, finding the coordinates of the midpoints and
    //creating the new edges in the same loop.
    ecode = mesh.impl.create_vertices(
        midpoint_coordinates.data(), nedges, NEW_NODES
    );
    //From the MOAB documentation:
    //> Entities allocated in sequence will typically have contiguous handles
    //Checking that.
    assert(NEW_NODES.psize() == 1);
    assert(NEW_NODES.front() == nodes.back() + 1);
    assert(NEW_NODES.back() == NEW_NODES.front() + nedges - 1);

    //We'll check that the new edges have contiguous handles as we go and
    //construct the new edge range at the end. (We don't need `EDGES.front()` to
    //follow `edges.back()`, but it should and anyway it'll make the contiguity
    //check a little simpler.)
    moab::EntityHandle most_recent_edge = edges.back();
    for (std::size_t i = 0; i < nedges; ++i) {
        const moab::EntityHandle edge = edges[i];
        const moab::EntityHandle midpoint = NEW_NODES[i];
        moab::EntityHandle EDGE_CONNECTIVITY[2];
        EDGE_CONNECTIVITY[0] = midpoint;
        double * const p = midpoint_coordinates.data() + 3 * i;
        double * const q = endpoint_coordinates.data();
        for (const moab::EntityHandle endpoint : mesh.connectivity(edge)) {
            EDGE_CONNECTIVITY[1] = endpoint;
            moab::EntityHandle EDGE;
            ecode = mesh.impl.create_element(
                moab::MBEDGE, EDGE_CONNECTIVITY, 2, EDGE
            );
            assert(EDGE == ++most_recent_edge);
            blas::axpy(3, 1, q + 3 * mesh.index(endpoint), 1, p, 1);
        }
        blas::scal(3, 0.5, p, 1);
    }
    ecode = mesh.impl.set_coords(NEW_NODES, midpoint_coordinates.data());
    MB_CHK_ERR(ecode);

    NODES.insert(nodes.front(), NEW_NODES.back());
    assert(NODES.psize() == 1);

    EDGES.insert(edges.back() + 1, most_recent_edge);
    assert(EDGES.psize() == 1);
    assert(EDGES.size() == 2 * nedges);
    return moab::MB_SUCCESS;
}

moab::ErrorCode UniformMeshRefiner::quadrisect_triangles(
    const MeshLevel &mesh,
    const moab::Range &NODES,
    moab::Range &EDGES,
    moab::Range &ELEMENTS
) {
    moab::ErrorCode ecode;
    assert(ELEMENTS.empty());
    assert(mesh.element_type == moab::MBTRI);
    const moab::Range &elements = mesh.entities[mesh.element_type];
    const std::size_t nnodes = mesh.ndof();
    const std::size_t nelements = elements.size();

    //We'll use this to check that elements are contiguous. (We don't need
    //`ELEMENTS.front()` to follow `elements.back()`, but it should and anyway
    //it'll make this check a little simpler.)
    moab::EntityHandle most_recent_element = elements.back();
    //Similarly for the edges.
    moab::EntityHandle most_recent_edge = EDGES.back();
    for (moab::EntityHandle element : elements) {
        const helpers::PseudoArray<
            const moab::EntityHandle
        > element_connectivity = mesh.connectivity(element);

        std::vector<moab::EntityHandle> edges;
        edges.reserve(3);
        ecode = mesh.impl.get_adjacencies(&element, 1, 1, false, edges);
        MB_CHK_ERR(ecode);
        assert(edges.size() == 3);

        //For each node in the original element, the midpoint of the opposite
        //edge. We use this to ensure that the new elements have the same
        //orientation as the original element.
        std::unordered_map<moab::EntityHandle, moab::EntityHandle> opposites;

        for (moab::EntityHandle edge : edges) {
            opposites.insert({
                find_node_opposite_edge(
                    element_connectivity, mesh.connectivity(edge)
                ),
                //The new nodes were created with the same ordering as the old
                //edges. The `i`th node is the midpoint of the `i`th edge.
                NODES[nnodes + mesh.index(edge)]
            });
        }

        //Add the elements containing an old node.
        moab::EntityHandle ELEMENT_CONNECTIVITY[3];
        for (std::size_t i = 0; i < 3; ++i) {
            const moab::EntityHandle node = element_connectivity[i];
            ELEMENT_CONNECTIVITY[0] = node;
            for (std::size_t j = 1; j < 3; ++j) {
                const std::size_t k = (i + (3 - j)) % 3;
                ELEMENT_CONNECTIVITY[j] = opposites.at(element_connectivity[k]);
            }
            moab::EntityHandle ELEMENT;
            ecode = mesh.impl.create_element(
                moab::MBTRI, ELEMENT_CONNECTIVITY, 3, ELEMENT
            );
            MB_CHK_ERR(ecode);
            assert(ELEMENT == ++most_recent_element);
        }

        //Add the element containing only new nodes.
        for (std::size_t i = 0; i < 3; ++i) {
            ELEMENT_CONNECTIVITY[i] = opposites.at(element_connectivity[i]);
        }
        {
            moab::EntityHandle ELEMENT;
            ecode = mesh.impl.create_element(
                moab::MBTRI, ELEMENT_CONNECTIVITY, 3, ELEMENT
            );
            MB_CHK_ERR(ecode);
            assert(ELEMENT == ++most_recent_element);
        }
        //Add the corresponding edges.
        moab::EntityHandle EDGE_CONNECTIVITY[2];
        for (std::size_t i = 0; i < 3; ++i) {
            EDGE_CONNECTIVITY[0] = ELEMENT_CONNECTIVITY[i];
            EDGE_CONNECTIVITY[1] = ELEMENT_CONNECTIVITY[(i + 1) % 3];
            moab::EntityHandle EDGE;
            ecode = mesh.impl.create_element(
                moab::MBEDGE, EDGE_CONNECTIVITY, 2, EDGE
            );
            MB_CHK_ERR(ecode);
            assert(EDGE == ++most_recent_edge);
        }
    }
    ELEMENTS.insert(elements.back() + 1, most_recent_element);
    assert(ELEMENTS.psize() == 1);
    assert(ELEMENTS.size() == 4 * nelements);

    EDGES.insert(EDGES.back() + 1, most_recent_edge);
    assert(EDGES.psize() == 1);
    return moab::MB_SUCCESS;
}

}
