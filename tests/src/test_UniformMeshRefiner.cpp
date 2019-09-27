#include "catch2/catch.hpp"

#include <iostream>

#include <array>
#include <set>

#include "moab/Core.hpp"

#include "UniformMeshRefiner.hpp"
#include "utilities.hpp"

#include "testing_utilities.hpp"

typedef std::array<double, 3> NodeCoordinates;
typedef std::set<NodeCoordinates> Element;

static moab::ErrorCode check_elements(
    mgard::MeshLevel &MESH, const std::set<Element> &expected_elements
) {
    moab::ErrorCode ecode;
    std::set<Element> elements;
    for (const moab::EntityHandle t : MESH.entities[MESH.element_type]) {
        moab::EntityHandle const *connectivity;
        int num_nodes;
        ecode = MESH.impl->get_connectivity(t, connectivity, num_nodes);
        MB_CHK_ERR(ecode);
        assert(num_nodes == 3);
        Element element;
        for (const moab::EntityHandle x : helpers::PseudoArray(
                connectivity, num_nodes
        )) {
            NodeCoordinates xyz;
            //Could look this up ahead of time.
            ecode = MESH.impl->get_coords(&x, 1, xyz.data());
            MB_CHK_ERR(ecode);
            element.insert(xyz);
        }
        elements.insert(element);
    }
    REQUIRE(elements == expected_elements);
    return moab::MB_SUCCESS;
}

TEST_CASE("refining multiple triangles", "[UniformMeshRefiner]") {
    moab::ErrorCode ecode;

    moab::Core mbcore;
    const std::size_t num_nodes = 5;
    const std::size_t num_edges = 7;
    const std::size_t num_elements = 3;

    const double coordinates[3 * num_nodes] = {
        0,  0, 0,
        8,  0, 0,
        0,  6, 0,
        4, -4, 0,
        8, -4, 0
    };
    const std::size_t edge_connectivity[2 * num_edges] = {
        0, 1,
        1, 2,
        2, 0,
        0, 3,
        3, 1,
        3, 4,
        4, 1
    };
    const std::size_t element_connectivity[3 * num_elements] = {
        0, 1, 2,
        0, 3, 1,
        3, 1, 4
    };

    mgard::MeshLevel mesh = make_mesh_level(
        mbcore,
        num_nodes,
        num_edges,
        num_elements,
        2,
        coordinates,
        edge_connectivity,
        element_connectivity
    );

    mgard::UniformMeshRefiner refiner;
    mgard::MeshLevel MESH = refiner(mesh);

    REQUIRE(MESH.topological_dimension == 2);
    REQUIRE(MESH.ndof() == 12);
    REQUIRE(MESH.entities[moab::MBEDGE].size() == 23);
    REQUIRE(MESH.entities[moab::MBTRI].size() == 12);

    ecode = check_elements(MESH, {
        //Element given by the coordinates of their vertices.
        {{0,  0, 0}, {4,  0, 0}, {0,  3, 0}},
        {{4,  0, 0}, {4,  3, 0}, {0,  3, 0}},
        {{4,  0, 0}, {8,  0, 0}, {4,  3, 0}},
        {{0,  3, 0}, {4,  3, 0}, {0,  6, 0}},
        {{0,  0, 0}, {4,  0, 0}, {2, -2, 0}},
        {{4,  0, 0}, {8,  0, 0}, {6, -2, 0}},
        {{2, -2, 0}, {4,  0, 0}, {6, -2, 0}},
        {{2, -2, 0}, {4, -4, 0}, {6, -2, 0}},
        {{4, -4, 0}, {6, -4, 0}, {6, -2, 0}},
        {{6, -4, 0}, {6, -2, 0}, {8, -2, 0}},
        {{6, -4, 0}, {8, -4, 0}, {8, -2, 0}},
        {{6, -2, 0}, {8, -2, 0}, {8,  0, 0}}
    });
    require_moab_success(ecode);
}

TEST_CASE("refining triangle multiply", "[UniformMeshRefiner]") {
    moab::ErrorCode ecode;

    moab::Core mbcore;
    const std::size_t num_nodes = 3;
    const std::size_t num_edges = 3;
    const std::size_t num_elements = 1;
    const double z = 2.478923;

    const double coordinates[3 * num_nodes] = {
        0,  0, z,
        10,  0, z,
        10,  6, z,
    };
    const std::size_t edge_connectivity[2 * num_edges] = {
        0, 1,
        1, 2,
        2, 0
    };
    const std::size_t element_connectivity[3 * num_elements] = {
        0, 1, 2
    };

    mgard::MeshLevel mesh = make_mesh_level(
        mbcore,
        num_nodes,
        num_edges,
        num_elements,
        2,
        coordinates,
        edge_connectivity,
        element_connectivity
    );
    mgard::UniformMeshRefiner refiner;
    mgard::MeshLevel MESH = refiner(refiner(mesh));
    REQUIRE(MESH.ndof() == 15);
    REQUIRE(MESH.entities[moab::MBEDGE].size() == 30);
    REQUIRE(MESH.entities[moab::MBTRI].size() == 16);

    ecode = check_elements(MESH, {
        {{  0,   0, z}, {2.5,   0, z}, {2.5, 1.5, z}},
        {{2.5,   0, z}, {  5,   0, z}, {  5, 1.5, z}},
        {{  5,   0, z}, {7.5,   0, z}, {7.5, 1.5, z}},
        {{7.5,   0, z}, { 10,   0, z}, { 10, 1.5, z}},
        {{2.5,   0, z}, {2.5, 1.5, z}, {  5, 1.5, z}},
        {{  5,   0, z}, {  5, 1.5, z}, {7.5, 1.5, z}},
        {{7.5,   0, z}, {7.5, 1.5, z}, { 10, 1.5, z}},
        {{2.5, 1.5, z}, {  5, 1.5, z}, {  5,   3, z}},
        {{  5, 1.5, z}, {7.5, 1.5, z}, {7.5,   3, z}},
        {{7.5, 1.5, z}, { 10, 1.5, z}, { 10,   3, z}},
        {{  5, 1.5, z}, {  5,   3, z}, {7.5,   3, z}},
        {{7.5, 1.5, z}, {7.5,   3, z}, { 10,   3, z}},
        {{  5,   3, z}, {7.5,   3, z}, {7.5, 4.5, z}},
        {{7.5,   3, z}, { 10,   3, z}, { 10, 4.5, z}},
        {{7.5,   3, z}, {7.5, 4.5, z}, { 10, 4.5, z}},
        {{7.5, 4.5, z}, { 10, 4.5, z}, { 10,   6, z}},
    });
    require_moab_success(ecode);
}
