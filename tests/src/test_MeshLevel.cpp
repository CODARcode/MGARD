#include "catch2/catch.hpp"

#include <cstddef>

#include "moab/Core.hpp"

#include "blaspp/blas.hh"

#include "MeshLevel.hpp"
#include "pcg.hpp"

#include "testing_utilities.hpp"

TEST_CASE("MeshLevel construction", "[MeshLevel]") {
    moab::Core mbcore;

    const std::size_t num_nodes = 6;
    const std::size_t num_edges = 9;
    const std::size_t num_tris = 4;
    const double coordinates[3 * num_nodes] = {
        0, 0, 0,
        5, 0, 0,
        12, 0, 0,
        0, 8, 0,
        5, 6, 0,
        9, 3, 0
    };
    const std::size_t edge_connectivity[2 * num_edges] = {
        0, 1,
        1, 3,
        3, 0,
        1, 2,
        2, 5,
        5, 1,
        1, 4,
        4, 3,
        5, 4
    };
    const std::size_t tri_connectivity[3 * num_tris] = {
        0, 1, 3,
        1, 2, 5,
        1, 4, 3,
        1, 5, 4
    };
    mgard::MeshLevel mesh = make_mesh_level(
        mbcore, num_nodes, num_edges, num_tris, 2,
        coordinates, edge_connectivity, tri_connectivity
    );
    REQUIRE(mesh.topological_dimension == 2);
    REQUIRE(mesh.element_type == moab::MBTRI);
    REQUIRE(mesh.ndof() == num_nodes);

    //Don't want to deal with a lot of nonrational lengths, so I'm only checking
    //a few.
    const moab::Range &edges = mesh.entities[moab::MBEDGE];
    REQUIRE(mesh.measure(edges[0]) == Approx(5));
    REQUIRE(mesh.measure(edges[3]) == Approx(7));
    REQUIRE(mesh.measure(edges[6]) == Approx(6));

    const moab::Range &elements = mesh.entities[mesh.element_type];
    double triangle_areas[num_tris] = {20, 10.5, 15, 12};
    SECTION("measures without precomputing") {
        //`precompute_measures` should be called by `measure`.
        for (std::size_t i = 0; i < num_tris; ++i) {
            REQUIRE(
                mesh.measure(elements[i]) == Approx(triangle_areas[i])
            );
        }
    }
    SECTION("measures with precomputing") {
        mesh.precompute_element_measures();
        for (std::size_t i = 0; i < num_tris; ++i) {
            REQUIRE(
                mesh.measure(elements[i]) == Approx(triangle_areas[i])
            );
        }
    }
}
