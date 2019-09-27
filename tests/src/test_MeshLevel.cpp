#include "catch2/catch.hpp"

#include <cstddef>

#include <set>

#include "moab/Core.hpp"

#include "blaspp/blas.hh"

#include "MeshLevel.hpp"
#include "pcg.hpp"

#include "testing_utilities.hpp"

typedef std::set<moab::EntityHandle> Edge;

static std::set<Edge> make_edges_from_offsets(
    const moab::Range &nodes,
    const std::set<std::set<std::size_t>> edges_by_offsets
) {
    std::set<Edge> edges;
    for (std::set<std::size_t> edge_by_offsets : edges_by_offsets) {
        Edge edge;
        for (std::size_t offset : edge_by_offsets) {
            edge.insert(nodes[offset]);
        }
        edges.insert(edge);
    }
    return edges;
}

static moab::ErrorCode check_edges(
    mgard::MeshLevel &mesh, const std::set<Edge> &expected_edges
) {
    moab::ErrorCode ecode;
    std::set<Edge> edges;
    for (const moab::EntityHandle e : mesh.entities[moab::MBEDGE]) {
        moab::EntityHandle const *connectivity;
        int num_nodes;
        ecode = mesh.impl->get_connectivity(e, connectivity, num_nodes);
        MB_CHK_ERR(ecode);
        assert(num_nodes == 2);
        edges.insert(Edge(connectivity, connectivity + num_nodes));
    }
    REQUIRE(edges == expected_edges);
    return moab::MB_SUCCESS;
}

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

TEST_CASE("edge generation", "[MeshLevel]") {
    moab::ErrorCode ecode;
    moab::Core mbcore;

    const std::size_t num_nodes = 5;
    const std::size_t num_tris = 4;

    const double coordinates[3 * num_nodes] = {
        0, 0, 1,
        3, 0, 2,
        3, 3, 2,
        0, 3, 0,
        2, 1, 3,
    };
    const std::size_t tri_connectivity[3 * num_tris] = {
        0, 1, 4,
        1, 2, 4,
        2, 3, 4,
        3, 0, 4
    };

    moab::Range nodes;
    ecode = mbcore.create_vertices(coordinates, num_nodes, nodes);
    require_moab_success(ecode);

    for (std::size_t i = 0; i < num_tris; ++i) {
        moab::EntityHandle triangle;
        moab::EntityHandle connectivity[3];
        for (std::size_t j = 0; j < 3; ++j) {
            connectivity[j] = nodes[tri_connectivity[3 * i + j]];
        }
        ecode = mbcore.create_element(
            moab::MBTRI, connectivity, 3, triangle
        );
        require_moab_success(ecode);
    }

    mgard::MeshLevel mesh(&mbcore);

    REQUIRE(mesh.ndof() == num_nodes);
    REQUIRE(mesh.entities[moab::MBTRI].size() == num_tris);
    REQUIRE(mesh.entities[moab::MBEDGE].size() == 8);
    check_edges(
        mesh,
        make_edges_from_offsets(
            mesh.entities[moab::MBVERTEX],
            {
                {0, 1},
                {1, 2},
                {2, 3},
                {3, 0},
                {0, 4},
                {1, 4},
                {2, 4},
                {3, 4}
            }
        )
    );
}
