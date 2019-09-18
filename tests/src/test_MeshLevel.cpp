#include "catch2/catch.hpp"

#include <cstddef>

#include "moab/Core.hpp"

#include "blaspp/blas.hh"

#include "MeshLevel.hpp"

static mgard::MeshLevel make_mesh_level(
    moab::Core &mbcore,
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
    MB_CHK_ERR_CONT(ecode);
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
        MB_CHK_ERR_CONT(ecode);
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
        MB_CHK_ERR_CONT(ecode);
        elements.insert(handle);
    }
    assert(elements.psize() == 1);
    return mgard::MeshLevel(&mbcore, nodes, edges, elements);
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

TEST_CASE("MeshLevel mass matrix matvec", "[MeshLevel]") {
    SECTION("triangles") {
        moab::Core mbcore;
        const std::size_t num_nodes = 5;
        const std::size_t num_edges = 7;
        const std::size_t num_tris = 3;
        const double coordinates[3 * num_nodes] = {
            0, 0, 1,
            3, 0, 1,
            0, 4, 1,
            6, 4, 1,
            1.5, -3, 1,
        };
        const std::size_t edge_connectivity[2 * num_edges] = {
            0, 1,
            1, 2,
            2, 0,
            1, 3,
            3, 2,
            0, 4,
            4, 1,
        };
        const std::size_t tri_connectivity[3 * num_tris] = {
            0, 1, 2,
            2, 1, 3,
            1, 0, 4,
        };
        mgard::MeshLevel mesh = make_mesh_level(
            mbcore, num_nodes, num_edges, num_tris, 2,
            coordinates, edge_connectivity, tri_connectivity
        );
        REQUIRE(mesh.topological_dimension == 2);
        REQUIRE(mesh.element_type == moab::MBTRI);

        //These products were obtained using the Python implementation.
        const std::size_t num_trials = 8;
        const double vs[num_trials][num_nodes] = {
            {1, 0, 0, 0, 0},
            {0, 1, 0, 0, 0},
            {0, 0, 1, 0, 0},
            {0, 0, 0, 1, 0},
            {0, 0, 0, 0, 1},
            {1, -1, 6, 0, -1},
            {15.1, 6.7, 0.4, -8.3, 30.4},
            {108.5, 90.4, -32.7, -61.1, -12.1}
        };
        const double expecteds[num_trials][num_nodes] = {
            {1.75, 0.875, 0.5, 0., 0.375},
            {0.875, 3.75, 1.5, 1., 0.375},
            {0.5, 1.5, 3., 1., 0.},
            {0., 1., 1., 2., 0.},
            {0.375, 0.375, 0., 0., 0.75},
            {3.5, 5.75, 17., 5., -0.75},
            {43.8875, 42.0375, 10.5, -9.5, 30.975},
            {248.0875, 319.25, 30.65, -64.5, 65.5125}
        };

        for (std::size_t i = 0; i < num_trials; ++i) {
            double b[num_nodes];
            mesh.mass_matrix_matvec(vs[i], b);
            bool all_close = true;
            for (std::size_t j = 0; j < num_nodes; ++j) {
                all_close = all_close && b[j] == Approx(expecteds[i][j]);
            }
            REQUIRE(all_close);
        }
    }

    SECTION("tetrahedra") {
        moab::Core mbcore;
        const std::size_t num_nodes = 5;
        const std::size_t num_edges = 9;
        const std::size_t num_tets = 2;
        const double coordinates[3 * num_nodes] = {
            0, 0, 0,
            5, 1, 0,
            -1, 3, 0,
            0, 0, 4,
            3, 2, 3
        };
        const std::size_t edge_connectivity[2 * num_edges] = {
            0, 1,
            0, 2,
            0, 3,
            1, 2,
            1, 3,
            2, 3,
            2, 4,
            1, 4,
            3, 4
        };
        const std::size_t tet_connectivity[4 * num_tets] = {
            0, 1, 2, 3,
            2, 1, 3, 4
        };
        mgard::MeshLevel mesh = make_mesh_level(
            mbcore, num_nodes, num_edges, num_tets, 3,
            coordinates, edge_connectivity, tet_connectivity
        );
        REQUIRE(mesh.topological_dimension == 3);
        REQUIRE(mesh.element_type == moab::MBTET);
        double u[num_nodes] = {-8, -7, 3, 0, 10};
        double b[num_nodes];
        mesh.mass_matrix_matvec(u, b);

        const moab::Range &elements = mesh.entities[mesh.element_type];
        double measures[num_tets];
        for (std::size_t i = 0; i < num_tets; ++i) {
            measures[i] = mesh.measure(elements[i]);
        }

        double expected[num_nodes];
        expected[0] = -20 * measures[0];
        expected[1] = -19 * measures[0] - 1 * measures[1];
        expected[2] = -9 * measures[0] + 9 * measures[1];
        expected[3] = -12 * measures[0] + 6 * measures[1];
        expected[4] = 16 * measures[1];
        blas::scal(num_nodes, 1.0 / 20.0, expected, 1);

        bool all_close = true;
        for (std::size_t i = 0; i < num_nodes; ++i) {
            all_close = all_close && b[i] == Approx(expected[i]);
        }
        REQUIRE(all_close);
    }
}
