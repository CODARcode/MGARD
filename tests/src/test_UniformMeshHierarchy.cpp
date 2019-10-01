#include "catch2/catch.hpp"

#include <cstddef>

#include <random>

#include "blaspp/blas.hh"

#include "moab/Core.hpp"

#include "testing_utilities.hpp"

#include "MeshLevel.hpp"
#include "UniformMeshHierarchy.hpp"

TEST_CASE("basic properties", "[UniformMeshHierarchy]") {
    const std::size_t num_nodes = 9;
    const std::size_t num_edges = 16;
    const std::size_t num_elements = 8;

    const double coordinates[3 * num_nodes] = {
         0,  0,  0,
        -6,  2, -1,
        -1,  4,  1,
         1,  7,  2,
         3,  4, -1,
         6,  4, -2,
         5,  0,  0,
         4, -2,  1,
         1,  2, -1
    };
    const std::size_t edge_connectivity[2 * num_edges] = {
        0, 1,
        1, 2,
        2, 3,
        3, 4,
        4, 5,
        5, 6,
        6, 7,
        7, 0,
        0, 2,
        2, 4,
        4, 6,
        6, 0,
        0, 8,
        2, 8,
        4, 8,
        6, 8
    };
    const std::size_t element_connectivity[3 * num_elements] = {
        0, 2, 1,
        0, 8, 2,
        8, 0, 6,
        4, 8, 6,
        2, 8, 4,
        0, 7, 6,
        6, 5, 4,
        4, 3, 2
    };

    moab::Core mbcore;
    const mgard::MeshLevel mesh = make_mesh_level(
        mbcore, num_nodes, num_edges, num_elements, 2,
        coordinates, edge_connectivity, element_connectivity
    );
    mgard::UniformMeshHierarchy hierarchy(mesh, 2);
    REQUIRE(hierarchy.L == 2);

    const std::size_t N = hierarchy.ndof();
    std::vector<double> u(N);
    const mgard::MeshLevel &MESH = hierarchy.meshes.back();

    SECTION("zero coordinates with linear function") {
        std::vector<double>::iterator p = u.begin();
        for (const moab::EntityHandle node : MESH.entities[moab::MBVERTEX]) {
            double const *x;
            double const *y;
            double const *z;
            moab::ErrorCode ecode = mbcore.get_coords(node, x, y, z);
            require_moab_success(ecode);
            *p++ = 5 * *x - 3 * *y + 2 * *z;
        }
        hierarchy.decompose(u.data());
        bool all_near_zero = true;
        for (
            std::vector<double>::iterator p = u.begin() + hierarchy.ndof(0);
            p != u.end();
            ++p
        ) {
            all_near_zero = all_near_zero && std::abs(*p) < 1e-6;
        }
        REQUIRE(all_near_zero);
    }

    std::random_device device;
    std::default_random_engine generator(device());
    std::uniform_real_distribution<double> distribution(-1, 1);

    SECTION("recompose inverts decompose") {
        for (double &value : u) {
            value = distribution(generator);
        }
        std::vector<double> copy = u;
        hierarchy.decompose(u.data());
        hierarchy.recompose(u.data());
        blas::axpy(N, -1, u.data(), 1, copy.data(), 1);
        std::vector<double> &errors = copy;
        REQUIRE(std::abs(blas::nrm2(N, errors.data(), 1)) < 1e-9 * N);
    }

    SECTION("recompose inverts decompose") {
        for (double &value : u) {
            value = distribution(generator);
        }
        std::vector<double> copy = u;
        hierarchy.recompose(u.data());
        hierarchy.decompose(u.data());
        blas::axpy(N, -1, u.data(), 1, copy.data(), 1);
        std::vector<double> &errors = copy;
        REQUIRE(std::abs(blas::nrm2(N, errors.data(), 1)) < 1e-9 * N);
    }

    SECTION("multilevel coefficients depend linearly on nodal coefficients") {
        const std::size_t N = u.size();
        std::vector<double> v(N);
        for (std::size_t i = 0; i < N; ++i) {
            u.at(i) = distribution(generator);
            v.at(i) = distribution(generator);
        }
        const double alpha = distribution(generator);
        std::vector<double> w(N);
        //`w = u + alpha * v`.
        blas::copy(N, u.data(), 1, w.data(), 1);
        blas::axpy(N, alpha, v.data(), 1, w.data(), 1);
        hierarchy.decompose(u.data());
        hierarchy.decompose(v.data());
        hierarchy.decompose(w.data());
        //Copy just overwrite `u` instead.
        std::vector<double> expected(N);
        blas::copy(N, u.data(), 1, expected.data(), 1);
        blas::axpy(N, alpha, v.data(), 1, expected.data(), 1);
        blas::axpy(N, -1, w.data(), 1, expected.data(), 1);
        std::vector<double> &errors = expected;
        REQUIRE(std::abs(blas::nrm2(N, errors.data(), 1)) < 1e-9 * N);
    }
}
