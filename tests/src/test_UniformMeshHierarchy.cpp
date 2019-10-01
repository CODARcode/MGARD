#include "catch2/catch.hpp"

#include <cstddef>

#include <random>

#include "blaspp/blas.hh"

#include "moab/Core.hpp"

#include "testing_utilities.hpp"

#include "MeshLevel.hpp"
#include "UniformMeshHierarchy.hpp"

TEST_CASE("basic properties", "[UniformMeshHierarchy]") {
    moab::ErrorCode ecode;
    moab::Core mbcore;
    ecode = mbcore.load_file(mesh_path("pyramid.msh").c_str());
    require_moab_success(ecode);
    const mgard::MeshLevel mesh(mbcore);
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
