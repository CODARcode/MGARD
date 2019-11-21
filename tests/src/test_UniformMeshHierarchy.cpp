#include "catch2/catch.hpp"

#include <cstddef>
#include <cmath>

#include <algorithm>
#include <fstream>
#include <limits>
#include <random>
#include <set>

#include "blas.hpp"

#include "moab/Core.hpp"

#include "testing_utilities.hpp"

#include "MeshLevel.hpp"
#include "UniformMeshHierarchy.hpp"

//These tests call `decompose` and `recompose` in the old style and read the
//transformed coefficients from the same identifier.

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
        require_vector_equality(u, copy);
    }

    SECTION("recompose inverts decompose") {
        for (double &value : u) {
            value = distribution(generator);
        }
        std::vector<double> copy = u;
        hierarchy.recompose(u.data());
        hierarchy.decompose(u.data());
        require_vector_equality(u, copy);
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
        blas::copy(N, u.data(), w.data());
        blas::axpy(N, alpha, v.data(), w.data());
        hierarchy.decompose(u.data());
        hierarchy.decompose(v.data());
        hierarchy.decompose(w.data());
        //Copy just overwrite `u` instead.
        std::vector<double> expected(N);
        blas::copy(N, u.data(), expected.data());
        blas::axpy(N, alpha, v.data(), expected.data());
        require_vector_equality(w, expected);
    }
}

TEST_CASE(
    "comparison with Python implementation: refinement and decomposition",
    "[UniformMeshHierarchy]"
) {
    moab::ErrorCode ecode;
    moab::Core mbcore;
    ecode = mbcore.load_file(mesh_path("circle.msh").c_str());
    require_moab_success(ecode);
    mgard::MeshLevel mesh(mbcore);
    mgard::UniformMeshHierarchy hierarchy(mesh, 2);

    const std::size_t N = hierarchy.ndof();
    const mgard::MeshLevel &MESH = hierarchy.meshes.back();
    const moab::Range &NODES = MESH.entities[moab::MBVERTEX];
    std::vector<double> coordinates(3 * N);
    ecode = MESH.impl.get_coords(NODES, coordinates.data());
    require_moab_success(ecode);

    std::vector<double> u_nc(N);

    std::set<std::size_t> accounted_for_indices;
    //This will have the ordering of our nodes, not the reference nodes.
    std::vector<double> ref_mc(N);

    double max_rel_xyz_err = 0;
    std::ifstream f;
    f.open(output_path("circle_L=2_coefficients.txt").c_str());
    double xyz[3];
    double nc;
    double mc;
    while (f >> xyz[0] >> xyz[1] >> xyz[2] >> nc >> mc) {
        //Fine because this mesh doesn't have a node at the origin.
        const double xyz_norm = blas::nrm2(3, xyz);
        double min_distance = std::numeric_limits<double>::max();
        std::size_t i;
        double const *_xyz = coordinates.data();
        double difference[3];
        for (std::size_t j = 0; j < N; ++j) {
            blas::copy(3, xyz, difference);
            blas::axpy(3, -1.0, _xyz, difference);
            const double distance = blas::nrm2(3, difference);
            if (distance < min_distance) {
                min_distance = distance;
                i = j;
            }
            _xyz += 3;
        }
        assert(min_distance != std::numeric_limits<double>::max());
        const double rel_xyz_err = min_distance / xyz_norm;
        max_rel_xyz_err = std::max(max_rel_xyz_err, rel_xyz_err);

        accounted_for_indices.insert(i);
        u_nc.at(i) = nc;
        ref_mc.at(i) = mc;
    }
    f.close();
    //Check that the references nodes were uniquely paired with ours.
    REQUIRE(accounted_for_indices.size() == N);
    REQUIRE(max_rel_xyz_err < 1e-6);

    std::vector<double> u_mc = u_nc;
    hierarchy.decompose(u_mc.data());

    double max_rel_mc_err = 0;
    for (std::size_t j = 0; j < N; ++j) {
        const double rel_mc_err = std::abs(
            (u_mc.at(j) - ref_mc.at(j)) / u_mc.at(j)
        );
        max_rel_mc_err = std::max(max_rel_mc_err, rel_mc_err);
    }
    //Expect some error.
    assert(max_rel_mc_err != 0);
    REQUIRE(max_rel_mc_err < 1e-6);
}
