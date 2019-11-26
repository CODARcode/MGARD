#include "catch2/catch.hpp"

#include <cassert>
#include <cstddef>
#include <cmath>

#include <random>
#include <vector>

#include "moab/Core.hpp"

#include "blas.hpp"

#include "data.hpp"
#include "MeshLevel.hpp"
#include "UniformMeshHierarchy.hpp"
#include "estimators.hpp"

#include "testing_utilities.hpp"

static const double inf = std::numeric_limits<double>::infinity();

TEST_CASE("comparison with Python implementation: estimators", "[estimators]") {
    moab::ErrorCode ecode;
    moab::Core mbcore;
    ecode = mbcore.load_file(mesh_path("slope.msh").c_str());
    require_moab_success(ecode);
    mgard::MeshLevel mesh(mbcore);
    mgard::UniformMeshHierarchy hierarchy(mesh, 5);
    const std::size_t N = hierarchy.ndof();

    std::vector<double> u_(N);
    const moab::Range &NODES = hierarchy.meshes.back().entities[moab::MBVERTEX];
    for (std::size_t i = 0; i < N; ++i) {
        double xyz[3];
        const moab::EntityHandle node = NODES[i];
        mbcore.get_coords(&node, 1, xyz);
        const double x = xyz[0];
        const double y = xyz[1];
        const double z = xyz[2];
        assert(z == 0);
        u_.at(i) = std::sin(10 * x - 15 * y) + 2 * std::exp(
            -1 / (1 + x * x + y * y)
        );
    }
    mgard::NodalCoefficients u_nc(u_.data());
    mgard::MultilevelCoefficients u_mc = hierarchy.decompose(u_nc);

    REQUIRE(
        mgard::estimator(u_mc, hierarchy, -1.5).unscaled ==
            Approx(14.408831895461581)
    );
    REQUIRE(
        mgard::estimator(u_mc, hierarchy, -1.0).unscaled ==
            Approx(14.414586335331334)
    );
    REQUIRE(
        mgard::estimator(u_mc, hierarchy, -0.5).unscaled ==
            Approx(14.454292254136842)
    );
    REQUIRE(
        mgard::estimator(u_mc, hierarchy,  0.0).unscaled ==
            Approx(15.34865518032767)
    );
    REQUIRE(
        mgard::estimator(u_mc, hierarchy,  0.5).unscaled ==
            Approx(32.333348842187284)
    );
    REQUIRE(
        mgard::estimator(u_mc, hierarchy,  1.0).unscaled ==
            Approx(162.87659345811159)
    );
    REQUIRE(
        mgard::estimator(u_mc, hierarchy,  1.5).unscaled ==
            Approx(914.1806446523887)
    );
}
