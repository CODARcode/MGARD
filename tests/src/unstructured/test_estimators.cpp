#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "moab/Core.hpp"

#include "blas.hpp"

#include "unstructured/MeshLevel.hpp"
#include "unstructured/UniformMeshHierarchy.hpp"
#include "unstructured/data.hpp"
#include "unstructured/estimators.hpp"
#include "unstructured/norms.hpp"

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
    [[maybe_unused]] const double z = xyz[2];
    assert(z == 0);
    u_.at(i) =
        std::sin(10 * x - 15 * y) + 2 * std::exp(-1 / (1 + x * x + y * y));
  }
  mgard::NodalCoefficients u_nc(u_.data());
  mgard::MultilevelCoefficients u_mc = hierarchy.decompose(u_nc);

  const mgard::RatioBounds factors =
      mgard::s_square_estimator_bounds(hierarchy);
  REQUIRE(factors.reliability == 1);
  // Because the unscaled estimator gets scaled by one to get the upper
  // estimate, we can use the unscaled estimator in the following.

  REQUIRE(mgard::estimator(u_mc, hierarchy, -1.5) ==
          Catch::Approx(14.408831895461581));
  REQUIRE(mgard::estimator(u_mc, hierarchy, -1.0) ==
          Catch::Approx(14.414586335331334));
  REQUIRE(mgard::estimator(u_mc, hierarchy, -0.5) ==
          Catch::Approx(14.454292254136842));
  REQUIRE(mgard::estimator(u_mc, hierarchy, 0.0) ==
          Catch::Approx(15.34865518032767));
  REQUIRE(mgard::estimator(u_mc, hierarchy, 0.5) ==
          Catch::Approx(32.333348842187284));
  REQUIRE(mgard::estimator(u_mc, hierarchy, 1.0) ==
          Catch::Approx(162.87659345811159));
  REQUIRE(mgard::estimator(u_mc, hierarchy, 1.5) ==
          Catch::Approx(914.1806446523887));
}

TEST_CASE("estimators should track norms", "[estimators]") {
  const std::string filename = GENERATE("pyramid.msh", "hexahedron.msh");
  moab::ErrorCode ecode;
  moab::Core mbcore;
  ecode = mbcore.load_file(mesh_path(filename).c_str());
  require_moab_success(ecode);
  mgard::MeshLevel mesh(mbcore);
  mgard::UniformMeshHierarchy hierarchy(mesh, 3);
  const std::size_t N = hierarchy.ndof();
  std::vector<float> smoothness_parameters = {-1.5, -1.0, -0.5, 0.0,
                                              0.5,  1.0,  1.5};

  std::vector<double> u_(N);
  std::random_device device;
  std::default_random_engine generator(device());
  std::uniform_real_distribution<double> distribution(-1, 1);

  std::generate(u_.begin(), u_.end(),
                [&]() -> double { return distribution(generator); });
  const mgard::NodalCoefficients<double> u_nc(u_.data());

  std::vector<double> norms(smoothness_parameters.size());
  std::transform(
      smoothness_parameters.begin(), smoothness_parameters.end(), norms.begin(),
      [&](const float s) -> float { return mgard::norm(u_nc, hierarchy, s); });

  const mgard::MultilevelCoefficients u_mc = hierarchy.decompose(u_nc);
  const mgard::RatioBounds factors =
      mgard::s_square_estimator_bounds(hierarchy);
  for (std::size_t i = 0; i < smoothness_parameters.size(); ++i) {
    const float s = smoothness_parameters.at(i);
    const double norm = norms.at(i);
    const double estimate = mgard::estimator(u_mc, hierarchy, s);
    REQUIRE((std::sqrt(factors.realism) * estimate <= norm &&
             norm <= std::sqrt(factors.reliability) * estimate));
  }
}
