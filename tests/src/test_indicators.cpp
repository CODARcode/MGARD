#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>

#include <random>
#include <string>
#include <vector>

#include "moab/Core.hpp"

#include "blas.hpp"

#include "IndicatorInput.hpp"
#include "MeshLevel.hpp"
#include "UniformMeshHierarchy.hpp"
#include "data.hpp"
#include "estimators.hpp"
#include "indicators.hpp"

#include "testing_utilities.hpp"

static double
unscaled_indicator(const mgard::MultilevelCoefficients<double> u_mc,
                   const mgard::MeshHierarchy &hierarchy, const float s) {
  double unscaled_square_indicator = 0;
  double const *p = u_mc.data;
  for (const mgard::IndicatorInput input :
       mgard::IndicatorInputRange(hierarchy)) {
    const double coefficient = *p++;
    unscaled_square_indicator +=
        coefficient * coefficient * mgard::square_indicator_factor(input, s);
  }
  return std::sqrt(unscaled_square_indicator);
}

TEST_CASE("comparison with Python implementation: indicators", "[indicators]") {
  moab::ErrorCode ecode;
  moab::Core mbcore;
  ecode = mbcore.load_file(mesh_path("seated.msh").c_str());
  require_moab_success(ecode);
  mgard::MeshLevel mesh(mbcore);
  mgard::UniformMeshHierarchy hierarchy(mesh, 3);
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
    u_.at(i) = (34 * std::sin(1 * x + 1 * y) + 21 * std::sin(2 * x + 3 * y) +
                13 * std::sin(5 * x + 8 * y)) /
               (13 + 21 + 34);
  }
  mgard::NodalCoefficients u_nc(u_.data());
  mgard::MultilevelCoefficients u_mc = hierarchy.decompose(u_nc);

  REQUIRE(unscaled_indicator(u_mc, hierarchy, -1.6) ==
          Catch::Approx(3.6534422088028458));
  REQUIRE(unscaled_indicator(u_mc, hierarchy, -0.8) ==
          Catch::Approx(4.7577598954771325));
  REQUIRE(unscaled_indicator(u_mc, hierarchy, 0.0) ==
          Catch::Approx(8.80618193171665));
  REQUIRE(unscaled_indicator(u_mc, hierarchy, 0.8) ==
          Catch::Approx(26.420258062021407));
  REQUIRE(unscaled_indicator(u_mc, hierarchy, 1.6) ==
          Catch::Approx(112.02614999556158));
}

TEST_CASE("indicators should track estimators", "[indicators]") {
  const std::string filename = GENERATE("lopsided.msh", "hexahedron.msh");
  moab::ErrorCode ecode;
  moab::Core mbcore;
  ecode = mbcore.load_file(mesh_path(filename).c_str());
  require_moab_success(ecode);
  mgard::MeshLevel mesh(mbcore);
  mgard::UniformMeshHierarchy hierarchy(mesh, 4);
  const std::size_t N = hierarchy.ndof();
  std::vector<float> smoothness_parameters = {-1.25, -0.75, -0.25, 0.0,
                                              0.25,  0.75,  1.25};

  std::vector<double> u_(N);
  std::random_device device;
  std::default_random_engine generator(device());
  std::uniform_real_distribution<double> distribution(-3, 0);
  for (double &value : u_) {
    value = distribution(generator);
  }
  // Could skip the decomposition and generate the multilevel coefficients
  // directly. Doing it this way in case I want to compute the norms or whatever
  // else later.
  const mgard::NodalCoefficients<double> u_nc(u_.data());

  const mgard::MultilevelCoefficients u_mc = hierarchy.decompose(u_nc);
  std::vector<double> square_estimators;
  for (const float s : smoothness_parameters) {
    const double estimate = mgard::estimator(u_mc, hierarchy, s);
    square_estimators.push_back(estimate * estimate);
  }

  std::vector<double> square_indicators;
  for (const float s : smoothness_parameters) {
    const double indicator = unscaled_indicator(u_mc, hierarchy, s);
    square_indicators.push_back(indicator * indicator);
  }

  const mgard::RatioBounds factors =
      mgard::s_square_indicator_bounds(hierarchy);
  for (std::size_t i = 0; i < smoothness_parameters.size(); ++i) {
    const double square_indicator = square_indicators.at(i);
    const double square_estimator = square_estimators.at(i);
    REQUIRE((factors.realism * square_indicator <= square_estimator &&
             square_estimator <= factors.reliability * square_indicator));
  }
}
