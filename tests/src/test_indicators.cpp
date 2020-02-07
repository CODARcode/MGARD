#include "catch2/catch.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>

#include <random>
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
                   const mgard::MeshHierarchy &hierarchy, const double s) {
  double unscaled_square_indicator = 0;
  for (const mgard::IndicatorInput input :
       mgard::IndicatorInputRange(hierarchy, u_mc)) {
    unscaled_square_indicator +=
        mgard::square_indicator_coefficient<double>(input, s).unscaled;
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
    const double z = xyz[2];
    assert(z == 0);
    u_.at(i) = (34 * std::sin(1 * x + 1 * y) + 21 * std::sin(2 * x + 3 * y) +
                13 * std::sin(5 * x + 8 * y)) /
               (13 + 21 + 34);
  }
  mgard::NodalCoefficients u_nc(u_.data());
  mgard::MultilevelCoefficients u_mc = hierarchy.decompose(u_nc);

  REQUIRE(unscaled_indicator(u_mc, hierarchy, -1.6) ==
          Approx(3.6534422088028458));
  REQUIRE(unscaled_indicator(u_mc, hierarchy, -0.8) ==
          Approx(4.7577598954771325));
  REQUIRE(unscaled_indicator(u_mc, hierarchy, 0.0) == Approx(8.80618193171665));
  REQUIRE(unscaled_indicator(u_mc, hierarchy, 0.8) ==
          Approx(26.420258062021407));
  REQUIRE(unscaled_indicator(u_mc, hierarchy, 1.6) ==
          Approx(112.02614999556158));
}

TEST_CASE("indicators should track estimators", "[indicators]") {
  moab::ErrorCode ecode;
  moab::Core mbcore;
  ecode = mbcore.load_file(mesh_path("lopsided.msh").c_str());
  require_moab_success(ecode);
  mgard::MeshLevel mesh(mbcore);
  mgard::UniformMeshHierarchy hierarchy(mesh, 4);
  const std::size_t N = hierarchy.ndof();
  std::vector<double> smoothness_parameters = {-1.25, -0.75, -0.25, 0.0,
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
  std::vector<mgard::SandwichBounds> estimators;
  for (const double s : smoothness_parameters) {
    estimators.push_back(mgard::estimator(u_mc, hierarchy, s));
  }

  std::vector<mgard::SandwichBounds> square_indicators;
  for (const double s : smoothness_parameters) {
    // Just initializing everything to zero.
    mgard::SandwichBounds sum({.realism = 0, .reliability = 0}, 0);
    for (const mgard::IndicatorInput input :
         mgard::IndicatorInputRange(hierarchy, u_mc)) {
      mgard::SandwichBounds summand =
          mgard::square_indicator_coefficient<double>(input, s);
      sum.unscaled += summand.unscaled;
      sum.lower += summand.lower;
      sum.upper += summand.upper;
    }
    square_indicators.push_back(sum);
  }

  for (std::size_t i = 0; i < smoothness_parameters.size(); ++i) {
    const mgard::SandwichBounds square_indicator = square_indicators.at(i);
    const mgard::SandwichBounds estimator = estimators.at(i);
    REQUIRE((std::sqrt(square_indicator.lower) <= estimator.unscaled &&
             estimator.unscaled <= std::sqrt(square_indicator.upper)));
  }
}
