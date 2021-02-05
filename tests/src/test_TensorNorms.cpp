#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"

#include <cmath>
#include <cstddef>
#include <cstdlib>

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

#include <random>

#include "blas.hpp"

#include "TensorMeshHierarchy.hpp"
#include "TensorMeshHierarchyIteration.hpp"
#include "TensorNorms.hpp"

#include "testing_random.hpp"
#include "testing_utilities.hpp"

namespace {

template <std::size_t N, typename Real>
void test_tensor_basic_norm_properties(
    const std::array<std::size_t, N> shape,
    std::default_random_engine &generator,
    std::uniform_real_distribution<Real> &node_spacing_distribution,
    std::uniform_real_distribution<Real> &nodal_coefficient_distribution) {
  const mgard::TensorMeshHierarchy<N, Real> hierarchy =
      hierarchy_with_random_spacing(generator, node_spacing_distribution,
                                    shape);
  const std::size_t ndof = hierarchy.ndof();
  std::vector<Real> u_(ndof);
  for (Real &x : u_) {
    x = nodal_coefficient_distribution(generator);
  }
  Real *const u = u_.data();
  std::vector<Real> v_(ndof);
  Real *const v = v_.data();
  std::vector<Real> w_(ndof);
  Real *const w = w_.data();

  const std::vector<Real> smoothness_parameters = {
      -1.5, -1.0, -0.5,
      0,    1e-9, 0.5,
      1.0,  1.5,  std::numeric_limits<Real>::infinity()};

  SECTION("absolute homogeneity") {
    const Real alpha = nodal_coefficient_distribution(generator);
    for (std::size_t i = 0; i < ndof; ++i) {
      v_.at(i) = alpha * u_.at(i);
    }
    TrialTracker tracker;
    for (const Real s : smoothness_parameters) {
      tracker += mgard::norm(hierarchy, v, s) ==
                 Catch::Approx(std::abs(alpha) * mgard::norm(hierarchy, u, s));
    }
    REQUIRE(tracker);
  }

  SECTION("triangle inequality") {
    for (std::size_t i = 0; i < ndof; ++i) {
      v_.at(i) = nodal_coefficient_distribution(generator);
      w_.at(i) = u_.at(i) + v_.at(i);
    }
    TrialTracker tracker;
    for (const Real s : smoothness_parameters) {
      tracker += mgard::norm(hierarchy, u, s) + mgard::norm(hierarchy, v, s) >=
                 mgard::norm(hierarchy, w, s);
    }
    REQUIRE(tracker);
  }
}

} // namespace

TEST_CASE("tensor basic norm properties", "[norms]") {
  std::default_random_engine generator;
  {
    std::uniform_real_distribution<float> node_spacing_distribution(1, 3);
    std::uniform_real_distribution<float> nodal_coefficient_distribution(-5, 5);
    test_tensor_basic_norm_properties<1, float>({35}, generator,
                                                node_spacing_distribution,
                                                nodal_coefficient_distribution);
    test_tensor_basic_norm_properties<2, float>({17, 17}, generator,
                                                node_spacing_distribution,
                                                nodal_coefficient_distribution);
  }
  {
    std::uniform_real_distribution<double> node_spacing_distribution(0.25,
                                                                     0.375);
    std::uniform_real_distribution<double> nodal_coefficient_distribution(-3,
                                                                          0.5);
    test_tensor_basic_norm_properties<3, double>(
        {7, 8, 9}, generator, node_spacing_distribution,
        nodal_coefficient_distribution);
    test_tensor_basic_norm_properties<4, double>(
        {3, 5, 2, 2}, generator, node_spacing_distribution,
        nodal_coefficient_distribution);
  }
}

namespace {

float f(const std::array<float, 3> xyz) {
  const float x = xyz.at(0);
  const float y = xyz.at(1);
  const float z = xyz.at(2);
  return std::sin(2 * x - 1 * y) + std::sin(-1 * x + 2 * z) +
         std::sin(-3 * y + 1 * z) + std::sin(4 * x * y * z);
}

} // namespace

TEST_CASE("comparison with Python implementation: tensor norms", "[norms]") {
  const mgard::TensorMeshHierarchy<3, float> hierarchy({9, 9, 9});
  const std::size_t ndof = hierarchy.ndof();
  std::vector<float> u_(ndof);
  float *const u = u_.data();
  for (mgard::TensorNode<3> node :
       mgard::ShuffledTensorNodeRange(hierarchy, hierarchy.L)) {
    hierarchy.at(u, node.multiindex) = f(coordinates(hierarchy, node));
  }

  const float inf_ = std::numeric_limits<float>::infinity();
  REQUIRE(mgard::norm(hierarchy, u, inf_) == Catch::Approx(2.9147100421936996));
  REQUIRE(mgard::norm(hierarchy, u, -1.5f) ==
          Catch::Approx(1.0933978732810643));
  REQUIRE(mgard::norm(hierarchy, u, -1.0f) ==
          Catch::Approx(1.0975934890537276));
  REQUIRE(mgard::norm(hierarchy, u, -0.5f) == Catch::Approx(1.106198751014936));
  REQUIRE(mgard::norm(hierarchy, u, 0.0f) == Catch::Approx(1.1242926017063057));
  REQUIRE(mgard::norm(hierarchy, u, 1e-9f) == Catch::Approx(1.124292601758146));
  REQUIRE(mgard::norm(hierarchy, u, 0.5f) == Catch::Approx(1.164206301367337));
  REQUIRE(mgard::norm(hierarchy, u, 1.0f) == Catch::Approx(1.2600881685595349));
  // Small error encountered here.
  REQUIRE(mgard::norm(hierarchy, u, 1.5f) ==
          Catch::Approx(1.5198059864642621).epsilon(0.001));
}

namespace {

template <std::size_t N, typename Real>
void test_component_norms_sum(
    std::default_random_engine &generator,
    std::uniform_real_distribution<float> &distribution,
    const std::array<std::size_t, N> shape, const Real s) {
  const mgard::TensorMeshHierarchy<N, Real> hierarchy =
      hierarchy_with_random_spacing(generator, distribution, shape);
  const std::size_t ndof = hierarchy.ndof();

  Real *const u = static_cast<Real *>(std::malloc(sizeof(Real) * ndof));
  generate_reasonable_function(hierarchy, s, generator, u);

  Real *const f = static_cast<Real *>(std::malloc(sizeof(Real) * ndof));
  {
    std::copy(u, u + ndof, f);
    const mgard::TensorMassMatrix<N, Real> M(hierarchy, hierarchy.L);
    M(f);
  }
  const std::vector<Real> square_norms =
      mgard::orthogonal_component_square_norms(hierarchy, u, f);
  std::free(f);

  REQUIRE(std::sqrt(std::accumulate(square_norms.begin(), square_norms.end(),
                                    static_cast<Real>(0))) ==
          Catch::Approx(mgard::norm<N, Real>(hierarchy, u, 0)));

  std::free(u);

  {
    TrialTracker tracker;
    for (const Real square_norm : square_norms) {
      tracker += square_norm >= 0;
    }
    REQUIRE(tracker);
  }
}

} // namespace

TEST_CASE("orthogonal component norms", "[norms]") {
  std::default_random_engine gen;
  // Node spacing distribution.
  std::uniform_real_distribution<float> dis(0.01, 0.02);

  test_component_norms_sum<1, float>(gen, dis, {33}, 0);
  test_component_norms_sum<1, float>(gen, dis, {160}, 0.25);
  test_component_norms_sum<2, float>(gen, dis, {9, 11}, 0.5);
  test_component_norms_sum<2, float>(gen, dis, {30, 29}, 0.75);
  test_component_norms_sum<3, float>(gen, dis, {5, 6, 21}, 1.0);
  test_component_norms_sum<3, float>(gen, dis, {10, 10, 10}, 1.25);
}
