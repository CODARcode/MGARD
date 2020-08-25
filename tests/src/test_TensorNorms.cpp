#include "catch2/catch.hpp"

#include <cmath>
#include <cstddef>

#include <array>
#include <limits>
#include <vector>

#include <random>

#include "blas.hpp"

#include "TensorMeshHierarchy.hpp"
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
                 Approx(std::abs(alpha) * mgard::norm(hierarchy, u, s));
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
  for (mgard::TensorNode<3, float> node : hierarchy.nodes(hierarchy.L)) {
    hierarchy.at(u, node.multiindex) = f(node.coordinates);
  }

  const float inf_ = std::numeric_limits<float>::infinity();
  REQUIRE(mgard::norm(hierarchy, u, inf_) == Approx(2.9147100421936996));
  REQUIRE(mgard::norm(hierarchy, u, -1.5f) == Approx(1.0933978732810643));
  REQUIRE(mgard::norm(hierarchy, u, -1.0f) == Approx(1.0975934890537276));
  REQUIRE(mgard::norm(hierarchy, u, -0.5f) == Approx(1.106198751014936));
  REQUIRE(mgard::norm(hierarchy, u, 0.0f) == Approx(1.1242926017063057));
  REQUIRE(mgard::norm(hierarchy, u, 1e-9f) == Approx(1.124292601758146));
  REQUIRE(mgard::norm(hierarchy, u, 0.5f) == Approx(1.164206301367337));
  REQUIRE(mgard::norm(hierarchy, u, 1.0f) == Approx(1.2600881685595349));
  // Small error encountered here.
  REQUIRE(mgard::norm(hierarchy, u, 1.5f) ==
          Approx(1.5198059864642621).epsilon(0.001));
}

namespace {

static const float alpha = -1.7855108138339222;

float f(int n1, int n2, int n3, float *v) {
  const std::size_t i = n1 * n2 * n3 / 4;
  return v[i - 1] - 4 * v[i] + v[i + 1];
}

float scaled_f(int n1, int n2, int n3, float *v) {
  return alpha * f(n1, n2, n3, v);
}

float g(int n1, int n2, int n3, float *v) {
  return v[n1] - v[n1 * n2] + v[n1 * n2 * (n3 / 2)];
}

float f_plus_g(int n1, int n2, int n3, float *v) {
  return f(n1, n2, n3, v) + g(n1, n2, n3, v);
}

void test_qoi_basic_norm_properties(
    const mgard::TensorMeshHierarchy<3, float> &hierarchy) {
  const std::array<size_t, 3> shape = hierarchy.meshes.at(hierarchy.L).shape;
  const int n1 = shape.at(0);
  const int n2 = shape.at(1);
  const int n3 = shape.at(2);
  const std::vector<float> smoothness_parameters = {
      -1.5, -1.0, -0.5,
      0,    1e-9, 0.5,
      1.0,  1.5,  std::numeric_limits<float>::infinity()};

  SECTION("absolute homogeneity") {
    TrialTracker tracker;
    for (const float s : smoothness_parameters) {
      tracker += mgard::norm(n1, n2, n3, scaled_f, s) ==
                 Approx(std::abs(alpha) * mgard::norm(n1, n2, n3, f, s));
    }
    REQUIRE(tracker);
  }

  SECTION("triangle inequality") {
    TrialTracker tracker;
    for (const float s : smoothness_parameters) {
      tracker +=
          mgard::norm(n1, n2, n3, f, s) + mgard::norm(n1, n2, n3, g, s) >=
          mgard::norm(n1, n2, n3, f_plus_g, s);
    }
    REQUIRE(tracker);
  }
}

} // namespace

TEST_CASE("QoI basic norm properties", "[norms]") {
  {
    const mgard::TensorMeshHierarchy<3, float> hierarchy({9, 5, 9});
    test_qoi_basic_norm_properties(hierarchy);
  }

  {
    const mgard::TensorMeshHierarchy<3, float> hierarchy({6, 7, 8});
    test_qoi_basic_norm_properties(hierarchy);
  }
}

namespace {

void populate_f_nodal_values(
    const mgard::TensorMeshHierarchy<3, float> &hierarchy, float *const w) {
  for (const mgard::TensorNode<3, float> node : hierarchy.nodes(hierarchy.L)) {
    hierarchy.at(w, node.multiindex) =
        std::exp(-std::pow(node.coordinates.at(0) - 0.1, 2) / 0.2 -
                 std::pow(node.coordinates.at(1) + 0.2, 2) / 0.5 -
                 std::pow(node.coordinates.at(2) - 0.7, 2) / 0.1);
  }
}

float dot_with_f(const int n1, const int n2, const int n3, float *v) {
  const mgard::TensorMeshHierarchy<3, float> hierarchy(
      {static_cast<std::size_t>(n1), static_cast<std::size_t>(n2),
       static_cast<std::size_t>(n3)});
  std::vector<float> w_(hierarchy.ndof());
  float *const w = w_.data();
  populate_f_nodal_values(hierarchy, w);
  const mgard::TensorMassMatrix<3, float> M(hierarchy, hierarchy.L);
  M(w);
  return blas::dotu(hierarchy.ndof(), w, v);
}

} // namespace

TEST_CASE("Riesz representative norm", "[norms]") {
  const mgard::TensorMeshHierarchy<3, float> hierarchy({15, 12, 13});
  const std::array<std::size_t, 3> &SHAPE = hierarchy.meshes.back().shape;
  std::vector<float> w_(hierarchy.ndof());
  float *const w = w_.data();
  populate_f_nodal_values(hierarchy, w);
  const std::vector<float> smoothness_parameters = {-1.5, -1, -0.5, 0,
                                                    0.5,  1,  1.5};
  for (const float s : smoothness_parameters) {
    REQUIRE(mgard::norm(hierarchy, w, s) ==
            Approx(mgard::norm(SHAPE.at(0), SHAPE.at(1), SHAPE.at(2),
                               dot_with_f, s))
                .epsilon(1e-4));
  }
}
