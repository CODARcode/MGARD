#include "catch2/catch.hpp"

#include <cmath>
#include <cstddef>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "moab/Core.hpp"

#include "blas.hpp"

#include "MeshLevel.hpp"
#include "UniformMeshHierarchy.hpp"
#include "data.hpp"
#include "norms.hpp"

#include "testing_random.hpp"
#include "testing_utilities.hpp"

static const double inf = std::numeric_limits<double>::infinity();

TEST_CASE("unstructured basic norm properties", "[norms]") {
  const std::string filename = GENERATE("pyramid.msh", "tetrahedron.msh");
  moab::ErrorCode ecode;
  moab::Core mbcore;
  ecode = mbcore.load_file(mesh_path(filename).c_str());
  require_moab_success(ecode);
  mgard::MeshLevel mesh(mbcore);
  mgard::UniformMeshHierarchy hierarchy(mesh, 4);
  const std::size_t N = hierarchy.ndof();
  std::vector<double> smoothness_parameters = {-1.5, -1.0, -0.5, 0.0,
                                               0.5,  1.0,  1.5,  inf};

  std::random_device device;
  std::default_random_engine generator(device());
  std::uniform_real_distribution<double> distribution(-1, 1);

  SECTION("absolute homogeneity") {
    std::vector<double> u_(N);
    // Likely not needed. Not checking now.
    std::fill(u_.begin(), u_.end(), 0);
    mgard::NodalCoefficients u(u_.data());
    {
      TrialTracker tracker;
      for (const double s : smoothness_parameters) {
        tracker += mgard::norm(u, hierarchy, s) == 0;
      }
      REQUIRE(tracker);
    }

    for (double &value : u_) {
      value = distribution(generator);
    }

    std::vector<double> copy_(N);
    mgard::NodalCoefficients copy(copy_.data());
    {
      TrialTracker tracker;
      for (const double s : smoothness_parameters) {
        blas::copy(N, u.data, copy.data);
        const double alpha = distribution(generator);
        blas::scal(N, alpha, copy.data);
        tracker += mgard::norm(copy, hierarchy, s) ==
                   Approx(std::abs(alpha) * mgard::norm(u, hierarchy, s));
      }
      REQUIRE(tracker);
    }
  }

  SECTION("triangle inequality") {
    std::vector<double> u_(N);
    std::vector<double> v_(N);
    std::vector<double> w_ = u_;
    for (double &value : u_) {
      value = distribution(generator);
    }
    for (double &value : v_) {
      value = distribution(generator);
    }
    mgard::NodalCoefficients u(u_.data());
    mgard::NodalCoefficients v(v_.data());
    mgard::NodalCoefficients w(w_.data());
    blas::axpy(N, 1.0, v.data, u.data);
    TrialTracker tracker;
    for (const double s : smoothness_parameters) {
      tracker += mgard::norm(w, hierarchy, s) <=
                 mgard::norm(u, hierarchy, s) + mgard::norm(v, hierarchy, s);
    }
    REQUIRE(tracker);
  }
}

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

TEST_CASE("comparison with Python implementation: unstructured norms",
          "[norms]") {
  moab::ErrorCode ecode;
  moab::Core mbcore;
  ecode = mbcore.load_file(mesh_path("circle.msh").c_str());
  require_moab_success(ecode);
  mgard::MeshLevel mesh(mbcore);
  mgard::UniformMeshHierarchy hierarchy(mesh, 2);
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
    u_.at(i) = std::sin(5 * x) + 2 * std::cos(32 * y) - std::sin(x * y * z);
  }
  mgard::NodalCoefficients u(u_.data());

  REQUIRE(mgard::norm(u, hierarchy, inf) == Approx(2.99974381309398));
  REQUIRE(mgard::norm(u, hierarchy, -1.5) == Approx(1.041534180771523));
  REQUIRE(mgard::norm(u, hierarchy, -1.0) == Approx(1.086120609647959));
  REQUIRE(mgard::norm(u, hierarchy, -0.5) == Approx(1.1720508380006622));
  REQUIRE(mgard::norm(u, hierarchy, 0.0) == Approx(1.3338133542219779));
  REQUIRE(mgard::norm(u, hierarchy, 1e-9) == Approx(1.3338133546552449));
  REQUIRE(mgard::norm(u, hierarchy, 0.5) == Approx(1.6305906723975383));
  REQUIRE(mgard::norm(u, hierarchy, 1.0) == Approx(2.1667011853555294));
  REQUIRE(mgard::norm(u, hierarchy, 1.5) == Approx(3.14182423518829));
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
