#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <limits>
#include <random>
#include <string>

#include "moab/Core.hpp"
#include "moab/EntityHandle.hpp"
#include "moab/EntityType.hpp"

#include "testing_utilities.hpp"

#include "MeshLevel.hpp"
#include "MultilevelCoefficientQuantizer.hpp"
#include "UniformMeshHierarchy.hpp"
#include "data.hpp"
#include "norms.hpp"

TEST_CASE("multilevel coefficient (de)quantization iteration",
          "[MultilevelCoefficientQuantizer]") {
  const std::string filename = GENERATE("triangle.msh", "tetrahedron.msh");
  moab::Core mbcore;
  const moab::ErrorCode ecode = mbcore.load_file(mesh_path(filename).c_str());
  require_moab_success(ecode);
  const mgard::MeshLevel _mesh(mbcore);
  const std::size_t L = 4;
  mgard::UniformMeshHierarchy hierarchy(_mesh, L);
  const std::size_t N = hierarchy.ndof();

  std::vector<double> u_(N);
  std::random_device device;
  std::default_random_engine generator(device());
  std::uniform_real_distribution<double> distribution(-3, 0);
  for (double &value : u_) {
    value = distribution(generator);
  }
  const mgard::MultilevelCoefficients<double> u(u_.data());

  const float s = 0.25;
  const float tolerance = 0.01;
  const mgard::MultilevelCoefficientQuantizer<double, int> quantizer(
      hierarchy, s, tolerance);
  const mgard::MultilevelCoefficientDequantizer<int, double> dequantizer(
      hierarchy, s, tolerance);

  std::vector<int> quantized;
  for (const int n : quantizer(u)) {
    quantized.push_back(n);
  }
  REQUIRE(quantized.size() == N);

  std::vector<double> dequantized;
  for (const double x : dequantizer(quantized.begin(), quantized.end())) {
    dequantized.push_back(x);
  }
  REQUIRE(dequantized.size() == N);
}

TEST_CASE("quantization respects error bound",
          "[MultilevelCoefficientQuantizer]") {
  const std::string filename = GENERATE("lopsided.msh", "hexahedron.msh");
  moab::Core mbcore;
  const moab::ErrorCode ecode = mbcore.load_file(mesh_path(filename).c_str());
  require_moab_success(ecode);
  const mgard::MeshLevel _mesh(mbcore);
  const std::size_t L = 3;
  mgard::UniformMeshHierarchy hierarchy(_mesh, L);
  const std::size_t N = hierarchy.ndof();

  std::vector<double> u_nc_(N);
  std::random_device device;
  std::default_random_engine generator(device());
  std::uniform_real_distribution<double> distribution(-3, 0);
  for (double &value : u_nc_) {
    value = distribution(generator);
  }
  const mgard::NodalCoefficients<double> u_nc(u_nc_.data());
  const mgard::MultilevelCoefficients<double> u_mc = hierarchy.decompose(u_nc);

  const std::vector<float> smoothness_parameters = {-2, 0, 0.9};
  const std::vector<float> tolerances = {0.0327, 0.1892, 1.1};

  for (const float s : smoothness_parameters) {
    for (const float tolerance : tolerances) {
      const mgard::MultilevelCoefficientQuantizer<double, int> quantizer(
          hierarchy, s, tolerance);
      std::vector<int> quantized;
      for (const int n : quantizer(u_mc)) {
        quantized.push_back(n);
      }

      std::vector<double> error_mc_;
      const mgard::MultilevelCoefficientDequantizer<int, double> dequantizer(
          hierarchy, s, tolerance);
      double const *p = u_mc.data;
      for (const double x : dequantizer(quantized.begin(), quantized.end())) {
        error_mc_.push_back(*p++ - x);
      }
      const mgard::MultilevelCoefficients<double> error_mc(error_mc_.data());
      const mgard::NodalCoefficients<double> error_nc =
          hierarchy.recompose(error_mc);

      REQUIRE(mgard::norm(error_nc, hierarchy, s) <= tolerance);
    }
  }
}

TEST_CASE("multilevel coefficient (de)quantization inversion",
          "[MultilevelCoefficientQuantizer]") {
  const std::string filename = GENERATE("triangle.msh", "tetrahedron.msh");
  moab::Core mbcore;
  const moab::ErrorCode ecode = mbcore.load_file(mesh_path(filename).c_str());
  require_moab_success(ecode);
  const mgard::MeshLevel _mesh(mbcore);
  const std::size_t L = 3;
  mgard::UniformMeshHierarchy hierarchy(_mesh, L);
  const std::size_t N = hierarchy.ndof();

  std::vector<short int> prequantized(N);
  std::random_device device;
  std::default_random_engine generator(device());
  std::uniform_int_distribution<short int> distribution(
      std::numeric_limits<short int>::min(),
      std::numeric_limits<short int>::max());
  for (short int &value : prequantized) {
    value = distribution(generator);
  }

  const std::vector<float> smoothness_parameters = {-1.25, 0.25, 0.75, 1.5};
  const std::vector<float> tolerances = {0.001, 0.01, 0.1, 1};

  for (const float s : smoothness_parameters) {
    for (const float tolerance : tolerances) {
      const mgard::MultilevelCoefficientDequantizer<short int, float>
          dequantizer(hierarchy, s, tolerance);
      const mgard::MultilevelCoefficientQuantizer<float, short int> quantizer(
          hierarchy, s, tolerance);
      std::vector<float> dequantized;
      std::vector<short int> requantized;
      for (const float x :
           dequantizer(prequantized.begin(), prequantized.end())) {
        dequantized.push_back(x);
      }
      const mgard::MultilevelCoefficients<float> u(dequantized.data());
      for (const short int n : quantizer(u)) {
        requantized.push_back(n);
      }
      TrialTracker tracker;
      for (std::size_t i = 0; i < N; ++i) {
        tracker += prequantized.at(i) == requantized.at(i);
      }
      REQUIRE(tracker);
    }
  }
}
