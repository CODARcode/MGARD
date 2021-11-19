#include "catch2/catch.hpp"

#include <algorithm>
#include <iterator>
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
  const mgard::MultilevelCoefficients<double> u(u_.data());
  std::random_device device;
  std::default_random_engine generator(device());
  std::uniform_real_distribution<double> distribution(-3, 0);
  std::generate(u.data, u.data + N,
                [&]() -> double { return distribution(generator); });

  using Quantizer = mgard::MultilevelCoefficientQuantizer<double, int>;
  using Dequantizer = mgard::MultilevelCoefficientDequantizer<int, double>;

  using It = typename Quantizer::iterator;
  using Jt = typename Dequantizer::template iterator<It>;

  const float s = 0.25;
  const float tolerance = 0.01;
  const Quantizer quantizer(hierarchy, s, tolerance);
  const Dequantizer dequantizer(hierarchy, s, tolerance);

  const mgard::RangeSlice<It> quantized = quantizer(u);
  REQUIRE(static_cast<std::size_t>(
              std::distance(quantized.begin(), quantized.end())) == N);

  const mgard::RangeSlice<Jt> dequantized =
      dequantizer(quantized.begin(), quantized.end());
  REQUIRE(static_cast<std::size_t>(
              std::distance(dequantized.begin(), dequantized.end())) == N);
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
  const mgard::NodalCoefficients<double> u_nc(u_nc_.data());
  std::random_device device;
  std::default_random_engine generator(device());
  std::uniform_real_distribution<double> distribution(-3, 0);
  std::generate(u_nc.data, u_nc.data + N,
                [&]() -> double { return distribution(generator); });
  const mgard::MultilevelCoefficients<double> u_mc = hierarchy.decompose(u_nc);

  const std::vector<float> smoothness_parameters = {-2, 0, 0.9};
  const std::vector<float> tolerances = {0.0327, 0.1892, 1.1};

  using Quantizer = mgard::MultilevelCoefficientQuantizer<double, int>;
  using It = typename Quantizer::iterator;

  using Dequantizer = mgard::MultilevelCoefficientDequantizer<int, double>;
  using Jt = typename Dequantizer::template iterator<It>;

  for (const float s : smoothness_parameters) {
    for (const float tolerance : tolerances) {
      const Quantizer quantizer(hierarchy, s, tolerance);
      const mgard::RangeSlice<It> quantized = quantizer(u_mc);

      std::vector<double> error_mc_(N);
      const mgard::MultilevelCoefficients<double> error_mc(error_mc_.data());
      const Dequantizer dequantizer(hierarchy, s, tolerance);
      const mgard::RangeSlice<Jt> dequantized =
          dequantizer(quantized.begin(), quantized.end());
      std::transform(u_mc.data, u_mc.data + N, dequantized.begin(),
                     error_mc.data, std::minus<double>());
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
  std::generate(prequantized.begin(), prequantized.end(),
                [&]() -> short int { return distribution(generator); });

  const std::vector<float> smoothness_parameters = {-1.25, 0.25, 0.75, 1.5};
  const std::vector<float> tolerances = {0.001, 0.01, 0.1, 1};

  using Dequantizer = mgard::MultilevelCoefficientDequantizer<short int, float>;
  using It = std::vector<short int>::iterator;
  using Jt = typename Dequantizer::template iterator<It>;

  using Quantizer =
      mgard::MultilevelCoefficientQuantizer<const float, short int>;
  using Kt = typename Quantizer::iterator;

  for (const float s : smoothness_parameters) {
    for (const float tolerance : tolerances) {
      const Dequantizer dequantizer(hierarchy, s, tolerance);
      const mgard::RangeSlice<Jt> dequantized_ =
          dequantizer(prequantized.begin(), prequantized.end());
      const std::vector<float> dequantized(dequantized_.begin(),
                                           dequantized_.end());

      const mgard::MultilevelCoefficients<const float> u(dequantized.data());
      const Quantizer quantizer(hierarchy, s, tolerance);
      const mgard::RangeSlice<Kt> requantized = quantizer(u);

      TrialTracker tracker;
      It p = prequantized.begin();
      Kt q = requantized.begin();
      for (std::size_t i = 0; i < N; ++i) {
        tracker += *p++ == *q++;
      }
      tracker += p == prequantized.end();
      tracker += q == requantized.end();
      REQUIRE(tracker);
    }
  }
}
