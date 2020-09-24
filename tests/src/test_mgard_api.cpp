#include "catch2/catch.hpp"

#include <cmath>
#include <cstdlib>
#include <numeric>
#include <vector>

#include "testing_random.hpp"
#include "testing_utilities.hpp"

#include "TensorNorms.hpp"
#include "blas.hpp"
#include "mgard_api.h"

namespace {

template <std::size_t N, typename Real>
void test_compression_decompression(
    const std::array<std::size_t, N> &shape,
    // Used for generating the data.
    const Real s, std::default_random_engine &generator,
    const std::vector<Real> &smoothness_parameters,
    const std::vector<Real> &tolerances) {
  std::uniform_real_distribution<Real> node_spacing_distribution(1, 2);
  const mgard::TensorMeshHierarchy<N, Real> hierarchy =
      hierarchy_with_random_spacing(generator, node_spacing_distribution,
                                    shape);
  const std::size_t ndof = hierarchy.ndof();

  Real *const u = static_cast<Real *>(std::malloc(ndof * sizeof(*u)));
  generate_reasonable_function(hierarchy, s, generator, u);
  Real *const v = static_cast<Real *>(std::malloc(ndof * sizeof(*v)));
  Real *const error = static_cast<Real *>(std::malloc(ndof * sizeof(*error)));

  TrialTracker tracker;
  for (const Real s : smoothness_parameters) {
    for (const Real tolerance : tolerances) {
      blas::copy(ndof, u, v);
      blas::copy(ndof, u, error);

      const mgard::CompressedDataset<N, Real> compressed =
          mgard::compress(hierarchy, v, s, tolerance);
      const mgard::DecompressedDataset<N, Real> decompressed =
          mgard::decompress(compressed);

      blas::axpy(ndof, static_cast<Real>(-1), decompressed.data(), error);
      tracker += mgard::norm(hierarchy, error, s) <= tolerance;
    }
  }
  REQUIRE(tracker);

  std::free(error);
  std::free(v);
  std::free(u);
}

} // namespace

TEMPLATE_TEST_CASE("compression followed by decompression", "[mgard_api]",
                   float, double) {
  std::default_random_engine generator;
  const std::vector<TestType> smoothness_parameters = {
      -1.5, -0.5, 0.0, 0.5, 1.5, std::numeric_limits<TestType>::infinity()};
  const std::vector<TestType> tolerances = {1, 0.1, 0.01, 0.001};

  test_compression_decompression<1, TestType>(
      {65}, 0.1, generator, smoothness_parameters, tolerances);
  test_compression_decompression<1, TestType>(
      {55}, 1.0, generator, smoothness_parameters, tolerances);
  test_compression_decompression<2, TestType>(
      {17, 19}, 0.3, generator, smoothness_parameters, tolerances);
  test_compression_decompression<2, TestType>(
      {18, 18}, 0.5, generator, smoothness_parameters, tolerances);
  test_compression_decompression<3, TestType>(
      {10, 5, 12}, 0, generator, smoothness_parameters, tolerances);
  test_compression_decompression<3, TestType>(
      {9, 9, 6}, 1.5, generator, smoothness_parameters, tolerances);
}

TEST_CASE("1D quadratic data", "[mgard_api]") {
  const mgard::TensorMeshHierarchy<1, double> hierarchy({64});
  const std::size_t ndof = hierarchy.ndof();
  double *const v = static_cast<double *>(std::malloc(ndof * sizeof(*v)));

  for (std::size_t i = 0; i < ndof; ++i) {
    v[i] = static_cast<double>(i * i) / ndof;
  }

  double *const error =
      static_cast<double *>(std::malloc(ndof * sizeof(*error)));
  blas::copy(ndof, v, error);

  const double s = std::numeric_limits<double>::infinity();
  const double tolerance = 0.001;
  const mgard::CompressedDataset<1, double> compressed =
      mgard::compress(hierarchy, v, s, tolerance);
  std::free(v);
  const mgard::DecompressedDataset<1, double> decompressed =
      mgard::decompress(compressed);

  blas::axpy(ndof, static_cast<double>(-1), decompressed.data(), error);
  const double achieved = mgard::norm(hierarchy, error, s);
  std::free(error);

  REQUIRE(achieved <= tolerance);
}

TEST_CASE("1D cosine data", "[mgard_api]") {
  const mgard::TensorMeshHierarchy<1, double> hierarchy({4096});
  const std::size_t ndof = hierarchy.ndof();
  double *const v = static_cast<double *>(std::malloc(ndof * sizeof(*v)));

  const double pi = 3.141592653589793;
  for (std::size_t i = 0; i < ndof; ++i) {
    v[i] = std::cos(2 * pi * i / ndof);
  }

  double *const error =
      static_cast<double *>(std::malloc(ndof * sizeof(*error)));
  blas::copy(ndof, v, error);

  const double s = std::numeric_limits<double>::infinity();
  const double tolerance = 0.000001;
  const mgard::CompressedDataset<1, double> compressed =
      mgard::compress(hierarchy, v, s, tolerance);
  std::free(v);
  const mgard::DecompressedDataset<1, double> decompressed =
      mgard::decompress(compressed);

  blas::axpy(ndof, static_cast<double>(-1), decompressed.data(), error);
  const double achieved = mgard::norm(hierarchy, error, s);
  std::free(error);

  REQUIRE(achieved <= tolerance);
}

TEST_CASE("2D cosine data", "[mgard_api]") {
  const mgard::TensorMeshHierarchy<2, float> hierarchy({256, 16});
  const std::size_t ndof = hierarchy.ndof();
  float *const v = static_cast<float *>(std::malloc(ndof * sizeof(*v)));

  for (const mgard::TensorNode<2, float> node : hierarchy.nodes(hierarchy.L)) {
    hierarchy.at(v, node.multiindex) =
        std::cos(12 * node.coordinates.at(0) - 5 * node.coordinates.at(1));
  }

  float *const error = static_cast<float *>(std::malloc(ndof * sizeof(*error)));
  blas::copy(ndof, v, error);

  const float s = std::numeric_limits<float>::infinity();
  const float tolerance = 0.001;
  const mgard::CompressedDataset<2, float> compressed =
      mgard::compress(hierarchy, v, s, tolerance);
  std::free(v);
  const mgard::DecompressedDataset<2, float> decompressed =
      mgard::decompress(compressed);

  blas::axpy(ndof, static_cast<float>(-1), decompressed.data(), error);
  const float achieved = mgard::norm(hierarchy, error, s);
  std::free(error);

  REQUIRE(achieved <= tolerance);
}
