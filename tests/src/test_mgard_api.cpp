#include "catch2/catch_template_test_macros.hpp"
#include "catch2/catch_test_macros.hpp"

#include <cmath>
#include <cstdlib>

#include <algorithm>
#include <numeric>
#include <vector>

#include "testing_random.hpp"
#include "testing_utilities.hpp"

#include "TensorMeshHierarchy.hpp"
#include "TensorMeshHierarchyIteration.hpp"
#include "TensorNorms.hpp"
#include "blas.hpp"
#include "mgard_api.h"
#include "shuffle.hpp"

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

  Real *const buffer = static_cast<Real *>(std::malloc(ndof * sizeof(*buffer)));
  generate_reasonable_function(hierarchy, s, generator, buffer);

  Real *const u = static_cast<Real *>(std::malloc(ndof * sizeof(*u)));
  // `generate_reasonable_function` uses `TensorMeshHierarchy::at`, so the
  // function will come out shuffled. `mgard::compress` shuffles its input,
  // though, so we need to unshuffle beforehand.
  mgard::unshuffle(hierarchy, buffer, u);

  Real *const v = static_cast<Real *>(std::malloc(ndof * sizeof(*v)));
  Real *const error = static_cast<Real *>(std::malloc(ndof * sizeof(*error)));

  TrialTracker tracker;
  for (const Real s : smoothness_parameters) {
    for (const Real tolerance : tolerances) {
      blas::copy(ndof, u, v);
      blas::copy(ndof, u, error);
      // `v` and `error` are now unshuffled.

      const mgard::CompressedDataset<N, Real> compressed =
          mgard::compress(hierarchy, v, s, tolerance);
      const mgard::DecompressedDataset<N, Real> decompressed =
          mgard::decompress(compressed);
      // `decompressed.data` is unshuffled.

      blas::axpy(ndof, static_cast<Real>(-1), decompressed.data(), error);
      // `error` is calculated, but it's unshuffled. `mgard::norm` expects its
      // input to be shuffled, so we shuffle into `buffer`.
      mgard::shuffle(hierarchy, error, buffer);
      tracker += mgard::norm(hierarchy, buffer, s) <= tolerance;
    }
  }
  REQUIRE(tracker);

  std::free(error);
  std::free(v);
  std::free(u);
  std::free(buffer);
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

namespace {

template <std::size_t N, typename Real>
void test_compression_error_bound(
    const mgard::TensorMeshHierarchy<N, Real> &hierarchy, Real *const v,
    const Real s, const Real tolerance) {
  const std::size_t ndof = hierarchy.ndof();
  Real *const error = static_cast<Real *>(std::malloc(ndof * sizeof(*error)));
  blas::copy(ndof, v, error);

  const mgard::CompressedDataset<N, Real> compressed =
      mgard::compress(hierarchy, v, s, tolerance);
  const mgard::DecompressedDataset<N, Real> decompressed =
      mgard::decompress(compressed);

  blas::axpy(ndof, static_cast<Real>(-1), decompressed.data(), error);
  const Real achieved = mgard::norm(hierarchy, error, s);
  std::free(error);

  REQUIRE(achieved <= tolerance);
}

} // namespace

TEST_CASE("1D quadratic data", "[mgard_api]") {
  {
    const mgard::TensorMeshHierarchy<1, float> hierarchy({64});
    const std::size_t ndof = hierarchy.ndof();
    float *const v = static_cast<float *>(std::malloc(ndof * sizeof(*v)));
    for (std::size_t i = 0; i < ndof; ++i) {
      v[i] = static_cast<float>(i * i) / ndof;
    }

    const float s = std::numeric_limits<float>::infinity();
    const float tolerance = 0.001;
    test_compression_error_bound<1, float>(hierarchy, v, s, tolerance);

    std::free(v);
  }
  {
    const mgard::TensorMeshHierarchy<1, double> hierarchy({65});
    const std::size_t ndof = hierarchy.ndof();
    double *const v = static_cast<double *>(std::malloc(ndof * sizeof(*v)));
    for (std::size_t i = 0; i < ndof; ++i) {
      v[i] = static_cast<double>(i * i) / ndof;
    }

    const double s = std::numeric_limits<double>::infinity();
    const double tolerance = 0.01;
    test_compression_error_bound<1, double>(hierarchy, v, s, tolerance);

    std::free(v);
  }
}

TEST_CASE("3D constant data", "[mgard_api]") {
  const mgard::TensorMeshHierarchy<3, float> hierarchy({16, 16, 16});
  const std::size_t ndof = hierarchy.ndof();
  float *const v = static_cast<float *>(std::malloc(ndof * sizeof(*v)));
  std::fill(v, v + ndof, 10);

  const float s = std::numeric_limits<float>::infinity();
  const float tolerance = 0.01;
  test_compression_error_bound<3, float>(hierarchy, v, s, tolerance);

  std::free(v);
}

TEST_CASE("1D cosine data", "[mgard_api]") {
  const mgard::TensorMeshHierarchy<1, double> hierarchy({4096});
  const std::size_t ndof = hierarchy.ndof();
  double *const v = static_cast<double *>(std::malloc(ndof * sizeof(*v)));
  const double pi = 3.141592653589793;
  for (std::size_t i = 0; i < ndof; ++i) {
    v[i] = std::cos(2 * pi * i / ndof);
  }

  const double s = std::numeric_limits<double>::infinity();
  const double tolerance = 0.000001;
  test_compression_error_bound<1, double>(hierarchy, v, s, tolerance);

  std::free(v);
}

TEST_CASE("2D cosine data", "[mgard_api]") {
  const mgard::TensorMeshHierarchy<2, float> hierarchy({256, 16});
  const std::size_t ndof = hierarchy.ndof();
  float *const v = static_cast<float *>(std::malloc(ndof * sizeof(*v)));
  for (const mgard::TensorNode<2> node :
       mgard::ShuffledTensorNodeRange(hierarchy, hierarchy.L)) {
    const std::array<float, 2> xy = coordinates(hierarchy, node);
    hierarchy.at(v, node.multiindex) = std::cos(12 * xy.at(0) - 5 * xy.at(1));
  }

  const float s = std::numeric_limits<float>::infinity();
  const float tolerance = 0.001;
  test_compression_error_bound<2, float>(hierarchy, v, s, tolerance);

  std::free(v);
}
