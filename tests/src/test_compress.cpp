#include "catch2/catch_template_test_macros.hpp"
#include "catch2/catch_test_macros.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "testing_random.hpp"
#include "testing_utilities.hpp"

#include "TensorMeshHierarchy.hpp"
#include "TensorMeshHierarchyIteration.hpp"
#include "TensorNorms.hpp"
#include "blas.hpp"
#include "compress.hpp"
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

  const Real * new_data = (const Real *)mgard::mgard_decompress(compressed.data(), compressed.size());
      blas::axpy(ndof, static_cast<Real>(-1), new_data, error);
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

TEMPLATE_TEST_CASE("compression followed by decompression", "[compress]", float,
                   double) {
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

  const Real * new_data = (const Real *)mgard::mgard_decompress(compressed.data(), compressed.size());
  blas::axpy(ndof, static_cast<Real>(-1), new_data, error);

  const Real achieved = mgard::norm(hierarchy, error, s);
  std::free(error);

  REQUIRE(achieved <= tolerance);
}

} // namespace

TEST_CASE("1D quadratic data", "[compress]") {
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

TEST_CASE("3D constant data", "[compress]") {
  const mgard::TensorMeshHierarchy<3, float> hierarchy({16, 16, 16});
  const std::size_t ndof = hierarchy.ndof();
  float *const v = static_cast<float *>(std::malloc(ndof * sizeof(*v)));
  std::fill(v, v + ndof, 10);

  const float s = std::numeric_limits<float>::infinity();
  const float tolerance = 0.01;
  test_compression_error_bound<3, float>(hierarchy, v, s, tolerance);

  std::free(v);
}

TEST_CASE("1D cosine data", "[compress]") {
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

TEST_CASE("2D cosine data", "[compress]") {
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

namespace {

template <std::size_t N, std::size_t M, typename Real>
void test_compression_on_flat_mesh(
    const mgard::TensorMeshHierarchy<N, Real> &hierarchy, Real const *const u,
    const mgard::CompressedDataset<N, Real> &expected,
    const std::array<std::size_t, M> shape, Real *v, TrialTracker &tracker) {
  const std::size_t ndof = hierarchy.ndof();
  const mgard::TensorMeshHierarchy<M, Real> flat_hierarchy =
      make_flat_hierarchy<N, M, Real>(hierarchy, shape);
  std::copy(u, u + ndof, v);
  const mgard::CompressedDataset<M, Real> obtained =
      mgard::compress(flat_hierarchy, v, expected.s, expected.tolerance);
  tracker += expected.size() == (obtained.size() - (M - N) * 8);
std::cout << "expected.size() = " << expected.size() << " obtained.size() = " << obtained.size() - (M - N) * 8 << "\n";
  const std::size_t metadata_expected = 4 + 19 + 1 + 1 + 1 + 1 + N * 8 + 8 + 8 + 8 + 4 + 1;
  const std::size_t metadata_obtained = 4 + 19 + 1 + 1 + 1 + 1 + M * 8 + 8 + 8 + 8 + 4 + 1;;
  tracker +=
      std::memcmp(expected.data() + metadata_expected, obtained.data() + metadata_obtained, expected.size() - metadata_expected) == 0;
}

} // namespace

TEST_CASE("compressing on 'flat' meshes", "[compress]") {
  std::default_random_engine gen(799875);
  std::uniform_real_distribution<float> dis(0.01, 0.011);
  const mgard::TensorMeshHierarchy<2, float> hierarchy =
      hierarchy_with_random_spacing<2, float>(gen, dis, {31, 8});
  const std::size_t ndof = hierarchy.ndof();
  float *const u = static_cast<float *>(std::malloc(ndof * sizeof(float)));
  float *const v = static_cast<float *>(std::malloc(ndof * sizeof(float)));
  {
    const float generation_s = 1.25;
    float *const buffer =
        static_cast<float *>(std::malloc(ndof * sizeof(float)));
    generate_reasonable_function(hierarchy, generation_s, gen, buffer);
    mgard::unshuffle(hierarchy, buffer, u);
    std::free(buffer);
  }

  const std::vector<float> smoothness_parameters = {
      -1, 0, 1, std::numeric_limits<float>::infinity()};
  const std::vector<float> tolerances = {0.1, 0.01, 0.001};

  TrialTracker tracker;
  for (const float s : smoothness_parameters) {
    for (const float tolerance : tolerances) {
      const mgard::CompressedDataset<2, float> expected =
          mgard::compress(hierarchy, u, s, tolerance);

      test_compression_on_flat_mesh<2, 3, float>(hierarchy, u, expected,
                                                 {31, 8, 1}, v, tracker);
      test_compression_on_flat_mesh<2, 3, float>(hierarchy, u, expected,
                                                 {31, 1, 8}, v, tracker);
      test_compression_on_flat_mesh<2, 5, float>(hierarchy, u, expected,
                                                 {1, 31, 1, 1, 8}, v, tracker);
    }
  }
  REQUIRE(tracker);

  std::free(v);
  std::free(u);
}

namespace {

template <std::size_t N, std::size_t M, typename Real>
void test_decompression_on_flat_mesh(
    const mgard::TensorMeshHierarchy<N, Real> &hierarchy,
    const mgard::CompressedDataset<N, Real> &compressed,
    const mgard::DecompressedDataset<N, Real> &expected,
    const std::array<std::size_t, M> shape, TrialTracker &tracker) {
  const std::size_t ndof = hierarchy.ndof();
  const mgard::TensorMeshHierarchy<M, Real> flat_hierarchy =
      make_flat_hierarchy<N, M, Real>(hierarchy, shape);
  void *const data = new unsigned char[compressed.size()];
  std::memcpy(data, compressed.data(), compressed.size());
  const mgard::CompressedDataset<M, Real> flat_compressed(
      flat_hierarchy, compressed.s, compressed.tolerance, data,
      compressed.size());
  const mgard::DecompressedDataset<M, Real> obtained =
      mgard::decompress(flat_compressed);
  // Originally we compared `expected.data()` and `obtained.data()` bitwise.
  // When using `-ffast-math` after precomputing the shuffled indices, though,
  // we were getting some small discrepancies.
  Real const *const p = expected.data();
  Real const *const q = obtained.data();
  for (std::size_t i = 0; i < ndof; ++i) {
    const Real expected_ = p[i];
    const Real obtained_ = q[i];
    tracker += std::abs((expected_ - obtained_) / expected_) < 1e-6;
  }
}

} // namespace
#if 0
TEST_CASE("decompressing on 'flat' meshes", "[compress]") {
  std::default_random_engine gen(780037);
  std::uniform_real_distribution<double> dis(2, 3);
  const mgard::TensorMeshHierarchy<3, double> hierarchy =
      hierarchy_with_random_spacing<3, double>(gen, dis, {6, 5, 7});
  const std::size_t ndof = hierarchy.ndof();
  double *const u = static_cast<double *>(std::malloc(ndof * sizeof(double)));
  double *const v = static_cast<double *>(std::malloc(ndof * sizeof(double)));
  {
    const double generation_s = 0.25;
    double *const buffer =
        static_cast<double *>(std::malloc(ndof * sizeof(double)));
    generate_reasonable_function(hierarchy, generation_s, gen, buffer);
    mgard::unshuffle(hierarchy, buffer, u);
    std::free(buffer);
  }

  const std::vector<double> smoothness_parameters = {
      -1.25, 0, 0.5, std::numeric_limits<double>::infinity()};
  const std::vector<double> tolerances = {0.1, 0.01, 0.001};

  TrialTracker tracker;
  for (const double s : smoothness_parameters) {
    for (const double tolerance : tolerances) {
      const mgard::CompressedDataset<3, double> compressed =
          mgard::compress(hierarchy, u, s, tolerance);
      const mgard::DecompressedDataset<3, double> expected =
          mgard::decompress(compressed);

      test_decompression_on_flat_mesh<3, 4, double>(
          hierarchy, compressed, expected, {6, 5, 1, 7}, tracker);
      test_decompression_on_flat_mesh<3, 4, double>(
          hierarchy, compressed, expected, {1, 6, 5, 7}, tracker);
      test_decompression_on_flat_mesh<3, 5, double>(
          hierarchy, compressed, expected, {6, 1, 5, 7, 1}, tracker);
    }
  }
  REQUIRE(tracker);

  std::free(v);
  std::free(u);
}
#endif
