#include "catch2/catch.hpp"

#include <array>
#include <numeric>

#include "TensorMeshHierarchy.hpp"
#include "TensorMultilevelCoefficientQuantizer.hpp"
#include "norms.hpp"

#include "testing_random.hpp"
#include "testing_utilities.hpp"

namespace {

template <std::size_t N, typename Real, typename Int>
void test_mc_quantization_iteration(const std::array<std::size_t, N> shape,
                                    const Real s, const Real tolerance) {
  const mgard::TensorMeshHierarchy<N, Real> hierarchy(shape);
  const std::size_t ndof = hierarchy.ndof();

  {
    std::vector<Real> u_(ndof);
    std::iota(u_.begin(), u_.end(), -20);
    Real *const u = u_.data();

    const mgard::TensorMultilevelCoefficientQuantizer<N, Real, Int> quantizer(
        hierarchy, s, tolerance);
    std::size_t count = 0;
    for (const Int n : quantizer(u)) {
      static_cast<void>(n);
      ++count;
    }
    REQUIRE(count == ndof);
  }

  {
    std::vector<Int> n_(ndof);
    std::iota(n_.begin(), n_.end(), 10);

    const mgard::TensorMultilevelCoefficientDequantizer<N, Int, Real>
        dequantizer(hierarchy, s, tolerance);
    std::size_t count = 0;
    for (const Real x : dequantizer(n_.begin(), n_.end())) {
      static_cast<void>(x);
      ++count;
    }
    REQUIRE(count == ndof);
  }
}

} // namespace

TEST_CASE("tensor multilevel coefficient (de)quantization iteration",
          "[TensorMultilevelCoefficientQuantizer]") {
  test_mc_quantization_iteration<1, float, int>(
      {65}, std::numeric_limits<float>::infinity(), 0.1);
  test_mc_quantization_iteration<2, double, long int>(
      {22, 19}, std::numeric_limits<double>::infinity(), 0.01);
  test_mc_quantization_iteration<3, float, long int>(
      {10, 9, 9}, std::numeric_limits<float>::infinity(), 0.25);
}

namespace {

template <std::size_t N, typename Real, typename Int>
void test_mc_dequantization_inversion(const std::array<std::size_t, N> shape,
                                      const Real s, const Real tolerance,
                                      std::default_random_engine &generator) {
  const mgard::TensorMeshHierarchy<N, Real> hierarchy(shape);
  const std::size_t ndof = hierarchy.ndof();

  const mgard::TensorMultilevelCoefficientQuantizer<N, Real, Int> quantizer(
      hierarchy, s, tolerance);
  const mgard::TensorMultilevelCoefficientDequantizer<N, Int, Real> dequantizer(
      hierarchy, s, tolerance);

  std::vector<Int> v_(ndof);
  // Quantization does not invert dequantization when `n` is very large.
  // (The condition involves `quantum` too, I'm sure.) Example:
  //   const int n = 80000001;
  //   const float quantum = 1e-4;
  //   const float dequantized = n * quantum;
  //   // This isn't exactly how quantization is done currently.
  //   const int requantized = dequantized / quantum;
  //   // `requantized` is 80000000 (not equal to `n`).
  std::uniform_int_distribution<Int> n_distribution(-1000, 1000);
  for (Int &n : v_) {
    n = n_distribution(generator);
  }

  using QntzdIt = typename std::vector<Int>::iterator;
  using DqntzrIt = typename mgard::TensorMultilevelCoefficientDequantizer<
      N, Int, Real>::template iterator<QntzdIt>;
  const mgard::RangeSlice<DqntzrIt> dequantized_v =
      dequantizer(v_.begin(), v_.end());
  const std::vector<Real> dequantized(dequantized_v.begin(),
                                      dequantized_v.end());

  TrialTracker tracker;
  typename std::vector<Int>::const_iterator p = v_.begin();
  for (const Int n : quantizer(dequantized.data())) {
    const Int original = *p++;
    tracker += n == original;
  }
  REQUIRE(tracker);
}

template <std::size_t N, typename Real, typename Int>
void test_mc_quantization_error(const std::array<std::size_t, N> shape,
                                const Real s, const Real tolerance,
                                std::default_random_engine &generator) {
  std::uniform_real_distribution<Real> nodal_spacing_distribution(1, 1.1);
  const mgard::TensorMeshHierarchy<N, Real> hierarchy =
      hierarchy_with_random_spacing(generator, nodal_spacing_distribution,
                                    shape);
  const std::size_t ndof = hierarchy.ndof();

  const mgard::TensorMultilevelCoefficientQuantizer<N, Real, Int> quantizer(
      hierarchy, s, tolerance);
  const mgard::TensorMultilevelCoefficientDequantizer<N, Int, Real> dequantizer(
      hierarchy, s, tolerance);

  std::vector<Real> u_(ndof);
  Real *const u = u_.data();

  std::uniform_real_distribution<Real> nodal_coefficient_distribution(-100,
                                                                      100);
  for (Real &x : u_) {
    x = nodal_coefficient_distribution(generator);
  }

  {
    using QntzrIt =
        typename mgard::TensorMultilevelCoefficientQuantizer<N, Real,
                                                             Int>::iterator;
    const mgard::RangeSlice<QntzrIt> quantized_u = quantizer(u);
    const std::vector<Int> quantized(quantized_u.begin(), quantized_u.end());

    std::vector<Real> error_(ndof);
    Real *const error = error_.data();

    typename std::vector<Real>::const_iterator p = u_.begin();
    typename std::vector<Real>::iterator q = error_.begin();
    for (const Real x : dequantizer(quantized.begin(), quantized.end())) {
      *q++ = *p++ - x;
    }
    REQUIRE(mgard::norm(hierarchy, error, s) <= tolerance);
  }
}

} // namespace

TEST_CASE("tensor multilevel coefficient (de)quantization inversion",
          "[TensorMultilevelCoefficientQuantizer]") {
  std::default_random_engine generator;
  // TODO: Test other smoothness parameters once (de)quantization support is
  // extended.
  const std::vector<float> smoothness_parameters = {
      std::numeric_limits<float>::infinity()};
  const std::vector<float> tolerances = {0.001, 0.01, 0.1, 1};

  for (const float s : smoothness_parameters) {
    for (const float tolerance : tolerances) {
      test_mc_dequantization_inversion<1, float, int>({65}, s, tolerance,
                                                      generator);
      test_mc_dequantization_inversion<2, float, int>({22, 17}, s, tolerance,
                                                      generator);
      test_mc_dequantization_inversion<3, float, int>({8, 8, 9}, s, tolerance,
                                                      generator);
    }
  }
}

TEST_CASE("tensor multilevel coefficient (de)quantization error",
          "[TensorMultilevelCoefficientQuantizer]") {
  std::default_random_engine generator;
  // TODO: Test other smoothness parameters once (de)quantization support is
  // extended.
  const std::vector<double> smoothness_parameters = {
      std::numeric_limits<double>::infinity()};
  const std::vector<double> tolerances = {0.5, 0.25, 0.015625};
  for (const double s : smoothness_parameters) {
    for (const double tolerance : tolerances) {
      test_mc_quantization_error<1, double, long int>({25}, s, tolerance,
                                                      generator);
      test_mc_quantization_error<2, double, long int>({17, 9}, s, tolerance,
                                                      generator);
      test_mc_quantization_error<3, double, long int>({10, 8, 10}, s, tolerance,
                                                      generator);
    }
  }
}
