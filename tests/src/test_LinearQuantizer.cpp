#include "catch2/catch.hpp"

#include <cmath>
#include <cstddef>

#include <iterator>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include "LinearQuantizer.hpp"

template <typename Real, typename Int>
static void test_quantization_error(const Real quantum) {
  std::random_device device;
  std::default_random_engine generator(device());
  std::uniform_real_distribution<Real> mantissa_distribution(-1, 1);
  std::uniform_int_distribution<Int> exponent_distribution(-2, 2);
  const mgard::LinearQuantizer<Real, Int> quantizer(quantum);
  const mgard::LinearDequantizer<Int, Real> dequantizer(quantum);
  bool all_successful = true;
  for (std::size_t i = 0; i < 100; ++i) {
    const Real x = mantissa_distribution(generator) *
                   std::pow(10, exponent_distribution(generator));
    const Int n = quantizer(x);
    const Real y = dequantizer(n);
    all_successful = all_successful && std::abs(x - y) < quantum;
  }
  REQUIRE(all_successful);
}

template <typename Real, typename Int>
static void test_quantization_domain(const Real quantum) {
  const mgard::LinearQuantizer<Real, Int> quantizer(quantum);
  // The largest and smallest values `dequantizer.operator()` will output.
  const Real largest_output = quantum * std::numeric_limits<Int>::max();
  const Real smallest_output = quantum * std::numeric_limits<Int>::min();
  // Haven't tested what happens when `quantum` is so large that all (finite)
  // values can be quantized.
  REQUIRE_NOTHROW(quantizer(largest_output));
  REQUIRE_NOTHROW(quantizer(largest_output + 0.499 * quantum));
  REQUIRE_THROWS(quantizer(largest_output + 0.501 * quantum));
  REQUIRE_NOTHROW(quantizer(smallest_output));
  REQUIRE_NOTHROW(quantizer(smallest_output - 0.499 * quantum));
  REQUIRE_THROWS(quantizer(smallest_output - 0.501 * quantum));
}

template <typename Real, typename Int>
static void test_dequantization_inversion(const Real quantum,
                                          const std::vector<Int> ns) {
  std::vector<Real> xs;
  const mgard::LinearDequantizer<Int, Real> dequantizer(quantum);
  for (const Int n : ns) {
    xs.push_back(dequantizer(n));
  }

  const mgard::LinearQuantizer<Real, Int> quantizer(quantum);
  typename std::vector<Int>::const_iterator p = ns.begin();
  bool all_equal = true;
  for (const Real x : xs) {
    all_equal = all_equal && quantizer(x) == *p++;
  }
  REQUIRE(all_equal);
}

TEST_CASE("quantization error", "[LinearQuantizer]") {
  test_quantization_error<float, int>(0.01);
  test_quantization_error<float, long int>(2.4);
  test_quantization_error<double, int>(0.5);
  test_quantization_error<double, long int>(0.89327);
}

TEST_CASE("quantization exceptions", "[LinearQuantizer]") {
  SECTION("quantum must be positive") {
    REQUIRE_THROWS(mgard::LinearQuantizer<float, long int>(-12.2));
    REQUIRE_THROWS(mgard::LinearQuantizer<double, int>(0));
    REQUIRE_THROWS(mgard::LinearDequantizer<long int, float>(-0.742));
    REQUIRE_THROWS(mgard::LinearDequantizer<int, double>(0));
  }

  SECTION("quantization domain") {
    test_quantization_error<float, short int>(0.1);
    test_quantization_error<float, short int>(0.1);
    test_quantization_error<double, long int>(24.2189);
  }
}

TEST_CASE("quantization of a range", "[LinearQuantizer]") {
  // This originally tested a quantizer applied to a range 'in its entirety'
  // (rather than element-by-element).
  SECTION("basic quantization iteration") {
    const mgard::LinearQuantizer<double, int> quantizer(0.5);
    const std::vector<double> xs = {0, 2.5, 2.49, 2.51, -10.5};
    std::vector<int> ns;
    for (const double x : xs) {
      ns.push_back(quantizer(x));
    }
    REQUIRE(ns == std::vector<int>({0, 5, 5, 5, -21}));
  }

  SECTION("basic dequantization iteration") {
    const mgard::LinearDequantizer<short int, float> dequantizer(1.25);
    const std::vector<short int> ns = {-28, 0, -5, 2387};
    std::vector<float> xs;
    for (const short int n : ns) {
      xs.push_back(dequantizer(n));
    }
    REQUIRE(xs == std::vector<float>({-35, 0, -6.25, 2983.75}));
  }

  SECTION("quantization inverts dequantization") {
    {
      test_dequantization_inversion<float, int>(
          0.03, {2, 18, 2897, -3289, 21782, 0, -12, -1783});
    }
    {
      test_dequantization_inversion<double, short>(
          0.238, {4, 66, 83, -51, -833, -328, 0, 0, -327});
    }
  }
}
