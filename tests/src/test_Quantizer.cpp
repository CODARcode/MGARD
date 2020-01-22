#include "catch2/catch.hpp"

#include <cmath>
#include <cstddef>

#include <iterator>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include "Quantizer.hpp"

template <typename Real, typename Int>
static void test_quantization_error(const Real quantum) {
  std::random_device device;
  std::default_random_engine generator(device());
  std::uniform_real_distribution<Real> mantissa_distribution(-1, 1);
  std::uniform_int_distribution<Int> exponent_distribution(-2, 2);
  const mgard::Quantizer<Real, Int> quantizer(quantum);
  bool all_successful = true;
  for (std::size_t i = 0; i < 100; ++i) {
    const Real x = mantissa_distribution(generator) *
                   std::pow(10, exponent_distribution(generator));
    const Int n = quantizer.quantize(x);
    const Real y = quantizer.dequantize(n);
    all_successful = all_successful && std::abs(x - y) < quantum;
  }
  REQUIRE(all_successful);
}

template <typename Real, typename Int>
static void test_quantization_domain(const Real quantum) {
  const mgard::Quantizer<Real, Int> quantizer(quantum);
  // The largest and smallest values `quantizer.dequantize` will output.
  const Real largest_output = quantum * std::numeric_limits<Int>::max();
  const Real smallest_output = quantum * std::numeric_limits<Int>::min();
  // Haven't tested what happens when `quantum` is so large that all (finite)
  // values can be quantized.
  REQUIRE_NOTHROW(quantizer.quantize(largest_output));
  REQUIRE_NOTHROW(quantizer.quantize(largest_output + 0.499 * quantum));
  REQUIRE_THROWS(quantizer.quantize(largest_output + 0.501 * quantum));
  REQUIRE_NOTHROW(quantizer.quantize(smallest_output));
  REQUIRE_NOTHROW(quantizer.quantize(smallest_output - 0.499 * quantum));
  REQUIRE_THROWS(quantizer.quantize(smallest_output - 0.501 * quantum));
}

template <typename Real, typename Int>
static void
test_iteration_inversion(const mgard::Quantizer<Real, Int> &quantizer,
                         const std::vector<Int> ns) {
  using It = typename std::vector<Int>::const_iterator;
  const mgard::DequantizedRange<Real, Int, It> dequantized =
      quantizer.dequantize(ns.begin(), ns.end());

  using Jt = typename mgard::DequantizedRange<Real, Int, It>::iterator;
  const mgard::QuantizedRange<Real, Int, Jt> quantized =
      quantizer.quantize(dequantized.begin(), dequantized.end());

  typename std::vector<Int>::const_iterator p = ns.begin();
  bool all_equal = true;
  for (const Int n : quantized) {
    all_equal = all_equal && n == *p++;
  }
  REQUIRE(all_equal);
}

TEST_CASE("quantization error", "[Quantizer]") {
  test_quantization_error<float, int>(0.01);
  test_quantization_error<float, long int>(2.4);
  test_quantization_error<double, int>(0.5);
  test_quantization_error<double, long int>(0.89327);
}

TEST_CASE("quantization exceptions", "[Quantizer]") {
  SECTION("quantum must be positive") {
    REQUIRE_THROWS(mgard::Quantizer<float, long int>(-12.2));
    REQUIRE_THROWS(mgard::Quantizer<double, int>(0));
  }

  SECTION("quantization domain", "[Quantizer]") {
    test_quantization_error<float, short int>(0.1);
    test_quantization_error<float, short int>(0.1);
    test_quantization_error<double, long int>(24.2189);
  }
}

TEST_CASE("quantization iterator", "[Quantizer]") {
  SECTION("basic quantization iteration") {
    const mgard::Quantizer<double, int> quantizer(0.5);
    const std::vector<double> xs = {0, 2.5, 2.49, 2.51, -10.5};
    std::vector<int> ns;
    for (const int n : quantizer.quantize(xs.begin(), xs.end())) {
      ns.push_back(n);
    }
    REQUIRE(ns == std::vector<int>({0, 5, 5, 5, -21}));
  }

  SECTION("basic dequantization iteration") {
    const mgard::Quantizer<float, short int> quantizer(1.25);
    const std::vector<short int> ns = {-28, 0, -5, 2387};
    std::vector<float> xs;
    for (const float x : quantizer.dequantize(ns.begin(), ns.end())) {
      xs.push_back(x);
    }
    REQUIRE(xs == std::vector<float>({-35, 0, -6.25, 2983.75}));
  }

  SECTION("iteration and quantization inverts dequantization") {
    {
      const mgard::Quantizer<float, int> quantizer(0.03);
      test_iteration_inversion<float, int>(
          quantizer, {2, 18, 2897, -3289, 21782, 0, -12, -1783});
    }
    {
      const mgard::Quantizer<double, short> quantizer(8.238);
      test_iteration_inversion<double, short>(
          quantizer, {4, 66, 83, -51, -833, -328, 0, 0, -327});
    }
  }
}
