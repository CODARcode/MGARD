#include "catch2/catch.hpp"

#include <cmath>
#include <cstddef>

#include <limits>
#include <random>

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
