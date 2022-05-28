#include "catch2/catch_test_macros.hpp"

#include <climits>

#include <algorithm>
#include <random>

#include "testing_utilities.hpp"

#include "huffman.hpp"

namespace {

void test_encoding_regression(long int *const quantized, const std::size_t N) {
  long int *const quantized_new = new long int[N];
  std::copy(quantized, quantized + N, quantized_new);

  unsigned char *hit;
  unsigned char *missed;
  unsigned char *frequencies;
  std::size_t bits_hit;
  std::size_t bytes_missed;
  std::size_t bytes_frequencies;
  mgard::huffman_encoding(quantized, N, hit, bits_hit, missed, bytes_missed,
                          frequencies, bytes_frequencies);

  unsigned char *hit_new;
  unsigned char *missed_new;
  unsigned char *frequencies_new;
  std::size_t bits_hit_new;
  std::size_t bytes_missed_new;
  std::size_t bytes_frequencies_new;
  mgard::huffman_encoding(quantized_new, N, hit_new, bits_hit_new, missed_new,
                          bytes_missed_new, frequencies_new,
                          bytes_frequencies_new);

  REQUIRE(bits_hit_new == bits_hit);
  const std::size_t bytes_hit = (bits_hit + CHAR_BIT - 1) / CHAR_BIT;
  REQUIRE(std::equal(hit, hit + bytes_hit, hit_new));

  REQUIRE(bytes_missed_new == bytes_missed);
  REQUIRE(std::equal(missed, missed + bytes_missed, missed_new));

  REQUIRE(bytes_frequencies_new == bytes_frequencies);
  REQUIRE(std::equal(frequencies, frequencies + bytes_frequencies,
                     frequencies_new));

  delete[] quantized_new;
}

void test_encoding_regression_constant(const std::size_t N, const long int q) {
  long int *const quantized = new long int[N];
  std::fill(quantized, quantized + N, q);
  test_encoding_regression(quantized, N);
  delete[] quantized;
}

//! Function object to generate periodict data.
struct PeriodicGenerator {
  //! Constructor.
  //!
  //!\param value Starting value.
  //!\param period Generator period.
  PeriodicGenerator(const std::size_t period, const long int value)
      : period(period), value(value), ncalls(0) {}

  //! Generator period.
  std::size_t period;

  //! Starting value.
  long int value;

  //! Number of times `operator()` has been called.
  std::size_t ncalls;

  long int operator()() {
    return value + static_cast<long int>(ncalls++ % period);
  }
};

void test_encoding_regression_periodic(const std::size_t N, const long int q,
                                       const std::size_t period) {
  long int *const quantized = new long int[N];
  std::generate(quantized, quantized + N, PeriodicGenerator(period, q));
  test_encoding_regression(quantized, N);
  delete[] quantized;
}

void test_encoding_regression_random(const std::size_t N, const long int a,
                                     const long int b,
                                     std::default_random_engine &gen) {
  std::uniform_int_distribution<long int> dis(a, b);
  long int *const quantized = new long int[N];
  std::generate(quantized, quantized + N, [&] { return dis(gen); });
  test_encoding_regression(quantized, N);
  delete[] quantized;
}

} // namespace

TEST_CASE("encoding regression", "[huffman]") {
  SECTION("constant data") {
    test_encoding_regression_constant(10, 0);
    test_encoding_regression_constant(100, 732);
    test_encoding_regression_constant(1000, -10);
  }

  SECTION("periodic data") {
    test_encoding_regression_periodic(10, -3, 3);
    test_encoding_regression_periodic(100, 0, 10);
    test_encoding_regression_periodic(1000, 51, 17);
  }

  SECTION("random data") {
    std::default_random_engine gen(131051);
    test_encoding_regression_random(10, 0, 1, gen);
    test_encoding_regression_random(100, -15, -5, gen);
    test_encoding_regression_random(1000, std::numeric_limits<int>::min(),
                                    std::numeric_limits<int>::max(), gen);
    test_encoding_regression_random(10000, -100, 100, gen);
  }
}
