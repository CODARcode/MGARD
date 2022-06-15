#include "catch2/catch_template_test_macros.hpp"
#include "catch2/catch_test_macros.hpp"

#include <climits>
#include <cstdint>

#include <algorithm>
#include <numeric>
#include <random>

#include "testing_utilities.hpp"

#include "huffman.hpp"
#include "huffman_regression.hpp"

namespace {

void test_encoding_regression(long int const *const quantized,
                              const std::size_t N) {
  const mgard::HuffmanEncodedStream out =
      mgard::regression::huffman_encoding(quantized, N);
  const mgard::HuffmanEncodedStream out_ =
      mgard::huffman_encoding(quantized, N);

  unsigned char const *const hit = out.hit.data.get();
  REQUIRE(out_.nbits == out.nbits);
  const std::size_t nbytes = (out.nbits + CHAR_BIT - 1) / CHAR_BIT;
  REQUIRE(std::equal(hit, hit + nbytes, out_.hit.data.get()));

  unsigned char const *const missed = out.missed.data.get();
  const std::size_t nmissed = out.missed.size;
  REQUIRE(out_.missed.size == nmissed);
  REQUIRE(std::equal(missed, missed + nmissed, out_.missed.data.get()));

  unsigned char const *const frequencies = out.frequencies.data.get();
  const std::size_t nfrequencies = out.frequencies.size;
  REQUIRE(out_.frequencies.size == nfrequencies);
  REQUIRE(std::equal(frequencies, frequencies + nfrequencies,
                     out_.frequencies.data.get()));
}

void test_decoding_regression(long int const *const quantized,
                              const std::size_t N) {
  const mgard::HuffmanEncodedStream encoded =
      mgard::regression::huffman_encoding(quantized, N);
  const mgard::HuffmanEncodedStream encoded_ =
      mgard::regression::huffman_encoding(quantized, N);

  const mgard::MemoryBuffer<long int> out =
      mgard::regression::huffman_decoding(encoded);
  const mgard::MemoryBuffer<long int> out_ = mgard::huffman_decoding(encoded_);

  REQUIRE(out.size == out_.size);
  REQUIRE(out.size == N);
  long int const *const p = out.data.get();
  long int const *const p_ = out_.data.get();
  REQUIRE(std::equal(p, p + out.size, p_));
}

template <typename T> void test_inversion(T const *const q, std::size_t N) {
  const mgard::MemoryBuffer<unsigned char> compressed =
      mgard::huffman_encode<T>(q, N);
  const mgard::MemoryBuffer<T> decompressed =
      mgard::huffman_decode<T>(compressed);
  REQUIRE(N == decompressed.size);
  REQUIRE(std::equal(q, q + N, decompressed.data.get()));
}

void test_encoding_regression_constant(const std::size_t N, const long int q) {
  long int *const quantized = new long int[N];
  std::fill(quantized, quantized + N, q);
  test_encoding_regression(quantized, N);
  delete[] quantized;
}

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

void test_decoding_regression_constant(const std::size_t N, const long int q) {
  long int *const quantized = new long int[N];
  std::fill(quantized, quantized + N, q);
  test_decoding_regression(quantized, N);
  delete[] quantized;
}

void test_decoding_regression_periodic(const std::size_t N, const long int q,
                                       const std::size_t period) {
  long int *const quantized = new long int[N];
  std::generate(quantized, quantized + N, PeriodicGenerator(period, q));
  test_decoding_regression(quantized, N);
  delete[] quantized;
}

void test_decoding_regression_random(const std::size_t N, const long int a,
                                     const long int b,
                                     std::default_random_engine &gen) {
  std::uniform_int_distribution<long int> dis(a, b);
  long int *const quantized = new long int[N];
  std::generate(quantized, quantized + N, [&] { return dis(gen); });
  test_decoding_regression(quantized, N);
  delete[] quantized;
}

template <typename T>
void test_inversion_constant(const std::size_t N, const T q) {
  T *const quantized = new T[N];
  std::fill(quantized, quantized + N, q);
  test_inversion(quantized, N);
  delete[] quantized;
}

template <typename T>
void test_inversion_periodic(const std::size_t N, const T q,
                             const std::size_t period) {
  T *const quantized = new T[N];
  std::generate(quantized, quantized + N, PeriodicGenerator(period, q));
  test_inversion(quantized, N);
  delete[] quantized;
}

template <typename T>
void test_inversion_random(const std::size_t N, const T a, const T b,
                           std::default_random_engine &gen) {
  std::uniform_int_distribution<T> dis(a, b);
  T *const quantized = new T[N];
  std::generate(quantized, quantized + N, [&] { return dis(gen); });
  test_inversion(quantized, N);
  delete[] quantized;
}

} // namespace

TEST_CASE("encoding regression", "[huffman] [regression]") {
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
    std::default_random_engine gen(726847);
    test_encoding_regression_random(10, 0, 1, gen);
    test_encoding_regression_random(100, -15, -5, gen);
    test_encoding_regression_random(1000, std::numeric_limits<int>::min(),
                                    std::numeric_limits<int>::max(), gen);
    test_encoding_regression_random(10000, -100, 100, gen);
  }
}

TEST_CASE("decoding regression", "[huffman] [regression]") {
  SECTION("constant data") {
    test_decoding_regression_constant(10, -11);
    test_decoding_regression_constant(100, 79);
    test_decoding_regression_constant(1000, -7296);
  }

  SECTION("periodic data") {
    test_decoding_regression_periodic(10, 12, 4);
    test_decoding_regression_periodic(100, -71, 9);
    test_decoding_regression_periodic(1000, 3280, 23);
  }

  SECTION("random data") {
    std::default_random_engine gen(363022);
    test_decoding_regression_random(10, 0, 1, gen);
    test_decoding_regression_random(100, -15, -5, gen);
    test_decoding_regression_random(1000, std::numeric_limits<int>::min(),
                                    std::numeric_limits<int>::max(), gen);
    test_decoding_regression_random(10000, -100, 100, gen);
  }
}

TEMPLATE_TEST_CASE("Huffman inversion", "[huffman]", std::int8_t, std::int16_t,
                   std::int32_t, std::int64_t) {
  std::default_random_engine gen_(454114);
  std::uniform_int_distribution<TestType> dis;
  SECTION("constant data") {
    test_inversion_constant<TestType>(10, dis(gen_));
    test_inversion_constant<TestType>(100, -dis(gen_));
    test_inversion_constant<TestType>(1000, dis(gen_));
  }

  SECTION("periodic data") {
    test_inversion_periodic<TestType>(10, -dis(gen_), 11);
    test_inversion_periodic<TestType>(100, dis(gen_), 10);
    test_inversion_periodic<TestType>(1000, -dis(gen_), 9);
  }

  SECTION("random data") {
    std::default_random_engine gen(950142);
    test_inversion_random<TestType>(10, 0, 1, gen);
    test_inversion_random<TestType>(100, -12, 11, gen);
    test_inversion_random<TestType>(1000, std::numeric_limits<TestType>::min(),
                                    std::numeric_limits<TestType>::max(), gen);
    test_inversion_random<TestType>(10000, -100, 100, gen);
  }
}

TEST_CASE("`HuffmanEncodedStream` serialization inversion", "[huffman]") {
  // This is not intended to be a valid `HuffmanEncodedStream`.
  const std::size_t nbits = 2718;
  const std::size_t nmissed = 896 * sizeof(int);
  const std::size_t ntable = 681 * 2 * sizeof(std::size_t);
  const mgard::HuffmanEncodedStream original(nbits, nmissed, ntable);
  {
    unsigned char *const p = original.hit.data.get();
    std::iota(p, p + original.hit.size, 1u);
  }
  {
    unsigned char *const p = original.missed.data.get();
    std::iota(p, p + nmissed, 90u);
  }
  {
    unsigned char *const p = original.frequencies.data.get();
    std::iota(p, p + ntable, 51u);
  }

  const mgard::MemoryBuffer<unsigned char> serialized =
      mgard::serialize_compress(original);
  const mgard::HuffmanEncodedStream deserialized =
      mgard::decompress_deserialize(serialized.data.get(), serialized.size);

  REQUIRE(original.nbits == deserialized.nbits);
  REQUIRE(original.hit.size == deserialized.hit.size);
  REQUIRE(original.missed.size == deserialized.missed.size);
  REQUIRE(original.frequencies.size == deserialized.frequencies.size);

  {
    unsigned char const *const p = original.hit.data.get();
    unsigned char const *const q = deserialized.hit.data.get();
    REQUIRE(std::equal(p, p + original.hit.size, q));
  }
  {
    unsigned char const *const p = original.missed.data.get();
    unsigned char const *const q = deserialized.missed.data.get();
    REQUIRE(std::equal(p, p + nmissed, q));
  }
  {
    unsigned char const *const p = original.frequencies.data.get();
    unsigned char const *const q = deserialized.frequencies.data.get();
    REQUIRE(std::equal(p, p + ntable, q));
  }
}
