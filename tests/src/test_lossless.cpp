#include "catch2/catch_template_test_macros.hpp"
#include "catch2/catch_test_macros.hpp"

#include <cstdint>

#include <algorithm>
#include <limits>
#include <random>

#include "format.hpp"
#include "lossless.hpp"
#include "lossless_regression.hpp"

#include "testing_utilities.hpp"

namespace {

// Generate a header for use with the deprecated Huffman serialization method.
mgard::pb::Header
deprecated_header(const mgard::pb::Encoding::Compressor compressor) {
  mgard::pb::Header header;
  header.mutable_quantization()->set_type(mgard::pb::Quantization::INT64_T);
  header.mutable_encoding()->set_preprocessor(mgard::pb::Encoding::SHUFFLE);
  header.mutable_encoding()->set_compressor(compressor);
  header.mutable_encoding()->set_serialization(mgard::pb::Encoding::DEPRECATED);
  return header;
}

void test_huffman_compression_regression(long int const *const src,
                                         const std::size_t srcLen) {
  std::vector<mgard::pb::Encoding::Compressor> compressors;
  compressors.push_back(mgard::pb::Encoding::CPU_HUFFMAN_ZLIB);
#ifdef MGARD_ZSTD
  compressors.push_back(mgard::pb::Encoding::CPU_HUFFMAN_ZSTD);
#endif

  for (mgard::pb::Encoding::Compressor compressor : compressors) {
    const mgard::pb::Header header = deprecated_header(compressor);
    const mgard::MemoryBuffer<unsigned char> out =
        mgard::regression::compress_memory_huffman(header, src, srcLen);
    unsigned char const *const p = out.data.get();

    const mgard::MemoryBuffer<unsigned char> out_ = mgard::compress(
        header, const_cast<long int *>(src), srcLen * sizeof(long int));
    unsigned char const *const p_ = out_.data.get();

    REQUIRE(out.size == out_.size);
    REQUIRE(std::equal(p, p + out.size, p_));
  }
}

void test_huffman_decompression_regression(long int const *const src,
                                           const std::size_t srcLen) {
  std::vector<mgard::pb::Encoding::Compressor> compressors;
  compressors.push_back(mgard::pb::Encoding::CPU_HUFFMAN_ZLIB);
#ifdef MGARD_ZSTD
  compressors.push_back(mgard::pb::Encoding::CPU_HUFFMAN_ZSTD);
#endif

  for (const mgard::pb::Encoding::Compressor compressor : compressors) {
    const mgard::pb::Header header = deprecated_header(compressor);

    const mgard::MemoryBuffer<unsigned char> compressed =
        mgard::regression::compress_memory_huffman(header, src, srcLen);
    const mgard::MemoryBuffer<unsigned char> compressed_(compressed.size);

    unsigned char *const q = compressed.data.get();
    unsigned char *const q_ = compressed_.data.get();
    std::copy(q, q + compressed.size, q_);

    mgard::MemoryBuffer<long int> out(srcLen);
    mgard::MemoryBuffer<long int> out_(srcLen);

    long int *const p = out.data.get();
    long int *const p_ = out_.data.get();

    mgard::regression::decompress_memory_huffman(header, q, compressed.size, p,
                                                 out.size * sizeof(long int));

    mgard::decompress(header, q_, compressed_.size, out_.data.get(),
                      out_.size * sizeof(long int));
    REQUIRE(std::equal(p, p + srcLen, p_));
  }
}

void test_hcr_constant(const std::size_t srcLen, const long int q) {
  long int *const src = new long int[srcLen];
  std::fill(src, src + srcLen, q);
  test_huffman_compression_regression(src, srcLen);
  delete[] src;
}

void test_hcr_periodic(const std::size_t srcLen, const long int initial,
                       const std::size_t period) {
  long int *const src = new long int[srcLen];
  std::generate(src, src + srcLen, PeriodicGenerator(period, initial));
  test_huffman_compression_regression(src, srcLen);
  delete[] src;
}

void test_hcr_random(const std::size_t srcLen, const long int a,
                     const long int b, std::default_random_engine &gen) {
  std::uniform_int_distribution<long int> dis(a, b);
  long int *const src = new long int[srcLen];
  std::generate(src, src + srcLen, [&] { return dis(gen); });
  test_huffman_compression_regression(src, srcLen);
  delete[] src;
}

void test_hdr_constant(const std::size_t srcLen, const long int q) {
  long int *const src = new long int[srcLen];
  std::fill(src, src + srcLen, q);
  test_huffman_decompression_regression(src, srcLen);
  delete[] src;
}

void test_hdr_periodic(const std::size_t srcLen, const long int initial,
                       const std::size_t period) {
  long int *const src = new long int[srcLen];
  std::generate(src, src + srcLen, PeriodicGenerator(period, initial));
  test_huffman_decompression_regression(src, srcLen);
  delete[] src;
}

void test_hdr_random(const std::size_t srcLen, const long int a,
                     const long int b, std::default_random_engine &gen) {
  std::uniform_int_distribution<long int> dis(a, b);
  long int *const src = new long int[srcLen];
  std::generate(src, src + srcLen, [&] { return dis(gen); });
  test_huffman_decompression_regression(src, srcLen);
  delete[] src;
}

} // namespace

TEST_CASE("Huffman compression regression", "[compressors] [regression]") {
  SECTION("constant data") {
    test_hcr_constant(5, -3);
    test_hcr_constant(25, 0);
    test_hcr_constant(625, 81);
  }

  SECTION("periodic data") {
    test_hcr_periodic(5, 0, 5);
    test_hcr_periodic(25, -4, 6);
    test_hcr_periodic(625, 22, 20);
  }

  SECTION("random data") {
    std::default_random_engine gen(131051);
    test_hcr_random(50, 0, 1, gen);
    test_hcr_random(25, -8, 16, gen);
    test_hcr_random(625, std::numeric_limits<int>::min(),
                    std::numeric_limits<int>::max(), gen);
    test_hcr_random(3125, -100, 100, gen);
  }
}

TEST_CASE("Huffman decompression regression", "[compressors] [regression]") {
  SECTION("constant data") {
    test_hdr_constant(4, -143485);
    test_hdr_constant(64, 0);
    test_hdr_constant(256, 67486);
  }

  SECTION("periodic data") {
    test_hdr_periodic(10, 0, 3);
    test_hdr_periodic(100, -570, 10);
    test_hdr_periodic(1000, 394, 19);
  }

  SECTION("random data") {
    std::default_random_engine gen(566222);
    test_hdr_random(100, 1, 2, gen);
    test_hdr_random(30, -7, 7, gen);
    test_hdr_random(900, std::numeric_limits<int>::min(),
                    std::numeric_limits<int>::max(), gen);
    test_hdr_random(2700, -60, 40, gen);
  }
}

#ifdef MGARD_ZSTD
namespace {

void test_zstd_identity(std::uniform_int_distribution<unsigned char> &dis,
                        std::default_random_engine &gen, const std::size_t n) {
  const auto f = [&]() -> unsigned char { return dis(gen); };
  unsigned char *const src = new unsigned char[n];
  std::generate(src, src + n, f);
  unsigned char *const src_ = new unsigned char[n];
  std::copy(src, src + n, src_);
  mgard::MemoryBuffer<unsigned char> dst = mgard::compress_zstd(src_, n);
  delete[] src_;
  unsigned char *const decompressed = new unsigned char[n];
  mgard::decompress_zstd(dst.data.get(), dst.size, decompressed, n);
  REQUIRE(std::equal(src, src + n, decompressed));
  delete[] decompressed;
  delete[] src;
}

} // namespace

TEST_CASE("zstd compression", "[compressors]") {
  std::uniform_int_distribution<unsigned char> dis;
  std::default_random_engine gen(158648);
  const std::vector<std::size_t> ns{10, 10, 1000, 10000};
  for (const std::size_t n : ns) {
    test_zstd_identity(dis, gen, n);
  }
}
#endif

namespace {

void test_zlib_identity(std::uniform_int_distribution<unsigned char> &dis,
                        std::default_random_engine &gen, const std::size_t n) {
  const auto f = [&]() -> unsigned char { return dis(gen); };
  unsigned char *const src = new unsigned char[n];
  std::generate(src, src + n, f);
  unsigned char *const src_ = new unsigned char[n];
  std::copy(src, src + n, src_);
  mgard::MemoryBuffer<unsigned char> dst = mgard::compress_zlib(src_, n);
  delete[] src_;
  unsigned char *const decompressed = new unsigned char[n];
  mgard::decompress_zlib(dst.data.get(), dst.size, decompressed, n);
  REQUIRE(std::equal(src, src + n, decompressed));
  delete[] decompressed;
  delete[] src;
}

} // namespace

TEST_CASE("zlib compression", "[compressors]") {
  std::uniform_int_distribution<unsigned char> dis;
  std::default_random_engine gen(252315);
  const std::vector<std::size_t> ns{10, 10, 1000, 10000};
  for (const std::size_t n : ns) {
    test_zlib_identity(dis, gen, n);
  }
}

namespace {

template <typename Int>
void test_cd_inversion(const mgard::pb::Header &header,
                       Int const *const quantized, const std::size_t n) {
  const std::size_t nbytes = sizeof(Int) * n;

  Int *const quantized_ = new Int[n];
  std::copy(quantized, quantized + n, quantized_);
  const mgard::MemoryBuffer<unsigned char> compressed =
      mgard::compress(header, quantized_, nbytes);
  delete[] quantized_;

  Int *const decompressed = new Int[n];
  mgard::decompress(header, compressed.data.get(), compressed.size,
                    decompressed, nbytes);
  REQUIRE(std::equal(quantized, quantized + n, decompressed));
  delete[] decompressed;
}

template <typename Int>
void test_cd_inversion_constant(const mgard::pb::Header &header,
                                const std::size_t N, const Int q) {
  Int *const quantized = new Int[N];
  std::fill(quantized, quantized + N, q);
  test_cd_inversion(header, quantized, N);
  delete[] quantized;
}

template <typename Int>
void test_cd_inversion_periodic(const mgard::pb::Header &header,
                                const std::size_t N, const Int q,
                                const std::size_t period) {
  Int *const quantized = new Int[N];
  std::generate(quantized, quantized + N, PeriodicGenerator(period, q));
  test_cd_inversion(header, quantized, N);
  delete[] quantized;
}

template <typename Int>
void test_cd_inversion_random(const mgard::pb::Header &header,
                              const std::size_t N, const Int a, const Int b,
                              std::default_random_engine &gen) {
  std::uniform_int_distribution<Int> dis(a, b);
  Int *const quantized = new Int[N];
  std::generate(quantized, quantized + N, [&] { return dis(gen); });
  test_cd_inversion(header, quantized, N);
  delete[] quantized;
}

template <typename Int>
mgard::pb::Quantization::Type type_to_quantization_type();

template <>
mgard::pb::Quantization::Type type_to_quantization_type<std::int8_t>() {
  return mgard::pb::Quantization::INT8_T;
}

template <>
mgard::pb::Quantization::Type type_to_quantization_type<std::int16_t>() {
  return mgard::pb::Quantization::INT16_T;
}

template <>
mgard::pb::Quantization::Type type_to_quantization_type<std::int32_t>() {
  return mgard::pb::Quantization::INT32_T;
}

template <>
mgard::pb::Quantization::Type type_to_quantization_type<std::int64_t>() {
  return mgard::pb::Quantization::INT64_T;
}

template <typename Int>
void test_cd_inversion_constant(const mgard::pb::Header &header) {
  test_cd_inversion_constant<Int>(header, 100, 98);
  test_cd_inversion_constant<Int>(header, 1000, 0);
  test_cd_inversion_constant<Int>(header, 10000, -62);
}

template <typename Int>
void test_cd_inversion_periodic(const mgard::pb::Header &header) {
  test_cd_inversion_periodic<Int>(header, 100, -5, 3);
  test_cd_inversion_periodic<Int>(header, 1000, 86, 60);
  test_cd_inversion_periodic<Int>(header, 10000, 7, 62);
}

template <typename Int>
void test_cd_inversion_random(const mgard::pb::Header &header) {
  std::default_random_engine gen(894584);
  test_cd_inversion_random<Int>(header, 100, 0, 3, gen);
  test_cd_inversion_random<Int>(header, 1000, std::numeric_limits<Int>::min(),
                                std::numeric_limits<Int>::max(), gen);
  test_cd_inversion_random<Int>(header, 10000, -110, 110, gen);
}

template <>
void test_cd_inversion_random<std::int64_t>(const mgard::pb::Header &header) {
  std::default_random_engine gen(952426);
  test_cd_inversion_random<std::int64_t>(header, 100, -1, 1, gen);
  // In the deprecated Huffman encoding function, the missed symbols are cast
  // from `long int` to `int`.
  test_cd_inversion_random<std::int64_t>(header, 1000,
                                         std::numeric_limits<int>::min(),
                                         std::numeric_limits<int>::max(), gen);
  test_cd_inversion_random<std::int64_t>(header, 10000, 0, 250, gen);
}

template <typename Int>
void test_cd_inversion(const mgard::pb::Header &header) {
  SECTION("constant data") { test_cd_inversion_constant<Int>(header); }
  SECTION("periodic data") { test_cd_inversion_periodic<Int>(header); }
  SECTION("random data") { test_cd_inversion_random<Int>(header); }
}

} // namespace

TEMPLATE_TEST_CASE("`compress`/`decompress` inversion", "[compressors]",
                   std::int8_t, std::int16_t, std::int32_t, std::int64_t) {
  mgard::pb::Header header;
  mgard::populate_defaults(header);
  mgard::pb::Quantization &q = *header.mutable_quantization();
  mgard::pb::Encoding &e = *header.mutable_encoding();

  const mgard::pb::Quantization::Type qtype =
      type_to_quantization_type<TestType>();
  q.set_type(qtype);

  SECTION("`CPU_ZLIB`") {
    e.set_compressor(mgard::pb::Encoding::CPU_ZLIB);
    test_cd_inversion<TestType>(header);
  }

#ifdef MGARD_ZSTD
  SECTION("`CPU_ZSTD`") {
    e.set_compressor(mgard::pb::Encoding::CPU_ZSTD);
    test_cd_inversion<TestType>(header);
  }
#endif

  // The deprecated Huffman serialization method requires the quantization type
  // to be `std::int64_t`.
  if (qtype == mgard::pb::Quantization::INT64_T) {
    SECTION("`CPU_HUFFMAN_ZLIB` with `DEPRECATED`") {
      e.set_compressor(mgard::pb::Encoding::CPU_HUFFMAN_ZLIB);
      e.set_serialization(mgard::pb::Encoding::DEPRECATED);
      test_cd_inversion<TestType>(header);
    }

#ifdef MGARD_ZSTD
    SECTION("`CPU_HUFFMAN_ZSTD` with `DEPRECATED`") {
      e.set_compressor(mgard::pb::Encoding::CPU_HUFFMAN_ZLIB);
      e.set_serialization(mgard::pb::Encoding::DEPRECATED);
      test_cd_inversion<TestType>(header);
    }
#endif
  }

  SECTION("`CPU_HUFFMAN_ZLIB` with `RFMH`") {
    e.set_compressor(mgard::pb::Encoding::CPU_HUFFMAN_ZLIB);
    e.set_serialization(mgard::pb::Encoding::RFMH);
    test_cd_inversion<TestType>(header);
  }

#ifdef MGARD_ZSTD
  SECTION("`CPU_HUFFMAN_ZSTD` with `RFMH`") {
    e.set_compressor(mgard::pb::Encoding::CPU_HUFFMAN_ZSTD);
    e.set_serialization(mgard::pb::Encoding::RFMH);
    test_cd_inversion<TestType>(header);
  }
#endif
}

// In the deprecated Huffman encoding function, the missed symbols are cast from
// `long int` to `int`.
TEST_CASE("deprecated Huffman inversion", "[compressors] [!shouldfail]") {
  std::default_random_engine gen(257100);
  const std::int64_t a =
      2 * static_cast<std::int64_t>(std::numeric_limits<int>::min());
  const std::int64_t b =
      2 * static_cast<std::int64_t>(std::numeric_limits<int>::max());

  SECTION("`CPU_HUFFMAN_ZLIB` with `DEPRECATED`") {
    // Conceivably this could pass if all the generated `std::int64_t`s are
    // representable as `int`s.
    test_cd_inversion_random<std::int64_t>(
        deprecated_header(mgard::pb::Encoding::CPU_HUFFMAN_ZLIB), 5000, a, b,
        gen);
  }

#ifdef MGARD_ZSTD
  SECTION("`CPU_HUFFMAN_ZSTD` with `DEPRECATED`") {
    // Conceivably this could pass if all the generated `std::int64_t`s are
    // representable as `int`s.
    test_cd_inversion_random<std::int64_t>(
        deprecated_header(mgard::pb::Encoding::CPU_HUFFMAN_ZSTD), 5000, a, b,
        gen);
  }
#endif
}
