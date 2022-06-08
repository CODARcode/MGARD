#include "catch2/catch_test_macros.hpp"

#include <cstdint>

#include <algorithm>
#include <limits>
#include <random>

#include "compressors.hpp"
#include "compressors_regression.hpp"
#include "format.hpp"

#include "testing_utilities.hpp"

namespace {

template <typename T>
void test_huffman_identity(std::default_random_engine &gen,
                           const std::size_t n) {
  std::uniform_int_distribution<T> dis(std::numeric_limits<T>::min());
  const auto f = [&]() -> T { return dis(gen); };
  std::vector<long int> src(n);
  std::generate(src.begin(), src.end(), f);
  std::vector<long int> src_(src);
  mgard::MemoryBuffer<unsigned char> compressed =
      mgard::compress_memory_huffman(src_.data(), n);
  long int *const decompressed = new long int[n];
  mgard::decompress_memory_huffman(compressed.data.get(), compressed.size,
                                   decompressed, n * sizeof(long int));
  REQUIRE(std::equal(src.begin(), src.end(), decompressed));
  delete[] decompressed;
}

void test_huffman_compression_regression(long int *const src,
                                         const std::size_t srcLen) {
  long int *const src_ = new long int[srcLen];
  std::copy(src, src + srcLen, src_);

  const mgard::MemoryBuffer<unsigned char> out =
      mgard::regression::compress_memory_huffman(src, srcLen);
  const mgard::MemoryBuffer<unsigned char> out_ =
      mgard::compress_memory_huffman(src_, srcLen);

  delete[] src_;

  REQUIRE(out.size == out_.size);
  unsigned char const *const p = out.data.get();
  unsigned char const *const p_ = out_.data.get();
  REQUIRE(std::equal(p, p + out.size, p_));
}

void test_huffman_decompression_regression(long int *const src,
                                           const std::size_t srcLen) {
  long int *const src_ = new long int[srcLen];
  std::copy(src, src + srcLen, src_);

  const mgard::MemoryBuffer<unsigned char> compressed =
      mgard::regression::compress_memory_huffman(src, srcLen);
  const mgard::MemoryBuffer<unsigned char> compressed_ =
      mgard::regression::compress_memory_huffman(src_, srcLen);

  delete[] src_;

  mgard::MemoryBuffer<long int> out(srcLen);
  mgard::MemoryBuffer<long int> out_(srcLen);

  unsigned char *const q = compressed.data.get();
  unsigned char *const q_ = compressed_.data.get();
  long int *const p = out.data.get();
  long int *const p_ = out_.data.get();

  mgard::regression::decompress_memory_huffman(q, compressed.size, p,
                                               out.size * sizeof(long int));
  mgard::decompress_memory_huffman(q_, compressed_.size, p_,
                                   out_.size * sizeof(long int));

  REQUIRE(std::equal(p, p + srcLen, p_));
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

TEST_CASE("Huffman compression", "[compressors] [!mayfail]") {
  std::default_random_engine gen(257100);
  const std::size_t n = 5000;
  SECTION("signed characters") { test_huffman_identity<signed char>(gen, n); }
  SECTION("short integers") { test_huffman_identity<short int>(gen, n); }
  SECTION("integers") { test_huffman_identity<int>(gen, n); }
  SECTION("long integers") { test_huffman_identity<long int>(gen, n); }
}

namespace {

void test_zstd_identity(std::uniform_int_distribution<unsigned char> &dis,
                        std::default_random_engine &gen, const std::size_t n) {
  const auto f = [&]() -> unsigned char { return dis(gen); };
  unsigned char *const src = new unsigned char[n];
  std::generate(src, src + n, f);
  unsigned char *const src_ = new unsigned char[n];
  std::copy(src, src + n, src_);
  mgard::MemoryBuffer<unsigned char> dst = mgard::compress_memory_zstd(src_, n);
  delete[] src_;
  unsigned char *const decompressed = new unsigned char[n];
  mgard::decompress_memory_zstd(dst.data.get(), dst.size, decompressed, n);
  REQUIRE(std::equal(src, src + n, decompressed));
  delete[] decompressed;
  delete[] src;
}

} // namespace

#ifdef MGARD_ZSTD
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
  mgard::MemoryBuffer<unsigned char> dst = mgard::compress_memory_z(src_, n);
  delete[] src_;
  unsigned char *const decompressed = new unsigned char[n];
  mgard::decompress_memory_z(dst.data.get(), dst.size, decompressed, n);
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

TEST_CASE("compression with header configuration", "[compressors]") {
  mgard::pb::Header header;
  // TODO: Once Huffman trees can be built for types other than `long int`, use
  // something other than `std::int64_t` here.
  mgard::populate_defaults(header);

  const std::size_t ndof = 10000;
  std::int64_t *const quantized = new std::int64_t[ndof];
  std::uniform_int_distribution<std::int64_t> dis(-250, 250);
  std::default_random_engine gen(419643);
  const auto f = [&]() -> std::int64_t { return dis(gen); };
  std::generate(quantized, quantized + ndof, f);
  const std::size_t quantizedLen = ndof * sizeof(*quantized);
  // `dst` must have the correct alignment for the quantization type.
  std::int64_t *const dst = new std::int64_t[ndof];

  std::int64_t *const quantized_ = new std::int64_t[ndof];
  std::copy(quantized, quantized + ndof, quantized_);
  const mgard::MemoryBuffer<unsigned char> compressed =
      mgard::compress(header, quantized_, quantizedLen);
  delete[] quantized_;

  const mgard::pb::Encoding &e = header.encoding();
  REQUIRE(e.preprocessor() == mgard::pb::Encoding::SHUFFLE);
#ifdef MGARD_ZSTD
  REQUIRE(e.compressor() == mgard::pb::Encoding::CPU_HUFFMAN_ZSTD);
  mgard::regression::decompress_memory_huffman(
      compressed.data.get(), compressed.size, dst, quantizedLen);
#else
  REQUIRE(e.compressor() == mgard::pb::Encoding::CPU_HUFFMAN_ZLIB);
  mgard::decompress_memory_z(compressed.data.get(), compressed.size, dst,
                             quantizedLen);
#endif
  REQUIRE(std::equal(quantized, quantized + ndof, dst));
  delete[] dst;
  delete[] quantized;
}

TEST_CASE("decompression with header configuration", "[compressors]") {
  mgard::pb::Header header;
  // TODO: Once Huffman trees can be built for types other than `long int`, use
  // something other than `std::int64_t` here.
  mgard::populate_defaults(header);

  const std::size_t ndof = 5000;
  std::int64_t *const quantized = new std::int64_t[ndof];
  std::uniform_int_distribution<std::int64_t> dis(-500, 500);
  std::default_random_engine gen(489063);
  const auto f = [&]() -> std::int64_t { return dis(gen); };
  std::generate(quantized, quantized + ndof, f);
  const std::size_t quantizedLen = ndof * sizeof(*quantized);
  // `dst` must have the correct alignment for the quantization type.
  std::int64_t *const dst = new std::int64_t[ndof];

  mgard::pb::Encoding &e = *header.mutable_encoding();
  SECTION("noop") {
    e.set_compressor(mgard::pb::Encoding::NOOP_COMPRESSOR);

    const std::size_t srcLen = quantizedLen;
    unsigned char *const src = new unsigned char[srcLen];
    {
      unsigned char const *const p =
          reinterpret_cast<unsigned char const *>(quantized);
      std::copy(p, p + quantizedLen, src);
    }

    mgard::decompress(header, src, srcLen,
                      reinterpret_cast<unsigned char *>(dst), quantizedLen);
    delete[] src;
    REQUIRE(std::equal(quantized, quantized + ndof, dst));
  }

  SECTION("zlib") {
    e.set_compressor(mgard::pb::Encoding::CPU_HUFFMAN_ZLIB);

    const mgard::MemoryBuffer<unsigned char> out =
        mgard::compress_memory_z(quantized, quantizedLen);

    const std::size_t srcLen = out.size * sizeof(*out.data.get());
    unsigned char *const src = new unsigned char[srcLen];
    {
      unsigned char const *const p = out.data.get();
      std::copy(p, p + srcLen, src);
    }
    mgard::decompress(header, src, srcLen,
                      reinterpret_cast<unsigned char *>(dst), quantizedLen);
    delete[] src;
    REQUIRE(std::equal(quantized, quantized + ndof, dst));
  }

#ifdef MGARD_ZSTD
  SECTION("zstd") {
    e.set_compressor(mgard::pb::Encoding::CPU_HUFFMAN_ZSTD);

    std::int64_t *const quantized_ = new std::int64_t[ndof];
    std::copy(quantized, quantized + ndof, quantized_);
    const mgard::MemoryBuffer<unsigned char> out =
        mgard::regression::compress_memory_huffman(quantized_, ndof);
    delete[] quantized_;

    const std::size_t srcLen = out.size;
    unsigned char *const src = new unsigned char[srcLen];
    {
      unsigned char const *const p = out.data.get();
      std::copy(p, p + srcLen, src);
    }
    mgard::decompress(header, src, srcLen,
                      reinterpret_cast<unsigned char *>(dst), quantizedLen);
    delete[] src;
    REQUIRE(std::equal(quantized, quantized + ndof, dst));
  }
#endif

  delete[] dst;
  delete[] quantized;
}

TEST_CASE("compression and decompression with header", "[compressors]") {
  mgard::pb::Header header;
  // TODO: Once Huffman trees can be built for types other than `long int`, use
  // something other than `std::int64_t` here.
  mgard::populate_defaults(header);

  const std::size_t ndof = 2500;
  std::int64_t *const quantized = new std::int64_t[ndof];
  std::uniform_int_distribution<std::int64_t> dis(-1000, 1000);
  std::default_random_engine gen(995719);
  const auto f = [&]() -> std::int64_t { return dis(gen); };
  std::generate(quantized, quantized + ndof, f);
  const std::size_t quantizedLen = ndof * sizeof(*quantized);
  // `dst` must have the correct alignment for the quantization type.
  std::int64_t *const dst = new std::int64_t[ndof];

  std::int64_t *const quantized_ = new std::int64_t[ndof];
  std::copy(quantized, quantized + ndof, quantized_);
  const mgard::MemoryBuffer<unsigned char> compressed =
      mgard::compress(header, quantized_, quantizedLen);
  delete[] quantized_;

  mgard::decompress(header, compressed.data.get(), compressed.size, dst,
                    quantizedLen);

  REQUIRE(std::equal(quantized, quantized + ndof, dst));
  delete[] dst;
  delete[] quantized;
}
