#include "catch2/catch_test_macros.hpp"

#include <cstdint>

#include <algorithm>
#include <limits>
#include <random>

#include "compressors.hpp"
#include "format.hpp"

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

} // namespace

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
      mgard::compress(quantized_, quantizedLen, header);
  delete[] quantized_;

  const mgard::pb::Encoding &e = header.encoding();
  REQUIRE(e.preprocessor() == mgard::pb::Encoding::SHUFFLE);
#ifdef MGARD_ZSTD
  REQUIRE(e.compressor() == mgard::pb::Encoding::CPU_HUFFMAN_ZSTD);
  mgard::decompress_memory_huffman(compressed.data.get(), compressed.size, dst,
                                   quantizedLen);
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
    e.set_compressor(mgard::pb::Encoding::NOOP);

    const std::size_t srcLen = quantizedLen;
    unsigned char *const src = new unsigned char[srcLen];
    {
      unsigned char const *const p =
          reinterpret_cast<unsigned char const *>(quantized);
      std::copy(p, p + quantizedLen, src);
    }

    mgard::decompress(src, srcLen, reinterpret_cast<unsigned char *>(dst),
                      quantizedLen, header);
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
    mgard::decompress(src, srcLen, reinterpret_cast<unsigned char *>(dst),
                      quantizedLen, header);
    delete[] src;
    REQUIRE(std::equal(quantized, quantized + ndof, dst));
  }

#ifdef MGARD_ZSTD
  SECTION("zstd") {
    e.set_compressor(mgard::pb::Encoding::CPU_HUFFMAN_ZSTD);

    std::int64_t *const quantized_ = new std::int64_t[ndof];
    std::copy(quantized, quantized + ndof, quantized_);
    const mgard::MemoryBuffer<unsigned char> out =
        mgard::compress_memory_huffman(quantized_, ndof);
    delete[] quantized_;

    const std::size_t srcLen = out.size;
    unsigned char *const src = new unsigned char[srcLen];
    {
      unsigned char const *const p = out.data.get();
      std::copy(p, p + srcLen, src);
    }
    mgard::decompress(src, srcLen, reinterpret_cast<unsigned char *>(dst),
                      quantizedLen, header);
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
      mgard::compress(quantized_, quantizedLen, header);
  delete[] quantized_;

  mgard::decompress(compressed.data.get(), compressed.size, dst, quantizedLen,
                    header);

  REQUIRE(std::equal(quantized, quantized + ndof, dst));
  delete[] dst;
  delete[] quantized;
}
