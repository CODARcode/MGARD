#include "catch2/catch_test_macros.hpp"

#include <cstdint>

#include <algorithm>
#include <limits>
#include <random>

#include "compressors.hpp"

namespace {

template <typename T>
void test_huffman_identity(std::default_random_engine &gen,
                           const std::size_t n) {
  std::uniform_int_distribution<T> dis(std::numeric_limits<T>::min());
  const auto f = [&]() -> double { return dis(gen); };
  std::vector<long int> src(n);
  std::generate(src.begin(), src.end(), f);
  std::vector<long int> src_(src);
  std::vector<unsigned char> compressed =
      mgard::compress_memory_huffman(src_.data(), n);
  long int *const decompressed = new long int[n];
  mgard::decompress_memory_huffman(compressed.data(), compressed.size(),
                                   decompressed, n * sizeof(long int));
  std::vector<long int> d(decompressed, decompressed + n);
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
  const auto f = [&]() -> double { return dis(gen); };
  unsigned char *const src = new unsigned char[n];
  std::generate(src, src + n, f);
  unsigned char *const src_ = new unsigned char[n];
  std::copy(src, src + n, src_);
  std::vector<std::uint8_t> dst = mgard::compress_memory_zstd(src_, n);
  delete[] src_;
  unsigned char *const decompressed = new unsigned char[n];
  mgard::decompress_memory_zstd(dst.data(), dst.size(), decompressed, n);
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
  const auto f = [&]() -> double { return dis(gen); };
  unsigned char *const src = new unsigned char[n];
  std::generate(src, src + n, f);
  unsigned char *const src_ = new unsigned char[n];
  std::copy(src, src + n, src_);
  std::vector<std::uint8_t> dst = mgard::compress_memory_z(src_, n);
  delete[] src_;
  unsigned char *const decompressed = new unsigned char[n];
  mgard::decompress_memory_z(dst.data(), dst.size(), decompressed, n);
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
