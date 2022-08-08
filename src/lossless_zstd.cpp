#include "lossless.hpp"

#include <cstdio>
#include <cstdlib>

#ifndef MGARD_ZSTD
#error "This file requires ZSTD."
#endif

#include <zstd.h>

namespace mgard {

/*! CHECK
 * Check that the condition holds. If it doesn't print a message and die.
 */
#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "%s:%d CHECK(%s) failed: ", __FILE__, __LINE__,     \
                   #cond);                                                     \
      std::fprintf(stderr, "" __VA_ARGS__);                                    \
      std::fprintf(stderr, "\n");                                              \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

/*! CHECK_ZSTD
 * Check the zstd error code and die if an error occurred after printing a
 * message.
 */
#define CHECK_ZSTD(fn, ...)                                                    \
  do {                                                                         \
    std::size_t const err = (fn);                                              \
    CHECK(!ZSTD_isError(err), "%s", ZSTD_getErrorName(err));                   \
  } while (0)

MemoryBuffer<unsigned char> compress_zstd(void const *const src,
                                          const std::size_t srcLen) {
  const std::size_t cBuffSize = ZSTD_compressBound(srcLen);
  unsigned char *const buffer = new unsigned char[cBuffSize];
  const std::size_t cSize = ZSTD_compress(buffer, cBuffSize, src, srcLen, 1);
  CHECK_ZSTD(cSize);
  return MemoryBuffer<unsigned char>(buffer, cSize);
}

void decompress_zstd(void const *const src, const std::size_t srcLen,
                     unsigned char *const dst, const std::size_t dstLen) {
  std::size_t const dSize = ZSTD_decompress(dst, dstLen, src, srcLen);
  CHECK_ZSTD(dSize);

  /* When zstd knows the content size, it will error if it doesn't match. */
  CHECK(dstLen == dSize, "Impossible because zstd will check this condition!");
}

#undef CHECK_ZSTD
#undef CHECK

} // namespace mgard
