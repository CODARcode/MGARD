#include "lossless.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

#include <zlib.h>

namespace mgard {

MemoryBuffer<unsigned char> compress_zlib(void const *const src,
                                          const std::size_t srcLen) {
  const std::size_t BUFSIZE = 2048 * 1024;
  std::vector<Bytef *> buffers;
  std::vector<std::size_t> bufferLengths;

  z_stream strm;
  strm.zalloc = Z_NULL;
  strm.zfree = Z_NULL;
  strm.next_in = static_cast<Bytef z_const *>(const_cast<void z_const *>(src));
  strm.avail_in = srcLen;
  buffers.push_back(strm.next_out = new Bytef[BUFSIZE]);
  bufferLengths.push_back(strm.avail_out = BUFSIZE);

  deflateInit(&strm, Z_BEST_COMPRESSION);

  while (strm.avail_in != 0) {
    [[maybe_unused]] const int res = deflate(&strm, Z_NO_FLUSH);
    assert(res == Z_OK);
    if (strm.avail_out == 0) {
      buffers.push_back(strm.next_out = new Bytef[BUFSIZE]);
      bufferLengths.push_back(strm.avail_out = BUFSIZE);
    }
  }

  int res = Z_OK;
  while (res == Z_OK) {
    if (strm.avail_out == 0) {
      buffers.push_back(strm.next_out = new Bytef[BUFSIZE]);
      bufferLengths.push_back(strm.avail_out = BUFSIZE);
    }
    res = deflate(&strm, Z_FINISH);
  }

  assert(res == Z_STREAM_END);
  bufferLengths.back() -= strm.avail_out;
  // Could just do `nbuffers * BUFSIZE - strm.avail_out`.
  const std::size_t bufferLen =
      std::accumulate(bufferLengths.begin(), bufferLengths.end(), 0);
  unsigned char *const buffer = new unsigned char[bufferLen];
  {
    const std::size_t nbuffers = buffers.size();
    unsigned char *p = buffer;
    for (std::size_t i = 0; i < nbuffers; ++i) {
      unsigned char const *const buffer = buffers.at(i);
      const std::size_t bufferLength = bufferLengths.at(i);
      std::copy(buffer, buffer + bufferLength, p);
      p += bufferLength;
      delete[] buffer;
    }
  }
  deflateEnd(&strm);

  return MemoryBuffer<unsigned char>(buffer, bufferLen);
}

void decompress_zlib(void const *const src, const std::size_t srcLen,
                     unsigned char *const dst, const std::size_t dstLen) {
  z_stream strm = {};
  strm.total_in = strm.avail_in = srcLen;
  strm.total_out = strm.avail_out = dstLen;
  strm.next_in = static_cast<Bytef z_const *>(const_cast<void z_const *>(src));
  strm.next_out = reinterpret_cast<Bytef *>(dst);

  strm.zalloc = Z_NULL;
  strm.zfree = Z_NULL;
  strm.opaque = Z_NULL;

  [[maybe_unused]] int res;
  res = inflateInit2(&strm, (15 + 32)); // 15 window bits, and the +32 tells
                                        // zlib to to detect if using gzip or
                                        // zlib
  assert(res == Z_OK);
  res = inflate(&strm, Z_FINISH);
  assert(res == Z_STREAM_END);
  res = inflateEnd(&strm);
  assert(res == Z_OK);
}

} // namespace mgard
