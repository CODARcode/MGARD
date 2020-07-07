#include "mgard_compress.hpp"

#include <cassert>
#include <cmath>

#include <zlib.h>

namespace mgard {

void compress_memory_z(void *const in_data, const std::size_t in_data_size,
                       std::vector<std::uint8_t> &out_data) {
  std::vector<std::uint8_t> buffer;

  const std::size_t BUFSIZE = 2048 * 1024;
  std::uint8_t temp_buffer[BUFSIZE];

  z_stream strm;
  strm.zalloc = Z_NULL;
  strm.zfree = Z_NULL;
  strm.next_in = static_cast<std::uint8_t *>(in_data);
  strm.avail_in = in_data_size;
  strm.next_out = temp_buffer;
  strm.avail_out = BUFSIZE;

  deflateInit(&strm, Z_BEST_COMPRESSION);

  while (strm.avail_in != 0) {
    const int res = deflate(&strm, Z_NO_FLUSH);
    assert(res == Z_OK);
    if (strm.avail_out == 0) {
      buffer.insert(buffer.end(), temp_buffer, temp_buffer + BUFSIZE);
      strm.next_out = temp_buffer;
      strm.avail_out = BUFSIZE;
    }
  }

  int res = Z_OK;
  while (res == Z_OK) {
    if (strm.avail_out == 0) {
      buffer.insert(buffer.end(), temp_buffer, temp_buffer + BUFSIZE);
      strm.next_out = temp_buffer;
      strm.avail_out = BUFSIZE;
    }
    res = deflate(&strm, Z_FINISH);
  }

  assert(res == Z_STREAM_END);
  buffer.insert(buffer.end(), temp_buffer,
                temp_buffer + BUFSIZE - strm.avail_out);
  deflateEnd(&strm);

  out_data.swap(buffer);
}

void decompress_memory_z(void *const src, const int srcLen, int *const dst,
                         const int dstLen) {
  z_stream strm = {0};
  strm.total_in = strm.avail_in = srcLen;
  strm.total_out = strm.avail_out = dstLen;
  strm.next_in = static_cast<Bytef *>(src);
  strm.next_out = reinterpret_cast<Bytef *>(dst);

  strm.zalloc = Z_NULL;
  strm.zfree = Z_NULL;
  strm.opaque = Z_NULL;

  int res;
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
