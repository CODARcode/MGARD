#include "mgard_compress.hpp"

#include <cassert>
#include <cmath>

#include <zlib.h>

namespace mgard {

void compress_memory_z(void *in_data, size_t in_data_size,
                       std::vector<uint8_t> &out_data) {
  std::vector<uint8_t> buffer;

  const size_t BUFSIZE = 2048 * 1024;
  uint8_t temp_buffer[BUFSIZE];

  z_stream strm;
  strm.zalloc = 0;
  strm.zfree = 0;
  strm.next_in = reinterpret_cast<uint8_t *>(in_data);
  strm.avail_in = in_data_size;
  strm.next_out = temp_buffer;
  strm.avail_out = BUFSIZE;

  deflateInit(&strm, Z_BEST_COMPRESSION);

  while (strm.avail_in != 0) {
    int res = deflate(&strm, Z_NO_FLUSH);
    assert(res == Z_OK);
    if (strm.avail_out == 0) {
      buffer.insert(buffer.end(), temp_buffer, temp_buffer + BUFSIZE);
      strm.next_out = temp_buffer;
      strm.avail_out = BUFSIZE;
    }
  }

  int deflate_res = Z_OK;
  while (deflate_res == Z_OK) {
    if (strm.avail_out == 0) {
      buffer.insert(buffer.end(), temp_buffer, temp_buffer + BUFSIZE);
      strm.next_out = temp_buffer;
      strm.avail_out = BUFSIZE;
    }
    deflate_res = deflate(&strm, Z_FINISH);
  }

  assert(deflate_res == Z_STREAM_END);
  buffer.insert(buffer.end(), temp_buffer,
                temp_buffer + BUFSIZE - strm.avail_out);
  deflateEnd(&strm);

  out_data.swap(buffer);
}

void decompress_memory_z(const void *src, int srcLen, int *dst, int dstLen) {
  z_stream strm = {0};
  strm.total_in = strm.avail_in = srcLen;
  strm.total_out = strm.avail_out = dstLen;
  strm.next_in = (Bytef *)src;
  strm.next_out = (Bytef *)dst;

  strm.zalloc = Z_NULL;
  strm.zfree = Z_NULL;
  strm.opaque = Z_NULL;

  int err = -1;
  int ret = -1;

  err = inflateInit2(&strm, (15 + 32)); // 15 window bits, and the +32 tells
                                        // zlib to to detect if using gzip or
                                        // zlib
  if (err == Z_OK) {
    err = inflate(&strm, Z_FINISH);
    if (err == Z_STREAM_END) {
      ret = strm.total_out;
    } else {
      inflateEnd(&strm);
      //             return err;
    }
  } else {
    inflateEnd(&strm);
    //        return err;
  }

  inflateEnd(&strm);
  //    return ret;
}

} // namespace mgard
