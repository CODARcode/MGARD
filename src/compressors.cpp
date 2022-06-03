#include "compressors.hpp"

#include <cassert>
#include <cmath>
#include <cstring>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include <zlib.h>

#include "format.hpp"
#include "huffman.hpp"
#include "utilities.hpp"

#ifdef MGARD_ZSTD
#include <zstd.h>
#endif

namespace mgard {

namespace {

std::size_t hit_buffer_size(const std::size_t nbits) {
  return nbits / CHAR_BIT + sizeof(unsigned int);
}

} // namespace

void decompress_memory_huffman(unsigned char *const src,
                               const std::size_t srcLen, long int *const dst,
                               const std::size_t dstLen) {
  std::size_t const *const sizes = reinterpret_cast<std::size_t const *>(src);
  const std::size_t nfrequencies = sizes[0];
  const std::size_t nbits = sizes[1];
  const std::size_t nmissed = sizes[2];
  const std::size_t nhit = hit_buffer_size(nbits);

  MemoryBuffer<unsigned char> buffer(nfrequencies + nhit + nmissed);
  {
    const std::size_t offset = 3 * sizeof(std::size_t);
    unsigned char const *const src_ = src + offset;
    const std::size_t srcLen_ = srcLen - offset;
    unsigned char *const dst_ = buffer.data.get();
    const std::size_t dstLen_ = buffer.size;

#ifndef MGARD_ZSTD
    decompress_memory_z(src_, srcLen_, dst_, dstLen_);
#else
    decompress_memory_zstd(src_, srcLen_, dst_, dstLen_);
#endif
  }

  HuffmanEncodedStream encoded(nbits, nhit, nmissed, nfrequencies);
  {
    unsigned char const *begin;
    unsigned char const *end;

    begin = buffer.data.get();
    end = begin + nfrequencies;
    std::copy(begin, end, encoded.frequencies.data.get());

    begin = end;
    end = begin + nhit;
    std::copy(begin, end, encoded.hit.data.get());

    begin = end;
    end = begin + nmissed;
    std::copy(begin, end, encoded.missed.data.get());
  }

  const MemoryBuffer<long int> decoded = huffman_decoding(encoded);
  {
    long int const *const p = decoded.data.get();
    if (decoded.size * sizeof(*p) != dstLen) {
      throw std::runtime_error(
          "mismatch between expected and obtained decompressed buffer sizes");
    }
    std::copy(p, p + decoded.size, dst);
  }
}

namespace {

using Constituent = std::pair<unsigned char const *, std::size_t>;

MemoryBuffer<unsigned char>
gather_constituents(const std::vector<Constituent> &constituents) {
  std::size_t nbuffer = 0;
  for (const Constituent &constituent : constituents) {
    nbuffer += constituent.second;
  }
  MemoryBuffer<unsigned char> buffer(nbuffer);
  unsigned char *p = buffer.data.get();
  for (const Constituent &constituent : constituents) {
    std::memcpy(p, constituent.first, constituent.second);
    p += constituent.second;
  }
  return buffer;
}

} // namespace

MemoryBuffer<unsigned char> compress_memory_huffman(long int *const src,
                                                    const std::size_t srcLen) {
  HuffmanEncodedStream encoded = huffman_encoding(src, srcLen);

  assert(not(encoded.hit.size % sizeof(unsigned int)));

  static_assert(CHAR_BIT == 8, "code written assuming `CHAR_BIT == 8`");
  static_assert(sizeof(unsigned int) == 4,
                "code written assuming `sizeof(unsigned int) == 4`");
  const std::size_t offset = encoded.nbits % (CHAR_BIT * sizeof(unsigned int));
  // Number of hit buffer padding bytes.
  const std::size_t nhbpb = offset ? offset / CHAR_BIT : sizeof(unsigned int);

  assert(encoded.hit.size + nhbpb == hit_buffer_size(encoded.nbits));

  const size_t npayload =
      encoded.hit.size + nhbpb + encoded.missed.size + encoded.frequencies.size;
  unsigned char *const payload = new unsigned char[npayload];
  unsigned char *bufp = payload;

  std::memcpy(bufp, encoded.frequencies.data.get(), encoded.frequencies.size);
  bufp += encoded.frequencies.size;

  std::memcpy(bufp, encoded.hit.data.get(), encoded.hit.size);
  bufp += encoded.hit.size;

  {
    const unsigned char zero{0};
    for (std::size_t i = 0; i < nhbpb; ++i) {
      std::memcpy(bufp, &zero, 1);
      bufp += 1;
    }
  }

  std::memcpy(bufp, encoded.missed.data.get(), encoded.missed.size);
  bufp += encoded.missed.size;

#ifndef MGARD_ZSTD
  const MemoryBuffer<unsigned char> out_data =
      compress_memory_z(payload, npayload);
#else
  const MemoryBuffer<unsigned char> out_data =
      compress_memory_zstd(payload, npayload);
#endif
  delete[] payload;
  bufp = nullptr;

  const std::size_t bufferLen = 3 * sizeof(size_t) + out_data.size;
  unsigned char *const buffer = new unsigned char[bufferLen];

  bufp = buffer;
  *(size_t *)bufp = encoded.frequencies.size;
  bufp += sizeof(size_t);

  *(size_t *)bufp = encoded.nbits;
  bufp += sizeof(size_t);

  *(size_t *)bufp = encoded.missed.size;
  bufp += sizeof(size_t);

  {
    unsigned char const *const p = out_data.data.get();
    std::copy(p, p + out_data.size, bufp);
  }
  return MemoryBuffer<unsigned char>(buffer, bufferLen);
}

MemoryBuffer<unsigned char>
compress_memory_huffman_rewritten(long int *const src,
                                  const std::size_t srcLen) {
  const HuffmanEncodedStream encoded = huffman_encoding(src, srcLen);

  assert(not(encoded.hit.size % sizeof(unsigned int)));

  static_assert(CHAR_BIT == 8, "code written assuming `CHAR_BIT == 8`");
  static_assert(sizeof(unsigned int) == 4,
                "code written assuming `sizeof(unsigned int) == 4`");
  const std::size_t offset = encoded.nbits % (CHAR_BIT * sizeof(unsigned int));
  // Number of hit buffer padding bytes.
  const std::size_t nhbpb = offset ? offset / CHAR_BIT : sizeof(unsigned int);

  assert(encoded.hit.size + nhbpb == hit_buffer_size(encoded.nbits));

  unsigned char const *hbpb = new unsigned char[nhbpb]();
  MemoryBuffer<unsigned char> payload = gather_constituents({
      {encoded.frequencies.data.get(), encoded.frequencies.size},
      {encoded.hit.data.get(), encoded.hit.size},
      {hbpb, nhbpb},
      {encoded.missed.data.get(), encoded.missed.size},
  });
  delete[] hbpb;

#ifndef MGARD_ZSTD
  const MemoryBuffer<unsigned char> out_data =
      compress_memory_z(payload.data.get(), payload.size);
#else
  const MemoryBuffer<unsigned char> out_data =
      compress_memory_zstd(payload.data.get(), payload.size);
#endif

  return gather_constituents(
      {{reinterpret_cast<unsigned char const *>(&encoded.frequencies.size),
        sizeof(encoded.frequencies.size)},
       {reinterpret_cast<unsigned char const *>(&encoded.nbits),
        sizeof(encoded.nbits)},
       {reinterpret_cast<unsigned char const *>(&encoded.missed.size),
        sizeof(encoded.missed.size)},
       {out_data.data.get(), out_data.size}});
}

#ifdef MGARD_ZSTD
/*! CHECK
 * Check that the condition holds. If it doesn't print a message and die.
 */
#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "%s:%d CHECK(%s) failed: ", __FILE__, __LINE__, #cond);  \
      fprintf(stderr, "" __VA_ARGS__);                                         \
      fprintf(stderr, "\n");                                                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

/*! CHECK_ZSTD
 * Check the zstd error code and die if an error occurred after printing a
 * message.
 */
/*! CHECK_ZSTD
 * Check the zstd error code and die if an error occurred after printing a
 * message.
 */
#define CHECK_ZSTD(fn, ...)                                                    \
  do {                                                                         \
    size_t const err = (fn);                                                   \
    CHECK(!ZSTD_isError(err), "%s", ZSTD_getErrorName(err));                   \
  } while (0)

MemoryBuffer<unsigned char> compress_memory_zstd(void const *const src,
                                                 const std::size_t srcLen) {
  const size_t cBuffSize = ZSTD_compressBound(srcLen);
  unsigned char *const buffer = new unsigned char[cBuffSize];
  const std::size_t cSize = ZSTD_compress(buffer, cBuffSize, src, srcLen, 1);
  CHECK_ZSTD(cSize);
  return MemoryBuffer<unsigned char>(buffer, cSize);
}
#endif

MemoryBuffer<unsigned char> compress_memory_z(void z_const *const src,
                                              const std::size_t srcLen) {
  const std::size_t BUFSIZE = 2048 * 1024;
  std::vector<Bytef *> buffers;
  std::vector<std::size_t> bufferLengths;

  z_stream strm;
  strm.zalloc = Z_NULL;
  strm.zfree = Z_NULL;
  strm.next_in = static_cast<Bytef z_const *>(src);
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

void decompress_memory_z(void z_const *const src, const std::size_t srcLen,
                         unsigned char *const dst, const std::size_t dstLen) {
  z_stream strm = {};
  strm.total_in = strm.avail_in = srcLen;
  strm.total_out = strm.avail_out = dstLen;
  strm.next_in = static_cast<Bytef z_const *>(src);
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

#ifdef MGARD_ZSTD
void decompress_memory_zstd(void const *const src, const std::size_t srcLen,
                            unsigned char *const dst,
                            const std::size_t dstLen) {
  size_t const dSize = ZSTD_decompress(dst, dstLen, src, srcLen);
  CHECK_ZSTD(dSize);

  /* When zstd knows the content size, it will error if it doesn't match. */
  CHECK(dstLen == dSize, "Impossible because zstd will check this condition!");
}
#endif

MemoryBuffer<unsigned char> compress(const pb::Header &header, void *const src,
                                     const std::size_t srcLen) {
  switch (header.encoding().compressor()) {
  case pb::Encoding::CPU_HUFFMAN_ZSTD:
#ifdef MGARD_ZSTD
  {
    if (header.quantization().type() != mgard::pb::Quantization::INT64_T) {
      throw std::runtime_error("Huffman tree not implemented for quantization "
                               "types other than `std::int64_t`");
    }
    // Quantization type size.
    const std::size_t qts = quantization_buffer(header, 1).size;
    if (srcLen % qts) {
      throw std::runtime_error("incorrect quantization buffer size");
    }
    return compress_memory_huffman(reinterpret_cast<long int *>(src),
                                   srcLen / qts);
  }
#else
    throw std::runtime_error("MGARD compiled without ZSTD support");
#endif
  case pb::Encoding::CPU_HUFFMAN_ZLIB:
    return compress_memory_z(src, srcLen);
  default:
    throw std::runtime_error("unrecognized lossless compressor");
  }
}

void decompress(const pb::Header &header, void *const src,
                const std::size_t srcLen, void *const dst,
                const std::size_t dstLen) {
  switch (read_encoding_compressor(header)) {
  case pb::Encoding::NOOP_COMPRESSOR:
    if (srcLen != dstLen) {
      throw std::invalid_argument(
          "source and destination lengths must be equal");
    }
    {
      unsigned char const *const p = static_cast<unsigned char const *>(src);
      unsigned char *const q = static_cast<unsigned char *>(dst);
      std::copy(p, p + srcLen, q);
    }
    break;
  case pb::Encoding::CPU_HUFFMAN_ZLIB:
    decompress_memory_z(const_cast<void z_const *>(src), srcLen,
                        static_cast<unsigned char *>(dst), dstLen);
    break;
  case pb::Encoding::CPU_HUFFMAN_ZSTD:
#ifdef MGARD_ZSTD
    decompress_memory_huffman(static_cast<unsigned char *>(src), srcLen,
                              static_cast<long int *>(dst), dstLen);
    break;
#else
    throw std::runtime_error("MGARD compiled without ZSTD support");
#endif
  default:
    throw std::runtime_error("unsupported lossless encoder");
  }
}

} // namespace mgard
