#include "compressors.hpp"

#include <cassert>
#include <cmath>
#include <cstring>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <type_traits>
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

namespace {

template <typename Int>
MemoryBuffer<unsigned char> compress_huffman_C_rfmh_(const pb::Header &header,
                                                     void const *const src,
                                                     const std::size_t srcLen) {
  check_quantization_buffer(header, src, srcLen);

  assert(header.encoding().serialization() == pb::Encoding::RFMH);

  return huffman_encode(static_cast<Int const *>(src), srcLen / sizeof(Int));
}

// `C` being either ZSTD or `zlib`.
MemoryBuffer<unsigned char> compress_huffman_C_rfmh(const pb::Header &header,
                                                    void *const src,
                                                    const std::size_t srcLen) {
  assert(header.encoding().serialization() == pb::Encoding::RFMH);

  switch (header.quantization().type()) {
  case pb::Quantization::INT8_T:
    return compress_huffman_C_rfmh_<std::int8_t>(header, src, srcLen);
  case pb::Quantization::INT16_T:
    return compress_huffman_C_rfmh_<std::int16_t>(header, src, srcLen);
  case pb::Quantization::INT32_T:
    return compress_huffman_C_rfmh_<std::int32_t>(header, src, srcLen);
  case pb::Quantization::INT64_T:
    return compress_huffman_C_rfmh_<std::int64_t>(header, src, srcLen);
  default:
    throw std::runtime_error("unrecognized quantization type");
  }
}

MemoryBuffer<unsigned char>
compress_huffman_C_deprecated(const pb::Header &header, void *const src,
                              const std::size_t srcLen) {
  check_quantization_buffer(header, src, srcLen);

  assert(header.encoding().serialization() == pb::Encoding::DEPRECATED);
  if (header.quantization().type() != mgard::pb::Quantization::INT64_T) {
    throw std::runtime_error(
        "deprecated Huffman coding not implemented for quantization "
        "types other than `std::int64_t`");
  }
  // I don't think it's strictly necessary that `std::int64_t` and `long int`
  // are the same type. We could think of `long int` as a generic byte type,
  // like `unsigned char`. Worth more attention if this assertion ever fails,
  // though. That might be a good time to remove the deprecated Huffman coding
  // functions.
  static_assert(std::is_same<std::int64_t, long int>::value,
                "deprecated Huffman coding written with assumption that "
                "`std::int64_t` is `long int`");

  return serialize_compress(
      header, huffman_encoding(reinterpret_cast<long int const *>(src),
                               srcLen / sizeof(long int)));
}

MemoryBuffer<unsigned char>
compress_huffman_zlib_deprecated(const pb::Header &header, void *const src,
                                 const std::size_t srcLen) {
  assert(header.encoding().compressor() == pb::Encoding::CPU_HUFFMAN_ZLIB);

  return compress_huffman_C_deprecated(header, src, srcLen);
}

#ifdef MGARD_ZSTD
MemoryBuffer<unsigned char>
compress_huffman_zstd_deprecated(const pb::Header &header, void *const src,
                                 const std::size_t srcLen) {
  assert(header.encoding().compressor() == pb::Encoding::CPU_HUFFMAN_ZSTD);

  return compress_huffman_C_deprecated(header, src, srcLen);
}
#endif

namespace {

// `decompress_memory_z` and `decompress_memory_zstd` need to know the size of
// the decompressed buffer before they can decompress. So, in addition to the
// compressed serialized Huffman tree (`compressed`), we need to store the size
// in bytes of the serialized Huffman tree (`nhuffman`).
MemoryBuffer<unsigned char> concatenate_nhuffman_and_compressed(
    const std::size_t nhuffman, const MemoryBuffer<unsigned char> &compressed) {
  MemoryBuffer<unsigned char> out(HEADER_SIZE_SIZE + compressed.size);
  unsigned char *p = out.data.get();

  // Size in bytes of the serialized Huffman tree.
  const std::array<unsigned char, HEADER_SIZE_SIZE> nhuffman_ =
      serialize_header_size(nhuffman);
  std::copy(nhuffman_.begin(), nhuffman_.end(), p);
  p += HEADER_SIZE_SIZE;

  unsigned char const *const q = compressed.data.get();
  std::copy(q, q + compressed.size, p);
  return out;
}

} // namespace

MemoryBuffer<unsigned char>
compress_huffman_zlib_rfmh(const pb::Header &header, void *const src,
                           const std::size_t srcLen) {
  assert(header.encoding().compressor() == pb::Encoding::CPU_HUFFMAN_ZLIB);
  assert(header.encoding().serialization() == pb::Encoding::RFMH);

  const MemoryBuffer<unsigned char> encoded =
      compress_huffman_C_rfmh(header, src, srcLen);
  const MemoryBuffer<unsigned char> compressed =
      compress_memory_z(encoded.data.get(), encoded.size);
  return concatenate_nhuffman_and_compressed(encoded.size, compressed);
}

#ifdef MGARD_ZSTD
MemoryBuffer<unsigned char>
compress_huffman_zstd_rfmh(const pb::Header &header, void *const src,
                           const std::size_t srcLen) {
  assert(header.encoding().compressor() == pb::Encoding::CPU_HUFFMAN_ZSTD);
  assert(header.encoding().serialization() == pb::Encoding::RFMH);

  const MemoryBuffer<unsigned char> encoded =
      compress_huffman_C_rfmh(header, src, srcLen);
  return concatenate_nhuffman_and_compressed(
      encoded.size, compress_memory_zstd(encoded.data.get(), encoded.size));
}
#endif

MemoryBuffer<unsigned char> compress_huffman_zlib(const pb::Header &header,
                                                  void *const src,
                                                  const std::size_t srcLen) {
  assert(header.encoding().compressor() == pb::Encoding::CPU_HUFFMAN_ZLIB);

  switch (header.encoding().serialization()) {
  case pb::Encoding::DEPRECATED:
    return compress_huffman_zlib_deprecated(header, src, srcLen);
  case pb::Encoding::RFMH:
    return compress_huffman_zlib_rfmh(header, src, srcLen);
  default:
    throw std::runtime_error("unrecognized Huffman serialization");
  }
}

#ifdef MGARD_ZSTD
MemoryBuffer<unsigned char> compress_huffman_zstd(const pb::Header &header,
                                                  void *const src,
                                                  const std::size_t srcLen) {
  assert(header.encoding().compressor() == pb::Encoding::CPU_HUFFMAN_ZSTD);

  switch (header.encoding().serialization()) {
  case pb::Encoding::DEPRECATED:
    return compress_huffman_zstd_deprecated(header, src, srcLen);
  case pb::Encoding::RFMH:
    return compress_huffman_zstd_rfmh(header, src, srcLen);
  default:
    throw std::runtime_error("unrecognized Huffman serialization");
  }
}
#endif

} // namespace

MemoryBuffer<unsigned char> compress(const pb::Header &header, void *const src,
                                     const std::size_t srcLen) {
  switch (header.encoding().compressor()) {
  case pb::Encoding::CPU_ZLIB:
    return compress_memory_z(src, srcLen);
  case pb::Encoding::CPU_ZSTD:
#ifdef MGARD_ZSTD
    return compress_memory_zstd(src, srcLen);
#else
    throw std::runtime_error("MGARD compiled without ZSTD support");
#endif
  case pb::Encoding::CPU_HUFFMAN_ZLIB:
    return compress_huffman_zlib(header, src, srcLen);
  case pb::Encoding::CPU_HUFFMAN_ZSTD:
#ifdef MGARD_ZSTD
    return compress_huffman_zstd(header, src, srcLen);
#else
    throw std::runtime_error("MGARD compiled without ZSTD support");
#endif
  default:
    throw std::runtime_error("unrecognized lossless compressor");
  }
}

void decompress_noop(void *const src, const std::size_t srcLen, void *const dst,
                     const std::size_t dstLen) {
  if (srcLen != dstLen) {
    throw std::invalid_argument("source and destination lengths must be equal");
  }
  {
    unsigned char const *const p = static_cast<unsigned char const *>(src);
    unsigned char *const q = static_cast<unsigned char *>(dst);
    std::copy(p, p + srcLen, q);
  }
}

namespace {

template <typename Int>
void decompress_huffman_C_rfmh_(const pb::Header &header,
                                const MemoryBuffer<unsigned char> &encoded,
                                void *const dst, const std::size_t dstLen) {
  check_quantization_buffer(header, dst, dstLen);

  assert(header.encoding().serialization() == pb::Encoding::RFMH);

  const MemoryBuffer<Int> decoded = huffman_decode<Int>(encoded);
  if (sizeof(Int) * decoded.size != dstLen) {
    throw std::runtime_error("size of destination buffer is incorrect");
  }
  unsigned char const *const p =
      reinterpret_cast<unsigned char const *>(decoded.data.get());
  std::copy(p, p + dstLen, static_cast<unsigned char *>(dst));
}

// `C` being either ZSTD or `zlib`.
void decompress_huffman_C_rfmh(const pb::Header &header,
                               const MemoryBuffer<unsigned char> &encoded,
                               void *const dst, const std::size_t dstLen) {
  assert(header.encoding().serialization() == pb::Encoding::RFMH);

  switch (header.quantization().type()) {
  case pb::Quantization::INT8_T:
    return decompress_huffman_C_rfmh_<std::int8_t>(header, encoded, dst,
                                                   dstLen);
  case pb::Quantization::INT16_T:
    return decompress_huffman_C_rfmh_<std::int16_t>(header, encoded, dst,
                                                    dstLen);
  case pb::Quantization::INT32_T:
    return decompress_huffman_C_rfmh_<std::int32_t>(header, encoded, dst,
                                                    dstLen);
  case pb::Quantization::INT64_T:
    return decompress_huffman_C_rfmh_<std::int64_t>(header, encoded, dst,
                                                    dstLen);
  default:
    throw std::runtime_error("unrecognized quantization type");
  }
}

void decompress_huffman_C_deprecated(const pb::Header &header, void *const src,
                                     const std::size_t srcLen, void *const dst,
                                     const std::size_t dstLen) {
  check_quantization_buffer(header, dst, dstLen);

  assert(header.encoding().serialization() == pb::Encoding::DEPRECATED);
  if (header.quantization().type() != mgard::pb::Quantization::INT64_T) {
    throw std::runtime_error(
        "deprecated Huffman coding not implemented for quantization "
        "types other than `std::int64_t`");
  }
  // I don't think it's strictly necessary that `std::int64_t` and `long int`
  // are the same type. We could think of `long int` as a generic byte type,
  // like `unsigned char`. Worth more attention if this assertion ever fails,
  // though. That might be a good time to remove the deprecated Huffman coding
  // functions.
  static_assert(std::is_same<std::int64_t, long int>::value,
                "deprecated Huffman coding written with assumption that "
                "`std::int64_t` is `long int`");

  const MemoryBuffer<long int> decoded =
      huffman_decoding(decompress_deserialize(
          header, reinterpret_cast<unsigned char const *>(src), srcLen));
  if (sizeof(long int) * decoded.size != dstLen) {
    throw std::runtime_error("size of destination buffer is incorrect");
  }
  {
    unsigned char const *const p =
        reinterpret_cast<unsigned char const *>(decoded.data.get());
    std::copy(p, p + dstLen, static_cast<unsigned char *>(dst));
  }
}

void decompress_huffman_zlib_deprecated(const pb::Header &header,
                                        void *const src,
                                        const std::size_t srcLen,
                                        void *const dst,
                                        const std::size_t dstLen) {
  assert(header.encoding().compressor() == pb::Encoding::CPU_HUFFMAN_ZLIB);

  return decompress_huffman_C_deprecated(header, src, srcLen, dst, dstLen);
}

#ifdef MGARD_ZSTD
void decompress_huffman_zstd_deprecated(const pb::Header &header,
                                        void *const src,
                                        const std::size_t srcLen,
                                        void *const dst,
                                        const std::size_t dstLen) {
  assert(header.encoding().compressor() == pb::Encoding::CPU_HUFFMAN_ZSTD);

  return decompress_huffman_C_deprecated(header, src, srcLen, dst, dstLen);
}
#endif

void decompress_huffman_zlib_rfmh(const pb::Header &header, void *const src,
                                  const std::size_t srcLen, void *const dst,
                                  const std::size_t dstLen) {
  assert(header.encoding().compressor() == pb::Encoding::CPU_HUFFMAN_ZLIB);
  assert(header.encoding().serialization() == pb::Encoding::RFMH);

  BufferWindow window(src, srcLen);
  // Read theSsze in bytes of the serialized Huffman tree.
  MemoryBuffer<unsigned char> encoded(read_header_size(window));
  decompress_memory_z(const_cast<unsigned char z_const *>(window.current),
                      window.end - window.current, encoded.data.get(),
                      encoded.size);

  return decompress_huffman_C_rfmh(header, encoded, dst, dstLen);
}

#ifdef MGARD_ZSTD
void decompress_huffman_zstd_rfmh(const pb::Header &header, void *const src,
                                  const std::size_t srcLen, void *const dst,
                                  const std::size_t dstLen) {
  assert(header.encoding().compressor() == pb::Encoding::CPU_HUFFMAN_ZSTD);
  assert(header.encoding().serialization() == pb::Encoding::RFMH);

  BufferWindow window(src, srcLen);
  // Read the size in bytes of the serialized Huffman tree.
  MemoryBuffer<unsigned char> encoded(read_header_size(window));
  decompress_memory_zstd(const_cast<unsigned char z_const *>(window.current),
                         window.end - window.current, encoded.data.get(),
                         encoded.size);

  return decompress_huffman_C_rfmh(header, encoded, dst, dstLen);
}
#endif

void decompress_huffman_zlib(const pb::Header &header, void *const src,
                             const std::size_t srcLen, void *const dst,
                             const std::size_t dstLen) {
  assert(header.encoding().compressor() == pb::Encoding::CPU_HUFFMAN_ZLIB);

  switch (header.encoding().serialization()) {
  case pb::Encoding::DEPRECATED:
    return decompress_huffman_zlib_deprecated(header, src, srcLen, dst, dstLen);
  case pb::Encoding::RFMH:
    return decompress_huffman_zlib_rfmh(header, src, srcLen, dst, dstLen);
  default:
    throw std::runtime_error("unrecognized Huffman serialization");
  }
}

#ifdef MGARD_ZSTD
void decompress_huffman_zstd(const pb::Header &header, void *const src,
                             const std::size_t srcLen, void *const dst,
                             const std::size_t dstLen) {
  assert(header.encoding().compressor() == pb::Encoding::CPU_HUFFMAN_ZSTD);

  switch (header.encoding().serialization()) {
  case pb::Encoding::DEPRECATED:
    return decompress_huffman_zstd_deprecated(header, src, srcLen, dst, dstLen);
  case pb::Encoding::RFMH:
    return decompress_huffman_zstd_rfmh(header, src, srcLen, dst, dstLen);
  default:
    throw std::runtime_error("unrecognized Huffman serialization");
  }
}
#endif

} // namespace

void decompress(const pb::Header &header, void *const src,
                const std::size_t srcLen, void *const dst,
                const std::size_t dstLen) {
  switch (header.encoding().compressor()) {
  case pb::Encoding::CPU_ZLIB:
    return decompress_memory_z(const_cast<void z_const *>(src), srcLen,
                               static_cast<unsigned char *>(dst), dstLen);
  case pb::Encoding::CPU_ZSTD:
#ifdef MGARD_ZSTD
    return decompress_memory_zstd(
        src, srcLen, reinterpret_cast<unsigned char *>(dst), dstLen);
#else
    throw std::runtime_error("MGARD compiled without ZSTD support");
#endif
  case pb::Encoding::CPU_HUFFMAN_ZLIB:
    return decompress_huffman_zlib(header, src, srcLen, dst, dstLen);
  case pb::Encoding::CPU_HUFFMAN_ZSTD:
#ifdef MGARD_ZSTD
    return decompress_huffman_zstd(header, src, srcLen, dst, dstLen);
#else
    throw std::runtime_error("MGARD compiled without ZSTD support");
#endif
  default:
    throw std::runtime_error("unsupported lossless encoder");
  }
}

} // namespace mgard
