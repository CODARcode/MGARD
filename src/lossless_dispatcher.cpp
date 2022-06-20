#include "lossless.hpp"

#include <cassert>
#include <cmath>
#include <cstring>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "format.hpp"
#include "huffman.hpp"
#include "utilities.hpp"

namespace mgard {

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
                                                    void const *const src,
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
compress_huffman_C_deprecated(const pb::Header &header, void const *const src,
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

MemoryBuffer<unsigned char> compress_huffman_zlib_deprecated(
    const pb::Header &header, void const *const src, const std::size_t srcLen) {
  assert(header.encoding().compressor() == pb::Encoding::CPU_HUFFMAN_ZLIB);

  return compress_huffman_C_deprecated(header, src, srcLen);
}

#ifdef MGARD_ZSTD
MemoryBuffer<unsigned char> compress_huffman_zstd_deprecated(
    const pb::Header &header, void const *const src, const std::size_t srcLen) {
  assert(header.encoding().compressor() == pb::Encoding::CPU_HUFFMAN_ZSTD);

  return compress_huffman_C_deprecated(header, src, srcLen);
}
#endif

namespace {

// `decompress_zlib` and `decompress_zstd` need to know the size of the
// decompressed buffer before they can decompress. So, in addition to the
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
compress_huffman_zlib_rfmh(const pb::Header &header, void const *const src,
                           const std::size_t srcLen) {
  assert(header.encoding().compressor() == pb::Encoding::CPU_HUFFMAN_ZLIB);
  assert(header.encoding().serialization() == pb::Encoding::RFMH);

  const MemoryBuffer<unsigned char> encoded =
      compress_huffman_C_rfmh(header, src, srcLen);
  const MemoryBuffer<unsigned char> compressed =
      compress_zlib(encoded.data.get(), encoded.size);
  return concatenate_nhuffman_and_compressed(encoded.size, compressed);
}

#ifdef MGARD_ZSTD
MemoryBuffer<unsigned char>
compress_huffman_zstd_rfmh(const pb::Header &header, void const *const src,
                           const std::size_t srcLen) {
  assert(header.encoding().compressor() == pb::Encoding::CPU_HUFFMAN_ZSTD);
  assert(header.encoding().serialization() == pb::Encoding::RFMH);

  const MemoryBuffer<unsigned char> encoded =
      compress_huffman_C_rfmh(header, src, srcLen);
  return concatenate_nhuffman_and_compressed(
      encoded.size, compress_zstd(encoded.data.get(), encoded.size));
}
#endif

MemoryBuffer<unsigned char> compress_huffman_zlib(const pb::Header &header,
                                                  void const *const src,
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
                                                  void const *const src,
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

MemoryBuffer<unsigned char> compress(const pb::Header &header,
                                     void const *const src,
                                     const std::size_t srcLen) {
  switch (header.encoding().compressor()) {
  case pb::Encoding::CPU_ZLIB:
    return compress_zlib(src, srcLen);
  case pb::Encoding::CPU_ZSTD:
#ifdef MGARD_ZSTD
    return compress_zstd(src, srcLen);
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

void decompress_noop(void const *const src, const std::size_t srcLen,
                     void *const dst, const std::size_t dstLen) {
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

void decompress_huffman_C_deprecated(const pb::Header &header,
                                     void const *const src,
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
                                        void const *const src,
                                        const std::size_t srcLen,
                                        void *const dst,
                                        const std::size_t dstLen) {
  assert(header.encoding().compressor() == pb::Encoding::CPU_HUFFMAN_ZLIB);

  return decompress_huffman_C_deprecated(header, src, srcLen, dst, dstLen);
}

#ifdef MGARD_ZSTD
void decompress_huffman_zstd_deprecated(const pb::Header &header,
                                        void const *const src,
                                        const std::size_t srcLen,
                                        void *const dst,
                                        const std::size_t dstLen) {
  assert(header.encoding().compressor() == pb::Encoding::CPU_HUFFMAN_ZSTD);

  return decompress_huffman_C_deprecated(header, src, srcLen, dst, dstLen);
}
#endif

void decompress_huffman_zlib_rfmh(const pb::Header &header,
                                  void const *const src,
                                  const std::size_t srcLen, void *const dst,
                                  const std::size_t dstLen) {
  assert(header.encoding().compressor() == pb::Encoding::CPU_HUFFMAN_ZLIB);
  assert(header.encoding().serialization() == pb::Encoding::RFMH);

  BufferWindow window(src, srcLen);
  // Read theSsze in bytes of the serialized Huffman tree.
  MemoryBuffer<unsigned char> encoded(read_header_size(window));
  decompress_zlib(window.current, window.end - window.current,
                  encoded.data.get(), encoded.size);

  return decompress_huffman_C_rfmh(header, encoded, dst, dstLen);
}

#ifdef MGARD_ZSTD
void decompress_huffman_zstd_rfmh(const pb::Header &header,
                                  void const *const src,
                                  const std::size_t srcLen, void *const dst,
                                  const std::size_t dstLen) {
  assert(header.encoding().compressor() == pb::Encoding::CPU_HUFFMAN_ZSTD);
  assert(header.encoding().serialization() == pb::Encoding::RFMH);

  BufferWindow window(src, srcLen);
  // Read the size in bytes of the serialized Huffman tree.
  MemoryBuffer<unsigned char> encoded(read_header_size(window));
  decompress_zstd(window.current, window.end - window.current,
                  encoded.data.get(), encoded.size);

  return decompress_huffman_C_rfmh(header, encoded, dst, dstLen);
}
#endif

void decompress_huffman_zlib(const pb::Header &header, void const *const src,
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
void decompress_huffman_zstd(const pb::Header &header, void const *const src,
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

void decompress(const pb::Header &header, void const *const src,
                const std::size_t srcLen, void *const dst,
                const std::size_t dstLen) {
  switch (header.encoding().compressor()) {
  case pb::Encoding::CPU_ZLIB:
    return decompress_zlib(src, srcLen, static_cast<unsigned char *>(dst),
                           dstLen);
  case pb::Encoding::CPU_ZSTD:
#ifdef MGARD_ZSTD
    return decompress_zstd(src, srcLen, reinterpret_cast<unsigned char *>(dst),
                           dstLen);
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
