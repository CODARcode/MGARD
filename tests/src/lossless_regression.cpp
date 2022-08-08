#include "lossless_regression.hpp"

#include <climits>
#include <cstring>

#include "huffman.hpp"
#include "huffman_regression.hpp"
#include "lossless.hpp"

namespace mgard {

namespace regression {

static_assert(CHAR_BIT == 8, "code written assuming `CHAR_BIT == 8`");

static_assert(sizeof(unsigned int) == 4,
              "code written assuming `sizeof(unsigned int) == 4`");

static_assert(sizeof(std::size_t) == 8,
              "code written assuming `sizeof(std::size_t) == 8`");

namespace {

std::size_t hit_buffer_size(const std::size_t nbits) {
  return nbits / CHAR_BIT + sizeof(unsigned int);
}

MemoryBuffer<unsigned char> compress_serialized(const pb::Header &header,
                                                unsigned char const *const p,
                                                const std::size_t n) {
  assert(header.encoding().serialization() == pb::Encoding::DEPRECATED);

  switch (header.encoding().compressor()) {
  case pb::Encoding::CPU_HUFFMAN_ZLIB:
    return compress_zlib(p, n);
  case pb::Encoding::CPU_HUFFMAN_ZSTD:
#ifdef MGARD_ZSTD
    return compress_zstd(p, n);
#else
    throw std::runtime_error("MGARD compiled without ZSTD support");
#endif
  default:
    throw std::runtime_error("unrecognized lossless compressor");
  }
}

} // namespace

// This code also makes endianness assumptions.

MemoryBuffer<unsigned char> compress_memory_huffman(const pb::Header &header,
                                                    long int const *const src,
                                                    const std::size_t srcLen) {
  HuffmanEncodedStream encoded =
      mgard::regression::huffman_encoding(src, srcLen);

  assert(not(encoded.hit.size % sizeof(unsigned int)));

  const std::size_t offset = encoded.nbits % (CHAR_BIT * sizeof(unsigned int));
  // Number of hit buffer padding bytes.
  const std::size_t nhbpb = offset ? offset / CHAR_BIT : sizeof(unsigned int);

  assert(encoded.hit.size + nhbpb == hit_buffer_size(encoded.nbits));

  const std::size_t npayload =
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

  const MemoryBuffer<unsigned char> out_data =
      compress_serialized(header, payload, npayload);

  delete[] payload;
  bufp = nullptr;

  const std::size_t bufferLen = 3 * sizeof(std::size_t) + out_data.size;
  unsigned char *const buffer = new unsigned char[bufferLen];

  bufp = buffer;
  *(std::size_t *)bufp = encoded.frequencies.size;
  bufp += sizeof(std::size_t);

  *(std::size_t *)bufp = encoded.nbits;
  bufp += sizeof(std::size_t);

  *(std::size_t *)bufp = encoded.missed.size;
  bufp += sizeof(std::size_t);

  {
    unsigned char const *const p = out_data.data.get();
    std::copy(p, p + out_data.size, bufp);
  }
  return MemoryBuffer<unsigned char>(buffer, bufferLen);
}

void decompress_memory_huffman(const pb::Header &header,
                               unsigned char const *const src,
                               const std::size_t srcLen, long int *const dst,
                               const std::size_t dstLen) {
  assert(header.encoding().serialization() == pb::Encoding::DEPRECATED);

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

    switch (header.encoding().compressor()) {
    case pb::Encoding::CPU_HUFFMAN_ZLIB:
      decompress_zlib(src_, srcLen_, dst_, dstLen_);
      break;
    case pb::Encoding::CPU_HUFFMAN_ZSTD:
#ifdef MGARD_ZSTD
      decompress_zstd(src_, srcLen_, dst_, dstLen_);
      break;
#else
      throw std::runtime_error("MGARD compiled without ZSTD support");
#endif
    default:
      throw std::runtime_error("unrecognized lossless compressor");
    }
  }

  HuffmanEncodedStream encoded(nbits, nmissed, nfrequencies);
  {
    unsigned char const *begin;
    unsigned char const *end;

    begin = buffer.data.get();
    end = begin + nfrequencies;
    std::copy(begin, end, encoded.frequencies.data.get());

    begin = end;
    assert(encoded.hit.size <= nhit);
    end = begin + encoded.hit.size;
    std::copy(begin, end, encoded.hit.data.get());

    // Skip any bytes between `begin + encoded.hit.size` and `begin + nhit`.
    begin = end + nhit - encoded.hit.size;
    end = begin + nmissed;
    std::copy(begin, end, encoded.missed.data.get());
  }

  const MemoryBuffer<long int> decoded =
      mgard::regression::huffman_decoding(encoded);
  {
    long int const *const p = decoded.data.get();
    if (decoded.size * sizeof(*p) != dstLen) {
      throw std::runtime_error(
          "mismatch between expected and obtained decompressed buffer sizes");
    }
    std::copy(p, p + decoded.size, dst);
  }
}

} // namespace regression

} // namespace mgard
