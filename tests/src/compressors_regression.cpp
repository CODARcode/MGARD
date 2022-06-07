#include "compressors_regression.hpp"

#include <climits>
#include <cstring>

#include "compressors.hpp"
#include "huffman.hpp"

namespace mgard {

static_assert(CHAR_BIT == 8, "code written assuming `CHAR_BIT == 8`");

static_assert(sizeof(unsigned int) == 4,
              "code written assuming `sizeof(unsigned int) == 4`");

static_assert(sizeof(std::size_t) == 8,
              "code written assuming `sizeof(std::size_t) == 8`");

// This code also makes endianness assumptions.

MemoryBuffer<unsigned char> compress_memory_huffman(long int *const src,
                                                    const std::size_t srcLen) {
  HuffmanEncodedStream encoded = huffman_encoding(src, srcLen);

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

#ifndef MGARD_ZSTD
  const MemoryBuffer<unsigned char> out_data =
      compress_memory_z(payload, npayload);
#else
  const MemoryBuffer<unsigned char> out_data =
      compress_memory_zstd(payload, npayload);
#endif
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

} // namespace mgard
