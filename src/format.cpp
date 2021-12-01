#include "format.hpp"

namespace mgard {

std::uint_least64_t deserialize_header_size(
    const std::array<unsigned char, HEADER_SIZE_SIZE> &bytes) {
  static_assert(CHAR_BIT * HEADER_SIZE_SIZE >= 64,
                "deserialization array too small");
  return deserialize<std::uint_least64_t, HEADER_SIZE_SIZE>(bytes);
}

std::uint_least32_t deserialize_header_crc32(
    const std::array<unsigned char, HEADER_CRC32_SIZE> &bytes) {
  static_assert(CHAR_BIT * HEADER_SIZE_SIZE >= 32,
                "deserialization array too small");
  return deserialize<std::uint_least32_t, HEADER_CRC32_SIZE>(bytes);
}

std::array<unsigned char, HEADER_SIZE_SIZE>
serialize_header_size(std::uint_least64_t size) {
  return serialize<std::uint_least64_t, HEADER_SIZE_SIZE>(size);
}

std::array<unsigned char, HEADER_CRC32_SIZE>
serialize_header_crc32(std::uint_least64_t crc32) {
  return serialize<std::uint_least32_t, HEADER_CRC32_SIZE>(crc32);
}

} // namespace mgard
