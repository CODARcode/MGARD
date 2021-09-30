#include <cstddef>

#include <algorithm>
#include <array>
#include <type_traits>

namespace mgard {

namespace {

//! Read bytes from metadata buffer without reordering.
template <typename T>
std::array<unsigned char, sizeof(T)> read_bytes(void const *const p) {
  std::array<unsigned char, sizeof(T)> bytes;
  unsigned char const *q = static_cast<unsigned char const *>(p);
  std::copy(q, q + sizeof(T), bytes.begin());
  return bytes;
}

// With C++20 we will be able to use `std::endian`.
template <typename T> constexpr bool big_endian() {
  if constexpr (std::is_enum<T>::value) {
    return big_endian<typename std::underlying_type<T>::type>();
  } else {
    static_assert(std::is_integral<T>::value && std::is_unsigned<T>::value,
                  "Endianness check requires unsigned integral type.");
    const T one = 0x01;
    return *reinterpret_cast<unsigned char const *>(&one);
  }
}

// TODO: This needs to be made more robust.
template <> constexpr bool big_endian<double>() {
  static_assert(sizeof(double) == 8, "Size of double assumed to be eight.");
  static_assert(std::numeric_limits<double>::is_iec559,
                "Double assumed to be binary64.");
  return true;
}

template <> constexpr bool big_endian<VersionNumber>() { return true; }

//! Reorder metadata bytes if necessary and return parsed value.
template <typename T>
T reorder_bytes(const std::array<unsigned char, sizeof(T)> &bytes) {
  static_assert(std::is_scalar<T>::value,
                "Can only read scalar types from metadata buffer.");
  T metadatum;
  unsigned char *q = reinterpret_cast<unsigned char *>(&metadatum);
  // Multibyte metadata values are stored in big-endian format.
  if constexpr (big_endian<T>()) {
    std::copy(bytes.cbegin(), bytes.cend(), q);
  } else {
    std::copy(bytes.crbegin(), bytes.crend(), q);
  };
  return metadatum;
}

// TODO: This can probably be rewritten using `std::is_standard_layout`.
template <>
VersionNumber
reorder_bytes(const std::array<unsigned char, sizeof(VersionNumber)> &bytes) {
  static_assert(sizeof(unsigned char) == sizeof(uint8_t),
                "Unexpected `uint8_t` size.");
  return {.major = static_cast<uint8_t>(bytes.at(0)),
          .minor = static_cast<uint8_t>(bytes.at(1)),
          .patch = static_cast<uint8_t>(bytes.at(2))};
}

//! Write byte representation of metadatum, reordering bytes if necessary.
template <typename T>
std::array<unsigned char, sizeof(T)> write_bytes(const T &t) {
  std::array<unsigned char, sizeof(T)> bytes;
  unsigned char const *const q = reinterpret_cast<unsigned char const *>(&t);
  unsigned char const *const r = q + sizeof(T);
  if (big_endian<T>()) {
    std::copy(q, r, bytes.begin());
  } else {
    std::copy(q, r, bytes.rbegin());
  }
  return bytes;
}

// TODO: This can probably be rewritten using `std::is_standard_layout`.
template <>
std::array<unsigned char, sizeof(VersionNumber)>
write_bytes(const VersionNumber &version) {
  static_assert(sizeof(VersionNumber) == 3, "Unexpected `VersionNumber` size.");
  std::array<unsigned char, sizeof(VersionNumber)> bytes;
  bytes.at(0) = *reinterpret_cast<unsigned char const *>(&version.major);
  bytes.at(1) = *reinterpret_cast<unsigned char const *>(&version.minor);
  bytes.at(2) = *reinterpret_cast<unsigned char const *>(&version.patch);
  return bytes;
}

} // namespace

template <typename T> T MetadataReader::read() {
  const T metadatum = reorder_bytes<T>(read_bytes<T>(p));
  p += sizeof(T);
  return metadatum;
}

template <typename T> void MetadataWriter::write(const T &t) {
  const std::array<unsigned char, sizeof(T)> bytes = write_bytes(t);
  buffer.insert(buffer.end(), bytes.begin(), bytes.end());
}

} // namespace mgard
