#include "format.hpp"

#include <stdexcept>

#include "MGARDConfig.hpp"

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

#ifdef MGARD_PROTOBUF
template <> pb::Dataset::Type type_to_dataset_type<float>() {
  return pb::Dataset::FLOAT;
}

template <> pb::Dataset::Type type_to_dataset_type<double>() {
  return pb::Dataset::DOUBLE;
}

namespace {

void set_version_number(pb::VersionNumber *const version_number,
                        const google::protobuf::uint64 major_,
                        const google::protobuf::uint64 minor_,
                        const google::protobuf::uint64 patch_) {
  version_number->set_major_(major_);
  version_number->set_minor_(minor_);
  version_number->set_patch_(patch_);
}

} // namespace

void populate_version_numbers(pb::Header &header) {
  set_version_number(header.mutable_mgard_version(), MGARD_VERSION_MAJOR,
                     MGARD_VERSION_MINOR, MGARD_VERSION_PATCH);
  set_version_number(header.mutable_file_format_version(),
                     MGARD_FILE_VERSION_MAJOR, MGARD_FILE_VERSION_MINOR,
                     MGARD_FILE_VERSION_PATCH);
}

BufferWindow::BufferWindow(void const *const data, const std::size_t size)
    : current(static_cast<unsigned char const *>(data)), end(current + size) {}

unsigned char const *BufferWindow::next(const std::size_t size) const {
  unsigned char const *const q = current + size;
  if (q > end) {
    throw std::runtime_error("next read will go past buffer endpoint");
  }
  return q;
}
#endif

} // namespace mgard
