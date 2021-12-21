#ifndef FORMAT_HPP
#define FORMAT_HPP
//!\file
//!\brief Self-describing file format for compressed datasets.

#include <cstdint>

#include <array>

#ifdef MGARD_PROTOBUF
#include "proto/mgard.pb.h"
#endif

namespace mgard {

//! Magic bytes for MGARD file format ('MGARD' in ASCII).
inline constexpr std::array<char, 5> SIGNATURE{0x4d, 0x47, 0x41, 0x52, 0x44};

// Size in bytes of the serialized header size.
inline constexpr std::size_t HEADER_SIZE_SIZE = 8;

// Size in bytes of the serialized header CRC32.
inline constexpr std::size_t HEADER_CRC32_SIZE = 4;

//! Deserialize header size.
//!
//!\param bytes Serialized header size.
//!\return Size in bytes of the header.
std::uint_least64_t deserialize_header_size(
    const std::array<unsigned char, HEADER_SIZE_SIZE> &bytes);

//! Deserialize header CRC32.
//!
//!\param bytes Serialized header CRC32.
//!\return CRC32 of the header.
std::uint_least32_t deserialize_header_crc32(
    const std::array<unsigned char, HEADER_CRC32_SIZE> &bytes);

//! Serialize header size.
//!
//!\param size Size in bytes of the header.
//!\return Serialized header size.
std::array<unsigned char, HEADER_SIZE_SIZE>
serialize_header_size(std::uint_least64_t size);

//! Serialize header CRC32.
//!
//!\param CRC32 of the header.
//!\return bytes Serialized header CRC32.
std::array<unsigned char, HEADER_CRC32_SIZE>
serialize_header_crc32(std::uint_least64_t crc32);

#ifdef MGARD_PROTOBUF
//! Return the `Dataset::Type` value corresponding to a floating point type.
template <typename Real> pb::Dataset::Type type_to_dataset_type();

//! Populate a header with the MGARD and file format version numbers.
//!
//!\param Header to be populated.
void populate_version_numbers(pb::Header &header);

//! Window of a buffer being parsed.
//!
//! The left endpoint of the window is advanced as we parse the buffer. The
//! right endpoint of the window is fixed.
struct BufferWindow {
  //! Constructor.
  //!
  //!\param begin Beginning of the buffer.
  //!\param size Size of the buffer.
  BufferWindow(void const *const data, const std::size_t size);

  //! Return the right endpoint of the next read, checking bounds.
  //!
  //!\param size Size in bytes of next read.
  //!\return Updated left endpoint after next read.
  unsigned char const *next(const std::size_t size) const;

  //! Left endpoint of the window.
  unsigned char const *current;

  //! Right endpoint of the window.
  unsigned char const *const end;
};
#endif

} // namespace mgard

#include "format.tpp"
#endif
