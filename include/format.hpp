#ifndef FORMAT_HPP
#define FORMAT_HPP
//!\file
//!\brief Self-describing file format for compressed datasets.

#include <cstdint>

#include <array>
#include <ostream>
#include <vector>

#include "proto/mgard.pb.h"

#ifdef __NVCC__
// NVCC breaks on `utilities.hpp`. See (we think) <https://github.com/CODARcode/
// MGARD/issues/126> and <https://forums.developer.nvidia.com/t/
// nvcc-preprocessor-bug-causes-compilation-failure/65962>.

//! Forward declaration.
template <typename T> struct MemoryBuffer;
#else
#include "utilities.hpp"
#endif

namespace mgard {

//! Magic bytes for MGARD file format ('MGARD' in ASCII).
inline constexpr std::array<char, 5> SIGNATURE{0x4d, 0x47, 0x41, 0x52, 0x44};

//! Size in bytes of the serialized header size.
inline constexpr std::size_t HEADER_SIZE_SIZE = 8;

//! Size in bytes of the serialized header CRC32.
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
//!\param crc32 CRC32 of the header.
//!\return bytes Serialized header CRC32.
std::array<unsigned char, HEADER_CRC32_SIZE>
serialize_header_crc32(std::uint_least64_t crc32);

//! Check that a pointer has the alignment for a type.
//!
//!\param p Pointer whose alignment will be checked.
template <typename T> void check_alignment(void const *const p);

//! Determine whether an integral type is big endian.
template <typename Int> bool big_endian();

//! Return the `Dataset::Type` value corresponding to a floating point type.
//!
//!\return `Dataset::Type` corresponding to `Real`.
template <typename Real> pb::Dataset::Type type_to_dataset_type();

//! Allocate a quantization buffer of the proper alignment and size.
//!
//!\param header Self-describing dataset header.
//!\param ndof Size of buffer (number of elements).
MemoryBuffer<unsigned char> quantization_buffer(const pb::Header &header,
                                                const std::size_t ndof);

//! Populate a header with the MGARD and file format version numbers.
//!
//!\param header Header to be populated.
void populate_version_numbers(pb::Header &header);

//! Populate a header with the default compression settings.
//!
//!\param header Header to be populated.
void populate_defaults(pb::Header &header);

//! Window of a buffer being parsed.
//!
//! The left endpoint of the window is advanced as we parse the buffer. The
//! right endpoint of the window is fixed.
struct BufferWindow {
  //! Constructor.
  //!
  //!\param data Beginning of the buffer.
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

//! Check that a self-describing buffer starts with the expected magic number.
//!
//! The buffer pointer will be advanced past the magic number.
//!
//!\param window Window into the self-describing buffer. The current position
//! should be the start of the magic number.
void check_magic_number(BufferWindow &window);

//! Read the size of the header from a self-describing buffer.
//!
//! The buffer pointer will be advanced past the header size.
//!
//!\param window Window into the self-describing buffer. The current position
//! should be the start of the header size.
//!\return Size of the header in bytes.
std::uint_least64_t read_header_size(BufferWindow &window);

//! Read the header CRC32 from a self-describing buffer.
//!
//! The buffer pointer will be advanced past the header CRC32.
//!
//!\param window Window into the self-describing buffer. The current position
//! should be the start of the header CRC32.
//!\return Expected header CRC32.
std::uint_least32_t read_header_crc32(BufferWindow &window);

//! Check that the header of a self-describing buffer has the expected CRC32.
//!
//! The buffer pointer will not be advanced.
//!
//!\param window Window into the self-describing buffer. The current position
//! should be the start of the header.
//!\param header_size Size in bytes of the header.
//!\param header_crc32 Expected CRC32 of the header.
void check_header_crc32(const BufferWindow &window,
                        const std::uint_least64_t header_size,
                        const std::uint_least32_t header_crc32);

//! Read the preheader and header of a self-describing buffer.
//!
//!\param window Window into the self-describing buffer. The current position
//! should be the start of the magic number.
pb::Header read_metadata(BufferWindow &window);

//! Write the preheader and header of a self-describing buffer.
//!
//!\param ostream Stream to write to.
//!\param header Header of the self-describing buffer.
void write_metadata(std::ostream &ostream, const pb::Header &header);

template <typename T>
//! Parse a message from a buffer window.
//!
//! The buffer pointer will be advanced past the header.
//!
//! This function was originally written to parse the header from a
//! self-describing buffer.
//
//!\param window Buffer window containing the serialized message. The current
//! position should be the start of the message.
//!\param nmessage Size in bytes of the message.
//!\return Parsed message.
T read_message(BufferWindow &window, const std::uint_least64_t nmessage);

//! Check that a dataset was compressed with a compatible version of MGARD.
//!
//!\param header Header of the compressed dataset.
void check_mgard_version(const pb::Header &header);

//! Check that a header was generated with a compatible version of MGARD.
//!
//!\param header Header of a self-describing buffer.
void check_file_format_version(const pb::Header &header);

//! Topology of a Cartesian grid.
//!
//! This struct summarizes a valid `pb::Domain` topology.
struct CartesianGridTopology {
  //! Dimension of the grid.
  std::size_t dimension;

  //! Shape of the grid.
  std::vector<std::size_t> shape;
};

//! Read and check the topology of a domain.
//!
//!\param domain Domain field of a self-describing header.
//!\return Topology of the domain.
CartesianGridTopology read_topology(const pb::Domain &domain);

//! Geometry of a Cartesian grid.
//!
//! This struct summarizes a valid `pb::Domain` geometry.
struct CartesianGridGeometry {
  //! Whether the nodes are uniformly spaced on `[0, 1]` in each dimension.
  bool uniform;

  //! Coordinates of the nodes if not `uniform`.
  std::vector<std::vector<double>> coordinates;
};

//! Read and check the geometry of a domain.
//!
//!\param domain Domain field of a self-describing header.
//!\param topology Topology of the domain.
//!\return Geometry of the domain.
CartesianGridGeometry read_geometry(const pb::Domain &domain,
                                    const CartesianGridTopology &topology);

//! Read and check the dataset type of a self-describing buffer.
//!
//!\param header Header of the self-describing buffer.
//!\return Dataset type of the self-describing buffer.
pb::Dataset::Type read_dataset_type(const pb::Header &header);

//! Parameters dictating how the compression error will be controlled.
//
//! This struct summarizes a valid `pb::ErrorControl` field.
struct ErrorControlParameters {
  //! Smoothness parameter.
  double s;

  //! Error tolerance.
  double tolerance;
};

//! Read and check the error control parameters of a self-describing buffer.
//!
//!\param header Header of the self-describing buffer.
//!\return Error control parameters of the self-describing buffer.
ErrorControlParameters read_error_control(const pb::Header &header);

//! Check the decomposition parameters of a self-describing buffer.
//!
//!\param header Header of the self-describing buffer.
void check_decomposition_parameters(const pb::Header &header);

//! Parameters dictating how the coefficients will be quantized.
//!
//! This struct summarizes a valid `pb::Quantization` field.
struct QuantizationParameters {
  //! Type used for quantizer output.
  pb::Quantization::Type type;

  //! Whether the quantizer output type is big endian.
  bool big_endian;
};

//! Read and check the quantization parameters of a self-describing buffer.
//!
//!\param header Header of the self-describing buffer.
//!\return Quantization parameters of the self-describing buffer.
QuantizationParameters read_quantization(const pb::Header &header);

//! Read and check the encoding compressor of a self-describing buffer.
//!
//!\param header Header of the self-describing buffer.
//!\return Encoding compressor of the self-describing buffer.
pb::Encoding::Compressor read_encoding_compressor(const pb::Header &header);

} // namespace mgard

#include "format.tpp"
#endif
