#ifndef FORMAT_HPP
#define FORMAT_HPP
//!\file
//!\brief Self-describing file format for compressed datasets.

#include <cstdint>

#include <array>
#include <vector>

namespace mgard {

//! Processor used to compress (determining algorithm).
enum Processor : uint8_t { CPU, GPU, LAST };

//! Mode in which compression error was controlled.
enum ErrorMode : uint8_t { Absolute, Relative, LAST };

//! Norm family in which compression error was controlled.
enum ErrorNorm : uint8_t { SupremumNorm, SNorm, LAST };

//! Compressor applied to quantized multilevel coefficients.
enum LosslessCompressor : uint8_t { zlib, zstd, LAST };

//! Precision of input dataset.
enum DataPrecision : uint8_t { binary32, binary64, LAST };

//! Structure of input dataset.
enum DataStructure : uint8_t {
  CartesianGridUniform,
  CartesianGridNonuniform,
  LAST
};

//! Version number. See <https://semver.org/>.
struct VersionNumber {
  //! Major version.
  uint8_t major;

  //! Minor version.
  uint8_t minor;

  //! Patch version.
  uint8_t patch;

  //! Pack the components into an array.
  std::array<uint8_t, 3> to_array() const;
};

//! Equality comparison.
bool operator==(const VersionNumber &a, const VersionNumber &b);

//! Inequality comparison.
bool operator!=(const VersionNumber &a, const VersionNumber &b);

//! 'Greater than or equal to' comparison.
bool operator>=(const VersionNumber &a, const VersionNumber &b);

//! 'Greater than' comparison.
bool operator>(const VersionNumber &a, const VersionNumber &b);

//! 'Less than' comparison.
bool operator<(const VersionNumber &a, const VersionNumber &b);

//! 'Less than or equal to' comparison.
bool operator<=(const VersionNumber &a, const VersionNumber &b);

//! Metadata byte reader.
struct MetadataReader {
  //! Constructor.
  //!
  //!\param metadata Metadata buffer.
  MetadataReader(unsigned char const *const p);

  //! Current position in metadata buffer.
  unsigned char const *p;

  //! Read a metadatum and advance in the buffer.
  template <typename T> T read();
};

//! Metadata byte writer.
struct MetadataWriter {
  //! Constructor.
  MetadataWriter(std::vector<unsigned char> &buffer);

  //! Metadata buffer.
  std::vector<unsigned char> &buffer;

  //! Write a metadatum and advance in the buffer.
  //!
  //!\param Metadatum to write.
  template <typename T> void write(const T &t);
};

} // namespace mgard

#include "format.tpp"
#endif
