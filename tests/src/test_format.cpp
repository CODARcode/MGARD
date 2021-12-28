#include "catch2/catch_test_macros.hpp"

#include <cstdint>
#include <random>

#include "testing_utilities.hpp"

#include "MGARDConfig.hpp"
#include "format.hpp"

#ifdef MGARD_PROTOBUF
#include <zlib.h>
#endif

namespace {
using SizeBytes = std::array<unsigned char, mgard::HEADER_SIZE_SIZE>;
using CRC32Bytes = std::array<unsigned char, mgard::HEADER_CRC32_SIZE>;
} // namespace

TEST_CASE("header size and CRC32 deserialization", "[format]") {
  {
    const SizeBytes bytes{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf7};
    REQUIRE(mgard::deserialize_header_size(bytes) == 247ULL);
  }
  {
    const SizeBytes bytes{0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6, 0x07, 0x18};
    REQUIRE(mgard::deserialize_header_size(bytes) == 11651590505119483672ULL);
  }
  {
    const CRC32Bytes bytes{0x00, 0xac, 0x00, 0x00};
    REQUIRE(mgard::deserialize_header_crc32(bytes) == 11272192UL);
  }
}

TEST_CASE("header size and CRC32 serialization", "[format]") {
  {
    const SizeBytes expected{0x02, 0x03, 0x05, 0x07, 0x0b, 0x0d, 0x11, 0x13};
    REQUIRE(mgard::serialize_header_size(144965140814303507ULL) == expected);
  }
  {
    const CRC32Bytes expected{0x00, 0x00, 0x07, 0xe1};
    REQUIRE(mgard::serialize_header_crc32(2017UL) == expected);
  }
  {
    const CRC32Bytes expected{0x00, 0xc8, 0x28, 0x5c};
    REQUIRE(mgard::serialize_header_crc32(13117532UL) == expected);
  }
}

TEST_CASE("deserialization inverts serialization", "[format]") {
  const std::size_t ntrials = 500;
  std::default_random_engine gen(963912);
  {
    std::uniform_int_distribution<std::uint_least64_t> dis;
    TrialTracker tracker;
    for (std::size_t i = 0; i < ntrials; ++i) {
      const uint_least64_t n = dis(gen);
      tracker +=
          n == mgard::deserialize_header_size(mgard::serialize_header_size(n));
    }
    REQUIRE(tracker);
  }
  {
    std::uniform_int_distribution<unsigned char> dis;
    TrialTracker tracker;
    for (std::size_t i = 0; i < ntrials; ++i) {
      CRC32Bytes bytes;
      for (unsigned char &byte : bytes) {
        byte = dis(gen);
      }
      tracker += bytes == mgard::serialize_header_crc32(
                              mgard::deserialize_header_crc32(bytes));
    }
    REQUIRE(tracker);
  }
}

#ifdef MGARD_PROTOBUF
namespace {

void check_version_number(const mgard::pb::VersionNumber &version_number,
                          const google::protobuf::uint64 major_,
                          const google::protobuf::uint64 minor_,
                          const google::protobuf::uint64 patch_) {
  REQUIRE(version_number.major_() == major_);
  REQUIRE(version_number.minor_() == minor_);
  REQUIRE(version_number.patch_() == patch_);
}

} // namespace

TEST_CASE("setting version numbers", "[format]") {
  mgard::pb::Header header;
  mgard::populate_version_numbers(header);
  check_version_number(header.mgard_version(), MGARD_VERSION_MAJOR,
                       MGARD_VERSION_MINOR, MGARD_VERSION_PATCH);
  check_version_number(header.file_format_version(), MGARD_FILE_VERSION_MAJOR,
                       MGARD_FILE_VERSION_MINOR, MGARD_FILE_VERSION_PATCH);
}

TEST_CASE("advancing buffer windows", "[format]") {
  const std::size_t N = 10;
  unsigned char const *const p = new unsigned char[N];
  mgard::BufferWindow window(p, N);
  REQUIRE(window.current == p);
  REQUIRE(window.end == p + N);
  window.current = window.next(3);
  window.current = window.next(4);
  window.current = window.next(1);
  REQUIRE_THROWS(window.next(3));
}

TEST_CASE("magic number", "[format]") {
  unsigned char buffer[5];
  for (std::size_t i = 0; i < 5; ++i) {
    buffer[i] = i;
  }
  mgard::BufferWindow window(buffer, 5);

  REQUIRE_THROWS(mgard::check_magic_number(window));

  for (std::size_t i = 0; i < 5; ++i) {
    buffer[i] = mgard::SIGNATURE.at(i);
  }
  window.current = buffer;
  REQUIRE_NOTHROW(mgard::check_magic_number(window));
  REQUIRE(window.current == buffer + 5);
}

TEST_CASE("reading header size and CRC32", "[format]") {
  const std::uint_least64_t header_size = 20;
  const std::uint_least32_t header_crc32 = 0x670b;
  const std::size_t n = mgard::HEADER_SIZE_SIZE + mgard::HEADER_CRC32_SIZE;
  unsigned char buffer[n];

  {
    unsigned char *p = buffer;
    for (unsigned char c : mgard::serialize_header_size(header_size)) {
      *p++ = c;
    }
    for (unsigned char c : mgard::serialize_header_crc32(header_crc32)) {
      *p++ = c;
    }
  }

  mgard::BufferWindow window(buffer, n);
  REQUIRE(mgard::read_header_size(window) == header_size);
  REQUIRE(mgard::read_header_crc32(window) == header_crc32);
}

TEST_CASE("checking header CRC32", "[format]") {
  const std::uint_least64_t header_size = 40;
  unsigned char buffer[header_size];
  for (std::uint_least64_t i = 0; i < header_size; ++i) {
    buffer[i] = i % 17;
  }

  mgard::BufferWindow window(buffer, header_size);
  uLong header_crc32 = crc32_z(0, Z_NULL, 0);
  header_crc32 = crc32_z(
      header_crc32, static_cast<const Bytef *>(window.current), header_size);

  REQUIRE_NOTHROW(check_header_crc32(window, header_size, header_crc32));
}

TEST_CASE("dataset types", "[format]") {
  REQUIRE(mgard::type_to_dataset_type<float>() == mgard::pb::Dataset::FLOAT);
  REQUIRE(mgard::type_to_dataset_type<double>() == mgard::pb::Dataset::DOUBLE);
}

TEST_CASE("quantization type sizes", "[format]") {
  mgard::pb::Header header;
  mgard::pb::Quantization &quantization = *header.mutable_quantization();

  quantization.set_type(mgard::pb::Quantization::INT8_T);
  REQUIRE(mgard::quantization_type_size(header) == 1);

  quantization.set_type(mgard::pb::Quantization::INT16_T);
  REQUIRE(mgard::quantization_type_size(header) == 2);

  quantization.set_type(mgard::pb::Quantization::INT32_T);
  REQUIRE(mgard::quantization_type_size(header) == 4);

  quantization.set_type(mgard::pb::Quantization::INT64_T);
  REQUIRE(mgard::quantization_type_size(header) == 8);
}

TEST_CASE("reading topology and geometry", "[format]") {
  mgard::pb::Domain domain;
  const std::size_t dimension = 3;
  const std::vector<std::size_t> shape{5, 5, 6};
  {
    domain.set_topology(mgard::pb::Domain::CARTESIAN_GRID);
    mgard::pb::CartesianGridTopology *const cgt =
        domain.mutable_cartesian_grid_topology();
    cgt->set_dimension(dimension);
    std::vector<std::size_t>::const_iterator p = shape.cbegin();
    cgt->add_shape(*p++);
    cgt->add_shape(*p++);
    REQUIRE_THROWS(mgard::read_topology(domain));
    cgt->add_shape(*p++);
  }
  const mgard::CartesianGridTopology cgt = mgard::read_topology(domain);
  REQUIRE(cgt.dimension == dimension);
  REQUIRE(cgt.shape == shape);

  { domain.set_geometry(mgard::pb::Domain::UNIT_CUBE); }
  {
    const mgard::CartesianGridGeometry cgg = read_geometry(domain, cgt);
    REQUIRE(cgg.uniform);
  }

  {
    domain.set_geometry(mgard::pb::Domain::EXPLICIT_CUBE);
    mgard::pb::ExplicitCubeGeometry &ecg =
        *domain.mutable_explicit_cube_geometry();
    for (const std::size_t n : shape) {
      for (std::size_t i = 0; i < n; ++i) {
        ecg.add_coordinates(static_cast<double>(i));
      }
    }
  }
  {
    const mgard::CartesianGridGeometry cgg = read_geometry(domain, cgt);
    REQUIRE(not cgg.uniform);
    TrialTracker tracker;
    for (std::size_t i = 0; i < dimension; ++i) {
      const std::vector<double> &xs = cgg.coordinates.at(i);
      for (std::size_t j = 0; j < shape.at(i); ++j) {
        tracker += xs.at(j) == static_cast<double>(j);
      }
    }
    REQUIRE(tracker);
  }

  // TODO: Test storage of coefficients in separate file.
}

TEST_CASE("reading dataset type", "[format]") {
  mgard::pb::Header header;
  mgard::pb::Dataset &d = *header.mutable_dataset();
  d.set_dimension(1);
  {
    d.set_type(mgard::pb::Dataset::FLOAT);
    REQUIRE(mgard::read_dataset_type(header) == mgard::pb::Dataset::FLOAT);
  }
  {
    d.set_type(mgard::pb::Dataset::DOUBLE);
    REQUIRE(mgard::read_dataset_type(header) == mgard::pb::Dataset::DOUBLE);
  }
}

TEST_CASE("reading error control parameters", "[format]") {
  mgard::pb::Header header;
  mgard::pb::ErrorControl &e = *header.mutable_error_control();
  {
    const double s = -0.25;
    const double tolerance = 0.01;
    e.set_mode(mgard::pb::ErrorControl::ABSOLUTE);
    e.set_norm(mgard::pb::ErrorControl::S_NORM);
    e.set_s(s);
    e.set_tolerance(tolerance);
    const mgard::ErrorControlParameters error_control =
        mgard::read_error_control(header);
    REQUIRE(error_control.s == s);
    REQUIRE(error_control.tolerance == tolerance);
  }
  {
    const double s = std::numeric_limits<double>::infinity();
    const double tolerance = 0.5;
    const double norm_of_original_data = 4;
    e.set_mode(mgard::pb::ErrorControl::RELATIVE);
    e.set_norm(mgard::pb::ErrorControl::L_INFINITY);
    e.set_s(s);
    e.set_tolerance(tolerance);
    e.set_norm_of_original_data(norm_of_original_data);
    const mgard::ErrorControlParameters error_control =
        mgard::read_error_control(header);
    REQUIRE(error_control.s == s);
    REQUIRE(error_control.tolerance == tolerance * norm_of_original_data);
  }
}

TEST_CASE("checking decomposition parameters", "[format]") {
  mgard::pb::Header header;
  mgard::pb::Decomposition &d = *header.mutable_decomposition();
  d.set_transform(mgard::pb::Decomposition::MULTILEVEL_COEFFICIENTS);
  {
    d.set_hierarchy(mgard::pb::Decomposition::GHOST_NODES);
    REQUIRE_THROWS(mgard::check_decomposition_parameters(header));
  }
  {
    d.set_hierarchy(mgard::pb::Decomposition::POWER_OF_TWO_PLUS_ONE);
    REQUIRE_NOTHROW(mgard::check_decomposition_parameters(header));
  }
}

TEST_CASE("reading quantization parameters", "[format]") {
  mgard::pb::Header header;
  mgard::pb::Quantization &q = *header.mutable_quantization();
  q.set_method(mgard::pb::Quantization::COEFFICIENTWISE_LINEAR);
  {
    q.set_bin_widths(mgard::pb::Quantization::PER_LEVEL);
    REQUIRE_THROWS(mgard::read_quantization(header));
  }
  {
    q.set_bin_widths(mgard::pb::Quantization::PER_COEFFICIENT);
    q.set_type(mgard::pb::Quantization::INT16_T);
    q.set_big_endian(true);
    const mgard::QuantizationParameters quantization =
        mgard::read_quantization(header);
    REQUIRE(quantization.type == mgard::pb::Quantization::INT16_T);
    REQUIRE(quantization.big_endian);
  }
}

TEST_CASE("reading encoding compressor", "[format]") {
  mgard::pb::Header header;
  mgard::pb::Encoding &e = *header.mutable_encoding();
  e.set_preprocessor(mgard::pb::Encoding::SHUFFLE);
  {
    e.set_compressor(mgard::pb::Encoding::GPU_HUFFMAN_LZ4);
    REQUIRE_THROWS(mgard::read_encoding_compressor(header));
  }
  {
    e.set_compressor(mgard::pb::Encoding::CPU_HUFFMAN_ZSTD);
    REQUIRE(mgard::read_encoding_compressor(header) ==
            mgard::pb::Encoding::CPU_HUFFMAN_ZSTD);
  }
}
#endif
