#include "catch2/catch_test_macros.hpp"

#include <random>

#include "testing_utilities.hpp"

#include "MGARDConfig.hpp"
#include "format.hpp"

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
#endif
