#include "format.hpp"

#include <array>

namespace mgard {

namespace {

// Version number as array.
std::array<uint8_t, 3> vnaa(const VersionNumber &a) {
  return {a.major, a.minor, a.patch};
}

} // namespace

bool operator==(const VersionNumber &a, const VersionNumber &b) {
  return operator==(vnaa(a), vnaa(b));
}

bool operator!=(const VersionNumber &a, const VersionNumber &b) {
  return operator!=(vnaa(a), vnaa(b));
}

bool operator>=(const VersionNumber &a, const VersionNumber &b) {
  return operator>=(vnaa(a), vnaa(b));
}

bool operator>(const VersionNumber &a, const VersionNumber &b) {
  return operator>(vnaa(a), vnaa(b));
}

bool operator<=(const VersionNumber &a, const VersionNumber &b) {
  return operator<=(vnaa(a), vnaa(b));
}

bool operator<(const VersionNumber &a, const VersionNumber &b) {
  return operator<(vnaa(a), vnaa(b));
}

MetadataReader::MetadataReader(unsigned char const *const p) : p(p) {}

MetadataWriter::MetadataWriter(std::vector<unsigned char> &buffer)
    : buffer(buffer) {}

} // namespace mgard
