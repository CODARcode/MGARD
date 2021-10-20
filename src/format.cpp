#include "format.hpp"

namespace mgard {

std::array<uint8_t, 3> VersionNumber::to_array() {
  return {a.major, a.minor, a.patch};
}

bool operator==(const VersionNumber &a, const VersionNumber &b) {
  return operator==(a.to_array(), b.to_array());
}

bool operator!=(const VersionNumber &a, const VersionNumber &b) {
  return operator!=(a.to_array(), b.to_array());
}

bool operator>=(const VersionNumber &a, const VersionNumber &b) {
  return operator>=(a.to_array(), b.to_array());
}

bool operator>(const VersionNumber &a, const VersionNumber &b) {
  return operator>(a.to_array(), b.to_array());
}

bool operator<=(const VersionNumber &a, const VersionNumber &b) {
  return operator<=(a.to_array(), b.to_array());
}

bool operator<(const VersionNumber &a, const VersionNumber &b) {
  return operator<(a.to_array(), b.to_array());
}

MetadataReader::MetadataReader(unsigned char const *const p) : p(p) {}

MetadataWriter::MetadataWriter(std::vector<unsigned char> &buffer)
    : buffer(buffer) {}

} // namespace mgard
