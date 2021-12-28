#include "compress_internal.hpp"

namespace mgard {

std::unique_ptr<unsigned char const []> decompress(void const *const data,
                                                   const std::size_t size,
                                                   const pb::Header &header) {
  check_mgard_version(header);
  check_file_format_version(header);
  const pb::Domain &domain = header.domain();
  const CartesianGridTopology topology = read_topology(domain);
  return decompress(data, size, header, topology.dimension);
}

} // namespace mgard
