#include "compress.hpp"

#include <stdexcept>

#include "compress_internal.hpp"
#include "format.hpp"

#ifndef MGARD_PROTOBUF
#error "This file shouldn't be compiled if ProtoBuf isn't found."
#endif

namespace mgard {

std::unique_ptr<unsigned char const []> decompress(void const *const data,
                                                   const std::size_t size) {
  BufferWindow window(data, size);
  const pb::Header header = read_metadata(window);
  const std::uint_least64_t header_size =
      window.current - static_cast<unsigned char const *>(data);
  if (header_size > size) {
    throw std::runtime_error("header size larger than overall size");
  }
  return decompress(static_cast<unsigned char const *>(data) + header_size,
                    size - header_size, header);
}

} // namespace mgard
