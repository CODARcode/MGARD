#include <cassert>
#include <climits>

#include <memory>
#include <stdexcept>
#include <type_traits>

namespace mgard {

namespace {

//! Deserialize an unsigned integer.
template <typename T, std::size_t N>
T deserialize(const std::array<unsigned char, N> &bytes) {
  static_assert(std::is_unsigned<T>::value,
                "can only deserialize unsigned integral types");
  T out = 0;
  // Multibyte metadata values are stored in big-endian format.
  using It = typename std::array<unsigned char, N>::const_iterator;
  for (It p = bytes.cbegin(); p != bytes.cend(); ++p) {
    out <<= CHAR_BIT;
    out ^= *p;
  }
  return out;
}

//! Serialize an unsigned integer.
template <typename T, std::size_t N>
std::array<unsigned char, N> serialize(T in) {
  static_assert(std::is_unsigned<T>::value,
                "can only serialize unsigned integral types");
  std::array<unsigned char, N> bytes;
  // Multibyte metadata values are stored in big-endian format.
  using It = typename std::array<unsigned char, N>::reverse_iterator;
  for (It p = bytes.rbegin(); p != bytes.rend(); ++p) {
    *p = in;
    in >>= CHAR_BIT;
  }
  assert(not in);
  return bytes;
}

} // namespace

template <typename T> void check_alignment(void const *const p) {
  if (p == nullptr) {
    throw std::invalid_argument("can't check alignment of null pointer");
  }
  void *q = const_cast<void *>(p);
  const std::size_t size = sizeof(T);
  std::size_t space = 2 * size;
  if (p != std::align(alignof(T), size, q, space)) {
    throw std::invalid_argument("pointer misaligned");
  }
}

template <typename Int> bool big_endian() {
  static_assert(std::is_integral<Int>::value,
                "can only check endianness of integral types");
  const Int n = 1;
  return not*reinterpret_cast<unsigned char const *>(&n);
}

template <typename T>
T read_message(BufferWindow &window, const std::uint_least64_t nmessage) {
  // The `CodedInputStream` constructor takes an `int`.
  if (nmessage > std::numeric_limits<int>::max()) {
    throw std::runtime_error("message is too large (size would overflow)");
  }
  // Check that the read will stay in the buffer.
  unsigned char const *const next = window.next(nmessage);
  T message;
  google::protobuf::io::CodedInputStream stream(
      static_cast<google::protobuf::uint8 const *>(window.current), nmessage);
  if (not message.ParseFromCodedStream(&stream)) {
    throw std::runtime_error(
        "message parsing encountered read or format error");
  }
  if (not stream.ConsumedEntireMessage()) {
    throw std::runtime_error("part of message left unparsed");
  }
  window.current = next;
  return message;
}

} // namespace mgard
