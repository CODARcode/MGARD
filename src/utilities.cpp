#include "utilities.hpp"

#include <climits>

#include <stdexcept>

namespace mgard {

Bits::Bits(unsigned char const *const begin, unsigned char const *const end,
           const unsigned char offset_end)
    : begin_(begin), end_(end), offset_end(offset_end) {
  if (offset_end >= CHAR_BIT) {
    throw std::invalid_argument(
        "offset must be smaller than number of bits in byte");
  }
}

Bits::Bits(unsigned char const *const begin, unsigned char const *const end)
    : Bits(begin, end, 0) {}

bool Bits::operator==(const Bits &other) const {
  return begin_ == other.begin_ and end_ == other.end_ and
         offset_end == other.offset_end;
}

bool Bits::operator!=(const Bits &other) const { return !operator==(other); }

Bits::iterator Bits::begin() const { return {*this, begin_, 0}; }

Bits::iterator Bits::end() const { return {*this, end_, offset_end}; }

Bits::iterator::iterator(const Bits &iterable, unsigned char const *const p,
                         const unsigned char offset)
    : iterable(iterable), p(p), offset(offset) {}

bool Bits::iterator::operator==(const Bits::iterator &other) const {
  return offset == other.offset and p == other.p and iterable == other.iterable;
}

bool Bits::iterator::operator!=(const Bits::iterator &other) const {
  return !operator==(other);
}

Bits::iterator &Bits::iterator::operator++() {
  ++offset;
  if (offset == CHAR_BIT) {
    ++p;
    offset = 0;
  }
  return *this;
}

Bits::iterator Bits::iterator::operator++(int) {
  const iterator tmp = *this;
  operator++();
  return tmp;
}

Bits::iterator::reference Bits::iterator::operator*() const {
  // Operator precedence: dereference, then left shift, then bitwise AND.
  return *p << offset & 0x80;
}

} // namespace mgard
