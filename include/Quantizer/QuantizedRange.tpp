#include <stdexcept>

namespace mgard {

//`QuantizedRange`.

// Public (member) functions.

template <typename Real, typename Int, typename It>
QuantizedRange<Real, Int, It>::QuantizedRange(
    const Quantizer<Real, Int> &quantizer, const It begin, const It end)
    : quantizer(quantizer), begin_(begin), end_(end) {}

template <typename Real, typename Int, typename It>
bool operator==(const QuantizedRange<Real, Int, It> &a,
                const QuantizedRange<Real, Int, It> &b) {
  return a.quantizer == b.quantizer && a.begin_ == b.begin_ && a.end_ == b.end_;
}

template <typename Real, typename Int, typename It>
typename QuantizedRange<Real, Int, It>::iterator
QuantizedRange<Real, Int, It>::begin() const {
  return iterator(*this, begin_);
}

template <typename Real, typename Int, typename It>
typename QuantizedRange<Real, Int, It>::iterator
QuantizedRange<Real, Int, It>::end() const {
  return iterator(*this, end_);
}

template <typename Real, typename Int, typename It>
bool operator!=(const QuantizedRange<Real, Int, It> &a,
                const QuantizedRange<Real, Int, It> &b) {
  return !operator==(a, b);
}

//`QuantizedRange::iterator`.

template <typename Real, typename Int, typename It>
QuantizedRange<Real, Int, It>::iterator::iterator(
    const QuantizedRange &iterable, const It inner)
    : iterable(iterable), inner(inner) {
  // This check is to prevent construction with an iterator that points, for
  // example, to the middle of a group of `float`s that are all supposed to be
  // mapped to the same `int`.
  if (inner != iterable.begin_ && inner != iterable.end_) {
    throw std::invalid_argument(
        "iterable must point to beginning of end of input range");
  }
}

template <typename Real, typename Int, typename It>
bool QuantizedRange<Real, Int, It>::iterator::
operator==(const QuantizedRange<Real, Int, It>::iterator &other) const {
  return iterable == other.iterable && inner == other.inner;
}

template <typename Real, typename Int, typename It>
bool QuantizedRange<Real, Int, It>::iterator::
operator!=(const QuantizedRange<Real, Int, It>::iterator &other) const {
  return !operator==(other);
}

template <typename Real, typename Int, typename It>
typename QuantizedRange<Real, Int, It>::iterator &
QuantizedRange<Real, Int, It>::iterator::operator++() {
  ++inner;
  return *this;
}

template <typename Real, typename Int, typename It>
typename QuantizedRange<Real, Int, It>::iterator
QuantizedRange<Real, Int, It>::iterator::operator++(int) {
  const QuantizedRange<Real, Int, It>::iterator tmp = *this;
  this->operator++();
  return tmp;
}

template <typename Real, typename Int, typename It>
Int QuantizedRange<Real, Int, It>::iterator::operator*() const {
  return iterable.quantizer.quantize(*inner);
}

template <typename Real, typename Int, typename It>
bool operator!=(const typename QuantizedRange<Real, Int, It>::iterator &a,
                const typename QuantizedRange<Real, Int, It>::iterator &b) {
  return !operator==(a, b);
}

} // namespace mgard
