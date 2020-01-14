#include <stdexcept>

namespace mgard {

//`DequantizedRange`.

// Public (member) functions.

template <typename Real, typename Int, typename It>
DequantizedRange<Real, Int, It>::DequantizedRange(
    const Quantizer<Real, Int> &quantizer, const It begin, const It end)
    : quantizer(quantizer), begin_(begin), end_(end) {}

template <typename Real, typename Int, typename It>
bool operator==(const DequantizedRange<Real, Int, It> &a,
                const DequantizedRange<Real, Int, It> &b) {
  return a.quantizer == b.quantizer && a.begin_ == b.begin_ && a.end_ == b.end_;
}

template <typename Real, typename Int, typename It>
typename DequantizedRange<Real, Int, It>::iterator
DequantizedRange<Real, Int, It>::begin() const {
  return iterator(*this, begin_);
}

template <typename Real, typename Int, typename It>
typename DequantizedRange<Real, Int, It>::iterator
DequantizedRange<Real, Int, It>::end() const {
  return iterator(*this, end_);
}

template <typename Real, typename Int, typename It>
bool operator!=(const DequantizedRange<Real, Int, It> &a,
                const DequantizedRange<Real, Int, It> &b) {
  return !operator==(a, b);
}

//`DequantizedRange::iterator`.

template <typename Real, typename Int, typename It>
DequantizedRange<Real, Int, It>::iterator::iterator(
    const DequantizedRange &iterable, const It inner)
    : iterable(iterable), inner(inner) {
  // See explanation in `QuantizedRange.tpp`.
  if (inner != iterable.begin_ && inner != iterable.end_) {
    throw std::invalid_argument(
        "iterable must point to beginning of end of input range");
  }
}

template <typename Real, typename Int, typename It>
bool DequantizedRange<Real, Int, It>::iterator::
operator==(const DequantizedRange<Real, Int, It>::iterator &other) const {
  return iterable == other.iterable && inner == other.inner;
}

template <typename Real, typename Int, typename It>
bool DequantizedRange<Real, Int, It>::iterator::
operator!=(const DequantizedRange<Real, Int, It>::iterator &other) const {
  return !operator==(other);
}

template <typename Real, typename Int, typename It>
typename DequantizedRange<Real, Int, It>::iterator &
DequantizedRange<Real, Int, It>::iterator::operator++() {
  ++inner;
  return *this;
}

template <typename Real, typename Int, typename It>
typename DequantizedRange<Real, Int, It>::iterator
DequantizedRange<Real, Int, It>::iterator::operator++(int) {
  const DequantizedRange<Real, Int, It>::iterator tmp = *this;
  this->operator++();
  return tmp;
}

template <typename Real, typename Int, typename It>
Real DequantizedRange<Real, Int, It>::iterator::operator*() const {
  return iterable.quantizer.dequantize(*inner);
}

template <typename Real, typename Int, typename It>
bool operator!=(const typename DequantizedRange<Real, Int, It>::iterator &a,
                const typename DequantizedRange<Real, Int, It>::iterator &b) {
  return !operator==(a, b);
}

} // namespace mgard
