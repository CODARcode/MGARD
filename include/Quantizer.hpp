#ifndef QUANTIZER_HPP
#define QUANTIZER_HPP
//!\file
//!\brief Quantizers to round floating point numbers in preparation for lossless
//! compression.

#include <type_traits>

namespace mgard {

template <typename Real, typename Int> class Quantizer {

  static_assert(std::is_floating_point<Real>::value);
  static_assert(std::is_integral<Int>::value);
  static_assert(std::is_signed<Int>::value);

public:
  //! Constructor.
  //!
  //!\param quantum Spacing between adjacent quantized numbers.
  Quantizer(const Real quantum);

  //! Quantize a floating point number to an integer.
  //!
  //!\param x Number to be quantized.
  Int quantize(const Real x) const;

  //! Dequantize an integer to a floating point number.
  //!
  //!\param n Number to be dequantized.
  Real dequantize(const Int n) const;

private:
  //! Spacing between adjacent quantized numbers.
  const Real quantum;

  //! Greatest value which cannot be quantized using `quantum`.
  const Real minimum;

  //! Least value which cannot be quantized using `quantum`.
  const Real maximum;

  virtual Int do_quantize(const Real x) const;
  virtual Real do_dequantize(const Int n) const;
};

} // namespace mgard

#include "Quantizer.tpp"
#endif
