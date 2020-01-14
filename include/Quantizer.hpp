#ifndef QUANTIZER_HPP
#define QUANTIZER_HPP
//!\file
//!\brief Quantizers to round floating point numbers in preparation for lossless
//! compression.

#include <type_traits>

namespace mgard {

// Forward declarations.
template <typename Real, typename Int> class Quantizer;

template <typename Real, typename Int>
bool operator==(const Quantizer<Real, Int> &a, const Quantizer<Real, Int> &b);

//! Linear quantizer for multilevel coefficients.
//!
//! In principle, we probably want to quantize the coefficients all together (or
//! at least in batches). We support (de)quantization of individual numbers to
//! ease integration with existing code.
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

  //! Equality comparison.
  friend bool operator==<>(const Quantizer &a, const Quantizer &b);

private:
  //! Spacing between adjacent quantized numbers.
  const Real quantum;

  //! Greatest value which cannot be quantized using `quantum`.
  const Real minimum;

  //! Least value which cannot be quantized using `quantum`.
  const Real maximum;
};

//! Inequality comparison.
template <typename Real, typename Int>
bool operator!=(const Quantizer<Real, Int> &a, const Quantizer<Real, Int> &b);


  virtual Int do_quantize(const Real x) const;
  virtual Real do_dequantize(const Int n) const;
};

} // namespace mgard

#include "Quantizer/Quantizer.tpp"
#endif
