#ifndef LINEARQUANTIZER_HPP
#define LINEARQUANTIZER_HPP
//!\file
//!\brief Quantizers to round floating point numbers in preparation for lossless
//! compression.

#include <type_traits>

namespace mgard {

//! Linear quantizer for multilevel coefficients.
template <typename Real, typename Int> class LinearQuantizer {

  static_assert(std::is_floating_point<Real>::value,
                "`Real` must be a floating point type");
  static_assert(std::is_integral<Int>::value, "`Int` must be an integral type");
  static_assert(std::is_signed<Int>::value, "`Int` must be a signed type");

public:
  //! Constructor.
  //!
  //!\param quantum Spacing between adjacent quantized numbers.
  LinearQuantizer(const Real quantum);

  //! Quantize a floating point number to an integer.
  //!
  //!\param x Number to be quantized.
  Int operator()(const Real x) const;

  //! Spacing between adjacent quantized numbers.
  const Real quantum;

private:
  //! Greatest value which cannot be quantized using `quantum`.
  const Real minimum;

  //! Least value which cannot be quantized using `quantum`.
  const Real maximum;
};

//! Equality comparison.
template <typename Real, typename Int>
bool operator==(const LinearQuantizer<Real, Int> &a,
                const LinearQuantizer<Real, Int> &b);

//! Inequality comparison.
template <typename Real, typename Int>
bool operator!=(const LinearQuantizer<Real, Int> &a,
                const LinearQuantizer<Real, Int> &b);

//! Linear dequantizer for multilevel coefficients.
template <typename Int, typename Real> class LinearDequantizer {

  static_assert(std::is_floating_point<Real>::value,
                "`Real` must be a floating point type");
  static_assert(std::is_integral<Int>::value, "`Int` must be an integral type");
  static_assert(std::is_signed<Int>::value, "`Int` must be a signed type");

public:
  //! Constructor.
  //!
  //!\param quantum Spacing between adjacent quantized numbers.
  LinearDequantizer(const Real quantum);

  //! Dequantize an integer to a floating point number.
  //!
  //!\param n Number to be dequantized.
  Real operator()(const Int n) const;

  //! Spacing between adjacent quantized numbers.
  const Real quantum;
};

//! Equality comparison.
template <typename Int, typename Real>
bool operator==(const LinearDequantizer<Int, Real> &a,
                const LinearDequantizer<Int, Real> &b);

//! Inequality comparison.
template <typename Int, typename Real>
bool operator!=(const LinearDequantizer<Int, Real> &a,
                const LinearDequantizer<Int, Real> &b);

} // namespace mgard

#include "LinearQuantizer.tpp"
#endif
