#ifndef MULTILEVELCOEFFICIENTQUANTIZER_HPP
#define MULTILEVELCOEFFICIENTQUANTIZER_HPP
//!\file
//!\brief Quantizer for multilevel coefficients.

#include <cstddef>

#include <iterator>

#include "IndicatorInput.hpp"
#include "MeshHierarchy.hpp"
#include "data.hpp"
#include "utilities.hpp"

namespace mgard {

//! Quantizer for multilevel coefficients. Each coefficient is quantized
//! according to its contribution to the error indicator.
template <typename Real, typename Int> class MultilevelCoefficientQuantizer {
public:
  //! Constructor.
  //!
  //!\param hierarchy Mesh hierarchy on which the coefficients to be quantized
  //! are defined.
  //!\param s Smoothness parameter. Determines the error norm in which
  //! quantization error is controlled.
  //!\param tolerance Quantization error tolerance for the entire set of
  //! multilevel coefficients.
  MultilevelCoefficientQuantizer(const MeshHierarchy &hierarchy, const float s,
                                 const float tolerance);

  //! Quantize a multilevel coefficient.
  //!
  //!\param input Auxiliary mesh data needed to compute the associated indicator
  //! coefficient factor.
  //!\param x Multilevel coefficient to be quantized.
  Int operator()(const IndicatorInput input, const Real x) const;

  //! Iterator used to traverse a quantized range. Note that the iterator is not
  //! used to iterate over the `MultilevelCoefficientQuantizer` itself.
  class iterator;

  //! Quantize a set of multilevel coefficients.
  //!
  //!\param u Multilevel coefficients to be quantized.
  RangeSlice<iterator> operator()(const MultilevelCoefficients<Real> u) const;

  //! Associated mesh hierarchy.
  const MeshHierarchy &hierarchy;

  //! Smoothness parameter.
  const float s;

  //! Global quantization error tolerance.
  const float tolerance;

private:
  //! Associated indicator input range.
  const IndicatorInputRange indicator_input_range;
};

//! Equality comparison.
template <typename Real, typename Int>
bool operator==(const MultilevelCoefficientQuantizer<Real, Int> &a,
                const MultilevelCoefficientQuantizer<Real, Int> &b);

//! Inequality comparison.
template <typename Real, typename Int>
bool operator==(const MultilevelCoefficientQuantizer<Real, Int> &a,
                const MultilevelCoefficientQuantizer<Real, Int> &b);

//! Iterator used to traverse a range of multilevel coefficients, quantizing as
//! it is dereferenced.
template <typename Real, typename Int>
class MultilevelCoefficientQuantizer<Real, Int>::iterator {
public:
  using iterator_category = std::input_iterator_tag;
  using value_type = Int;
  using difference_type = std::ptrdiff_t;
  using pointer = value_type *;
  using reference = value_type;

  //! Constructor.
  //!
  //!\param quantizer Associated multilevel coefficient quantizer.
  //!\param inner_input Position in the auxiliary data range.
  //!\param inner_mc Position in the multilevel coefficient range.
  iterator(const MultilevelCoefficientQuantizer &quantizer,
           const IndicatorInputRange::iterator inner_input,
           Real const *const inner_mc);

  //! Equality comparison.
  bool operator==(const iterator &other) const;

  //! Inequality comparison.
  bool operator!=(const iterator &other) const;

  //! Preincrement.
  iterator &operator++();

  //! Postincrement.
  iterator operator++(int);

  //! Dereference.
  reference operator*() const;

private:
  //! Associated multilevel coefficient quantizer;
  const MultilevelCoefficientQuantizer &quantizer;

  //! Iterator to current indicator input.
  IndicatorInputRange::iterator inner_input;

  //! Iterator to current multilevel coefficient.
  Real const *inner_mc;
};

//! Dequantizer for multilevel coefficients. Acts as an approximate inverse of
//! `MultilevelCoefficientQuantizer`.
template <typename Int, typename Real> class MultilevelCoefficientDequantizer {
public:
  //! Constructor.
  //!
  //!\param hierarchy Mesh hierarchy on which the dequantized coefficients
  //! will be defined.
  //!\param s Smoothness parameter. Determines the error norm in which
  //! quantization error was controlled.
  //!\param tolerance Quantization error tolerance.
  MultilevelCoefficientDequantizer(const MeshHierarchy &hierarchy,
                                   const float s, const float tolerance);

  //! Dequantize a quantized multilevel coefficient.
  //!
  //!\param input Auxiliary mesh data needed to compute the associated indicator
  //! coefficient factor.
  //!\param n Quantized multilevel coefficient to be dequantized.
  Real operator()(const IndicatorInput input, const Int n) const;

  //! Iterator used to traverse a dequantized range. Note that the iterator is
  //! not used to iterate over the `MultilevelCoefficientDequantizer` itself.
  template <typename It> class iterator;

  //! Dequantize a set of quantized multilevel coefficients.
  //!
  //!\param begin Beginning of quantized range.
  //!\param end End of quantized range.
  //!
  //! `begin` must point to the first quantized multilevel coefficient, and
  //! `end` must point one beyond the last. In particular, this function cannot
  //! be used to dequantize a proper subset of the multilevel coefficients,
  //! because the auxiliary mesh data needed to compute the quantization factors
  //! is fixed by the choice of `hierarchy` in the constructor.
  template <typename It>
  RangeSlice<iterator<It>> operator()(const It begin, const It end) const;

  //! Associated mesh hierarchy.
  const MeshHierarchy &hierarchy;

  //! Smoothness parameter.
  const float s;

  //! Global quantization error tolerance.
  const float tolerance;

private:
  //! Associated indicator input range.
  const IndicatorInputRange indicator_input_range;
};

//! Equality comparison.
template <typename Int, typename Real>
bool operator==(const MultilevelCoefficientDequantizer<Int, Real> &a,
                const MultilevelCoefficientDequantizer<Int, Real> &b);

//! Inequality comparison.
template <typename Int, typename Real>
bool operator==(const MultilevelCoefficientDequantizer<Int, Real> &a,
                const MultilevelCoefficientDequantizer<Int, Real> &b);

//! Iterator used to traverse a range of quantized multilevel coefficients,
//! dequantizing as it is dereferenced.
template <typename Int, typename Real>
template <typename It>
class MultilevelCoefficientDequantizer<Int, Real>::iterator {

  using T = typename std::iterator_traits<It>::value_type;
  static_assert(std::is_same<T, Int>::value, "`It` must dereference to `Int`");

public:
  using iterator_category = std::input_iterator_tag;
  using value_type = Real;
  using difference_type = std::ptrdiff_t;
  using pointer = value_type *;
  using reference = value_type;

  //! Constructor.
  //!
  //!\param dequantizer Associated multilevel coefficient dequantizer.
  //!\param inner_input Position in the auxiliary data range.
  //!\param inner_qc Position in the quantized multilevel coefficient range.
  iterator(const MultilevelCoefficientDequantizer &dequantizer,
           const IndicatorInputRange::iterator inner_input, const It inner_qc);

  //! Equality comparison.
  bool operator==(const iterator &other) const;

  //! Inequality comparison.
  bool operator!=(const iterator &other) const;

  //! Preincrement.
  iterator &operator++();

  //! Postincrement.
  iterator operator++(int);

  //! Dereference.
  reference operator*() const;

private:
  //! Associated multilevel coefficient dequantizer.
  const MultilevelCoefficientDequantizer &dequantizer;

  //! Iterator to current indicator input.
  IndicatorInputRange::iterator inner_input;

  //! Iterator to current quantized multilevel coefficient.
  It inner_qc;
};

} // namespace mgard

#include "MultilevelCoefficientQuantizer.tpp"
#endif
