#ifndef QUANTIZER_HPP
#define QUANTIZER_HPP
//!\file
//!\brief Quantizers to round floating point numbers in preparation for lossless
//! compression.

#include <iterator>
#include <type_traits>

namespace mgard {

// Forward declarations.
template <typename Real, typename Int, typename It> class QuantizedRange;

template <typename Real, typename Int, typename It> class DequantizedRange;

template <typename Real, typename Int> class Quantizer;

//! Equality comparison.
template <typename Real, typename Int>
bool operator==(const Quantizer<Real, Int> &a, const Quantizer<Real, Int> &b);

//! Linear quantizer for multilevel coefficients.
//!
//! In principle, we probably want to quantize the coefficients all together (or
//! at least in batches). We support (de)quantization of individual numbers to
//! ease integration with existing code.
template <typename Real, typename Int> class Quantizer {

  static_assert(std::is_floating_point<Real>::value,
                "`Real` must be a floating point type");
  static_assert(std::is_integral<Int>::value, "`Int` must be an integral type");
  static_assert(std::is_signed<Int>::value, "`Int` must be a signed type");

public:
  //! Constructor.
  //!
  //!\param quantum Spacing between adjacent quantized numbers.
  Quantizer(const Real quantum);

  // I would prefer the (de)quantization member functions to delegate to virtual
  // functions, but member function templates cannot be virtual, so this can't
  // be done for the iterator overloads.

  //! Quantize a floating point number to an integer.
  //!
  //!\param x Number to be quantized.
  Int quantize(const Real x) const;

  //! Quantize a range of floating point numbers.
  //!
  //!\param begin Beginning of range of numbers to quantize.
  //!\param end End of range of numbers to quantize.
  template <typename It>
  QuantizedRange<Real, Int, It> quantize(const It begin, const It end) const;

  //! Dequantize an integer to a floating point number.
  //!
  //!\param n Number to be dequantized.
  Real dequantize(const Int n) const;

  //! Dequantize a range of integers.
  //!
  //!\param begin Beginning of range of numbers to dequantize.
  //!\param end End of range of numbers to dequantize.
  template <typename It>
  DequantizedRange<Real, Int, It> dequantize(const It begin,
                                             const It end) const;

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

// Forward declarations.
template <typename Real, typename Int, typename It> class QuantizedRange;

//! Equality comparison.
template <typename Real, typename Int, typename It>
bool operator==(const QuantizedRange<Real, Int, It> &a,
                const QuantizedRange<Real, Int, It> &b);

// I have failed to define `operator==` as a friend to the inner iterator class
// using the 'introvert' method given in <https://stackoverflow.com/a/4661372>.
// The problem seems to be caused by some combination of the following.
//
//* `QuantizedRange` is a class template.
//* `iterator` is a nested class.
//* The 'introvert' method requires the nested class to be forward declared.
//
// See, possibly, <https://stackoverflow.com/q/57504699>.

//! Range obtained by quantizing some other range.
template <typename Real, typename Int, typename It> class QuantizedRange {

  using T = typename std::iterator_traits<It>::value_type;
  static_assert(std::is_same<T, Real>::value,
                "`It` must dereference to `Real`");

public:
  //! Constructor.
  //!
  //!\param quantizer Quantizer.
  //!\param begin Beginning of range of numbers to quantize.
  //!\param end End of range of numbers to quantize.
  QuantizedRange(const Quantizer<Real, Int> &quantizer, const It begin,
                 const It end);

  //! Equality comparison.
  friend bool operator==<>(const QuantizedRange &a, const QuantizedRange &b);

  //! Associated quantizer.
  const Quantizer<Real, Int> &quantizer;

  // Forward declaration.
  class iterator;

  //! Return an iterator to the beginning of the quantized range.
  iterator begin() const;

  //! Return an iterator to the end of the quantized range.
  iterator end() const;

private:
  const It begin_;
  const It end_;
};

//! Inequality comparison.
template <typename Real, typename Int, typename It>
bool operator!=(const QuantizedRange<Real, Int, It> &a,
                const QuantizedRange<Real, Int, It> &b);

//! Iterator for quantized range.
template <typename Real, typename Int, typename It>
class QuantizedRange<Real, Int, It>::iterator
    : public std::iterator<std::input_iterator_tag, Int> {
public:
  //! Constructor.
  //!
  //!\param iterable Quantized range to which this iterator is associated.
  //!\param inner Iterator that dereferences to range elements.
  iterator(const QuantizedRange &iterable, const It inner);

  // See note about above attempt to make this a nonmember friend.

  //! Equality comparison.
  bool operator==(const iterator &other) const;

  //! Inequality comparison.
  bool operator!=(const iterator &other) const;

  //! Preincrement.
  iterator &operator++();

  //! Postincrement.
  iterator operator++(int);

  //! Dereference.
  Int operator*() const;

private:
  const QuantizedRange &iterable;
  It inner;
};

// Forward declarations.
template <typename Real, typename Int, typename It> class DequantizedRange;

//! Equality comparison.
template <typename Real, typename Int, typename It>
bool operator==(const DequantizedRange<Real, Int, It> &a,
                const DequantizedRange<Real, Int, It> &b);

//! Range obtained by dequantizing some other range.
template <typename Real, typename Int, typename It> class DequantizedRange {

  using T = typename std::iterator_traits<It>::value_type;
  static_assert(std::is_same<T, Int>::value, "`It` must dereference to `Int`");

public:
  //! Constructor.
  //!
  //!\param quantizer Quantizer.
  //!\param begin Beginning of range of numbers to dequantize.
  //!\param end End of range of numbers to dequantize.
  DequantizedRange(const Quantizer<Real, Int> &quantizer, const It begin,
                   const It end);

  //! Equality comparison.
  friend bool operator==
      <>(const DequantizedRange &a, const DequantizedRange &b);

  //! Associated quantizer.
  const Quantizer<Real, Int> &quantizer;

  // Forward declaration.
  class iterator;

  //! Return an iterator to the beginning of the dequantized range.
  iterator begin() const;

  //! Return an iterator to the end of the dequantized range.
  iterator end() const;

private:
  const It begin_;
  const It end_;
};

//! Inequality comparison.
template <typename Real, typename Int, typename It>
bool operator!=(const DequantizedRange<Real, Int, It> &a,
                const DequantizedRange<Real, Int, It> &b);

//! Iterator for dequantized range.
template <typename Real, typename Int, typename It>
class DequantizedRange<Real, Int, It>::iterator
    : public std::iterator<std::input_iterator_tag, Real> {
public:
  //! Constructor.
  //!
  //!\param iterable Dequantized range to which this iterator is
  //! associated.
  //!\param inner Iterator that dereferences to range elements.
  iterator(const DequantizedRange &iterable, const It inner);

  //! Equality comparison.
  bool operator==(const iterator &other) const;

  //! Inequality comparison.
  bool operator!=(const iterator &other) const;

  //! Preincrement.
  iterator &operator++();

  //! Postincrement.
  iterator operator++(int);

  //! Dereference.
  Real operator*() const;

private:
  const DequantizedRange &iterable;
  It inner;
};

} // namespace mgard

#include "Quantizer/DequantizedRange.tpp"
#include "Quantizer/QuantizedRange.tpp"
#include "Quantizer/Quantizer.tpp"
#endif
