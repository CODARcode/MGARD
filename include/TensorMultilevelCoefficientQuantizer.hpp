#ifndef TENSORMULTILEVELCOEFFICIENTQUANTIZER_HPP
#define TENSORMULTILEVELCOEFFICIENTQUANTIZER_HPP
//!\file
//!\brief Quantizer for multilevel coefficients on tensor product grids.

#include "LinearQuantizer.hpp"
#include "utilities.hpp"

namespace mgard {

//! Forward declaration.
template <std::size_t N, typename Real, typename Int>
struct TensorQuantizedRange;

//! Quantizer for multilevel coefficients on tensor product grids. Each
//! coefficient is quantized according to its contribution to the error
//! indicator.
template <std::size_t N, typename Real, typename Int>
class TensorMultilevelCoefficientQuantizer {
public:
  //! Constructor.
  //!
  //!\param hierarchy Mesh hierarchy on which the coefficients to be quantized
  //! are defined.
  //!\param s Smoothness parameter. Determines the error norm in which
  //! quantization error is controlled.
  //!\param tolerance Quantization error tolerance for the entire set of
  //! multilevel coefficients.
  TensorMultilevelCoefficientQuantizer(
      const TensorMeshHierarchy<N, Real> &hierarchy, const Real s,
      const Real tolerance);

  //! Quantize a multilevel coefficient.
  //!
  //!\param coefficient Multilevel coefficient to be quantized, along with its
  //! auxiliary mesh data.
  Int operator()(const SituatedCoefficient<N, Real> coefficient) const;

  //! Iterator used to traverse a quantized range. Note that the iterator is not
  //! used to iterate over the `TensorMultilevelCoefficientQuantizer` itself.
  class iterator;

  //! Quantize a set of multilevel coefficients.
  //!
  //!\param u Multilevel coefficients to be quantized.
  TensorQuantizedRange<N, Real, Int> operator()(Real *const u) const;

  //! Associated mesh hierarchy.
  const TensorMeshHierarchy<N, Real> &hierarchy;

  //! Smoothness parameter.
  const Real s;

  //! Global quantization error tolerance.
  const Real tolerance;

private:
  //! Quantizer to use when controlling error in the supremum norm.
  const LinearQuantizer<Real, Int> supremum_quantizer;
};

//! Equality comparison.
template <std::size_t N, typename Real, typename Int>
bool operator==(const TensorMultilevelCoefficientQuantizer<N, Real, Int> &a,
                const TensorMultilevelCoefficientQuantizer<N, Real, Int> &b);

//! Inequality comparison.
template <std::size_t N, typename Real, typename Int>
bool operator==(const TensorMultilevelCoefficientQuantizer<N, Real, Int> &a,
                const TensorMultilevelCoefficientQuantizer<N, Real, Int> &b);

//! Iterable of quantized (as the object is iterated over) multilevel
//! coefficients.
//
// I am introducing this object to avoid a dangling reference, and hope to
// reimplement by splitting `SituatedCoefficient` later. It should probably
// inherit from `RangeSlice`, but that would require (it seems) a user-defined
// constructor for `RangeSlice`.
template <std::size_t N, typename Real, typename Int>
class TensorQuantizedRange {
public:
  using It =
      typename TensorMultilevelCoefficientQuantizer<N, Real, Int>::iterator;

  //! Constructor.
  //!
  //!\param quantizer Associated multilevel coefficient quantizer.
  //!\param u Multilevel coefficients to be quantized.
  TensorQuantizedRange(
      const TensorMultilevelCoefficientQuantizer<N, Real, Int> &quantizer,
      Real *const u);

  //! Return an iterator to the beginning of the quantized range.
  It begin() const;

  //! Return an iterator to the end of the quantized range.
  It end() const;

private:
  //! Multilevel coefficients to be quantized.
  //
  // This is stored so that the values don't go out of scope before iterators to
  // them (`begin_` and `end_`) do.
  const TensorLevelValues<N, Real> values;

  //! Iterator to the beginning of the quantized range.
  const It begin_;

  //! Iterator to the end of the quantized range.
  const It end_;
};

//! Iterator used to traverse a range of multilevel coefficients, quantizing as
//! it is dereferenced.
template <std::size_t N, typename Real, typename Int>
class TensorMultilevelCoefficientQuantizer<N, Real, Int>::iterator {
public:
  using iterator_category = std::input_iterator_tag;
  using value_type = Int;
  using difference_type = std::ptrdiff_t;
  using pointer = value_type *;
  using reference = value_type &;

  //! Constructor.
  //!
  //!\param quantizer Associated multilevel coefficient quantizer.
  //!\param inner_input Position in the auxiliary data range.
  //!\param inner_mc Position in the multilevel coefficient range.
  iterator(const TensorMultilevelCoefficientQuantizer &quantizer,
           const typename TensorLevelValues<N, Real>::iterator inner);

  //! Equality comparison.
  bool operator==(const iterator &other) const;

  //! Inequality comparison.
  bool operator!=(const iterator &other) const;

  //! Preincrement.
  iterator &operator++();

  //! Postincrement.
  iterator operator++(int);

  //! Dereference.
  value_type operator*() const;

private:
  //! Associated multilevel coefficient quantizer;
  const TensorMultilevelCoefficientQuantizer &quantizer;

  //! Iterator to current coefficient.
  typename TensorLevelValues<N, Real>::iterator inner;
};

} // namespace mgard

#include "TensorMultilevelCoefficientQuantizer.tpp"
#endif
