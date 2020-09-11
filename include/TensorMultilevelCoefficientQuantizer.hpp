#ifndef TENSORMULTILEVELCOEFFICIENTQUANTIZER_HPP
#define TENSORMULTILEVELCOEFFICIENTQUANTIZER_HPP
//!\file
//!\brief Quantizer for multilevel coefficients on tensor product grids.

#include "LinearQuantizer.hpp"
#include "TensorMeshHierarchy.hpp"
#include "utilities.hpp"

namespace mgard {

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
  //!\param node Auxiliary node data corresponding to the coefficient.
  //!\param coefficient Multilevel coefficient to be quantized.
  Int operator()(const TensorNode<N, Real> node, const Real coefficient) const;

  //! Iterator used to traverse a quantized range. Note that the iterator is not
  //! used to iterate over the `TensorMultilevelCoefficientQuantizer` itself.
  class iterator;

  //! Quantize a set of multilevel coefficients.
  //!
  //!\param u Multilevel coefficients to be quantized.
  RangeSlice<iterator> operator()(Real const *const u) const;

  //! Associated mesh hierarchy.
  const TensorMeshHierarchy<N, Real> &hierarchy;

  //! Smoothness parameter.
  const Real s;

  //! Global quantization error tolerance.
  const Real tolerance;

private:
  //! Nodes of the finest mesh in the hierarchy.
  const TensorNodeRange<N, Real> nodes;

  //! Quantizer to use when controlling error in the supremum norm.
  const LinearQuantizer<Real, Int> supremum_quantizer;
};

//! Equality comparison.
template <std::size_t N, typename Real, typename Int>
bool operator==(const TensorMultilevelCoefficientQuantizer<N, Real, Int> &a,
                const TensorMultilevelCoefficientQuantizer<N, Real, Int> &b);

//! Inequality comparison.
template <std::size_t N, typename Real, typename Int>
bool operator!=(const TensorMultilevelCoefficientQuantizer<N, Real, Int> &a,
                const TensorMultilevelCoefficientQuantizer<N, Real, Int> &b);

//! Iterator used to traverse a range of multilevel coefficients, quantizing as
//! it is dereferenced.
template <std::size_t N, typename Real, typename Int>
class TensorMultilevelCoefficientQuantizer<N, Real, Int>::iterator {
public:
  using iterator_category = std::input_iterator_tag;
  using value_type = Int;
  using difference_type = std::ptrdiff_t;
  using pointer = value_type *;
  using reference = value_type;

  //! Constructor.
  //!
  //!\param quantizer Associated multilevel coefficient quantizer.
  //!\param inner_node Position in the node range.
  //!\param inner_mc Position in the multilevel coefficient range.
  iterator(const TensorMultilevelCoefficientQuantizer &quantizer,
           const typename TensorNodeRange<N, Real>::iterator inner_node,
           Real const *const inner_coeff);

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
  const TensorMultilevelCoefficientQuantizer &quantizer;

  //! Iterator to current node.
  typename TensorNodeRange<N, Real>::iterator inner_node;

  //! Iterator to current coefficient.
  Real const *inner_coeff;
};

//! Dequantizer for multilevel coefficients on tensor product grids.
template <std::size_t N, typename Int, typename Real>
class TensorMultilevelCoefficientDequantizer {
public:
  //! Constructor.
  //!
  //!\param hierarchy Mesh hierarchy on which the dequantized coefficients
  //! will be defined.
  //!\param s Smoothness parameter. Determines the error norm in which
  //! quantization error was controlled.
  //!\param tolerance Quantization error tolerance for the entire set of
  //! multilevel coefficients.
  TensorMultilevelCoefficientDequantizer(
      const TensorMeshHierarchy<N, Real> &hierarchy, const Real s,
      const Real tolerance);

  //! Dequantize a multilevel coefficient.
  //!
  //!\param node Auxiliary node data corresponding to the coefficient.
  //!\param coefficient Multilevel coefficient to be dequantized.
  Real operator()(const TensorNode<N, Real> node, const Int n) const;

  //! Iterator used to traverse a quantized range. Note that the iterator is not
  //! used to iterate over the `TensorMultilevelCoefficientDequantizer` itself.
  template <typename It> class iterator;

  //! Quantize a set of multilevel coefficients.
  //!
  //!\param begin Iterator to the beginning of the quantized coefficient range.
  //!\param end Iterator to the end of the quantized coefficient range.
  template <typename It>
  RangeSlice<iterator<It>> operator()(const It begin, const It end) const;

  //! Associated mesh hierarchy.
  const TensorMeshHierarchy<N, Real> &hierarchy;

  //! Smoothness parameter.
  const Real s;

  //! Global quantization error tolerance.
  const Real tolerance;

private:
  //! Nodes of the finest mesh in the hierarchy.
  const TensorNodeRange<N, Real> nodes;

  //! Dequantizer to use when controlling error in the supremum norm.
  const LinearDequantizer<Int, Real> supremum_dequantizer;
};

//! Equality comparison.
template <std::size_t N, typename Int, typename Real>
bool operator==(const TensorMultilevelCoefficientDequantizer<N, Int, Real> &a,
                const TensorMultilevelCoefficientDequantizer<N, Int, Real> &b);

//! Inequality comparison.
template <std::size_t N, typename Int, typename Real>
bool operator!=(const TensorMultilevelCoefficientDequantizer<N, Int, Real> &a,
                const TensorMultilevelCoefficientDequantizer<N, Int, Real> &b);

//! Iterator used to traverse a range of quantized multilevel coefficients,
//! dequantizing as it is dereferenced.
template <std::size_t N, typename Int, typename Real>
template <typename It>
class TensorMultilevelCoefficientDequantizer<N, Int, Real>::iterator {
public:
  using iterator_category = std::input_iterator_tag;
  using value_type = Real;
  using difference_type = std::ptrdiff_t;
  using pointer = value_type *;
  using reference = value_type;

  //! Constructor.
  //!
  //!\param dequantizer Associated multilevel coefficient dequantizer.
  //!\param inner_node Position in the node range.
  //!\param inner_coeff Position in the quantized multilevel coefficient range.
  iterator(const TensorMultilevelCoefficientDequantizer &dequantizer,
           const typename TensorNodeRange<N, Real>::iterator inner_node,
           const It inner_coeff);

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
  //! Associated multilevel coefficient dequantizer;
  const TensorMultilevelCoefficientDequantizer &dequantizer;

  //! Iterator to current node.
  typename TensorNodeRange<N, Real>::iterator inner_node;

  //! Iterator to current quantized coefficient.
  It inner_coeff;
};

} // namespace mgard

#include "TensorMultilevelCoefficientQuantizer.tpp"
#endif
