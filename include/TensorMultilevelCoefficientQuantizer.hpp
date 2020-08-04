#ifndef TENSORMULTILEVELCOEFFICIENTQUANTIZER_HPP
#define TENSORMULTILEVELCOEFFICIENTQUANTIZER_HPP
//!\file
//!\brief Quantizer for multilevel coefficients on tensor product grids.

#include "LinearQuantizer.hpp"
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
  RangeSlice<iterator> operator()(Real *const u) const;

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
  using reference = value_type &;

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
  value_type operator*() const;

private:
  //! Associated multilevel coefficient quantizer;
  const TensorMultilevelCoefficientQuantizer &quantizer;

  //! Iterator to current node.
  typename TensorNodeRange<N, Real>::iterator inner_node;

  //! Iterator to current coefficient.
  Real const *inner_coeff;
};

} // namespace mgard

#include "TensorMultilevelCoefficientQuantizer.tpp"
#endif
