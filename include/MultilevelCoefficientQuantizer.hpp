#ifndef MULTILEVELCOEFFICIENTQUANTIZER_HPP
#define MULTILEVELCOEFFICIENTQUANTIZER_HPP
//!\file
//!\brief Quantizer for multilevel coefficients.

#include <iterator>

#include "MeshHierarchy.hpp"
#include "data.hpp"

namespace mgard {

// Forward declarations.
template <typename Real, typename Int> class MultilevelCoefficientQuantizer;

template <typename Real, typename Int>
class MultilevelCoefficientQuantizedRange;

template <typename Real, typename Int>
class MultilevelCoefficientDequantizedRange;

template <typename Real, typename Int>
bool operator==(const MultilevelCoefficientQuantizer<Real, Int> &a,
                const MultilevelCoefficientQuantizer<Real, Int> &b);

//! Quantizer for multilevel coefficients. Each coefficient is quantized
//! according to its contribution to the error indicator.
template <typename Real, typename Int> class MultilevelCoefficientQuantizer {
public:
  //! Constructor.
  //!
  //!\param hierarchy Mesh hierarchy on which the coefficients to be quantized
  //! will be defined.
  //!\param s Smoothness parameter. Determines the error norm in which
  //! quantization error is controlled.
  //!\param tolerance Quantization error tolerance.
  MultilevelCoefficientQuantizer(const MeshHierarchy &hierarchy, const float s,
                                 const float tolerance);

  //! Equality comparison.
  friend bool operator==<>(const MultilevelCoefficientQuantizer &a,
                           const MultilevelCoefficientQuantizer &b);

  //! Quantize a set of multilevel coefficients.
  //!
  //!\param begin Beginning of range to quantize.
  //!\param end End of range to quantize.
  MultilevelCoefficientQuantizedRange<Real, Int> quantize(const It begin,
                                                          const It end) const;

  //!\overload
  //!
  //!\param u Multilevel coefficients to be quantized.
  MultilevelCoefficientQuantizedRange<Real, Int>
  quantize(const MultilevelCoefficients<Real> u) const;

  //! Dequantize a set of quantized multilevel coefficients.
  //!
  //!\param It Beginning of quantized range.
  //!\param It End of quantized range.
  template <typename It>
  MultilevelCoefficientDequantizedRange<Real, Int>
  dequantize(const It begin, const It end) const;

private:
  //! Associated mesh hierarchy.
  const MeshHierarchy &hierarchy;

  //! Smoothness parameter.
  const float s;

  //! Quantization error tolerance.
  const float tolerance;
};

template <typename Real, typename Int>
bool operator!=(const MultilevelCoefficientQuantizedRange<Real, Int> &a,
                const MultilevelCoefficientQuantizedRange<Real, Int> &b);

} // namespace mgard

#include "MultilevelCoefficientQuantizer/MultilevelCoefficientQuantizer.tpp"
#endif
