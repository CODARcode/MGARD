#ifndef MULTILEVELCOEFFICIENTQUANTIZER_HPP
#define MULTILEVELCOEFFICIENTQUANTIZER_HPP
//!\file
//!\brief Quantizer for multilevel coefficients.

#include <iterator>

#include "MeshHierarchy.hpp"
#include "data.hpp"

namespace mgard {

// Forward declarations.
template <typename Int> class MultilevelCoefficientQuantizer;

template <typename Int> class MultilevelCoefficientQuantizedRange;

template <typename Int> class MultilevelCoefficientDequantizedRange;

template <typename Int>
bool operator==(const MultilevelCoefficientQuantizer<Int> &a,
                const MultilevelCoefficientQuantizer<Int> &b);

//! Quantizer for multilevel coefficients. Each coefficient is quantized
//! according to its contribution to the error indicator.
template <typename Int> class MultilevelCoefficientQuantizer {
public:
  //! Constructor.
  //!
  //!\param hierarchy Mesh hierarchy on which the coefficients to be quantized
  //! will be defined.
  //!\param s Smoothness parameter. Determines the error norm in which
  //! quantization error is controlled.
  //!\param tolerance Quantization error tolerance.
  MultilevelCoefficientQuantizer(const MeshHierarchy &hierarchy, const double s,
                                 const double tolerance);

  //! Equality comparison.
  friend bool operator==<>(const MultilevelCoefficientQuantizer &a,
                           const MultilevelCoefficientQuantizer &b);

  //! Quantize a set of multilevel coefficients.
  //!
  //!\param begin Beginning of range to quantize.
  //!\param end End of range to quantize.
  MultilevelCoefficientQuantizedRange<Int> quantize(const It begin,
                                                    const It end) const;

  //!\overload
  //!
  //!\param u Multilevel coefficients to be quantized.
  MultilevelCoefficientQuantizedRange<Int>
  quantize(const MultilevelCoefficients<double> u) const;

  //! Dequantize a set of quantized multilevel coefficients.
  //!
  //!\param It Beginning of quantized range.
  //!\param It End of quantized range.
  template <typename It>
  MultilevelCoefficientDequantizedRange<Int> dequantize(const It begin,
                                                        const It end) const;

private:
  //! Associated mesh hierarchy.
  const MeshHierarchy &hierarchy;

  //! Smoothness parameter.
  const double s;

  //! Quantization error tolerance.
  const double tolerance;
};

template <typename Int>
bool operator!=(const MultilevelCoefficientQuantizedRange<Int> &a,
                const MultilevelCoefficientQuantizedRange<Int> &b);

} // namespace mgard

#include "MultilevelCoefficientQuantizer/MultilevelCoefficientQuantizer.tpp"
#endif
