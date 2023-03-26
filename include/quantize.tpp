#include <algorithm>
#include <memory>
#include <stdexcept>

#include "TensorMultilevelCoefficientQuantizer.hpp"
#include "format.hpp"
#include "utilities.hpp"

namespace mgard {

namespace {

// This function checks that the endianness of `Int` matches the endianness
// requested/expected by the header. Ultimately we should shuffle bytes rather
// than throwing an exception if there's a mismatch.
template <typename Int> void check_endianness(const pb::Header &header) {
  const QuantizationParameters quantization = read_quantization(header);
  if (big_endian<Int>() != quantization.big_endian) {
    throw std::runtime_error("quantization type endianness mismatch");
  }
}

// This function has the same signature as `quantize`, so it's named
// differently.

//! Quantize an array of multilevel coefficients.
//!
//!\param[in] hierarchy Hierarchy on which the coefficients are defined.
//!\param[in] header Header for the self-describing buffer.
//!\param[in] s Smoothness parameter. Determines the error norm in which
//! quantization error is controlled.
//!\param[in] tolerance Quantization error tolerance for the entire set of
//! multilevel coefficients.
//!\param[in] coefficients Buffer of multilevel coefficients.
//!\param[out] quantized Buffer of quantized multilevel coefficients.
template <std::size_t N, typename Real, typename Int>
void quantize_(const TensorMeshHierarchy<N, Real> &hierarchy,
               const pb::Header &header, const Real s, const Real tolerance,
               Real const *const coefficients, Int *const quantized) {
  check_alignment<Int>(quantized);
  check_endianness<Int>(header);

  using Qntzr = TensorMultilevelCoefficientQuantizer<N, Real, Int>;
  const Qntzr quantizer(hierarchy, s, tolerance);
  using It = typename Qntzr::iterator;
  const RangeSlice<It> quantized_ = quantizer(coefficients);
  std::copy(quantized_.begin(), quantized_.end(), quantized);
}

// QG roi-adaptive
template <std::size_t N, typename Real, typename Int>
void quantize_roi_(const TensorMeshHierarchy<N, Real> &hierarchy,
                   const pb::Header &header, const Real s, const Real tolerance,
                   const size_t scalar, const Real *cmap,
                   Real const *const coefficients, Int *const quantized) {
  check_alignment<Int>(quantized);
  check_endianness<Int>(header);

  using QntzrAdp = TensorMultilevelCoefficientAdpQuantizer<N, Real, Int>;
  const QntzrAdp quantizer(hierarchy, s, tolerance, scalar);
  using It = typename QntzrAdp::iterator;
  const RangeSlice<It> quantized_ = quantizer(coefficients, cmap);
  std::copy(quantized_.begin(), quantized_.end(), quantized);
}

// This function has the same signature as `dequantize`, so it's named
// differently.

//! Dequantize an array of quantized multilevel coefficients.
//!
//!\param[in] compressed Compressed dataset of the self-describing buffer.
//!\param[in] quantized Buffer of quantized multilevel coefficients.
//!\param[out] dequantized Buffer of dequantized multilevel coefficients.
template <std::size_t N, typename Int, typename Real>
void dequantize_(const CompressedDataset<N, Real> &compressed,
                 Int const *const quantized, Real *const dequantized) {
  check_alignment<Int>(quantized);
  check_endianness<Int>(compressed.header);

  const std::size_t ndof = compressed.hierarchy.ndof();

  using Dqntzr = TensorMultilevelCoefficientDequantizer<N, Int, Real>;
  const Dqntzr dequantizer(compressed.hierarchy, compressed.s,
                           compressed.tolerance);
  using It = typename Dqntzr::template iterator<Int const *>;
  const RangeSlice<It> dequantized_ = dequantizer(quantized, quantized + ndof);
  std::copy(dequantized_.begin(), dequantized_.end(), dequantized);
}

} // namespace

template <std::size_t N, typename Real>
void quantize(const TensorMeshHierarchy<N, Real> &hierarchy,
              const pb::Header &header, const Real s, const Real tolerance,
              Real const *const coefficients, void *const quantized) {
  const QuantizationParameters quantization = read_quantization(header);
  switch (quantization.type) {
  case pb::Quantization::INT8_T:
    return quantize_(hierarchy, header, s, tolerance, coefficients,
                     static_cast<std::int8_t *>(quantized));
  case pb::Quantization::INT16_T:
    return quantize_(hierarchy, header, s, tolerance, coefficients,
                     static_cast<std::int16_t *>(quantized));
  case pb::Quantization::INT32_T:
    return quantize_(hierarchy, header, s, tolerance, coefficients,
                     static_cast<std::int32_t *>(quantized));
  case pb::Quantization::INT64_T:
    return quantize_(hierarchy, header, s, tolerance, coefficients,
                     static_cast<std::int64_t *>(quantized));
  default:
    throw std::runtime_error("unrecognized quantization type");
  }
}

// QG: roi-adaptive
template <std::size_t N, typename Real>
void quantize_roi(const TensorMeshHierarchy<N, Real> &hierarchy,
                  const pb::Header &header, const Real s, const Real tolerance,
                  const size_t scalar, const Real *cmap,
                  Real const *const coefficients, void *const quantized) {
  const QuantizationParameters quantization = read_quantization(header);
  switch (quantization.type) {
  case pb::Quantization::INT8_T:
    return quantize_roi_(hierarchy, header, s, tolerance, scalar, cmap,
                         coefficients, static_cast<std::int8_t *>(quantized));
  case pb::Quantization::INT16_T:
    return quantize_roi_(hierarchy, header, s, tolerance, scalar, cmap,
                         coefficients, static_cast<std::int16_t *>(quantized));
  case pb::Quantization::INT32_T:
    return quantize_roi_(hierarchy, header, s, tolerance, scalar, cmap,
                         coefficients, static_cast<std::int32_t *>(quantized));
  case pb::Quantization::INT64_T:
    return quantize_roi_(hierarchy, header, s, tolerance, scalar, cmap,
                         coefficients, static_cast<std::int64_t *>(quantized));
  default:
    throw std::runtime_error("unrecognized quantization type");
  }
}

template <std::size_t N, typename Real>
void dequantize(const CompressedDataset<N, Real> &compressed,
                void const *const quantized, Real *const dequantized) {
  const QuantizationParameters quantization =
      read_quantization(compressed.header);
  switch (quantization.type) {
  case pb::Quantization::INT8_T:
    return dequantize_(compressed, static_cast<std::int8_t const *>(quantized),
                       dequantized);
  case pb::Quantization::INT16_T:
    return dequantize_(compressed, static_cast<std::int16_t const *>(quantized),
                       dequantized);
  case pb::Quantization::INT32_T:
    return dequantize_(compressed, static_cast<std::int32_t const *>(quantized),
                       dequantized);
  case pb::Quantization::INT64_T:
    return dequantize_(compressed, static_cast<std::int64_t const *>(quantized),
                       dequantized);
  default:
    throw std::runtime_error("unrecognized quantization type");
  }
}

} // namespace mgard
