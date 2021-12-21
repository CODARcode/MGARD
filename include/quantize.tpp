#include <algorithm>
#include <memory>
#include <stdexcept>

#include <iostream>

#include "TensorMultilevelCoefficientQuantizer.hpp"
#include "utilities.hpp"

namespace mgard {

namespace {

template <typename Int> void check_alignment(void const *const p) {
  if (p == nullptr) {
    throw std::invalid_argument("can't check alignment of null pointer");
  }
  void *q = const_cast<void *>(p);
  const std::size_t size = sizeof(Int);
  std::size_t space = 2 * size;
  if (p != std::align(alignof(Int), size, q, space)) {
    throw std::invalid_argument("pointer misaligned");
  }
}

// This function checks that the endianness of `Int` matches the endianness
// requested/expected by the header. Ultimately we should shuffle bytes rather
// than throwing an exception if there's a mismatch.
template <typename Int> void check_endianness(const pb::Header &header) {
  const QuantizationParameters quantization = read_quantization(header);
  const Int n = 1;
  if ((*reinterpret_cast<unsigned char const *>(&n) == 1) ==
      quantization.big_endian) {
    throw std::runtime_error("quantization type endianness mismatch");
  }
}

//! Quantize an array of multilevel coefficients.
//!
//!\param[in] hierarchy Hierarchy on which the coefficients are defined.
//!\param[in] s Smoothness parameter. Determines the error norm in which
//! quantization error is controlled.
//!\param[in] tolerance Quantization error tolerance for the entire set of
//! multilevel coefficients.
//!\param[in] coefficients Buffer of multilevel coefficients.
//!\param[out] quantized Buffer of quantized multilevel coefficients.
//!\param[in] header Header for the self-describing buffer.
template <std::size_t N, typename Real, typename Int>
void quantize(const TensorMeshHierarchy<N, Real> &hierarchy, const Real s,
              const Real tolerance, Real const *const coefficients,
              Int *const quantized, const pb::Header &header) {
  check_alignment<Int>(static_cast<void const *>(quantized));
  check_endianness<Int>(header);

  using Qntzr = TensorMultilevelCoefficientQuantizer<N, Real, Int>;
  const Qntzr quantizer(hierarchy, s, tolerance);
  using It = typename Qntzr::iterator;
  const RangeSlice<It> quantized_ = quantizer(coefficients);
  std::copy(quantized_.begin(), quantized_.end(), quantized);
}

//! Dequantize an array of quantized multilevel coefficients.
//!
//!\param[in] compressed Compressed dataset of the self-describing buffer.
//!\param[in] quantized Buffer of quantized multilevel coefficients.
//!\param[out] dequantized Buffer of dequantized multilevel coefficients.
//!\param[in] header Header of the self-describing buffer.
template <std::size_t N, typename Int, typename Real>
void dequantize(const CompressedDataset<N, Real> &compressed,
                Int const *const quantized, Real *const dequantized,
                const pb::Header &header) {
  check_alignment<Int>(static_cast<void const *>(quantized));
  check_endianness<Int>(header);

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
void quantize(const TensorMeshHierarchy<N, Real> &hierarchy, const Real s,
              const Real tolerance, Real const *const coefficients,
              void *const quantized, const pb::Header &header) {
  const QuantizationParameters quantization = read_quantization(header);
  switch (quantization.type) {
  case pb::Quantization::INT8_T:
    return quantize(hierarchy, s, tolerance, coefficients,
                    static_cast<std::int8_t *>(quantized), header);
  case pb::Quantization::INT16_T:
    return quantize(hierarchy, s, tolerance, coefficients,
                    static_cast<std::int16_t *>(quantized), header);
  case pb::Quantization::INT32_T:
    return quantize(hierarchy, s, tolerance, coefficients,
                    static_cast<std::int32_t *>(quantized), header);
  case pb::Quantization::INT64_T:
    return quantize(hierarchy, s, tolerance, coefficients,
                    static_cast<std::int64_t *>(quantized), header);
  default:
    throw std::runtime_error("unrecognized quantization type");
  }
}

template <std::size_t N, typename Real>
void dequantize(const CompressedDataset<N, Real> &compressed,
                void const *const quantized, Real *const dequantized,
                const pb::Header &header) {
  const QuantizationParameters quantization = read_quantization(header);
  switch (quantization.type) {
  case pb::Quantization::INT8_T:
    return dequantize(compressed, static_cast<std::int8_t const *>(quantized),
                      dequantized, header);
  case pb::Quantization::INT16_T:
    return dequantize(compressed, static_cast<std::int16_t const *>(quantized),
                      dequantized, header);
  case pb::Quantization::INT32_T:
    return dequantize(compressed, static_cast<std::int32_t const *>(quantized),
                      dequantized, header);
  case pb::Quantization::INT64_T:
    return dequantize(compressed, static_cast<std::int64_t const *>(quantized),
                      dequantized, header);
  default:
    throw std::runtime_error("unrecognized quantization type");
  }
}

} // namespace mgard
