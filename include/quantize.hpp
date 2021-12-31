#ifndef QUANTIZE_HPP
#define QUANTIZE_HPP
//!\file
//!\brief Quantize multilevel coefficient arrays for self-describing buffers.

#include "proto/mgard.pb.h"

#include "TensorMeshHierarchy.hpp"
#include "compress.hpp"

namespace mgard {

//! Quantize an array of multilevel coefficients.
//!
//! The buffer pointed to by `quantized` must be large enough for the quantized
//! coefficients, and `quantized` must have the correct alignment for an object
//! of the quantization type.
//!
//!\param[in] hierarchy Hierarchy on which the coefficients are defined.
//!\param[in] header Header for the self-describing buffer.
//!\param[in] s Smoothness parameter. Determines the error norm in which
//! quantization error is controlled.
//!\param[in] tolerance Quantization error tolerance for the entire set of
//! multilevel coefficients.
//!\param[in] coefficients Buffer of multilevel coefficients.
//!\param[out] quantized Buffer of quantized multilevel coefficients.
template <std::size_t N, typename Real>
void quantize(const TensorMeshHierarchy<N, Real> &hierarchy,
              const pb::Header &header, const Real s, const Real tolerance,
              Real const *const coefficients, void *const quantized);

//! Dequantize an array of quantized multilevel coefficients.
//!
//! `quantized` must have the correct alignment for an object of the
//! quantization type.
//!
//!\param[in] compressed Compressed dataset of the self-describing buffer.
//!\param[in] quantized Buffer of quantized multilevel coefficients.
//!\param[out] dequantized Buffer of dequantized multilevel coefficients.
template <std::size_t N, typename Real>
void dequantize(const CompressedDataset<N, Real> &compressed,
                void const *const quantized, Real *const dequantized);

} // namespace mgard

#include "quantize.tpp"
#endif
