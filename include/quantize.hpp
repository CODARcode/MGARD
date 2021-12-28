#ifndef QUANTIZE_HPP
#define QUANTIZE_HPP
//!\file Quantize multilevel coefficient arrays for self-describing buffers.

#include "TensorMeshHierarchy.hpp"
#include "compress.hpp"

#ifdef MGARD_PROTOBUF
#include "proto/mgard.pb.h"
#endif

namespace mgard {

#ifdef MGARD_PROTOBUF
//! Quantize an array of multilevel coefficients.
//!
//! The buffer pointed to by `quantized` must be large enough for the quantized
//! coefficients, and `quantized` must have the correct alignment for an object
//! of the quantization type.
//!
//! The relevant fields of `header` will be populated.
//!
//!\param[in] hierarchy Hierarchy on which the coefficients are defined.
//!\param[in] s Smoothness parameter. Determines the error norm in which
//! quantization error is controlled.
//!\param[in] tolerance Quantization error tolerance for the entire set of
//! multilevel coefficients.
//!\param[in] coefficients Buffer of multilevel coefficients.
//!\param[out] quantized Buffer of quantized multilevel coefficients.
//!\param[in, out] header Header for the self-describing buffer.
template <std::size_t N, typename Real>
void quantize(const TensorMeshHierarchy<N, Real> &hierarchy, const Real s,
              const Real tolerance, Real const *const coefficients,
              void *const quantized, const pb::Header &header);

//! Dequantize an array of quantized multilevel coefficients.
//!
//! `quantized` must have the correct alignment for an object of the
//! quantization type.
//!
//!\param[in] compressed Compressed dataset of the self-describing buffer.
//!\param[in] quantized Buffer of quantized multilevel coefficients.
//!\param[out] dequantized Buffer of dequantized multilevel coefficients.
//!\param[in] header Header of the self-describing buffer.
template <std::size_t N, typename Real>
void dequantize(const CompressedDataset<N, Real> &compressed,
                void const *const quantized, Real *const dequantized,
                const pb::Header &header);
#endif

} // namespace mgard

#include "quantize.tpp"
#endif
