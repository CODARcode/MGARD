/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_160_IN_CACHE_DATA_REFACTORING
#define MGARD_X_160_IN_CACHE_DATA_REFACTORING

// #include "Common.h"
#include "../../Hierarchy/Hierarchy.h"
#include "../../RuntimeX/RuntimeXPublic.h"

namespace mgard_x_160 {

namespace data_refactoring {

namespace in_cache_block {

// Block-local (in-cache) refactoring for D = 1, 2 and 3. Every dimension of a
// block holds 8 fine nodes that coarsen to 5, so one block emits 5^D coarse
// values and 8^D - 5^D coefficients. Higher D is a no-op: the hybrid
// hierarchy has no block-local stage there.

template <DIM D, typename T, typename DeviceType>
void decompose(SubArray<D, T, DeviceType> v, SubArray<D, T, DeviceType> coarse,
               SubArray<1, T, DeviceType> coeff, bool orthogonal_projection,
               int queue_idx);

// Fused decompose+quantize: same block decomposition as decompose(), but the
// coefficients are quantized in-kernel and written as Q symbols. Reads v with
// bounds checks, so v does not need to be padded to a multiple of 8. Uses the
// per-block quantizers (indexed by the block's row-major index) when
// use_block_quantizers is set (ROI mode), the scalar quantizer otherwise.
template <DIM D, typename T, typename Q, typename DeviceType>
void decompose_quantize(SubArray<D, T, DeviceType> v,
                        SubArray<D, T, DeviceType> coarse,
                        SubArray<1, Q, DeviceType> quantized_coeff, T quantizer,
                        SubArray<1, T, DeviceType> block_quantizers,
                        bool use_block_quantizers, bool prep_huffman,
                        SIZE dict_size, bool orthogonal_projection,
                        int queue_idx);

template <DIM D, typename T, typename DeviceType>
void recompose(SubArray<D, T, DeviceType> v, SubArray<D, T, DeviceType> coarse,
               SubArray<1, T, DeviceType> coeff, bool orthogonal_projection,
               int queue_idx);

// Fused dequantize+recompose: same block recomposition as recompose(), but the
// coefficients are read as Q symbols and dequantized in-kernel. Writes v with
// bounds checks, so v does not need to be padded to a multiple of 8. Uses the
// per-block quantizers (indexed by the block's row-major index) when
// use_block_quantizers is set (ROI mode), the scalar quantizer otherwise.
template <DIM D, typename T, typename Q, typename DeviceType>
void recompose_dequantize(SubArray<D, T, DeviceType> v,
                          SubArray<D, T, DeviceType> coarse,
                          SubArray<1, Q, DeviceType> quantized_coeff,
                          T quantizer,
                          SubArray<1, T, DeviceType> block_quantizers,
                          bool use_block_quantizers, bool prep_huffman,
                          SIZE dict_size, bool orthogonal_projection,
                          int queue_idx);

} // namespace in_cache_block

} // namespace data_refactoring

} // namespace mgard_x_160

#endif
