/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#include <iostream>

#include "../../Hierarchy/Hierarchy.h"
#include "../../RuntimeX/RuntimeX.h"
#include "Autocorrelation8x8x8.hpp"
#include "DataRefactoring.h"
#include "Decompose8.hpp"
#include "Decompose8x8.hpp"
#include "Decompose8x8x8.hpp"
#include "Recompose8.hpp"
#include "Recompose8x8.hpp"
#include "Recompose8x8x8.hpp"

#include <iostream>

#ifndef MGARD_X_160_IN_CACHE_BLOCK_DATA_REFACTORING_HPP
#define MGARD_X_160_IN_CACHE_BLOCK_DATA_REFACTORING_HPP

namespace mgard_x_160 {

namespace data_refactoring {

namespace in_cache_block {

template <DIM D, typename T, typename DeviceType>
void decompose(SubArray<D, T, DeviceType> v, SubArray<D, T, DeviceType> coarse,
               SubArray<1, T, DeviceType> coeff, bool orthogonal_projection,
               int queue_idx) {
  // One kernel per dimensionality: the block geometry (8 fine nodes to 5
  // coarse ones per dimension) is shared, but the coefficient layout and the
  // number of transform passes are not.
  if constexpr (D == 1) {
    DeviceLauncher<DeviceType>::Execute(
        Decompose8Kernel<D, T, DeviceType>(v, coarse, coeff,
                                           orthogonal_projection),
        queue_idx);
  } else if constexpr (D == 2) {
    DeviceLauncher<DeviceType>::Execute(
        Decompose8x8Kernel<D, T, DeviceType>(v, coarse, coeff,
                                             orthogonal_projection),
        queue_idx);
  } else if constexpr (D == 3) {
    DeviceLauncher<DeviceType>::Execute(
        Decompose8x8x8Kernel<D, T, DeviceType>(v, coarse, coeff,
                                               orthogonal_projection),
        queue_idx);
  }
}

template <DIM D, typename T, typename Q, typename DeviceType>
void decompose_quantize(SubArray<D, T, DeviceType> v,
                        SubArray<D, T, DeviceType> coarse,
                        SubArray<1, Q, DeviceType> quantized_coeff, T quantizer,
                        SubArray<1, T, DeviceType> block_quantizers,
                        bool use_block_quantizers, bool prep_huffman,
                        SIZE dict_size, bool orthogonal_projection,
                        int queue_idx) {
  if constexpr (D == 1) {
    DeviceLauncher<DeviceType>::Execute(
        DecomposeQuantize8Kernel<D, T, Q, DeviceType>(
            v, coarse, quantized_coeff, quantizer, block_quantizers,
            use_block_quantizers, prep_huffman, dict_size,
            orthogonal_projection),
        queue_idx);
  } else if constexpr (D == 2) {
    DeviceLauncher<DeviceType>::Execute(
        DecomposeQuantize8x8Kernel<D, T, Q, DeviceType>(
            v, coarse, quantized_coeff, quantizer, block_quantizers,
            use_block_quantizers, prep_huffman, dict_size,
            orthogonal_projection),
        queue_idx);
  } else if constexpr (D == 3) {
    DeviceLauncher<DeviceType>::Execute(
        DecomposeQuantize8x8x8Kernel<D, T, Q, DeviceType>(
            v, coarse, quantized_coeff, quantizer, block_quantizers,
            use_block_quantizers, prep_huffman, dict_size,
            orthogonal_projection),
        queue_idx);
  }
}

template <DIM D, typename T, typename DeviceType>
void recompose(SubArray<D, T, DeviceType> v, SubArray<D, T, DeviceType> coarse,
               SubArray<1, T, DeviceType> coeff, bool orthogonal_projection,
               int queue_idx) {
  if constexpr (D == 1) {
    DeviceLauncher<DeviceType>::Execute(
        Recompose8Kernel<D, T, DeviceType>(v, coarse, coeff,
                                           orthogonal_projection),
        queue_idx);
  } else if constexpr (D == 2) {
    DeviceLauncher<DeviceType>::Execute(
        Recompose8x8Kernel<D, T, DeviceType>(v, coarse, coeff,
                                             orthogonal_projection),
        queue_idx);
  } else if constexpr (D == 3) {
    DeviceLauncher<DeviceType>::Execute(
        Recompose8x8x8Kernel<D, T, DeviceType>(v, coarse, coeff,
                                               orthogonal_projection),
        queue_idx);
  }
}

template <DIM D, typename T, typename Q, typename DeviceType>
void recompose_dequantize(SubArray<D, T, DeviceType> v,
                          SubArray<D, T, DeviceType> coarse,
                          SubArray<1, Q, DeviceType> quantized_coeff,
                          T quantizer,
                          SubArray<1, T, DeviceType> block_quantizers,
                          bool use_block_quantizers, bool prep_huffman,
                          SIZE dict_size, bool orthogonal_projection,
                          int queue_idx) {
  if constexpr (D == 1) {
    DeviceLauncher<DeviceType>::Execute(
        RecomposeDequantize8Kernel<D, T, Q, DeviceType>(
            v, coarse, quantized_coeff, quantizer, block_quantizers,
            use_block_quantizers, prep_huffman, dict_size,
            orthogonal_projection),
        queue_idx);
  } else if constexpr (D == 2) {
    DeviceLauncher<DeviceType>::Execute(
        RecomposeDequantize8x8Kernel<D, T, Q, DeviceType>(
            v, coarse, quantized_coeff, quantizer, block_quantizers,
            use_block_quantizers, prep_huffman, dict_size,
            orthogonal_projection),
        queue_idx);
  } else if constexpr (D == 3) {
    DeviceLauncher<DeviceType>::Execute(
        RecomposeDequantize8x8x8Kernel<D, T, Q, DeviceType>(
            v, coarse, quantized_coeff, quantizer, block_quantizers,
            use_block_quantizers, prep_huffman, dict_size,
            orthogonal_projection),
        queue_idx);
  }
}

} // namespace in_cache_block

} // namespace data_refactoring

} // namespace mgard_x_160

#endif
