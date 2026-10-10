/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_160_HYBRID_HIERARCHY_COMPRESSOR_H
#define MGARD_X_160_HYBRID_HIERARCHY_COMPRESSOR_H

#include "../DataRefactoring/BlockLocalHierarchyDataRefactor.hpp"
#include "../DataRefactoring/HybridHierarchyDataRefactor.hpp"
#include "../RuntimeX/RuntimeXPublic.h"

// #include "CompressionLowLevelWorkspace.hpp"

#include "../Hierarchy/HybridHierarchy.h"
#include "../Lossless/Lossless.hpp"
#include "../Quantization/HybridHierarchyLinearQuantization.hpp"
#include "../Quantization/LocalQuantization.hpp"
#include "../Utilities/ProjectionMode.h"
#include "LossyCompressorInterface.hpp"
#include "NormCalculator.hpp"

namespace mgard_x_160 {

template <DIM D, typename T, typename DeviceType>
class HybridHierarchyCompressor
    : public LossyCompressorInterface<D, T, DeviceType> {
public:
  using HierarchyType = HybridHierarchy<D, T, DeviceType>;
  //   using BlockLocalHierarchyDataRefactorType =
  //       data_refactoring::BlockLocalHierarchyDataRefactor<D, T, DeviceType>;
  using HybridHierarchyDataRefactorType =
      data_refactoring::HybridHierarchyDataRefactor<D, T, DeviceType>;
  using LosslessCompressorType =
      ComposedLosslessCompressor<QUANTIZED_INT, HUFFMAN_CODE, DeviceType>;
  //   using LocalQuantizerType = LocalQuantizer<D, T, QUANTIZED_INT,
  //   DeviceType>;
  using HybridQuantizerType =
      HybridHierarchyQuantizer<D, T, QUANTIZED_INT, DeviceType>;

  HybridHierarchyCompressor();

  HybridHierarchyCompressor(HybridHierarchy<D, T, DeviceType> &hierarchy,
                            Config config);

  void Adapt(HybridHierarchy<D, T, DeviceType> &hierarchy, Config config,
             int queue_idx);

  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape, Config config);

  void CalculateNorm(Array<D, T, DeviceType> &original_data,
                     enum error_bound_type ebtype, T s, T &norm, int queue_idx);

  void Decompose(Array<D, T, DeviceType> &original_data, int queue_idx);

  void Quantize(Array<D, T, DeviceType> &original_data,
                enum error_bound_type ebtype, T tol, T s, T norm,
                int queue_idx);

  // Fused Decompose+Quantize (single pass over the local levels); used by
  // Compress() instead of Decompose()+Quantize() when
  // hybrid_quantizer.CanFuseQuantize(s) holds.
  void DecomposeQuantize(Array<D, T, DeviceType> &original_data,
                         enum error_bound_type ebtype, T tol, T s, T norm,
                         int queue_idx);

  void LosslessCompress(Array<1, Byte, DeviceType> &compressed_data,
                        int queue_idx);

  void Serialize(Array<1, Byte, DeviceType> &compressed_data, int queue_idx);

  void Deserialize(Array<1, Byte, DeviceType> &compressed_data, int queue_idx);

  void Recompose(Array<D, T, DeviceType> &decompressed_data,
                 bool orthogonal_projection, int queue_idx);

  void Dequantize(Array<D, T, DeviceType> &decompressed_data,
                  enum error_bound_type ebtype, T tol, T s, T norm,
                  int queue_idx);

  // Dequantize + recompose. Overrides the plain two-call version: picks the
  // fused single-pass implementation below when the configuration allows it,
  // and falls back to Dequantize() + Recompose() otherwise. This is the only
  // place that choice is made, for both the pipelines and Decompress().
  void DequantizeRecompose(Array<D, T, DeviceType> &decompressed_data,
                           enum error_bound_type ebtype, T tol, T s, T norm,
                           int queue_idx);

  // Fused Dequantize+Recompose: one pass over the local levels, coefficients
  // never round-trip through global memory as T. Requires
  // hybrid_quantizer.CanFuseQuantize(s); throws otherwise. Call
  // DequantizeRecompose() instead unless you specifically want the fused path.
  void DequantizeRecomposeFused(Array<D, T, DeviceType> &decompressed_data,
                                enum error_bound_type ebtype, T tol, T s,
                                T norm, int queue_idx);

  void LosslessDecompress(Array<1, Byte, DeviceType> &compressed_data,
                          int queue_idx);

  void Compress(Array<D, T, DeviceType> &original_data,
                enum error_bound_type ebtype, T tol, T s, T &norm,
                Array<1, Byte, DeviceType> &compressed_data, int queue_idx);
  void Decompress(Array<1, Byte, DeviceType> &compressed_data,
                  enum error_bound_type ebtype, T tol, T s, T &norm,
                  Array<D, T, DeviceType> &decompressed_data, int queue_idx);

  static SIZE calculate_padded_size(std::vector<SIZE> shape, Config config);
  static SIZE
  calculate_padded_size(HybridHierarchy<D, T, DeviceType> &hierarchy,
                        Config config);

  bool initialized;
  HybridHierarchy<D, T, DeviceType> *hierarchy;
  Config config;
  bool orthogonal_projection = true;
  Array<1, T, DeviceType> norm_tmp_array;
  Array<1, T, DeviceType> norm_array;
  //   Array<1, T, DeviceType> local_decomposed_array;
  //   Array<1, QUANTIZED_INT, DeviceType> local_quantized_array;
  Array<1, T, DeviceType> hybrid_decomposed_array;
  Array<1, QUANTIZED_INT, DeviceType> hybrid_quantized_array;
  //   BlockLocalHierarchyDataRefactorType local_refactor;
  //   LocalQuantizerType local_quantizer;
  HybridHierarchyDataRefactorType hybrid_refactor;
  HybridQuantizerType hybrid_quantizer;
  LosslessCompressorType lossless_compressor;
};

} // namespace mgard_x_160

#endif
