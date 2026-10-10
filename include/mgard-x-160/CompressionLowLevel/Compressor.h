/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_160_COMPRESSOR_H
#define MGARD_X_160_COMPRESSOR_H

#include <limits>

#include "../RuntimeX/RuntimeXPublic.h"

#include "../DataRefactoring/DataRefactor.hpp"

// #include "CompressionLowLevelWorkspace.hpp"

#include "NormCalculator.hpp"

#include "../Hierarchy/Hierarchy.h"

#include "../Lossless/Lossless.hpp"
#include "../Quantization/LinearQuantization.hpp"
#include "../Utilities/ProjectionMode.h"

#include "LossyCompressorInterface.hpp"

namespace mgard_x_160 {

// D-aware counterpart of infer_orthogonal_projection in
// Utilities/ProjectionMode.h, kept so the shared call sites compile unchanged.
template <DIM D, typename T>
inline bool infer_orthogonal_projection(compression_projection_mode_type mode,
                                        T s) {
  // MGARD 1.6.0 data: always the orthogonal basis (see Utilities/
  // ProjectionMode.h).
  return true;
}

template <DIM D, typename T, typename DeviceType>
class Compressor : public LossyCompressorInterface<D, T, DeviceType> {
public:
  using HierarchyType = Hierarchy<D, T, DeviceType>;
  using DataRefactorType = data_refactoring::DataRefactor<D, T, DeviceType>;
  using LosslessCompressorType =
      ComposedLosslessCompressor<QUANTIZED_INT, HUFFMAN_CODE, DeviceType>;
  using LinearQuantizerType = LinearQuantizer<D, T, QUANTIZED_INT, DeviceType>;

public:
  Compressor();

  Compressor(Hierarchy<D, T, DeviceType> &hierarchy, Config config);

  void Adapt(Hierarchy<D, T, DeviceType> &hierarchy, Config config,
             int queue_idx);

  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape, Config config);

  void CalculateNorm(Array<D, T, DeviceType> &original_data,
                     enum error_bound_type ebtype, T s, T &norm, int queue_idx);

  void Decompose(Array<D, T, DeviceType> &original_data,
                 bool orthogonal_projection, int queue_idx);

  void Quantize(Array<D, T, DeviceType> &original_data,
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

  // Dequantize + recompose as one step of the decompression pipelines. Here it
  // is simply the two calls in sequence; a compressor that can do better (see
  // HybridHierarchyCompressor, which fuses them into one pass over the local
  // levels) overrides this and decides for itself. The pipelines call this
  // rather than the two methods so they do not have to know which compressor
  // they are driving.
  void DequantizeRecompose(Array<D, T, DeviceType> &decompressed_data,
                           enum error_bound_type ebtype, T tol, T s, T norm,
                           int queue_idx);

  void LosslessDecompress(Array<1, Byte, DeviceType> &compressed_data,
                          int queue_idx);

  void Compress(Array<D, T, DeviceType> &original_data,
                enum error_bound_type ebtype, T tol, T s, T &norm,
                Array<1, Byte, DeviceType> &compressed_data, int queue_idx);
  void Decompress(Array<1, Byte, DeviceType> &compressed_data,
                  enum error_bound_type ebtype, T tol, T s, T &norm,
                  Array<D, T, DeviceType> &decompressed_data, int queue_idx);

  bool initialized;
  Hierarchy<D, T, DeviceType> *hierarchy;
  Config config;
  // Whether the last (de)compose should use orthogonal projection. Derived
  // from config.projection_mode and s (see infer_orthogonal_projection)
  // during Compress/Decompress/(De)quantize and consumed by Recompose, which
  // does not receive s. Defaults to true so the orthogonal path is used
  // unless the config/s combination resolves to the hierarchical fast path.
  bool orthogonal_projection = true;
  Array<1, T, DeviceType> norm_tmp_array;
  Array<1, T, DeviceType> norm_array;
  Array<D, QUANTIZED_INT, DeviceType> quantized_array;
  DataRefactorType refactor;
  LinearQuantizerType quantizer;
  LosslessCompressorType lossless_compressor;
};

} // namespace mgard_x_160

#endif