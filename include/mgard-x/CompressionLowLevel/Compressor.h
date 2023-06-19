/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_COMPRESSOR_H
#define MGARD_X_COMPRESSOR_H

#include "../RuntimeX/RuntimeXPublic.h"

#include "../DataRefactoring/DataRefactor.hpp"

// #include "CompressionLowLevelWorkspace.hpp"

#include "NormCalculator.hpp"

#include "../Hierarchy/Hierarchy.h"

#include "../Lossless/Lossless.hpp"
#include "../Quantization/LinearQuantization.hpp"

#include "LossyCompressorInterface.hpp"

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
class Compressor : public LossyCompressorInterface<D, T, DeviceType> {
public:
  using HierarchyType = Hierarchy<D, T, DeviceType>;
  using DataRefactorType = data_refactoring::DataRefactor<D, T, DeviceType>;
  using LosslessCompressorType =
      ComposedLosslessCompressor<QUANTIZED_UNSIGNED_INT, HUFFMAN_CODE,
                                 DeviceType>;
  using LinearQuantizerType = LinearQuantizer<D, T, QUANTIZED_INT, DeviceType>;

public:
  Compressor();

  Compressor(Hierarchy<D, T, DeviceType> &hierarchy, Config config);

  void Adapt(Hierarchy<D, T, DeviceType> &hierarchy, Config config,
             int queue_idx);

  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape, Config config);

  void CalculateNorm(Array<D, T, DeviceType> &original_data,
                     enum error_bound_type ebtype, T s, T &norm, int queue_idx);

  void Decompose(Array<D, T, DeviceType> &original_data, int queue_idx);

  void Quantize(Array<D, T, DeviceType> &original_data,
                enum error_bound_type ebtype, T tol, T s, T norm,
                int queue_idx);

  void LosslessCompress(Array<1, Byte, DeviceType> &compressed_data,
                        int queue_idx);

  void Recompose(Array<D, T, DeviceType> &decompressed_data, int queue_idx);

  void Dequantize(Array<D, T, DeviceType> &decompressed_data,
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
  Array<1, T, DeviceType> norm_tmp_array;
  Array<1, T, DeviceType> norm_array;
  Array<D, QUANTIZED_INT, DeviceType> quantized_array;
  DataRefactorType refactor;
  LinearQuantizerType quantizer;
  LosslessCompressorType lossless_compressor;
};

} // namespace mgard_x

#endif