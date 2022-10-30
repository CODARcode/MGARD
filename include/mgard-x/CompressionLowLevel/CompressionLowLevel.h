/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_COMPRESSION_LOW_LEVEL_H
#define MGARD_X_COMPRESSION_LOW_LEVEL_H

#include "../RuntimeX/RuntimeXPublic.h"

#include "../DataRefactoring/DataRefactoring.hpp"

#include "CompressionLowLevelWorkspace.hpp"

#include "NormCalculator.hpp"

#include "../Hierarchy/Hierarchy.h"

#include "../Lossless/Lossless.hpp"
#include "../Quantization/LinearQuantization.hpp"

#include "LossyCompressorInterface.hpp"

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
void compress(Hierarchy<D, T, DeviceType> &hierarchy,
              Array<D, T, DeviceType> &original_array,
              enum error_bound_type type, T tol, T s, T &norm, Config config,
              CompressionLowLevelWorkspace<D, T, DeviceType> &workspace,
              Array<1, Byte, DeviceType> &compressed_array);

template <DIM D, typename T, typename DeviceType>
void decompress(Hierarchy<D, T, DeviceType> &hierarchy,
                Array<1, unsigned char, DeviceType> &compressed_array,
                enum error_bound_type type, T tol, T s, T norm, Config config,
                CompressionLowLevelWorkspace<D, T, DeviceType> &workspace,
                Array<D, T, DeviceType> &decompressed_array);

template <DIM D, typename T, typename DeviceType>
class Compressor : public LossyCompressorInterface<D, T, DeviceType> {
  using DataRefactorType = DataRefactor<D, T, DeviceType>;
  using LosslessCompressorType =
      ComposedLosslessCompressor<QUANTIZED_UNSIGNED_INT, HUFFMAN_CODE,
                                 DeviceType>;
  using LinearQuantizerType = LinearQuantizer<D, T, QUANTIZED_INT, DeviceType>;

public:
  Compressor(Hierarchy<D, T, DeviceType> hierarchy, Config config)
      : hierarchy(hierarchy), config(config), refactor(hierarchy, config),
        lossless_compressor(hierarchy.total_num_elems(), config),
        quantizer(hierarchy, config) {

    norm_tmp_array = Array<1, T, DeviceType>({hierarchy.total_num_elems()});
    norm_array = Array<1, T, DeviceType>({1});
    quantized_array = Array<D, QUANTIZED_INT, DeviceType>(
        hierarchy.level_shape(hierarchy.l_target()), false, false);
  }

  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape, SIZE l_target,
                                        SIZE dict_size, SIZE chunk_size,
                                        double estimated_outlier_ratio = 1.0) {
    SIZE total_num_elems = 1;
    for (DIM d = 0; d < D; d++)
      total_num_elems *= shape[d];
    size_t size = 0;
    size += sizeof(T) * total_num_elems + 1;
    size += DataRefactorType::EstimateMemoryFootprint(shape);
    size += LinearQuantizerType::EstimateMemoryFootprint(l_target);
    size += LosslessCompressorType::EstimateMemoryFootprint(
        total_num_elems, dict_size, chunk_size, estimated_outlier_ratio);
    return size;
  }

  void CalculateNorm(Array<D, T, DeviceType> &original_data,
                     enum error_bound_type ebtype, T s, T &norm,
                     int queue_idx) {
    if (ebtype == error_bound_type::REL) {
      norm = norm_calculator(original_data, SubArray(norm_tmp_array),
                             SubArray(norm_array), s,
                             config.normalize_coordinates);
    }
  }

  void Decompose(Array<D, T, DeviceType> &original_data, int queue_idx) {
    refactor.Decompose(original_data, queue_idx);
  }

  void Quantize(Array<D, T, DeviceType> &original_data,
                enum error_bound_type ebtype, T tol, T s, T norm,
                int queue_idx) {
    quantizer.Quantize(original_data, ebtype, tol, s, norm, quantized_array,
                       lossless_compressor, queue_idx);
  }

  void LosslessCompress(Array<1, Byte, DeviceType> &compressed_data,
                        int queue_idx) {
    Array<1, QUANTIZED_UNSIGNED_INT, DeviceType> quantized_liearized_array(
        {hierarchy.total_num_elems()},
        (QUANTIZED_UNSIGNED_INT *)quantized_array.data());
    lossless_compressor.Compress(quantized_liearized_array, compressed_data,
                                 queue_idx);
  }

  void Recompose(Array<D, T, DeviceType> &decompressed_data, int queue_idx) {
    refactor.Recompose(decompressed_data, queue_idx);
  }

  void Dequantize(Array<D, T, DeviceType> &decompressed_data,
                  enum error_bound_type ebtype, T tol, T s, T norm,
                  int queue_idx) {
    quantizer.Dequantize(quantized_array, ebtype, tol, s, norm,
                         decompressed_data, lossless_compressor, queue_idx);
  }

  void LosslessDecompress(Array<1, Byte, DeviceType> &compressed_data,
                          int queue_idx) {
    Array<1, QUANTIZED_UNSIGNED_INT, DeviceType> quantized_liearized_data(
        {hierarchy.total_num_elems()},
        (QUANTIZED_UNSIGNED_INT *)quantized_array.data());
    lossless_compressor.Decompress(compressed_data, quantized_liearized_data,
                                   queue_idx);
  }

  void Compress(Array<D, T, DeviceType> &original_data,
                enum error_bound_type ebtype, T tol, T s, T &norm,
                Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {

    config.apply();

    DeviceRuntime<DeviceType>::SelectDevice(config.dev_id);
    log::info("Select device: " + DeviceRuntime<DeviceType>::GetDeviceName());
    Timer timer_total;
    for (int d = D - 1; d >= 0; d--) {
      if (hierarchy.level_shape(hierarchy.l_target(), d) !=
          original_data.shape(d)) {
        log::err("The shape of input array does not match the shape initilized "
                 "in hierarchy!");
        return;
      }
    }

    if (log::level & log::TIME)
      timer_total.start();

    CalculateNorm(original_data, ebtype, s, norm, queue_idx);
    Decompose(original_data, queue_idx);
    Quantize(original_data, ebtype, tol, s, norm, queue_idx);
    LosslessCompress(compressed_data, queue_idx);

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(0);
      timer_total.end();
      timer_total.print("Low-level compression");
      log::time(
          "Low-level compression throughput: " +
          std::to_string((double)(hierarchy.total_num_elems() * sizeof(T)) /
                         timer_total.get() / 1e9) +
          " GB/s");
      timer_total.clear();
    }
  }
  void Decompress(Array<1, Byte, DeviceType> &compressed_data,
                  enum error_bound_type ebtype, T tol, T s, T &norm,
                  Array<D, T, DeviceType> &decompressed_data, int queue_idx) {
    config.apply();

    DeviceRuntime<DeviceType>::SelectDevice(config.dev_id);
    log::info("Select device: " + DeviceRuntime<DeviceType>::GetDeviceName());
    Timer timer_total, timer_each;

    if (log::level & log::TIME)
      timer_total.start();

    decompressed_data.resize(hierarchy.level_shape(hierarchy.l_target()));
    LosslessDecompress(compressed_data, queue_idx);
    Dequantize(decompressed_data, ebtype, tol, s, norm, queue_idx);
    Recompose(decompressed_data, queue_idx);

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(0);
      timer_total.end();
      timer_total.print("Low-level decompression");
      log::time(
          "Low-level decompression throughput: " +
          std::to_string((double)(hierarchy.total_num_elems() * sizeof(T)) /
                         timer_total.get() / 1e9) +
          " GB/s");
      timer_total.clear();
    }
  }

  Hierarchy<D, T, DeviceType> hierarchy;
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