/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "../Utilities/Types.h"

#include "../Config/Config.h"
#include "../Hierarchy/Hierarchy.h"
#include "../RuntimeX/RuntimeX.h"
#include "CompressorCache.hpp"
#include "HybridHierarchyCompressor.h"

#ifndef MGARD_X_HYBRID_HIERARCHY_COMPRESSOR_HPP
#define MGARD_X_HYBRID_HIERARCHY_COMPRESSOR_HPP

namespace mgard_x {

static bool debug_print_compression = true;

template <DIM D, typename T, typename DeviceType>
HybridHierarchyCompressor<D, T, DeviceType>::HybridHierarchyCompressor()
    : initialized(false) {}

template <DIM D, typename T, typename DeviceType>
HybridHierarchyCompressor<D, T, DeviceType>::HybridHierarchyCompressor(
    Hierarchy<D, T, DeviceType> &hierarchy, Config config)
    : initialized(true), hierarchy(&hierarchy), config(config),
      refactor(hierarchy, config), hybrid_refactor(hierarchy, config),
      lossless_compressor(hierarchy.total_num_elems(), config),
      quantizer(hierarchy, config), hybrid_quantizer(hierarchy, config) {

  norm_array = Array<1, T, DeviceType>({1});
  // Reuse workspace. Warning:
  if (sizeof(QUANTIZED_INT) <= sizeof(T)) {
    // Reuse workspace if possible
    norm_tmp_array = Array<1, T, DeviceType>({hierarchy.total_num_elems()},
                                             (T *)refactor.w_array.data());
    quantized_array = Array<D, QUANTIZED_INT, DeviceType>(
        hierarchy.level_shape(hierarchy.l_target()),
        (QUANTIZED_INT *)refactor.w_array.data());
  } else {
    norm_tmp_array = Array<1, T, DeviceType>({hierarchy.total_num_elems()});
    quantized_array = Array<D, QUANTIZED_INT, DeviceType>(
        hierarchy.level_shape(hierarchy.l_target()), false, false);
  }
  std::vector<SIZE> shape = hierarchy.level_shape(hierarchy.l_target());
  SIZE decomposed_size = hybrid_refactor.DecomposedDataSize();

  decomposed_array = Array<1, T, DeviceType>({decomposed_size});
  hybrid_quantized_array =
      Array<1, QUANTIZED_INT, DeviceType>({decomposed_size});
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::Adapt(
    Hierarchy<D, T, DeviceType> &hierarchy, Config config, int queue_idx) {
  this->initialized = true;
  this->hierarchy = &hierarchy;
  this->config = config;
  refactor.Adapt(hierarchy, config, queue_idx);
  hybrid_refactor.Adapt(hierarchy, config, queue_idx);
  lossless_compressor.Adapt(hierarchy.total_num_elems(), config, queue_idx);
  quantizer.Adapt(hierarchy, config, queue_idx);
  hybrid_quantizer.Adapt(hierarchy, config, queue_idx);
  norm_array.resize({1}, queue_idx);
  // Reuse workspace. Warning:
  if (sizeof(QUANTIZED_INT) <= sizeof(T)) {
    // Reuse workspace if possible
    norm_tmp_array = Array<1, T, DeviceType>({hierarchy.total_num_elems()},
                                             (T *)refactor.w_array.data());
    quantized_array = Array<D, QUANTIZED_INT, DeviceType>(
        hierarchy.level_shape(hierarchy.l_target()),
        (QUANTIZED_INT *)refactor.w_array.data());
  } else {
    norm_tmp_array.resize({hierarchy.total_num_elems()}, queue_idx);
    quantized_array.resize(hierarchy.level_shape(hierarchy.l_target()),
                           queue_idx);
  }
  std::vector<SIZE> shape = hierarchy.level_shape(hierarchy.l_target());
  SIZE decomposed_size = hybrid_refactor.DecomposedDataSize();

  decomposed_array.resize({decomposed_size}, queue_idx);
  hybrid_quantized_array.resize({decomposed_size}, queue_idx);
}

template <DIM D, typename T, typename DeviceType>
size_t HybridHierarchyCompressor<D, T, DeviceType>::EstimateMemoryFootprint(
    std::vector<SIZE> shape, Config config) {
  Hierarchy<D, T, DeviceType> hierarchy;
  hierarchy.EstimateMemoryFootprint(shape);
  size_t size = 0;
  size += DataRefactorType::EstimateMemoryFootprint(shape);
  size += LinearQuantizerType::EstimateMemoryFootprint(shape);
  size += LosslessCompressorType::EstimateMemoryFootprint(
      hierarchy.total_num_elems(), config);
  size += sizeof(T);
  if (sizeof(QUANTIZED_INT) > sizeof(T)) {
    size += sizeof(T) * hierarchy.total_num_elems();
    size += sizeof(QUANTIZED_INT) * hierarchy.total_num_elems();
  }
  return size;
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::CalculateNorm(
    Array<D, T, DeviceType> &original_data, enum error_bound_type ebtype, T s,
    T &norm, int queue_idx) {
  if (ebtype == error_bound_type::REL) {
    norm =
        norm_calculator(original_data, SubArray(norm_tmp_array),
                        SubArray(norm_array), s, config.normalize_coordinates);
  }
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::Decompose(
    Array<D, T, DeviceType> &original_data, int queue_idx) {
  hybrid_refactor.Decompose(original_data, decomposed_array, queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::Quantize(
    Array<D, T, DeviceType> &original_data, enum error_bound_type ebtype, T tol,
    T s, T norm, int queue_idx) {
  hybrid_quantizer.Quantize(decomposed_array, ebtype, tol, s, norm,
                            hybrid_quantized_array, lossless_compressor,
                            queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::LosslessCompress(
    Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
  Array<1, QUANTIZED_UNSIGNED_INT, DeviceType> hybrid_quantized_liearized_array(
      {hierarchy->total_num_elems()},
      (QUANTIZED_UNSIGNED_INT *)hybrid_quantized_array.data());

  // SIZE ori_size = 512*512*512;
  // SIZE coarse_size = 320*320*320;

  //   SIZE ori_size = 64*64*64;
  // SIZE coarse_size = 40*40*40;
  // SIZE coeff_size = ori_size-coarse_size;

  // Array<1, QUANTIZED_UNSIGNED_INT, DeviceType>
  // hybrid_quantized_liearized_array1(
  //     {coeff_size},
  //     (QUANTIZED_UNSIGNED_INT *)hybrid_quantized_array.data()+coarse_size);

  // Array<1, QUANTIZED_UNSIGNED_INT, DeviceType>
  // hybrid_quantized_liearized_array2(
  //     {coarse_size},
  //     (QUANTIZED_UNSIGNED_INT *)hybrid_quantized_array.data());

  // PrintSubarray("coeff", SubArray(hybrid_quantized_liearized_array1));
  // PrintSubarray("coarse", SubArray(hybrid_quantized_liearized_array2));
  // PrintSubarray("hh", SubArray(hybrid_quantized_liearized_array));

  // DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
  // DumpSubArray("quantized_hh.dat",
  // SubArray(hybrid_quantized_liearized_array));
  // DumpSubArray("quantized_coarse.dat",
  // SubArray(hybrid_quantized_liearized_array2));
  // DumpSubArray("quantized_coeff.dat",
  // SubArray(hybrid_quantized_liearized_array1));
  // DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
  // std::cout << "done dumping....\n";

  // LosslessCompressorType lossless_compressor1(coeff_size, config);

  // lossless_compressor1.Compress(hybrid_quantized_liearized_array1,
  //                              compressed_data, queue_idx);

  // LosslessCompressorType lossless_compressor2(coarse_size, config);

  // lossless_compressor2.Compress(hybrid_quantized_liearized_array2,
  //                              compressed_data, queue_idx);

  lossless_compressor.Compress(hybrid_quantized_liearized_array,
                               compressed_data, queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::Recompose(
    Array<D, T, DeviceType> &decompressed_data, int queue_idx) {
  refactor.Recompose(decompressed_data, queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::Dequantize(
    Array<D, T, DeviceType> &decompressed_data, enum error_bound_type ebtype,
    T tol, T s, T norm, int queue_idx) {
  quantizer.Dequantize(decompressed_data, ebtype, tol, s, norm, quantized_array,
                       lossless_compressor, queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::LosslessDecompress(
    Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
  Array<1, QUANTIZED_UNSIGNED_INT, DeviceType> quantized_liearized_data(
      {hierarchy->total_num_elems()},
      (QUANTIZED_UNSIGNED_INT *)quantized_array.data());
  lossless_compressor.Decompress(compressed_data, quantized_liearized_data,
                                 queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::Compress(
    Array<D, T, DeviceType> &original_data, enum error_bound_type ebtype, T tol,
    T s, T &norm, Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {

  config.apply();

  DeviceRuntime<DeviceType>::SelectDevice(config.dev_id);
  log::info("Select device: " + DeviceRuntime<DeviceType>::GetDeviceName());
  Timer timer_total;
  for (int d = D - 1; d >= 0; d--) {
    if (hierarchy->level_shape(hierarchy->l_target(), d) !=
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
  if (config.compress_with_dryrun) {
    Dequantize(original_data, ebtype, tol, s, norm, queue_idx);
    Recompose(original_data, queue_idx);
  }

  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncQueue(0);
    timer_total.end();
    timer_total.print("Low-level compression");
    log::time(
        "Low-level compression throughput: " +
        std::to_string((double)(hierarchy->total_num_elems() * sizeof(T)) /
                       timer_total.get() / 1e9) +
        " GB/s");
    timer_total.clear();
  }
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::Decompress(
    Array<1, Byte, DeviceType> &compressed_data, enum error_bound_type ebtype,
    T tol, T s, T &norm, Array<D, T, DeviceType> &decompressed_data,
    int queue_idx) {
  config.apply();

  DeviceRuntime<DeviceType>::SelectDevice(config.dev_id);
  log::info("Select device: " + DeviceRuntime<DeviceType>::GetDeviceName());
  Timer timer_total, timer_each;

  if (log::level & log::TIME)
    timer_total.start();

  decompressed_data.resize(hierarchy->level_shape(hierarchy->l_target()));
  LosslessDecompress(compressed_data, queue_idx);
  Dequantize(decompressed_data, ebtype, tol, s, norm, queue_idx);
  Recompose(decompressed_data, queue_idx);

  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncQueue(0);
    timer_total.end();
    timer_total.print("Low-level decompression");
    log::time(
        "Low-level decompression throughput: " +
        std::to_string((double)(hierarchy->total_num_elems() * sizeof(T)) /
                       timer_total.get() / 1e9) +
        " GB/s");
    timer_total.clear();
  }
}

} // namespace mgard_x

#endif