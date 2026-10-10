/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#include "../Utilities/Types.h"

#include "../Config/Config.h"
#include "../Hierarchy/Hierarchy.h"
#include "../RuntimeX/RuntimeX.h"
#include "Compressor.h"
#include "CompressorCache.hpp"

#ifndef MGARD_X_160_COMPRESSOR_HPP
#define MGARD_X_160_COMPRESSOR_HPP

namespace mgard_x_160 {

static bool debug_print_compression = true;

template <DIM D, typename T, typename DeviceType>
Compressor<D, T, DeviceType>::Compressor() : initialized(false) {}

template <DIM D, typename T, typename DeviceType>
Compressor<D, T, DeviceType>::Compressor(Hierarchy<D, T, DeviceType> &hierarchy,
                                         Config config)
    : initialized(true), hierarchy(&hierarchy), config(config),
      refactor(hierarchy, config),
      lossless_compressor(hierarchy.total_num_elems(), config),
      quantizer(hierarchy, config) {

  norm_array = Array<1, T, DeviceType>({1});
  norm_tmp_array = Array<1, T, DeviceType>({hierarchy.total_num_elems()},
                                           (T *)refactor.w_array.data());
  // Reuse workspace. Warning:
  if (sizeof(QUANTIZED_INT) <= sizeof(T)) {
    // Reuse workspace if possible
    // norm_tmp_array = Array<1, T, DeviceType>({hierarchy.total_num_elems()},
    //                                          (T *)refactor.w_array.data());
    quantized_array = Array<D, QUANTIZED_INT, DeviceType>(
        hierarchy.level_shape(hierarchy.l_target()),
        (QUANTIZED_INT *)refactor.w_array.data());
  } else {
    // norm_tmp_array = Array<1, T, DeviceType>({hierarchy.total_num_elems()});
    quantized_array = Array<D, QUANTIZED_INT, DeviceType>(
        hierarchy.level_shape(hierarchy.l_target()));
  }
}

template <DIM D, typename T, typename DeviceType>
void Compressor<D, T, DeviceType>::Adapt(Hierarchy<D, T, DeviceType> &hierarchy,
                                         Config config, int queue_idx) {
  this->initialized = true;
  this->hierarchy = &hierarchy;
  this->config = config;
  refactor.Adapt(hierarchy, config, queue_idx);
  quantizer.Adapt(hierarchy, config, queue_idx);
  lossless_compressor.Adapt(hierarchy.total_num_elems(), config, queue_idx);
  norm_array.resize({1}, queue_idx);
  // Reuse workspace.
  norm_tmp_array = Array<1, T, DeviceType>({hierarchy.total_num_elems()},
                                           (T *)refactor.w_array.data());
  if (sizeof(QUANTIZED_INT) <= sizeof(T)) {
    // Reuse workspace if possible
    // norm_tmp_array = Array<1, T, DeviceType>({hierarchy.total_num_elems()},
    //                                          (T *)refactor.w_array.data());
    quantized_array = Array<D, QUANTIZED_INT, DeviceType>(
        hierarchy.level_shape(hierarchy.l_target()),
        (QUANTIZED_INT *)refactor.w_array.data());
  } else {
    // norm_tmp_array.resize({hierarchy.total_num_elems()}, queue_idx);
    quantized_array.resize(hierarchy.level_shape(hierarchy.l_target()),
                           queue_idx);
  }
}

template <DIM D, typename T, typename DeviceType>
size_t
Compressor<D, T, DeviceType>::EstimateMemoryFootprint(std::vector<SIZE> shape,
                                                      Config config) {
  Hierarchy<D, T, DeviceType> hierarchy;
  hierarchy.EstimateMemoryFootprint(shape);
  size_t size = 0;
  size += DataRefactorType::EstimateMemoryFootprint(shape);
  log::dbg(
      "Data refactor space: " +
      std::to_string(
          (double)(DataRefactorType::EstimateMemoryFootprint(shape)) / 1e9) +
      " GB");
  size += LinearQuantizerType::EstimateMemoryFootprint(shape);
  log::dbg(
      "Quantizer space: " +
      std::to_string(
          (double)(LinearQuantizerType::EstimateMemoryFootprint(shape)) / 1e9) +
      " GB");
  size += LosslessCompressorType::EstimateMemoryFootprint(
      hierarchy.total_num_elems(), config);
  log::dbg(
      "Lossless space: " +
      std::to_string((double)(LosslessCompressorType::EstimateMemoryFootprint(
                         hierarchy.total_num_elems(), config)) /
                     1e9) +
      " GB");
  size += sizeof(T);
  if (sizeof(QUANTIZED_INT) > sizeof(T)) {
    // size += sizeof(T) * hierarchy.total_num_elems();
    size += sizeof(QUANTIZED_INT) * hierarchy.total_num_elems();
  }
  return size;
}

template <DIM D, typename T, typename DeviceType>
void Compressor<D, T, DeviceType>::CalculateNorm(
    Array<D, T, DeviceType> &original_data, enum error_bound_type ebtype, T s,
    T &norm, int queue_idx) {
  if (ebtype == error_bound_type::REL) {
    norm =
        norm_calculator(original_data, SubArray(norm_tmp_array),
                        SubArray(norm_array), s, config.normalize_coordinates);
  }
}

template <DIM D, typename T, typename DeviceType>
void Compressor<D, T, DeviceType>::Decompose(
    Array<D, T, DeviceType> &original_data, bool orthogonal_projection,
    int queue_idx) {
  refactor.Decompose(SubArray(original_data), orthogonal_projection, queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void Compressor<D, T, DeviceType>::Quantize(
    Array<D, T, DeviceType> &original_data, enum error_bound_type ebtype, T tol,
    T s, T norm, int queue_idx) {
  orthogonal_projection =
      infer_orthogonal_projection<D>(config.projection_mode, s);
  quantizer.Quantize(original_data, ebtype, tol, s, norm, quantized_array,
                     lossless_compressor, queue_idx, orthogonal_projection);
}

template <DIM D, typename T, typename DeviceType>
void Compressor<D, T, DeviceType>::LosslessCompress(
    Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
  Array<1, QUANTIZED_INT, DeviceType> quantized_liearized_array(
      {hierarchy->total_num_elems()}, (QUANTIZED_INT *)quantized_array.data());
  lossless_compressor.Compress(quantized_liearized_array, compressed_data,
                               queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void Compressor<D, T, DeviceType>::Serialize(
    Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
  lossless_compressor.Serialize(compressed_data, queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void Compressor<D, T, DeviceType>::Deserialize(
    Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
  lossless_compressor.Deserialize(compressed_data, queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void Compressor<D, T, DeviceType>::Recompose(
    Array<D, T, DeviceType> &decompressed_data, bool orthogonal_projection,
    int queue_idx) {
  // The generic decompress pipeline always passes true here, but the basis was
  // decided from s during Dequantize/Decompress and stored in
  // this->orthogonal_projection. Prefer the stored decision so the hierarchical
  // L-infinity fast path stays consistent between decompose and recompose; the
  // caller's argument is only honored when it requests the (default) orthogonal
  // path, so explicit orthogonal callers are never overridden.
  refactor.Recompose(SubArray(decompressed_data),
                     orthogonal_projection && this->orthogonal_projection,
                     queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void Compressor<D, T, DeviceType>::DequantizeRecompose(
    Array<D, T, DeviceType> &decompressed_data, enum error_bound_type ebtype,
    T tol, T s, T norm, int queue_idx) {
  Dequantize(decompressed_data, ebtype, tol, s, norm, queue_idx);
  Recompose(decompressed_data, true, queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void Compressor<D, T, DeviceType>::Dequantize(
    Array<D, T, DeviceType> &decompressed_data, enum error_bound_type ebtype,
    T tol, T s, T norm, int queue_idx) {
  orthogonal_projection =
      infer_orthogonal_projection<D>(config.projection_mode, s);
  decompressed_data.resize(hierarchy->level_shape(hierarchy->l_target()));
  quantizer.Dequantize(decompressed_data, ebtype, tol, s, norm, quantized_array,
                       lossless_compressor, queue_idx, orthogonal_projection);
}

template <DIM D, typename T, typename DeviceType>
void Compressor<D, T, DeviceType>::LosslessDecompress(
    Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
  Array<1, QUANTIZED_INT, DeviceType> quantized_liearized_data(
      {hierarchy->total_num_elems()}, (QUANTIZED_INT *)quantized_array.data());
  lossless_compressor.Decompress(compressed_data, quantized_liearized_data,
                                 queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void Compressor<D, T, DeviceType>::Compress(
    Array<D, T, DeviceType> &original_data, enum error_bound_type ebtype, T tol,
    T s, T &norm, Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {

  config.apply();

  DeviceRuntime<DeviceType>::SelectDevice(config.dev_id);
  log::info("Select device: " + DeviceRuntime<DeviceType>::GetDeviceName());
  Timer timer_total;
  for (int d = D - 1; d >= 0; d--) {
    if (hierarchy->level_shape(hierarchy->l_target(), d) !=
        original_data.shape(d)) {
      log::err("The shape of input array does not match the shape initialized "
               "in hierarchy!");
      return;
    }
  }

  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    timer_total.start();
  }

  // Resolve config.projection_mode against s (Auto defaults to the cheap
  // hierarchical basis under L-infinity, orthogonal otherwise). Remember the
  // decision so Recompose (which has no s) reconstructs consistently.
  orthogonal_projection =
      infer_orthogonal_projection<D>(config.projection_mode, s);

  CalculateNorm(original_data, ebtype, s, norm, queue_idx);
  Decompose(original_data, orthogonal_projection, queue_idx);
  Quantize(original_data, ebtype, tol, s, norm, queue_idx);
  LosslessCompress(compressed_data, queue_idx);
  Serialize(compressed_data, queue_idx);
  if (config.compress_with_dryrun) {
    Dequantize(original_data, ebtype, tol, s, norm, queue_idx);
    Recompose(original_data, orthogonal_projection, queue_idx);
  }

  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    timer_total.end();
    timer_total.print("Low-level compression",
                      hierarchy->total_num_elems() * sizeof(T));
    timer_total.clear();
  }
}

template <DIM D, typename T, typename DeviceType>
void Compressor<D, T, DeviceType>::Decompress(
    Array<1, Byte, DeviceType> &compressed_data, enum error_bound_type ebtype,
    T tol, T s, T &norm, Array<D, T, DeviceType> &decompressed_data,
    int queue_idx) {
  config.apply();

  DeviceRuntime<DeviceType>::SelectDevice(config.dev_id);
  log::info("Select device: " + DeviceRuntime<DeviceType>::GetDeviceName());
  Timer timer_total, timer_each;

  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    timer_total.start();
  }

  // Must mirror the orthogonal-projection decision made during Compress so the
  // hierarchical fast path (s == inf) reconstructs consistently.
  orthogonal_projection =
      infer_orthogonal_projection<D>(config.projection_mode, s);

  decompressed_data.resize(hierarchy->level_shape(hierarchy->l_target()));
  Deserialize(compressed_data, queue_idx);
  LosslessDecompress(compressed_data, queue_idx);
  Dequantize(decompressed_data, ebtype, tol, s, norm, queue_idx);
  Recompose(decompressed_data, orthogonal_projection, queue_idx);

  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    timer_total.end();
    timer_total.print("Low-level decompression",
                      hierarchy->total_num_elems() * sizeof(T));
    timer_total.clear();
  }
}

} // namespace mgard_x_160

#endif