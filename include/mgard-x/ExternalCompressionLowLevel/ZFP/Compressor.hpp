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

#include "../../Utilities/Types.h"

#include "../../Config/Config.h"
#include "../../Hierarchy/Hierarchy.h"
#include "../../RuntimeX/RuntimeX.h"
#include "Compressor.h"

#ifndef MGARD_X_ZFP_COMPRESSOR_HPP
#define MGARD_X_ZFP_COMPRESSOR_HPP

namespace mgard_x {

namespace zfp {
static bool debug_print_compression = true;

template <DIM D, typename T, typename DeviceType>
Compressor<D, T, DeviceType>::Compressor() : initialized(false) {}

template <DIM D, typename T, typename DeviceType>
Compressor<D, T, DeviceType>::Compressor(Hierarchy<D, T, DeviceType> &hierarchy,
                                         Config config)
    : initialized(true), hierarchy(&hierarchy), config(config) {
  zfp_stream = Array<1, ZFPWord, DeviceType>({hierarchy.total_num_elems()});
}

template <DIM D, typename T, typename DeviceType>
void Compressor<D, T, DeviceType>::Adapt(Hierarchy<D, T, DeviceType> &hierarchy,
                                         Config config, int queue_idx) {
  this->initialized = true;
  this->hierarchy = &hierarchy;
  this->config = config;
  zfp_stream.resize({hierarchy.total_num_elems()});
}

template <DIM D, typename T, typename DeviceType>
size_t
Compressor<D, T, DeviceType>::EstimateMemoryFootprint(std::vector<SIZE> shape,
                                                      Config config) {
  Hierarchy<D, T, DeviceType> hierarchy;
  hierarchy.EstimateMemoryFootprint(shape);
  size_t size = 0;
  size += sizeof(ZFPWord) * hierarchy.total_num_elems();
  return size;
}

template <DIM D, typename T, typename DeviceType>
void Compressor<D, T, DeviceType>::Decompose(
    Array<D, T, DeviceType> &original_data, int queue_idx) {
  // Do nothing
}

template <DIM D, typename T, typename DeviceType>
void Compressor<D, T, DeviceType>::Quantize(
    Array<D, T, DeviceType> &original_data, enum error_bound_type ebtype, T tol,
    T s, T norm, int queue_idx) {
  int rate = (int)tol;
  uint n = 1u << (2 * D);
  uint bits = (uint)floor(n * rate + 0.5);
  encode(original_data, zfp_stream, bits, queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void Compressor<D, T, DeviceType>::LosslessCompress(
    Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
  compressed_data.resize({(SIZE)zfp_stream.shape(0) * sizeof(ZFPWord)});
  MemoryManager<DeviceType>::Copy1D(
      compressed_data.data(), (Byte *)zfp_stream.data(),
      zfp_stream.shape(0) * sizeof(ZFPWord), queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void Compressor<D, T, DeviceType>::Recompose(
    Array<D, T, DeviceType> &decompressed_data, int queue_idx) {
  // Do nothing
}

template <DIM D, typename T, typename DeviceType>
void Compressor<D, T, DeviceType>::Dequantize(
    Array<D, T, DeviceType> &decompressed_data, enum error_bound_type ebtype,
    T tol, T s, T norm, int queue_idx) {
  decompressed_data.resize(hierarchy->level_shape(hierarchy->l_target()));
  int rate = (int)tol;
  uint n = 1u << (2 * D);
  uint bits = (uint)floor(n * rate + 0.5);
  decode(zfp_stream, decompressed_data, bits, queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void Compressor<D, T, DeviceType>::LosslessDecompress(
    Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
  zfp_stream.resize({(SIZE)compressed_data.shape(0) / sizeof(ZFPWord)});
  MemoryManager<DeviceType>::Copy1D((Byte *)zfp_stream.data(),
                                    compressed_data.data(),
                                    compressed_data.shape(0), queue_idx);
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

  Decompose(original_data, queue_idx);
  Quantize(original_data, ebtype, tol, s, norm, queue_idx);
  LosslessCompress(compressed_data, queue_idx);

  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
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

  decompressed_data.resize(hierarchy->level_shape(hierarchy->l_target()));
  LosslessDecompress(compressed_data, queue_idx);
  Dequantize(decompressed_data, ebtype, tol, s, norm, queue_idx);
  Recompose(decompressed_data, queue_idx);

  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
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
} // namespace zfp

} // namespace mgard_x

#endif