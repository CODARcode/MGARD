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

#ifndef MGARD_X_COMPRESSOR_CACHE_HPP
#define MGARD_X_COMPRESSOR_CACHE_HPP

namespace mgard_x {

template <DIM D, typename T, typename DeviceType, typename CompressorType>
class CompressorBundle {
public:
  CompressorType *compressor = nullptr;
  Array<D, T, DeviceType> *device_subdomain_buffer = nullptr;
  Array<1, Byte, DeviceType> *device_compressed_buffer = nullptr;

  bool initialized = false;

  void Release() {
    log::info("Releasing compressor cache");
    delete compressor;
    delete[] device_subdomain_buffer;
    delete[] device_compressed_buffer;
    initialized = false;
  }

  void Initialize() {
    log::info("Initializing compressor cache");
    compressor = new CompressorType();
    device_subdomain_buffer = new Array<D, T, DeviceType>[2];
    device_compressed_buffer = new Array<1, Byte, DeviceType>[2];
    initialized = true;
  }

  void SafeInitialize() {
    if (!initialized) {
      Initialize();
    }
  }

  void SafeRelease() {
    if (initialized) {
      Release();
    }
  }

  size_t CacheSize() {
    size_t size = 0;
    if (initialized) {
      size += CompressorType::EstimateMemoryFootprint(
          compressor->hierarchy.level_shape(compressor->hierarchy.l_target()),
          compressor->config);
      size += compressor->hierarchy.total_num_elems() * sizeof(T) * 4;
    }
    return size;
  }
};

template <DIM D, typename T, typename DeviceType, typename CompressorType>
class CompressorCache {
public:
  static inline CompressorBundle<D, T, DeviceType, CompressorType> cache;
};

} // namespace mgard_x

#endif
