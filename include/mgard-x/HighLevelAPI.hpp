/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "compress_x.hpp"
#include "mgard-x/Handle.hpp" 
#include "mgard-x/Metadata.hpp"
#include "mgard-x/RuntimeX/RuntimeX.h"

#ifndef MGARD_X_HIGH_LEVEL_API_HPP
#define MGARD_X_HIGH_LEVEL_API_HPP

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
void compress(std::vector<SIZE> shape, T tol, T s, enum error_bound_type mode,
              const void *original_data, void *&compressed_data,
              size_t &compressed_size, Config config, bool output_pre_allocated) {
  Handle<D, T, DeviceType> handle(shape, config);
  mgard_x::Array<D, T, DeviceType> in_array(shape);
  in_array.loadData((const T *)original_data);
  Array<1, unsigned char, DeviceType> compressed_array =
      compress<D, T, DeviceType>(handle, in_array, mode, tol, s);
  compressed_size = compressed_array.getShape()[0];
  if (MemoryManager<DeviceType>::IsDevicePointer(original_data)) {
    if (!output_pre_allocated) {
      MemoryManager<DeviceType>::Malloc1D(compressed_data, compressed_size, 0);
    }
    MemoryManager<DeviceType>::Copy1D(compressed_data, (void*)compressed_array.get_dv(),
                                compressed_size, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
  } else {
    if (!output_pre_allocated) {
      compressed_data = (unsigned char *)malloc(compressed_size);
    }
    memcpy(compressed_data, compressed_array.getDataHost(), compressed_size);
  }
}

template <DIM D, typename T, typename DeviceType>
void compress(std::vector<SIZE> shape, T tol, T s, enum error_bound_type mode,
              const void *original_data, void *&compressed_data,
              size_t &compressed_size, Config config, std::vector<T *> coords,
              bool output_pre_allocated) {
  Handle<D, T, DeviceType> handle(shape, coords, config);
  mgard_x::Array<D, T, DeviceType> in_array(shape);
  in_array.loadData((const T *)original_data);
  Array<1, unsigned char, DeviceType> compressed_array =
      compress<D, T, DeviceType>(handle, in_array, mode, tol, s);
  compressed_size = compressed_array.getShape()[0];
  if (MemoryManager<DeviceType>::IsDevicePointer(original_data)) {
    if (!output_pre_allocated) {
      MemoryManager<DeviceType>::Malloc1D(compressed_data, compressed_size, 0);
    }
    MemoryManager<DeviceType>::Copy1D(compressed_data, (void*)compressed_array.get_dv(),
                                compressed_size, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
  } else {
    if (!output_pre_allocated) {
      compressed_data = (unsigned char *)malloc(compressed_size);
    }
    memcpy(compressed_data, compressed_array.getDataHost(), compressed_size);
  }
}

template <DIM D, typename T, typename DeviceType>
void decompress(std::vector<SIZE> shape, const void *compressed_data,
                size_t compressed_size, void *&decompressed_data,
                std::vector<T *> coords, Config config,
                bool output_pre_allocated) {
  size_t original_size = 1;
  for (int i = 0; i < D; i++) {
    original_size *= shape[i];
  }
  Handle<D, T, DeviceType> handle(shape, coords, config);
  std::vector<SIZE> compressed_shape(1);
  compressed_shape[0] = compressed_size;
  Array<1, unsigned char, DeviceType> compressed_array(compressed_shape);
  compressed_array.loadData((const unsigned char *)compressed_data);
  Array<D, T, DeviceType> out_array = decompress<D, T, DeviceType>(handle, compressed_array);

  if (MemoryManager<DeviceType>::IsDevicePointer(compressed_data)) {
    if (!output_pre_allocated) {
      MemoryManager<DeviceType>::Malloc1D(decompressed_data, original_size * sizeof(T), 0);
    }
    MemoryManager<DeviceType>::Copy1D(decompressed_data, (void*)out_array.get_dv(),
                                original_size * sizeof(T), 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);

  } else {
    if (!output_pre_allocated) {
      decompressed_data = (T *)malloc(original_size * sizeof(T));
    }
    memcpy(decompressed_data, out_array.getDataHost(),
           original_size * sizeof(T));
  }
}

template <DIM D, typename T, typename DeviceType>
void decompress(std::vector<SIZE> shape, const void *compressed_data,
                size_t compressed_size, void *&decompressed_data,
                Config config, bool output_pre_allocated) {
  size_t original_size = 1;
  for (int i = 0; i < D; i++)
    original_size *= shape[i];
  Handle<D, T, DeviceType> handle(shape, config);
  std::vector<SIZE> compressed_shape(1);
  compressed_shape[0] = compressed_size;
  Array<1, unsigned char, DeviceType> compressed_array(compressed_shape);
  compressed_array.loadData((const unsigned char *)compressed_data);
  Array<D, T, DeviceType> out_array = decompress<D, T, DeviceType>(handle, compressed_array);
  if (MemoryManager<DeviceType>::IsDevicePointer(compressed_data)) {
    if (!output_pre_allocated) {
      MemoryManager<DeviceType>::Malloc1D(decompressed_data, original_size * sizeof(T), 0);
    }
    MemoryManager<DeviceType>::Copy1D(decompressed_data, (void*)out_array.get_dv(),
                                original_size * sizeof(T), 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
  } else {
    if (!output_pre_allocated) {
      decompressed_data = (T *)malloc(original_size * sizeof(T));
    }
    memcpy(decompressed_data, out_array.getDataHost(),
           original_size * sizeof(T));
  }
}

template<typename DeviceType>
void BeginAutoTuning() {
  AutoTuner<DeviceType>::ProfileKernels = true;
}

template<typename DeviceType>
void EndAutoTuning() {
  AutoTuner<DeviceType>::ProfileKernels = false;
}


}

#endif