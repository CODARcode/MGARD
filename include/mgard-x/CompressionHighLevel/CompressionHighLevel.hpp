/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "../Config/Config.h"
#include "../Hierarchy/Hierarchy.h"
#include "../RuntimeX/RuntimeX.h"

#include "../CompressionLowLevel/Compressor.h"

#if MGARD_ENABLE_EXTERNAL_COMPRESSOR
#include "../ExternalCompressionLowLevel/ZFP/Compressor.h"
#endif

#include "../CompressionLowLevel/CompressorCache.hpp"
#include "../CompressionLowLevel/HybridHierarchyCompressor.h"
#include "../CompressionLowLevel/NormCalculator.hpp"

#include "../DomainDecomposer/DomainDecomposer.hpp"
#include "../Metadata/Metadata.hpp"
#include "compress_x.hpp"

#ifndef MGARD_X_COMPRESSION_HIGH_LEVEL_API_HPP
#define MGARD_X_COMPRESSION_HIGH_LEVEL_API_HPP

#define OUTPUT_SAFTY_OVERHEAD 1e6

namespace mgard_x {

template <DIM D, typename T, typename DeviceType, typename CompressorType>
T calc_subdomain_norm_series_w_prefetch(
    DomainDecomposer<D, T, CompressorType, DeviceType> &domain_decomposer,
    T s) {
  Timer timer_series;
  if (log::level & log::TIME)
    timer_series.start();

  DeviceRuntime<DeviceType>::SyncQueue(0);
  Array<1, T, DeviceType> norm_array({1});
  SubArray<1, T, DeviceType> norm_subarray(norm_array);
  T norm = 0;

  // Two buffers one for current and one for next
  Array<D, T, DeviceType> device_subdomain_buffer[2];
  // Pre-allocate to the size of the first subdomain
  // Following subdomains should be no bigger than the first one
  // We shouldn't need to reallocate in the future
  device_subdomain_buffer[0].resize(domain_decomposer.subdomain_shape(0));
  device_subdomain_buffer[1].resize(domain_decomposer.subdomain_shape(0));

  // Pre-fetch the first subdomain to one buffer
  int current_buffer = 0;
  domain_decomposer.copy_subdomain(
      device_subdomain_buffer[current_buffer], 0,
      subdomain_copy_direction::OriginalToSubdomain, 0);

  for (SIZE curr_subdomain_id = 0;
       curr_subdomain_id < domain_decomposer.num_subdomains();
       curr_subdomain_id++) {
    SIZE next_subdomain_id;
    int next_buffer = (current_buffer + 1) % 2;
    if (curr_subdomain_id + 1 < domain_decomposer.num_subdomains()) {
      // Prefetch the next subdomain
      next_subdomain_id = curr_subdomain_id + 1;
      // Copy data
      domain_decomposer.copy_subdomain(
          device_subdomain_buffer[next_buffer], next_subdomain_id,
          subdomain_copy_direction::OriginalToSubdomain, 1);
    }

    // device_input_buffer has to be not pitched to avoid copy for
    // linearization
    assert(!device_subdomain_buffer[current_buffer].isPitched());
    // Disable normalize_coordinate since we do not want to void dividing
    // total_elems
    T local_norm =
        norm_calculator(device_subdomain_buffer[current_buffer],
                        SubArray<1, T, DeviceType>(), norm_subarray, s, false);
    if (s == std::numeric_limits<T>::infinity()) {
      norm = std::max(norm, local_norm);
    } else {
      norm += local_norm * local_norm;
    }
    current_buffer = next_buffer;
    DeviceRuntime<DeviceType>::SyncQueue(1);
  }
  if (log::level & log::TIME) {
    timer_series.end();
    timer_series.print("Calculate subdomains norm series");
    timer_series.clear();
  }

  DeviceRuntime<DeviceType>::SyncDevice();
  return norm;
}

template <DIM D, typename T, typename DeviceType, typename CompressorType>
T calc_norm_decomposed_w_prefetch(
    DomainDecomposer<D, T, CompressorType, DeviceType> &domain_decomposer, T s,
    bool normalize_coordinates, SIZE total_num_elem) {

  // Process a series of subdomains according to the subdomain id list
  T norm = calc_subdomain_norm_series_w_prefetch(domain_decomposer, s);
  if (s != std::numeric_limits<T>::infinity()) {
    if (!normalize_coordinates) {
      norm = std::sqrt(norm);
    } else {
      norm = std::sqrt(norm / total_num_elem);
    }
  }
  if (s == std::numeric_limits<T>::infinity()) {
    log::info("L_inf norm: " + std::to_string(norm));
  } else {
    log::info("L_2 norm: " + std::to_string(norm));
  }
  return norm;
}

template <DIM D, typename T, typename DeviceType, typename CompressorType>
T calc_norm_decomposed(
    DomainDecomposer<D, T, CompressorType, DeviceType> &domain_decomposer, T s,
    bool normalize_coordinates, SIZE total_num_elem) {
  Array<D, T, DeviceType> device_input_buffer;
  T norm = 0;
  for (int subdomain_id = 0; subdomain_id < domain_decomposer.num_subdomains();
       subdomain_id++) {
    domain_decomposer.copy_subdomain(
        device_input_buffer, subdomain_id,
        subdomain_copy_direction::OriginalToSubdomain, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
    Array<1, T, DeviceType> norm_array({1});
    SubArray<1, T, DeviceType> norm_subarray(norm_array);
    // device_input_buffer has to be not pitched to avoid copy for
    // linearization
    assert(!device_input_buffer.isPitched());
    // Disable normalize_coordinate since we do not want to void dividing
    // total_elems
    T local_norm =
        norm_calculator(device_input_buffer, SubArray<1, T, DeviceType>(),
                        norm_subarray, s, false);
    if (s == std::numeric_limits<T>::infinity()) {
      norm = std::max(norm, local_norm);
    } else {
      norm += local_norm * local_norm;
    }
  }
  if (s != std::numeric_limits<T>::infinity()) {
    if (!normalize_coordinates) {
      norm = std::sqrt(norm);
    } else {
      norm = std::sqrt(norm / total_num_elem);
    }
  }
  if (s == std::numeric_limits<T>::infinity()) {
    log::info("L_inf norm: " + std::to_string(norm));
  } else {
    log::info("L_2 norm: " + std::to_string(norm));
  }
  return norm;
}

template <typename T>
T calc_local_abs_tol(enum error_bound_type ebtype, T norm, T tol, T s,
                     SIZE num_subdomain) {
  T local_abs_tol;
  if (ebtype == error_bound_type::REL) {
    if (s == std::numeric_limits<T>::infinity()) {
      log::info("L_inf norm: " + std::to_string(norm));
      local_abs_tol = tol * norm;
    } else {
      log::info("L_2 norm: " + std::to_string(norm));
      local_abs_tol = std::sqrt((tol * norm) * (tol * norm) / num_subdomain);
    }
  } else {
    if (s == std::numeric_limits<T>::infinity()) {
      local_abs_tol = tol;
    } else {
      local_abs_tol = std::sqrt((tol * tol) / num_subdomain);
    }
  }
  log::info("local abs tol: " + std::to_string(local_abs_tol));
  return local_abs_tol;
}

template <typename DeviceType>
void load(Config &config, Metadata<DeviceType> &metadata) {
  config.domain_decomposition = metadata.ddtype;
  config.decomposition = metadata.decomposition;
  config.lossless = metadata.ltype;
  config.huff_dict_size = metadata.huff_dict_size;
  config.huff_block_size = metadata.huff_block_size;
  config.reorder = metadata.reorder;
}

template <DIM D, typename T, typename DeviceType>
bool is_same(Hierarchy<D, T, DeviceType> &hierarchy1,
             Hierarchy<D, T, DeviceType> &hierarchy2) {
  if (hierarchy1.data_structure() ==
          data_structure_type::Cartesian_Grid_Non_Uniform ||
      hierarchy2.data_structure() ==
          data_structure_type::Cartesian_Grid_Non_Uniform) {
    return false;
  }
  for (DIM d = 0; d < D; d++) {
    if (hierarchy1.level_shape(hierarchy1.l_target(), d) !=
        hierarchy2.level_shape(hierarchy2.l_target(), d)) {
      return false;
    }
  }
  return true;
}

template <DIM D, typename T, typename DeviceType, typename CompressorType>
enum compress_status_type compress_pipeline(
    DomainDecomposer<D, T, CompressorType, DeviceType> &domain_decomposer,
    T local_tol, T s, T &norm, enum error_bound_type local_ebtype,
    Config &config, Byte *compressed_subdomain_data,
    SIZE &compressed_subdomain_size) {
  Timer timer_series;
  if (log::level & log::TIME)
    timer_series.start();

  using Cache = CompressorCache<D, T, DeviceType, CompressorType>;
  using HierarchyType = typename CompressorType::HierarchyType;
  CompressorType &compressor = *Cache::cache.compressor;
  Array<D, T, DeviceType> *device_subdomain_buffer =
      Cache::cache.device_subdomain_buffer;
  Array<1, Byte, DeviceType> *device_compressed_buffer =
      Cache::cache.device_compressed_buffer;

  if (!Cache::cache.InHierarchyCache(domain_decomposer.subdomain_shape(0),
                                     domain_decomposer.uniform)) {
    Cache::cache.ClearHierarchyCache();
  }

  for (SIZE id = 0; id < domain_decomposer.num_subdomains(); id++) {
    if (!Cache::cache.InHierarchyCache(domain_decomposer.subdomain_shape(id),
                                       domain_decomposer.uniform)) {
      Cache::cache.InsertHierarchyCache(
          domain_decomposer.subdomain_hierarchy(id));
    }
  }

  log::info("Adjust device buffers");
  std::vector<SIZE> shape = domain_decomposer.subdomain_shape(0);
  SIZE num_elements = 1;
  for (int i = 0; i < shape.size(); i++)
    num_elements *= shape[i];
  device_subdomain_buffer[0].resize(shape);
  device_subdomain_buffer[1].resize(shape);
  device_compressed_buffer[0].resize(
      {domain_decomposer.subdomain_compressed_buffer_size(0)});
  device_compressed_buffer[1].resize(
      {domain_decomposer.subdomain_compressed_buffer_size(0)});

  DeviceRuntime<DeviceType>::SyncDevice();

  if (log::level & log::TIME) {
    timer_series.end();
    timer_series.print("Prepare device environment");
    timer_series.clear();
    timer_series.start();
  }

  // For serialization
  SIZE byte_offset = 0;

  // Pre-fetch the first subdomain to one buffer
  int current_buffer = 0;
  int current_queue = current_buffer;
  domain_decomposer.copy_subdomain(
      device_subdomain_buffer[current_buffer], 0,
      subdomain_copy_direction::OriginalToSubdomain, current_queue);

  for (SIZE curr_subdomain_id = 0;
       curr_subdomain_id < domain_decomposer.num_subdomains();
       curr_subdomain_id++) {
    SIZE next_subdomain_id;
    int next_buffer = (current_buffer + 1) % 2;
    int next_queue = next_buffer;
    HierarchyType &hierarchy = Cache::cache.GetHierarchyCache(
        domain_decomposer.subdomain_shape(curr_subdomain_id));
    if (curr_subdomain_id + 1 < domain_decomposer.num_subdomains()) {
      // Prefetch the next subdomain
      next_subdomain_id = curr_subdomain_id + 1;
      // Copy data
      domain_decomposer.copy_subdomain(
          device_subdomain_buffer[next_buffer], next_subdomain_id,
          subdomain_copy_direction::OriginalToSubdomain, next_queue);
    }

    log::info("Adapt Compressor to hierarchy");
    compressor.Adapt(hierarchy, config, current_queue);

    std::stringstream ss;
    for (DIM d = 0; d < D; d++) {
      ss << compressor.hierarchy->level_shape(compressor.hierarchy->l_target(),
                                              d)
         << " ";
    }
    log::info("Compressing subdomain " + std::to_string(curr_subdomain_id) +
              " with shape: " + ss.str());
    compressor.Compress(
        device_subdomain_buffer[current_buffer], local_ebtype, local_tol, s,
        norm, device_compressed_buffer[current_buffer], current_queue);

    SIZE compressed_size = device_compressed_buffer[current_buffer].shape(0);
    double CR = (double)compressor.hierarchy->total_num_elems() * sizeof(T) /
                compressed_size;
    log::info("Subdomain CR: " + std::to_string(CR));
    if (CR < 1.0) {
      log::info("Using uncompressed data instead");
      domain_decomposer.copy_subdomain(
          device_subdomain_buffer[current_buffer], curr_subdomain_id,
          subdomain_copy_direction::OriginalToSubdomain, current_queue);
      SIZE linearized_width = 1;
      for (DIM d = 0; d < D - 1; d++)
        linearized_width *= device_subdomain_buffer[current_buffer].shape(d);
      MemoryManager<DeviceType>::CopyND(
          device_compressed_buffer[current_buffer].data(),
          device_subdomain_buffer[current_buffer].shape(D - 1) * sizeof(T),
          (Byte *)device_subdomain_buffer[current_buffer].data(),
          device_subdomain_buffer[current_buffer].ld(D - 1) * sizeof(T),
          device_subdomain_buffer[current_buffer].shape(D - 1) * sizeof(T),
          linearized_width, current_queue);
      compressed_size = compressor.hierarchy->total_num_elems() * sizeof(T);
    }

    // Check if we have enough space
    if (compressed_size >
        compressed_subdomain_size - byte_offset - sizeof(SIZE)) {
      log::err("Output too large (original size: " +
               std::to_string((double)compressor.hierarchy->total_num_elems() *
                              sizeof(T) / 1e9) +
               " GB, compressed size: " +
               std::to_string((double)compressed_size / 1e9) +
               " GB, leftover buffer space: " +
               std::to_string((double)(compressed_subdomain_size - byte_offset -
                                       sizeof(SIZE)) /
                              1e9) +
               " GB)");
      return compress_status_type::OutputTooLargeFailure;
    }

    Serialize<SIZE, DeviceType>(compressed_subdomain_data, &compressed_size, 1,
                                byte_offset, current_queue);
    Serialize<Byte, DeviceType>(compressed_subdomain_data,
                                device_compressed_buffer[current_buffer].data(),
                                compressed_size, byte_offset, current_queue);
    if (config.compress_with_dryrun) {
      domain_decomposer.copy_subdomain(
          device_subdomain_buffer[current_buffer], curr_subdomain_id,
          subdomain_copy_direction::SubdomainToOriginal, current_queue);
    }
    current_buffer = next_buffer;
    current_queue = next_queue;
  }
  compressed_subdomain_size = byte_offset;
  DeviceRuntime<DeviceType>::SyncDevice();
  if (log::level & log::TIME) {
    timer_series.end();
    timer_series.print("Compress subdomains series with prefetch");
    timer_series.clear();
  }
  return compress_status_type::Success;
}

template <DIM D, typename T, typename DeviceType, typename CompressorType>
enum compress_status_type decompress_pipeline(
    DomainDecomposer<D, T, CompressorType, DeviceType> &domain_decomposer,
    T local_tol, T s, T norm, enum error_bound_type local_ebtype,
    Config &config, Byte *compressed_subdomain_data) {
  Timer timer_series;
  if (log::level & log::TIME)
    timer_series.start();

  SIZE byte_offset = 0;
  using Cache = CompressorCache<D, T, DeviceType, CompressorType>;
  using HierarchyType = typename CompressorType::HierarchyType;
  CompressorType &compressor = *Cache::cache.compressor;
  Array<D, T, DeviceType> *device_subdomain_buffer =
      Cache::cache.device_subdomain_buffer;
  Array<1, Byte, DeviceType> *device_compressed_buffer =
      Cache::cache.device_compressed_buffer;

  if (!Cache::cache.InHierarchyCache(domain_decomposer.subdomain_shape(0),
                                     domain_decomposer.uniform)) {
    Cache::cache.ClearHierarchyCache();
  }

  for (SIZE id = 0; id < domain_decomposer.num_subdomains(); id++) {
    if (!Cache::cache.InHierarchyCache(domain_decomposer.subdomain_shape(id),
                                       domain_decomposer.uniform)) {
      Cache::cache.InsertHierarchyCache(
          domain_decomposer.subdomain_hierarchy(id));
    }
  }

  log::info("Adjust device buffers");
  std::vector<SIZE> shape = domain_decomposer.subdomain_shape(0);
  SIZE num_elements = 1;
  for (int i = 0; i < shape.size(); i++)
    num_elements *= shape[i];
  device_subdomain_buffer[0].resize(shape);
  device_subdomain_buffer[1].resize(shape);
  device_compressed_buffer[0].resize(
      {domain_decomposer.subdomain_compressed_buffer_size(0)});
  device_compressed_buffer[1].resize(
      {domain_decomposer.subdomain_compressed_buffer_size(0)});

  DeviceRuntime<DeviceType>::SyncDevice();

  if (log::level & log::TIME) {
    timer_series.end();
    timer_series.print("Prepare device environment");
    timer_series.clear();
    timer_series.start();
  }

  // Pre-fetch the first subdomain on queue 0
  int current_buffer = 0;
  int current_queue = current_buffer;

  SIZE compressed_size;
  SIZE *compressed_size_ptr = &compressed_size;
  Byte *compressed_data;

  Deserialize<SIZE, DeviceType>(compressed_subdomain_data, compressed_size_ptr,
                                1, byte_offset, false, current_queue);
  DeviceRuntime<DeviceType>::SyncQueue(current_queue);
  Deserialize<Byte, DeviceType>(compressed_subdomain_data, compressed_data,
                                compressed_size, byte_offset, true,
                                current_queue);

  device_compressed_buffer[current_buffer].resize({compressed_size});

  MemoryManager<DeviceType>::Copy1D(
      device_compressed_buffer[current_buffer].data(), compressed_data,
      compressed_size, current_queue);

  for (SIZE curr_subdomain_id = 0;
       curr_subdomain_id < domain_decomposer.num_subdomains();
       curr_subdomain_id++) {
    SIZE next_subdomain_id;
    int next_buffer = (current_buffer + 1) % 2;
    int next_queue = next_buffer;
    HierarchyType &hierarchy = Cache::cache.GetHierarchyCache(
        domain_decomposer.subdomain_shape(curr_subdomain_id));
    if (curr_subdomain_id + 1 < domain_decomposer.num_subdomains()) {
      // Prefetch the next subdomain
      next_subdomain_id = curr_subdomain_id + 1;

      // Deserialize and copy next compressed data on queue 1
      SIZE compressed_size;
      SIZE *compressed_size_ptr = &compressed_size;
      Byte *compressed_data;

      Deserialize<SIZE, DeviceType>(compressed_subdomain_data,
                                    compressed_size_ptr, 1, byte_offset, false,
                                    next_queue);
      DeviceRuntime<DeviceType>::SyncQueue(next_queue);
      Deserialize<Byte, DeviceType>(compressed_subdomain_data, compressed_data,
                                    compressed_size, byte_offset, true,
                                    next_queue);

      device_compressed_buffer[next_buffer].resize({compressed_size});

      MemoryManager<DeviceType>::Copy1D(
          device_compressed_buffer[next_buffer].data(), compressed_data,
          compressed_size, next_queue);
    }

    log::info("Adapt Compressor to hierarchy");
    compressor.Adapt(hierarchy, config, current_queue);

    double CR = (double)compressor.hierarchy->total_num_elems() * sizeof(T) /
                compressed_size;
    log::info("Subdomain CR: " + std::to_string(CR));
    if (CR > 1.0) {
      std::stringstream ss;
      for (DIM d = 0; d < D; d++) {
        ss << compressor.hierarchy->level_shape(
                  compressor.hierarchy->l_target(), d)
           << " ";
      }
      log::info("Decompressing subdomain " + std::to_string(curr_subdomain_id) +
                " with shape: " + ss.str());
      compressor.LosslessDecompress(device_compressed_buffer[current_buffer],
                                    current_queue);
    }
    if (curr_subdomain_id > 0) {
      // We delay D2H since since it can delay the D2H in lossless decompession
      // and dequantization
      int previous_buffer = std::abs((current_buffer - 1) % 2);
      int previous_queue = previous_buffer;
      SIZE prev_subdomain_id = curr_subdomain_id - 1;
      domain_decomposer.copy_subdomain(
          device_subdomain_buffer[previous_buffer], prev_subdomain_id,
          subdomain_copy_direction::SubdomainToOriginal, previous_queue);
    }
    if (CR > 1.0) {
      compressor.Dequantize(device_subdomain_buffer[current_buffer],
                            local_ebtype, local_tol, s, norm, current_queue);
      compressor.Recompose(device_subdomain_buffer[current_buffer],
                           current_queue);
    } else {
      log::info("Skipping decompression as original data was saved instead");
      device_subdomain_buffer[current_buffer].resize(
          {compressor.hierarchy->level_shape(
              compressor.hierarchy->l_target())});
      SIZE linearized_width = 1;
      for (DIM d = 0; d < D - 1; d++)
        linearized_width *= device_subdomain_buffer[current_buffer].shape(d);
      MemoryManager<DeviceType>::CopyND(
          device_subdomain_buffer[current_buffer].data(),
          device_subdomain_buffer[current_buffer].ld(D - 1),
          (T *)device_compressed_buffer[current_buffer].data(),
          device_subdomain_buffer[current_buffer].shape(D - 1),
          device_subdomain_buffer[current_buffer].shape(D - 1),
          linearized_width, current_queue);
    }

    // Need to ensure decompession is complete without blocking other operations
    DeviceRuntime<DeviceType>::SyncQueue(current_queue);
    current_buffer = next_buffer;
    current_queue = next_queue;
  }

  // Copy the last subdomain
  int previous_buffer = std::abs((current_buffer - 1) % 2);
  int previous_queue = previous_buffer;
  SIZE prev_subdomain_id = domain_decomposer.num_subdomains() - 1;
  domain_decomposer.copy_subdomain(
      device_subdomain_buffer[previous_buffer], prev_subdomain_id,
      subdomain_copy_direction::SubdomainToOriginal, previous_queue);

  DeviceRuntime<DeviceType>::SyncDevice();
  if (log::level & log::TIME) {
    timer_series.end();
    timer_series.print("Decompress subdomains series with prefetch");
    timer_series.clear();
  }
  return compress_status_type::Success;
}

template <typename T> int max_dim(std::vector<T> &shape) {
  int max_d = 0;
  T max_n = 0;
  for (int i = 0; i < shape.size(); i++) {
    if (max_n < shape[i]) {
      max_d = i;
      max_n = shape[i];
    }
  }
  return max_d;
}

template <typename T> int min_dim(std::vector<T> &shape) {
  int min_d = 0;
  T min_n = SIZE_MAX;
  for (int i = 0; i < shape.size(); i++) {
    if (min_n > shape[i]) {
      min_d = i;
      min_n = shape[i];
    }
  }
  return min_d;
}

template <typename T> std::vector<T> find_refactors(T n) {
  std::vector<T> factors;
  T z = 2;
  while (z * z <= n) {
    if (n % z == 0) {
      factors.push_back(z);
      n /= z;
    } else {
      z++;
    }
  }
  if (n > 1) {
    factors.push_back(n);
  }
  return factors;
}

template <typename T> void adjust_shape(std::vector<T> &shape, Config config) {
  log::info("Using shape adjustment");
  int num_timesteps;
  if (config.domain_decomposition == domain_decomposition_type::TemporalDim) {
    // If do shape adjustment with temporal dim domain decomposition
    // the temporal dim has to be the first dim
    assert(config.temporal_dim == 0);
    num_timesteps = shape[0] / config.temporal_dim_size;
    shape[0] = config.temporal_dim_size;
  }
  int max_d = max_dim(shape);
  SIZE max_n = shape[max_d];
  std::vector<SIZE> factors = find_refactors(max_n);
  // std::cout << "factors: ";
  // for (SIZE f : factors) std::cout << f << " ";
  // std::cout << "\n";
  shape[max_d] = 1;
  for (int i = factors.size() - 1; i >= 0; i--) {
    int min_d = min_dim(shape);
    shape[min_d] *= factors[i];
    // std::cout << "multiple " << factors[i] <<
    // " to dim " << min_d << ": " << shape[min_d] << "\n";
  }
  if (config.domain_decomposition == domain_decomposition_type::TemporalDim) {
    shape[0] *= num_timesteps;
  }
  // std::cout << "shape: ";
  // for (SIZE n : shape) {
  //   std::cout <<  n << "\n";
  // }
  std::stringstream ss;
  for (DIM d = 0; d < shape.size(); d++) {
    ss << shape[d] << " ";
  }
  log::info("Shape adjusted to " + ss.str());
}

template <DIM D, typename T, typename DeviceType, typename CompressorType>
enum compress_status_type
general_compress(std::vector<SIZE> shape, T tol, T s,
                 enum error_bound_type ebtype, const void *original_data,
                 void *&compressed_data, size_t &compressed_size, Config config,
                 bool uniform, std::vector<T *> coords,
                 bool output_pre_allocated) {

  DeviceRuntime<DeviceType>::Initialize();
  size_t total_num_elem = 1;
  for (int i = 0; i < D; i++)
    total_num_elem *= shape[i];

  config.apply();

  log::info("adjust_shape: " + std::to_string(config.adjust_shape));
  if (config.adjust_shape) {
    adjust_shape(shape, config);
  }

  Timer timer_total, timer_each;
  if (log::level & log::TIME)
    timer_total.start();

  using Cache = CompressorCache<D, T, DeviceType, CompressorType>;
  Cache::cache.SafeInitialize();

  bool reduce_memory_footprint_original =
      MemoryManager<DeviceType>::ReduceMemoryFootprint;
  if (MemoryManager<DeviceType>::ReduceMemoryFootprint) {
    log::info("Original ReduceMemoryFootprint: 1");
  } else {
    log::info("Original ReduceMemoryFootprint: 0");
  }

  DomainDecomposer<D, T, CompressorType, DeviceType> domain_decomposer;
  if (uniform) {
    domain_decomposer =
        DomainDecomposer<D, T, CompressorType, DeviceType>(shape, config);
  } else {
    domain_decomposer = DomainDecomposer<D, T, CompressorType, DeviceType>(
        shape, config, coords);
  }
  domain_decomposer.set_original_data((T *)original_data);

  T norm = 1;
  T local_tol = tol;
  enum error_bound_type local_ebtype;

  if (config.decomposition == decomposition_type::MultiDim) {
    log::info("Multilevel Decomposition: multi-dimensional");
  } else if (config.decomposition == decomposition_type::SingleDim) {
    log::info("Multilevel Decomposition: single-dimensional");
  } else if (config.decomposition == decomposition_type::Hybrid) {
    log::info("Multilevel Decomposition: hybrid");
  }

  log::info("tol: " + std::to_string(tol));
  log::info("s: " + std::to_string(s));
  log::info("coordinate normalization: " +
            std::to_string(config.normalize_coordinates));
  if (!domain_decomposer.domain_decomposed()) {
    local_tol = tol;
    local_ebtype = ebtype;
  } else {
    if (log::level & log::TIME)
      timer_each.start();
    if (ebtype == error_bound_type::REL) {
      // norm = calc_norm_decomposed(domain_decomposer, s,
      //                             config.normalize_coordinates,
      //                             total_num_elem);
      norm = calc_norm_decomposed_w_prefetch(
          domain_decomposer, s, config.normalize_coordinates, total_num_elem);
    }
    local_tol = calc_local_abs_tol(ebtype, norm, tol, s,
                                   domain_decomposer.num_subdomains());
    // Force to use ABS mode when do domain decomposition
    local_ebtype = error_bound_type::ABS;
    if (log::level & log::TIME) {
      timer_each.end();
      timer_each.print("Calculate norm of decomposed domain");
      timer_each.clear();
    }
  }

  if (log::level & log::TIME)
    timer_each.start();
  // Use consistance memory space between input and output data
  size_t output_buffer_size;
  if (!output_pre_allocated) {
    output_buffer_size = total_num_elem * sizeof(T) + OUTPUT_SAFTY_OVERHEAD;
    if (MemoryManager<DeviceType>::IsDevicePointer(original_data)) {
      DeviceRuntime<DeviceType>::SelectDevice(
          MemoryManager<DeviceType>::GetPointerDevice(original_data));
      MemoryManager<DeviceType>::Malloc1D(compressed_data, output_buffer_size);
    } else {
      compressed_data = (unsigned char *)malloc(output_buffer_size);
    }
  } else {
    // compressed_size stores pre-allocated buffer size
    output_buffer_size = compressed_size;
  }
  log::info("Output preallocated: " + std::to_string(output_pre_allocated));

  bool input_previously_pinned = true;
  if (!MemoryManager<DeviceType>::IsDevicePointer((void *)original_data)) {
    input_previously_pinned =
        MemoryManager<DeviceType>::CheckHostRegister((void *)original_data);
    if (!input_previously_pinned && config.prefetch) {
      MemoryManager<DeviceType>::HostRegister((void *)original_data,
                                              total_num_elem * sizeof(T));
    }
    log::info("Input previously pinned: " +
              std::to_string(input_previously_pinned));
  } else {
    log::info("Input on device");
  }

  bool output_previously_pinned = true;
  if (!MemoryManager<DeviceType>::IsDevicePointer((void *)compressed_data)) {
    output_previously_pinned =
        MemoryManager<DeviceType>::CheckHostRegister((void *)compressed_data);
    if (!output_previously_pinned && config.prefetch) {
      MemoryManager<DeviceType>::HostRegister((void *)compressed_data,
                                              output_buffer_size);
    }
    log::info("Output previously pinned: " +
              std::to_string(output_previously_pinned));
  } else {
    log::info("Output on device");
  }

  // Estimate metadata size
  Metadata<DeviceType> m;
  if (uniform) {
    m.FillForCompression(
        ebtype, tol, s, norm, config.decomposition, config.reorder,
        config.lossless, config.huff_dict_size, config.huff_block_size, shape,
        domain_decomposer.domain_decomposed(), config.domain_decomposition,
        domain_decomposer.domain_decomposed_dim(),
        domain_decomposer.domain_decomposed_size());
  } else {
    m.FillForCompression(
        ebtype, tol, s, norm, config.decomposition, config.reorder,
        config.lossless, config.huff_dict_size, config.huff_block_size, shape,
        domain_decomposer.domain_decomposed(), config.domain_decomposition,
        domain_decomposer.domain_decomposed_dim(),
        domain_decomposer.domain_decomposed_size(), coords);
  }

  uint32_t metadata_size;
  SERIALIZED_TYPE *serizalied_meta = m.Serialize(metadata_size);
  MemoryManager<DeviceType>::Free(serizalied_meta);

  SIZE byte_offset = metadata_size;
  Byte *compressed_subdomain_data = (Byte *)compressed_data + byte_offset;
  SIZE compressed_subdomain_size = output_buffer_size - metadata_size;

  if (log::level & log::TIME) {
    timer_each.end();
    timer_each.print("Prepare input and output buffer");
    timer_each.clear();
  }

  enum compress_status_type compress_status;
  if (log::level & log::TIME)
    timer_each.start();
  DeviceRuntime<DeviceType>::SelectDevice(config.dev_id);
  bool old_sync_set = DeviceRuntime<DeviceType>::SyncAllKernelsAndCheckErrors;
  if (!config.prefetch) {
    DeviceRuntime<DeviceType>::SyncAllKernelsAndCheckErrors = true;
  } else {
    DeviceRuntime<DeviceType>::SyncAllKernelsAndCheckErrors = false;
  }
  compress_status = compress_pipeline(
      domain_decomposer, local_tol, s, norm, local_ebtype, config,
      compressed_subdomain_data, compressed_subdomain_size);

  DeviceRuntime<DeviceType>::SyncAllKernelsAndCheckErrors = old_sync_set;

  if (log::level & log::TIME) {
    timer_each.end();
    timer_each.print("Aggregated low-level compression");
    log::time("Aggregated low-level compression throughput: " +
              std::to_string((double)(total_num_elem * sizeof(T)) /
                             timer_each.get() / 1e9) +
              " GB/s");
    timer_each.clear();
  }

  if (log::level & log::TIME)
    timer_each.start();
  if (uniform) {
    m.FillForCompression(
        ebtype, tol, s, norm, config.decomposition, config.reorder,
        config.lossless, config.huff_dict_size, config.huff_block_size, shape,
        domain_decomposer.domain_decomposed(), config.domain_decomposition,
        domain_decomposer.domain_decomposed_dim(),
        domain_decomposer.domain_decomposed_size());
  } else {
    m.FillForCompression(
        ebtype, tol, s, norm, config.decomposition, config.reorder,
        config.lossless, config.huff_dict_size, config.huff_block_size, shape,
        domain_decomposer.domain_decomposed(), config.domain_decomposition,
        domain_decomposer.domain_decomposed_dim(),
        domain_decomposer.domain_decomposed_size(), coords);
  }

  serizalied_meta = m.Serialize(metadata_size);

  // Serialize
  byte_offset = 0;
  Serialize<SERIALIZED_TYPE, DeviceType>(
      (Byte *)compressed_data, serizalied_meta, metadata_size, byte_offset, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  MemoryManager<DeviceType>::Free(serizalied_meta);
  byte_offset += compressed_subdomain_size;
  DeviceRuntime<DeviceType>::SyncQueue(0);
  compressed_size = byte_offset;

  if (!input_previously_pinned && config.prefetch) {
    MemoryManager<DeviceType>::HostUnregister((void *)original_data);
  }
  if (!output_previously_pinned && config.prefetch) {
    MemoryManager<DeviceType>::HostUnregister((void *)compressed_data);
  }

  if (!config.cache_compressor)
    Cache::cache.SafeRelease();
  DeviceRuntime<DeviceType>::Finalize();

  MemoryManager<DeviceType>::ReduceMemoryFootprint =
      reduce_memory_footprint_original;
  if (MemoryManager<DeviceType>::ReduceMemoryFootprint) {
    log::info("ReduceMemoryFootprint restored to 1");
  } else {
    log::info("ReduceMemoryFootprint restored to 0");
  }

  if (log::level & log::TIME) {
    timer_each.end();
    timer_each.print("Serialization");
    timer_each.clear();
    timer_total.end();
    timer_total.print("High-level compression");
    log::time("High-level compression throughput: " +
              std::to_string((double)(total_num_elem * sizeof(T)) /
                             timer_total.get() / 1e9) +
              " GB/s");
    timer_total.clear();
  }

  return compress_status;
}

template <DIM D, typename T, typename DeviceType>
enum compress_status_type
compress(std::vector<SIZE> shape, T tol, T s, enum error_bound_type ebtype,
         const void *original_data, void *&compressed_data,
         size_t &compressed_size, Config config, bool output_pre_allocated) {
  if (config.compressor == compressor_type::MGARD) {
    if (config.decomposition != decomposition_type::Hybrid) {
      return general_compress<D, T, DeviceType, Compressor<D, T, DeviceType>>(
          shape, tol, s, ebtype, original_data, compressed_data,
          compressed_size, config, true, std::vector<T *>(0),
          output_pre_allocated);
    } else {
      return general_compress<D, T, DeviceType,
                              HybridHierarchyCompressor<D, T, DeviceType>>(
          shape, tol, s, ebtype, original_data, compressed_data,
          compressed_size, config, true, std::vector<T *>(0),
          output_pre_allocated);
    }
  } else if (config.compressor == compressor_type::ZFP) {
#if MGARD_ENABLE_EXTERNAL_COMPRESSOR
    return general_compress<D, T, DeviceType,
                            zfp::Compressor<D, T, DeviceType>>(
        shape, tol, s, ebtype, original_data, compressed_data, compressed_size,
        config, true, std::vector<T *>(0), output_pre_allocated);
#else
    log::err("MGARD not built with external compressor ZFP");
    return compress_status_type::Failure;
#endif
  }
}

template <DIM D, typename T, typename DeviceType>
enum compress_status_type
compress(std::vector<SIZE> shape, T tol, T s, enum error_bound_type ebtype,
         const void *original_data, void *&compressed_data,
         size_t &compressed_size, Config config, std::vector<T *> coords,
         bool output_pre_allocated) {
  if (config.compressor == compressor_type::MGARD) {
    if (config.decomposition != decomposition_type::Hybrid) {
      return general_compress<D, T, DeviceType, Compressor<D, T, DeviceType>>(
          shape, tol, s, ebtype, original_data, compressed_data,
          compressed_size, config, false, coords, output_pre_allocated);
    } else {
      return general_compress<D, T, DeviceType,
                              HybridHierarchyCompressor<D, T, DeviceType>>(
          shape, tol, s, ebtype, original_data, compressed_data,
          compressed_size, config, false, coords, output_pre_allocated);
    }
  } else if (config.compressor == compressor_type::ZFP) {
#if MGARD_ENABLE_EXTERNAL_COMPRESSOR
    return general_compress<D, T, DeviceType,
                            zfp::Compressor<D, T, DeviceType>>(
        shape, tol, s, ebtype, original_data, compressed_data, compressed_size,
        config, false, coords, output_pre_allocated);
#else
    log::err("MGARD not built with external compressor ZFP");
    return compress_status_type::Failure;
#endif
  }
}

template <DIM D, typename T, typename DeviceType, typename CompressorType>
enum compress_status_type
general_decompress(std::vector<SIZE> shape, const void *compressed_data,
                   size_t compressed_size, void *&decompressed_data,
                   Config config, bool output_pre_allocated) {
  DeviceRuntime<DeviceType>::Initialize();
  size_t total_num_elem = 1;
  for (int i = 0; i < D; i++)
    total_num_elem *= shape[i];

  config.apply();

  Timer timer_total, timer_each;
  if (log::level & log::TIME)
    timer_total.start();
  if (log::level & log::TIME)
    timer_each.start();

  using Cache = CompressorCache<D, T, DeviceType, CompressorType>;
  Cache::cache.SafeInitialize();

  bool reduce_memory_footprint_original =
      MemoryManager<DeviceType>::ReduceMemoryFootprint;
  if (MemoryManager<DeviceType>::ReduceMemoryFootprint) {
    log::info("Original ReduceMemoryFootprint: 1");
  } else {
    log::info("Original ReduceMemoryFootprint: 0");
  }

  // Use consistance memory space between input and output data
  if (!output_pre_allocated) {
    if (MemoryManager<DeviceType>::IsDevicePointer(compressed_data)) {
      DeviceRuntime<DeviceType>::SelectDevice(
          MemoryManager<DeviceType>::GetPointerDevice(compressed_data));
      MemoryManager<DeviceType>::Malloc1D(decompressed_data,
                                          total_num_elem * sizeof(T));
    } else {
      decompressed_data = (void *)malloc(total_num_elem * sizeof(T));
    }
  }
  log::info("Output preallocated: " + std::to_string(output_pre_allocated));

  bool input_previously_pinned = true;
  if (!MemoryManager<DeviceType>::IsDevicePointer((void *)compressed_data)) {
    input_previously_pinned =
        MemoryManager<DeviceType>::CheckHostRegister((void *)compressed_data);
    if (!input_previously_pinned && config.prefetch) {
      MemoryManager<DeviceType>::HostRegister((void *)compressed_data,
                                              compressed_size);
    }
    log::info("Input previously pinned: " +
              std::to_string(input_previously_pinned));
  } else {
    log::info("Input on device");
  }
  bool output_previously_pinned = true;
  if (!MemoryManager<DeviceType>::IsDevicePointer((void *)decompressed_data)) {
    output_previously_pinned =
        MemoryManager<DeviceType>::CheckHostRegister((void *)decompressed_data);
    if (!output_previously_pinned && config.prefetch) {
      MemoryManager<DeviceType>::HostRegister((void *)decompressed_data,
                                              total_num_elem * sizeof(T));
    }
    log::info("Output previously pinned: " +
              std::to_string(output_previously_pinned));
  } else {
    log::info("Output on device");
  }

  if (log::level & log::TIME) {
    timer_each.end();
    timer_each.print("Prepare input and output buffer");
    timer_each.clear();
  }

  if (log::level & log::TIME)
    timer_each.start();

  Metadata<DeviceType> m;
  m.Deserialize((SERIALIZED_TYPE *)compressed_data);
  load(config, m);

  std::vector<T *> coords(D);
  if (m.dstype == data_structure_type::Cartesian_Grid_Non_Uniform) {
    for (DIM d = 0; d < D; d++) {
      coords[d] = new T[shape[d]];
      for (SIZE i = 0; i < shape[d]; i++) {
        coords[d][i] = (float)m.coords[d][i];
      }
    }
  }

  if (config.decomposition == decomposition_type::MultiDim) {
    log::info("Multilevel Decomposition: multi-dimensional");
  } else if (config.decomposition == decomposition_type::SingleDim) {
    log::info("Multilevel Decomposition: single-dimensional");
  } else if (config.decomposition == decomposition_type::Hybrid) {
    log::info("Multilevel Decomposition: hybrid");
  }
  log::info("tol: " + std::to_string(m.tol));
  log::info("s: " + std::to_string(m.s));

  // Initialize DomainDecomposer
  DomainDecomposer<D, T, CompressorType, DeviceType> domain_decomposer;

  if (m.dstype == data_structure_type::Cartesian_Grid_Uniform) {
    domain_decomposer = DomainDecomposer<D, T, CompressorType, DeviceType>(
        shape, m.domain_decomposed, m.domain_decomposed_dim,
        m.domain_decomposed_size, config);
  } else {
    domain_decomposer = DomainDecomposer<D, T, CompressorType, DeviceType>(
        shape, m.domain_decomposed, m.domain_decomposed_dim,
        m.domain_decomposed_size, config, coords);
  }
  domain_decomposer.set_original_data((T *)decompressed_data);

  // Preparing decompression parameters
  T local_tol;
  enum error_bound_type local_ebtype;

  if (log::level & log::TIME)
    timer_each.start();

  if (!m.domain_decomposed) {
    local_tol = m.tol;
    local_ebtype = m.ebtype;
  } else {
    local_tol = calc_local_abs_tol(m.ebtype, m.norm, m.tol, m.s,
                                   domain_decomposer.num_subdomains());
    // Force to use ABS mode when do domain decomposition
    local_ebtype = error_bound_type::ABS;
  }

  // Deserialize compressed data
  Byte *compressed_subdomain_data = (Byte *)compressed_data + m.metadata_size;

  if (log::level & log::TIME) {
    timer_each.end();
    timer_each.print("Deserialization");
    timer_each.clear();
  }
  enum compress_status_type decompress_status;
  if (log::level & log::TIME)
    timer_each.start();
  DeviceRuntime<DeviceType>::SelectDevice(config.dev_id);
  bool old_sync_set = DeviceRuntime<DeviceType>::SyncAllKernelsAndCheckErrors;
  if (!config.prefetch) {
    DeviceRuntime<DeviceType>::SyncAllKernelsAndCheckErrors = true;
  } else {
    DeviceRuntime<DeviceType>::SyncAllKernelsAndCheckErrors = false;
  }
  decompress_status =
      decompress_pipeline(domain_decomposer, local_tol, (T)m.s, (T)m.norm,
                          local_ebtype, config, compressed_subdomain_data);
  DeviceRuntime<DeviceType>::SyncAllKernelsAndCheckErrors = old_sync_set;
  if (log::level & log::TIME) {
    timer_each.end();
    timer_each.print("Aggregated low-level decompression");
    log::time("Aggregated low-level decompression throughput: " +
              std::to_string((double)(total_num_elem * sizeof(T)) /
                             timer_each.get() / 1e9) +
              " GB/s");
    timer_each.clear();
  }

  if (!input_previously_pinned && config.prefetch) {
    MemoryManager<DeviceType>::HostUnregister((void *)compressed_data);
  }
  if (!output_previously_pinned && config.prefetch) {
    MemoryManager<DeviceType>::HostUnregister((void *)decompressed_data);
  }

  if (m.dstype == data_structure_type::Cartesian_Grid_Non_Uniform) {
    for (DIM d = 0; d < D; d++)
      delete[] coords[d];
  }

  if (!config.cache_compressor)
    Cache::cache.SafeRelease();
  DeviceRuntime<DeviceType>::Finalize();

  MemoryManager<DeviceType>::ReduceMemoryFootprint =
      reduce_memory_footprint_original;
  if (MemoryManager<DeviceType>::ReduceMemoryFootprint) {
    log::info("ReduceMemoryFootprint restored to 1");
  } else {
    log::info("ReduceMemoryFootprint restored to 0");
  }

  if (log::level & log::TIME) {
    timer_total.end();
    timer_total.print("High-level decompression");
    log::time("High-level decompression throughput: " +
              std::to_string((double)(total_num_elem * sizeof(T)) /
                             timer_total.get() / 1e9) +
              " GB/s");
    timer_total.clear();
  }

  return decompress_status;
}

template <DIM D, typename T, typename DeviceType>
enum compress_status_type
decompress(std::vector<SIZE> shape, const void *compressed_data,
           size_t compressed_size, void *&decompressed_data, Config config,
           bool output_pre_allocated) {
  if (config.compressor == compressor_type::MGARD) {
    if (config.decomposition != decomposition_type::Hybrid) {
      return general_decompress<D, T, DeviceType, Compressor<D, T, DeviceType>>(
          shape, compressed_data, compressed_size, decompressed_data, config,
          output_pre_allocated);
    } else {
      return general_decompress<D, T, DeviceType,
                                HybridHierarchyCompressor<D, T, DeviceType>>(
          shape, compressed_data, compressed_size, decompressed_data, config,
          output_pre_allocated);
    }
  } else if (config.compressor == compressor_type::ZFP) {
#if MGARD_ENABLE_EXTERNAL_COMPRESSOR
    return general_decompress<D, T, DeviceType,
                              zfp::Compressor<D, T, DeviceType>>(
        shape, compressed_data, compressed_size, decompressed_data, config,
        output_pre_allocated);
#else
    log::err("MGARD not built with external compressor ZFP");
    return compress_status_type::Failure;
#endif
  }
}

template <typename DeviceType>
enum compress_status_type
compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol, double s,
         enum error_bound_type mode, const void *original_data,
         void *&compressed_data, size_t &compressed_size, Config config,
         bool output_pre_allocated) {
  if (dtype == data_type::Float) {
    if (D == 1) {
      return compress<1, float, DeviceType>(shape, tol, s, mode, original_data,
                                            compressed_data, compressed_size,
                                            config, output_pre_allocated);
    } else if (D == 2) {
      return compress<2, float, DeviceType>(shape, tol, s, mode, original_data,
                                            compressed_data, compressed_size,
                                            config, output_pre_allocated);
    } else if (D == 3) {
      return compress<3, float, DeviceType>(shape, tol, s, mode, original_data,
                                            compressed_data, compressed_size,
                                            config, output_pre_allocated);
    } else if (D == 4) {
      return compress<4, float, DeviceType>(shape, tol, s, mode, original_data,
                                            compressed_data, compressed_size,
                                            config, output_pre_allocated);
    } else if (D == 5) {
      return compress<5, float, DeviceType>(shape, tol, s, mode, original_data,
                                            compressed_data, compressed_size,
                                            config, output_pre_allocated);
    } else {
      return compress_status_type::NotSupportHigherNumberOfDimensionsFailure;
    }
  } else if (dtype == data_type::Double) {
    if (D == 1) {
      return compress<1, double, DeviceType>(shape, tol, s, mode, original_data,
                                             compressed_data, compressed_size,
                                             config, output_pre_allocated);
    } else if (D == 2) {
      return compress<2, double, DeviceType>(shape, tol, s, mode, original_data,
                                             compressed_data, compressed_size,
                                             config, output_pre_allocated);
    } else if (D == 3) {
      return compress<3, double, DeviceType>(shape, tol, s, mode, original_data,
                                             compressed_data, compressed_size,
                                             config, output_pre_allocated);
    } else if (D == 4) {
      return compress<4, double, DeviceType>(shape, tol, s, mode, original_data,
                                             compressed_data, compressed_size,
                                             config, output_pre_allocated);
    } else if (D == 5) {
      return compress<5, double, DeviceType>(shape, tol, s, mode, original_data,
                                             compressed_data, compressed_size,
                                             config, output_pre_allocated);
    } else {
      return compress_status_type::NotSupportHigherNumberOfDimensionsFailure;
    }
  } else {
    return compress_status_type::NotSupportDataTypeFailure;
  }
}

template <typename DeviceType>
enum compress_status_type
compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol, double s,
         enum error_bound_type mode, const void *original_data,
         void *&compressed_data, size_t &compressed_size,
         bool output_pre_allocated) {

  Config config;
  return compress<DeviceType>(D, dtype, shape, tol, s, mode, original_data,
                              compressed_data, compressed_size, config,
                              output_pre_allocated);
}

template <typename DeviceType>
enum compress_status_type
compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol, double s,
         enum error_bound_type mode, const void *original_data,
         void *&compressed_data, size_t &compressed_size,
         std::vector<const Byte *> coords, Config config,
         bool output_pre_allocated) {

  if (dtype == data_type::Float) {
    std::vector<float *> float_coords;
    for (auto &coord : coords)
      float_coords.push_back((float *)coord);
    if (D == 1) {
      return compress<1, float, DeviceType>(
          shape, tol, s, mode, original_data, compressed_data, compressed_size,
          config, float_coords, output_pre_allocated);
    } else if (D == 2) {
      return compress<2, float, DeviceType>(
          shape, tol, s, mode, original_data, compressed_data, compressed_size,
          config, float_coords, output_pre_allocated);
    } else if (D == 3) {
      return compress<3, float, DeviceType>(
          shape, tol, s, mode, original_data, compressed_data, compressed_size,
          config, float_coords, output_pre_allocated);
    } else if (D == 4) {
      return compress<4, float, DeviceType>(
          shape, tol, s, mode, original_data, compressed_data, compressed_size,
          config, float_coords, output_pre_allocated);
    } else if (D == 5) {
      return compress<5, float, DeviceType>(
          shape, tol, s, mode, original_data, compressed_data, compressed_size,
          config, float_coords, output_pre_allocated);
    } else {
      return compress_status_type::NotSupportHigherNumberOfDimensionsFailure;
    }
  } else if (dtype == data_type::Double) {
    std::vector<double *> double_coords;
    for (auto &coord : coords)
      double_coords.push_back((double *)coord);
    if (D == 1) {
      return compress<1, double, DeviceType>(
          shape, tol, s, mode, original_data, compressed_data, compressed_size,
          config, double_coords, output_pre_allocated);
    } else if (D == 2) {
      return compress<2, double, DeviceType>(
          shape, tol, s, mode, original_data, compressed_data, compressed_size,
          config, double_coords, output_pre_allocated);
    } else if (D == 3) {
      return compress<3, double, DeviceType>(
          shape, tol, s, mode, original_data, compressed_data, compressed_size,
          config, double_coords, output_pre_allocated);
    } else if (D == 4) {
      return compress<4, double, DeviceType>(
          shape, tol, s, mode, original_data, compressed_data, compressed_size,
          config, double_coords, output_pre_allocated);
    } else if (D == 5) {
      return compress<5, double, DeviceType>(
          shape, tol, s, mode, original_data, compressed_data, compressed_size,
          config, double_coords, output_pre_allocated);
    } else {
      return compress_status_type::NotSupportHigherNumberOfDimensionsFailure;
    }
  } else {
    return compress_status_type::NotSupportDataTypeFailure;
  }
}

template <typename DeviceType>
enum compress_status_type
compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol, double s,
         enum error_bound_type mode, const void *original_data,
         void *&compressed_data, size_t &compressed_size,
         std::vector<const Byte *> coords, bool output_pre_allocated) {
  Config config;
  return compress<DeviceType>(D, dtype, shape, tol, s, mode, original_data,
                              compressed_data, compressed_size, coords, config,
                              output_pre_allocated);
}

template <typename DeviceType>
enum compress_status_type
decompress(const void *compressed_data, size_t compressed_size,
           void *&decompressed_data, Config config, bool output_pre_allocated) {
  Metadata<DeviceType> meta;
  meta.Deserialize((SERIALIZED_TYPE *)compressed_data);
  load(config, meta);

  std::vector<SIZE> shape(meta.total_dims);
  for (DIM d = 0; d < shape.size(); d++)
    shape[d] = (SIZE)meta.shape[d];
  data_type dtype = meta.dtype;
  if (dtype == data_type::Float) {
    if (shape.size() == 1) {
      return decompress<1, float, DeviceType>(
          shape, compressed_data, compressed_size, decompressed_data, config,
          output_pre_allocated);
    } else if (shape.size() == 2) {
      return decompress<2, float, DeviceType>(
          shape, compressed_data, compressed_size, decompressed_data, config,
          output_pre_allocated);
    } else if (shape.size() == 3) {
      return decompress<3, float, DeviceType>(
          shape, compressed_data, compressed_size, decompressed_data, config,
          output_pre_allocated);
    } else if (shape.size() == 4) {
      return decompress<4, float, DeviceType>(
          shape, compressed_data, compressed_size, decompressed_data, config,
          output_pre_allocated);
    } else if (shape.size() == 5) {
      return decompress<5, float, DeviceType>(
          shape, compressed_data, compressed_size, decompressed_data, config,
          output_pre_allocated);
    } else {
      return compress_status_type::NotSupportHigherNumberOfDimensionsFailure;
    }
  } else if (dtype == data_type::Double) {
    if (shape.size() == 1) {
      enum compress_status_type s = decompress<1, double, DeviceType>(
          shape, compressed_data, compressed_size, decompressed_data, config,
          output_pre_allocated);
      return s;
    } else if (shape.size() == 2) {
      return decompress<2, double, DeviceType>(
          shape, compressed_data, compressed_size, decompressed_data, config,
          output_pre_allocated);
    } else if (shape.size() == 3) {
      return decompress<3, double, DeviceType>(
          shape, compressed_data, compressed_size, decompressed_data, config,
          output_pre_allocated);
    } else if (shape.size() == 4) {
      return decompress<4, double, DeviceType>(
          shape, compressed_data, compressed_size, decompressed_data, config,
          output_pre_allocated);
    } else if (shape.size() == 5) {
      return decompress<5, double, DeviceType>(
          shape, compressed_data, compressed_size, decompressed_data, config,
          output_pre_allocated);
    } else {
      return compress_status_type::NotSupportHigherNumberOfDimensionsFailure;
    }
  } else {
    return compress_status_type::NotSupportDataTypeFailure;
  }
}

template <typename DeviceType>
enum compress_status_type
decompress(const void *compressed_data, size_t compressed_size,
           void *&decompressed_data, bool output_pre_allocated) {
  Config config;
  return decompress<DeviceType>(compressed_data, compressed_size,
                                decompressed_data, config,
                                output_pre_allocated);
}

template <typename DeviceType>
enum compress_status_type decompress(const void *compressed_data,
                                     size_t compressed_size,
                                     void *&decompressed_data, data_type &dtype,
                                     std::vector<mgard_x::SIZE> &shape,
                                     Config config, bool output_pre_allocated) {
  Metadata<DeviceType> meta;
  meta.Deserialize((SERIALIZED_TYPE *)compressed_data);
  load(config, meta);

  shape = std::vector<SIZE>(meta.total_dims);
  for (DIM d = 0; d < shape.size(); d++)
    shape[d] = (SIZE)meta.shape[d];
  dtype = meta.dtype;

  if (dtype == data_type::Float) {
    if (shape.size() == 1) {
      return decompress<1, float, DeviceType>(
          shape, compressed_data, compressed_size, decompressed_data, config,
          output_pre_allocated);
    } else if (shape.size() == 2) {
      return decompress<2, float, DeviceType>(
          shape, compressed_data, compressed_size, decompressed_data, config,
          output_pre_allocated);
    } else if (shape.size() == 3) {
      return decompress<3, float, DeviceType>(
          shape, compressed_data, compressed_size, decompressed_data, config,
          output_pre_allocated);
    } else if (shape.size() == 4) {
      return decompress<4, float, DeviceType>(
          shape, compressed_data, compressed_size, decompressed_data, config,
          output_pre_allocated);
    } else if (shape.size() == 5) {
      return decompress<5, float, DeviceType>(
          shape, compressed_data, compressed_size, decompressed_data, config,
          output_pre_allocated);
    } else {
      return compress_status_type::NotSupportHigherNumberOfDimensionsFailure;
    }
  } else if (dtype == data_type::Double) {
    if (shape.size() == 1) {
      return decompress<1, double, DeviceType>(
          shape, compressed_data, compressed_size, decompressed_data, config,
          output_pre_allocated);
    } else if (shape.size() == 2) {
      return decompress<2, double, DeviceType>(
          shape, compressed_data, compressed_size, decompressed_data, config,
          output_pre_allocated);
    } else if (shape.size() == 3) {
      return decompress<3, double, DeviceType>(
          shape, compressed_data, compressed_size, decompressed_data, config,
          output_pre_allocated);
    } else if (shape.size() == 4) {
      return decompress<4, double, DeviceType>(
          shape, compressed_data, compressed_size, decompressed_data, config,
          output_pre_allocated);
    } else if (shape.size() == 5) {
      return decompress<5, double, DeviceType>(
          shape, compressed_data, compressed_size, decompressed_data, config,
          output_pre_allocated);
    } else {
      return compress_status_type::NotSupportHigherNumberOfDimensionsFailure;
    }
  } else {
    return compress_status_type::NotSupportDataTypeFailure;
  }
}

template <typename DeviceType>
enum compress_status_type
decompress(const void *compressed_data, size_t compressed_size,
           void *&decompressed_data, data_type &dtype,
           std::vector<mgard_x::SIZE> &shape, bool output_pre_allocated) {
  Config config;
  return decompress<DeviceType>(compressed_data, compressed_size,
                                decompressed_data, dtype, shape, config,
                                output_pre_allocated);
}

template <DIM D, typename T, typename DeviceType> void release_cache() {
  using Cache1 =
      CompressorCache<D, T, DeviceType, Compressor<D, T, DeviceType>>;
  Cache1::cache.SafeRelease();
  using Cache2 = CompressorCache<D, T, DeviceType,
                                 HybridHierarchyCompressor<D, T, DeviceType>>;
  Cache2::cache.SafeRelease();
#if MGARD_ENABLE_EXTERNAL_COMPRESSOR
  using Cache3 =
      CompressorCache<D, T, DeviceType, zfp::Compressor<D, T, DeviceType>>;
  Cache3::cache.SafeRelease();
#endif
}

template <typename T, typename DeviceType> void release_cache() {
  release_cache<1, T, DeviceType>();
  release_cache<2, T, DeviceType>();
  release_cache<3, T, DeviceType>();
  release_cache<4, T, DeviceType>();
  release_cache<5, T, DeviceType>();
}

template <typename DeviceType> enum compress_status_type release_cache() {
  release_cache<float, DeviceType>();
  release_cache<double, DeviceType>();
  return compress_status_type::Success;
}

template <typename DeviceType> void pin_memory(void *ptr, SIZE num_bytes) {
  MemoryManager<DeviceType>::HostRegister((Byte *)ptr, num_bytes);
}

template <typename DeviceType> bool check_memory_pinned(void *ptr) {
  return MemoryManager<DeviceType>::CheckHostRegister((Byte *)ptr);
}

template <typename DeviceType> void unpin_memory(void *ptr) {
  MemoryManager<DeviceType>::HostUnregister((Byte *)ptr);
}

} // namespace mgard_x

#endif