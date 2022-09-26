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

#include "../Config/Config.h"
#include "../Hierarchy/Hierarchy.hpp"
#include "../RuntimeX/RuntimeX.h"

#include "../CompressionLowLevel/CompressionLowLevel.h"
#include "../CompressionLowLevel/NormCalculator.hpp"

#include "Metadata.hpp"
#include "DomainDecomposer.hpp"
#include "compress_x.hpp"

#ifndef MGARD_X_COMPRESSION_HIGH_LEVEL_API_HPP
#define MGARD_X_COMPRESSION_HIGH_LEVEL_API_HPP

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
T calc_subdomain_norm_series_w_prefetch(DomainDecomposer<D, T, DeviceType>& domain_decomposer,
                               std::vector<SIZE>& subdomain_ids, T s) {
  assert(subdomain_ids.size() > 0);
  Timer timer_series;
  if (log::level & log::TIME) timer_series.start();

  DeviceRuntime<DeviceType>::SyncQueue(0);
  Array<1, T, DeviceType> norm_array({1});
  SubArray<1, T, DeviceType> norm_subarray(norm_array);
  T norm = 0;

  // Two buffers one for current and one for next
  Array<D, T, DeviceType> device_subdomain_buffer[2];
  // Pre-allocate to the size of the first subdomain
  // Following subdomains should be no bigger than the first one
  // We shouldn't need to reallocate in the future
  device_subdomain_buffer[0].resize(domain_decomposer.subdomain_shape(subdomain_ids[0]));
  if (subdomain_ids.size() > 1) {
    device_subdomain_buffer[1].resize(domain_decomposer.subdomain_shape(subdomain_ids[0]));
  }

  // Pre-fetch the first subdomain to one buffer
  int current_buffer = 0;
  domain_decomposer.copy_subdomain(device_subdomain_buffer[current_buffer], subdomain_ids[0], ORIGINAL_TO_SUBDOMAIN, 0);
  

  for (SIZE i = 0; i < subdomain_ids.size(); i++) {
    SIZE curr_subdomain_id = subdomain_ids[i];
    SIZE next_subdomain_id;
    int next_buffer = (current_buffer + 1) % 2;
    if (i+1 < subdomain_ids.size()) {
      // Prefetch the next subdomain
      next_subdomain_id = subdomain_ids[i+1];
      // Copy data
      domain_decomposer.copy_subdomain(device_subdomain_buffer[next_buffer], next_subdomain_id, ORIGINAL_TO_SUBDOMAIN, 1);
    }

    // device_input_buffer[dev_id] has to be not pitched to avoid copy for linearization
    assert(!device_subdomain_buffer[current_buffer].isPitched());
    // Disable normalize_coordinate since we do not want to void dividing
    // total_elems
    T local_norm =
        norm_calculator(device_subdomain_buffer[current_buffer], SubArray<1, T, DeviceType>(),
                        norm_subarray, s, false);
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

template <DIM D, typename T, typename DeviceType>
T calc_norm_decomposed_w_prefetch(DomainDecomposer<D, T, DeviceType> &domain_decomposer,
                                  T s, bool normalize_coordinates, SIZE total_num_elem,
                                  int adjusted_num_dev) {

  T norm = 0;
  // Set the number of threads equal to the number of devices
  // So that each thread is responsible for one device
#if MGARD_ENABLE_MULTI_DEVICE
  omp_set_num_threads(adjusted_num_dev);
#endif
#pragma omp parallel for
  for (SIZE dev_id = 0; dev_id < adjusted_num_dev; dev_id++) {  
    DeviceRuntime<DeviceType>::SelectDevice(dev_id);
    // Create a series of subdomain ids that are assigned to the current device
    std::vector<SIZE> subdomain_ids;
    for (SIZE subdomain_id = dev_id; subdomain_id < domain_decomposer.num_subdomains(); subdomain_id+=adjusted_num_dev) {
      subdomain_ids.push_back(subdomain_id);
    }
    // Process a series of subdomains according to the subdomain id list
    T local_norm = calc_subdomain_norm_series_w_prefetch(domain_decomposer,subdomain_ids, s);
    #pragma omp critical
    {
      if (s == std::numeric_limits<T>::infinity()) {
        norm = std::max(norm, local_norm);
      } else {
        norm += local_norm;
      }
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

template <DIM D, typename T, typename DeviceType>
T calc_norm_decomposed(std::vector<Array<D, T, DeviceType>> &device_input_buffer,
                       DomainDecomposer<D, T, DeviceType> &domain_decomposer,
                       T s, bool normalize_coordinates, SIZE total_num_elem) {
  T norm = 0;
#if MGARD_ENABLE_MULTI_DEVICE
  omp_set_num_threads(domain_decomposer.num_devices());
#endif
#pragma omp parallel for
  for (int subdomain_id = 0; subdomain_id < domain_decomposer.num_subdomains(); subdomain_id++) {
    SIZE dev_id = DeviceRuntime<DeviceType>::GetDevice();
#if MGARD_ENABLE_MULTI_DEVICE
    dev_id = omp_get_thread_num();
    DeviceRuntime<DeviceType>::SelectDevice(dev_id);
#endif
    // Each device pick a subdomain and calculate norm
    domain_decomposer.copy_subdomain(device_input_buffer[dev_id], subdomain_id, ORIGINAL_TO_SUBDOMAIN, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
    Array<1, T, DeviceType> norm_array({1});
    SubArray<1, T, DeviceType> norm_subarray(norm_array);
    // device_input_buffer[dev_id] has to be not pitched to avoid copy for linearization
    assert(!device_input_buffer[dev_id].isPitched());
    // Disable normalize_coordinate since we do not want to void dividing
    // total_elems
    T local_norm =
        norm_calculator(device_input_buffer[dev_id], SubArray<1, T, DeviceType>(),
                        norm_subarray, s, false);
    #pragma omp critical
    {
      if (s == std::numeric_limits<T>::infinity()) {
        norm = std::max(norm, local_norm);
      } else {
        norm += local_norm * local_norm;
      }
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
  config.decomposition = metadata.decomposition;
  config.lossless = metadata.ltype;
  config.huff_dict_size = metadata.huff_dict_size;
  config.huff_block_size = metadata.huff_block_size;
  config.reorder = metadata.reorder;
}

template <DIM D, typename T, typename DeviceType>
bool is_same(Hierarchy<D, T, DeviceType>& hierarchy1, Hierarchy<D, T, DeviceType>& hierarchy2) {
  if (hierarchy1.data_structure() == data_structure_type::Cartesian_Grid_Non_Uniform ||
      hierarchy2.data_structure() == data_structure_type::Cartesian_Grid_Non_Uniform) {
    return false;
  }
  for (DIM d = 0; d < D; d++) {
    if (hierarchy1.level_shape(hierarchy1.l_target(), d) != hierarchy2.level_shape(hierarchy2.l_target(), d)) {
      return false;
    }
  }
  return true;
}

template <DIM D, typename T, typename DeviceType>
void compress_subdomain(DomainDecomposer<D, T, DeviceType>& domain_decomposer,
                       SIZE subdomain_id,
                       std::vector<Hierarchy<D, T, DeviceType>>& subdomain_hierarchies,
                       T local_tol, T s, T &norm, enum error_bound_type local_ebtype, Config &config,
                       std::vector<Byte*>& compressed_subdomain_data,
                       std::vector<SIZE>& compressed_subdomain_size) {

  Timer timer_series;
  if (log::level & log::TIME) timer_series.start();
  Array<D, T, DeviceType> device_subdomain_buffer;
  Array<1, Byte, DeviceType> device_compressed_buffer;
  // Copy data    
  domain_decomposer.copy_subdomain(device_subdomain_buffer, subdomain_id, ORIGINAL_TO_SUBDOMAIN, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  // Trigger the copy constructor to copy hierarchy to the current device
  Hierarchy<D, T, DeviceType> hierarchy = subdomain_hierarchies[subdomain_id];
  CompressionLowLevelWorkspace workspace(hierarchy, 0.1);
  std::stringstream ss;
  for (DIM d = 0; d < D; d++) {
    ss << hierarchy.level_shape(hierarchy.l_target(), d) << " ";
  }
  log::info("Compressing subdomain " + std::to_string(subdomain_id) +
            " with shape: " + ss.str());
  compress<D, T, DeviceType>(hierarchy, device_subdomain_buffer, local_ebtype,
                             local_tol, s, norm, config, workspace, device_compressed_buffer);
  MemoryManager<DeviceType>::MallocHost(compressed_subdomain_data[subdomain_id],
                                        device_compressed_buffer.shape(0));
  MemoryManager<DeviceType>::Copy1D(compressed_subdomain_data[subdomain_id], 
                                    device_compressed_buffer.data(), device_compressed_buffer.shape(0));
  compressed_subdomain_size[subdomain_id] = device_compressed_buffer.shape(0);
  if (log::level & log::TIME) {
    timer_series.end();
    timer_series.print("Compress single subdomain");
    timer_series.clear();
  }
  DeviceRuntime<DeviceType>::SyncDevice();
}

template <DIM D, typename T, typename DeviceType>
void compress_subdomain_series(DomainDecomposer<D, T, DeviceType>& domain_decomposer,
                               std::vector<SIZE>& subdomain_ids,
                               std::vector<Hierarchy<D, T, DeviceType>>& subdomain_hierarchies,
                               T local_tol, T s, T &norm, enum error_bound_type local_ebtype, Config &config,
                               std::vector<Byte*>& compressed_subdomain_data,
                               std::vector<SIZE>& compressed_subdomain_size) {
  assert(subdomain_ids.size() > 0);
  assert(subdomain_hierarchies.size() > 0);
  Timer timer_series;
  if (log::level & log::TIME) timer_series.start();
  Array<D, T, DeviceType> device_subdomain_buffer;
  Array<1, Byte, DeviceType> device_compressed_buffer;
  for (SIZE i = 0; i < subdomain_ids.size(); i++) {
    SIZE subdomain_id = subdomain_ids[i];
    // Copy data    
    domain_decomposer.copy_subdomain(device_subdomain_buffer, subdomain_id, ORIGINAL_TO_SUBDOMAIN, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
    // Trigger the copy constructor to copy hierarchy to the current device
    Hierarchy<D, T, DeviceType> hierarchy = subdomain_hierarchies[subdomain_id];
    CompressionLowLevelWorkspace workspace(hierarchy, 0.1);
    std::stringstream ss;
    for (DIM d = 0; d < D; d++) {
      ss << hierarchy.level_shape(hierarchy.l_target(), d) << " ";
    }
    log::info("Compressing subdomain " + std::to_string(subdomain_id) +
              " with shape: " + ss.str());
    if (local_ebtype == error_bound_type::REL) {
      norm = norm_calculator(device_subdomain_buffer, workspace.norm_tmp_subarray,
                             workspace.norm_subarray, s,
                             config.normalize_coordinates);
    }

    Decompose(hierarchy, device_subdomain_buffer, config, workspace.data_refactoring_workspace, 0);

    LinearQuanziation(hierarchy, device_subdomain_buffer,
                      config, local_ebtype, local_tol, s, norm, workspace, 0);

    LosslessCompress(hierarchy, device_compressed_buffer, config, workspace);

    MemoryManager<DeviceType>::MallocHost(compressed_subdomain_data[subdomain_id],
                                          device_compressed_buffer.shape(0));
    MemoryManager<DeviceType>::Copy1D(compressed_subdomain_data[subdomain_id], 
                                      device_compressed_buffer.data(), device_compressed_buffer.shape(0));
    compressed_subdomain_size[subdomain_id] = device_compressed_buffer.shape(0);
  }
  if (log::level & log::TIME) {
    timer_series.end();
    timer_series.print("Compress subdomains series");
    timer_series.clear();
  }

  DeviceRuntime<DeviceType>::SyncDevice();
}

template <DIM D, typename T, typename DeviceType>
void compress_subdomain_series_w_prefetch(DomainDecomposer<D, T, DeviceType>& domain_decomposer,
                               std::vector<SIZE>& subdomain_ids,
                               std::vector<Hierarchy<D, T, DeviceType>>& subdomain_hierarchies,
                               T local_tol, T s, T &norm, enum error_bound_type local_ebtype, Config &config,
                               std::vector<Byte*>& compressed_subdomain_data,
                               std::vector<SIZE>& compressed_subdomain_size) {
  assert(subdomain_ids.size() > 0);
  assert(subdomain_hierarchies.size() > 0);
  Timer timer_series;
  if (log::level & log::TIME) timer_series.start();
  Hierarchy<D, T, DeviceType> hierarchy = subdomain_hierarchies[subdomain_ids[0]];
  CompressionLowLevelWorkspace workspace(hierarchy, 0.1);

  // Two buffers one for current and one for next
  Array<D, T, DeviceType> device_subdomain_buffer[2];
  Array<1, Byte, DeviceType> device_compressed_buffer;
  // Pre-allocate to the size of the first subdomain
  // Following subdomains should be no bigger than the first one
  // We shouldn't need to reallocate in the future
  device_subdomain_buffer[0].resize(domain_decomposer.subdomain_shape(subdomain_ids[0]));
  device_subdomain_buffer[1].resize(domain_decomposer.subdomain_shape(subdomain_ids[0]));
  device_compressed_buffer.resize({(SIZE)(hierarchy.total_num_elems()*sizeof(T))});

  // Pre-fetch the first subdomain to one buffer
  int current_buffer = 0;
  domain_decomposer.copy_subdomain(device_subdomain_buffer[current_buffer], subdomain_ids[0], ORIGINAL_TO_SUBDOMAIN, 0);
  

  for (SIZE i = 0; i < subdomain_ids.size(); i++) {
    SIZE curr_subdomain_id = subdomain_ids[i];
    SIZE next_subdomain_id;
    int next_buffer = (current_buffer + 1) % 2;
    if (i+1 < subdomain_ids.size()) {
      // Prefetch the next subdomain
      next_subdomain_id = subdomain_ids[i+1];
      // Copy data
      domain_decomposer.copy_subdomain(device_subdomain_buffer[next_buffer], next_subdomain_id, ORIGINAL_TO_SUBDOMAIN, 1);
    }
    std::stringstream ss;
    if (!is_same(hierarchy, subdomain_hierarchies[curr_subdomain_id])) {
      hierarchy = subdomain_hierarchies[curr_subdomain_id];
      workspace = CompressionLowLevelWorkspace(hierarchy, 0.1);
    }
    
    for (DIM d = 0; d < D; d++) {
      ss << hierarchy.level_shape(hierarchy.l_target(), d) << " ";
    }
    log::info("Compressing subdomain " + std::to_string(curr_subdomain_id) +
              " with shape: " + ss.str());
    
    if (local_ebtype == error_bound_type::REL) {
      norm = norm_calculator(device_subdomain_buffer[current_buffer], workspace.norm_tmp_subarray,
                             workspace.norm_subarray, s,
                             config.normalize_coordinates);
    }

    Decompose(hierarchy, device_subdomain_buffer[current_buffer], config, workspace.data_refactoring_workspace, 0);

    LinearQuanziation(hierarchy, device_subdomain_buffer[current_buffer],
                      config, local_ebtype, local_tol, s, norm, workspace, 0);

    LosslessCompress(hierarchy, device_compressed_buffer, config, workspace);

    MemoryManager<DeviceType>::MallocHost(compressed_subdomain_data[curr_subdomain_id],
                                          device_compressed_buffer.shape(0));
    MemoryManager<DeviceType>::Copy1D(compressed_subdomain_data[curr_subdomain_id], 
                                      device_compressed_buffer.data(), device_compressed_buffer.shape(0), 2);
    compressed_subdomain_size[curr_subdomain_id] = device_compressed_buffer.shape(0);

    current_buffer = next_buffer;
    // Make sure prefetch is done
    DeviceRuntime<DeviceType>::SyncQueue(1);
  }
  if (log::level & log::TIME) {
    timer_series.end();
    timer_series.print("Compress subdomains series with prefetch");
    timer_series.clear();
  }

  DeviceRuntime<DeviceType>::SyncDevice();
}


template <DIM D, typename T, typename DeviceType>
void decompress_subdomain(DomainDecomposer<D, T, DeviceType>& domain_decomposer,
                                 SIZE subdomain_id,
                                 std::vector<Hierarchy<D, T, DeviceType>>& subdomain_hierarchies,
                                 T local_tol, T s, T norm, enum error_bound_type local_ebtype, Config &config,
                                 std::vector<Byte*>& compressed_subdomain_data,
                                 std::vector<SIZE>& compressed_subdomain_size) {
  assert(subdomain_hierarchies.size() > 0);
  Timer timer_series;
  if (log::level & log::TIME) timer_series.start();
  Array<D, T, DeviceType> device_subdomain_buffer;
  Array<1, Byte, DeviceType> device_compressed_buffer;
  // Copy data
  device_compressed_buffer = Array<1, Byte, DeviceType>({compressed_subdomain_size[subdomain_id]});
  MemoryManager<DeviceType>::Copy1D(device_compressed_buffer.data(), compressed_subdomain_data[subdomain_id],
                                    compressed_subdomain_size[subdomain_id]);
  MemoryManager<DeviceType>::FreeHost(compressed_subdomain_data[subdomain_id]);

  // Trigger the copy constructor to copy hierarchy to the current device
  Hierarchy<D, T, DeviceType> hierarchy = subdomain_hierarchies[subdomain_id];

  CompressionLowLevelWorkspace workspace(hierarchy, 0.0);
  std::stringstream ss;
  for (DIM d = 0; d < D; d++) {
    ss << hierarchy.level_shape(hierarchy.l_target(), d) << " ";
  }
  log::info("Decompressing subdomain " + std::to_string(subdomain_id) +
            " with shape: " + ss.str());
  decompress<D, T, DeviceType>(
      hierarchy, device_compressed_buffer, local_ebtype, local_tol, s,
      norm, config, workspace, device_subdomain_buffer);

  domain_decomposer.copy_subdomain(device_subdomain_buffer, subdomain_id, SUBDOMAIN_TO_ORIGINAL, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  if (log::level & log::TIME) {
    timer_series.end();
    timer_series.print("Decompress single subdomain");
    timer_series.clear();
  }
  DeviceRuntime<DeviceType>::SyncDevice();
}

template <DIM D, typename T, typename DeviceType>
void decompress_subdomain_series(DomainDecomposer<D, T, DeviceType>& domain_decomposer,
                                 std::vector<SIZE>& subdomain_ids,
                                 std::vector<Hierarchy<D, T, DeviceType>>& subdomain_hierarchies,
                                 T local_tol, T s, T norm, enum error_bound_type local_ebtype, Config &config,
                                 std::vector<Byte*>& compressed_subdomain_data,
                                 std::vector<SIZE>& compressed_subdomain_size) {
  assert(subdomain_ids.size() > 0);
  assert(subdomain_hierarchies.size() > 0);
  Timer timer_series;
  if (log::level & log::TIME) timer_series.start();
  Array<D, T, DeviceType> device_subdomain_buffer;
  Array<1, Byte, DeviceType> device_compressed_buffer;
  for (SIZE i = 0; i < subdomain_ids.size(); i++) {
    SIZE subdomain_id = subdomain_ids[i];
    // Copy data
    device_compressed_buffer = Array<1, Byte, DeviceType>({compressed_subdomain_size[subdomain_id]});
    MemoryManager<DeviceType>::Copy1D(device_compressed_buffer.data(), compressed_subdomain_data[subdomain_id],
                                      compressed_subdomain_size[subdomain_id]);
    MemoryManager<DeviceType>::FreeHost(compressed_subdomain_data[subdomain_id]);

    // Trigger the copy constructor to copy hierarchy to the current device
    Hierarchy<D, T, DeviceType> hierarchy = subdomain_hierarchies[subdomain_id];

    CompressionLowLevelWorkspace workspace(hierarchy, 0.0);
    std::stringstream ss;
    for (DIM d = 0; d < D; d++) {
      ss << hierarchy.level_shape(hierarchy.l_target(), d) << " ";
    }
    log::info("Decompressing subdomain " + std::to_string(subdomain_id) +
              " with shape: " + ss.str());
    LosslessDecompress(hierarchy, device_compressed_buffer, config, workspace);

    LinearDequanziation(hierarchy, device_subdomain_buffer, config, local_ebtype, local_tol, s, norm,
                        workspace, 0);
    Recompose(hierarchy, device_subdomain_buffer, config, workspace.data_refactoring_workspace, 0);

    domain_decomposer.copy_subdomain(device_subdomain_buffer, subdomain_id, SUBDOMAIN_TO_ORIGINAL, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
  }
  if (log::level & log::TIME) {
    timer_series.end();
    timer_series.print("Decompress subdomains series");
    timer_series.clear();
  }
  DeviceRuntime<DeviceType>::SyncDevice();
}

template <DIM D, typename T, typename DeviceType>
void decompress_subdomain_series_w_prefetch(DomainDecomposer<D, T, DeviceType>& domain_decomposer,
                                 std::vector<SIZE>& subdomain_ids,
                                 std::vector<Hierarchy<D, T, DeviceType>>& subdomain_hierarchies,
                                 T local_tol, T s, T norm, enum error_bound_type local_ebtype, Config &config,
                                 std::vector<Byte*>& compressed_subdomain_data,
                                 std::vector<SIZE>& compressed_subdomain_size) {
  assert(subdomain_ids.size() > 0);
  assert(subdomain_hierarchies.size() > 0);
  Timer timer_series;
  if (log::level & log::TIME) timer_series.start();
  Hierarchy<D, T, DeviceType> hierarchy = subdomain_hierarchies[subdomain_ids[0]];
  CompressionLowLevelWorkspace workspace(hierarchy, 0.1);
  // Two buffers one for current and one for next
  Array<D, T, DeviceType> device_subdomain_buffer[2];
  Array<1, Byte, DeviceType> device_compressed_buffer[2];
  device_subdomain_buffer[0].resize(domain_decomposer.subdomain_shape(subdomain_ids[0]));
  device_subdomain_buffer[1].resize(domain_decomposer.subdomain_shape(subdomain_ids[0]));
  device_compressed_buffer[0].resize({(SIZE)(hierarchy.total_num_elems()*sizeof(T))});
  device_compressed_buffer[1].resize({(SIZE)(hierarchy.total_num_elems()*sizeof(T))});

  // Pre-fetch the first subdomain on queue 0
  int current_buffer = 0;
  device_compressed_buffer[current_buffer].resize({compressed_subdomain_size[subdomain_ids[0]]});
  MemoryManager<DeviceType>::Copy1D(device_compressed_buffer[current_buffer].data(), compressed_subdomain_data[subdomain_ids[0]],
                                      compressed_subdomain_size[subdomain_ids[0]], 0);

  for (SIZE i = 0; i < subdomain_ids.size(); i++) {
    SIZE curr_subdomain_id = subdomain_ids[i];
    SIZE next_subdomain_id;
    int next_buffer = (current_buffer + 1) % 2;
    if (i+1 < subdomain_ids.size()) {
      // Prefetch the next subdomain
      next_subdomain_id = subdomain_ids[i+1];
      // Copy next compressed data on queue 1
      device_compressed_buffer[next_buffer].resize({compressed_subdomain_size[next_subdomain_id]});
      MemoryManager<DeviceType>::Copy1D(device_compressed_buffer[next_buffer].data(), compressed_subdomain_data[next_subdomain_id],
                                        compressed_subdomain_size[next_subdomain_id], 1);
    }
    
    if (!is_same(hierarchy, subdomain_hierarchies[curr_subdomain_id])) {
      hierarchy = subdomain_hierarchies[curr_subdomain_id];
      workspace = CompressionLowLevelWorkspace(hierarchy, 0.1);
    }

    std::stringstream ss;
    for (DIM d = 0; d < D; d++) {
      ss << hierarchy.level_shape(hierarchy.l_target(), d) << " ";
    }
    log::info("Decompressing subdomain " + std::to_string(curr_subdomain_id) +
              " with shape: " + ss.str());

    LosslessDecompress(hierarchy, device_compressed_buffer[current_buffer], config, workspace);

    LinearDequanziation(hierarchy, device_subdomain_buffer[current_buffer], config, local_ebtype, local_tol, s, norm,
                        workspace, 0);

    if (i > 0) {
      // We delay D2H since since it can delay the D2H in lossless decompession and dequantization 
      int previous_buffer = std::abs((current_buffer - 1) % 2);
      SIZE prev_subdomain_id = subdomain_ids[i-1];
      domain_decomposer.copy_subdomain(device_subdomain_buffer[previous_buffer], prev_subdomain_id, SUBDOMAIN_TO_ORIGINAL, 2);
    }
    Recompose(hierarchy, device_subdomain_buffer[current_buffer], config, workspace.data_refactoring_workspace, 0);

    // Make sure the next subdomain is copied before the next iteration
    DeviceRuntime<DeviceType>::SyncQueue(1);

    current_buffer = next_buffer;

  }

  // Copy the last subdomain
  int previous_buffer = std::abs((current_buffer - 1) % 2);
  SIZE prev_subdomain_id = subdomain_ids[subdomain_ids.size()-1];
  domain_decomposer.copy_subdomain(device_subdomain_buffer[previous_buffer], prev_subdomain_id, SUBDOMAIN_TO_ORIGINAL, 2);

  for (SIZE i = 0; i < subdomain_ids.size(); i++) {
    MemoryManager<DeviceType>::FreeHost(compressed_subdomain_data[subdomain_ids[i]]);
  }
  
  DeviceRuntime<DeviceType>::SyncDevice();
  if (log::level & log::TIME) {
    timer_series.end();
    timer_series.print("Decompress subdomains series with prefetch");
    timer_series.clear();
  }
  DeviceRuntime<DeviceType>::SyncDevice();
}

template <DIM D, typename T, typename DeviceType>
void general_compress(std::vector<SIZE> shape, T tol, T s,
                      enum error_bound_type ebtype, const void *original_data,
                      void *&compressed_data, size_t &compressed_size,
                      Config config, bool uniform, std::vector<T *> coords,
                      bool output_pre_allocated) {

  size_t total_num_elem = 1;
  for (int i = 0; i < D; i++)
    total_num_elem *= shape[i];

  bool previously_pinned = !MemoryManager<DeviceType>::IsDevicePointer((void*)original_data) &&
                            MemoryManager<DeviceType>::CheckHostRegister((void*)original_data);
  if (!previously_pinned) {
    MemoryManager<DeviceType>::HostRegister((void*)original_data,
                                            total_num_elem*sizeof(T));
  }

  Hierarchy<D, T, DeviceType> hierarchy;
  DIM domain_decomposed_dim;
  SIZE domain_decomposed_size;
  SIZE num_subdomains;

  config.apply();

  if (config.num_dev <= 0) {
    log::err("Number of device needs to be greater than 0.");
    exit(-1);
  }
  int adjusted_num_dev =
      std::min(DeviceRuntime<DeviceType>::GetDeviceCount(), config.num_dev);
  if (adjusted_num_dev != config.num_dev) {
    log::info("Using " + std::to_string(adjusted_num_dev) +
              " devices (adjusted from " + std::to_string(config.num_dev) +
              " devices)");
  } else {
    log::info("Using " + std::to_string(adjusted_num_dev) + " devices.");
  }

  Timer timer_total, timer_each;
  if (log::level & log::TIME)
    timer_total.start();
  if (log::level & log::TIME)
    timer_each.start();

  DomainDecomposer<D, T, DeviceType> domain_decomposer((T*)original_data, shape, adjusted_num_dev);

  if (!domain_decomposer.domain_decomposed()) {
    if (uniform) {
      hierarchy = Hierarchy<D, T, DeviceType>(shape, config);
    } else {
      hierarchy = Hierarchy<D, T, DeviceType>(shape, coords, config);
    }
    num_subdomains = 1;
  } else {
    if (uniform) {
      hierarchy = Hierarchy<D, T, DeviceType>(shape, domain_decomposer.domain_decomposed_dim(),
                                              domain_decomposer.domain_decomposed_size(), config);
    } else {
      hierarchy = Hierarchy<D, T, DeviceType>(
          shape, domain_decomposer.domain_decomposed_dim(), domain_decomposer.domain_decomposed_size(), coords, config);
    }
    num_subdomains = domain_decomposer.num_subdomains();
    assert(num_subdomains == hierarchy.hierarchy_chunck.size());
  }

  if (log::level & log::TIME) {
    timer_each.end();
    timer_each.print("Building Hierarchy");
    timer_each.clear();
  }

#ifndef MGARDX_COMPILE_CUDA
  if (config.lossless == lossless_type::Huffman_LZ4) {
    config.lossless = lossless_type::Huffman_Zstd;
  }
#endif

  // Buffers for holding input data and they are reused between subdomains
  std::vector<Array<D, T, DeviceType>> device_input_buffer(adjusted_num_dev);
  // We store compressed data on host in case they are too large for devices
  std::vector<Byte*> compressed_subdomain_data(num_subdomains);
  std::vector<SIZE> compressed_subdomain_size(num_subdomains);
  std::vector<Hierarchy<D, T, DeviceType>> subdomain_hierarchies;
  T norm = 1;
  T local_tol = tol;
  enum error_bound_type local_ebtype;

  if (!hierarchy.domain_decomposed) {
    subdomain_hierarchies.push_back(hierarchy);
    local_tol = tol;
    local_ebtype = ebtype;
  } else {
    log::info("Orignial domain is decomposed into " +
              std::to_string(hierarchy.hierarchy_chunck.size()) +
              " sub-domains");
    
    if (log::level & log::TIME)
      timer_each.start();
    subdomain_hierarchies = hierarchy.hierarchy_chunck;
    if (ebtype == error_bound_type::REL) {
      // norm = calc_norm_decomposed(device_input_buffer, domain_decomposer, s,
      //                             config.normalize_coordinates, total_num_elem);
      norm = calc_norm_decomposed_w_prefetch(domain_decomposer, s, config.normalize_coordinates, 
                                             total_num_elem, adjusted_num_dev);
    }
    local_tol = calc_local_abs_tol(ebtype, norm, tol, s, num_subdomains);
    // Force to use ABS mode when do domain decomposition
    local_ebtype = error_bound_type::ABS;
    // Fast copy for domain decomposition need we disable pitched memory
    // allocation
    MemoryManager<DeviceType>::ReduceMemoryFootprint = true;
    if (log::level & log::TIME) {
      timer_each.end();
      timer_each.print("Calculate norm of decomposed domain");
      timer_each.clear();
    }
  }

  if (log::level & log::TIME)
    timer_each.start();
    // Set the number of threads equal to the number of devices
    // So that each thread is responsible for one device
#if MGARD_ENABLE_MULTI_DEVICE
  omp_set_num_threads(adjusted_num_dev);
#endif
#pragma omp parallel for firstprivate(config)
  for (SIZE dev_id = 0; dev_id < adjusted_num_dev; dev_id++) {  
#if MGARD_ENABLE_MULTI_DEVICE
    config.dev_id = dev_id;
#endif
    DeviceRuntime<DeviceType>::SelectDevice(config.dev_id);
    // Create a series of subdomain ids that are assigned to the current device
    std::vector<SIZE> subdomain_ids;
    for (SIZE subdomain_id = dev_id; subdomain_id < num_subdomains; subdomain_id+=adjusted_num_dev) {
      subdomain_ids.push_back(subdomain_id);
    }
    if (subdomain_ids.size() == 1) {
      compress_subdomain(domain_decomposer, subdomain_ids[0], subdomain_hierarchies,
                              local_tol, s, norm, local_ebtype, config,
                              compressed_subdomain_data, compressed_subdomain_size);
    } else {
      // Compress a series of subdomains according to the subdomain id list
      // compress_subdomain_series(domain_decomposer, subdomain_ids, subdomain_hierarchies,
      //                           local_tol, s, norm, local_ebtype, config,
      //                           compressed_subdomain_data, compressed_subdomain_size);
      compress_subdomain_series_w_prefetch(domain_decomposer, subdomain_ids, subdomain_hierarchies,
                                local_tol, s, norm, local_ebtype, config,
                                compressed_subdomain_data, compressed_subdomain_size);
    }
  }

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
  // Initialize metadata
  Metadata<DeviceType> m;
  if (uniform) {
    m.Fill(ebtype, tol, s, norm, config.decomposition, config.reorder,
           config.lossless, config.huff_dict_size, config.huff_block_size,
           shape, hierarchy.domain_decomposed, hierarchy.domain_decomposed_dim,
           hierarchy.domain_decomposed_size);
  } else {
    m.Fill(ebtype, tol, s, norm, config.decomposition, config.reorder,
           config.lossless, config.huff_dict_size, config.huff_block_size,
           shape, hierarchy.domain_decomposed, hierarchy.domain_decomposed_dim,
           hierarchy.domain_decomposed_size, coords);
  }

  uint32_t metadata_size;
  SERIALIZED_TYPE *serizalied_meta = m.Serialize(metadata_size);

  // Estimate final output size
  SIZE byte_offset = 0;
  advance_with_align<SERIALIZED_TYPE>(byte_offset, metadata_size);
  for (uint32_t i = 0; i < num_subdomains; i++) {
    advance_with_align<SIZE>(byte_offset, 1);
    align_byte_offset<uint64_t>(byte_offset); // for zero copy when deserialize
    advance_with_align<Byte>(byte_offset, compressed_subdomain_size[i]);
  }

  compressed_size = byte_offset;

  // Use consistance memory space between input and output data
  if (!output_pre_allocated) {
    if (MemoryManager<DeviceType>::IsDevicePointer(original_data)) {
      DeviceRuntime<DeviceType>::SelectDevice(
          MemoryManager<DeviceType>::GetPointerDevice(original_data));
      MemoryManager<DeviceType>::Malloc1D(compressed_data, compressed_size);
    } else {
      compressed_data = (unsigned char *)malloc(compressed_size);
    }
  }

  // Serialize
  byte_offset = 0;
  Serialize<SERIALIZED_TYPE, DeviceType>(
      (Byte *)compressed_data, serizalied_meta, metadata_size, byte_offset);
  for (uint32_t i = 0; i < num_subdomains; i++) {
    SIZE subdomain_compressed_size = compressed_subdomain_size[i];
    Serialize<SIZE, DeviceType>((Byte *)compressed_data,
                                &subdomain_compressed_size, 1, byte_offset);
    align_byte_offset<uint64_t>(byte_offset);
    Serialize<Byte, DeviceType>((Byte *)compressed_data,
                                compressed_subdomain_data[i],
                                subdomain_compressed_size, byte_offset);
    MemoryManager<DeviceType>::FreeHost(compressed_subdomain_data[i]);
  }
  MemoryManager<DeviceType>::Free(serizalied_meta);

  if (!previously_pinned) {
    MemoryManager<DeviceType>::HostUnregister((void*)original_data);
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
}

template <DIM D, typename T, typename DeviceType>
void compress(std::vector<SIZE> shape, T tol, T s, enum error_bound_type ebtype,
              const void *original_data, void *&compressed_data,
              size_t &compressed_size, Config config,
              bool output_pre_allocated) {

  general_compress<D, T, DeviceType>(
      shape, tol, s, ebtype, original_data, compressed_data, compressed_size,
      config, true, std::vector<T *>(0), output_pre_allocated);
}

template <DIM D, typename T, typename DeviceType>
void compress(std::vector<SIZE> shape, T tol, T s, enum error_bound_type ebtype,
              const void *original_data, void *&compressed_data,
              size_t &compressed_size, Config config, std::vector<T *> coords,
              bool output_pre_allocated) {
  general_compress<D, T, DeviceType>(shape, tol, s, ebtype, original_data,
                                     compressed_data, compressed_size, config,
                                     false, coords, output_pre_allocated);
}

template <DIM D, typename T, typename DeviceType>
void decompress(std::vector<SIZE> shape, const void *compressed_data,
                size_t compressed_size, void *&decompressed_data, Config config,
                bool output_pre_allocated) {
  size_t total_num_elem = 1;
  for (int i = 0; i < D; i++)
    total_num_elem *= shape[i];

  if (config.num_dev <= 0) {
    log::err("Number of device needs to be greater than 0.");
    exit(-1);
  }

  int adjusted_num_dev =
      std::min(DeviceRuntime<DeviceType>::GetDeviceCount(), config.num_dev);
  if (adjusted_num_dev != config.num_dev) {
    log::info("Using " + std::to_string(adjusted_num_dev) +
              " devices (adjusted from " + std::to_string(config.num_dev) +
              " devices)");
  } else {
    log::info("Using " + std::to_string(adjusted_num_dev) + " devices.");
  }

  config.apply();

  Timer timer_total, timer_each;
  if (log::level & log::TIME)
    timer_total.start();
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

  if (log::level & log::TIME) {
    timer_each.end();
    timer_each.print("Deserialize metadata");
    timer_each.clear();
  }

  if (log::level & log::TIME)
    timer_each.start();

  Hierarchy<D, T, DeviceType> hierarchy;
  SIZE num_subdomains;

  if (!m.domain_decomposed) {
    if (m.dstype == data_structure_type::Cartesian_Grid_Uniform) {
      hierarchy = Hierarchy<D, T, DeviceType>(shape, config);
    } else {
      hierarchy = Hierarchy<D, T, DeviceType>(shape, coords, config);
    }
    num_subdomains = 1;
  } else {
    if (m.dstype == data_structure_type::Cartesian_Grid_Uniform) {
      hierarchy = Hierarchy<D, T, DeviceType>(shape, m.domain_decomposed_dim,
                                              m.domain_decomposed_size, config);
    } else {
      hierarchy =
          Hierarchy<D, T, DeviceType>(shape, m.domain_decomposed_dim,
                                      m.domain_decomposed_size, coords, config);
    }
    num_subdomains =
        (shape[m.domain_decomposed_dim] - 1) / m.domain_decomposed_size + 1;
    assert(num_subdomains == hierarchy.hierarchy_chunck.size());
  }

  if (m.dstype == data_structure_type::Cartesian_Grid_Non_Uniform) {
    for (DIM d = 0; d < D; d++)
      delete[] coords[d];
  }

  if (log::level & log::TIME) {
    timer_each.end();
    timer_each.print("Building Hierarchy");
    timer_each.clear();
  }

  // Preparing decompression parameters
  std::vector<Hierarchy<D, T, DeviceType>> subdomain_hierarchies;
  T local_tol;
  enum error_bound_type local_ebtype;

  if (log::level & log::TIME)
    timer_each.start();

  if (!m.domain_decomposed) {
    subdomain_hierarchies.push_back(hierarchy);
    local_tol = m.tol;
    local_ebtype = m.ebtype;
  } else {
    subdomain_hierarchies = hierarchy.hierarchy_chunck;
    log::info("Orignial domain was decomposed into " +
              std::to_string(subdomain_hierarchies.size()) +
              " subdomains during compression");
    local_tol =
        calc_local_abs_tol(m.ebtype, m.norm, m.tol, m.s, num_subdomains);
    // Force to use ABS mode when do domain decomposition
    local_ebtype = error_bound_type::ABS;
    // Fast copy for domain decomposition need we disable pitched memory
    // allocation
    MemoryManager<DeviceType>::ReduceMemoryFootprint = true;
  }

  // Deserialize compressed data
  std::vector<Byte*> compressed_subdomain_data(num_subdomains);
  std::vector<SIZE> compressed_subdomain_size(num_subdomains);
  SIZE byte_offset = m.metadata_size;
  for (uint32_t i = 0; i < subdomain_hierarchies.size(); i++) {
    DeviceRuntime<DeviceType>::SelectDevice(i % adjusted_num_dev);
    SIZE temp_size;
    Byte *temp_data;
    Deserialize<SIZE, DeviceType>((Byte *)compressed_data, &temp_size, 1,
                                  byte_offset);
    MemoryManager<DeviceType>::MallocHost(temp_data, temp_size);
    align_byte_offset<uint64_t>(byte_offset);
    Deserialize<Byte, DeviceType>((Byte *)compressed_data, temp_data, temp_size,
                                  byte_offset);
    compressed_subdomain_data[i] = temp_data;
    compressed_subdomain_size[i] = temp_size;
  }

  if (log::level & log::TIME) {
    timer_each.end();
    timer_each.print("Deserialize data");
    timer_each.clear();
  }

  // Use consistance memory space between input and output data
  if (!output_pre_allocated) {
    if (log::level & log::TIME)
      timer_each.start();
    if (MemoryManager<DeviceType>::IsDevicePointer(compressed_data)) {
      DeviceRuntime<DeviceType>::SelectDevice(
          MemoryManager<DeviceType>::GetPointerDevice(compressed_data));
      MemoryManager<DeviceType>::Malloc1D(decompressed_data,
                                          total_num_elem * sizeof(T));
    } else {
       MemoryManager<DeviceType>::MallocHost(decompressed_data, total_num_elem * sizeof(T));
    }
    if (log::level & log::TIME) {
      timer_each.end();
      timer_each.print("Prepare output buffer");
      timer_each.clear();
    }
  }

  // Initialize DomainDecomposer
  DomainDecomposer<D, T, DeviceType> domain_decomposer((T*)decompressed_data, shape, adjusted_num_dev,
                                                      m.domain_decomposed, m.domain_decomposed_dim, m.domain_decomposed_size);

  if (log::level & log::TIME)
    timer_each.start();
    // decompress
    // Set the number of threads equal to the number of devices
    // So that each thread is responsible for one device
#if MGARD_ENABLE_MULTI_DEVICE
  omp_set_num_threads(adjusted_num_dev);
#endif
#pragma omp parallel for firstprivate(config)
  for (SIZE dev_id = 0; dev_id < adjusted_num_dev; dev_id++) {  
#if MGARD_ENABLE_MULTI_DEVICE
    config.dev_id = dev_id;
#endif
    DeviceRuntime<DeviceType>::SelectDevice(config.dev_id);
    // Create a series of subdomain ids that are assigned to the current device
    std::vector<SIZE> subdomain_ids;
    for (SIZE subdomain_id = dev_id; subdomain_id < num_subdomains; subdomain_id+=adjusted_num_dev) {
      subdomain_ids.push_back(subdomain_id);
    }
    if (subdomain_ids.size() == 1) {
      decompress_subdomain(domain_decomposer, subdomain_ids[0], subdomain_hierarchies,
                              local_tol, (T)m.s, (T)m.norm, local_ebtype, config,
                              compressed_subdomain_data, compressed_subdomain_size);
    } else {
      // Decompress a series of subdomains according to the subdomain id list
      // decompress_subdomain_series(domain_decomposer, subdomain_ids, subdomain_hierarchies,
      //                           local_tol, (T)m.s, (T)m.norm, local_ebtype, config,
      //                           compressed_subdomain_data, compressed_subdomain_size);
      decompress_subdomain_series_w_prefetch(domain_decomposer, subdomain_ids, subdomain_hierarchies,
                                local_tol, (T)m.s, (T)m.norm, local_ebtype, config,
                                compressed_subdomain_data, compressed_subdomain_size);
    }
  }

  if (log::level & log::TIME) {
    timer_each.end();
    timer_each.print("Aggregated low-level decompression");
    log::time("Aggregated high-level decompression throughput: " +
              std::to_string((double)(total_num_elem * sizeof(T)) /
                             timer_each.get() / 1e9) +
              " GB/s");
    timer_each.clear();
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
}

template <typename DeviceType>
void compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol,
              double s, enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size, Config config,
              bool output_pre_allocated) {
  if (dtype == data_type::Float) {
    if (D == 1) {
      compress<1, float, DeviceType>(shape, tol, s, mode, original_data,
                                     compressed_data, compressed_size, config,
                                     output_pre_allocated);
    } else if (D == 2) {
      compress<2, float, DeviceType>(shape, tol, s, mode, original_data,
                                     compressed_data, compressed_size, config,
                                     output_pre_allocated);
    } else if (D == 3) {
      compress<3, float, DeviceType>(shape, tol, s, mode, original_data,
                                     compressed_data, compressed_size, config,
                                     output_pre_allocated);
    } else if (D == 4) {
      compress<4, float, DeviceType>(shape, tol, s, mode, original_data,
                                     compressed_data, compressed_size, config,
                                     output_pre_allocated);
    } else if (D == 5) {
      compress<5, float, DeviceType>(shape, tol, s, mode, original_data,
                                     compressed_data, compressed_size, config,
                                     output_pre_allocated);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else if (dtype == data_type::Double) {
    if (D == 1) {
      compress<1, double, DeviceType>(shape, tol, s, mode, original_data,
                                      compressed_data, compressed_size, config,
                                      output_pre_allocated);
    } else if (D == 2) {
      compress<2, double, DeviceType>(shape, tol, s, mode, original_data,
                                      compressed_data, compressed_size, config,
                                      output_pre_allocated);
    } else if (D == 3) {
      compress<3, double, DeviceType>(shape, tol, s, mode, original_data,
                                      compressed_data, compressed_size, config,
                                      output_pre_allocated);
    } else if (D == 4) {
      compress<4, double, DeviceType>(shape, tol, s, mode, original_data,
                                      compressed_data, compressed_size, config,
                                      output_pre_allocated);
    } else if (D == 5) {
      compress<5, double, DeviceType>(shape, tol, s, mode, original_data,
                                      compressed_data, compressed_size, config,
                                      output_pre_allocated);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else {
    log::err("do not support types other than double and float!");
    exit(-1);
  }
}

template <typename DeviceType>
void compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol,
              double s, enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size,
              bool output_pre_allocated) {

  Config config;
  compress<DeviceType>(D, dtype, shape, tol, s, mode, original_data,
                       compressed_data, compressed_size, config,
                       output_pre_allocated);
}

template <typename DeviceType>
void compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol,
              double s, enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size,
              std::vector<const Byte *> coords, Config config,
              bool output_pre_allocated) {

  if (dtype == data_type::Float) {
    std::vector<float *> float_coords;
    for (auto &coord : coords)
      float_coords.push_back((float *)coord);
    if (D == 1) {
      compress<1, float, DeviceType>(shape, tol, s, mode, original_data,
                                     compressed_data, compressed_size, config,
                                     float_coords, output_pre_allocated);
    } else if (D == 2) {
      compress<2, float, DeviceType>(shape, tol, s, mode, original_data,
                                     compressed_data, compressed_size, config,
                                     float_coords, output_pre_allocated);
    } else if (D == 3) {
      compress<3, float, DeviceType>(shape, tol, s, mode, original_data,
                                     compressed_data, compressed_size, config,
                                     float_coords, output_pre_allocated);
    } else if (D == 4) {
      compress<4, float, DeviceType>(shape, tol, s, mode, original_data,
                                     compressed_data, compressed_size, config,
                                     float_coords, output_pre_allocated);
    } else if (D == 5) {
      compress<5, float, DeviceType>(shape, tol, s, mode, original_data,
                                     compressed_data, compressed_size, config,
                                     float_coords, output_pre_allocated);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else if (dtype == data_type::Double) {
    std::vector<double *> double_coords;
    for (auto &coord : coords)
      double_coords.push_back((double *)coord);
    if (D == 1) {
      compress<1, double, DeviceType>(shape, tol, s, mode, original_data,
                                      compressed_data, compressed_size, config,
                                      double_coords, output_pre_allocated);
    } else if (D == 2) {
      compress<2, double, DeviceType>(shape, tol, s, mode, original_data,
                                      compressed_data, compressed_size, config,
                                      double_coords, output_pre_allocated);
    } else if (D == 3) {
      compress<3, double, DeviceType>(shape, tol, s, mode, original_data,
                                      compressed_data, compressed_size, config,
                                      double_coords, output_pre_allocated);
    } else if (D == 4) {
      compress<4, double, DeviceType>(shape, tol, s, mode, original_data,
                                      compressed_data, compressed_size, config,
                                      double_coords, output_pre_allocated);
    } else if (D == 5) {
      compress<5, double, DeviceType>(shape, tol, s, mode, original_data,
                                      compressed_data, compressed_size, config,
                                      double_coords, output_pre_allocated);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else {
    log::err("do not support types other than double and float!");
    exit(-1);
  }
}

template <typename DeviceType>
void compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol,
              double s, enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size,
              std::vector<const Byte *> coords, bool output_pre_allocated) {
  Config config;
  compress<DeviceType>(D, dtype, shape, tol, s, mode, original_data,
                       compressed_data, compressed_size, coords, config,
                       output_pre_allocated);
}

template <typename DeviceType>
void decompress(const void *compressed_data, size_t compressed_size,
                void *&decompressed_data, Config config,
                bool output_pre_allocated) {
  Metadata<DeviceType> meta;
  meta.Deserialize((SERIALIZED_TYPE *)compressed_data);

  std::vector<SIZE> shape(meta.total_dims);
  for (DIM d = 0; d < shape.size(); d++)
    shape[d] = (SIZE)meta.shape[d];
  data_type dtype = meta.dtype;
  if (dtype == data_type::Float) {
    if (shape.size() == 1) {
      decompress<1, float, DeviceType>(shape, compressed_data, compressed_size,
                                       decompressed_data, config,
                                       output_pre_allocated);
    } else if (shape.size() == 2) {
      decompress<2, float, DeviceType>(shape, compressed_data, compressed_size,
                                       decompressed_data, config,
                                       output_pre_allocated);
    } else if (shape.size() == 3) {
      decompress<3, float, DeviceType>(shape, compressed_data, compressed_size,
                                       decompressed_data, config,
                                       output_pre_allocated);
    } else if (shape.size() == 4) {
      decompress<4, float, DeviceType>(shape, compressed_data, compressed_size,
                                       decompressed_data, config,
                                       output_pre_allocated);
    } else if (shape.size() == 5) {
      decompress<5, float, DeviceType>(shape, compressed_data, compressed_size,
                                       decompressed_data, config,
                                       output_pre_allocated);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else if (dtype == data_type::Double) {
    if (shape.size() == 1) {
      decompress<1, double, DeviceType>(shape, compressed_data, compressed_size,
                                        decompressed_data, config,
                                        output_pre_allocated);
    } else if (shape.size() == 2) {
      decompress<2, double, DeviceType>(shape, compressed_data, compressed_size,
                                        decompressed_data, config,
                                        output_pre_allocated);
    } else if (shape.size() == 3) {
      decompress<3, double, DeviceType>(shape, compressed_data, compressed_size,
                                        decompressed_data, config,
                                        output_pre_allocated);
    } else if (shape.size() == 4) {
      decompress<4, double, DeviceType>(shape, compressed_data, compressed_size,
                                        decompressed_data, config,
                                        output_pre_allocated);
    } else if (shape.size() == 5) {
      decompress<5, double, DeviceType>(shape, compressed_data, compressed_size,
                                        decompressed_data, config,
                                        output_pre_allocated);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else {
    log::err("do not support types other than double and float!");
    exit(-1);
  }
}

template <typename DeviceType>
void decompress(const void *compressed_data, size_t compressed_size,
                void *&decompressed_data, bool output_pre_allocated) {
  Config config;
  decompress<DeviceType>(compressed_data, compressed_size, decompressed_data,
                         config, output_pre_allocated);
}

template <typename DeviceType>
void decompress(const void *compressed_data, size_t compressed_size,
                void *&decompressed_data, data_type &dtype,
                std::vector<mgard_x::SIZE> &shape, Config config,
                bool output_pre_allocated) {
  Metadata<DeviceType> meta;
  meta.Deserialize((SERIALIZED_TYPE *)compressed_data);

  shape = std::vector<SIZE>(meta.total_dims);
  for (DIM d = 0; d < shape.size(); d++)
    shape[d] = (SIZE)meta.shape[d];
  dtype = meta.dtype;

  if (dtype == data_type::Float) {
    if (shape.size() == 1) {
      decompress<1, float, DeviceType>(shape, compressed_data, compressed_size,
                                       decompressed_data, config,
                                       output_pre_allocated);
    } else if (shape.size() == 2) {
      decompress<2, float, DeviceType>(shape, compressed_data, compressed_size,
                                       decompressed_data, config,
                                       output_pre_allocated);
    } else if (shape.size() == 3) {
      decompress<3, float, DeviceType>(shape, compressed_data, compressed_size,
                                       decompressed_data, config,
                                       output_pre_allocated);
    } else if (shape.size() == 4) {
      decompress<4, float, DeviceType>(shape, compressed_data, compressed_size,
                                       decompressed_data, config,
                                       output_pre_allocated);
    } else if (shape.size() == 5) {
      decompress<5, float, DeviceType>(shape, compressed_data, compressed_size,
                                       decompressed_data, config,
                                       output_pre_allocated);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else if (dtype == data_type::Double) {
    if (shape.size() == 1) {
      decompress<1, double, DeviceType>(shape, compressed_data, compressed_size,
                                        decompressed_data, config,
                                        output_pre_allocated);
    } else if (shape.size() == 2) {
      decompress<2, double, DeviceType>(shape, compressed_data, compressed_size,
                                        decompressed_data, config,
                                        output_pre_allocated);
    } else if (shape.size() == 3) {
      decompress<3, double, DeviceType>(shape, compressed_data, compressed_size,
                                        decompressed_data, config,
                                        output_pre_allocated);
    } else if (shape.size() == 4) {
      decompress<4, double, DeviceType>(shape, compressed_data, compressed_size,
                                        decompressed_data, config,
                                        output_pre_allocated);
    } else if (shape.size() == 5) {
      decompress<5, double, DeviceType>(shape, compressed_data, compressed_size,
                                        decompressed_data, config,
                                        output_pre_allocated);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else {
    log::err("do not support types other than double and float!");
    exit(-1);
  }
}

template <typename DeviceType>
void decompress(const void *compressed_data, size_t compressed_size,
                void *&decompressed_data, data_type &dtype,
                std::vector<mgard_x::SIZE> &shape, bool output_pre_allocated) {
  Config config;
  decompress<DeviceType>(compressed_data, compressed_size, decompressed_data,
                         dtype, shape, config, output_pre_allocated);
}

} // namespace mgard_x

#endif