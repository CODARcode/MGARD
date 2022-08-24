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

#include "../Hierarchy/Hierarchy.hpp"
#include "../RuntimeX/RuntimeX.h"
#include "Metadata.hpp"
#include "../Config/Config.hpp"
#include "compress_x.hpp"

// #if MGARD_ENABLE_OPENMP
#include <omp.h>
// #endif

#include "../CompressionLowLevel/CompressionLowLevel.h"

#ifndef MGARD_X_COMPRESSION_HIGH_LEVEL_API_HPP
#define MGARD_X_COMPRESSION_HIGH_LEVEL_API_HPP

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
size_t estimate_memory_usgae(std::vector<SIZE> shape) {
  size_t estimate_memory_usgae = 0;
  size_t total_elem = 1;
  for (DIM d = 0; d < D; d++) {
    total_elem *= shape[d];
  }
  size_t pitch_size = 1;
  if (!MemoryManager<DeviceType>::ReduceMemoryFootprint) { // pitch is enable
    T *dummy;
    SIZE ld;
    MemoryManager<DeviceType>::MallocND(dummy, 1, 1, ld, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
    MemoryManager<DeviceType>::Free(dummy);
    pitch_size = ld * sizeof(T);
  }

  size_t hierarchy_space = 0;
  // space need for hiearachy
  int nlevel = std::numeric_limits<int>::max();
  for (DIM i = 0; i < shape.size(); i++) {
    int n = shape[i];
    int l = 0;
    while (n > 2) {
      n = n / 2 + 1;
      l++;
    }
    nlevel = std::min(nlevel, l);
  }
  nlevel--;
  for (DIM d = 0; d < D; d++) {
    hierarchy_space += shape[d] * 2 * sizeof(T); // dist
    hierarchy_space += shape[d] * 2 * sizeof(T); // ratio
  }
  SIZE max_dim = 0;
  for (DIM d = 0; d < D; d++) {
    max_dim = std::max(max_dim, shape[d]);
  }

  hierarchy_space +=
      D * (nlevel + 1) * roundup(max_dim * sizeof(T), pitch_size); // volume
  for (DIM d = 0; d < D; d++) {
    hierarchy_space += shape[d] * 2 * sizeof(T); // am
    hierarchy_space += shape[d] * 2 * sizeof(T); // bm
  }

  size_t input_space = roundup(shape[D - 1] * sizeof(T), pitch_size);
  for (DIM d = 0; d < D - 1; d++) {
    input_space *= shape[d];
  }

  size_t norm_workspace = roundup(shape[D - 1] * sizeof(T), pitch_size);
  for (DIM d = 0; d < D - 1; d++) {
    norm_workspace *= shape[d];
  }
  estimate_memory_usgae = std::max(
      estimate_memory_usgae, hierarchy_space + input_space + norm_workspace);

  size_t decomposition_workspace =
      roundup(shape[D - 1] * sizeof(T), pitch_size);
  for (DIM d = 0; d < D - 1; d++) {
    decomposition_workspace *= shape[d];
  }
  estimate_memory_usgae =
      std::max(estimate_memory_usgae,
               hierarchy_space + input_space + decomposition_workspace);

  size_t quantization_workspace =
      sizeof(QUANTIZED_INT) * total_elem + // quantized
      sizeof(LENGTH) * total_elem +        // outlier index
      sizeof(QUANTIZED_INT) * total_elem;  // outlier

  estimate_memory_usgae =
      std::max(estimate_memory_usgae,
               hierarchy_space + input_space + quantization_workspace);

  size_t huffman_workspace =
      sizeof(QUANTIZED_INT) *
      total_elem; // fix-length encoding
                  // space taken by codebook generation is ignored

  estimate_memory_usgae = std::max(
      estimate_memory_usgae, hierarchy_space + input_space +
                                 quantization_workspace + huffman_workspace);

  double estimated_output_ratio = 0.7;

  return estimate_memory_usgae + (double)input_space * estimated_output_ratio;
}

template <DIM D, typename T, typename DeviceType>
bool need_domain_decomposition(std::vector<SIZE> shape) {
  return estimate_memory_usgae<D, T, DeviceType>(shape) >=
         DeviceRuntime<DeviceType>::GetAvailableMemory();
}

template <DIM D, typename T, typename DeviceType>
bool generate_domain_decomposition_strategy(std::vector<SIZE> shape, 
                                            DIM &domain_decomposed_dim, 
                                            SIZE &domain_decomposed_size) {

  if (!need_domain_decomposition<D, T, DeviceType>(shape)) {
    return false;
  }
  // determine max dimension
  DIM max_dim = 0;
  for (DIM d = 0; d < D; d++) {
    if (shape[d] > max_dim) {
      max_dim = shape[d];
      domain_decomposed_dim = d;
    }
  }

  // domain decomposition strategy
  std::vector<SIZE> chunck_shape = shape;
  while (need_domain_decomposition<D, T, DeviceType>(chunck_shape)) {
    chunck_shape[domain_decomposed_dim] /= 2;
  }
  domain_decomposed_size = chunck_shape[domain_decomposed_dim];
  return true;
}

template <DIM D, typename T, typename DeviceType>
void domain_decompose(T *data, std::vector<Array<D, T, DeviceType>> &decomposed_data,
                      std::vector<SIZE> shape, DIM domain_decomposed_dim,
                      SIZE domain_decomposed_size) {

  std::vector<SIZE> chunck_shape = shape;
  chunck_shape[domain_decomposed_dim] = domain_decomposed_size;
  SIZE elem_per_chunk = 1;
  for (DIM d = 0; d < chunck_shape.size(); d++) {
    elem_per_chunk *= chunck_shape[d];
  }
  bool pitched = false;

  for (SIZE i = 0; i < shape[domain_decomposed_dim];
       i += domain_decomposed_size) {
    DeviceRuntime<DeviceType>::SelectDevice((i / domain_decomposed_size) % DeviceRuntime<DeviceType>::GetDeviceCount());
    Array<D, T, DeviceType> subdomain_array({chunck_shape}, pitched);
    SIZE dst_ld = subdomain_array.ld(D-1);//domain_decomposed_size;
    if (domain_decomposed_dim <= D-2) {
      for (int d = D-1; d > domain_decomposed_dim; d--) {
        dst_ld *= shape[d];
      }
    }

    SIZE src_ld = 1;
    for (int d = D-1; d > domain_decomposed_dim - 1; d--) {
      src_ld *= shape[d];
    }


    SIZE n1 = domain_decomposed_size;
    SIZE n2 = 1;
     for (int d = domain_decomposed_dim - 1; d >= 0; d--) {
      n2 *= shape[d];
    }

    T *src_ptr = data + n1 * (i / domain_decomposed_size);
    MemoryManager<DeviceType>::CopyND(subdomain_array.data(), dst_ld, src_ptr, src_ld, n1,
                                      n2, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
    decomposed_data.push_back(subdomain_array);
  }
  SIZE leftover_dim_size =
      shape[domain_decomposed_dim] % domain_decomposed_size;
  if (leftover_dim_size != 0) {
    std::vector<SIZE> leftover_shape = shape;
    leftover_shape[domain_decomposed_dim] = leftover_dim_size;
    SIZE elem_leftover = 1;
    for (DIM d = 0; d < leftover_shape.size(); d++) {
      elem_leftover *= leftover_shape[d];
    }

    Array<D, T, DeviceType> subdomain_array({leftover_shape}, pitched);

    SIZE dst_ld = subdomain_array.ld(D-1);
    if (domain_decomposed_dim <= D-2) {
      for (int d = D-1; d > domain_decomposed_dim; d--) {
        dst_ld *= shape[d];
      }
    }
    SIZE src_ld = 1;
    for (int d = D-1; d > domain_decomposed_dim - 1; d--) {
      src_ld *= shape[d];
    }
    SIZE n1 = leftover_dim_size;
    SIZE n2 = 1;
    for (int d = domain_decomposed_dim - 1; d >= 0; d--) {
      n2 *= shape[d];
    }

    T *src_ptr = data + n1 * decomposed_data.size();

    MemoryManager<DeviceType>::CopyND(subdomain_array.data(), dst_ld, src_ptr, src_ld,
                                      n1, n2, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
    decomposed_data.push_back(subdomain_array);
  }
}

template <DIM D, typename T, typename DeviceType>
void domain_recompose(std::vector<Array<D, T, DeviceType>>& decomposed_data, T *data,
                      std::vector<SIZE> shape, DIM domain_decomposed_dim,
                      SIZE domain_decomposed_size) {
  std::vector<SIZE> chunck_shape = shape;
  chunck_shape[domain_decomposed_dim] = domain_decomposed_size;
  SIZE elem_per_chunk = 1;
  for (DIM d = 0; d < chunck_shape.size(); d++) {
    elem_per_chunk *= chunck_shape[d];
  }
  for (SIZE i = 0; i < shape[domain_decomposed_dim];
       i += domain_decomposed_size) {
    Array<D, T, DeviceType> &chunck_data = decomposed_data[i / domain_decomposed_size];

    SIZE dst_ld = chunck_data.ld(D-1);
    if (domain_decomposed_dim <= D-2) {
      for (int d = D-1; d > domain_decomposed_dim; d--) {
        dst_ld *= shape[d];
      }
    }

    SIZE src_ld = 1;
    for (int d = D-1; d > domain_decomposed_dim - 1; d--) {
      src_ld *= shape[d];
    }


    SIZE n1 = domain_decomposed_size;
    SIZE n2 = 1;
     for (int d = domain_decomposed_dim - 1; d >= 0; d--) {
      n2 *= shape[d];
    }

    T *src_ptr = data + n1 * (i / domain_decomposed_size);
    MemoryManager<DeviceType>::CopyND(src_ptr, src_ld, chunck_data.data(), dst_ld, n1,
                                      n2, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
  }
  SIZE leftover_dim_size =
      shape[domain_decomposed_dim] % domain_decomposed_size;
  if (leftover_dim_size != 0) {
    std::vector<SIZE> leftover_shape = shape;
    leftover_shape[domain_decomposed_dim] = leftover_dim_size;
    SIZE elem_leftover = 1;
    for (DIM d = 0; d < leftover_shape.size(); d++) {
      elem_leftover *= leftover_shape[d];
    }
    Array<D, T, DeviceType> &leftover_data = decomposed_data[decomposed_data.size() - 1];

    SIZE dst_ld = leftover_data.ld(D-1);
    if (domain_decomposed_dim <= D-2) {
      for (int d = D-1; d > domain_decomposed_dim; d--) {
        dst_ld *= shape[d];
      }
    }
    SIZE src_ld = 1;
    for (int d = D-1; d > domain_decomposed_dim - 1; d--) {
      src_ld *= shape[d];
    }
    SIZE n1 = leftover_dim_size;
    SIZE n2 = 1;
    for (int d = domain_decomposed_dim - 1; d >= 0; d--) {
      n2 *= shape[d];
    }

    T *src_ptr = data + n1 * decomposed_data.size();
    MemoryManager<DeviceType>::CopyND(src_ptr, src_ld, leftover_data.data(), dst_ld,
                                      n1, n2, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
  }
}

template <DIM D, typename T, typename DeviceType>
T calc_norm_decomposed(std::vector<Array<D, T, DeviceType>> &decomposed_data, T s) {
  T norm = 0;
  Array<1, T, DeviceType> norm_array({1});
  SubArray<1, T, DeviceType> norm_subarray(norm_array);
  for (int i = 0; i < decomposed_data.size(); i++) {
    SubArray chunck_in_subarray = SubArray(decomposed_data[i]).Linearize();
    if (s == std::numeric_limits<T>::infinity()) {
      DeviceCollective<DeviceType>::AbsMax(chunck_in_subarray.shape(0), chunck_in_subarray,
                                           norm_subarray, 0);
      DeviceRuntime<DeviceType>::SyncQueue(0);
      norm = std::max(norm, norm_array.hostCopy()[0]);
    } else {
      DeviceCollective<DeviceType>::SquareSum(
          chunck_in_subarray.shape(0), chunck_in_subarray, norm_subarray, 0);
      DeviceRuntime<DeviceType>::SyncQueue(0);
      norm += norm_array.hostCopy()[0];
    }
  }
  if (s != std::numeric_limits<T>::infinity()) {
    norm = std::sqrt(norm);
  }
  return norm;
}

template <typename DeviceType>
void load(Config& config, Metadata<DeviceType>& metadata) {
  config.decomposition = metadata.decomposition;
  config.lossless = metadata.ltype;
  config.huff_dict_size = metadata.huff_dict_size;
  config.huff_block_size = metadata.huff_block_size;
  config.reorder = metadata.reorder;
}

template <DIM D, typename T, typename DeviceType>
void general_compress(std::vector<SIZE> shape, T tol, T s, enum error_bound_type type,
              const void *original_data, void *&compressed_data,
              size_t &compressed_size, Config config, bool uniform, std::vector<T *> coords,
              bool output_pre_allocated) {

  size_t original_size = 1;
  for (int i = 0; i < D; i++)
    original_size *= shape[i];

  Hierarchy<D, T, DeviceType> hierarchy;
  DIM domain_decomposed_dim; 
  SIZE domain_decomposed_size;

  config.apply();

  Timer timer_total, timer_each;
  if (log::level & log::TIME) timer_total.start();
  if (log::level & log::TIME) timer_each.start();

  if (!need_domain_decomposition<D, T, DeviceType>(shape)) {
    if (uniform) {
      hierarchy = Hierarchy<D, T, DeviceType>(shape, config.uniform_coord_mode);
    } else {
      hierarchy = Hierarchy<D, T, DeviceType>(shape, coords);
    }
  } else {   
    generate_domain_decomposition_strategy<D, T, DeviceType>(shape, 
                                          domain_decomposed_dim, domain_decomposed_size);
    if (uniform) {
      hierarchy = Hierarchy<D, T, DeviceType>(shape, domain_decomposed_dim, domain_decomposed_size,
                                              config.uniform_coord_mode);
    } else {
      hierarchy = Hierarchy<D, T, DeviceType>(shape, domain_decomposed_dim, domain_decomposed_size,
                                              coords);
    }
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

  std::vector<Array<D, T, DeviceType>> subdomain_data;
  std::vector<Array<1, Byte, DeviceType>> compressed_subdomain_data;
  std::vector<Hierarchy<D, T, DeviceType>> subdomain_hierarchy;
  T norm = 1;
  T local_tol = tol;

  if (!hierarchy.domain_decomposed) {
    if (log::level & log::TIME) timer_each.start();
    Array<D, T, DeviceType> in_array({shape});
    in_array.load((T*)original_data);
    subdomain_data.push_back(in_array);
    subdomain_hierarchy.push_back(hierarchy);
    local_tol = tol;
    if (log::level & log::TIME) {
      timer_each.end();
      timer_each.print("Load data");
      timer_each.clear();
    }
  } else {
    log::info("Orignial domain is decomposed into " +
               std::to_string(hierarchy.hierarchy_chunck.size()) + " sub-domains");
    if (log::level & log::TIME) timer_each.start();
    domain_decompose<D, T, DeviceType>((T *)original_data, subdomain_data,
                                       shape, hierarchy.domain_decomposed_dim,
                                       hierarchy.domain_decomposed_size);
    if (log::level & log::TIME) {
      timer_each.end();
      timer_each.print("Domain decomposition");
      timer_each.clear();
    }

    if (log::level & log::TIME) timer_each.start();
    subdomain_hierarchy = hierarchy.hierarchy_chunck;
    if (type == error_bound_type::REL) {
      norm = calc_norm_decomposed(subdomain_data, s);
      if (s == std::numeric_limits<T>::infinity()) {
        log::info("L_inf norm: " + std::to_string(norm));
      } else {
        log::info("L_2 norm: " + std::to_string(norm));
      }
    }
    if (s != std::numeric_limits<T>::infinity()) {
      local_tol = std::sqrt((tol * tol) / subdomain_data.size());
      log::info("local bound: " + std::to_string(local_tol));
    }
    if (log::level & log::TIME) {
      timer_each.end();
      timer_each.print("Calculate norm of decomposed domain");
      timer_each.clear();
    }
  }
  
  if (log::level & log::TIME) timer_each.start();

  // Set the number of threads equal to the number of devices
  // So that each thread is responsible for one device
  omp_set_num_threads(DeviceRuntime<DeviceType>::GetDeviceCount());
  #pragma omp parallel for schedule(dynamic) private(curr_dev_id)
  for (SIZE i = 0; i < subdomain_data.size(); i++) {
    // Select device based on where is input data is
    DeviceRuntime<DeviceType>::SelectDevice(subdomain_data[i].resideDevice());
    config.dev_id = subdomain_data[i].resideDevice();

    // Trigger the copy constructor to copy hierarchy to corresponding device
    Hierarchy<D, T, DeviceType> hierarchy = subdomain_hierarchy[i];

    std::stringstream ss;
    for (DIM d = 0; d < D; d++)
      ss << subdomain_hierarchy[i].level_shape(
                       subdomain_hierarchy[i].l_target(), d)
                << " ";
    log::info("Compressing decomposed domain " + std::to_string(i) + "/" +
               std::to_string(subdomain_data.size()) + " with shape: " + ss.str() +
               "using thread " + std::to_string(omp_get_thread_num()) + "/" +
               std::to_string(omp_get_num_threads()) + "on device " + std::to_string(subdomain_data[i].resideDevice()));
    Array<1, Byte, DeviceType> compressed_array =
        compress<D, T, DeviceType>(hierarchy,
                                   subdomain_data[i], type, local_tol, s, norm,
                                   config);
    compressed_subdomain_data.push_back(compressed_array);
  }

  if (log::level & log::TIME) {
    timer_each.end();
    timer_each.print("Aggregated low-level compression");
    log::time("Aggregated low-level compression throughput: "
              + std::to_string((double)(original_size * sizeof(T)) / timer_each.get() / 1e9)
              + " GB/s");
    timer_each.clear();
  }

  if (log::level & log::TIME) timer_each.start();
  // Initialize metadata
  Metadata<DeviceType> m;
  if (uniform) {
    m.Fill(type, tol, s, norm, config.decomposition, config.reorder, config.lossless,
           config.huff_dict_size, config.huff_block_size,
           shape, hierarchy.domain_decomposed,
           hierarchy.domain_decomposed_dim, hierarchy.domain_decomposed_size);
  } else {
    m.Fill(type, tol, s, norm, config.decomposition, config.reorder, config.lossless,
           config.huff_dict_size, config.huff_block_size,
           shape, hierarchy.domain_decomposed,
           hierarchy.domain_decomposed_dim, hierarchy.domain_decomposed_size, coords);
  }

  uint32_t metadata_size;
  SERIALIZED_TYPE *serizalied_meta = m.Serialize(metadata_size);

  // Estimate final output size
  SIZE byte_offset = 0;
  advance_with_align<SERIALIZED_TYPE>(byte_offset, metadata_size);
  for (uint32_t i = 0; i < subdomain_data.size(); i++) {
    advance_with_align<SIZE>(byte_offset, 1);
    align_byte_offset<uint64_t>(
        byte_offset); // for zero copy when deserialize
    advance_with_align<Byte>(byte_offset, compressed_subdomain_data[i].shape(0));
  }

  compressed_size = byte_offset;

  // Use consistance memory space between input and output data
  if (!output_pre_allocated) {
    if (MemoryManager<DeviceType>::IsDevicePointer(original_data)) {
      MemoryManager<DeviceType>::Malloc1D(compressed_data, compressed_size);
    } else {
      compressed_data = (unsigned char *)malloc(compressed_size);
    }
  }

  // Serialize
  byte_offset = 0;
  Serialize<SERIALIZED_TYPE, DeviceType>(
      (Byte *)compressed_data, serizalied_meta, metadata_size, byte_offset);
  for (uint32_t i = 0; i < subdomain_data.size(); i++) {
    SIZE subdomain_compressed_size = compressed_subdomain_data[i].shape(0);
    Serialize<SIZE, DeviceType>((Byte *)compressed_data,
                                &subdomain_compressed_size, 1,
                                byte_offset);
    align_byte_offset<uint64_t>(byte_offset);
    Serialize<Byte, DeviceType>((Byte *)compressed_data,
                                compressed_subdomain_data[i].data(),
                                subdomain_compressed_size, byte_offset);
  }
  MemoryManager<DeviceType>::Free(serizalied_meta);

  if (log::level & log::TIME) {
    timer_each.end();
    timer_each.print("Serialization");
    timer_each.clear();
    timer_total.end();
    timer_total.print("High-level compression");
    log::time("High-level compression throughput: "
              + std::to_string((double)(original_size * sizeof(T)) / timer_total.get() / 1e9)
              + " GB/s");
    timer_total.clear();
  }
}

template <DIM D, typename T, typename DeviceType>
void compress(std::vector<SIZE> shape, T tol, T s, enum error_bound_type type,
              const void *original_data, void *&compressed_data,
              size_t &compressed_size, Config config,
              bool output_pre_allocated) {

  general_compress<D, T, DeviceType>(shape, tol, s, type, original_data, compressed_data,
                                     compressed_size, config, true, std::vector<T*>(0), output_pre_allocated);
}

template <DIM D, typename T, typename DeviceType>
void compress(std::vector<SIZE> shape, T tol, T s, enum error_bound_type type,
              const void *original_data, void *&compressed_data,
              size_t &compressed_size, Config config, std::vector<T *> coords,
              bool output_pre_allocated) {
  general_compress<D, T, DeviceType>(shape, tol, s, type, original_data, compressed_data,
                                     compressed_size, config, false, coords, output_pre_allocated);
}

template <DIM D, typename T, typename DeviceType>
void decompress(std::vector<SIZE> shape, const void *compressed_data,
                size_t compressed_size, void *&decompressed_data, Config config,
                bool output_pre_allocated) {
  size_t original_size = 1;
  for (int i = 0; i < D; i++)
    original_size *= shape[i];

  config.apply();

  Timer timer_total, timer_each;
  if (log::level & log::TIME) timer_total.start();
  if (log::level & log::TIME) timer_each.start();

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

  if (log::level & log::TIME) timer_each.start();

  Hierarchy<D, T, DeviceType> hierarchy;

  if (!m.domain_decomposed) {
    if (m.dstype == data_structure_type::Cartesian_Grid_Uniform) {
      hierarchy = Hierarchy<D, T, DeviceType> (shape, config.uniform_coord_mode);
    } else {
      hierarchy = Hierarchy<D, T, DeviceType> (shape, coords);
    }
  } else {
    if (m.dstype == data_structure_type::Cartesian_Grid_Uniform) {
      hierarchy = Hierarchy<D, T, DeviceType>(shape, m.domain_decomposed_dim,
                                            m.domain_decomposed_size,
                                            config.uniform_coord_mode);
    } else {
      hierarchy = Hierarchy<D, T, DeviceType>(shape, m.domain_decomposed_dim,
                                            m.domain_decomposed_size,
                                            coords);
    }
  }

  if (m.dstype == data_structure_type::Cartesian_Grid_Non_Uniform) {
    for (DIM d = 0; d < D; d++) delete[] coords[d];
  }

  if (log::level & log::TIME) {
    timer_each.end();
    timer_each.print("Building Hierarchy");
    timer_each.clear();
  }

  std::vector<Array<D, T, DeviceType>> subdomain_data;
  std::vector<Array<1, Byte, DeviceType>> compressed_subdomain_data;
  std::vector<Hierarchy<D, T, DeviceType>> subdomain_hierarchy;

  if (log::level & log::TIME) timer_each.start();

  if (!m.domain_decomposed) {
    subdomain_hierarchy.push_back(hierarchy);
  } else {
    subdomain_hierarchy = hierarchy.hierarchy_chunck;
    log::info("Orignial domain was decomposed into " +
               std::to_string(subdomain_hierarchy.size()) + 
               " sub-domains during compression");
      if (m.s != std::numeric_limits<T>::infinity()) {
        m.tol = std::sqrt((m.tol * m.tol) / subdomain_hierarchy.size());
        log::info("local bound: " + std::to_string(m.tol));
      }
  }

  SIZE byte_offset = m.metadata_size;
  for (uint32_t i = 0; i < subdomain_hierarchy.size(); i++) {
    DeviceRuntime<DeviceType>::SelectDevice(i % DeviceRuntime<DeviceType>::GetDeviceCount());
    SIZE temp_size;
    Byte *temp_data;
    Deserialize<SIZE, DeviceType>((Byte *)compressed_data, &temp_size, 1,
                                  byte_offset);
    MemoryManager<DeviceType>::MallocHost(temp_data, temp_size, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
    align_byte_offset<uint64_t>(byte_offset);
    Deserialize<Byte, DeviceType>((Byte *)compressed_data, temp_data,
                                  temp_size, byte_offset);
    Array<1, Byte, DeviceType> compressed_array({temp_size});
    compressed_array.load(temp_data);
    compressed_subdomain_data.push_back(compressed_array);
    MemoryManager<DeviceType>::FreeHost(temp_data);
  }

  if (log::level & log::TIME) {
    timer_each.end();
    timer_each.print("Deserialize data");
    timer_each.clear();
  }

  if (log::level & log::TIME) timer_each.start();
  // decompress
  // Set the number of threads equal to the number of devices
  // So that each thread is responsible for one device
  omp_set_num_threads(DeviceRuntime<DeviceType>::GetDeviceCount());
  #pragma omp parallel for schedule(dynamic) private(curr_dev_id)
  for (uint32_t i = 0; i < subdomain_hierarchy.size(); i++) {

    // Select device based on where is input data is
    DeviceRuntime<DeviceType>::SelectDevice(compressed_subdomain_data[i].resideDevice());
    config.dev_id = compressed_subdomain_data[i].resideDevice();

    // Trigger the copy constructor to copy hierarchy to corresponding device
    Hierarchy<D, T, DeviceType> hierarchy = subdomain_hierarchy[i];

    std::stringstream ss;
    for (DIM d = 0; d < D; d++)
      ss << subdomain_hierarchy[i].level_shape(
                       subdomain_hierarchy[i].l_target(), d)
                << " ";
    log::info("Compressing decomposed domain " + std::to_string(i) + "/" +
               std::to_string(subdomain_data.size()) + " with shape: " + ss.str() +
               "using thread " + std::to_string(omp_get_thread_num()) + "/" +
               std::to_string(omp_get_num_threads()) + "on device " + std::to_string(compressed_subdomain_data[i].resideDevice()));

    // PrintSubarray("input of decompress", SubArray(compressed_subdomain_data[i]));
    Array<D, T, DeviceType> out_array = decompress<D, T, DeviceType>(
        hierarchy, compressed_subdomain_data[i], m.ebtype,
        m.tol, m.s, m.norm, config);
    subdomain_data.push_back(out_array);
  }

  if (log::level & log::TIME) {
    timer_each.end();
    timer_each.print("Aggregated low-level decompression");
    log::time("Aggregated high-level decompression throughput: "
              + std::to_string((double)(original_size * sizeof(T)) / timer_each.get() / 1e9)
              + " GB/s");
    timer_each.clear();
  }

  // Use consistance memory space between input and output data
  if (!output_pre_allocated) {
    if (log::level & log::TIME) timer_each.start();
    if (MemoryManager<DeviceType>::IsDevicePointer(compressed_data)) {
      MemoryManager<DeviceType>::Malloc1D(decompressed_data, original_size);
    } else {
      decompressed_data = (T *)malloc(original_size * sizeof(T));
    }
    if (log::level & log::TIME) {
      timer_each.end();
      timer_each.print("Prepare output buffer");
      timer_each.clear();
    }
  }

  if (!hierarchy.domain_decomposed) {
    if (log::level & log::TIME) timer_each.start();
    MemoryManager<DeviceType>::Copy1D(decompressed_data,
                                      (void *)subdomain_data[0].data(),
                                      original_size * sizeof(T), 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
    if (log::level & log::TIME) {
      timer_each.end();
      timer_each.print("Copy output");
      timer_each.clear();
    }
  } else {
    if (log::level & log::TIME) timer_each.start();
    // domain recomposition
    domain_recompose<D, T, DeviceType>(subdomain_data, (T *)decompressed_data,
                                       shape, hierarchy.domain_decomposed_dim,
                                       hierarchy.domain_decomposed_size);
    if (log::level & log::TIME) {
      timer_each.end();
      timer_each.print("Domain recomposition");
      timer_each.clear();
    }
  }

  if (log::level & log::TIME) {
    timer_total.end();
    timer_total.print("High-level decompression");
    log::time("High-level decompression throughput: "
              + std::to_string((double)(original_size * sizeof(T)) / timer_total.get() / 1e9)
              + " GB/s");
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
      decompress<1, float, DeviceType>(shape, compressed_data,
                                       compressed_size, decompressed_data,
                                       config, output_pre_allocated);
    } else if (shape.size() == 2) {
      decompress<2, float, DeviceType>(shape, compressed_data,
                                       compressed_size, decompressed_data,
                                       config, output_pre_allocated);
    } else if (shape.size() == 3) {
      decompress<3, float, DeviceType>(shape, compressed_data,
                                       compressed_size, decompressed_data,
                                       config, output_pre_allocated);
    } else if (shape.size() == 4) {
      decompress<4, float, DeviceType>(shape, compressed_data,
                                       compressed_size, decompressed_data,
                                       config, output_pre_allocated);
    } else if (shape.size() == 5) {
      decompress<5, float, DeviceType>(shape, compressed_data,
                                       compressed_size, decompressed_data,
                                       config, output_pre_allocated);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else if (dtype == data_type::Double) {
    if (shape.size() == 1) {
      decompress<1, double, DeviceType>(shape, compressed_data,
                                        compressed_size, decompressed_data,
                                        config, output_pre_allocated);
    } else if (shape.size() == 2) {
      decompress<2, double, DeviceType>(shape, compressed_data,
                                        compressed_size, decompressed_data,
                                        config, output_pre_allocated);
    } else if (shape.size() == 3) {
      decompress<3, double, DeviceType>(shape, compressed_data,
                                        compressed_size, decompressed_data,
                                        config, output_pre_allocated);
    } else if (shape.size() == 4) {
      decompress<4, double, DeviceType>(shape, compressed_data,
                                        compressed_size, decompressed_data,
                                        config, output_pre_allocated);
    } else if (shape.size() == 5) {
      decompress<5, double, DeviceType>(shape, compressed_data,
                                        compressed_size, decompressed_data,
                                        config, output_pre_allocated);
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
      decompress<1, float, DeviceType>(shape, compressed_data,
                                       compressed_size, decompressed_data,
                                       config, output_pre_allocated);
    } else if (shape.size() == 2) {
      decompress<2, float, DeviceType>(shape, compressed_data,
                                       compressed_size, decompressed_data,
                                       config, output_pre_allocated);
    } else if (shape.size() == 3) {
      decompress<3, float, DeviceType>(shape, compressed_data,
                                       compressed_size, decompressed_data,
                                       config, output_pre_allocated);
    } else if (shape.size() == 4) {
      decompress<4, float, DeviceType>(shape, compressed_data,
                                       compressed_size, decompressed_data,
                                       config, output_pre_allocated);
    } else if (shape.size() == 5) {
      decompress<5, float, DeviceType>(shape, compressed_data,
                                       compressed_size, decompressed_data,
                                       config, output_pre_allocated);
    } else {
      log::err("do not support higher than five dimentions");
      exit(-1);
    }
  } else if (dtype == data_type::Double) {
    if (shape.size() == 1) {
      decompress<1, double, DeviceType>(shape, compressed_data,
                                        compressed_size, decompressed_data,
                                        config, output_pre_allocated);
    } else if (shape.size() == 2) {
      decompress<2, double, DeviceType>(shape, compressed_data,
                                        compressed_size, decompressed_data,
                                        config, output_pre_allocated);
    } else if (shape.size() == 3) {
      decompress<3, double, DeviceType>(shape, compressed_data,
                                        compressed_size, decompressed_data,
                                        config, output_pre_allocated);
    } else if (shape.size() == 4) {
      decompress<4, double, DeviceType>(shape, compressed_data,
                                        compressed_size, decompressed_data,
                                        config, output_pre_allocated);
    } else if (shape.size() == 5) {
      decompress<5, double, DeviceType>(shape, compressed_data,
                                        compressed_size, decompressed_data,
                                        config, output_pre_allocated);
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