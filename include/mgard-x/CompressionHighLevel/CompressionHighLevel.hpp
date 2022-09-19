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
#include "Metadata.hpp"
#include "compress_x.hpp"

#include "../CompressionLowLevel/CompressionLowLevel.h"
#include "../CompressionLowLevel/NormCalculator.hpp"

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

  Array<1, T, DeviceType> array_with_pitch({1});
  size_t pitch_size = array_with_pitch.ld(0) * sizeof(T);

  // log::info("pitch_size: " + std::to_string(pitch_size));

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

  // log::info("hierarchy_space: " +
  // std::to_string((double)hierarchy_space/1e9));

  size_t input_space = roundup(shape[D - 1] * sizeof(T), pitch_size);
  for (DIM d = 0; d < D - 1; d++) {
    input_space *= shape[d];
  }

  // log::info("input_space: " + std::to_string((double)input_space/1e9));

  CompressionLowLevelWorkspace<D, T, DeviceType> compression_workspace;

  estimate_memory_usgae =
      std::max(estimate_memory_usgae,
               hierarchy_space + input_space +
                   compression_workspace.estimate_size(shape, 64, 0.1));

  return estimate_memory_usgae;
}

template <DIM D, typename T, typename DeviceType>
bool need_domain_decomposition(std::vector<SIZE> shape) {
  size_t estm = estimate_memory_usgae<D, T, DeviceType>(shape);
  size_t aval = DeviceRuntime<DeviceType>::GetAvailableMemory();
  log::info("Estimated memory usage: " + std::to_string((double)estm / 1e9) +
            "GB, Available: " + std::to_string((double)aval / 1e9) + "GB");
  return estm >= aval;
}

template <DIM D, typename T, typename DeviceType>
bool generate_domain_decomposition_strategy(std::vector<SIZE> shape,
                                            DIM &domain_decomposed_dim,
                                            SIZE &domain_decomposed_size,
                                            SIZE num_dev) {
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

  // First device by the number of devices
  chunck_shape[domain_decomposed_dim] =
      std::ceil((double)chunck_shape[domain_decomposed_dim] / num_dev);

  // Then check if each chunk can fit into device memory
  while (need_domain_decomposition<D, T, DeviceType>(chunck_shape)) {
    // Divide by 2 and round up
    chunck_shape[domain_decomposed_dim] =
        (chunck_shape[domain_decomposed_dim] - 1) / 2 + 1;
  }
  domain_decomposed_size = chunck_shape[domain_decomposed_dim];
  log::info(
      "domain_decomposed_dim: " + std::to_string(domain_decomposed_dim) +
      ", domain_decomposed_size: " + std::to_string(domain_decomposed_size));
  return true;
}

template <DIM D>
void calc_domain_decompose_parameter(std::vector<SIZE> shape,
                                     DIM domain_decomposed_dim,
                                     SIZE domain_decomposed_size, SIZE &dst_ld,
                                     SIZE &src_ld, SIZE &n1, SIZE &n2) {
  dst_ld = domain_decomposed_size;
  for (int d = D - 1; d > (int)domain_decomposed_dim; d--) {
    dst_ld *= shape[d];
  }
  // std::cout << "dst_ld: " << dst_ld << "\n";
  src_ld = 1;
  for (int d = D - 1; d >= (int)domain_decomposed_dim; d--) {
    src_ld *= shape[d];
  }
  // std::cout << "src_ld: " << src_ld << "\n";
  n1 = domain_decomposed_size;
  for (int d = D - 1; d > (int)domain_decomposed_dim; d--) {
    n1 *= shape[d];
  }
  // std::cout << "n1: " << n1 << "\n";
  n2 = 1;
  for (int d = 0; d < (int)domain_decomposed_dim; d++) {
    n2 *= shape[d];
  }
  // std::cout << "n2: " << n2 << "\n";
}

#define ORIGINAL_TO_SUBDOMAIN 0
#define SUBDOMAIN_TO_ORIGINAL 1

template <DIM D, typename T, typename DeviceType>
void copy_subdomain(T *data, Array<D, T, DeviceType> &subdomain_data,
                    std::vector<SIZE> shape, DIM domain_decomposed_dim,
                    SIZE domain_decomposed_size, int subdomain_idx,
                    int option) {
  // Pitched memory allocation has to be disable for the correctness of the
  // following copies
  assert(MemoryManager<DeviceType>::ReduceMemoryFootprint == true);
  bool pitched = false;

  SIZE dst_ld, src_ld, n1, n2;
  calc_domain_decompose_parameter<D>(shape, domain_decomposed_dim,
                                     domain_decomposed_size, dst_ld, src_ld, n1,
                                     n2);
  data += n1 * subdomain_idx;
  if (subdomain_idx < shape[domain_decomposed_dim] / domain_decomposed_size) {
    if (option == ORIGINAL_TO_SUBDOMAIN) {
      std::vector<SIZE> chunck_shape = shape;
      chunck_shape[domain_decomposed_dim] = domain_decomposed_size;
      subdomain_data = Array<D, T, DeviceType>(chunck_shape, pitched);
    }
  } else {
    SIZE leftover_dim_size =
        shape[domain_decomposed_dim] % domain_decomposed_size;
    calc_domain_decompose_parameter<D>(shape, domain_decomposed_dim,
                                       leftover_dim_size, dst_ld, src_ld, n1,
                                       n2);
    if (option == ORIGINAL_TO_SUBDOMAIN) {
      std::vector<SIZE> leftover_shape = shape;
      leftover_shape[domain_decomposed_dim] = leftover_dim_size;
      subdomain_data = Array<D, T, DeviceType>(leftover_shape, pitched);
    }
  }

  if (option == ORIGINAL_TO_SUBDOMAIN) {
    MemoryManager<DeviceType>::CopyND(subdomain_data.data(), dst_ld, data,
                                      src_ld, n1, n2, 0);
  } else if (option == SUBDOMAIN_TO_ORIGINAL) {
    MemoryManager<DeviceType>::CopyND(data, src_ld, subdomain_data.data(),
                                      dst_ld, n1, n2, 0);
  } else {
    log::err("copy_subdomain: wrong option.");
    exit(-1);
  }
  DeviceRuntime<DeviceType>::SyncQueue(0);
}

template <DIM D, typename T, typename DeviceType>
T calc_norm_decomposed(std::vector<Array<D, T, DeviceType>> &subdomain_data,
                       T s, bool normalize_coordinates, SIZE total_num_elem) {
  T norm = 0;

  for (int i = 0; i < subdomain_data.size(); i++) {
    DeviceRuntime<DeviceType>::SelectDevice(subdomain_data[i].resideDevice());
    Array<1, T, DeviceType> norm_array({1});
    SubArray<1, T, DeviceType> norm_subarray(norm_array);
    // subdomain_data[i] has to be not pitched to avoid copy for linearization
    assert(!subdomain_data[i].isPitched());
    // Disable normalize_coordinate since we do not want to void dividing
    // total_elems
    T local_norm =
        norm_calculator(subdomain_data[i], SubArray<1, T, DeviceType>(),
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
void general_compress(std::vector<SIZE> shape, T tol, T s,
                      enum error_bound_type ebtype, const void *original_data,
                      void *&compressed_data, size_t &compressed_size,
                      Config config, bool uniform, std::vector<T *> coords,
                      bool output_pre_allocated) {

  size_t total_num_elem = 1;
  for (int i = 0; i < D; i++)
    total_num_elem *= shape[i];

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

  if (!need_domain_decomposition<D, T, DeviceType>(shape) &&
      adjusted_num_dev == 1) {
    if (uniform) {
      hierarchy = Hierarchy<D, T, DeviceType>(shape, config);
    } else {
      hierarchy = Hierarchy<D, T, DeviceType>(shape, coords, config);
    }
    num_subdomains = 1;
  } else {
    generate_domain_decomposition_strategy<D, T, DeviceType>(
        shape, domain_decomposed_dim, domain_decomposed_size, adjusted_num_dev);
    if (uniform) {
      hierarchy = Hierarchy<D, T, DeviceType>(shape, domain_decomposed_dim,
                                              domain_decomposed_size, config);
    } else {
      hierarchy = Hierarchy<D, T, DeviceType>(
          shape, domain_decomposed_dim, domain_decomposed_size, coords, config);
    }
    num_subdomains =
        (shape[domain_decomposed_dim] - 1) / domain_decomposed_size + 1;
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

  std::vector<Array<D, T, DeviceType>> subdomain_data(num_subdomains);
  std::vector<Array<1, Byte, DeviceType>> compressed_subdomain_data(
      num_subdomains);
  std::vector<Hierarchy<D, T, DeviceType>> subdomain_hierarchy;
  T norm = 1;
  T local_tol = tol;
  enum error_bound_type local_ebtype;

  if (!hierarchy.domain_decomposed) {
    if (log::level & log::TIME)
      timer_each.start();
    subdomain_data[0] = Array<D, T, DeviceType>({shape});
    subdomain_data[0].load((T *)original_data);
    subdomain_hierarchy.push_back(hierarchy);
    local_tol = tol;
    local_ebtype = ebtype;
    if (log::level & log::TIME) {
      timer_each.end();
      timer_each.print("Load data");
      timer_each.clear();
    }
  } else {
    log::info("Orignial domain is decomposed into " +
              std::to_string(hierarchy.hierarchy_chunck.size()) +
              " sub-domains");
    if (log::level & log::TIME)
      timer_each.start();
    for (int i = 0; i < num_subdomains; i++) {
      DeviceRuntime<DeviceType>::SelectDevice(i % adjusted_num_dev);
      copy_subdomain<D, T, DeviceType>((T *)original_data, subdomain_data[i],
                                       shape, hierarchy.domain_decomposed_dim,
                                       hierarchy.domain_decomposed_size, i,
                                       ORIGINAL_TO_SUBDOMAIN);
    }

    if (log::level & log::TIME) {
      timer_each.end();
      timer_each.print("Domain decomposition");
      timer_each.clear();
    }

    if (log::level & log::TIME)
      timer_each.start();
    subdomain_hierarchy = hierarchy.hierarchy_chunck;
    if (ebtype == error_bound_type::REL) {
      norm = calc_norm_decomposed(subdomain_data, s,
                                  config.normalize_coordinates, total_num_elem);
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
  for (SIZE i = 0; i < subdomain_data.size(); i++) {
    // Select device based on where is input data is
    DeviceRuntime<DeviceType>::SelectDevice(subdomain_data[i].resideDevice());
    config.dev_id = subdomain_data[i].resideDevice();

    // Trigger the copy constructor to copy hierarchy to corresponding device
    Hierarchy<D, T, DeviceType> hierarchy = subdomain_hierarchy[i];

    CompressionLowLevelWorkspace workspace(hierarchy, 0.1);

    std::stringstream ss;
    for (DIM d = 0; d < D; d++)
      ss << subdomain_hierarchy[i].level_shape(
                subdomain_hierarchy[i].l_target(), d)
         << " ";
#if MGARD_ENABLE_MULTI_DEVICE
    log::info("Compressing subdomain " + std::to_string(i + 1) + "/" +
              std::to_string(subdomain_data.size()) +
              " with shape: " + ss.str() + "on thread " +
              std::to_string(omp_get_thread_num() + 1) + "/" +
              std::to_string(omp_get_num_threads()) + " device " +
              std::to_string(subdomain_data[i].resideDevice()));
#else
    log::info("Compressing subdomain " + std::to_string(i + 1) + "/" +
              std::to_string(subdomain_data.size()) +
              " with shape: " + ss.str());
#endif

    compressed_subdomain_data[i] =
        compress<D, T, DeviceType>(hierarchy, subdomain_data[i], local_ebtype,
                                   local_tol, s, norm, config, workspace);
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
  for (uint32_t i = 0; i < subdomain_data.size(); i++) {
    advance_with_align<SIZE>(byte_offset, 1);
    align_byte_offset<uint64_t>(byte_offset); // for zero copy when deserialize
    advance_with_align<Byte>(byte_offset,
                             compressed_subdomain_data[i].shape(0));
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
  for (uint32_t i = 0; i < subdomain_data.size(); i++) {
    SIZE subdomain_compressed_size = compressed_subdomain_data[i].shape(0);
    Serialize<SIZE, DeviceType>((Byte *)compressed_data,
                                &subdomain_compressed_size, 1, byte_offset);
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

  std::vector<Array<D, T, DeviceType>> subdomain_data(num_subdomains);
  std::vector<Array<1, Byte, DeviceType>> compressed_subdomain_data(
      num_subdomains);
  std::vector<Hierarchy<D, T, DeviceType>> subdomain_hierarchy;
  T local_tol;
  enum error_bound_type local_ebtype;

  if (log::level & log::TIME)
    timer_each.start();

  if (!m.domain_decomposed) {
    subdomain_hierarchy.push_back(hierarchy);
    local_tol = m.tol;
    local_ebtype = m.ebtype;
  } else {
    subdomain_hierarchy = hierarchy.hierarchy_chunck;
    log::info("Orignial domain was decomposed into " +
              std::to_string(subdomain_hierarchy.size()) +
              " subdomains during compression");
    local_tol =
        calc_local_abs_tol(m.ebtype, m.norm, m.tol, m.s, num_subdomains);
    // Force to use ABS mode when do domain decomposition
    local_ebtype = error_bound_type::ABS;
    // Fast copy for domain decomposition need we disable pitched memory
    // allocation
    MemoryManager<DeviceType>::ReduceMemoryFootprint = true;
  }

  SIZE byte_offset = m.metadata_size;
  for (uint32_t i = 0; i < subdomain_hierarchy.size(); i++) {
    DeviceRuntime<DeviceType>::SelectDevice(i % adjusted_num_dev);
    SIZE temp_size;
    Byte *temp_data;
    Deserialize<SIZE, DeviceType>((Byte *)compressed_data, &temp_size, 1,
                                  byte_offset);
    MemoryManager<DeviceType>::MallocHost(temp_data, temp_size);
    align_byte_offset<uint64_t>(byte_offset);
    Deserialize<Byte, DeviceType>((Byte *)compressed_data, temp_data, temp_size,
                                  byte_offset);
    compressed_subdomain_data[i] = Array<1, Byte, DeviceType>({temp_size});
    compressed_subdomain_data[i].load(temp_data);
    MemoryManager<DeviceType>::FreeHost(temp_data);
  }

  if (log::level & log::TIME) {
    timer_each.end();
    timer_each.print("Deserialize data");
    timer_each.clear();
  }

  if (log::level & log::TIME)
    timer_each.start();
    // decompress
    // Set the number of threads equal to the number of devices
    // So that each thread is responsible for one device
#if MGARD_ENABLE_MULTI_DEVICE
  omp_set_num_threads(adjusted_num_dev);
#endif
#pragma omp parallel for firstprivate(config)
  for (uint32_t i = 0; i < subdomain_hierarchy.size(); i++) {

    // Select device based on where is input data is
    DeviceRuntime<DeviceType>::SelectDevice(
        compressed_subdomain_data[i].resideDevice());
    config.dev_id = compressed_subdomain_data[i].resideDevice();

    // Trigger the copy constructor to copy hierarchy to corresponding device
    Hierarchy<D, T, DeviceType> hierarchy = subdomain_hierarchy[i];

    CompressionLowLevelWorkspace workspace(hierarchy, 0.0);

    std::stringstream ss;
    for (DIM d = 0; d < D; d++)
      ss << subdomain_hierarchy[i].level_shape(
                subdomain_hierarchy[i].l_target(), d)
         << " ";
#if MGARD_ENABLE_MULTI_DEVICE
    log::info("Decompressing subdomain " + std::to_string(i + 1) + "/" +
              std::to_string(compressed_subdomain_data.size()) +
              " with shape: " + ss.str() + "on thread " +
              std::to_string(omp_get_thread_num() + 1) + "/" +
              std::to_string(omp_get_num_threads()) + " device " +
              std::to_string(compressed_subdomain_data[i].resideDevice()));
#else
    log::info("Decompressing subdomain " + std::to_string(i + 1) + "/" +
              std::to_string(compressed_subdomain_data.size()) +
              " with shape: " + ss.str());
#endif

    // PrintSubarray("input of decompress",
    // SubArray(compressed_subdomain_data[i]));
    subdomain_data[i] = decompress<D, T, DeviceType>(
        hierarchy, compressed_subdomain_data[i], local_ebtype, local_tol, m.s,
        m.norm, config, workspace);
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
      decompressed_data = malloc(total_num_elem * sizeof(T));
    }
    if (log::level & log::TIME) {
      timer_each.end();
      timer_each.print("Prepare output buffer");
      timer_each.clear();
    }
  }

  if (!hierarchy.domain_decomposed) {
    if (log::level & log::TIME)
      timer_each.start();
    // MemoryManager<DeviceType>::Copy1D(decompressed_data,
    //                                   (void *)subdomain_data[0].data(),
    //                                   total_num_elem * sizeof(T), 0);
    MemoryManager<DeviceType>::CopyND(
        decompressed_data, subdomain_data[0].shape(D - 1) * sizeof(T),
        (void *)subdomain_data[0].data(),
        subdomain_data[0].ld(D - 1) * sizeof(T),
        subdomain_data[0].shape(D - 1) * sizeof(T),
        (SIZE)hierarchy.linearized_width(), 0);

    DeviceRuntime<DeviceType>::SyncQueue(0);
    if (log::level & log::TIME) {
      timer_each.end();
      timer_each.print("Copy output");
      timer_each.clear();
    }
  } else {
    if (log::level & log::TIME)
      timer_each.start();

    for (int i = 0; i < num_subdomains; i++) {
      copy_subdomain<D, T, DeviceType>(
          (T *)decompressed_data, subdomain_data[i], shape,
          hierarchy.domain_decomposed_dim, hierarchy.domain_decomposed_size, i,
          SUBDOMAIN_TO_ORIGINAL);
    }

    if (log::level & log::TIME) {
      timer_each.end();
      timer_each.print("Domain recomposition");
      timer_each.clear();
    }
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