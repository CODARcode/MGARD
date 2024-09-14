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

#if MGARD_ENABLE_OPENMP
#include "CPUPipelines.hpp"
#endif
#include "ErrorToleranceCalculator.hpp"
#include "GPUPipelines.hpp"
#include "ShapeAdjustment.hpp"

#define OUTPUT_SAFTY_OVERHEAD 1e6

namespace mgard_x {

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

  using Cache = CompressorCache<D, T, DeviceType, CompressorType>;
  if (std::is_same<DeviceType, CUDA>::value ||
      std::is_same<DeviceType, HIP>::value ||
      std::is_same<DeviceType, SYCL>::value ||
      std::is_same<DeviceType, SERIAL>::value) {
    Cache::cache.SafeInitialize(2, 1);
  } else {
    Cache::cache.SafeInitialize(domain_decomposer.num_subdomains(),
                                domain_decomposer.num_subdomains());
  }

  T norm = 1;
  T local_tol = tol;
  enum error_bound_type local_ebtype;

  if (config.compressor == compressor_type::MGARD) {
    log::info("Compressor type: MGARD");
  } else if (config.compressor == compressor_type::ZFP) {
    log::info("Compressor type: ZFP");
  }

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
    if (!input_previously_pinned && config.auto_pin_host_buffers) {
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
    if (!output_previously_pinned && config.auto_pin_host_buffers) {
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
  if constexpr (std::is_same<DeviceType, CUDA>::value ||
                std::is_same<DeviceType, HIP>::value ||
                std::is_same<DeviceType, SYCL>::value ||
                std::is_same<DeviceType, SERIAL>::value) {
    compress_status = compress_pipeline_gpu(
        domain_decomposer, local_tol, s, norm, local_ebtype, config,
        compressed_subdomain_data, compressed_subdomain_size);
  } else {
    compress_status = compress_pipeline_cpu(
        domain_decomposer, local_tol, s, norm, local_ebtype, config,
        compressed_subdomain_data, compressed_subdomain_size);
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

  if (!input_previously_pinned && config.auto_pin_host_buffers) {
    MemoryManager<DeviceType>::HostUnregister((void *)original_data);
  }
  if (!output_previously_pinned && config.auto_pin_host_buffers) {
    MemoryManager<DeviceType>::HostUnregister((void *)compressed_data);
  }

  if (config.auto_cache_release)
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
    if (!input_previously_pinned && config.auto_pin_host_buffers) {
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
    if (!output_previously_pinned && config.auto_pin_host_buffers) {
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
  m.InitializeConfig(config);

  std::vector<T *> coords(D);
  if (m.dstype == data_structure_type::Cartesian_Grid_Non_Uniform) {
    for (DIM d = 0; d < D; d++) {
      coords[d] = new T[shape[d]];
      for (SIZE i = 0; i < shape[d]; i++) {
        coords[d][i] = (float)m.coords[d][i];
      }
    }
  }

  if (config.compressor == compressor_type::MGARD) {
    log::info("Compressor type: MGARD");
  } else if (config.compressor == compressor_type::ZFP) {
    log::info("Compressor type: ZFP");
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

  using Cache = CompressorCache<D, T, DeviceType, CompressorType>;
  if (std::is_same<DeviceType, CUDA>::value ||
      std::is_same<DeviceType, HIP>::value ||
      std::is_same<DeviceType, SYCL>::value ||
      std::is_same<DeviceType, SERIAL>::value) {
    Cache::cache.SafeInitialize(2, 1);
  } else {
    Cache::cache.SafeInitialize(domain_decomposer.num_subdomains(),
                                domain_decomposer.num_subdomains());
  }

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

  if constexpr (std::is_same<DeviceType, CUDA>::value ||
                std::is_same<DeviceType, HIP>::value ||
                std::is_same<DeviceType, SYCL>::value ||
                std::is_same<DeviceType, SERIAL>::value) {
    decompress_status = decompress_pipeline_gpu(
        domain_decomposer, local_tol, (T)m.s, (T)m.norm, local_ebtype, config,
        compressed_subdomain_data);
  } else {
    decompress_status = decompress_pipeline_cpu(
        domain_decomposer, local_tol, (T)m.s, (T)m.norm, local_ebtype, config,
        compressed_subdomain_data);
  }
  if (log::level & log::TIME) {
    timer_each.end();
    timer_each.print("Aggregated low-level decompression");
    log::time("Aggregated low-level decompression throughput: " +
              std::to_string((double)(total_num_elem * sizeof(T)) /
                             timer_each.get() / 1e9) +
              " GB/s");
    timer_each.clear();
  }

  if (!input_previously_pinned && config.auto_pin_host_buffers) {
    MemoryManager<DeviceType>::HostUnregister((void *)compressed_data);
  }
  if (!output_previously_pinned && config.auto_pin_host_buffers) {
    MemoryManager<DeviceType>::HostUnregister((void *)decompressed_data);
  }

  if (m.dstype == data_structure_type::Cartesian_Grid_Non_Uniform) {
    for (DIM d = 0; d < D; d++)
      delete[] coords[d];
  }

  if (config.auto_cache_release)
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
  meta.InitializeConfig(config);

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
  meta.InitializeConfig(config);

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