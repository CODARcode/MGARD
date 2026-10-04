/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "../Config/Config.h"
#include "../Hierarchy/Hierarchy.h"
#include "../RuntimeX/RuntimeX.h"
#include "../Utilities/Types.h"
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
    HybridHierarchy<D, T, DeviceType> &hierarchy, Config config)
    : initialized(true), hierarchy(&hierarchy), config(config),
      hybrid_refactor(hierarchy, config),
      lossless_compressor(calculate_padded_size(hierarchy, config), config),
      hybrid_quantizer(hierarchy, config) {
  norm_array = Array<1, T, DeviceType>({1});
  hybrid_decomposed_array =
      Array<1, T, DeviceType>({hybrid_refactor.DecomposedDataSize()});

  // Reuse workspace. Warning:
  if (sizeof(QUANTIZED_INT) <= sizeof(T)) {
    if (config.num_local_refactoring_level > 0) {
      norm_tmp_array = Array<1, T, DeviceType>(
          {hierarchy.total_num_elems()},
          (T *)hybrid_refactor.local_refactor.coarse_buffers[0].data());
      // hybrid_quantized_array = Array<1, QUANTIZED_INT, DeviceType>(
      //     {hybrid_refactor.DecomposedDataSize()},
      //     (QUANTIZED_INT*)hybrid_refactor.local_refactor.coarse_buffers[0]
      //         .data());
    } else {
      // Reuse space from global refactor
      norm_tmp_array = Array<1, T, DeviceType>(
          {hierarchy.total_num_elems()},
          (T *)hybrid_refactor.global_refactor.w_array.data());
      // hybrid_quantized_array = Array<1, QUANTIZED_INT, DeviceType>(
      //     {hybrid_refactor.DecomposedDataSize()},
      //     (QUANTIZED_INT*)hybrid_refactor.global_refactor.w_array.data());
    }
  } else {
    // if space is not enough
    norm_tmp_array = Array<1, T, DeviceType>({hierarchy.total_num_elems()});
    // hybrid_quantized_array = Array<1, QUANTIZED_INT, DeviceType>(
    //     {hybrid_refactor.DecomposedDataSize()});
  }
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::Adapt(
    HybridHierarchy<D, T, DeviceType> &hierarchy, Config config,
    int queue_idx) {
  hierarchy.check_config(config);
  this->initialized = true;
  this->hierarchy = &hierarchy;
  this->config = config;
  hybrid_refactor.Adapt(hierarchy, config, queue_idx);
  lossless_compressor.Adapt(calculate_padded_size(hierarchy, config), config,
                            queue_idx);
  hybrid_quantizer.Adapt(hierarchy, config, queue_idx);
  norm_array.resize({1}, queue_idx);
  hybrid_decomposed_array.resize({hybrid_refactor.DecomposedDataSize()},
                                 queue_idx);
  hybrid_quantized_array = Array<1, QUANTIZED_INT, DeviceType>(
      {hybrid_refactor.DecomposedDataSize()});

  // Reuse workspace
  if (sizeof(QUANTIZED_INT) <= sizeof(T)) {
    if (config.num_local_refactoring_level > 0) {
      norm_tmp_array = Array<1, T, DeviceType>(
          {hierarchy.total_num_elems()},
          (T *)hybrid_refactor.local_refactor.coarse_buffers[0].data());
      // hybrid_quantized_array = Array<1, QUANTIZED_INT, DeviceType>(
      //     {hybrid_refactor.DecomposedDataSize()},
      //     (QUANTIZED_INT*)hybrid_refactor.local_refactor.coarse_buffers[0]
      //         .data());
    } else {
      // Reuse space from global refactor
      norm_tmp_array = Array<1, T, DeviceType>(
          {hierarchy.total_num_elems()},
          (T *)hybrid_refactor.global_refactor.w_array.data());
      // hybrid_quantized_array = Array<1, QUANTIZED_INT, DeviceType>(
      //     {hybrid_refactor.DecomposedDataSize()},
      //     (QUANTIZED_INT*)hybrid_refactor.global_refactor.w_array.data());
    }
  } else {
    norm_tmp_array.resize({hierarchy.total_num_elems()}, queue_idx);
    // hybrid_quantized_array.resize({hybrid_refactor.DecomposedDataSize()},
    //                               queue_idx);
  }
}

// May not be accurate
template <DIM D, typename T, typename DeviceType>
size_t HybridHierarchyCompressor<D, T, DeviceType>::EstimateMemoryFootprint(
    std::vector<SIZE> shape, Config config) {
  size_t size = 0;
  // size +=
  // BlockLocalHierarchyDataRefactorType::EstimateMemoryFootprint(shape);
  size +=
      HybridHierarchyDataRefactorType::EstimateMemoryFootprint(shape, config);
  // log::info(
  //     "Data refactor space: " +
  //     std::to_string(
  //         (double)(BlockLocalHierarchyDataRefactorType::EstimateMemoryFootprint(
  //             shape)) /
  //         1e9) +
  //     " GB");
  // size += LocalQuantizerType::EstimateMemoryFootprint(shape);
  size += HybridQuantizerType::EstimateMemoryFootprint(shape);
  // log::info(
  //     "Quantizer space: " +
  //     std::to_string(
  //         (double)(LocalQuantizerType::EstimateMemoryFootprint(shape)) / 1e9)
  //         +
  //     " GB");
  size += LosslessCompressorType::EstimateMemoryFootprint(
      calculate_padded_size(shape, config), config);
  // log::info(
  //     "Lossless space: " +
  //     std::to_string((double)(LosslessCompressorType::EstimateMemoryFootprint(
  //                        hierarchy.total_num_elems(), config)) /
  //                    1e9) +
  //     " GB");
  size += sizeof(T);
  if (sizeof(QUANTIZED_INT) > sizeof(T)) {
    size += sizeof(T) * calculate_padded_size(shape, config);
    size += sizeof(QUANTIZED_INT) * calculate_padded_size(shape, config);
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
  // DumpSubArray("/home/leonli/TestInCacheBlock/org.txt",SubArray(original_data));
  // PrintSubarray("Original before decompose", SubArray(original_data));
  // SubArray<D, T, DeviceType> temp({3,3,3}, original_data.data());
  // PrintSubarray("Orginal 8x8x8 before decompose", temp);
  // hybrid_refactor.Decompose(original_data, decomposed_array, queue_idx);
  hybrid_refactor.Decompose(SubArray(original_data),
                            SubArray(hybrid_decomposed_array), queue_idx,
                            orthogonal_projection);
  // PrintSubarray("Decomposed after decompose",
  // SubArray(local_decomposed_array));
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::Quantize(
    Array<D, T, DeviceType> &original_data, enum error_bound_type ebtype, T tol,
    T s, T norm, int queue_idx) {
  // Only called from Compress(), which has already resolved
  // orthogonal_projection and set it on hybrid_quantizer.
  SIZE total_num_elems_1D = hybrid_refactor.DecomposedDataSize();

  SubArray<1, T, DeviceType> data_subarray({total_num_elems_1D},
                                           hybrid_decomposed_array.data());
  hybrid_quantizer.Quantize(data_subarray, ebtype, tol, s, norm,
                            hybrid_quantized_array, lossless_compressor,
                            queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::DecomposeQuantize(
    Array<D, T, DeviceType> &original_data, enum error_bound_type ebtype, T tol,
    T s, T norm, int queue_idx) {
  // Only called from Compress(), which has already resolved
  // orthogonal_projection and set it on hybrid_quantizer.
  SubArray<1, T, DeviceType> decomposed_subarray(hybrid_decomposed_array);
  SubArray<1, QUANTIZED_INT, DeviceType> quantized_subarray(
      hybrid_quantized_array);
  hybrid_quantizer.DecomposeQuantize(
      hybrid_refactor, SubArray(original_data), decomposed_subarray,
      quantized_subarray, ebtype, tol, s, norm, lossless_compressor, queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::LosslessCompress(
    Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
  lossless_compressor.Compress(hybrid_quantized_array, compressed_data,
                               queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::Serialize(
    Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
  lossless_compressor.Serialize(compressed_data, queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::Deserialize(
    Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
  lossless_compressor.Deserialize(compressed_data, queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::Recompose(
    Array<D, T, DeviceType> &decompressed_data, bool orthogonal_projection,
    int queue_idx) {
  this->orthogonal_projection = orthogonal_projection;
  // PrintSubarray("Decomposed before recompose",
  // SubArray(local_decomposed_array));
  hybrid_refactor.Recompose(SubArray(decompressed_data),
                            SubArray(hybrid_decomposed_array), queue_idx,
                            this->orthogonal_projection);

  // SubArray<D, T, DeviceType> temp({3,3,3}, decompressed_data.data());
  // PrintSubarray("Orginal 8x8x8 after decompose", temp);
  // PrintSubarray("Decompressed after recompose", SubArray(decompressed_data));
  // DumpSubArray("/home/leonli/TestInCacheBlock/decomp.txt",SubArray(decompressed_data));
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::Dequantize(
    Array<D, T, DeviceType> &decompressed_data, enum error_bound_type ebtype,
    T tol, T s, T norm, int queue_idx) {
  // Only called from DequantizeRecompose(), which has already resolved
  // orthogonal_projection and set it on hybrid_quantizer.
  SIZE total_num_elems_1D = hybrid_refactor.DecomposedDataSize();
  SubArray<1, T, DeviceType> decompressed_data_subarray(
      {total_num_elems_1D}, hybrid_decomposed_array.data());
  // Direct calculation
  hybrid_quantizer.Dequantize(decompressed_data_subarray, ebtype, tol, s, norm,
                              hybrid_quantized_array, lossless_compressor,
                              queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::DequantizeRecompose(
    Array<D, T, DeviceType> &decompressed_data, enum error_bound_type ebtype,
    T tol, T s, T norm, int queue_idx) {
  // The decompress-side resolution point: the domain-decomposition pipelines
  // (GPUPipelines/CPUPipelines) call this directly, bypassing Decompress(), so
  // it has to resolve for itself rather than relying on a caller having done
  // it already.
  orthogonal_projection =
      infer_orthogonal_projection(config.projection_mode, s);
  hybrid_quantizer.SetOrthogonalProjection(orthogonal_projection);
  if (config.fuse_dequantize_recompose && hybrid_quantizer.CanFuseQuantize(s)) {
    log::info("Local dequantize+recompose kernels: fused");
    DequantizeRecomposeFused(decompressed_data, ebtype, tol, s, norm,
                             queue_idx);
  } else {
    log::info("Local dequantize+recompose kernels: separate (" +
              (config.fuse_dequantize_recompose
                   ? hybrid_quantizer.WhyCannotFuseQuantize(s)
                   : std::string("kernel fusion disabled")) +
              ")");
    Dequantize(decompressed_data, ebtype, tol, s, norm, queue_idx);
    Recompose(decompressed_data, orthogonal_projection, queue_idx);
  }
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::DequantizeRecomposeFused(
    Array<D, T, DeviceType> &decompressed_data, enum error_bound_type ebtype,
    T tol, T s, T norm, int queue_idx) {
  SubArray<1, T, DeviceType> decomposed_subarray(hybrid_decomposed_array);
  SubArray<1, QUANTIZED_INT, DeviceType> quantized_subarray(
      hybrid_quantized_array);
  // Build the output view from the hierarchy shape (like the unfused
  // Recompose does for its final copy) rather than the array metadata, which
  // the pipelines may not have resized yet.
  SubArray<D, T, DeviceType> data_subarray(
      hierarchy->level_shape(hierarchy->l_target()), decompressed_data.data());
  hybrid_quantizer.DequantizeRecompose(
      hybrid_refactor, data_subarray, decomposed_subarray, quantized_subarray,
      ebtype, tol, s, norm, lossless_compressor, queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void HybridHierarchyCompressor<D, T, DeviceType>::LosslessDecompress(
    Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
  lossless_compressor.Decompress(compressed_data, hybrid_quantized_array,
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
  Timer timer_compress_kernel;
  for (int d = D - 1; d >= 0; d--) {
    if (hierarchy->level_shape(hierarchy->l_target(), d) !=
        original_data.shape(d)) {
      log::err("The shape of input array does not match the shape initilized "
               "in hierarchy!");
      return;
    }
  }

  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    timer_total.start();
  }

  // The compress-side resolution point: Decompose()/DecomposeQuantize()/
  // Quantize() below all just consume the result.
  orthogonal_projection =
      infer_orthogonal_projection(config.projection_mode, s);
  hybrid_quantizer.SetOrthogonalProjection(orthogonal_projection);

  CalculateNorm(original_data, ebtype, s, norm, queue_idx);
  // log::info(std::to_string(original_data.totalNumElems()));
  // PrintSubarray("Original before decompose", SubArray(original_data));
  // log::info("Before decompose()");

  if (log::level & log::TIME)
    timer_compress_kernel.start();
  if (config.fuse_decompose_quantize && hybrid_quantizer.CanFuseQuantize(s)) {
    log::info("Local decompose+quantize kernels: fused");
    DecomposeQuantize(original_data, ebtype, tol, s, norm, queue_idx);
  } else {
    log::info("Local decompose+quantize kernels: separate (" +
              (config.fuse_decompose_quantize
                   ? hybrid_quantizer.WhyCannotFuseQuantize(s)
                   : std::string("kernel fusion disabled")) +
              ")");
    Decompose(original_data, queue_idx);
    // log::info("After decompose()");
    // log::info(std::to_string(original_data.totalNumElems()));
    // PrintSubarray("Original after decompose", SubArray(original_data));
    // // PrintSubarray("Decomposed", SubArray(decomposed_array));
    // log::info("Before quantize");
    Quantize(original_data, ebtype, tol, s, norm, queue_idx);
  }
  // log::info("After quantize");
  // log::info("Num of Original data after quantization:");
  // log::info(std::to_string(original_data.totalNumElems()));
  // PrintSubarray("Quantized", SubArray(local_quantized_array));
  // PrintSubarray("Compressed data before lossless",SubArray(compressed_data));
  // log::info("Before lossless");
  LosslessCompress(compressed_data, queue_idx);
  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncQueue(0);
    timer_compress_kernel.end();
    timer_compress_kernel.print("Compression Kernel");
    timer_compress_kernel.clear();
  }
  Serialize(compressed_data, queue_idx);
  // log::info("After lossless");
  // PrintSubarray("Quantized data before
  // lossless",SubArray(local_quantized_array)); PrintSubarray("Compressed data
  // after lossless",SubArray(compressed_data)); From printing result, we found
  // lossless didn't do anything to compressed_data
  if (config.compress_with_dryrun) {
    DequantizeRecompose(original_data, ebtype, tol, s, norm, queue_idx);
  }

  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncQueue(0);
    timer_total.end();
    timer_total.print("Low-level compression",
                      hierarchy->total_num_elems() * sizeof(T));
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
  Deserialize(compressed_data, queue_idx);
  LosslessDecompress(compressed_data, queue_idx);
  // DequantizeRecompose resolves orthogonal_projection itself.
  DequantizeRecompose(decompressed_data, ebtype, tol, s, norm, queue_idx);

  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncQueue(0);
    timer_total.end();
    timer_total.print("Low-level decompression",
                      hierarchy->total_num_elems() * sizeof(T));
    timer_total.clear();
  }
}

// Only calculating padding to 8x8x8 for once
template <DIM D, typename T, typename DeviceType>
SIZE HybridHierarchyCompressor<D, T, DeviceType>::calculate_padded_size(
    std::vector<SIZE> shape, Config config) {
  SIZE L = config.num_local_refactoring_level;
  SIZE total_num_elems_1D = 1;
  if (L > 0) {
    std::vector<std::vector<SIZE>> fine_shapes, coarse_shapes;
    HybridHierarchy<D, T, DeviceType>::compute_local_shapes(
        shape, L, fine_shapes, coarse_shapes);
    for (SIZE l = 0; l < L; l++) {
      SIZE last_level_size = 1, curr_level_size = 1;
      for (DIM d = 0; d < D; d++) {
        last_level_size *= fine_shapes[l][d];
        curr_level_size *= coarse_shapes[l][d];
      }
      total_num_elems_1D += (last_level_size - curr_level_size);
      if (l == L - 1) {
        total_num_elems_1D += curr_level_size;
      }
    }
  } else {
    total_num_elems_1D = 1;
    for (DIM d = 0; d < D; d++) {
      total_num_elems_1D *= shape[d];
    }
  }

  return total_num_elems_1D;
}

template <DIM D, typename T, typename DeviceType>
SIZE HybridHierarchyCompressor<D, T, DeviceType>::calculate_padded_size(
    HybridHierarchy<D, T, DeviceType> &hierarchy, Config config) {
  return calculate_padded_size(hierarchy.input_shape(), config);
}

} // namespace mgard_x

#endif
