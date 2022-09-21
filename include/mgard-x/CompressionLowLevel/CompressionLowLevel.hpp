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
#include "../Hierarchy/Hierarchy.hpp"
#include "../RuntimeX/RuntimeX.h"
#include "CompressionLowLevel.h"

#include "../DataRefactoring/MultiDimension/DataRefactoring.h"
#include "../DataRefactoring/SingleDimension/DataRefactoring.h"

#include "NormCalculator.hpp"

#include "../Quantization/LinearQuantization.hpp"

#include "../Linearization/LevelLinearizer.hpp"

#include "../Lossless/ParallelHuffman/Huffman.hpp"

#ifdef MGARDX_COMPILE_CUDA
#include "../Lossless/Cascaded.hpp"
#include "../Lossless/LZ4.hpp"
#endif

#include "../Lossless/CPU.hpp"
#include "../Lossless/Zstd.hpp"

#ifndef MGARD_X_COMPRESSION_LOW_LEVEL_HPP
#define MGARD_X_COMPRESSION_LOW_LEVEL_HPP

namespace mgard_x {

static bool debug_print_compression = true;

template <DIM D, typename T, typename DeviceType>
Array<1, unsigned char, DeviceType>
compress(Hierarchy<D, T, DeviceType> &hierarchy,
         Array<D, T, DeviceType> &original_array, enum error_bound_type type,
         T tol, T s, T &norm, Config config,
         CompressionLowLevelWorkspace<D, T, DeviceType> &workspace) {

  config.apply();

  DeviceRuntime<DeviceType>::SelectDevice(config.dev_id);
  log::info("Select device: " + DeviceRuntime<DeviceType>::GetDeviceName());
  Timer timer_total, timer_each;
  for (int d = D - 1; d >= 0; d--) {
    if (hierarchy.level_shape(hierarchy.l_target(), d) !=
        original_array.shape(d)) {
      log::err("The shape of input array does not match the shape initilized "
               "in hierarchy!");
      std::vector<SIZE> empty_shape;
      empty_shape.push_back(1);
      Array<1, unsigned char, DeviceType> empty(empty_shape);
      return empty;
    }
  }

  // Workspaces
  if (!workspace.pre_allocated) {
    if (log::level & log::TIME)
      timer_total.start();
    // Allocate workspace if not pre-allocated
    workspace = CompressionLowLevelWorkspace(hierarchy);
    if (log::level & log::TIME) {
      timer_each.end();
      timer_each.print("Allocate workspace");
      timer_each.clear();
    }
  }

  SubArray in_subarray(original_array);
  SIZE total_elems = hierarchy.total_num_elems();

  if (log::level & log::TIME)
    timer_total.start();

  // Norm
  if (type == error_bound_type::REL) {
    norm = norm_calculator(original_array, workspace.norm_tmp_subarray,
                           workspace.norm_subarray, s,
                           config.normalize_coordinates);
  }

  // Decomposition
  if (config.decomposition == decomposition_type::MultiDim) {
    decompose<D, T, DeviceType>(hierarchy, in_subarray,
                                workspace.refactoring_w_subarray,
                                workspace.refactoring_b_subarray, 0, 0);
  } else if (config.decomposition == decomposition_type::SingleDim) {
    decompose_single<D, T, DeviceType>(hierarchy, in_subarray, 0, 0);
  }

  // Quantization
  bool prep_huffman =
      config.lossless != lossless_type::CPU_Lossless; // always do Huffman

  SubArray<2, SIZE, DeviceType> level_ranges_subarray(hierarchy.level_ranges(),
                                                      true);
  SubArray<3, T, DeviceType> level_volumes_subarray(hierarchy.level_volumes());
  LENGTH outlier_count;

#ifndef MGARDX_COMPILE_CUDA
  if (config.lossless == lossless_type::Huffman_LZ4) {
    log::warn("LZ4 is only available in CUDA. Switched to Zstd");
    config.lossless = lossless_type::Huffman_Zstd;
  }
#endif

  T *quantizers = new T[hierarchy.l_target() + 1];
  calc_quantizers<D, T>(total_elems, quantizers, type, tol, s, norm,
                        hierarchy.l_target(), config.decomposition, true);
  MemoryManager<DeviceType>::Copy1D(workspace.quantizers_subarray.data(),
                                    quantizers, hierarchy.l_target() + 1);
  delete[] quantizers;

  DeviceRuntime<DeviceType>::SyncQueue(0);

  bool done_quantization = false;
  while (!done_quantization) {
    LevelwiseLinearQuantizerND<D, T, MGARDX_QUANTIZE, DeviceType>().Execute(
        level_ranges_subarray, hierarchy.l_target(),
        workspace.quantizers_subarray, level_volumes_subarray, s,
        config.huff_dict_size, in_subarray, workspace.quantized_subarray,
        prep_huffman, config.reorder, workspace.outlier_count_subarray,
        workspace.outlier_idx_subarray, workspace.outliers_subarray, 0);

    MemoryManager<DeviceType>::Copy1D(
        &outlier_count, workspace.outlier_count_subarray.data(), 1, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
    if (outlier_count <= workspace.outliers_subarray.shape(0)) {
      // outlier buffer has sufficient size
      done_quantization = true;
    } else {
      log::info("Not enough workspace for outliers. Re-allocating to " +
                std::to_string(outlier_count));
      workspace.outlier_idx_array =
          Array<1, LENGTH, DeviceType>({(SIZE)outlier_count});
      workspace.outliers_array =
          Array<1, QUANTIZED_INT, DeviceType>({(SIZE)outlier_count});
      workspace.outlier_idx_subarray = SubArray(workspace.outlier_idx_array);
      workspace.outliers_subarray = SubArray(workspace.outliers_array);
      workspace.outlier_count_array.memset(0);
    }
  }

  // if (debug_print) {
  // PrintSubarray("decomposed", SubArray(original_array));
  // PrintSubarray("quantized_subarray", quantized_subarray);
  // PrintSubarray("quantized outliers_array", outliers_subarray);
  // PrintSubarray("quantized outlier_idx_array", outlier_idx_subarray);
  // }

  Array<1, Byte, DeviceType> compressed_array;
  SubArray<1, Byte, DeviceType> compressed_subarray;

  if (config.lossless != lossless_type::CPU_Lossless) {
    // Huffman compression
    // Cast to 1D unsigned integers when do CPU compression
    SubArray<1, QUANTIZED_UNSIGNED_INT, DeviceType> cast_quantized_subarray =
        SubArray<1, QUANTIZED_UNSIGNED_INT, DeviceType>(
            {total_elems},
            (QUANTIZED_UNSIGNED_INT *)workspace.quantized_subarray.data());
    compressed_array =
        HuffmanCompress<QUANTIZED_UNSIGNED_INT, HUFFMAN_CODE, DeviceType>(
            cast_quantized_subarray, config.huff_block_size,
            config.huff_dict_size, outlier_count,
            workspace.outlier_idx_subarray, workspace.outliers_subarray,
            workspace.huffman_subarray, workspace.status_subarray);
    compressed_subarray = SubArray(compressed_array);
  } else {
    // Cast to 1D signed integers when do CPU compression
    SubArray<1, QUANTIZED_INT, DeviceType> cast_linearized_subarray =
        SubArray<1, QUANTIZED_INT, DeviceType>(
            {total_elems},
            (QUANTIZED_INT *)workspace.quantized_subarray.data());

    compressed_array =
        CPUCompress<QUANTIZED_INT, DeviceType>(cast_linearized_subarray);
    compressed_subarray = SubArray(compressed_array);
  }

  // PrintSubarray("compressed_subarray",
  // compressed_subarray);

#ifdef MGARDX_COMPILE_CUDA
  // LZ4 compression
  if (config.lossless == lossless_type::Huffman_LZ4) {
    compressed_array = LZ4Compress(compressed_subarray, config.lz4_block_size);
    compressed_subarray = SubArray(compressed_array);
  }
#endif

  if (config.lossless == lossless_type::Huffman_Zstd) {
    compressed_array =
        ZstdCompress(compressed_subarray, config.zstd_compress_level);
    compressed_subarray = SubArray(compressed_array);
  }

  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncQueue(0);
    timer_total.end();
    timer_total.print("Low-level compression");
    log::time("Low-level compression throughput: " +
              std::to_string((double)(total_elems * sizeof(T)) /
                             timer_total.get() / 1e9) +
              " GB/s");
    timer_total.clear();
  }

  return compressed_array;
}

template <DIM D, typename T, typename DeviceType>
Array<D, T, DeviceType>
decompress(Hierarchy<D, T, DeviceType> &hierarchy,
           Array<1, unsigned char, DeviceType> &compressed_array,
           enum error_bound_type type, T tol, T s, T norm, Config config,
           CompressionLowLevelWorkspace<D, T, DeviceType> &workspace) {

  config.apply();

  DeviceRuntime<DeviceType>::SelectDevice(config.dev_id);
  log::info("Select device: " + DeviceRuntime<DeviceType>::GetDeviceName());
  Timer timer_total, timer_each;

  // Workspaces
  if (!workspace.pre_allocated) {
    if (log::level & log::TIME)
      timer_total.start();
    // Allocate workspace if not pre-allocated
    workspace = CompressionLowLevelWorkspace(hierarchy);
    if (log::level & log::TIME) {
      timer_each.end();
      timer_each.print("Allocate workspace");
      timer_each.clear();
    }
  }

  SIZE total_elems = hierarchy.total_num_elems();

  SubArray compressed_subarray(compressed_array);

  LENGTH outlier_count;

  Array<1, Byte, DeviceType> lossless_compressed_array;

  if (log::level & log::TIME)
    timer_total.start();

  if (config.lossless == lossless_type::Huffman_LZ4) {
#ifdef MGARDX_COMPILE_CUDA
    lossless_compressed_array =
        LZ4Decompress<Byte, DeviceType>(compressed_subarray);
    compressed_subarray = SubArray(lossless_compressed_array);
#else
    log::err("LZ4 is only available in CUDA. Portable LZ4 is in development. "
             "Please use the CUDA backend to decompress for now.");
    exit(-1);
#endif
  }

  if (config.lossless == lossless_type::Huffman_Zstd) {
    lossless_compressed_array =
        ZstdDecompress<Byte, DeviceType>(compressed_subarray);
    compressed_subarray = SubArray(lossless_compressed_array);
  }

  // PrintSubarray("compressed_subarray",
  // compressed_subarray);

  if (config.lossless != lossless_type::CPU_Lossless) {
    // Huffman compression
    SubArray<1, QUANTIZED_UNSIGNED_INT, DeviceType> cast_quantized_subarray(
        {total_elems},
        (QUANTIZED_UNSIGNED_INT *)workspace.quantized_subarray.data());
    HuffmanDecompress<QUANTIZED_UNSIGNED_INT, HUFFMAN_CODE, DeviceType>(
        compressed_subarray, cast_quantized_subarray, outlier_count,
        workspace.outlier_idx_subarray, workspace.outliers_subarray);
  } else {
    Array<1, QUANTIZED_INT, DeviceType> quantized_array =
        CPUDecompress<QUANTIZED_INT, DeviceType>(compressed_subarray);
    // We have to copy since quantized_array will be deleted
    MemoryManager<DeviceType>::Copy1D(workspace.quantized_subarray.data(),
                                      quantized_array.data(), total_elems);
  }

  // if (debug_print_compression) {
  // PrintSubarray("Quantized primary", SubArray(primary));
  // }

  Array<D, T, DeviceType> decompressed_data(
      hierarchy.level_shape(hierarchy.l_target()));
  SubArray<D, T, DeviceType> decompressed_subarray(decompressed_data);
  MemoryManager<DeviceType>::Copy1D(workspace.outlier_count_subarray.data(),
                                    &outlier_count, 1);
  SubArray<2, SIZE, DeviceType> level_ranges_subarray(hierarchy.level_ranges(),
                                                      true);
  SubArray<3, T, DeviceType> level_volumes_subarray(hierarchy.level_volumes());

  bool prep_huffman = config.lossless != lossless_type::CPU_Lossless;

  T *quantizers = new T[hierarchy.l_target() + 1];
  calc_quantizers<D, T>(total_elems, quantizers, type, tol, s, norm,
                        hierarchy.l_target(), config.decomposition, false);
  MemoryManager<DeviceType>::Copy1D(workspace.quantizers_subarray.data(),
                                    quantizers, hierarchy.l_target() + 1);
  delete[] quantizers;

  LevelwiseLinearQuantizerND<D, T, MGARDX_DEQUANTIZE, DeviceType>().Execute(
      level_ranges_subarray, hierarchy.l_target(),
      workspace.quantizers_subarray, level_volumes_subarray, s,
      config.huff_dict_size, decompressed_subarray,
      workspace.quantized_subarray, prep_huffman, config.reorder,
      workspace.outlier_count_subarray, workspace.outlier_idx_subarray,
      workspace.outliers_subarray, 0);

  // if (debug_prnt) {
  // PrintSubarray("Dequanzed primary", SubArray(dqv_array));
  // PrintSubarray("decompressed_subarray", decompressed_subarray);
  // }

  if (config.decomposition == decomposition_type::MultiDim) {
    recompose<D, T, DeviceType>(
        hierarchy, decompressed_subarray, workspace.refactoring_w_subarray,
        workspace.refactoring_b_subarray, hierarchy.l_target(), 0);
  } else if (config.decomposition == decomposition_type::SingleDim) {
    recompose_single<D, T, DeviceType>(hierarchy, decompressed_subarray,
                                       hierarchy.l_target(), 0);
  }

  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncQueue(0);
    timer_total.end();
    timer_total.print("Low-level decompression");
    log::time("Low-level decompression throughput: " +
              std::to_string((double)(total_elems * sizeof(T)) /
                             timer_total.get() / 1e9) +
              " GB/s");
    timer_total.clear();
  }

  // if (debug_print_compression) {
  //   PrintSubarray2("decompressed_subarray", SubArray<D, T,
  //   DeviceType>(decompressed_subarray));
  // }

  return decompressed_data;
}
} // namespace mgard_x

#endif