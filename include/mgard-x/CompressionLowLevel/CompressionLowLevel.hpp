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

#include "../Hierarchy/Hierarchy.hpp"
#include "../Config/Config.hpp"
#include "../RuntimeX/RuntimeX.h"
#include "CompressionLowLevel.h"

#include "../DataRefactoring/MultiDimension/DataRefactoring.h"
#include "../DataRefactoring/SingleDimension/DataRefactoring.h"

#include "../Quantization/LinearQuantization.hpp"

#include "../Linearization/LevelLinearizer.hpp"

#include "../Lossless/ParallelHuffman/Huffman.hpp"

#ifdef MGARDX_COMPILE_CUDA
#include "../Lossless/Cascaded.hpp"
#include "../Lossless/LZ4.hpp"
#endif

#include "../Lossless/CPU.hpp"
#include "../Lossless/Zstd.hpp"

// for debugging
// #include "../cuda/CommonInternal.h"
// #include "../cuda/DataRefactoring.h"
// #include "../cuda/SubArray.h"

#ifndef MGARD_X_COMPRESSION_LOW_LEVEL_HPP
#define MGARD_X_COMPRESSION_LOW_LEVEL_HPP

#define BLOCK_SIZE 64

using namespace std::chrono;

namespace mgard_x {

static bool debug_print_compression = true;

template <DIM D, typename T, typename DeviceType>
Array<1, unsigned char, DeviceType>
compress(Hierarchy<D, T, DeviceType> &hierarchy,
         Array<D, T, DeviceType> &in_array, enum error_bound_type type, T tol,
         T s, T &norm, Config config, CompressionLowLevelWorkspace<D, T, DeviceType>& workspace) {

  config.apply();

  DeviceRuntime<DeviceType>::SelectDevice(config.dev_id);
  log::info("Select device: " + DeviceRuntime<DeviceType>::GetDeviceName());
  Timer timer_total, timer_each;
  for (int d = D - 1; d >= 0; d--) {
    if (hierarchy.level_shape(hierarchy.l_target(), d) != in_array.shape(d)) {
      log::err("The shape of input array does not match the shape initilized in hierarchy!");
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
    workspace = CompressionLowLevelWorkspace(hierarchy, config);
    if (log::level & log::TIME) {
      timer_each.end();
      timer_each.print("Allocate workspace");
      timer_each.clear();
    }
  }

  Array<1, T, DeviceType> &quantizers_array = workspace.quantizers_array;
  Array<D, QUANTIZED_INT, DeviceType> &quantized_array = workspace.quantized_array;
  Array<1, LENGTH, DeviceType> &outlier_count_array = workspace.outlier_count_array;
  Array<1, LENGTH, DeviceType> &outlier_idx_array = workspace.outlier_idx_array;
  Array<1, QUANTIZED_INT, DeviceType> &outliers_array = workspace.outliers_array;
  Array<1, QUANTIZED_INT, DeviceType> &quantized_linearized_array = workspace.quantized_linearized_array;

  SubArray in_subarray(in_array);
  SIZE total_elems = hierarchy.total_num_elems();

  if (log::level & log::TIME)
    timer_total.start();

  if (type == error_bound_type::REL) {
    if (log::level & log::TIME)
      timer_each.start();

    Array<1, T, DeviceType> temp_array;
    SubArray<1, T, DeviceType> temp_subarray;
    Array<1, T, DeviceType> norm_array({1});
    SubArray<1, T, DeviceType> norm_subarray(norm_array);
    if (!in_array.isPitched()) { // zero copy
      log::info("Use zero copy when calculating norm\n");
      temp_subarray =
          SubArray<1, T, DeviceType>({total_elems}, in_array.data());
    } else { // need to linearized
      log::info("Explicit copy used when calculating norm\n");
      temp_array = Array<1, T, DeviceType>({(SIZE)total_elems}, false);
      MemoryManager<DeviceType>::CopyND(
          temp_array.data(), in_array.shape(D - 1), in_array.data(),
          in_array.ld(D - 1), in_array.shape(D - 1),
          (SIZE)hierarchy.linearized_width(), 0);
      temp_subarray = SubArray<1, T, DeviceType>(temp_array);
    }
    DeviceRuntime<DeviceType>::SyncQueue(0);
    if (s == std::numeric_limits<T>::infinity()) {
      DeviceCollective<DeviceType>::AbsMax(total_elems, temp_subarray,
                                           norm_subarray, 0);
      DeviceRuntime<DeviceType>::SyncQueue(0);
      norm = norm_array.hostCopy()[0];
      log::info("L_inf norm: " +  std::to_string(norm));
    } else {
      DeviceCollective<DeviceType>::SquareSum(total_elems, temp_subarray,
                                              norm_subarray, 0);
      DeviceRuntime<DeviceType>::SyncQueue(0);
      norm = norm_array.hostCopy()[0];
      if (config.uniform_coord_mode == 0) {
        norm = std::sqrt(norm);
      } else {
        norm = std::sqrt(norm / total_elems);
      }
      log::info("L_2 norm: " +  std::to_string(norm));
    }
    if (log::level & log::TIME) {
      timer_each.end();
      timer_each.print("Calculate norm");
      timer_each.clear();
    }
  }

  // Decomposition
  if (log::level & log::TIME)
    timer_each.start();
  if (config.decomposition == decomposition_type::MultiDim) {
    decompose<D, T, DeviceType>(hierarchy, in_subarray, 0, 0);
  } else if (config.decomposition == decomposition_type::SingleDim) {
    decompose_single<D, T, DeviceType>(hierarchy, in_subarray, 0, 0);
  }

  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncQueue(0);
    timer_each.end();
    timer_each.print("Decomposition");
    timer_each.clear();
  }

  // Quantization
  bool prep_huffman =
      config.lossless != lossless_type::CPU_Lossless; // always do Huffman

  if (log::level & log::TIME)
    timer_each.start();

  SubArray<1, T, DeviceType> quantizers_subarray(quantizers_array);
  SubArray<D, QUANTIZED_INT, DeviceType> quantized_subarray(quantized_array);
  SubArray<1, LENGTH, DeviceType> outlier_count_subarray(outlier_count_array);
  SubArray<1, LENGTH, DeviceType> outlier_idx_subarray(outlier_idx_array);
  SubArray<1, QUANTIZED_INT, DeviceType> outliers_subarray(outliers_array);
  SubArray<2, SIZE, DeviceType> level_ranges_subarray(hierarchy.level_ranges());
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
  quantizers_array.load(quantizers);
  delete[] quantizers;

  DeviceRuntime<DeviceType>::SyncQueue(0);

  bool done_quantization = false;
  while (!done_quantization) {
    LevelwiseLinearQuantizerND<D, T, MGARDX_QUANTIZE, DeviceType>().Execute(
        level_ranges_subarray,
        hierarchy.l_target(), quantizers_subarray,
        level_volumes_subarray, s,
        config.huff_dict_size, in_subarray,
        quantized_subarray, prep_huffman, outlier_count_subarray,
        outlier_idx_subarray, outliers_subarray, 0);

    MemoryManager<DeviceType>::Copy1D(&outlier_count, outlier_count_subarray.data(), 
                                     1, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
    if (outlier_count <= outliers_subarray.shape(0)) {
      // outlier buffer has sufficient size
      done_quantization = true;
    } else {
      log::info("Not enough workspace for outliers. Re-allocating to " + std::to_string(outlier_count));
      outlier_idx_array = Array<1, LENGTH, DeviceType>({(SIZE)outlier_count});
      outliers_array = Array<1, QUANTIZED_INT, DeviceType>({(SIZE)outlier_count});
      outlier_idx_subarray = SubArray(outlier_idx_array);
      outliers_subarray = SubArray(outliers_array);
      outlier_count_array.memset(0);
    }
  }

  log::info("Outlier ratio: " + std::to_string(outlier_count) + "/"
              + std::to_string(total_elems) + " ("
              + std::to_string((double)100 * outlier_count / total_elems) + "%)");

  if (log::level & log::TIME) {
    
    timer_each.end();
    timer_each.print("Quantization");
    timer_each.clear();
  }

  

  // if (debug_print) {
  // PrintSubarray("decomposed", SubArray(in_array));
  // PrintSubarray("quantized_subarray", quantized_subarray);
  // PrintSubarray("quantized outliers_array", outliers_subarray);
  // PrintSubarray("quantized outlier_idx_array", outlier_idx_subarray);
  // }

  Array<1, Byte, DeviceType> lossless_compressed_array;
  SubArray<1, Byte, DeviceType> lossless_compressed_subarray;

  SubArray<1, QUANTIZED_UNSIGNED_INT, DeviceType>
      unsigned_quantized_linearized_subarray;
  SubArray<1, QUANTIZED_INT, DeviceType> quantized_linearized_subarray;

  if (config.reorder != 0) {
    if (log::level & log::TIME)
      timer_each.start();
    if (config.reorder == 1) {
      LevelLinearizer<D, QUANTIZED_INT, Interleave, DeviceType>().Execute(
          SubArray<2, SIZE, DeviceType>(hierarchy.level_ranges(), true),
          hierarchy.l_target(), quantized_subarray,
          SubArray<1, QUANTIZED_INT, DeviceType>(quantized_linearized_array),
          0);
    } else {
      log::err("wrong reodering type");
    }
    DeviceRuntime<DeviceType>::SyncDevice();
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(0);
      timer_each.end();
      timer_each.print("Level linearizer type: " +
                       std::to_string(config.reorder));
      timer_each.clear();
    }
    unsigned_quantized_linearized_subarray =
        SubArray<1, QUANTIZED_UNSIGNED_INT, DeviceType>(
            {total_elems},
            (QUANTIZED_UNSIGNED_INT *)quantized_linearized_array.data());
    quantized_linearized_subarray = SubArray<1, QUANTIZED_INT, DeviceType>(
        {total_elems}, (QUANTIZED_INT *)quantized_linearized_array.data());
  } else {
    // Cast to QUANTIZED_UNSIGNED_INT
    unsigned_quantized_linearized_subarray =
        SubArray<1, QUANTIZED_UNSIGNED_INT, DeviceType>(
            {total_elems}, (QUANTIZED_UNSIGNED_INT *)quantized_array.data());
    quantized_linearized_subarray = SubArray<1, QUANTIZED_INT, DeviceType>(
        {total_elems}, (QUANTIZED_INT *)quantized_array.data());
  }

  if (config.lossless != lossless_type::CPU_Lossless) {
    // Huffman compression
    if (log::level & log::TIME)
      timer_each.start();
    lossless_compressed_array =
        HuffmanCompress<QUANTIZED_UNSIGNED_INT, uint64_t, DeviceType>(
            unsigned_quantized_linearized_subarray, config.huff_block_size,
            config.huff_dict_size, outlier_count, outlier_idx_subarray,
            outliers_subarray);
    lossless_compressed_subarray = SubArray(lossless_compressed_array);
    log::info("Huffman block size: " + std::to_string(config.huff_block_size));
    log::info("Huffman dictionary size: " + std::to_string(config.huff_dict_size));
    log::info("Huffman compress ratio: "
               + std::to_string(total_elems * sizeof(QUANTIZED_UNSIGNED_INT)) + "/"
               + std::to_string(lossless_compressed_subarray.shape(0)) + " ("
               + std::to_string((double)total_elems * sizeof(QUANTIZED_UNSIGNED_INT) /
                     lossless_compressed_subarray.shape(0)) + ")");
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(0);
      timer_each.end();
      timer_each.print("Huffman compress");
      timer_each.clear();
    }
  } else {
    if (log::level & log::TIME)
      timer_each.start();
    // PrintSubarray("quantized_linearized_subarray",
    // quantized_linearized_subarray);
    lossless_compressed_array =
        CPUCompress<QUANTIZED_INT, DeviceType>(quantized_linearized_subarray);
    lossless_compressed_subarray = SubArray(lossless_compressed_array);
    log::info("CPU lossless compress ratio: "
                + std::to_string(total_elems * sizeof(QUANTIZED_INT)) + "/"
                + std::to_string(lossless_compressed_subarray.shape(0)) + " ("
                + std::to_string((double)total_elems * sizeof(QUANTIZED_INT) /
                       lossless_compressed_subarray.shape(0)) + ")");
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(0);
      timer_each.end();
      timer_each.print("CPU lossless");
      timer_each.clear();
    }
  }

  // PrintSubarray("lossless_compressed_subarray",
  // lossless_compressed_subarray);

#ifdef MGARDX_COMPILE_CUDA
  // LZ4 compression
  if (config.lossless == lossless_type::Huffman_LZ4) {
    if (log::level & log::TIME)
      timer_each.start();
    SIZE lz4_before_size = lossless_compressed_subarray.shape(0);
    lossless_compressed_array =
        LZ4Compress(lossless_compressed_subarray, config.lz4_block_size);
    lossless_compressed_subarray = SubArray(lossless_compressed_array);
    SIZE lz4_after_size = lossless_compressed_subarray.shape(0);
    log::info("LZ4 block size: " + std::to_string(config.lz4_block_size));
    log::info("LZ4 compress ratio: "
              + std::to_string((double)lz4_before_size / lz4_after_size));
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(0);
      timer_each.end();
      timer_each.print("LZ4 compress");
      timer_each.clear();
    }
  }
#endif

  if (config.lossless == lossless_type::Huffman_Zstd) {
    if (log::level & log::TIME)
      timer_each.start();
    SIZE zstd_before_size = lossless_compressed_subarray.shape(0);
    lossless_compressed_array =
        ZstdCompress(lossless_compressed_subarray, config.zstd_compress_level);
    lossless_compressed_subarray = SubArray(lossless_compressed_array);
    SIZE zstd_after_size = lossless_compressed_subarray.shape(0);
    log::info("Zstd compression level: " + std::to_string(config.zstd_compress_level));
    log::info("Zstd compress ratio: "
              + std::to_string((double)zstd_before_size / zstd_after_size));
    if (log::level & log::TIME) {
      timer_each.end();
      timer_each.print("Zstd Compress");
      timer_each.clear();
    }
  }

  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncQueue(0);
    timer_total.end();
    timer_total.print("Low-level compression");
    log::time("Low-level compression throughput: "
              + std::to_string((double)(total_elems * sizeof(T)) / timer_total.get() / 1e9)
              + " GB/s");
    timer_total.clear();
  }

  return lossless_compressed_array;
}

template <DIM D, typename T, typename DeviceType>
Array<D, T, DeviceType>
decompress(Hierarchy<D, T, DeviceType> &hierarchy,
           Array<1, unsigned char, DeviceType> &compressed_array,
           enum error_bound_type type, T tol, T s, T norm, Config config,
           CompressionLowLevelWorkspace<D, T, DeviceType>& workspace) {

  config.apply();

  DeviceRuntime<DeviceType>::SelectDevice(config.dev_id);
  log::info("Select device: " + DeviceRuntime<DeviceType>::GetDeviceName());
  Timer timer_total, timer_each;

  // Workspaces
  if (!workspace.pre_allocated) {
    if (log::level & log::TIME)
    timer_total.start();
    // Allocate workspace if not pre-allocated
    workspace = CompressionLowLevelWorkspace(hierarchy, config);
    if (log::level & log::TIME) {
      timer_each.end();
      timer_each.print("Allocate workspace");
      timer_each.clear();
    }
  }

  Array<1, T, DeviceType> &quantizers_array = workspace.quantizers_array;
  Array<D, QUANTIZED_INT, DeviceType> &quantized_array = workspace.quantized_array;
  Array<1, LENGTH, DeviceType> &outlier_count_array = workspace.outlier_count_array;

  SIZE total_elems = hierarchy.total_num_elems();

  SubArray compressed_subarray(compressed_array);

  LENGTH outlier_count;

  SubArray<1, Byte, DeviceType> lossless_compressed_subarray =
      compressed_subarray;

  SubArray<1, LENGTH, DeviceType> outlier_idx_subarray;
  SubArray<1, QUANTIZED_INT, DeviceType> outliers_subarray;

  Array<1, Byte, DeviceType> lossless_compressed_array;
  Array<1, Byte, DeviceType> huffman_array;

  if (log::level & log::TIME)
    timer_total.start();

  if (config.lossless == lossless_type::Huffman_LZ4) {
#ifdef MGARDX_COMPILE_CUDA
    if (log::level & log::TIME)
      timer_each.start();
    lossless_compressed_array =
        LZ4Decompress<Byte, DeviceType>(lossless_compressed_subarray);
    lossless_compressed_subarray = SubArray(lossless_compressed_array);
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(0);
      timer_each.end();
      timer_each.print("LZ4 decompress");
      timer_each.clear();
    }
#else
    log::err("LZ4 is only available in CUDA. Portable LZ4 is in development. Please use the CUDA backend to decompress for now.");
    exit(-1);
#endif
  }

  if (config.lossless == lossless_type::Huffman_Zstd) {
    if (log::level & log::TIME)
      timer_each.start();
    lossless_compressed_array =
        ZstdDecompress<Byte, DeviceType>(lossless_compressed_subarray);
    lossless_compressed_subarray = SubArray(lossless_compressed_array);
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(0);
      timer_each.end();
      timer_each.print("Zstd decompress");
      timer_each.clear();
    }
  }

  QUANTIZED_UNSIGNED_INT *unsigned_dqv;

  if (debug_print_compression) {
    // PrintSubarray("Huffman lossless_compressed_subarray",
    // lossless_compressed_subarray);
  }

  // PrintSubarray("lossless_compressed_subarray",
  // lossless_compressed_subarray);

  if (config.lossless != lossless_type::CPU_Lossless) {
    // Huffman compression
    if (log::level & log::TIME)
      timer_each.start();
    Array<1, QUANTIZED_UNSIGNED_INT, DeviceType>
        unsigned_quantized_linearized_array =
            HuffmanDecompress<QUANTIZED_UNSIGNED_INT, uint64_t, DeviceType>(
                lossless_compressed_subarray, outlier_count,
                outlier_idx_subarray, outliers_subarray);

    if (config.reorder != 0) {
      if (log::level & log::TIME)
        timer_each.start();
      if (config.reorder == 1) {
        LevelLinearizer<D, QUANTIZED_INT, Reposition, DeviceType>().Execute(
            SubArray<2, SIZE, DeviceType>(hierarchy.level_ranges(), true),
            hierarchy.l_target(), SubArray(quantized_array),
            SubArray<1, QUANTIZED_INT, DeviceType>(
                {total_elems},
                (QUANTIZED_INT *)unsigned_quantized_linearized_array.data()),
            0);
      } else {
        log::err("wrong reodering option.");
        exit(-1);
      }
      if (log::level & log::TIME) {
        DeviceRuntime<DeviceType>::SyncQueue(0);
        timer_each.end();
        timer_each.print("Level linearization");
        timer_each.clear();
      }
    } else {
      MemoryManager<DeviceType>::Copy1D(
          quantized_array.data(),
          (QUANTIZED_INT *)unsigned_quantized_linearized_array.data(),
          total_elems, 0);
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(0);
      timer_each.end();
      timer_each.print("Huffman decompress");
      timer_each.clear();
    }
  } else {
    if (log::level & log::TIME)
      timer_each.start();
    Array<1, QUANTIZED_INT, DeviceType> quantized_linearized_array =
        CPUDecompress<QUANTIZED_INT, DeviceType>(lossless_compressed_subarray);

    if (config.reorder != 0) {
      if (log::level & log::TIME)
        timer_each.start();
      if (config.reorder == 1) {
        LevelLinearizer<D, QUANTIZED_INT, Reposition, DeviceType>().Execute(
            SubArray<2, SIZE, DeviceType>(hierarchy.level_ranges(), true),
            hierarchy.l_target(), SubArray(quantized_array),
            SubArray<1, QUANTIZED_INT, DeviceType>(quantized_linearized_array),
            0);
      } else {
        log::err("wrong reodering type.");
        exit(-1);
      }
      if (log::level & log::TIME) {
        DeviceRuntime<DeviceType>::SyncQueue(0);
        timer_each.end();
        timer_each.print("Level linearization");
        timer_each.clear();
      }
    } else {
      // PrintSubarray("quantized_linearized_array",
      // SubArray(quantized_linearized_array));
      MemoryManager<DeviceType>::Copy1D(
          quantized_array.data(),
          (QUANTIZED_INT *)quantized_linearized_array.data(), total_elems, 0);
    }
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(0);
      timer_each.end();
      timer_each.print("CPU lossless");
      timer_each.clear();
    }
  }

  // if (debug_print_compression) {
  // PrintSubarray("Quantized primary", SubArray(primary));
  // }

  if (log::level & log::TIME)
    timer_each.start();

  Array<D, T, DeviceType> decompressed_data(
      hierarchy.level_shape(hierarchy.l_target()));
  SubArray<D, T, DeviceType> decompressed_subarray(decompressed_data);
  outlier_count_array.load(&outlier_count);
  SubArray<2, SIZE, DeviceType> level_ranges_subarray(hierarchy.level_ranges());
  SubArray<3, T, DeviceType> level_volumes_subarray(hierarchy.level_volumes());
  SubArray<D, QUANTIZED_INT, DeviceType> quantized_subarray(quantized_array);
  SubArray<1, LENGTH, DeviceType> outlier_count_subarray(outlier_count_array);

  bool prep_huffman = config.lossless != lossless_type::CPU_Lossless;

  T *quantizers = new T[hierarchy.l_target() + 1];
  calc_quantizers<D, T>(total_elems, quantizers, type, tol, s, norm,
                        hierarchy.l_target(), config.decomposition, false);
  quantizers_array.load(quantizers);
  SubArray<1, T, DeviceType> quantizers_subarray(quantizers_array);
  delete[] quantizers;

  LevelwiseLinearQuantizerND<D, T, MGARDX_DEQUANTIZE, DeviceType>().Execute(
      level_ranges_subarray,
      hierarchy.l_target(), quantizers_subarray,
      level_volumes_subarray, s,
      config.huff_dict_size, decompressed_subarray,
      quantized_subarray, prep_huffman,
      outlier_count_subarray,
      outlier_idx_subarray, outliers_subarray, 0);

  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncQueue(0);
    timer_each.end();
    timer_each.print("Dequantization");
    timer_each.clear();
  }

  // if (debug_prnt) {
  // PrintSubarray("Dequanzed primary", SubArray(dqv_array));
  // PrintSubarray("decompressed_subarray", decompressed_subarray);
  // }

  if (log::level & log::TIME)
    timer_each.start();
  if (config.decomposition == decomposition_type::MultiDim) {
    recompose<D, T, DeviceType>(hierarchy, decompressed_subarray,
                                hierarchy.l_target(), 0);
  } else if (config.decomposition == decomposition_type::SingleDim) {
    recompose_single<D, T, DeviceType>(hierarchy, decompressed_subarray,
                                       hierarchy.l_target(), 0);
  }

  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncQueue(0);
    timer_each.end();
    timer_each.print("Recomposition");
    timer_each.clear();
  }

  if (log::level & log::TIME) {
    DeviceRuntime<DeviceType>::SyncQueue(0);
    timer_total.end();
    timer_total.print("Low-level decompression");
    log::time("Low-level decompression throughput: "
              + std::to_string((double)(total_elems * sizeof(T)) / timer_total.get() / 1e9)
              + " GB/s");
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