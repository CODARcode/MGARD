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


#include "Types.h"

#include "Hierarchy.hpp"
#include "RuntimeX/RuntimeX.h"
#include "CompressionWorkflow.h"
#include "DataRefactoring/MultiDimension/DataRefactoring.h"
#include "Quantization/LinearQuantization.hpp"
#include "Lossless/ParallelHuffman/Huffman.hpp"

#ifdef MGARDX_COMPILE_CUDA
#include "Lossless/LZ4.hpp"
#include "Lossless/Cascaded.hpp"
#endif

#include "Lossless/Zstd.hpp"
#include "Utilities/CheckEndianess.h"

// for debugging
// #include "../cuda/CommonInternal.h"
// #include "../cuda/DataRefactoring.h"
// #include "../cuda/SubArray.h"

#define BLOCK_SIZE 64

using namespace std::chrono;

namespace mgard_x {

static bool debug_print = true;

template <DIM D, typename T, typename DeviceType>
Array<1, unsigned char, DeviceType> compress(Hierarchy<D, T, DeviceType> &hierarchy, Array<D, T, DeviceType> &in_array,
                                 enum error_bound_type type, T tol, T s, T &norm, Config config) {
  DeviceRuntime<DeviceType>::SelectDevice(config.dev_id);
  Timer timer_total, timer_each;
  for (DIM i = 0; i < D; i++) {
    if (hierarchy.shape[i] != in_array.getShape()[i]) {
      std::cout << log::log_err
                << "The shape of input array does not match the shape "
                   "initilized in hierarchy!\n";
      std::vector<SIZE> empty_shape;
      empty_shape.push_back(1);
      Array<1, unsigned char, DeviceType> empty(empty_shape);
      return empty;
    }
  }

  SubArray in_subarray(in_array);
  SIZE total_elems = hierarchy.dofs[0][0] * hierarchy.dofs[1][0] * hierarchy.linearized_depth;

  if (config.timing) timer_total.start();
  // norm = (T)1.0;

  if (type == error_bound_type::REL && norm == 1.0) {
    if (config.timing) timer_each.start();

    Array<1, T, DeviceType> temp_array;
    SubArray<1, T, DeviceType> temp_subarray;
    Array<1, T, DeviceType> norm_array({1});
    SubArray<1, T, DeviceType> norm_subarray(norm_array);
    if (MemoryManager<DeviceType>::ReduceMemoryFootprint) { // zero copy
      temp_subarray = SubArray<1, T, DeviceType>({total_elems}, in_array.get_dv());
    } else { // need to linearized
      temp_array = Array<1, T, DeviceType>(
            {(SIZE)(hierarchy.dofs[0][0] * hierarchy.dofs[1][0] * hierarchy.linearized_depth)}, 
            false);
      MemoryManager<DeviceType>().CopyND(
                        temp_array.get_dv(), hierarchy.dofs[0][0],
                        in_array.get_dv(), in_array.get_ldvs_h()[0], 
                        hierarchy.dofs[0][0], (SIZE)(hierarchy.dofs[1][0] * hierarchy.linearized_depth),
                        0);
      temp_subarray = SubArray<1, T, DeviceType>(temp_array);
    }
    DeviceRuntime<DeviceType>::SyncQueue(0);
    if (s == std::numeric_limits<T>::infinity()) {
      DeviceCollective<DeviceType>::AbsMax(total_elems, temp_subarray, norm_subarray, 0);
      DeviceRuntime<DeviceType>::SyncQueue(0);
      norm = norm_array.getDataHost()[0];
    } else {
      DeviceCollective<DeviceType>::SquareSum(total_elems, temp_subarray, norm_subarray, 0);
      DeviceRuntime<DeviceType>::SyncQueue(0);
      norm = norm_array.getDataHost()[0];
      norm = std::sqrt(norm);
    }
    if (config.timing) {
      timer_each.end();
      timer_each.print("Calculating norm");
      timer_each.clear();
      if (s == std::numeric_limits<T>::infinity()) {
        std::cout << log::log_info << "L_inf norm: " << norm << std::endl;
      } else {
        std::cout << log::log_info << "L_2 norm: " << norm << std::endl;
      }
    }
  }
  
  // Decomposition
  if (config.timing) timer_each.start();
  decompose<D, T, DeviceType>(hierarchy, in_subarray, hierarchy.l_target, 0);


  if (config.timing) {
    DeviceRuntime<DeviceType>::SyncQueue(0);
    timer_each.end();
    timer_each.print("Decomposition");
    timer_each.clear();
  }

  // Quantization
  bool prep_huffman = true; // always do Huffman

  if (config.timing) timer_each.start();

  Array<D, QUANTIZED_INT, DeviceType> dqv_array(hierarchy.shape_org, false);

  LENGTH estimate_outlier_count = (double)total_elems * 1;
  LENGTH zero = 0, outlier_count;
  Array<1, LENGTH, DeviceType> outlier_count_array({1});
  Array<1, LENGTH, DeviceType> outlier_idx_array({(SIZE)estimate_outlier_count});
  Array<1, QUANTIZED_INT, DeviceType> outliers_array({(SIZE)estimate_outlier_count});
  MemoryManager<DeviceType>::Copy1D(outlier_count_array.get_dv(), &zero, 1, 0);
  MemoryManager<DeviceType>::Memset1D(outlier_idx_array.get_dv(), estimate_outlier_count, 0);
  MemoryManager<DeviceType>::Memset1D(outliers_array.get_dv(), estimate_outlier_count, 0);


#ifndef MGARDX_COMPILE_CUDA
  if (config.lossless == lossless_type::Huffman_LZ4) {
    std::cout << log::log_warn << "LZ4 only available in CUDA. Switched to Zstd.\n";
    config.lossless = lossless_type::Huffman_Zstd;
  }
#endif

  T *quantizers = new T[hierarchy.l_target + 1];
  size_t dof = 1;
  for (int d = 0; d < D; d++) dof *= hierarchy.dofs[d][0];
  calc_quantizers<D, T>(dof, quantizers, type, tol, s, norm, hierarchy.l_target, false);
  Array<1, T, DeviceType> quantizers_array({hierarchy.l_target + 1});
  quantizers_array.loadData(quantizers);
  SubArray<1, T, DeviceType> quantizers_subarray(quantizers_array);
  delete [] quantizers;

  SubArray<1, LENGTH, DeviceType> outlier_idx_subarray(outlier_idx_array);
  SubArray<1, QUANTIZED_INT, DeviceType> outliers_subarray(outliers_array);

  LevelwiseLinearQuantizeND<D, T, DeviceType>().Execute(
          SubArray<1, SIZE, DeviceType>(hierarchy.ranges), hierarchy.l_target, quantizers_subarray,
          SubArray<2, T, DeviceType>(hierarchy.volumes_array), 
          s, config.huff_dict_size, SubArray<D, T, DeviceType>(in_array),
          SubArray<D, QUANTIZED_INT, DeviceType>(dqv_array), prep_huffman,
          SubArray<1, SIZE, DeviceType>(hierarchy.shapes[0], true),
          SubArray<1, LENGTH, DeviceType>(outlier_count_array), outlier_idx_subarray,
          outliers_subarray,
          0);
  MemoryManager<DeviceType>::Copy1D(&outlier_count, outlier_count_array.get_dv(), 1, 0);
  DeviceRuntime<DeviceType>::SyncDevice();
  // m.huff_outlier_count = outlier_count;
  if (config.timing) {
    timer_each.end();
    timer_each.print("Quantization");
    timer_each.clear();
    std::cout << log::log_info << "Outlier ratio: " << 
    outlier_count << "/" << total_elems << " (" <<
    (double)100*outlier_count/total_elems << "%)\n"; 
  }
  if (debug_print) {
     // PrintSubarray("decompresed", SubArray<D, T, Serial>(in_array_serial));
    // PrintSubarray("quantized parimary", SubArray<D, QUANTIZED_INT, DeviceType>(dqv_array));
    // std::cout << "outlier_count: " << outlier_count << std::endl;
    // PrintSubarray("quantized outliers_array", SubArray<1, QUANTIZED_INT, DeviceType>(outliers_array));
    // PrintSubarray("quantized outlier_idx_array", SubArray<1, LENGTH, DeviceType>(outlier_idx_array));
  }

  // Huffman compression
  if (config.timing) timer_each.start();

  Array<1, Byte, DeviceType> lossless_compressed_array;
  SubArray<1, Byte, DeviceType> lossless_compressed_subarray;

  // Cast to QUANTIZED_UNSIGNED_INT
  SubArray<1, QUANTIZED_UNSIGNED_INT, DeviceType> qv({total_elems},
                                                  (QUANTIZED_UNSIGNED_INT*)dqv_array.get_dv());
  lossless_compressed_array =
  HuffmanCompress<QUANTIZED_UNSIGNED_INT, uint64_t, DeviceType>(
      qv, config.huff_block_size, config.huff_dict_size, outlier_count, outlier_idx_subarray, outliers_subarray);
  lossless_compressed_subarray = SubArray(lossless_compressed_array);
  
  if (config.timing) {
    timer_each.end();
    timer_each.print("Huffman Compress");
    std::cout << log::log_info << "Huffman block size: " << 
      config.huff_block_size << "\n"; 
    std::cout << log::log_info << "Huffman dictionary size: " << 
      config.huff_dict_size << "\n"; 
    std::cout << log::log_info << "Huffman compress ratio: " << 
      total_elems*sizeof(QUANTIZED_UNSIGNED_INT) << "/" <<
      lossless_compressed_subarray.getShape(0) << " (" <<
      (double)total_elems*sizeof(QUANTIZED_UNSIGNED_INT) / lossless_compressed_subarray.getShape(0) << ")\n"; 
    timer_each.clear();
  }


#ifdef MGARDX_COMPILE_CUDA
  // LZ4 compression
  if (config.lossless == lossless_type::Huffman_LZ4) {
    if (config.timing) timer_each.start();
    SIZE lz4_before_size = lossless_compressed_subarray.getShape(0);
    lossless_compressed_array = 
    LZ4Compress(lossless_compressed_subarray, config.lz4_block_size);
    lossless_compressed_subarray = SubArray(lossless_compressed_array);
    SIZE lz4_after_size = lossless_compressed_subarray.getShape(0);
    if (config.timing) {
      timer_each.end();
      timer_each.print("LZ4 Compress");
      std::cout << log::log_info << "LZ4 block size: " << 
      config.lz4_block_size << "\n"; 
      std::cout << log::log_info << "LZ4 compress ratio: " << 
      (double)lz4_before_size / lz4_after_size << "\n"; 
      timer_each.clear();
    }
  } 
#endif

  if (config.lossless == lossless_type::Huffman_Zstd) {
    if (config.timing) timer_each.start();
    SIZE zstd_before_size = lossless_compressed_subarray.getShape(0);
    lossless_compressed_array = 
    ZstdCompress(lossless_compressed_subarray, config.zstd_compress_level);
    lossless_compressed_subarray = SubArray(lossless_compressed_array);
    SIZE zstd_after_size = lossless_compressed_subarray.getShape(0);
    if (config.timing) {
      timer_each.end();
      timer_each.print("Zstd Compress");
      std::cout << log::log_info << "Zstd compression level: " << 
      config.zstd_compress_level << "\n"; 
      std::cout << log::log_info << "Zstd compress ratio: " << 
      (double)zstd_before_size / zstd_after_size << "\n"; 
      timer_each.clear();
    }
  }

  if (config.timing) {
    timer_total.end();
    timer_total.print("Overall Compress");
    std::cout << log::log_time << "Compression Throughput: " <<
      (double)(total_elems*sizeof(T))/timer_total.get()/1e9 << " GB/s\n";
    timer_total.clear();
  }

  return lossless_compressed_array;
}

template <DIM D, typename T, typename DeviceType>
Array<D, T, DeviceType> decompress(Hierarchy<D, T, DeviceType> &hierarchy,
                       Array<1, unsigned char, DeviceType> &compressed_array, 
                       enum error_bound_type type, T tol, T s, T norm, Config config) {
  DeviceRuntime<DeviceType>::SelectDevice(config.dev_id);
  Timer timer_total, timer_each;

  SIZE total_elems = hierarchy.dofs[0][0] * hierarchy.dofs[1][0] * hierarchy.linearized_depth;

  SubArray compressed_subarray(compressed_array);

  LENGTH outlier_count;

  SubArray<1, Byte, DeviceType> lossless_compressed_subarray = compressed_subarray;

  SubArray<1, LENGTH, DeviceType> outlier_idx_subarray;
  SubArray<1, QUANTIZED_INT, DeviceType> outliers_subarray;

  Array<1, Byte, DeviceType> lossless_compressed_array;
  Array<1, Byte, DeviceType> huffman_array;

  if (config.timing) timer_total.start();

  if (config.lossless == lossless_type::Huffman_LZ4) {
#ifdef MGARDX_COMPILE_CUDA
    if (config.timing) timer_each.start();
    lossless_compressed_array = LZ4Decompress<Byte, DeviceType>(lossless_compressed_subarray);
    lossless_compressed_subarray = SubArray(lossless_compressed_array);
    DeviceRuntime<DeviceType>::SyncDevice();
    if (config.timing) {
      timer_each.end();
      timer_each.print("LZ4 Decompress");
      timer_each.clear();
    }
#else
  std::cout << log::log_err << "LZ4 only available in CUDA. Portable LZ4 is in development. Please use the CUDA backend to decompress for now.\n";
  exit(-1);
#endif
  }


  if (config.lossless == lossless_type::Huffman_Zstd) {
    if (config.timing) timer_each.start();
    lossless_compressed_array = ZstdDecompress<Byte, DeviceType>(lossless_compressed_subarray);
    lossless_compressed_subarray =  SubArray(lossless_compressed_array);
    DeviceRuntime<DeviceType>::SyncDevice();
    if (config.timing) {
      timer_each.end();
      timer_each.print("Zstd Decompress");
      timer_each.clear();
    }
  }

  QUANTIZED_UNSIGNED_INT * unsigned_dqv;
  
  if (debug_print) {
    // PrintSubarray("Huffman lossless_compressed_subarray", lossless_compressed_subarray);
  }

  if (config.timing) timer_each.start();
  Array<1, QUANTIZED_UNSIGNED_INT, DeviceType> primary =
  HuffmanDecompress<QUANTIZED_UNSIGNED_INT, uint64_t, DeviceType>(lossless_compressed_subarray,
                                                                  outlier_count,
                                                                  outlier_idx_subarray,
                                                                  outliers_subarray);
  DeviceRuntime<DeviceType>::SyncDevice();
  if (config.timing) {
    timer_each.end();
    timer_each.print("Huffman Decompress");
    timer_each.clear();
  }

  // if (debug_print) {
  //   PrintSubarray("Quantized primary", SubArray(primary));
  // }


  if (config.timing) timer_each.start();

  std::vector<SIZE> decompressed_shape(D);
  for (int i = 0; i < D; i++)
    decompressed_shape[i] = hierarchy.shape[i];
  std::reverse(decompressed_shape.begin(), decompressed_shape.end());
  Array<D, T, DeviceType> decompressed_data(decompressed_shape);
  SubArray<D, T, DeviceType> decompressed_subarray(decompressed_data);

  bool prep_huffman = true;

  Array<D, QUANTIZED_INT, DeviceType> dqv_array(hierarchy.shape_org, false);
  MemoryManager<DeviceType>::Copy1D(dqv_array.get_dv(), (QUANTIZED_INT*)primary.get_dv(), total_elems, 0);

  T *quantizers = new T[hierarchy.l_target + 1];
  size_t dof = 1;
  for (int d = 0; d < D; d++) dof *= hierarchy.dofs[d][0];
  calc_quantizers<D, T>(dof, quantizers, type, tol, s, norm, hierarchy.l_target, false);
  Array<1, T, DeviceType> quantizers_array({hierarchy.l_target + 1});
  quantizers_array.loadData(quantizers);
  SubArray<1, T, DeviceType> quantizers_subarray(quantizers_array);
  delete [] quantizers;

  LevelwiseLinearDequantizeND<D, T, DeviceType>().Execute(
            SubArray<1, SIZE, DeviceType>(hierarchy.ranges), hierarchy.l_target, 
            quantizers_subarray,
            SubArray<2, T, DeviceType>(hierarchy.volumes_array), 
            s, config.huff_dict_size, decompressed_subarray,
            SubArray<D, QUANTIZED_INT, DeviceType>(dqv_array), prep_huffman,
            SubArray<1, SIZE, DeviceType>(hierarchy.shapes[0], true),
            outlier_count, outlier_idx_subarray, outliers_subarray, 0);

  DeviceRuntime<DeviceType>::SyncDevice();

  // hierarchy.sync_all();
  if (config.timing) {
    timer_each.end();
    timer_each.print("Dequantization");
    timer_each.clear();
  }


  // if (debug_print) {
  //   PrintSubarray2("Dequanzed primary", SubArray(primary));
  // }

  if (config.timing) timer_each.start();
  recompose<D, T, DeviceType>(hierarchy, decompressed_subarray, hierarchy.l_target, 0);
  // hierarchy.sync_all();
  if (config.timing) {
    timer_each.end();
    timer_each.print("Recomposition");
    timer_each.clear();
  }

  // hierarchy.sync_all();
  if (config.timing) {
    timer_total.end();
    timer_total.print("Overall Decompression");
    std::cout << log::log_time << "Decompression Throughput: " << 
      (double)(total_elems*sizeof(T))/timer_total.get()/1e9 << " GB/s\n";
    timer_total.clear();
  }

  // if (debug_print) {
  //   PrintSubarray2("decompressed_subarray", SubArray<D, T, DeviceType>(decompressed_subarray));
  // }


  return decompressed_data;
}

} // namespace mgard_x
