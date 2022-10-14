/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "ParallelHuffman/Huffman.hpp"
#ifdef MGARDX_COMPILE_CUDA
#include "../Lossless/Cascaded.hpp"
#include "../Lossless/LZ4.hpp"
#endif
#include "CPU.hpp"
#include "Zstd.hpp"

#ifndef MGARD_X_LOSSLESS_TEMPLATE_HPP
#define MGARD_X_LOSSLESS_TEMPLATE_HPP

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
void LosslessCompress(
    Hierarchy<D, T, DeviceType> &hierarchy,
    Array<1, Byte, DeviceType> &compressed_array, Config config,
    CompressionLowLevelWorkspace<D, T, DeviceType> &workspace) {
  SIZE total_elems = hierarchy.total_num_elems();
  if (config.lossless != lossless_type::CPU_Lossless) {
    // Huffman compression
    // Cast to 1D unsigned integers when do CPU compression
    SubArray<1, QUANTIZED_UNSIGNED_INT, DeviceType> cast_quantized_subarray =
        SubArray<1, QUANTIZED_UNSIGNED_INT, DeviceType>(
            {total_elems},
            (QUANTIZED_UNSIGNED_INT *)workspace.quantized_subarray.data());
    HuffmanCompress<QUANTIZED_UNSIGNED_INT, HUFFMAN_CODE, DeviceType>(
        cast_quantized_subarray, config.huff_block_size, config.huff_dict_size,
        workspace.outlier_count, workspace.outlier_idx_subarray,
        workspace.outliers_subarray, compressed_array,
        workspace.huffman_workspace);

    if (config.lossless == lossless_type::Huffman_LZ4) {
#ifdef MGARDX_COMPILE_CUDA
      LZ4Compress(compressed_array, config.lz4_block_size);
#else
      config.lossless = lossless_type::Huffman;
      log::warn("LZ for is only available on CUDA devices. Portable version is "
                "in development.");
#endif
    }

    if (config.lossless == lossless_type::Huffman_Zstd) {
      ZstdCompress(compressed_array, config.zstd_compress_level);
    }
  } else {
    // Cast to 1D signed integers when do CPU compression
    SubArray<1, QUANTIZED_INT, DeviceType> cast_linearized_subarray =
        SubArray<1, QUANTIZED_INT, DeviceType>(
            {total_elems},
            (QUANTIZED_INT *)workspace.quantized_subarray.data());

    compressed_array =
        CPUCompress<QUANTIZED_INT, DeviceType>(cast_linearized_subarray);
  }
}

template <DIM D, typename T, typename DeviceType>
void LosslessDecompress(
    Hierarchy<D, T, DeviceType> &hierarchy,
    Array<1, Byte, DeviceType> &compressed_array, Config config,
    CompressionLowLevelWorkspace<D, T, DeviceType> &workspace) {
  SIZE total_elems = hierarchy.total_num_elems();
  SubArray compressed_subarray(compressed_array);
  if (config.lossless != lossless_type::CPU_Lossless) {
    if (config.lossless == lossless_type::Huffman_LZ4) {
#ifdef MGARDX_COMPILE_CUDA
      LZ4Decompress(compressed_array);
      compressed_subarray = SubArray(compressed_array);
#else
      log::err("LZ4 is only available in CUDA. Portable LZ4 is in development. "
               "Please use the CUDA backend to decompress for now.");
      exit(-1);
#endif
    }

    if (config.lossless == lossless_type::Huffman_Zstd) {
      ZstdDecompress(compressed_array);
      compressed_subarray = SubArray(compressed_array);
    }

    Array<1, QUANTIZED_UNSIGNED_INT, DeviceType> cast_quantized_array(
        {total_elems},
        (QUANTIZED_UNSIGNED_INT *)workspace.quantized_subarray.data());
    HuffmanDecompress<QUANTIZED_UNSIGNED_INT, HUFFMAN_CODE, DeviceType>(
        compressed_subarray, cast_quantized_array, workspace.outlier_count,
        workspace.outlier_idx_subarray, workspace.outliers_subarray,
        workspace.huffman_workspace);
  } else {
    Array<1, QUANTIZED_INT, DeviceType> quantized_array =
        CPUDecompress<QUANTIZED_INT, DeviceType>(compressed_subarray);
    // We have to copy since quantized_array will be deleted
    MemoryManager<DeviceType>::Copy1D(workspace.quantized_subarray.data(),
                                      quantized_array.data(), total_elems);
  }
}

} // namespace mgard_x

#endif