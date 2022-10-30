/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "LosslessCompressorInterface.hpp"
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
    HuffmanCompress(cast_quantized_subarray, config.huff_block_size,
                    config.huff_dict_size,
                    workspace.huffman_workspace.outlier_count,
                    workspace.huffman_workspace.outlier_idx_subarray,
                    workspace.huffman_workspace.outlier_subarray,
                    compressed_array, workspace.huffman_workspace, 0);

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
    HuffmanDecompress(compressed_subarray, cast_quantized_array,
                      workspace.huffman_workspace.outlier_count,
                      workspace.huffman_workspace.outlier_idx_subarray,
                      workspace.huffman_workspace.outlier_subarray,
                      workspace.huffman_workspace, 0);
  } else {
    Array<1, QUANTIZED_INT, DeviceType> quantized_array =
        CPUDecompress<QUANTIZED_INT, DeviceType>(compressed_subarray);
    // We have to copy since quantized_array will be deleted
    MemoryManager<DeviceType>::Copy1D(workspace.quantized_subarray.data(),
                                      quantized_array.data(), total_elems);
  }
}

template <typename T, typename H, typename DeviceType>
class ComposedLosslessCompressor
    : public LosslessCompressorInterface<T, DeviceType> {
public:
  using S = typename std::make_signed<T>::type;
  using Q = typename std::make_unsigned<T>::type;

  ComposedLosslessCompressor(SIZE n, Config config,
                             double estimated_outlier_ratio = 1.0)
      : n(n), config(config),
        huffman(n, config.huff_dict_size, config.huff_block_size,
                estimated_outlier_ratio) {
    static_assert(!std::is_floating_point<T>::value,
                  "ComposedLosslessCompressor: Type of T must be integer.");
  }

  static size_t EstimateMemoryFootprint(SIZE primary_count, SIZE dict_size,
                                        SIZE chunk_size,
                                        double estimated_outlier_ratio = 1) {
    return Huffman<Q, S, H, DeviceType>::EstimateMemoryFootprint(
        primary_count, dict_size, chunk_size, estimated_outlier_ratio);
  }

  void Compress(Array<1, T, DeviceType> &original_data,
                Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {

    huffman.CompressPrimary(original_data, compressed_data, queue_idx);

    if (config.lossless == lossless_type::Huffman_LZ4) {
      LZ4Compress(compressed_data, config.lz4_block_size);
    }

    if (config.lossless == lossless_type::Huffman_Zstd) {
      ZstdCompress(compressed_data, config.zstd_compress_level);
    }
  }

  void Decompress(Array<1, Byte, DeviceType> &compressed_data,
                  Array<1, T, DeviceType> &decompressed_data, int queue_idx) {

    if (config.lossless == lossless_type::Huffman_LZ4) {
      LZ4Decompress(compressed_data);
    }

    if (config.lossless == lossless_type::Huffman_Zstd) {
      ZstdDecompress(compressed_data);
    }

    huffman.DecompressPrimary(compressed_data, decompressed_data, queue_idx);
  }

  SIZE n;
  Config config;
  Huffman<Q, S, H, DeviceType> huffman;
};

} // namespace mgard_x

#endif