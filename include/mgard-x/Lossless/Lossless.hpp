/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "CPU.hpp"
#include "Cascaded.hpp"
#include "LZ4.hpp"
#include "LosslessCompressorInterface.hpp"
#include "ParallelHuffman/Huffman.hpp"
#include "Zstd.hpp"

#ifndef MGARD_X_LOSSLESS_TEMPLATE_HPP
#define MGARD_X_LOSSLESS_TEMPLATE_HPP

namespace mgard_x {

template <typename T, typename H, typename DeviceType>
class ComposedLosslessCompressor
    : public LosslessCompressorInterface<T, DeviceType> {
public:
  using S = typename std::make_signed<T>::type;
  using Q = typename std::make_unsigned<T>::type;

  ComposedLosslessCompressor() : initialized(false) {}

  ComposedLosslessCompressor(SIZE n, Config config)
      : initialized(true), n(n), config(config),
        huffman(n, config.huff_dict_size, config.huff_block_size,
                config.estimate_outlier_ratio) {
    static_assert(!std::is_floating_point<T>::value,
                  "ComposedLosslessCompressor: Type of T must be integer.");
    if (config.lossless == lossless_type::Huffman_LZ4) {
      lz4.Resize(n * sizeof(H), config.lz4_block_size, 0);
    }
    if (config.lossless == lossless_type::Huffman_Zstd) {
      zstd.Resize(n * sizeof(H), config.zstd_compress_level, 0);
    }
    DeviceRuntime<DeviceType>::SyncQueue(0);
  }

  void Adapt(SIZE n, Config config, int queue_idx) {
    this->initialized = true;
    this->n = n;
    this->config = config;
    huffman.Resize(n, config.huff_dict_size, config.huff_block_size,
                   config.estimate_outlier_ratio, queue_idx);
    if (config.lossless == lossless_type::Huffman_LZ4) {
      lz4.Resize(n * sizeof(H), config.lz4_block_size, queue_idx);
    }
    if (config.lossless == lossless_type::Huffman_Zstd) {
      zstd.Resize(n * sizeof(H), config.zstd_compress_level, queue_idx);
    }
  }

  static size_t EstimateMemoryFootprint(SIZE primary_count, Config config) {
    size_t size = Huffman<Q, S, H, DeviceType>::EstimateMemoryFootprint(
        primary_count, config.huff_dict_size, config.huff_block_size,
        config.estimate_outlier_ratio);
    if (config.lossless == lossless_type::Huffman_LZ4) {
      size += LZ4<DeviceType>::EstimateMemoryFootprint(
          primary_count * sizeof(H), config.lz4_block_size);
    }
    if (config.lossless == lossless_type::Huffman_Zstd) {
      size +=
          Zstd<DeviceType>::EstimateMemoryFootprint(primary_count * sizeof(H));
    }
    return size;
  }

  void Compress(Array<1, T, DeviceType> &original_data,
                Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {

    huffman.CompressPrimary(original_data, compressed_data, queue_idx);

    if (config.lossless == lossless_type::Huffman_LZ4) {
      lz4.Compress(compressed_data, queue_idx);
    }

    if (config.lossless == lossless_type::Huffman_Zstd) {
      zstd.Compress(compressed_data, queue_idx);
    }
  }

  void Decompress(Array<1, Byte, DeviceType> &compressed_data,
                  Array<1, T, DeviceType> &decompressed_data, int queue_idx) {

    if (config.lossless == lossless_type::Huffman_LZ4) {
      lz4.Decompress(compressed_data, queue_idx);
    }

    if (config.lossless == lossless_type::Huffman_Zstd) {
      zstd.Decompress(compressed_data, queue_idx);
    }

    huffman.DecompressPrimary(compressed_data, decompressed_data, queue_idx);
  }

  bool initialized;
  SIZE n;
  Config config;
  Huffman<Q, S, H, DeviceType> huffman;
  LZ4<DeviceType> lz4;
  Zstd<DeviceType> zstd;
};

} // namespace mgard_x

#endif