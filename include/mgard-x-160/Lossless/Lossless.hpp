/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#include "BlockDelta/BlockDelta.hpp"
#include "CPU.hpp"
#include "LZ4/LZ4.hpp"
#include "LosslessCompressorInterface.hpp"
#include "ParallelHuffman/Huffman.hpp"
#include "ParallelRLE/ZeroRunLengthEncoding.hpp"
#include "SymbolRans/SymbolRans.hpp"
#include "Zstd.hpp"
#include "rANS/Rans.hpp"

#ifndef MGARD_X_160_LOSSLESS_TEMPLATE_HPP
#define MGARD_X_160_LOSSLESS_TEMPLATE_HPP

namespace mgard_x_160 {

template <typename T, typename H, typename DeviceType>
class ComposedLosslessCompressor
    : public LosslessCompressorInterface<T, DeviceType> {
public:
  using S = typename std::make_signed<T>::type;
  using Q = typename std::make_unsigned<T>::type;

  ComposedLosslessCompressor() : initialized(false) {}

  // Whether the configured lossless path actually uses the (workspace-heavy)
  // Huffman backend. BlockDelta and standalone LZ4 have their own
  // (de)compressors and never touch the Huffman workspace; every other type is
  // Huffman / Huffman+LZ4 / Huffman+Zstd (or CPU_Lossless falling through to
  // Huffman in Compress()).
  static bool uses_huffman(enum lossless_type lossless) {
    return lossless != lossless_type::BlockDelta &&
           lossless != lossless_type::LZ4 &&
           lossless != lossless_type::ZeroRLE_Rans &&
           lossless != lossless_type::SymbolRans;
  }

  // Worst-case byte size of the RLE0 blob fed to rANS: a (count, symbol) pair
  // per element when nothing repeats (uint32 count + T symbol), plus slack.
  static SIZE rle_rans_bound(SIZE n) {
    return n * (sizeof(uint32_t) + sizeof(T)) + 64;
  }

  ComposedLosslessCompressor(SIZE n, Config config)
      : initialized(true), n(n), config(config) {
    static_assert(!std::is_floating_point<T>::value,
                  "ComposedLosslessCompressor: Type of T must be integer.");
    if (uses_huffman(config.lossless)) {
      huffman.Resize(n, config.huff_dict_size, config.huff_block_size,
                     config.estimate_outlier_ratio, 0);
    }
    if (config.lossless == lossless_type::Huffman_LZ4) {
      lz4.Resize(n * sizeof(H), config.lz4_block_size, 0);
    }
    if (config.lossless == lossless_type::LZ4) {
      lz4.Resize(n * sizeof(T), config.lz4_block_size, 0);
    }
    if (config.lossless == lossless_type::Huffman_Zstd) {
      zstd.Resize(n * sizeof(H), config.zstd_compress_level, 0);
    }
    if (config.lossless == lossless_type::BlockDelta) {
      blockdelta.Resize(n, config.block_delta_block_size,
                        config.block_delta_mode, 0);
    }
    if (config.lossless == lossless_type::ZeroRLE_Rans) {
      zerorle.Resize(n, 0);
      rans.Resize(rle_rans_bound(n), 256, 0);
    }
    if (config.lossless == lossless_type::SymbolRans) {
      symbolrans.Resize(n, config.huff_dict_size, config.estimate_outlier_ratio,
                        0);
    }
    DeviceRuntime<DeviceType>::SyncQueue(0);
  }

  void Adapt(SIZE n, Config config, int queue_idx) {
    this->initialized = true;
    this->n = n;
    this->config = config;
    if (uses_huffman(config.lossless)) {
      huffman.Resize(n, config.huff_dict_size, config.huff_block_size,
                     config.estimate_outlier_ratio, queue_idx);
    }
    if (config.lossless == lossless_type::Huffman_LZ4) {
      lz4.Resize(n * sizeof(H), config.lz4_block_size, queue_idx);
    }
    if (config.lossless == lossless_type::LZ4) {
      lz4.Resize(n * sizeof(T), config.lz4_block_size, queue_idx);
    }
    if (config.lossless == lossless_type::Huffman_Zstd) {
      zstd.Resize(n * sizeof(H), config.zstd_compress_level, queue_idx);
    }
    if (config.lossless == lossless_type::BlockDelta) {
      blockdelta.Resize(n, config.block_delta_block_size,
                        config.block_delta_mode, queue_idx);
    }
    if (config.lossless == lossless_type::ZeroRLE_Rans) {
      zerorle.Resize(n, queue_idx);
      rans.Resize(rle_rans_bound(n), 256, queue_idx);
    }
    if (config.lossless == lossless_type::SymbolRans) {
      symbolrans.Resize(n, config.huff_dict_size, config.estimate_outlier_ratio,
                        queue_idx);
    }
  }

  static size_t EstimateMemoryFootprint(SIZE primary_count, Config config) {
    size_t size = 0;
    if (uses_huffman(config.lossless)) {
      size += Huffman<Q, S, H, DeviceType>::EstimateMemoryFootprint(
          primary_count, config.huff_dict_size, config.huff_block_size,
          config.estimate_outlier_ratio);
    }
    if (config.lossless == lossless_type::Huffman_LZ4) {
      size += LZ4<DeviceType>::EstimateMemoryFootprint(
          primary_count * sizeof(H), config.lz4_block_size);
    }
    if (config.lossless == lossless_type::LZ4) {
      size += LZ4<DeviceType>::EstimateMemoryFootprint(
          primary_count * sizeof(T), config.lz4_block_size);
    }
    if (config.lossless == lossless_type::Huffman_Zstd) {
      size +=
          Zstd<DeviceType>::EstimateMemoryFootprint(primary_count * sizeof(H));
    }
    if (config.lossless == lossless_type::BlockDelta) {
      size += BlockDeltaLossless<T, DeviceType>::EstimateMemoryFootprint(
          primary_count, config.block_delta_block_size);
    }
    if (config.lossless == lossless_type::ZeroRLE_Rans) {
      // RLE0 blob + rANS scratch (~2x the blob) dominate.
      size += rle_rans_bound(primary_count) * 3;
    }
    if (config.lossless == lossless_type::SymbolRans) {
      size += SymbolRans<Q, S, DeviceType>::EstimateMemoryFootprint(
          primary_count, config.huff_dict_size, config.estimate_outlier_ratio);
    }
    return size;
  }

  void Compress(Array<1, T, DeviceType> &original_data,
                Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {

    if (config.lossless == lossless_type::BlockDelta) {
      blockdelta.Compress(original_data, compressed_data, queue_idx);
      return;
    }

    if (config.lossless == lossless_type::LZ4) {
      // LZ4 directly on the raw quantized stream (no Huffman). View the T array
      // as bytes into compressed_data, then compress it in place.
      SIZE nbytes = original_data.shape(0) * sizeof(T);
      compressed_data.resize({nbytes}, queue_idx);
      MemoryManager<DeviceType>::Copy1D(compressed_data.data(),
                                        (Byte *)original_data.data(), nbytes,
                                        queue_idx);
      lz4.Compress(compressed_data, queue_idx);
      return;
    }

    if (config.lossless == lossless_type::ZeroRLE_Rans) {
      // Zero-RLE the quantized stream into a (counts, symbols) byte blob, then
      // entropy-code that blob with rANS. Self-contained: the rANS output is
      // the final compressed stream (Serialize/Deserialize are no-ops here).
      zerorle.Compress(original_data, rle_bytes, 0.0, queue_idx);
      rans.Compress(rle_bytes, compressed_data, queue_idx);
      return;
    }

    if (config.lossless == lossless_type::SymbolRans) {
      // Outlier separation + symbol-alphabet rANS over the dict_size primary.
      symbolrans.Compress(original_data, compressed_data, queue_idx);
      return;
    }

    huffman.Compress(original_data, compressed_data, 0.0, queue_idx);

    if (config.lossless == lossless_type::Huffman_LZ4) {
      huffman.Serialize(compressed_data, queue_idx);
      lz4.Compress(compressed_data, queue_idx);
    }

    if (config.lossless == lossless_type::Huffman_Zstd) {
      huffman.Serialize(compressed_data, queue_idx);
      zstd.Compress(compressed_data, queue_idx);
    }
  }

  void Serialize(Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
    if (config.lossless == lossless_type::Huffman) {
      huffman.Serialize(compressed_data, queue_idx);
    }
    if (config.lossless == lossless_type::BlockDelta) {
      blockdelta.Serialize(compressed_data, queue_idx);
    }
  }

  void Deserialize(Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
    if (config.lossless == lossless_type::Huffman) {
      huffman.Deserialize(compressed_data, queue_idx);
    }
    if (config.lossless == lossless_type::BlockDelta) {
      blockdelta.Deserialize(compressed_data, queue_idx);
    }
  }

  void Decompress(Array<1, Byte, DeviceType> &compressed_data,
                  Array<1, T, DeviceType> &decompressed_data, int queue_idx) {

    if (config.lossless == lossless_type::BlockDelta) {
      // Deserialize (the memory-movement stage) was already run separately by
      // the pipeline; Decompress is computation only.
      blockdelta.Decompress(compressed_data, decompressed_data, queue_idx);
      return;
    }

    if (config.lossless == lossless_type::LZ4) {
      // Inverse of the standalone LZ4 path: decompress to the raw byte stream,
      // then reinterpret it back into the quantized T array.
      lz4.Decompress(compressed_data, queue_idx);
      SIZE nbytes = compressed_data.shape(0);
      decompressed_data.resize({(SIZE)(nbytes / sizeof(T))}, queue_idx);
      MemoryManager<DeviceType>::Copy1D((Byte *)decompressed_data.data(),
                                        compressed_data.data(), nbytes,
                                        queue_idx);
      return;
    }

    if (config.lossless == lossless_type::ZeroRLE_Rans) {
      // Inverse of the RLE0 -> rANS path: rANS-decode back to the RLE0 blob,
      // then expand it to the quantized stream.
      rans.Deserialize(compressed_data, queue_idx);
      rans.Decompress(compressed_data, rle_bytes, queue_idx);
      zerorle.Deserialize(rle_bytes, queue_idx);
      zerorle.Decompress(rle_bytes, decompressed_data, queue_idx);
      return;
    }

    if (config.lossless == lossless_type::SymbolRans) {
      symbolrans.Decompress(compressed_data, decompressed_data, queue_idx);
      return;
    }

    if (config.lossless == lossless_type::Huffman_LZ4) {
      lz4.Decompress(compressed_data, queue_idx);
      huffman.Deserialize(compressed_data, queue_idx);
    }

    if (config.lossless == lossless_type::Huffman_Zstd) {
      zstd.Decompress(compressed_data, queue_idx);
      huffman.Deserialize(compressed_data, queue_idx);
    }

    huffman.Decompress(compressed_data, decompressed_data, queue_idx);
  }

  bool initialized;
  SIZE n;
  Config config;
  Huffman<Q, S, H, DeviceType> huffman;
  LZ4<DeviceType> lz4;
  Zstd<DeviceType> zstd;
  BlockDeltaLossless<T, DeviceType> blockdelta;
  parallel_rle::ZeroRunLengthEncoding<T, uint32_t, uint32_t, DeviceType>
      zerorle;
  rans::Rans<Byte, DeviceType> rans;
  Array<1, Byte, DeviceType> rle_bytes;
  SymbolRans<Q, S, DeviceType> symbolrans;
};

} // namespace mgard_x_160

#endif
