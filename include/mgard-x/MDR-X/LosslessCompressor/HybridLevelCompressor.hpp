#ifndef _MDR_HYBRID_LEVEL_COMPRESSOR_HPP
#define _MDR_HYBRID_LEVEL_COMPRESSOR_HPP

#include "../../Lossless/ParallelHuffman/Huffman.hpp"
#include "../../Lossless/ParallelRLE/RunLengthEncoding.hpp"
#include "../../Lossless/Zstd.hpp"
// #include "../RefactorUtils.hpp"
#include "LevelCompressorInterface.hpp"
#include "LosslessCompressor.hpp"
#include <cstring>

namespace mgard_x {
namespace MDR {

// interface for lossless compressor
template <typename T_bitplane, typename DeviceType>
class HybridLevelCompressor
    : public concepts::LevelCompressorInterface<T_bitplane, DeviceType> {
public:
  using T_compress = u_int8_t;
  // using T_compress = u_int16_t;

  static constexpr int byte_ratio = sizeof(T_bitplane) / sizeof(T_compress);
  static constexpr int _huff_dict_size = 256;
  static constexpr int _huff_block_size = 1024;
  static constexpr int num_merged_bitplanes = 4;

  SIZE size_threshold = 1e6;
  float cr_threshold = 2.0;

  HybridLevelCompressor() : initialized(false) {}
  HybridLevelCompressor(SIZE max_n, Config config) {
    this->initialized = true;
    Adapt(max_n * byte_ratio, config, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
  }
  ~HybridLevelCompressor(){};

  void Adapt(SIZE max_n, SIZE max_bitplanes, Config config, int queue_idx) {
    this->initialized = true;
    this->config = config;
    huffman.Resize(max_n * byte_ratio * num_merged_bitplanes, _huff_dict_size,
                   _huff_block_size, config.estimate_outlier_ratio, queue_idx);
    rle.Resize(max_n * byte_ratio * num_merged_bitplanes, queue_idx);
    zstd.Resize(max_n * sizeof(T_bitplane), config.zstd_compress_level,
                queue_idx);
  }
  static size_t EstimateMemoryFootprint(SIZE max_n, Config config) {
    size_t size = 0;
    size += Huffman<T_bitplane, T_bitplane, HUFFMAN_CODE, DeviceType>::
        EstimateMemoryFootprint(max_n * byte_ratio * num_merged_bitplanes,
                                _huff_dict_size, _huff_block_size,
                                config.estimate_outlier_ratio);
    size += parallel_rle::RunLengthEncoding<
        T_compress, u_int32_t, u_int32_t,
        DeviceType>::EstimateMemoryFootprint(max_n * byte_ratio *
                                             num_merged_bitplanes);
    size +=
        Zstd<DeviceType>::EstimateMemoryFootprint(max_n * sizeof(T_bitplane));
    return size;
  }

  void
  compress_level(SubArray<2, T_bitplane, DeviceType> &encoded_bitplanes,
                 std::vector<Array<1, Byte, DeviceType>> &compressed_bitplanes,
                 int level_idx, int queue_idx) {

    std::vector<float> cr, time;
    bool huffman_success, rle_success, zstd_success;
    for (SIZE bitplane_idx = 0; bitplane_idx < encoded_bitplanes.shape(0);
         bitplane_idx++) {
      if (bitplane_idx % num_merged_bitplanes == 0) {
        SIZE merged_bitplane_size =
            encoded_bitplanes.shape(1) * byte_ratio * num_merged_bitplanes;
        Timer timer;
        timer.start();
        T_compress *bitplane = (T_compress *)encoded_bitplanes(bitplane_idx, 0);

        Array<1, T_compress, DeviceType> encoded_bitplane(
            {merged_bitplane_size}, bitplane);
        int old_log_level = log::level;
        log::level = 0;
        huffman_success = false;
        rle_success = false;
        zstd_success = false;
        // cr_threshold = 2.0;
        if (merged_bitplane_size > size_threshold &&
            config.lossless == lossless_type::Huffman_Zstd) {
          zstd_success =
              compress_zstd((Byte *)bitplane, merged_bitplane_size,
                            compressed_bitplanes[bitplane_idx], queue_idx);
        } else if (merged_bitplane_size > size_threshold) {
          rle_success =
              rle.Compress(encoded_bitplane, compressed_bitplanes[bitplane_idx],
                           cr_threshold, queue_idx);
          if (rle_success) {
            rle.Serialize(compressed_bitplanes[bitplane_idx], queue_idx);
          } else {
            ATOMIC_IDX zero = 0;
            MemoryManager<DeviceType>::Copy1D(
                huffman.workspace.outlier_count_subarray.data(), &zero, 1,
                queue_idx);
            MemoryManager<DeviceType>::Copy1D(
                &huffman.outlier_count,
                huffman.workspace.outlier_count_subarray.data(), 1, queue_idx);
            huffman_success = huffman.CompressPrimary(
                encoded_bitplane, compressed_bitplanes[bitplane_idx],
                cr_threshold, queue_idx);
            if (huffman_success) {
              huffman.Serialize(compressed_bitplanes[bitplane_idx], queue_idx);
            }
          }
        }

        if (!huffman_success && !rle_success && !zstd_success) {
          // direct copy
          compressed_bitplanes[bitplane_idx].resize({merged_bitplane_size});
          MemoryManager<DeviceType>::Copy1D(
              compressed_bitplanes[bitplane_idx].data(), (Byte *)bitplane,
              merged_bitplane_size, queue_idx);
        }

        log::level = old_log_level;
        cr.push_back((float)merged_bitplane_size /
                     compressed_bitplanes[bitplane_idx].shape(0));

        timer.end();
        time.push_back(timer.get());
        timer.clear();
        // timer.print("Compressing bitplane", merged_bitplane_size);
        // timer.clear();
      } else {
        compressed_bitplanes[bitplane_idx].resize({0}, queue_idx);
      }
    }
    // std::string cr_string = "";
    // for (auto x : cr) {
    //   cr_string += std::to_string(x) + ", ";
    // }
    // log::info("CR: " + cr_string);

    // std::string time_string = "";
    // for (auto x : time) {
    //   time_string += std::to_string(x) + " ";
    // }
    // log::info("Time: " + time_string);
  }

  // decompress level, create new buffer and overwrite original streams; will
  // not change stream sizes
  void decompress_level(
      std::vector<Array<1, Byte, DeviceType>> &compressed_bitplanes,
      SubArray<2, T_bitplane, DeviceType> &encoded_bitplanes,
      uint8_t starting_bitplane, uint8_t num_bitplanes, int level_idx,
      int queue_idx) {

    std::vector<float> time;
    for (SIZE bitplane_idx = starting_bitplane;
         bitplane_idx < starting_bitplane + num_bitplanes; bitplane_idx++) {
      if (bitplane_idx % num_merged_bitplanes == 0) {
        Timer timer;
        timer.start();
        T_compress *bitplane = (T_compress *)encoded_bitplanes(bitplane_idx, 0);
        SIZE merged_bitplane_size =
            encoded_bitplanes.shape(1) * byte_ratio * num_merged_bitplanes;

        Array<1, T_compress, DeviceType> encoded_bitplane(
            {merged_bitplane_size}, bitplane);
        int old_log_level = log::level;
        log::level = 0;

        // Huffman
        if (huffman.Verify(compressed_bitplanes[bitplane_idx], queue_idx)) {
          huffman.Deserialize(compressed_bitplanes[bitplane_idx], queue_idx);
          huffman.DecompressPrimary(compressed_bitplanes[bitplane_idx],
                                    encoded_bitplane, queue_idx);
          // RLE
        } else if (rle.Verify(compressed_bitplanes[bitplane_idx], queue_idx)) {
          rle.Deserialize(compressed_bitplanes[bitplane_idx], queue_idx);
          rle.Decompress(compressed_bitplanes[bitplane_idx], encoded_bitplane,
                         queue_idx);
        } else if (is_zstd(compressed_bitplanes[bitplane_idx],
                           merged_bitplane_size, queue_idx)) {
          decompress_zstd(compressed_bitplanes[bitplane_idx], (Byte *)bitplane,
                          merged_bitplane_size, queue_idx);
        } else {
          // Direct copy
          MemoryManager<DeviceType>::Copy1D(
              (uint8_t *)bitplane, compressed_bitplanes[bitplane_idx].data(),
              merged_bitplane_size, queue_idx);
        }
        log::level = old_log_level;
        timer.end();
        time.push_back(timer.get());
        timer.clear();
      }
    }
    // std::string time_string = "";
    // for (auto x : time) {
    //   time_string += std::to_string(x) + " ";
    // }
    // log::info("Time: " + time_string);
  }

  // ZSTD stage (Config::lossless == Huffman_Zstd): replaces RLE/byte Huffman
  // for groups above size_threshold and is kept whenever it is smaller than
  // the raw group. Stored as [signature][Zstd stream].
  static constexpr Byte zstd_signature[7] = {'M', 'G', 'X', 'Z', 'S', 'T', 'D'};

  bool compress_zstd(Byte *group, SIZE n, Array<1, Byte, DeviceType> &out,
                     int queue_idx) {
    Array<1, Byte, DeviceType> buffer({n});
    MemoryManager<DeviceType>::Copy1D(buffer.data(), group, n, queue_idx);
    zstd.Compress(buffer, queue_idx);
    SIZE size = buffer.shape(0);
    if (size + sizeof(zstd_signature) >= n) {
      return false;
    }
    out.resize({(SIZE)(size + sizeof(zstd_signature))}, queue_idx);
    MemoryManager<DeviceType>::Copy1D(out.data(), (Byte *)zstd_signature,
                                      sizeof(zstd_signature), queue_idx);
    MemoryManager<DeviceType>::Copy1D(out.data() + sizeof(zstd_signature),
                                      buffer.data(), size, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    return true;
  }

  // A raw group is exactly n bytes; a ZSTD group is smaller and signed.
  bool is_zstd(Array<1, Byte, DeviceType> &data, SIZE n, int queue_idx) {
    if (data.shape(0) >= n || data.shape(0) <= sizeof(zstd_signature)) {
      return false;
    }
    Byte signature[sizeof(zstd_signature)];
    MemoryManager<DeviceType>::Copy1D(signature, data.data(),
                                      sizeof(zstd_signature), queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    return std::memcmp(signature, zstd_signature, sizeof(zstd_signature)) == 0;
  }

  void decompress_zstd(Array<1, Byte, DeviceType> &data, Byte *group, SIZE n,
                       int queue_idx) {
    SIZE size = data.shape(0) - sizeof(zstd_signature);
    Array<1, Byte, DeviceType> buffer({size});
    MemoryManager<DeviceType>::Copy1D(
        buffer.data(), data.data() + sizeof(zstd_signature), size, queue_idx);
    zstd.Decompress(buffer, queue_idx);
    MemoryManager<DeviceType>::Copy1D(group, buffer.data(), n, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
  }

  // release the buffer created
  void decompress_release() {}

  void print() const {}
  bool initialized;
  Huffman<T_compress, T_compress, uint64_t, DeviceType> huffman;
  parallel_rle::RunLengthEncoding<T_compress, u_int32_t, u_int32_t, DeviceType>
      rle;
  Zstd<DeviceType> zstd;
  Config config;
};

} // namespace MDR
} // namespace mgard_x
#endif
