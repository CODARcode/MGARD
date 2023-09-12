#ifndef _MDR_DEFAULT_LEVEL_COMPRESSOR_HPP
#define _MDR_DEFAULT_LEVEL_COMPRESSOR_HPP

#include "../../Lossless/ParallelHuffman/Huffman.hpp"
#include "../../Lossless/Zstd.hpp"
// #include "../RefactorUtils.hpp"
#include "LevelCompressorInterface.hpp"
#include "LosslessCompressor.hpp"

namespace mgard_x {
namespace MDR {

// interface for lossless compressor
template <typename T, typename DeviceType>
class DefaultLevelCompressor
    : public concepts::LevelCompressorInterface<T, DeviceType> {
public:
  DefaultLevelCompressor(SIZE max_n, Config config)
      : huffman(max_n, config.huff_dict_size, config.huff_block_size,
                config.estimate_outlier_ratio),
        config(config) {
    zstd.Resize(max_n * sizeof(T), config.zstd_compress_level, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
  }
  ~DefaultLevelCompressor(){};

  static size_t EstimateMemoryFootprint(SIZE max_n, Config config) {
    size_t size = 0;
    size += Huffman<T, T, HUFFMAN_CODE, DeviceType>::EstimateMemoryFootprint(
        max_n, config.huff_dict_size, config.huff_block_size,
        config.estimate_outlier_ratio);
    size += Zstd<DeviceType>::EstimateMemoryFootprint(max_n * sizeof(T));
    return size;
  }
  // compress level, overwrite and free original streams; rewrite streams sizes
  void
  compress_level(std::vector<SIZE> &bitplane_sizes,
                 Array<2, T, DeviceType> &encoded_bitplanes,
                 std::vector<Array<1, Byte, DeviceType>> &compressed_bitplanes,
                 int queue_idx) {

    SubArray<2, T, DeviceType> encoded_bitplanes_subarray(encoded_bitplanes);
    for (SIZE bitplane_idx = 0;
         bitplane_idx < encoded_bitplanes_subarray.shape(0); bitplane_idx++) {
      T *bitplane = encoded_bitplanes_subarray(bitplane_idx, queue_idx);
      // MDR::Zstd
      // T *bitplane_host = new T[bitplane_sizes[bitplane_idx]];

      // MemoryManager<DeviceType>::Copy1D(
      //     bitplane_host, bitplane, bitplane_sizes[bitplane_idx] / sizeof(T),
      //     0);
      // DeviceRuntime<DeviceType>::SyncQueue(0);

      // Byte *compressed_host = NULL;
      // SIZE compressed_bitplane_size =
      //     ::MDR::ZSTD::compress((uint8_t *)bitplane_host,
      //                           bitplane_sizes[bitplane_idx],
      //                           &compressed_host);
      // Array<1, Byte, DeviceType> compressed_bitplane(
      //     {compressed_bitplane_size});
      // compressed_bitplane.load(compressed_host);
      // compressed_bitplanes[bitplane_idx] = compressed_bitplane;
      // bitplane_sizes[bitplane_idx] = compressed_bitplane_size;

      // Huffman
      // Array<1, T, DeviceType>
      // encoded_bitplane({encoded_bitplanes_subarray.shape(1)}, bitplane);
      // huffman.Compress(encoded_bitplane, compressed_bitplanes[bitplane_idx],
      // queue_idx); bitplane_sizes[bitplane_idx] =
      // compressed_bitplanes[bitplane_idx].shape(0);

      Array<1, Byte, DeviceType> compressed_bitplane(
          {bitplane_sizes[bitplane_idx]});
      MemoryManager<DeviceType>::Copy1D(
          compressed_bitplane.data(), (uint8_t *)bitplane,
          bitplane_sizes[bitplane_idx], queue_idx);
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      int old_log_level = log::level;
      log::level = log::ERR;
      zstd.Compress(compressed_bitplane, queue_idx);
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      log::level = old_log_level;
      compressed_bitplanes[bitplane_idx] = compressed_bitplane;
      bitplane_sizes[bitplane_idx] = compressed_bitplane.shape(0);
    }
  }

  // decompress level, create new buffer and overwrite original streams; will
  // not change stream sizes
  void decompress_level(
      std::vector<SIZE> &bitplane_sizes,
      std::vector<Array<1, Byte, DeviceType>> &compressed_bitplanes,
      Array<2, T, DeviceType> &encoded_bitplanes, uint8_t starting_bitplane,
      uint8_t num_bitplanes, int queue_idx) {

    SubArray<2, T, DeviceType> encoded_bitplanes_subarray(encoded_bitplanes);

    for (SIZE bitplane_idx = starting_bitplane; bitplane_idx < num_bitplanes;
         bitplane_idx++) {
      T *bitplane = encoded_bitplanes_subarray(bitplane_idx, 0);
      // MDR::Zstd
      // SIZE compressed_size = bitplane_sizes[starting_bitplane +
      // bitplane_idx]; Byte *compressed_host = new Byte[compressed_size];
      // MemoryManager<DeviceType>::Copy1D(
      //     compressed_host,
      //     compressed_bitplanes[starting_bitplane + bitplane_idx].data(),
      //     compressed_size, 0);
      // DeviceRuntime<DeviceType>::SyncQueue(0);

      // Byte *bitplane_host = NULL;
      // SIZE decompressed_size = ::MDR::ZSTD::decompress(
      //     compressed_host, compressed_size, &bitplane_host);

      // MemoryManager<DeviceType>::Copy1D(bitplane, (T *)bitplane_host,
      //                                   decompressed_size / sizeof(T), 0);
      // DeviceRuntime<DeviceType>::SyncQueue(0);

      // Huffman
      // Array<1, T, DeviceType>
      // encoded_bitplane({encoded_bitplanes_subarray.shape(1)}, bitplane);
      // huffman.Decompress(compressed_bitplanes[bitplane_idx],
      // encoded_bitplane, queue_idx);
      // std::cout << "decompress level: " << bitplane_idx << "\n";
      int old_log_level = log::level;
      log::level = log::ERR;
      zstd.Decompress(compressed_bitplanes[bitplane_idx], queue_idx);
      log::level = old_log_level;
      MemoryManager<DeviceType>::Copy1D(
          (uint8_t *)bitplane, compressed_bitplanes[bitplane_idx].data(),
          compressed_bitplanes[bitplane_idx].shape(0), queue_idx);
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    }
  }

  // release the buffer created
  void decompress_release() {}

  void print() const {}

  Huffman<T, T, HUFFMAN_CODE, DeviceType> huffman;
  Zstd<DeviceType> zstd;
  Config config;
};

} // namespace MDR
} // namespace mgard_x
#endif
