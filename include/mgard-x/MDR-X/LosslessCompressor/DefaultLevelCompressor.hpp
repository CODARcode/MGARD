#ifndef _MDR_DEFAULT_LEVEL_COMPRESSOR_HPP
#define _MDR_DEFAULT_LEVEL_COMPRESSOR_HPP

#include "../../Lossless/ParallelHuffman/Huffman.hpp"
#include "../RefactorUtils.hpp"
#include "LevelCompressorInterface.hpp"
#include "LosslessCompressor.hpp"
namespace MDR {
// compress all layers
class DefaultLevelCompressor : public concepts::LevelCompressorInterface {
public:
  DefaultLevelCompressor() {}
  uint8_t compress_level(std::vector<uint8_t *> &streams,
                         std::vector<uint32_t> &stream_sizes) const {
    for (int i = 0; i < streams.size(); i++) {
      uint8_t *compressed = NULL;
      auto compressed_size =
          ZSTD::compress(streams[i], stream_sizes[i], &compressed);
      free(streams[i]);
      streams[i] = compressed;
      stream_sizes[i] = compressed_size;
    }
    return 0;
  }
  void decompress_level(std::vector<const uint8_t *> &streams,
                        const std::vector<uint32_t> &stream_sizes,
                        uint8_t starting_bitplane, uint8_t num_bitplanes,
                        uint8_t stopping_index) {
    for (int i = 0; i < num_bitplanes; i++) {
      uint8_t *decompressed = NULL;
      auto decompressed_size = ZSTD::decompress(
          streams[i], stream_sizes[starting_bitplane + i], &decompressed);
      buffer.push_back(decompressed);
      streams[i] = decompressed;
    }
  }
  void decompress_release() {
    for (int i = 0; i < buffer.size(); i++) {
      free(buffer[i]);
    }
    buffer.clear();
  }
  void print() const {
    std::cout << "Default level lossless compressor" << std::endl;
  }
  ~DefaultLevelCompressor() { decompress_release(); }

private:
  std::vector<uint8_t *> buffer;
};
} // namespace MDR

namespace mgard_x {
namespace MDR {

// interface for lossless compressor
template <typename T, typename DeviceType>
class DefaultLevelCompressor
    : public concepts::LevelCompressorInterface<T, DeviceType> {
public:
  DefaultLevelCompressor(SIZE max_n, Config config)
      : huffman(max_n, config.huff_dict_size, config.huff_block_size,
                config.estimate_outlier_ratio) {}
  ~DefaultLevelCompressor(){};

  static size_t EstimateMemoryFootprint(SIZE max_n, Config config) {
    size_t size = 0;
    size += Huffman<T, T, HUFFMAN_CODE, DeviceType>::EstimateMemoryFootprint(
        max_n, config.huff_dict_size, config.huff_block_size,
        config.estimate_outlier_ratio);
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
      T *bitplane = encoded_bitplanes_subarray(bitplane_idx, 0);
      T *bitplane_host = new T[bitplane_sizes[bitplane_idx]];

      MemoryManager<DeviceType>::Copy1D(
          bitplane_host, bitplane, bitplane_sizes[bitplane_idx] / sizeof(T), 0);
      DeviceRuntime<DeviceType>::SyncQueue(0);

      Byte *compressed_host = NULL;
      SIZE compressed_bitplane_size =
          ::MDR::ZSTD::compress((uint8_t *)bitplane_host,
                                bitplane_sizes[bitplane_idx], &compressed_host);
      Array<1, Byte, DeviceType> compressed_bitplane(
          {compressed_bitplane_size});
      compressed_bitplane.load(compressed_host);
      compressed_bitplanes[bitplane_idx] = compressed_bitplane;
      bitplane_sizes[bitplane_idx] = compressed_bitplane_size;

      // Array<1, T, DeviceType>
      // encoded_bitplane({encoded_bitplanes_subarray.shape(1)}, bitplane);
      // huffman.Compress(encoded_bitplane, compressed_bitplanes[bitplane_idx],
      // queue_idx); bitplane_sizes[bitplane_idx] =
      // compressed_bitplanes[bitplane_idx].shape(0);
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

    for (SIZE bitplane_idx = 0; bitplane_idx < num_bitplanes; bitplane_idx++) {
      SIZE compressed_size = bitplane_sizes[starting_bitplane + bitplane_idx];
      Byte *compressed_host = new Byte[compressed_size];
      MemoryManager<DeviceType>::Copy1D(
          compressed_host,
          compressed_bitplanes[starting_bitplane + bitplane_idx].data(),
          compressed_size, 0);
      DeviceRuntime<DeviceType>::SyncQueue(0);

      Byte *bitplane_host = NULL;
      SIZE decompressed_size = ::MDR::ZSTD::decompress(
          compressed_host, compressed_size, &bitplane_host);

      T *bitplane = encoded_bitplanes_subarray(bitplane_idx, 0);
      MemoryManager<DeviceType>::Copy1D(bitplane, (T *)bitplane_host,
                                        decompressed_size / sizeof(T), 0);
      DeviceRuntime<DeviceType>::SyncQueue(0);

      // Array<1, T, DeviceType>
      // encoded_bitplane({encoded_bitplanes_subarray.shape(1)}, bitplane);
      // huffman.Decompress(compressed_bitplanes[bitplane_idx],
      // encoded_bitplane, queue_idx);
    }
  }

  // release the buffer created
  void decompress_release() {}

  void print() const {}

  Huffman<T, T, HUFFMAN_CODE, DeviceType> huffman;
};

} // namespace MDR
} // namespace mgard_x
#endif
