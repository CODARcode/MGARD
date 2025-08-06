#ifndef _MDR_DEFAULT_LEVEL_COMPRESSOR_HPP
#define _MDR_DEFAULT_LEVEL_COMPRESSOR_HPP

#include "../RefactorUtils.hpp"
#include "LevelCompressorInterface.hpp"
#include "LosslessCompressor.hpp"
namespace mgard_x {
namespace MDR {
// compress all layers
class DefaultLevelCompressor : public concepts::LevelCompressorInterface {
public:
  DefaultLevelCompressor() {}
  uint8_t compress_level(std::vector<uint8_t *> &streams,
                         std::vector<SIZE> &stream_sizes) const {
    Timer timer;
    for (int i = 0; i < streams.size(); i++) {
      uint8_t *compressed = NULL;
      timer.start();
      auto compressed_size =
          ZSTD::compress(streams[i], stream_sizes[i], &compressed);
      free(streams[i]);
      timer.end();
      streams[i] = compressed;
      stream_sizes[i] = compressed_size;
    }
    timer.print("Lossless: ");
    return 0;
  }
  void decompress_level(std::vector<const uint8_t *> &streams,
                        const std::vector<SIZE> &stream_sizes,
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
} // namespace mgard_x

namespace mgard_m {
namespace MDR {

// interface for lossless compressor
template <typename HandleType, mgard_x::DIM D, typename T>
class DefaultLevelCompressor
    : public concepts::LevelCompressorInterface<HandleType, D, T> {
public:
  DefaultLevelCompressor(HandleType &handle) : handle(handle) {}
  ~DefaultLevelCompressor(){};

  // compress level, overwrite and free original streams; rewrite streams sizes
  uint8_t
  compress_level(std::vector<mgard_x::SIZE> &bitplane_sizes,
                 mgard_x::Array<2, T, mgard_x::CUDA> &encoded_bitplanes,
                 std::vector<mgard_x::Array<1, mgard_x::Byte, mgard_x::CUDA>>
                     &compressed_bitplanes) {

    mgard_x::SubArray<2, T, mgard_x::CUDA> encoded_bitplanes_subarray(
        encoded_bitplanes);
    for (mgard_x::SIZE bitplane_idx = 0;
         bitplane_idx < encoded_bitplanes_subarray.getShape(1);
         bitplane_idx++) {
      T *bitplane = encoded_bitplanes_subarray(bitplane_idx, 0);
      T *bitplane_host = new T[bitplane_sizes[bitplane_idx]];
      // mgard_x::cudaMemcpyAsyncHelper(handle, bitplane_host, bitplane,
      //                                   bitplane_sizes[bitplane_idx],
      //                                   mgard_x::AUTO, 0);
      // handle.sync(0);
      MemoryManager<CUDA>::Copy1D(bitplane_host, bitplane,
                                  bitplane_sizes[bitplane_idx] / sizeof(T), 0);
      DeviceRuntime<CUDA>::SyncQueue(0);

      mgard_x::Byte *compressed_host = NULL;
      mgard_x::SIZE compressed_bitplane_size = mgard_x::MDR::ZSTD::compress(
          (uint8_t *)bitplane_host, bitplane_sizes[bitplane_idx],
          &compressed_host);
      mgard_x::Array<1, mgard_x::Byte, mgard_x::CUDA> compressed_bitplane(
          {compressed_bitplane_size});
      compressed_bitplane.loadData(compressed_host);
      compressed_bitplanes.push_back(compressed_bitplane);
      bitplane_sizes[bitplane_idx] = compressed_bitplane_size;
    }
    return 0;
  }

  // decompress level, create new buffer and overwrite original streams; will
  // not change stream sizes
  void
  decompress_level(std::vector<mgard_x::SIZE> &bitplane_sizes,
                   std::vector<mgard_x::Array<1, mgard_x::Byte, mgard_x::CUDA>>
                       &compressed_bitplanes,
                   mgard_x::Array<2, T, mgard_x::CUDA> &encoded_bitplanes,
                   uint8_t starting_bitplane, uint8_t num_bitplanes,
                   uint8_t stopping_index) {

    mgard_x::SubArray<2, T, mgard_x::CUDA> encoded_bitplanes_subarray(
        encoded_bitplanes);

    for (mgard_x::SIZE bitplane_idx = 0; bitplane_idx < num_bitplanes;
         bitplane_idx++) {
      mgard_x::SubArray<1, mgard_x::Byte, mgard_x::CUDA>
          compressed_bitplane_subarray(compressed_bitplanes[bitplane_idx]);
      mgard_x::Byte *compressed_host =
          new mgard_x::Byte[bitplane_sizes[bitplane_idx]];
      // mgard_x::cudaMemcpyAsyncHelper(handle, compressed_host,
      // compressed_bitplane_subarray.data(),
      //                                   bitplane_sizes[starting_bitplane +
      //                                   bitplane_idx], mgard_x::AUTO, 0);
      // handle.sync(0);
      MemoryManager<CUDA>::Copy1D(
          compressed_host, compressed_bitplane_subarray.data(),
          bitplane_sizes[starting_bitplane + bitplane_idx], 0);
      DeviceRuntime<CUDA>::SyncQueue(0);

      mgard_x::Byte *bitplane_host = NULL;
      bitplane_sizes[bitplane_idx] = mgard_x::MDR::ZSTD::decompress(
          compressed_host, bitplane_sizes[bitplane_idx], &bitplane_host);
      T *bitplane = encoded_bitplanes_subarray(bitplane_idx, 0);
      // mgard_x::cudaMemcpyAsyncHelper(handle, bitplane, bitplane_host,
      //                                   bitplane_sizes[bitplane_idx],
      //                                   mgard_x::AUTO, 0);
      // handle.sync(0);
      MemoryManager<CUDA>::Copy1D(bitplane, bitplane_host,
                                  bitplane_sizes[bitplane_idx] / sizeof(T), 0);
      DeviceRuntime<CUDA>::SyncQueue(0);
    }
  }

  // release the buffer created
  void decompress_release() {}

  void print() const {}

private:
  HandleType &handle;
};

} // namespace MDR
} // namespace mgard_m
#endif
