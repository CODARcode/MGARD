#ifndef _MDR_DEFAULT_LEVEL_COMPRESSOR_HPP
#define _MDR_DEFAULT_LEVEL_COMPRESSOR_HPP

#include "LevelCompressorInterface.hpp"
#include "LosslessCompressor.hpp"
#include "../RefactorUtils.hpp"
namespace mgard_cuda {
namespace MDR {
    // compress all layers
    class DefaultLevelCompressor : public concepts::LevelCompressorInterface {
    public:
        DefaultLevelCompressor(){}
        uint8_t compress_level(std::vector<uint8_t*>& streams, std::vector<uint32_t>& stream_sizes) const {
            Timer timer;
            for(int i=0; i<streams.size(); i++){
                uint8_t * compressed = NULL;
                timer.start();
                auto compressed_size = ZSTD::compress(streams[i], stream_sizes[i], &compressed);
                free(streams[i]);
                timer.end();
                streams[i] = compressed;
                stream_sizes[i] = compressed_size;
            }
            timer.print("Lossless: ");
            return 0;
        }
        void decompress_level(std::vector<const uint8_t*>& streams, const std::vector<uint32_t>& stream_sizes, uint8_t starting_bitplane, uint8_t num_bitplanes, uint8_t stopping_index) {
            for(int i=0; i<num_bitplanes; i++){
                uint8_t * decompressed = NULL;
                auto decompressed_size = ZSTD::decompress(streams[i], stream_sizes[starting_bitplane + i], &decompressed);
                buffer.push_back(decompressed);
                streams[i] = decompressed;
            }
        }
        void decompress_release(){
            for(int i=0; i<buffer.size(); i++){
                free(buffer[i]);
            }
            buffer.clear();
        }
        void print() const {
            std::cout << "Default level lossless compressor" << std::endl;
        }
        ~DefaultLevelCompressor(){
            decompress_release();
        }
    private:
        std::vector<uint8_t*> buffer;
    };
}
}

namespace mgard_m {
namespace MDR {

// interface for lossless compressor 
template<typename HandleType, mgard_cuda::DIM D, typename T>
class DefaultLevelCompressor : public concepts::LevelCompressorInterface<HandleType, D, T> {
public:

    DefaultLevelCompressor(HandleType& handle): handle(handle) {}
    ~DefaultLevelCompressor() {};

    // compress level, overwrite and free original streams; rewrite streams sizes
    uint8_t compress_level(std::vector<mgard_cuda::SIZE>& bitplane_sizes,
                           mgard_cuda::Array<2, T>& encoded_bitplanes,
                           std::vector<mgard_cuda::Array<1, mgard_cuda::Byte>>& compressed_bitplanes) {

      mgard_cuda::SubArray<2, T> encoded_bitplanes_subarray(encoded_bitplanes);
      for (mgard_cuda::SIZE bitplane_idx = 0; bitplane_idx < encoded_bitplanes_subarray.shape[1]; bitplane_idx++) {
        T * bitplane = encoded_bitplanes_subarray(bitplane_idx, 0);
        T * bitplane_host = new T[bitplane_sizes[bitplane_idx]];
        mgard_cuda::cudaMemcpyAsyncHelper(handle, bitplane_host, bitplane,
                                          bitplane_sizes[bitplane_idx], mgard_cuda::AUTO, 0);
        handle.sync(0);
        mgard_cuda::Byte * compressed_host = NULL;
        mgard_cuda::SIZE compressed_bitplane_size = 
          mgard_cuda::MDR::ZSTD::compress((uint8_t*)bitplane_host, bitplane_sizes[bitplane_idx], &compressed_host);
        mgard_cuda::Array<1, mgard_cuda::Byte> compressed_bitplane({compressed_bitplane_size});
        compressed_bitplane.loadData(compressed_host);
        compressed_bitplanes.push_back(compressed_bitplane);
        bitplane_sizes[bitplane_idx] = compressed_bitplane_size;
      }
      return 0;
    }

    // decompress level, create new buffer and overwrite original streams; will not change stream sizes
    void decompress_level(std::vector<mgard_cuda::SIZE>& bitplane_sizes,
                          std::vector<mgard_cuda::Array<1, mgard_cuda::Byte>>& compressed_bitplanes, 
                          mgard_cuda::Array<2, T>& encoded_bitplanes,
                          uint8_t starting_bitplane, uint8_t num_bitplanes, uint8_t stopping_index) {

      mgard_cuda::SubArray<2, T> encoded_bitplanes_subarray(encoded_bitplanes);

      for (mgard_cuda::SIZE bitplane_idx = 0; bitplane_idx < num_bitplanes; bitplane_idx++) {
        mgard_cuda::SubArray<1, mgard_cuda::Byte> compressed_bitplane_subarray(compressed_bitplanes[bitplane_idx]);
        mgard_cuda::Byte * compressed_host = new mgard_cuda::Byte[bitplane_sizes[bitplane_idx]];
        mgard_cuda::cudaMemcpyAsyncHelper(handle, compressed_host, compressed_bitplane_subarray.data(),
                                          bitplane_sizes[starting_bitplane + bitplane_idx], mgard_cuda::AUTO, 0);
        handle.sync(0);
        mgard_cuda::Byte * bitplane_host = NULL;
        bitplane_sizes[bitplane_idx] = mgard_cuda::MDR::ZSTD::decompress(compressed_host, bitplane_sizes[bitplane_idx], &bitplane_host);
        T * bitplane = encoded_bitplanes_subarray(bitplane_idx, 0);
        mgard_cuda::cudaMemcpyAsyncHelper(handle, bitplane, bitplane_host,
                                          bitplane_sizes[bitplane_idx], mgard_cuda::AUTO, 0);
        handle.sync(0);
      }
    }

    // release the buffer created
    void decompress_release() {}

    void print() const {}

private:
    HandleType& handle;

};

}
}
#endif
