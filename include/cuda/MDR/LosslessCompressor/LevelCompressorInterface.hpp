#ifndef _MDR_LEVEL_COMPRESSOR_INTERFACE_HPP
#define _MDR_LEVEL_COMPRESSOR_INTERFACE_HPP
namespace mgard_cuda {
namespace MDR {
    namespace concepts {

        // interface for lossless compressor 
        class LevelCompressorInterface {
        public:

            virtual ~LevelCompressorInterface() = default;

            // compress level, overwrite and free original streams; rewrite streams sizes
            virtual uint8_t compress_level(std::vector<uint8_t*>& streams, std::vector<uint32_t>& stream_sizes) const = 0;

            // decompress level, create new buffer and overwrite original streams; will not change stream sizes
            virtual void decompress_level(std::vector<const uint8_t*>& streams, const std::vector<uint32_t>& stream_sizes, uint8_t starting_bitplane, uint8_t num_bitplanes, uint8_t stopping_index) = 0;

            // release the buffer created
            virtual void decompress_release() = 0;

            virtual void print() const = 0;
        };
    }
}
}

namespace mgard_m {
namespace MDR {
    namespace concepts {

        // interface for lossless compressor 
        template<typename HandleType, mgard_cuda::DIM D, typename T>
        class LevelCompressorInterface {
        public:

            virtual ~LevelCompressorInterface() = default;

            // compress level, overwrite and free original streams; rewrite streams sizes
            virtual uint8_t compress_level(std::vector<mgard_cuda::SIZE>& bitplane_sizes,
                                           mgard_cuda::Array<2, T>& encoded_bitplanes,
                                           std::vector<mgard_cuda::Array<1, mgard_cuda::Byte>>& compressed_bitplanes) = 0;

            // decompress level, create new buffer and overwrite original streams; will not change stream sizes
            virtual void decompress_level(std::vector<mgard_cuda::SIZE>& bitplane_sizes,
                                          std::vector<mgard_cuda::Array<1, mgard_cuda::Byte>>& compressed_bitplanes, 
                                          mgard_cuda::Array<2, T>& encoded_bitplanes,
                                          uint8_t starting_bitplane, uint8_t num_bitplanes, uint8_t stopping_index) = 0;

            // release the buffer created
            virtual void decompress_release() = 0;

            virtual void print() const = 0;
        };
    }
}
}


#endif
