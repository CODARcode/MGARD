#ifndef _MDR_NULL_LEVEL_COMPRESSOR_HPP
#define _MDR_NULL_LEVEL_COMPRESSOR_HPP

#include "LevelCompressorInterface.hpp"

namespace MDR {
    // Null lossless compressor
    class NullLevelCompressor : public concepts::LevelCompressorInterface {
    public:
        NullLevelCompressor(){}
        uint8_t compress_level(std::vector<uint8_t*>& streams, std::vector<uint32_t>& stream_sizes) const { return 0;}
        void decompress_level(std::vector<const uint8_t*>& streams, const std::vector<uint32_t>& stream_sizes, uint8_t starting_bitplane, uint8_t num_bitplanes, uint8_t stopping_index){}
        void decompress_release(){}
        void print() const {
            std::cout << "Null level compressor" << std::endl;
        }
    };
}
#endif
