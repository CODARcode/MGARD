#ifndef _MDR_ADAPTIVE_LEVEL_COMPRESSOR_HPP
#define _MDR_ADAPTIVE_LEVEL_COMPRESSOR_HPP

#include "LevelCompressorInterface.hpp"
#include "LosslessCompressor.hpp"

namespace MDR {
    #define CR_THRESHOLD 1.05
    // compress all layers
    class AdaptiveLevelCompressor : public concepts::LevelCompressorInterface {
    public:
        AdaptiveLevelCompressor(int l = 26) : latter_index(l) {}
        uint8_t compress_level(std::vector<uint8_t*>& streams, std::vector<uint32_t>& stream_sizes) const {
            int stopping_index = stream_sizes.size();
            for(int i=0; i<streams.size(); i++){
                uint8_t * compressed = NULL;
                auto compressed_size = ZSTD::compress(streams[i], stream_sizes[i], &compressed);
                free(streams[i]);
                // std::cout << compressed_size << " " << stream_sizes[i] << " " << stream_sizes[i] * 1.0 / compressed_size << std::endl;
                // skip the first
                float ratio = stream_sizes[i] * 1.0 / compressed_size;
                streams[i] = compressed;
                stream_sizes[i] = compressed_size;
                if(i && (ratio < CR_THRESHOLD)){
                    stopping_index = i;
                    break;
                }
            }
            int latter_start_index = (stopping_index < latter_index) ? latter_index : stopping_index + 1;
            for(int i=latter_start_index; i<streams.size(); i++){
                uint8_t * compressed = NULL;
                auto compressed_size = ZSTD::compress(streams[i], stream_sizes[i], &compressed);
                free(streams[i]);
                streams[i] = compressed;
                stream_sizes[i] = compressed_size;
            }
            return stopping_index;
        }
        void decompress_level(std::vector<const uint8_t*>& streams, const std::vector<uint32_t>& stream_sizes, uint8_t starting_bitplane, uint8_t num_bitplanes, uint8_t stopping_index) {
            for(int i=0; i<num_bitplanes; i++){
                int bitplane_index = starting_bitplane + i;
                if((bitplane_index <= stopping_index) || (bitplane_index >= latter_index)){
                    uint8_t * decompressed = NULL;
                    auto decompressed_size = ZSTD::decompress(streams[i], stream_sizes[bitplane_index], &decompressed);
                    buffer.push_back(decompressed);
                    streams[i] = decompressed;                    
                }
            }
        }
        void decompress_release(){
            for(int i=0; i<buffer.size(); i++){
                if(buffer[i]) free(buffer[i]);
            }
            buffer.clear();
        }
        void print() const {
            std::cout << "Adaptive level lossless compressor" << std::endl;
        }
        ~AdaptiveLevelCompressor(){
            decompress_release();
        }
    private:
        int latter_index;
        std::vector<uint8_t*> buffer;
    };
}
#endif
