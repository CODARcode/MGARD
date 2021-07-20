#ifndef _MDR_DEFAULT_LEVEL_COMPRESSOR_HPP
#define _MDR_DEFAULT_LEVEL_COMPRESSOR_HPP

#include "LevelCompressorInterface.hpp"
#include "LosslessCompressor.hpp"
#include "RefactorUtils.hpp"

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
#endif
