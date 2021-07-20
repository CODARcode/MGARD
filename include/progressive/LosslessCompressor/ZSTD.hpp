#ifndef _MDR_ZSTD_HPP
#define _MDR_ZSTD_HPP

#include "zstd.h"

namespace MDR {
    namespace ZSTD{
        #define ZSTD_LEVEL 3 //default setting of level is 3
        // ZSTD lossless compressor
        uint32_t compress(uint8_t* data, uint32_t dataLength, uint8_t** compressBytes) {
            uint32_t outSize = 0; 
            size_t estimatedCompressedSize = 0;
            if(dataLength < 100) 
                estimatedCompressedSize = 200;
            else
                estimatedCompressedSize = dataLength*1.2;
            *compressBytes = (uint8_t*)malloc(estimatedCompressedSize);
            *reinterpret_cast<size_t*>(*compressBytes) = dataLength;
            outSize = ZSTD_compress(*compressBytes + sizeof(size_t), estimatedCompressedSize, data, dataLength, ZSTD_LEVEL); 
            return outSize + sizeof(size_t);
        }
        uint32_t decompress(const uint8_t* compressBytes, uint32_t cmpSize, uint8_t** oriData) {
            uint32_t outSize = 0;
            outSize = *reinterpret_cast<const size_t*>(compressBytes);
            *oriData = (uint8_t*)malloc(outSize);
            ZSTD_decompress(*oriData, outSize, compressBytes + sizeof(size_t), cmpSize - sizeof(size_t));
            return outSize;
        }
    }
}
#endif
