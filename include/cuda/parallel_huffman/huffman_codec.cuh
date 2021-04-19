#ifndef DEFLATE_CUH
#define DEFLATE_CUH

#include <stddef.h>

template <typename Q, typename H>
__global__ void EncodeFixedLen(Q*, H*, size_t, H*);

template <typename Q>
__global__ void Deflate(Q*, size_t, size_t*, int);

template <typename H, typename T>
__device__ void InflateChunkwise(H*, T*, size_t, uint8_t*);

template <typename Q, typename H>
__global__ void Decode(H*, size_t*, Q*, size_t, int, int, uint8_t*, size_t);

#endif
