#ifndef CUDA_MEM_CUH
#define CUDA_MEM_CUH

#include <cstdint>
#include <cuda_runtime.h>

namespace mem {

enum MemcpyDirection { h2d, d2h };

template <typename T> T *CreateCUDASpace(size_t l, uint8_t i = 0);

template <typename T>
void CopyBetweenSpaces(T *src, T *dst, MemcpyDirection direct);

template <typename T> T *CreateDeviceSpaceAndMemcpyFromHost(T *var, size_t l);

template <typename T> T *CreateHostSpaceAndMemcpyFromDevice(T *d_var, size_t l);
} // namespace mem

#endif
