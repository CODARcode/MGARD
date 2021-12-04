/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGRAD_CUDA_GPK_FUNCTOR
#define MGRAD_CUDA_GPK_FUNCTOR

namespace mgard_cuda {

template <typename T> __device__ inline T lerp(T v0, T v1, T t) {
#ifdef MGARD_CUDA_FMA
  if (sizeof(T) == sizeof(double)) {
    return fma(t, v1, fma(-t, v0, v0));
  } else if (sizeof(T) == sizeof(float)) {
    return fmaf(t, v1, fmaf(-t, v0, v0));
  }
#else
  T r = v0 + v0 * t * -1;
  r = r + t * v1;
  return r;
#endif
}

} // namespace mgard_cuda

#endif