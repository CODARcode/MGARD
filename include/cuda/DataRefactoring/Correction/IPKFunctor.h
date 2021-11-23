/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#ifndef MGRAD_CUDA_IPK_FUNCTOR
#define MGRAD_CUDA_IPK_FUNCTOR

namespace mgard_cuda {

template <typename T>
__device__ inline T tridiag_forward2(T prev, T am, T bm, T curr) {

#ifdef MGARD_CUDA_FMA
  if (sizeof(T) == sizeof(double)) {
    // printf("forward: %f < %f %f %f %f\n", fma(prev, am * bm, curr),
    //         curr, prev, am , bm);
    return fma(prev, am * bm, curr);
  } else if (sizeof(T) == sizeof(float)) {
    // printf("forward: %f < %f %f %f %f\n", fmaf(prev, am * bm, curr),
    //         curr, prev, am , bm);
    return fmaf(prev, am * bm, curr);
  }
#else
  // printf("forward: %f < %f %f %f %f\n", curr - prev * (am * bm),
  //           curr, prev, am , bm);
  return curr - prev * (am / bm);
#endif
}

template <typename T>
__device__ inline T tridiag_backward2(T prev, T am, T bm, T curr) {

#ifdef MGARD_CUDA_FMA
  if (sizeof(T) == sizeof(double)) {
    // printf("backward: %f < %f %f %f %f\n", fma(am, prev, curr) * bm,
    //         curr, prev, am , bm);
    return fma(am, prev, curr) * bm;
  } else if (sizeof(T) == sizeof(float)) {
    // printf("backward: %f < %f %f %f %f\n", fmaf(am, prev, curr) * bm,
    //         curr, prev, am , bm);
    return fmaf(am, prev, curr) * bm;
  }
#else
  // printf("backward: %f < %f %f %f %f\n", (curr - am * prev) * bm,
  //           curr, prev, am , bm);
  return (curr - am * prev) / bm;
#endif
}

}

#endif