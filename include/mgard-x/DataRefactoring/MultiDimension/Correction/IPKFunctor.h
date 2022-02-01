/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_IPK_FUNCTOR
#define MGARD_X_IPK_FUNCTOR

namespace mgard_x {

template <typename T>
MGARDX_EXEC T tridiag_forward2(T prev, T am, T bm, T curr) {

#ifdef MGARD_X_FMA
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
MGARDX_EXEC T tridiag_backward2(T prev, T am, T bm, T curr) {

#ifdef MGARD_X_FMA
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

} // namespace mgard_x

#endif