/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_160_GPK_FUNCTOR
#define MGARD_X_160_GPK_FUNCTOR

namespace mgard_x_160 {

template <typename T> MGARDX_EXEC T lerp(T v0, T v1, T t) {
#ifdef MGARD_X_160_FMA
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

} // namespace mgard_x_160

#endif