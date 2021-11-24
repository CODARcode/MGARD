/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_LPK_FUNCTOR
#define MGARD_X_LPK_FUNCTOR

namespace mgard_x {

template <typename T>
__device__ inline T mass_trans(T a, T b, T c, T d, T e, T h1, T h2, T h3, T h4,
                               T r1, T r2, T r3, T r4) {
  T tb, tc, td, tb1, tb2, tc1, tc2, td1, td2;
#ifdef MGARD_X_FMA
  if (sizeof(T) == sizeof(double)) {
    tb1 = fma(c, h2/6, a * h1/6);
    tb2 = fma(b, h2/6, b * h1/6);

    tc1 = fma(d, h3/6, b * h2/6);
    tc2 = fma(c, h3/6, c * h2/6);

    td1 = fma(c, h4/6, e * h3/6);
    td2 = fma(d, h4/6, d * h3/6);

    tb = fma(2, tb2, tb1);
    tc = fma(2, tc2, tc1);
    td = fma(2, td2, td1);
    return fma(td, r4, fma(tb, r1, tc));
  } else if (sizeof(T) == sizeof(float)) {
    tb1 = fmaf(c, h2/6, a * h1/6);
    tb2 = fmaf(b, h2/6, b * h1/6);

    tc1 = fmaf(d, h3/6, b * h2/6);
    tc2 = fmaf(c, h3/6, c * h2/6);

    td1 = fmaf(c, h4/6, e * h3/6);
    td2 = fmaf(d, h4/6, d * h3/6);

    tb = fmaf(2, tb2, tb1);
    tc = fmaf(2, tc2, tc1);
    td = fmaf(2, td2, td1);
    return fmaf(td, r4, fmaf(tb, r1, tc));
  }
#else

  if (h1 + h2 != 0) {
    r1 = h1 / (h1 + h2);
  } else {
    r1 = 0.0;
  }
  if (h3 + h4 != 0) {
    r4 = h4 / (h3 + h4);
  } else {
    r4 = 0.0;
  }

  // printf("%f %f %f %f %f (%f %f %f %f)\n", a, b, c, d, e, h1, h2, h3, h4);
  tb = a * (h1/6) + b * ((h1 + h2)/3) + c * (h2/6);
  tc = b * (h2/6) + c * ((h2 + h3)/3) + d * (h3/6);
  td = c * (h3/6) + d * ((h3 + h4)/3) + e * (h4/6);
  tc += tb * r1 + td * r4;
  return tc;
#endif
}

}

#endif