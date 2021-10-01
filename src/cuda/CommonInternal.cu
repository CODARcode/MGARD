/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#include "cuda/CommonInternal.h"
#include <iomanip>
#include <iostream>

namespace mgard_cuda {
bool is_2kplus1_cuda(double num) {
  float frac_part, f_level, int_part;

  f_level = std::log2(num - 1);
  frac_part = modff(f_level, &int_part);

  if (frac_part == 0) {
    return 1;
  } else {
    return 0;
  }
}

// __device__ int get_idx(const int ld, const int i, const int j) {
//   return ld * i + j;
// }

// // ld2 = nrow
// // ld1 = pitch
// __device__ int get_idx(const int ld1, const int ld2, const int z, const int
// y,
//                        const int x) {
//   return ld2 * ld1 * z + ld1 * y + x;
// }



template int check_shape<1>(std::vector<SIZE> shape);
template int check_shape<2>(std::vector<SIZE> shape);
template int check_shape<3>(std::vector<SIZE> shape);
template int check_shape<4>(std::vector<SIZE> shape);
template int check_shape<5>(std::vector<SIZE> shape);

template <typename T> T max_norm_cuda(const T *v, size_t size) {
  double norm = 0;

  for (int i = 0; i < size; ++i) {
    T ntest = std::abs(v[i]);
    if (ntest > norm)
      norm = ntest;
  }
  return norm;
}

template double max_norm_cuda<double>(const double *v, size_t size);
template float max_norm_cuda<float>(const float *v, size_t size);

template <typename T> __device__ T _get_dist(T *coords, int i, int j) {
  return coords[j] - coords[i];
}

template __device__ double _get_dist<double>(double *coords, int i, int j);
template __device__ float _get_dist<float>(float *coords, int i, int j);

__host__ __device__ int get_lindex_cuda(const int n, const int no,
                                        const int i) {
  // no: original number of points
  // n : number of points at next coarser level (L-1) with  2^k+1 nodes
  int lindex;
  //    return floor((no-2)/(n-2)*i);
  if (i != n - 1) {
    lindex = floor(((double)no - 2.0) / ((double)n - 2.0) * i);
  } else if (i == n - 1) {
    lindex = no - 1;
  }

  return lindex;
}

} // namespace mgard_cuda