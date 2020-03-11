#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"
#include "mgard_cuda_helper_internal.h"
#include <fstream>

namespace mgard_2d {
namespace mgard_common {

__host__ __device__
int get_index_cuda(const int ncol, const int i, const int j) {
  return ncol * i + j;
}

template <typename T>
__host__ __device__ T 
interp_2d_cuda(T q11, T q12, T q21, T q22,
                        T x1, T x2, T y1, T y2, T x,
                        T y) {
  T x2x1, y2y1, x2x, y2y, yy1, xx1;
  x2x1 = x2 - x1;
  y2y1 = y2 - y1;
  x2x = x2 - x;
  y2y = y2 - y;
  yy1 = y - y1;
  xx1 = x - x1;
  return 1.0 / (x2x1 * y2y1) *
         (q11 * x2x * y2y + q21 * xx1 * y2y + q12 * x2x * yy1 +
          q22 * xx1 * yy1);
}

template <typename T>
__host__ __device__ T 
get_h_cuda(const T * coords, int i, int stride) {
  return (i + stride - i);
}

template <typename T>
__host__ __device__
T get_dist_cuda(const T * coords, int i, int j) {
  return (j - i);
}


template <typename T>
__device__ T 
_get_dist(T * coords, int i, int j) {
  return coords[j] - coords[i];
}



template  __host__ __device__ double 
interp_2d_cuda<double>(double q11, double q12, double q21, double q22,
                        double x1, double x2, double y1, double y2, double x,
                        double y);
template  __host__ __device__ float 
interp_2d_cuda<float>(float q11, float q12, float q21, float q22,
                        float x1, float x2, float y1, float y2, float x,
                        float y);

template __host__ __device__ double 
get_h_cuda<double>(const double * coords, int i, int stride);
template __host__ __device__ float 
get_h_cuda<float>(const float * coords, int i, int stride);


template __host__ __device__
double get_dist_cuda<double>(const double * coords, int i, int j);
template __host__ __device__
float get_dist_cuda<float>(const float * coords, int i, int j);

template __device__ double 
_get_dist<double>(double * coords, int i, int j);
template __device__ float 
_get_dist<float>(float * coords, int i, int j);

} //end namespace mgard_common
}