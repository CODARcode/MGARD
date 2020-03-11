#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"
#include "mgard_cuda_compact_helper.h"
#include "mgard_cuda_helper_internal.h"
#include <fstream>
#include <cmath>

namespace mgard_2d {
namespace mgard_gen {  

__host__ __device__ int
get_lindex_cuda(const int n, const int no, const int i) {
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

template <typename T>
__host__ __device__ T 
get_h_l_cuda(const T * coords, const int n,
             const int no, int i, int stride) {

  //    return (*get_ref_cuda(coords, n, no, i+stride) - *get_ref_cuda(coords, n, no, i));
  return (get_lindex_cuda(n, no, i + stride) - get_lindex_cuda(n, no, i));
}

template <typename T>
__device__ T 
_get_h_l(const T * coords, int i, int stride) {

  //    return (*get_ref_cuda(coords, n, no, i+stride) - *get_ref_cuda(coords, n, no, i));
  return coords[i + stride] - coords[i];
}


template __host__ __device__ double 
get_h_l_cuda<double>(const double * coords, const int n,
             const int no, int i, int stride);
template __host__ __device__ float 
get_h_l_cuda<float>(const float * coords, const int n,
             const int no, int i, int stride);

template __device__ double 
_get_h_l<double>(const double * coords, int i, int stride);
template __device__ float 
_get_h_l<float>(const float * coords, int i, int stride);


}
}