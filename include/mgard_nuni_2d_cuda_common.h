#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"

namespace mgard_2d {
namespace mgard_common {

__host__ __device__
int get_index_cuda(const int ncol, const int i, const int j);

template <typename T>
__host__ __device__ T 
interp_2d_cuda(T q11, T q12, T q21, T q22,
                        T x1, T x2, T y1, T y2, T x,
                        T y);

template <typename T>
__host__ __device__ T 
get_h_cuda(const T * coords, int i, int stride);

template <typename T>
__host__ __device__
T get_dist_cuda(const T * coords, int i, int j);

template <typename T>
__device__ T 
_get_dist(T * coords, int i, int j);

}
}
