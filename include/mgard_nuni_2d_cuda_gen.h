#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"


namespace mgard_2d {
namespace mgard_gen {

__host__ __device__ int
get_lindex_cuda(const int n, const int no, const int i);

template <typename T>
__host__ __device__ T 
get_h_l_cuda(const T * coords, const int n,
             const int no, int i, int stride);

template <typename T>
__device__ T 
_get_h_l(const T * coords, int i, int stride);

}
}