#include "cuda/mgard_cuda_common.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace mgard_cuda {

template <typename T, int D>
void refactor_reo(mgard_cuda_handle<T, D> &handle, T *dv, thrust::device_vector<int> lds, int l_target);

template <typename T, int D>
void recompose_reo(mgard_cuda_handle<T, D> &handle, T *dv, thrust::device_vector<int> lds, int l_target);

}