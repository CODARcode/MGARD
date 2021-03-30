#include "cuda/mgard_cuda_common.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace mgard_cuda {

template <typename T, int D, int OP>
void lwpk(mgard_cuda_handle<T, D> &handle, thrust::device_vector<int> shape,
                   T *dv, thrust::device_vector<int> ldvs, T *dwork, thrust::device_vector<int> ldws,
                   int queue_idx);

template <typename T, int D, int OP>
void lwpk(mgard_cuda_handle<T, D> &handle, 
          int * shape_h, int * shape_d,
          T *dv, int * ldvs, 
          T *dwork, int * ldws,
          int queue_idx);

}