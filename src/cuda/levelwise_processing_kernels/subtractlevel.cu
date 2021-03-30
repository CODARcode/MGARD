#include "cuda/mgard_cuda_common_internal.h"
#include "cuda/levelwise_processing_kernel.h"
#include "cuda/levelwise_processing_kernel.hpp"

namespace mgard_cuda {

#define KERNELS(T, D) \
        template void lwpk<T, D, SUBTRACT>(\
        mgard_cuda_handle<T, D> &handle, thrust::device_vector<int> shape,\
        T *dv, thrust::device_vector<int> ldvs, T *dwork, thrust::device_vector<int> ldws,\
        int queue_idx);

KERNELS(double, 1)
KERNELS(float,  1)
KERNELS(double, 2)
KERNELS(float,  2)
KERNELS(double, 3)
KERNELS(float,  3)
KERNELS(double, 4)
KERNELS(float,  4)
KERNELS(double, 5)
KERNELS(float,  5)

#undef KERNELS

#define KERNELS(T, D) \
        template void lwpk<T, D, SUBTRACT>(\
        mgard_cuda_handle<T, D> &handle,\
        int * shape_h, int * shape_d,\
        T *dv, int * ldvs,\
        T *dwork, int * ldws,\
        int queue_idx);

KERNELS(double, 1)
KERNELS(float,  1)
KERNELS(double, 2)
KERNELS(float,  2)
KERNELS(double, 3)
KERNELS(float,  3)
KERNELS(double, 4)
KERNELS(float,  4)
KERNELS(double, 5)
KERNELS(float,  5)

#undef KERNELS

}