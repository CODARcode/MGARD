#include "cuda/mgard_cuda_common.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace mgard_cuda {

template <typename T, int D>
void levelwise_linear_quantize(mgard_cuda_handle<T, D> &handle, 
                     int * shapes, int l_target,
                     quant_meta<T> m, 
                     T *dv, int * ldvs, 
                     int *dwork, int * ldws, 
                     bool prep_huffmam,
                     int * shape,
                     size_t * outlier_count, unsigned int * outlier_idx, int * outliers,
                    int queue_idx);


template <typename T, int D>
void levelwise_linear_dequantize(mgard_cuda_handle<T, D> &handle, 
                    int * shapes, int l_target,
                    quant_meta<T> m,
                    int *dv, int * ldvs, 
                    T *dwork, int * ldws, 
                    size_t outlier_count, unsigned int * outlier_idx, int * outliers,
                    int queue_idx);

}