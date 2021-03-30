#include "cuda/mgard_cuda_common.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace mgard_cuda {

template <typename T, int D>
void ipk_1_3d(mgard_cuda_handle<T, D> &handle, int nr, int nc, int nf_c, T* am, T *bm, T * ddist_f, T *dv,
                                 int lddv1, int lddv2, int queue_idx, int config);

template <typename T, int D>
void ipk_2_3d(mgard_cuda_handle<T, D> &handle, int nr, int nc_c, int nf_c, T* am, T *bm, T * ddist_c, T *dv,
                                 int lddv1, int lddv2, int queue_idx, int config);

template <typename T, int D>
void ipk_3_3d(mgard_cuda_handle<T, D> &handle, int nr_c, int nc_c, int nf_c, T* am, T *bm, T * ddist_r, T *dv,
                                 int lddv1, int lddv2, int queue_idx, int config);

}