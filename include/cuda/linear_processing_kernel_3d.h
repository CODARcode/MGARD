#include "cuda/mgard_cuda_common.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace mgard_cuda {

template <typename T, int D>
void lpk_reo_1_3d(mgard_cuda_handle<T, D> &handle, 
                               int nr, int nc, int nf, int nf_c,
                               int zero_r, int zero_c, int zero_f,
                               T *ddist_f, T *dratio_f, 
                               T *dv1, int lddv11, int lddv12,
                               T *dv2, int lddv21, int lddv22,
                               T *dw, int lddw1, int lddw2, 
                               int queue_idx, int config);

template <typename T, int D>
void lpk_reo_2_3d(mgard_cuda_handle<T, D> &handle, 
                         int nr, int nc, int nf_c, int nc_c,
                         T *ddist_c, T *dratio_c,
                         T *dv1, int lddv11, int lddv12,
                         T *dv2, int lddv21, int lddv22,
                         T *dw, int lddw1, int lddw2,
                         int queue_idx, int config);

template <typename T, int D>
void lpk_reo_3_3d(mgard_cuda_handle<T, D> &handle, 
                         int nr, int nc_c, int nf_c, int nr_c, 
                         T *ddist_r, T *dratio_r, 
                         T *dv1, int lddv11, int lddv12,
                         T *dv2, int lddv21, int lddv22,
                         T *dw, int lddw1, int lddw2, 
                         int queue_idx, int config);

}