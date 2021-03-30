#include "cuda/mgard_cuda_common_internal.h"
#include "cuda/linear_processing_kernel_3d.h"
#include "cuda/linear_processing_kernel_3d.hpp"

namespace mgard_cuda {

#define KERNELS(T, D) \
                        template void lpk_reo_1_3d<T, D>(\
                        mgard_cuda_handle<T, D> &handle, \
                               int nr, int nc, int nf, int nf_c,\
                               int zero_r, int zero_c, int zero_f,\
                               T *ddist_f, T *dratio_f, \
                               T *dv1, int lddv11, int lddv12,\
                               T *dv2, int lddv21, int lddv22,\
                               T *dw, int lddw1, int lddw2, int queue_idx, int config);

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

} //end namespace