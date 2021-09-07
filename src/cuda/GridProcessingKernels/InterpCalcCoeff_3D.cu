/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */
#include "cuda/CommonInternal.h"

#include "cuda/GridProcessingKernel3D.h"
#include "cuda/GridProcessingKernel3D.hpp"

namespace mgard_cuda {

#define KERNELS(D, T)                                                          \
  template void gpk_reo_3d<D, T>(                                              \
      Handle<D, T> & handle, SIZE nr, SIZE nc, SIZE nf, T * dratio_r,          \
      T * dratio_c, T * dratio_f, T * dv, SIZE lddv1, SIZE lddv2, T * dw,      \
      SIZE lddw1, SIZE lddw2, T * dwf, SIZE lddwf1, SIZE lddwf2, T * dwc,      \
      SIZE lddwc1, SIZE lddwc2, T * dwr, SIZE lddwr1, SIZE lddwr2, T * dwcf,   \
      SIZE lddwcf1, SIZE lddwcf2, T * dwrf, SIZE lddwrf1, SIZE lddwrf2,        \
      T * dwrc, SIZE lddwrc1, SIZE lddwrc2, T * dwrcf, SIZE lddwrcf1,          \
      SIZE lddwrcf2, int queue_idx, int config);

KERNELS(1, double)
KERNELS(1, float)
KERNELS(2, double)
KERNELS(2, float)
KERNELS(3, double)
KERNELS(3, float)

#undef KERNELS

} // namespace mgard_cuda