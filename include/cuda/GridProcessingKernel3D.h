/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGRAD_CUDA_GRID_PROCESSING_KERNEL_3D
#define MGRAD_CUDA_GRID_PROCESSING_KERNEL_3D

#include "Common.h"
#include "CommonInternal.h"

namespace mgard_cuda {

template <uint32_t D, typename T>
void gpk_reo_3d(Handle<D, T> &handle, int nr, int nc, int nf, T *dratio_r,
                T *dratio_c, T *dratio_f, T *dv, int lddv1, int lddv2, T *dw,
                int lddw1, int lddw2, T *dwf, int lddwf1, int lddwf2, T *dwc,
                int lddwc1, int lddwc2, T *dwr, int lddwr1, int lddwr2, T *dwcf,
                int lddwcf1, int lddwcf2, T *dwrf, int lddwrf1, int lddwrf2,
                T *dwrc, int lddwrc1, int lddwrc2, T *dwrcf, int lddwrcf1,
                int lddwrcf2, int queue_idx, int config);

template <uint32_t D, typename T>
void gpk_rev_3d(Handle<D, T> &handle, int nr, int nc, int nf, T *dratio_r,
                T *dratio_c, T *dratio_f, T *dv, int lddv1, int lddv2, T *dw,
                int lddw1, int lddw2, T *dwf, int lddwf1, int lddwf2, T *dwc,
                int lddwc1, int lddwc2, T *dwr, int lddwr1, int lddwr2, T *dwcf,
                int lddwcf1, int lddwcf2, T *dwrf, int lddwrf1, int lddwrf2,
                T *dwrc, int lddwrc1, int lddwrc2, T *dwrcf, int lddwrcf1,
                int lddwrcf2, int svr, int svc, int svf, int nvr, int nvc,
                int nvf, int queue_idx, int config);

} // namespace mgard_cuda

#endif