/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#ifndef MGRAD_CUDA_GRID_PROCESSING_KERNEL
#define MGRAD_CUDA_GRID_PROCESSING_KERNEL

#include "../../Common.h"

namespace mgard_cuda {


template <DIM D_GLOBAL, DIM D_LOCAL, typename T, bool INTERPOLATION,
          bool CALC_COEFF, int TYPE>
void gpk_reo(Handle<D_GLOBAL, T> &handle, SIZE *shape_h, SIZE *shape_d,
             SIZE *shape_c_d, SIZE *ldvs, SIZE *ldws, DIM unprocessed_n,
             DIM *unprocessed_dims, DIM curr_dim_r, DIM curr_dim_c,
             DIM curr_dim_f, T *dratio_r, T *dratio_c, T *dratio_f, T *dv,
             LENGTH lddv1, LENGTH lddv2, T *dw, LENGTH lddw1, LENGTH lddw2, T *dwf,
             LENGTH lddwf1, LENGTH lddwf2, T *dwc, LENGTH lddwc1, LENGTH lddwc2, T *dwr,
             LENGTH lddwr1, LENGTH lddwr2, T *dwcf, LENGTH lddwcf1, LENGTH lddwcf2, T *dwrf,
             LENGTH lddwrf1, LENGTH lddwrf2, T *dwrc, LENGTH lddwrc1, LENGTH lddwrc2,
             T *dwrcf, LENGTH lddwrcf1, LENGTH lddwrcf2, int queue_idx, int config);

template <DIM D_GLOBAL, DIM D_LOCAL, typename T, bool INTERPOLATION,
          bool COEFF_RESTORE, int TYPE>
void gpk_rev(Handle<D_GLOBAL, T> &handle, SIZE *shape_h, SIZE *shape_d,
             SIZE *shape_c_d, SIZE *ldvs, SIZE *ldws, DIM unprocessed_n,
             DIM *unprocessed_dims, DIM curr_dim_r, DIM curr_dim_c,
             DIM curr_dim_f, T *dratio_r, T *dratio_c, T *dratio_f, T *dv,
             LENGTH lddv1, LENGTH lddv2, T *dw, LENGTH lddw1, LENGTH lddw2, T *dwf,
             LENGTH lddwf1, LENGTH lddwf2, T *dwc, LENGTH lddwc1, LENGTH lddwc2, T *dwr,
             LENGTH lddwr1, LENGTH lddwr2, T *dwcf, LENGTH lddwcf1, LENGTH lddwcf2, T *dwrf,
             LENGTH lddwrf1, LENGTH lddwrf2, T *dwrc, LENGTH lddwrc1, LENGTH lddwrc2,
             T *dwrcf, LENGTH lddwrcf1, LENGTH lddwrcf2, SIZE svr, SIZE svc, SIZE svf,
             SIZE nvr, SIZE nvc, SIZE nvf, int queue_idx, int config);

template <DIM D_GLOBAL, DIM D_LOCAL, typename T, bool INTERPOLATION, bool CALC_COEFF, int TYPE, typename DeviceType> class GpkReo;

} // namespace mgard_cuda

#endif