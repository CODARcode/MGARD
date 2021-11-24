/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */
#include "cuda/CommonInternal.h"
 
#include "cuda/DataRefactoring/Coefficient/GridProcessingKernel.h"
#include "cuda/DataRefactoring/Coefficient/GridProcessingKernel.hpp"

namespace mgard_x {

#define KERNELS(D_GLOBAL, D_LOCAL, T, INTERPOLATION, COEFF_RESTORE, TYPE)      \
  template void                                                                \
  gpk_rev<D_GLOBAL, D_LOCAL, T, INTERPOLATION, COEFF_RESTORE, TYPE>(           \
      Handle<D_GLOBAL, T> & handle, SIZE *shape_h, SIZE *shape_d,                \
      SIZE *shape_c_d, SIZE *ldvs, SIZE *ldws, DIM unprocessed_n,                 \
      DIM *unprocessed_dims, DIM curr_dim_r, DIM curr_dim_c, DIM curr_dim_f,   \
      T *dratio_r, T *dratio_c, T *dratio_f, T *dv, LENGTH lddv1, LENGTH lddv2,      \
      T *dw, LENGTH lddw1, LENGTH lddw2, T *dwf, LENGTH lddwf1, LENGTH lddwf2, T *dwc,     \
      LENGTH lddwc1, LENGTH lddwc2, T *dwr, LENGTH lddwr1, LENGTH lddwr2, T *dwcf,         \
      LENGTH lddwcf1, LENGTH lddwcf2, T *dwrf, LENGTH lddwrf1, LENGTH lddwrf2, T *dwrc,    \
      LENGTH lddwrc1, LENGTH lddwrc2, T *dwrcf, LENGTH lddwrcf1, LENGTH lddwrcf2, SIZE svr, \
      SIZE svc, SIZE svf, SIZE nvr, SIZE nvc, SIZE nvf, int queue_idx, int config);

KERNELS(1, 1, double, false, true, 1)
KERNELS(1, 1, float, false, true, 1)
KERNELS(2, 2, double, false, true, 1)
KERNELS(2, 2, float, false, true, 1)
KERNELS(3, 3, double, false, true, 1)
KERNELS(3, 3, float, false, true, 1)

KERNELS(4, 2, double, false, true, 2)
KERNELS(4, 2, float, false, true, 2)
KERNELS(5, 2, double, false, true, 2)
KERNELS(5, 2, float, false, true, 2)

KERNELS(4, 3, double, false, true, 2)
KERNELS(4, 3, float, false, true, 2)
KERNELS(5, 3, double, false, true, 2)
KERNELS(5, 3, float, false, true, 2)

// #undef KERNELS

} // namespace mgard_x