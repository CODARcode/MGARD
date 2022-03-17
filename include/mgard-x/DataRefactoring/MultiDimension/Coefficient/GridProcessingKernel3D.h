/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_GRID_PROCESSING_KERNEL_3D
#define MGARD_X_GRID_PROCESSING_KERNEL_3D

#include "../../Common.h"

namespace mgard_x {

template <DIM D, typename T>
void gpk_reo_3d(Handle<D, T> &handle, SIZE nr, SIZE nc, SIZE nf, T *dratio_r,
                T *dratio_c, T *dratio_f, T *dv, SIZE lddv1, SIZE lddv2, T *dw,
                SIZE lddw1, SIZE lddw2, T *dwf, SIZE lddwf1, SIZE lddwf2,
                T *dwc, SIZE lddwc1, SIZE lddwc2, T *dwr, SIZE lddwr1,
                SIZE lddwr2, T *dwcf, SIZE lddwcf1, SIZE lddwcf2, T *dwrf,
                SIZE lddwrf1, SIZE lddwrf2, T *dwrc, SIZE lddwrc1, SIZE lddwrc2,
                T *dwrcf, SIZE lddwrcf1, SIZE lddwrcf2, int queue_idx,
                int config);

template <DIM D, typename T>
void gpk_rev_3d(Handle<D, T> &handle, SIZE nr, SIZE nc, SIZE nf, T *dratio_r,
                T *dratio_c, T *dratio_f, T *dv, SIZE lddv1, SIZE lddv2, T *dw,
                SIZE lddw1, SIZE lddw2, T *dwf, SIZE lddwf1, SIZE lddwf2,
                T *dwc, SIZE lddwc1, SIZE lddwc2, T *dwr, SIZE lddwr1,
                SIZE lddwr2, T *dwcf, SIZE lddwcf1, SIZE lddwcf2, T *dwrf,
                SIZE lddwrf1, SIZE lddwrf2, T *dwrc, SIZE lddwrc1, SIZE lddwrc2,
                T *dwrcf, SIZE lddwrcf1, SIZE lddwrcf2, SIZE svr, SIZE svc,
                SIZE svf, SIZE nvr, SIZE nvc, SIZE nvf, int queue_idx,
                int config);

} // namespace mgard_x

#endif