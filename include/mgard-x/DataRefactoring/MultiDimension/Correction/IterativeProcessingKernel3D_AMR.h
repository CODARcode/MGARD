/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_ITERATIVE_PROCESSING_KERNEL_3D_AMR
#define MGARD_X_ITERATIVE_PROCESSING_KERNEL_3D_AMR

#include "../../Common.h"
#include "../../CommonInternal.h"

namespace mgard_x {

template <uint32_t D, typename T>
void ipk_1_3d_amr(Handle<D, T> &handle, int nr, int nc, int nf_c, T *am, T *bm,
                  T *ddist_f, T *dv, int lddv1, int lddv2, bool retrieve,
                  int block_size, T *fv, int ldfv1, int ldfv2, T *bv, int ldbv1,
                  int ldbv2, int queue_idx, int config);

// template <uint32_t D, typename T>
// void ipk_2_3d(Handle<D, T> &handle, int nr, int nc_c, int nf_c, T *am, T *bm,
//               T *ddist_c, T *dv, int lddv1, int lddv2, int queue_idx,
//               int config);

// template <uint32_t D, typename T>
// void ipk_3_3d(Handle<D, T> &handle, int nr_c, int nc_c, int nf_c, T *am, T
// *bm,
//               T *ddist_r, T *dv, int lddv1, int lddv2, int queue_idx,
//               int config);

} // namespace mgard_x

#endif