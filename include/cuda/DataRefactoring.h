/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGRAD_CUDA_DATA_REFACTORING
#define MGRAD_CUDA_DATA_REFACTORING

#include "Common.h"

namespace mgard_cuda {

// template <DIM D, typename T>
// void calc_coeff_pointers(Handle<D, T> &handle, DIM curr_dims[3], DIM l,
// SubArray<D, T> doutput,
//                          SubArray<D, T> &dcoarse,
//                          SubArray<D, T> &dcoeff_f,
//                          SubArray<D, T> &dcoeff_c,
//                          SubArray<D, T> &dcoeff_r,
//                          SubArray<D, T> &dcoeff_cf,
//                          SubArray<D, T> &dcoeff_rf,
//                          SubArray<D, T> &dcoeff_rc,
//                          SubArray<D, T> &dcoeff_rcf);

// template <DIM D, typename T>
// void calc_coefficients_3d(Handle<D, T> &handle, SubArray<D, T> dinput,
//                         SubArray<D, T> &doutput, SIZE l, int queue_idx);

// template <DIM D, typename T>
// void coefficients_restore_3d(Handle<D, T> &handle, SubArray<D, T> dinput,
//                         SubArray<D, T> &doutput, SIZE l, int queue_idx);

// template <DIM D, typename T>
// void calc_correction_3d(Handle<D, T> &handle, SubArray<D, T> dcoeff,
//                         SubArray<D, T> &dcorrection, SIZE l, int queue_idx);

// template <DIM D, typename T>
// void calc_coefficients_nd(Handle<D, T> &handle, SubArray<D, T> dinput1,
//                           SubArray<D, T> dinput2,
//                         SubArray<D, T> &doutput, SIZE l, int queue_idx);

// template <DIM D, typename T>
// void coefficients_restore_nd(Handle<D, T> &handle, SubArray<D, T> dinput1,
//                              SubArray<D, T> dinput2,
//                              SubArray<D, T> &doutput, SIZE l, int queue_idx);

// template <DIM D, typename T>
// void calc_correction_nd(Handle<D, T> &handle, SubArray<D, T> dcoeff,
//                         SubArray<D, T> &dcorrection, SIZE l, int queue_idx);

template <DIM D, typename T>
void decompose(Handle<D, T> &handle, T *dv, std::vector<SIZE> ldvs_h,
               SIZE *ldvs_d, SIZE l_target, int queue_idx);

template <DIM D, typename T>
void recompose(Handle<D, T> &handle, T *dv, std::vector<SIZE> ldvs_h,
               SIZE *ldvs_d, SIZE l_target, int queue_idx);

} // namespace mgard_cuda

#endif