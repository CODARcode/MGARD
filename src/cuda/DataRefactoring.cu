/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */


#include "cuda/DataRefactoring.hpp"

#include <iostream>

#include <chrono>
namespace mgard_cuda {

#define KERNELS(D, T)                                                          \
  template void calc_coeff_pointers<D, T>(Handle<D, T> &handle, DIM curr_dims[3], DIM l, \
                         SubArray<D, T> doutput, \
                         SubArray<D, T> &dcoarse, \
                         SubArray<D, T> &dcoeff_f, \
                         SubArray<D, T> &dcoeff_c, \
                         SubArray<D, T> &dcoeff_r, \
                         SubArray<D, T> &dcoeff_cf, \
                         SubArray<D, T> &dcoeff_rf, \
                         SubArray<D, T> &dcoeff_rc, \
                         SubArray<D, T> &dcoeff_rcf); \
  template void decompose<D, T>(Handle<D, T> & handle, T * dv,                 \
                                std::vector<SIZE> ldvs_h, SIZE * ldvs_d, SIZE l_target, int queue_idx);   \
  template void recompose<D, T>(Handle<D, T> & handle, T * dv,                 \
                                std::vector<SIZE> ldvs_h, SIZE * ldvs_d, SIZE l_target, int queue_idx);

KERNELS(1, double)
KERNELS(1, float)
KERNELS(2, double)
KERNELS(2, float)
KERNELS(3, double)
KERNELS(3, float)
KERNELS(4, double)
KERNELS(4, float)
KERNELS(5, double)
KERNELS(5, float)
#undef KERNELS

#define KERNELS(D, T)     \
template void calc_coefficients_3d<D, T>(Handle<D, T> &handle, SubArray<D, T> dinput, \
                        SubArray<D, T> &doutput, SIZE l, int queue_idx);\
template void coefficients_restore_3d<D, T>(Handle<D, T> &handle, SubArray<D, T> dinput, \
                      SubArray<D, T> &doutput, SIZE l, int queue_idx);\
template void calc_correction_3d<D, T>(Handle<D, T> &handle, SubArray<D, T> dcoeff, \
                      SubArray<D, T> &dcorrection, SIZE l, int queue_idx);
KERNELS(1, double)
KERNELS(1, float)
KERNELS(2, double)
KERNELS(2, float)
KERNELS(3, double)
KERNELS(3, float)
#undef KERNELS


#define KERNELS(D, T)     \
template void calc_coefficients_nd<D, T>(Handle<D, T> &handle, SubArray<D, T> dinput1, \
                          SubArray<D, T> dinput2, \
                          SubArray<D, T> &doutput, SIZE l, int queue_idx);\
template void coefficients_restore_nd<D, T>(Handle<D, T> &handle, SubArray<D, T> dinput1, \
                          SubArray<D, T> dinput2, \
                          SubArray<D, T> &doutput, SIZE l, int queue_idx);\
template void calc_correction_nd<D, T>(Handle<D, T> &handle, SubArray<D, T> dcoeff, \
                          SubArray<D, T> &dcorrection, SIZE l, int queue_idx);

KERNELS(4, double)
KERNELS(4, float)
KERNELS(5, double)
KERNELS(5, float)
#undef KERNELS

} // namespace mgard_cuda
