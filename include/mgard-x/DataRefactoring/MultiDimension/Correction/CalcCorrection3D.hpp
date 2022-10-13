/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../../../Hierarchy/Hierarchy.h"
#include "../../../RuntimeX/RuntimeX.h"

#include "../DataRefactoring.h"

#include "IterativeProcessingKernel3D.hpp"
#include "LinearProcessingKernel3D.hpp"

#ifndef MGARD_X_DATA_REFACTORING_CALC_CORRECTION_3D
#define MGARD_X_DATA_REFACTORING_CALC_CORRECTION_3D

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
void CalcCorrection3D(Hierarchy<D, T, DeviceType> &hierarchy,
                      SubArray<D, T, DeviceType> dcoeff,
                      SubArray<D, T, DeviceType> &dcorrection, SIZE l,
                      int queue_idx) {

  std::string prefix = "decomp_";
  if (sizeof(T) == sizeof(double))
    prefix += "d_";
  if (sizeof(T) == sizeof(float))
    prefix += "f_";
  for (int d = 0; d < D; d++)
    prefix +=
        std::to_string(hierarchy.level_shape(hierarchy.l_target(), d)) + "_";

  SIZE f = hierarchy.level_shape(l, D - 1);
  SIZE c = hierarchy.level_shape(l, D - 2);
  SIZE r = hierarchy.level_shape(l, D - 3);
  SIZE ff = hierarchy.level_shape(l - 1, D - 1);
  SIZE cc = hierarchy.level_shape(l - 1, D - 2);
  SIZE rr = hierarchy.level_shape(l - 1, D - 3);

  SubArray dist_f(hierarchy.dist(l, D - 1));
  SubArray dist_c(hierarchy.dist(l, D - 2));
  SubArray dist_r(hierarchy.dist(l, D - 3));

  SubArray dist_ff(hierarchy.dist(l - 1, D - 1));
  SubArray dist_cc(hierarchy.dist(l - 1, D - 2));
  SubArray dist_rr(hierarchy.dist(l - 1, D - 3));

  SubArray ratio_f(hierarchy.ratio(l, D - 1));
  SubArray ratio_c(hierarchy.ratio(l, D - 2));
  SubArray ratio_r(hierarchy.ratio(l, D - 3));

  SubArray am_ff(hierarchy.am(l - 1, D - 1));
  SubArray am_cc(hierarchy.am(l - 1, D - 2));
  SubArray am_rr(hierarchy.am(l - 1, D - 3));

  SubArray bm_ff(hierarchy.bm(l - 1, D - 1));
  SubArray bm_cc(hierarchy.bm(l - 1, D - 2));
  SubArray bm_rr(hierarchy.bm(l - 1, D - 3));

  SubArray<D, T, DeviceType> dw_in1, dw_in2, dw_out;

  if (D >= 1) {
    dw_in1 = dcoeff;
    dw_in1.resize({r, c, ff});
    dw_in2 = dcoeff;
    dw_in2.offset({0, 0, ff});
    dw_in2.resize({r, c, f - ff});
    dw_out = dcorrection;
    dw_out.resize({r, c, ff});

    DeviceLauncher<DeviceType>::Execute(
        Lpk1Reo3DKernel<D, T, DeviceType>(r, c, f, ff, rr, cc, ff, dist_f,
                                          ratio_f, dw_in1, dw_in2, dw_out),
        queue_idx);

    verify_matrix_cuda(r, c, ff, dw_out.data(), dw_out.ld(D - 1),
                       dw_out.ld(D - 2), dw_out.ld(D - 1),
                       prefix + "lpk_reo_1_3d" + "_level_" + std::to_string(l),
                       multidim_refactoring_store, multidim_refactoring_verify);

    if (multidim_refactoring_debug_print) {
      PrintSubarray("after mass_trans_multiply_1_cpt", dw_out);
    }
  }

  if (D >= 2) {
    dw_in1 = dw_out;
    dw_in1.resize({r, cc, ff});
    dw_in2 = dw_out;
    dw_in2.offset({0, cc, 0});
    dw_in2.resize({r, c - cc, ff});
    dw_out.offset({0, 0, ff});
    dw_out.resize({r, cc, ff});

    DeviceLauncher<DeviceType>::Execute(
        Lpk2Reo3DKernel<D, T, DeviceType>(r, c, ff, cc, dist_c, ratio_c, dw_in1,
                                          dw_in2, dw_out),
        queue_idx);

    verify_matrix_cuda(r, cc, ff, dw_out.data(), dw_out.ld(D - 1),
                       dw_out.ld(D - 2), dw_out.ld(D - 1),
                       prefix + "lpk_reo_2_3d" + "_level_" + std::to_string(l),
                       multidim_refactoring_store, multidim_refactoring_verify);

    if (multidim_refactoring_debug_print) {
      PrintSubarray("after mass_trans_multiply_2_cpt", dw_out);
    }
  }

  if (D == 3) {
    dw_in1 = dw_out;
    dw_in1.resize({rr, cc, ff});
    dw_in2 = dw_out;
    dw_in2.offset({rr, 0, 0});
    dw_in2.resize({r - rr, cc, r - ff});
    dw_out.offset({0, cc, ff});
    dw_out.resize({rr, cc, ff});

    DeviceLauncher<DeviceType>::Execute(
        Lpk3Reo3DKernel<D, T, DeviceType>(r, cc, ff, rr, dist_r, ratio_r,
                                          dw_in1, dw_in2, dw_out),
        queue_idx);

    verify_matrix_cuda(rr, cc, ff, dw_out.data(), dw_out.ld(D - 1),
                       dw_out.ld(D - 2), dw_out.ld(D - 1),
                       prefix + "lpk_reo_3_3d" + "_level_" + std::to_string(l),
                       multidim_refactoring_store, multidim_refactoring_verify);

    if (multidim_refactoring_debug_print) {
      PrintSubarray("after mass_trans_multiply_3_cpt", dw_out);
    }
  }

  if (D >= 1) {
    DeviceLauncher<DeviceType>::Execute(
        Ipk1Reo3DKernel<D, T, DeviceType>(rr, cc, ff, am_ff, bm_ff, dist_ff,
                                          dw_out),
        queue_idx);
    verify_matrix_cuda(rr, cc, ff, dw_out.data(), dw_out.ld(D - 1),
                       dw_out.ld(D - 2), dw_out.ld(D - 1),
                       prefix + "ipk_1_3d" + "_level_" + std::to_string(l),
                       multidim_refactoring_store, multidim_refactoring_verify);

    if (multidim_refactoring_debug_print) {
      PrintSubarray("after solve_tridiag_1_cpt", dw_out);
    }
  }
  if (D >= 2) {
    DeviceLauncher<DeviceType>::Execute(
        Ipk2Reo3DKernel<D, T, DeviceType>(rr, cc, ff, am_cc, bm_cc, dist_cc,
                                          dw_out),
        queue_idx);
    verify_matrix_cuda(rr, cc, ff, dw_out.data(), dw_out.ld(D - 1),
                       dw_out.ld(D - 2), dw_out.ld(D - 1),
                       prefix + "ipk_2_3d" + "_level_" + std::to_string(l),
                       multidim_refactoring_store, multidim_refactoring_verify);

    if (multidim_refactoring_debug_print) {
      PrintSubarray("after solve_tridiag_2_cpt", dw_out);
    }
  }
  if (D == 3) {
    DeviceLauncher<DeviceType>::Execute(
        Ipk3Reo3DKernel<D, T, DeviceType>(rr, cc, ff, am_rr, bm_rr, dist_rr,
                                          dw_out),
        queue_idx);
    verify_matrix_cuda(rr, cc, ff, dw_out.data(), dw_out.ld(D - 1),
                       dw_out.ld(D - 2), dw_out.ld(D - 1),
                       prefix + "ipk_3_3d" + "_level_" + std::to_string(l),
                       multidim_refactoring_store, multidim_refactoring_verify);

    if (multidim_refactoring_debug_print) {
      PrintSubarray("after solve_tridiag_3_cpt", dw_out);
    }
  }
  // final correction output
  dcorrection = dw_out;
}

} // namespace mgard_x

#endif