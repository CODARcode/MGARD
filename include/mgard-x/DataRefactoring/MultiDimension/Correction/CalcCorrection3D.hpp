/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../../../Hierarchy/Hierarchy.hpp"
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
    prefix += std::to_string(hierarchy.shape[d]) + "_";

  SubArray<D, T, DeviceType> dw_in1, dw_in2, dw_out;

  if (D >= 1) {
    dw_in1 = dcoeff;
    dw_in1.resize(
        {hierarchy.dofs[0][l + 1], hierarchy.dofs[1][l], hierarchy.dofs[2][l]});
    dw_in2 = dcoeff;
    dw_in2.offset({hierarchy.dofs[0][l + 1], 0, 0});
    dw_in2.resize({hierarchy.dofs[0][l] - hierarchy.dofs[0][l + 1],
                   hierarchy.dofs[1][l], hierarchy.dofs[2][l]});
    dw_out = dcorrection;
    dw_out.resize(
        {hierarchy.dofs[0][l + 1], hierarchy.dofs[1][l], hierarchy.dofs[2][l]});

    Lpk1Reo3D<D, T, DeviceType>().Execute(
        hierarchy.dofs[2][l], hierarchy.dofs[1][l], hierarchy.dofs[0][l],
        hierarchy.dofs[0][l + 1], hierarchy.dofs[2][l + 1],
        hierarchy.dofs[1][l + 1], hierarchy.dofs[0][l + 1],
        SubArray(hierarchy.dist_array[0][l]),
        SubArray(hierarchy.ratio_array[0][l]), dw_in1, dw_in2, dw_out,
        queue_idx);

    verify_matrix_cuda(hierarchy.dofs[2][l], hierarchy.dofs[1][l],
                       hierarchy.dofs[0][l + 1], dw_out.data(), dw_out.getLd(0),
                       dw_out.getLd(1), dw_out.getLd(0),
                       prefix + "lpk_reo_1_3d" + "_level_" + std::to_string(l),
                       multidim_refactoring_store, multidim_refactoring_verify);

    if (multidim_refactoring_debug_print) {
      PrintSubarray("after mass_trans_multiply_1_cpt", dw_out);
    }
  }

  if (D >= 2) {
    dw_in1 = dw_out;
    dw_in1.resize({hierarchy.dofs[0][l + 1], hierarchy.dofs[1][l + 1],
                   hierarchy.dofs[2][l]});
    dw_in2 = dw_out;
    dw_in2.offset({0, hierarchy.dofs[1][l + 1], 0});
    dw_in2.resize({hierarchy.dofs[0][l + 1],
                   hierarchy.dofs[1][l] - hierarchy.dofs[1][l + 1],
                   hierarchy.dofs[2][l]});
    dw_out.offset({hierarchy.dofs[0][l + 1], 0, 0});
    dw_out.resize({hierarchy.dofs[0][l + 1], hierarchy.dofs[1][l + 1],
                   hierarchy.dofs[2][l]});

    Lpk2Reo3D<D, T, DeviceType>().Execute(
        hierarchy.dofs[2][l], hierarchy.dofs[1][l], hierarchy.dofs[0][l + 1],
        hierarchy.dofs[1][l + 1], SubArray(hierarchy.dist_array[1][l]),
        SubArray(hierarchy.ratio_array[1][l]), dw_in1, dw_in2, dw_out,
        queue_idx);

    verify_matrix_cuda(hierarchy.dofs[2][l], hierarchy.dofs[1][l + 1],
                       hierarchy.dofs[0][l + 1], dw_out.data(), dw_out.getLd(0),
                       dw_out.getLd(1), dw_out.getLd(0),
                       prefix + "lpk_reo_2_3d" + "_level_" + std::to_string(l),
                       multidim_refactoring_store, multidim_refactoring_verify);

    if (multidim_refactoring_debug_print) {
      PrintSubarray("after mass_trans_multiply_2_cpt", dw_out);
    }
  }

  if (D == 3) {
    dw_in1 = dw_out;
    dw_in1.resize({hierarchy.dofs[0][l + 1], hierarchy.dofs[1][l + 1],
                   hierarchy.dofs[2][l + 1]});
    dw_in2 = dw_out;
    dw_in2.offset({0, 0, hierarchy.dofs[2][l + 1]});
    dw_in2.resize({hierarchy.dofs[0][l + 1], hierarchy.dofs[1][l + 1],
                   hierarchy.dofs[2][l] - hierarchy.dofs[2][l + 1]});
    dw_out.offset({hierarchy.dofs[0][l + 1], hierarchy.dofs[1][l + 1], 0});
    dw_out.resize({hierarchy.dofs[0][l + 1], hierarchy.dofs[1][l + 1],
                   hierarchy.dofs[2][l + 1]});

    Lpk3Reo3D<D, T, DeviceType>().Execute(
        hierarchy.dofs[2][l], hierarchy.dofs[1][l + 1],
        hierarchy.dofs[0][l + 1], hierarchy.dofs[2][l + 1],
        SubArray(hierarchy.dist_array[2][l]),
        SubArray(hierarchy.ratio_array[2][l]), dw_in1, dw_in2, dw_out,
        queue_idx);

    verify_matrix_cuda(hierarchy.dofs[2][l + 1], hierarchy.dofs[1][l + 1],
                       hierarchy.dofs[0][l + 1], dw_out.data(), dw_out.getLd(0),
                       dw_out.getLd(1), dw_out.getLd(0),
                       prefix + "lpk_reo_3_3d" + "_level_" + std::to_string(l),
                       multidim_refactoring_store, multidim_refactoring_verify);

    if (multidim_refactoring_debug_print) {
      PrintSubarray("after mass_trans_multiply_3_cpt", dw_out);
    }
  }

  if (D >= 1) {
    Ipk1Reo3D<D, T, DeviceType>().Execute(
        hierarchy.dofs[2][l + 1], hierarchy.dofs[1][l + 1],
        hierarchy.dofs[0][l + 1], SubArray(hierarchy.am_array[0][l + 1]),
        SubArray(hierarchy.bm_array[0][l + 1]),
        SubArray(hierarchy.dist_array[0][l + 1]), dw_out, queue_idx);
    verify_matrix_cuda(hierarchy.dofs[2][l + 1], hierarchy.dofs[1][l + 1],
                       hierarchy.dofs[0][l + 1], dw_out.data(), dw_out.getLd(0),
                       dw_out.getLd(1), dw_out.getLd(0),
                       prefix + "ipk_1_3d" + "_level_" + std::to_string(l),
                       multidim_refactoring_store, multidim_refactoring_verify);

    if (multidim_refactoring_debug_print) {
      PrintSubarray("after solve_tridiag_1_cpt", dw_out);
    }
  }
  if (D >= 2) {
    Ipk2Reo3D<D, T, DeviceType>().Execute(
        hierarchy.dofs[2][l + 1], hierarchy.dofs[1][l + 1],
        hierarchy.dofs[0][l + 1], SubArray(hierarchy.am_array[1][l + 1]),
        SubArray(hierarchy.bm_array[1][l + 1]),
        SubArray(hierarchy.dist_array[1][l + 1]), dw_out, queue_idx);

    verify_matrix_cuda(hierarchy.dofs[2][l + 1], hierarchy.dofs[1][l + 1],
                       hierarchy.dofs[0][l + 1], dw_out.data(), dw_out.getLd(0),
                       dw_out.getLd(1), dw_out.getLd(0),
                       prefix + "ipk_2_3d" + "_level_" + std::to_string(l),
                       multidim_refactoring_store, multidim_refactoring_verify);

    if (multidim_refactoring_debug_print) {
      PrintSubarray("after solve_tridiag_2_cpt", dw_out);
    }
  }
  if (D == 3) {
    Ipk3Reo3D<D, T, DeviceType>().Execute(
        hierarchy.dofs[2][l + 1], hierarchy.dofs[1][l + 1],
        hierarchy.dofs[0][l + 1], SubArray(hierarchy.am_array[2][l + 1]),
        SubArray(hierarchy.bm_array[2][l + 1]),
        SubArray(hierarchy.dist_array[2][l + 1]), dw_out, queue_idx);

    verify_matrix_cuda(hierarchy.dofs[2][l + 1], hierarchy.dofs[1][l + 1],
                       hierarchy.dofs[0][l + 1], dw_out.data(), dw_out.getLd(0),
                       dw_out.getLd(1), dw_out.getLd(0),
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