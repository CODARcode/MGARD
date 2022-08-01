/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../../../Hierarchy/Hierarchy.hpp"
#include "../../../RuntimeX/RuntimeX.h"

#include "../DataRefactoring.h"

#include "IterativeProcessingKernel.hpp"
#include "LinearProcessingKernel.hpp"

#ifndef MGARD_X_DATA_REFACTORING_CALC_CORRECTION_ND
#define MGARD_X_DATA_REFACTORING_CALC_CORRECTION_ND

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
void CalcCorrectionND(Hierarchy<D, T, DeviceType> &hierarchy,
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

  SubArray<D, T, DeviceType> dw_in1 = dcoeff;
  SubArray<D, T, DeviceType> dw_in2 = dcoeff;
  SubArray<D, T, DeviceType> dw_out = dcorrection;

  // start correction calculation
  int prev_dim_r, prev_dim_c, prev_dim_f;
  int curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;

  dw_in1.resize(curr_dim_f, hierarchy.dofs[curr_dim_f][l + 1]);
  dw_in2.offset(curr_dim_f, hierarchy.dofs[curr_dim_f][l + 1]);
  dw_in2.resize(curr_dim_f, hierarchy.dofs[curr_dim_f][l] -
                                hierarchy.dofs[curr_dim_f][l + 1]);
  dw_out.resize(curr_dim_f, hierarchy.dofs[curr_dim_f][l + 1]);

  dw_in1.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_in2.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_out.project(curr_dim_f, curr_dim_c, curr_dim_r);

  Lpk1Reo<D, T, DeviceType>().Execute(
      SubArray<1, SIZE, DeviceType>(hierarchy.shapes[l], true),
      SubArray<1, SIZE, DeviceType>(hierarchy.shapes[l + 1], true),
      hierarchy.processed_n[0],
      SubArray<1, DIM, DeviceType>(hierarchy.processed_dims[0], true),
      curr_dim_r, curr_dim_c, curr_dim_f,
      SubArray(hierarchy.dist_array[curr_dim_f][l]),
      SubArray(hierarchy.ratio_array[curr_dim_f][l]), dw_in1, dw_in2, dw_out,
      queue_idx);

  if (multidim_refactoring_debug_print) { // debug
    PrintSubarray4D(format("decomposition: after MR-1D[{}]", l), dw_out);
  }

  // mass trans 2D
  prev_dim_f = curr_dim_f;
  prev_dim_c = curr_dim_c;
  prev_dim_r = curr_dim_r;
  curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;

  dw_in1 = dw_out;
  dw_in2 = dw_out;
  dw_in1.resize(curr_dim_c, hierarchy.dofs[curr_dim_c][l + 1]);
  dw_in2.offset(curr_dim_c, hierarchy.dofs[curr_dim_c][l + 1]);
  dw_in2.resize(curr_dim_c, hierarchy.dofs[curr_dim_c][l] -
                                hierarchy.dofs[curr_dim_c][l + 1]);
  dw_out.offset(prev_dim_f, hierarchy.dofs[curr_dim_f][l + 1]);
  dw_out.resize(curr_dim_c, hierarchy.dofs[curr_dim_c][l + 1]);

  dw_in1.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_in2.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_out.project(curr_dim_f, curr_dim_c, curr_dim_r);

  Lpk2Reo<D, T, DeviceType>().Execute(
      SubArray<1, SIZE, DeviceType>(hierarchy.shapes[l], true),
      SubArray<1, SIZE, DeviceType>(hierarchy.shapes[l + 1], true),
      hierarchy.processed_n[1],
      SubArray<1, DIM, DeviceType>(hierarchy.processed_dims[1], true),
      curr_dim_r, curr_dim_c, curr_dim_f,
      SubArray(hierarchy.dist_array[curr_dim_c][l]),
      SubArray(hierarchy.ratio_array[curr_dim_c][l]), dw_in1, dw_in2, dw_out,
      queue_idx);

  if (multidim_refactoring_debug_print) { // debug
    PrintSubarray4D(format("decomposition: after MR-2D[{}]", l), dw_out);
  }

  // mass trans 3D

  prev_dim_f = curr_dim_f;
  prev_dim_c = curr_dim_c;
  prev_dim_r = curr_dim_r;
  curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;

  dw_in1 = dw_out;
  dw_in2 = dw_out;
  dw_in1.resize(curr_dim_r, hierarchy.dofs[curr_dim_r][l + 1]);
  dw_in2.offset(curr_dim_r, hierarchy.dofs[curr_dim_r][l + 1]);
  dw_in2.resize(curr_dim_r, hierarchy.dofs[curr_dim_r][l] -
                                hierarchy.dofs[curr_dim_r][l + 1]);
  dw_out.offset(prev_dim_c, hierarchy.dofs[curr_dim_c][l + 1]);
  dw_out.resize(curr_dim_r, hierarchy.dofs[curr_dim_r][l + 1]);

  dw_in1.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_in2.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_out.project(curr_dim_f, curr_dim_c, curr_dim_r);

  Lpk3Reo<D, T, DeviceType>().Execute(
      SubArray<1, SIZE, DeviceType>(hierarchy.shapes[l], true),
      SubArray<1, SIZE, DeviceType>(hierarchy.shapes[l + 1], true),
      hierarchy.processed_n[2],
      SubArray<1, DIM, DeviceType>(hierarchy.processed_dims[2], true),
      curr_dim_r, curr_dim_c, curr_dim_f,
      SubArray(hierarchy.dist_array[curr_dim_r][l]),
      SubArray(hierarchy.ratio_array[curr_dim_r][l]), dw_in1, dw_in2, dw_out,
      queue_idx);

  if (multidim_refactoring_debug_print) { // debug
    PrintSubarray4D(format("decomposition: after MR-3D[{}]", l), dw_out);
  }

  // mass trans 4D+
  for (int i = 3; i < D; i++) {
    prev_dim_f = curr_dim_f;
    prev_dim_c = curr_dim_c;
    prev_dim_r = curr_dim_r;
    curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = i;
    dw_in1 = dw_out;
    dw_in2 = dw_out;
    dw_in1.resize(curr_dim_r, hierarchy.dofs[curr_dim_r][l + 1]);
    dw_in2.offset(curr_dim_r, hierarchy.dofs[curr_dim_r][l + 1]);
    dw_in2.resize(curr_dim_r, hierarchy.dofs[curr_dim_r][l] -
                                  hierarchy.dofs[curr_dim_r][l + 1]);
    dw_out.offset(prev_dim_r, hierarchy.dofs[prev_dim_r][l + 1]);
    dw_out.resize(curr_dim_r, hierarchy.dofs[curr_dim_r][l + 1]);

    dw_in1.project(curr_dim_f, curr_dim_c, curr_dim_r);
    dw_in2.project(curr_dim_f, curr_dim_c, curr_dim_r);
    dw_out.project(curr_dim_f, curr_dim_c, curr_dim_r);
    Lpk3Reo<D, T, DeviceType>().Execute(
        SubArray<1, SIZE, DeviceType>(hierarchy.shapes[l], true),
        SubArray<1, SIZE, DeviceType>(hierarchy.shapes[l + 1], true),
        hierarchy.processed_n[i],
        SubArray<1, DIM, DeviceType>(hierarchy.processed_dims[i], true),
        curr_dim_r, curr_dim_c, curr_dim_f,
        SubArray(hierarchy.dist_array[curr_dim_r][l]),
        SubArray(hierarchy.ratio_array[curr_dim_r][l]), dw_in1, dw_in2, dw_out,
        queue_idx);

    if (multidim_refactoring_debug_print) { // debug
      PrintSubarray4D(format("decomposition: after MR-{}D[{}]", i + 1, l),
                      dw_out);
    }
  }

  curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;
  dw_in1.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_in2.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_out.project(curr_dim_f, curr_dim_c, curr_dim_r);
  Ipk1Reo<D, T, DeviceType>().Execute(
      curr_dim_r, curr_dim_c, curr_dim_f,
      SubArray(hierarchy.am_array[curr_dim_f][l + 1]),
      SubArray(hierarchy.bm_array[curr_dim_f][l + 1]), dw_out, queue_idx);

  if (multidim_refactoring_debug_print) { // debug
    PrintSubarray4D(format("decomposition: after TR-1D[{}]", l), dw_out);
  } // debug

  curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;
  dw_in1.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_in2.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_out.project(curr_dim_f, curr_dim_c, curr_dim_r);
  Ipk2Reo<D, T, DeviceType>().Execute(
      curr_dim_r, curr_dim_c, curr_dim_f,
      SubArray(hierarchy.am_array[curr_dim_c][l + 1]),
      SubArray(hierarchy.bm_array[curr_dim_c][l + 1]), dw_out, queue_idx);

  if (multidim_refactoring_debug_print) { // debug
    PrintSubarray4D(format("decomposition: after TR-2D[{}]", l), dw_out);
  } // debug

  curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;
  dw_in1.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_in2.project(curr_dim_f, curr_dim_c, curr_dim_r);
  dw_out.project(curr_dim_f, curr_dim_c, curr_dim_r);
  Ipk3Reo<D, T, DeviceType>().Execute(
      curr_dim_r, curr_dim_c, curr_dim_f,
      SubArray(hierarchy.am_array[curr_dim_r][l + 1]),
      SubArray(hierarchy.bm_array[curr_dim_r][l + 1]), dw_out, queue_idx);

  if (multidim_refactoring_debug_print) { // debug
    PrintSubarray4D(format("decomposition: after TR-3D[{}]", l), dw_out);
  } // debug

  // mass trans 4D+
  for (int i = 3; i < D; i++) {
    curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = i;
    dw_in1.project(curr_dim_f, curr_dim_c, curr_dim_r);
    dw_in2.project(curr_dim_f, curr_dim_c, curr_dim_r);
    dw_out.project(curr_dim_f, curr_dim_c, curr_dim_r);
    Ipk3Reo<D, T, DeviceType>().Execute(
        curr_dim_r, curr_dim_c, curr_dim_f,
        SubArray(hierarchy.am_array[curr_dim_r][l + 1]),
        SubArray(hierarchy.bm_array[curr_dim_r][l + 1]), dw_out, queue_idx);
    if (multidim_refactoring_debug_print) { // debug
      PrintSubarray4D(format("decomposition: after TR-{}D[{}]", i + 1, l),
                      dw_out);
    } // debug
  }

  dcorrection = dw_out;
}

} // namespace mgard_x

#endif