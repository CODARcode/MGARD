/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../../../Hierarchy/Hierarchy.h"
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
    prefix +=
        std::to_string(hierarchy.level_shape(hierarchy.l_target(), d)) + "_";

  SubArray<D, T, DeviceType> dw_in1 = dcoeff;
  SubArray<D, T, DeviceType> dw_in2 = dcoeff;
  SubArray<D, T, DeviceType> dw_out = dcorrection;

  SubArray<1, SIZE, DeviceType> shape, shape_c;
  shape = SubArray<1, SIZE, DeviceType>(hierarchy.level_shape_array(l));
  shape_c = SubArray<1, SIZE, DeviceType>(hierarchy.level_shape_array(l - 1));

  DIM processed_n;
  SubArray<1, DIM, DeviceType> processed_dims;

  SubArray<1, T, DeviceType> dist, ratio, am, bm;

  // start correction calculation
  int prev_dim_f, prev_dim_c, prev_dim_r;
  int curr_dim_f, curr_dim_c, curr_dim_r;

  curr_dim_f = D - 1, curr_dim_c = D - 2, curr_dim_r = D - 3;

  dw_in1.resize(curr_dim_f, hierarchy.level_shape(l - 1, curr_dim_f));
  dw_in2.offset_dim(curr_dim_f, hierarchy.level_shape(l - 1, curr_dim_f));
  dw_in2.resize(curr_dim_f, hierarchy.level_shape(l, curr_dim_f) -
                                hierarchy.level_shape(l - 1, curr_dim_f));
  dw_out.resize(curr_dim_f, hierarchy.level_shape(l - 1, curr_dim_f));
  prev_dim_f = curr_dim_f;
  prev_dim_c = curr_dim_c;
  prev_dim_r = curr_dim_r;

  dw_in1.project(curr_dim_r, curr_dim_c, curr_dim_f);
  dw_in2.project(curr_dim_r, curr_dim_c, curr_dim_f);
  dw_out.project(curr_dim_r, curr_dim_c, curr_dim_f);

  dist = SubArray(hierarchy.dist(l, curr_dim_f));
  ratio = SubArray(hierarchy.ratio(l, curr_dim_f));

  processed_dims = SubArray(hierarchy.processed(0, processed_n));

  DeviceLauncher<DeviceType>::Execute(
      Lpk1ReoKernel<D, T, DeviceType>(
          shape, shape_c, processed_n, processed_dims, curr_dim_r, curr_dim_c,
          curr_dim_f, dist, ratio, dw_in1, dw_in2, dw_out),
      queue_idx);

  if (multidim_refactoring_debug_print) { // debug
    PrintSubarray4D(format("decomposition: after MR-1D[{}]", l), dw_out);
  }

  // mass trans 2D
  dw_in1 = dw_out;
  dw_in2 = dw_out;

  curr_dim_f = D - 1, curr_dim_c = D - 2, curr_dim_r = D - 3;
  dw_in1.resize(curr_dim_c, hierarchy.level_shape(l - 1, curr_dim_c));
  dw_in2.offset_dim(curr_dim_c, hierarchy.level_shape(l - 1, curr_dim_c));
  dw_in2.resize(curr_dim_c, hierarchy.level_shape(l, curr_dim_c) -
                                hierarchy.level_shape(l - 1, curr_dim_c));
  dw_out.offset_dim(prev_dim_f, hierarchy.level_shape(l - 1, curr_dim_f));
  dw_out.resize(curr_dim_c, hierarchy.level_shape(l - 1, curr_dim_c));
  prev_dim_f = curr_dim_f;
  prev_dim_c = curr_dim_c;
  prev_dim_r = curr_dim_r;

  dw_in1.project(curr_dim_r, curr_dim_c, curr_dim_f);
  dw_in2.project(curr_dim_r, curr_dim_c, curr_dim_f);
  dw_out.project(curr_dim_r, curr_dim_c, curr_dim_f);

  dist = SubArray(hierarchy.dist(l, curr_dim_c));
  ratio = SubArray(hierarchy.ratio(l, curr_dim_c));

  processed_dims = SubArray(hierarchy.processed(1, processed_n));

  DeviceLauncher<DeviceType>::Execute(
      Lpk2ReoKernel<D, T, DeviceType>(
          shape, shape_c, processed_n, processed_dims, curr_dim_r, curr_dim_c,
          curr_dim_f, dist, ratio, dw_in1, dw_in2, dw_out),
      queue_idx);

  if (multidim_refactoring_debug_print) { // debug
    PrintSubarray4D(format("decomposition: after MR-2D[{}]", l), dw_out);
  }

  // mass trans 3D
  dw_in1 = dw_out;
  dw_in2 = dw_out;

  curr_dim_f = D - 1, curr_dim_c = D - 2, curr_dim_r = D - 3;
  dw_in1.resize(curr_dim_r, hierarchy.level_shape(l - 1, curr_dim_r));
  dw_in2.offset_dim(curr_dim_r, hierarchy.level_shape(l - 1, curr_dim_r));
  dw_in2.resize(curr_dim_r, hierarchy.level_shape(l, curr_dim_r) -
                                hierarchy.level_shape(l - 1, curr_dim_r));
  dw_out.offset_dim(prev_dim_c, hierarchy.level_shape(l - 1, curr_dim_c));
  dw_out.resize(curr_dim_r, hierarchy.level_shape(l - 1, curr_dim_r));
  prev_dim_f = curr_dim_f;
  prev_dim_c = curr_dim_c;
  prev_dim_r = curr_dim_r;

  dw_in1.project(curr_dim_r, curr_dim_c, curr_dim_f);
  dw_in2.project(curr_dim_r, curr_dim_c, curr_dim_f);
  dw_out.project(curr_dim_r, curr_dim_c, curr_dim_f);

  dist = SubArray(hierarchy.dist(l, curr_dim_r));
  ratio = SubArray(hierarchy.ratio(l, curr_dim_r));

  processed_dims = SubArray(hierarchy.processed(2, processed_n));

  DeviceLauncher<DeviceType>::Execute(
      Lpk3ReoKernel<D, T, DeviceType>(
          shape, shape_c, processed_n, processed_dims, curr_dim_r, curr_dim_c,
          curr_dim_f, dist, ratio, dw_in1, dw_in2, dw_out),
      queue_idx);

  if (multidim_refactoring_debug_print) { // debug
    PrintSubarray4D(format("decomposition: after MR-3D[{}]", l), dw_out);
  }

  // mass trans 4D+
  for (int i = 3; i < D; i++) {
    dw_in1 = dw_out;
    dw_in2 = dw_out;

    curr_dim_f = D - 1, curr_dim_c = D - 2, curr_dim_r = D - (i + 1);
    dw_in1.resize(curr_dim_r, hierarchy.level_shape(l - 1, curr_dim_r));
    dw_in2.offset_dim(curr_dim_r, hierarchy.level_shape(l - 1, curr_dim_r));
    dw_in2.resize(curr_dim_r, hierarchy.level_shape(l, curr_dim_r) -
                                  hierarchy.level_shape(l - 1, curr_dim_r));
    dw_out.offset_dim(prev_dim_r, hierarchy.level_shape(l - 1, prev_dim_r));
    dw_out.resize(curr_dim_r, hierarchy.level_shape(l - 1, curr_dim_r));
    prev_dim_f = curr_dim_f;
    prev_dim_c = curr_dim_c;
    prev_dim_r = curr_dim_r;

    dw_in1.project(curr_dim_r, curr_dim_c, curr_dim_f);
    dw_in2.project(curr_dim_r, curr_dim_c, curr_dim_f);
    dw_out.project(curr_dim_r, curr_dim_c, curr_dim_f);

    dist = SubArray(hierarchy.dist(l, curr_dim_r));
    ratio = SubArray(hierarchy.ratio(l, curr_dim_r));

    processed_dims = SubArray(hierarchy.processed(i, processed_n));

    DeviceLauncher<DeviceType>::Execute(
        Lpk3ReoKernel<D, T, DeviceType>(
            shape, shape_c, processed_n, processed_dims, curr_dim_r, curr_dim_c,
            curr_dim_f, dist, ratio, dw_in1, dw_in2, dw_out),
        queue_idx);

    if (multidim_refactoring_debug_print) { // debug
      PrintSubarray4D(format("decomposition: after MR-{}D[{}]", i + 1, l),
                      dw_out);
    }
  }

  curr_dim_f = D - 1, curr_dim_c = D - 2, curr_dim_r = D - 3;
  dw_in1.project(curr_dim_r, curr_dim_c, curr_dim_f);
  dw_in2.project(curr_dim_r, curr_dim_c, curr_dim_f);
  dw_out.project(curr_dim_r, curr_dim_c, curr_dim_f);
  am = hierarchy.am(l - 1, curr_dim_f);
  bm = hierarchy.bm(l - 1, curr_dim_f);

  DeviceLauncher<DeviceType>::Execute(
      Ipk1ReoKernel<D, T, DeviceType>(curr_dim_r, curr_dim_c, curr_dim_f, am,
                                      bm, dw_out),
      queue_idx);

  if (multidim_refactoring_debug_print) { // debug
    PrintSubarray4D(format("decomposition: after TR-1D[{}]", l), dw_out);
  } // debug

  curr_dim_f = D - 1, curr_dim_c = D - 2, curr_dim_r = D - 3;
  dw_in1.project(curr_dim_r, curr_dim_c, curr_dim_f);
  dw_in2.project(curr_dim_r, curr_dim_c, curr_dim_f);
  dw_out.project(curr_dim_r, curr_dim_c, curr_dim_f);
  am = hierarchy.am(l - 1, curr_dim_c);
  bm = hierarchy.bm(l - 1, curr_dim_c);

  DeviceLauncher<DeviceType>::Execute(
      Ipk2ReoKernel<D, T, DeviceType>(curr_dim_r, curr_dim_c, curr_dim_f, am,
                                      bm, dw_out),
      queue_idx);

  if (multidim_refactoring_debug_print) { // debug
    PrintSubarray4D(format("decomposition: after TR-2D[{}]", l), dw_out);
  } // debug

  curr_dim_f = D - 1, curr_dim_c = D - 2, curr_dim_r = D - 3;
  dw_in1.project(curr_dim_r, curr_dim_c, curr_dim_f);
  dw_in2.project(curr_dim_r, curr_dim_c, curr_dim_f);
  dw_out.project(curr_dim_r, curr_dim_c, curr_dim_f);
  am = hierarchy.am(l - 1, curr_dim_r);
  bm = hierarchy.bm(l - 1, curr_dim_r);

  DeviceLauncher<DeviceType>::Execute(
      Ipk3ReoKernel<D, T, DeviceType>(curr_dim_r, curr_dim_c, curr_dim_f, am,
                                      bm, dw_out),
      queue_idx);

  if (multidim_refactoring_debug_print) { // debug
    PrintSubarray4D(format("decomposition: after TR-3D[{}]", l), dw_out);
  } // debug

  // mass trans 4D+
  for (int i = 3; i < D; i++) {
    curr_dim_f = D - 1, curr_dim_c = D - 2, curr_dim_r = D - (i + 1);
    dw_in1.project(curr_dim_r, curr_dim_c, curr_dim_f);
    dw_in2.project(curr_dim_r, curr_dim_c, curr_dim_f);
    dw_out.project(curr_dim_r, curr_dim_c, curr_dim_f);
    am = hierarchy.am(l - 1, curr_dim_r);
    bm = hierarchy.bm(l - 1, curr_dim_r);

    DeviceLauncher<DeviceType>::Execute(
        Ipk3ReoKernel<D, T, DeviceType>(curr_dim_r, curr_dim_c, curr_dim_f, am,
                                        bm, dw_out),
        queue_idx);

    if (multidim_refactoring_debug_print) { // debug
      PrintSubarray4D(format("decomposition: after TR-{}D[{}]", i + 1, l),
                      dw_out);
    } // debug
  }

  dcorrection = dw_out;
}

} // namespace mgard_x

#endif
