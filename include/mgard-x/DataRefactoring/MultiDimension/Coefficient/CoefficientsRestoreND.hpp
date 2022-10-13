/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../../../Hierarchy/Hierarchy.h"
#include "../../../RuntimeX/RuntimeX.h"

#include "../DataRefactoring.h"

#include "CalcCoefficientsPointers.hpp"
#include "GridProcessingKernel.hpp"

#ifndef MGARD_X_DATA_REFACTORING_COEFFICIENTS_RESTORE_ND
#define MGARD_X_DATA_REFACTORING_COEFFICIENTS_RESTORE_ND

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
void CoefficientsRestoreND(Hierarchy<D, T, DeviceType> &hierarchy,
                           SubArray<D, T, DeviceType> dinput1,
                           SubArray<D, T, DeviceType> dinput2,
                           SubArray<D, T, DeviceType> &doutput, SIZE l,
                           int queue_idx) {

  std::string prefix = "decomp_";
  if (sizeof(T) == sizeof(double))
    prefix += "d_";
  if (sizeof(T) == sizeof(float))
    prefix += "f_";
  for (int d = 0; d < D; d++)
    prefix +=
        std::to_string(hierarchy.level_shape(hierarchy.l_target(), d)) + "_";

  SubArray<D, T, DeviceType> dcoarse, dcoeff_f, dcoeff_c, dcoeff_r, dcoeff_cf,
      dcoeff_rf, dcoeff_rc, dcoeff_rcf;

  DIM curr_dims[3];

  SubArray<1, T, DeviceType> ratio_r, ratio_c, ratio_f;
  SubArray<1, SIZE, DeviceType> shape, shape_c;
  DIM unprocessed_n;
  SubArray<1, DIM, DeviceType> unprocessed_dims;

  shape = SubArray<1, SIZE, DeviceType>(hierarchy.level_shape_array(l), true);
  shape_c =
      SubArray<1, SIZE, DeviceType>(hierarchy.level_shape_array(l - 1), true);

  int unprocessed_idx = 0;
  unprocessed_dims = hierarchy.unprocessed(unprocessed_idx, unprocessed_n);
  // printf("interpolate-restore 1-3D\n");

  curr_dims[0] = D - 3;
  curr_dims[1] = D - 2;
  curr_dims[2] = D - 1;

  dinput1.project(curr_dims[0], curr_dims[1], curr_dims[2]);
  doutput.project(curr_dims[0], curr_dims[1], curr_dims[2]);

  CalcCoefficientsPointers(hierarchy, curr_dims, l, dinput1, dcoarse, dcoeff_f,
                           dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf, dcoeff_rc,
                           dcoeff_rcf);
  ratio_r = SubArray(hierarchy.ratio(l, curr_dims[0]));
  ratio_c = SubArray(hierarchy.ratio(l, curr_dims[1]));
  ratio_f = SubArray(hierarchy.ratio(l, curr_dims[2]));
  DeviceLauncher<DeviceType>::Execute(
      GpkRevKernel<D, 3, T, true, false, 1, DeviceType>(
          shape, shape_c, unprocessed_n, unprocessed_dims, curr_dims[0],
          curr_dims[1], curr_dims[2], ratio_r, ratio_c, ratio_f, doutput,
          dcoarse, dcoeff_f, dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf,
          dcoeff_rc, dcoeff_rcf, 0, 0, 0,
          hierarchy.level_shape(l, curr_dims[0]),
          hierarchy.level_shape(l, curr_dims[1]),
          hierarchy.level_shape(l, curr_dims[2])),
      queue_idx);

  for (DIM d = 3; d < D; d += 2) {
    CopyND(doutput, dinput1, queue_idx);

    // printf("interpolate-restore %u-%uD\n", d+1, d+2);
    curr_dims[0] = D - (d + 1 + 1);
    curr_dims[1] = D - (d + 1);
    curr_dims[2] = D - 1;

    dinput1.project(curr_dims[0], curr_dims[1], curr_dims[2]);
    doutput.project(curr_dims[0], curr_dims[1], curr_dims[2]);

    CalcCoefficientsPointers(hierarchy, curr_dims, l, dinput1, dcoarse,
                             dcoeff_f, dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf,
                             dcoeff_rc, dcoeff_rcf);
    ratio_r = SubArray(hierarchy.ratio(l, curr_dims[0]));
    ratio_c = SubArray(hierarchy.ratio(l, curr_dims[1]));
    ratio_f = SubArray(hierarchy.ratio(l, curr_dims[2]));

    if (D - d == 1) {
      unprocessed_idx += 1;
      unprocessed_dims = hierarchy.unprocessed(unprocessed_idx, unprocessed_n);
      DeviceLauncher<DeviceType>::Execute(
          GpkRevKernel<D, 2, T, true, false, 2, DeviceType>(
              shape, shape_c, unprocessed_n, unprocessed_dims, curr_dims[0],
              curr_dims[1], curr_dims[2], ratio_r, ratio_c, ratio_f, doutput,
              dcoarse, dcoeff_f, dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf,
              dcoeff_rc, dcoeff_rcf, 0, 0, 0,
              hierarchy.level_shape(l, curr_dims[0]),
              hierarchy.level_shape(l, curr_dims[1]),
              hierarchy.level_shape(l, curr_dims[2])),
          queue_idx);

    } else { // D - d >= 2
      unprocessed_idx += 2;
      unprocessed_dims = hierarchy.unprocessed(unprocessed_idx, unprocessed_n);
      DeviceLauncher<DeviceType>::Execute(
          GpkRevKernel<D, 3, T, true, false, 2, DeviceType>(
              shape, shape_c, unprocessed_n, unprocessed_dims, curr_dims[0],
              curr_dims[1], curr_dims[2], ratio_r, ratio_c, ratio_f, doutput,
              dcoarse, dcoeff_f, dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf,
              dcoeff_rc, dcoeff_rcf, 0, 0, 0,
              hierarchy.level_shape(l, curr_dims[0]),
              hierarchy.level_shape(l, curr_dims[1]),
              hierarchy.level_shape(l, curr_dims[2])),
          queue_idx);
    }
  }
  // Done interpolation-restore on doutput

  if (multidim_refactoring_debug_print) { // debug
    PrintSubarray4D("After interpolation reverse-reorder", doutput);
  } // debug

  unprocessed_idx = 0;
  unprocessed_dims = hierarchy.unprocessed(unprocessed_idx, unprocessed_n);

  // printf("reorder-restore 1-3D\n");

  curr_dims[0] = D - 3;
  curr_dims[1] = D - 2;
  curr_dims[2] = D - 1;

  dinput2.project(curr_dims[0], curr_dims[1], curr_dims[2]);
  dinput1.project(curr_dims[0], curr_dims[1],
                  curr_dims[2]); // reuse input1 as temp space

  CalcCoefficientsPointers(hierarchy, curr_dims, l, dinput2, dcoarse, dcoeff_f,
                           dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf, dcoeff_rc,
                           dcoeff_rcf);

  ratio_r = SubArray(hierarchy.ratio(l, curr_dims[0]));
  ratio_c = SubArray(hierarchy.ratio(l, curr_dims[1]));
  ratio_f = SubArray(hierarchy.ratio(l, curr_dims[2]));
  DeviceLauncher<DeviceType>::Execute(
      GpkRevKernel<D, 3, T, false, false, 1, DeviceType>(
          shape, shape_c, unprocessed_n, unprocessed_dims, curr_dims[0],
          curr_dims[1], curr_dims[2], ratio_r, ratio_c, ratio_f, dinput1,
          dcoarse, dcoeff_f, dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf,
          dcoeff_rc, dcoeff_rcf, 0, 0, 0,
          hierarchy.level_shape(l, curr_dims[0]),
          hierarchy.level_shape(l, curr_dims[1]),
          hierarchy.level_shape(l, curr_dims[2])),
      queue_idx);

  DIM D_reduced = D % 2 == 0 ? D - 1 : D - 2;
  for (DIM d = 3; d < D_reduced; d += 2) {
    // printf("reorder-reverse\n");
    // copy back to input2 for reordering again
    CopyND(dinput1, dinput2, queue_idx);

    // printf("reorder-restore %u-%uD\n", d+1, d+2);

    curr_dims[0] = D - (d + 1 + 1);
    curr_dims[1] = D - (d + 1);
    curr_dims[2] = D - 1;

    dinput2.project(curr_dims[0], curr_dims[1], curr_dims[2]);
    dinput1.project(curr_dims[0], curr_dims[1],
                    curr_dims[2]); // reuse input1 as temp output

    CalcCoefficientsPointers(hierarchy, curr_dims, l, dinput2, dcoarse,
                             dcoeff_f, dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf,
                             dcoeff_rc, dcoeff_rcf);
    ratio_r = SubArray(hierarchy.ratio(l, curr_dims[0]));
    ratio_c = SubArray(hierarchy.ratio(l, curr_dims[1]));
    ratio_f = SubArray(hierarchy.ratio(l, curr_dims[2]));

    unprocessed_idx += 2;
    unprocessed_dims = hierarchy.unprocessed(unprocessed_idx, unprocessed_n);
    DeviceLauncher<DeviceType>::Execute(
        GpkRevKernel<D, 3, T, false, false, 2, DeviceType>(
            shape, shape_c, unprocessed_n, unprocessed_dims, curr_dims[0],
            curr_dims[1], curr_dims[2], ratio_r, ratio_c, ratio_f, dinput1,
            dcoarse, dcoeff_f, dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf,
            dcoeff_rc, dcoeff_rcf, 0, 0, 0,
            hierarchy.level_shape(l, curr_dims[0]),
            hierarchy.level_shape(l, curr_dims[1]),
            hierarchy.level_shape(l, curr_dims[2])),
        queue_idx);
  }

  // printf("coeff-restore %u-%dD\n", D_reduced+1, D_reduced+2);
  curr_dims[0] = D - (D_reduced + 1 + 1);
  curr_dims[1] = D - (D_reduced + 1);
  curr_dims[2] = D - 1;

  dinput1.project(curr_dims[0], curr_dims[1], curr_dims[2]);
  doutput.project(curr_dims[0], curr_dims[1], curr_dims[2]);

  CalcCoefficientsPointers(hierarchy, curr_dims, l, dinput1, dcoarse, dcoeff_f,
                           dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf, dcoeff_rc,
                           dcoeff_rcf);
  ratio_r = SubArray(hierarchy.ratio(l, curr_dims[0]));
  ratio_c = SubArray(hierarchy.ratio(l, curr_dims[1]));
  ratio_f = SubArray(hierarchy.ratio(l, curr_dims[2]));

  if (D - D_reduced == 1) {
    // printf("coeff-restore %u-%dD\n", D_reduced+1, D_reduced+1);
    unprocessed_idx += 1;
    unprocessed_dims = hierarchy.unprocessed(unprocessed_idx, unprocessed_n);
    DeviceLauncher<DeviceType>::Execute(
        GpkRevKernel<D, 2, T, false, true, 2, DeviceType>(
            shape, shape_c, unprocessed_n, unprocessed_dims, curr_dims[0],
            curr_dims[1], curr_dims[2], ratio_r, ratio_c, ratio_f, doutput,
            dcoarse, dcoeff_f, dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf,
            dcoeff_rc, dcoeff_rcf, 0, 0, 0,
            hierarchy.level_shape(l, curr_dims[0]),
            hierarchy.level_shape(l, curr_dims[1]),
            hierarchy.level_shape(l, curr_dims[2])),
        queue_idx);
  } else { // D - D_reduced >= 2
    // printf("coeff-restore %u-%dD\n", D_reduced+1, D_reduced+2);
    unprocessed_idx += 2;
    unprocessed_dims = hierarchy.unprocessed(unprocessed_idx, unprocessed_n);
    DeviceLauncher<DeviceType>::Execute(
        GpkRevKernel<D, 3, T, false, true, 2, DeviceType>(
            shape, shape_c, unprocessed_n, unprocessed_dims, curr_dims[0],
            curr_dims[1], curr_dims[2], ratio_r, ratio_c, ratio_f, doutput,
            dcoarse, dcoeff_f, dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf,
            dcoeff_rc, dcoeff_rcf, 0, 0, 0,
            hierarchy.level_shape(l, curr_dims[0]),
            hierarchy.level_shape(l, curr_dims[1]),
            hierarchy.level_shape(l, curr_dims[2])),
        queue_idx);
  }

  if (multidim_refactoring_debug_print) { // debug
    PrintSubarray4D("After coeff restore", doutput);
  } // debug
}

} // namespace mgard_x

#endif