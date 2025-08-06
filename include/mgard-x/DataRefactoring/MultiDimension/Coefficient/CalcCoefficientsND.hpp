/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../../../Hierarchy/Hierarchy.hpp"
#include "../../../RuntimeX/RuntimeX.h"

#include "../DataRefactoring.h"

#include "CalcCoefficientsPointers.hpp"
#include "GridProcessingKernel.hpp"

#ifndef MGARD_X_DATA_REFACTORING_CALC_COEFFICIENTS_ND
#define MGARD_X_DATA_REFACTORING_CALC_COEFFICIENTS_ND

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
void CalcCoefficientsND(Hierarchy<D, T, DeviceType> &hierarchy,
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
    prefix += std::to_string(hierarchy.shape[d]) + "_";
  // printf("interpolate 1-3D\n");

  SubArray<D, T, DeviceType> dcoarse, dcoeff_f, dcoeff_c, dcoeff_r, dcoeff_cf,
      dcoeff_rf, dcoeff_rc, dcoeff_rcf;

  DIM curr_dims[3];

  int unprocessed_idx = 0;
  curr_dims[0] = 0;
  curr_dims[1] = 1;
  curr_dims[2] = 2;
  dinput1.project(curr_dims[0], curr_dims[1], curr_dims[2]);
  doutput.project(curr_dims[0], curr_dims[1], curr_dims[2]);

  CalcCoefficientsPointers(hierarchy, curr_dims, l, doutput, dcoarse, dcoeff_f,
                           dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf, dcoeff_rc,
                           dcoeff_rcf);

  GpkReo<D, 3, T, true, false, 1, DeviceType>().Execute(
      SubArray<1, SIZE, DeviceType>(hierarchy.shapes[l], true),
      SubArray<1, SIZE, DeviceType>(hierarchy.shapes[l + 1], true),
      hierarchy.unprocessed_n[unprocessed_idx],
      SubArray(hierarchy.unprocessed_dims[unprocessed_idx]), curr_dims[2],
      curr_dims[1], curr_dims[0],
      SubArray(hierarchy.ratio_array[curr_dims[2]][l]),
      SubArray(hierarchy.ratio_array[curr_dims[1]][l]),
      SubArray(hierarchy.ratio_array[curr_dims[0]][l]), dinput1, dcoarse,
      dcoeff_f, dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf, dcoeff_rc, dcoeff_rcf,
      queue_idx);

  for (DIM d = 3; d < D; d += 2) {
    // copy back to input1 for interpolation again
    // LwpkReo<D, T, COPY, DeviceType>().Execute(doutput, dinput1, queue_idx);
    CopyND(doutput, dinput1, queue_idx);

    // printf("interpolate %u-%uD\n", d+1, d+2);
    curr_dims[0] = 0;
    curr_dims[1] = d;
    curr_dims[2] = d + 1;
    dinput1.project(curr_dims[0], curr_dims[1], curr_dims[2]);
    doutput.project(curr_dims[0], curr_dims[1], curr_dims[2]);
    CalcCoefficientsPointers(hierarchy, curr_dims, l, doutput, dcoarse,
                             dcoeff_f, dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf,
                             dcoeff_rc, dcoeff_rcf);

    if (D - d == 1) {
      unprocessed_idx += 1;

      GpkReo<D, 2, T, true, false, 2, DeviceType>().Execute(
          SubArray<1, SIZE, DeviceType>(hierarchy.shapes[l], true),
          SubArray<1, SIZE, DeviceType>(hierarchy.shapes[l + 1], true),
          hierarchy.unprocessed_n[unprocessed_idx],
          SubArray(hierarchy.unprocessed_dims[unprocessed_idx]), curr_dims[2],
          curr_dims[1], curr_dims[0],
          SubArray(hierarchy.ratio_array[curr_dims[2]][l]),
          SubArray(hierarchy.ratio_array[curr_dims[1]][l]),
          SubArray(hierarchy.ratio_array[curr_dims[0]][l]), dinput1, dcoarse,
          dcoeff_f, dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf, dcoeff_rc,
          dcoeff_rcf, queue_idx);

    } else { // D - d >= 2
      unprocessed_idx += 2;
      GpkReo<D, 3, T, true, false, 2, DeviceType>().Execute(
          SubArray<1, SIZE, DeviceType>(hierarchy.shapes[l], true),
          SubArray<1, SIZE, DeviceType>(hierarchy.shapes[l + 1], true),
          hierarchy.unprocessed_n[unprocessed_idx],
          SubArray(hierarchy.unprocessed_dims[unprocessed_idx]),
          // unprocessed_dims_subarray,
          curr_dims[2], curr_dims[1], curr_dims[0],
          // ratio_r, ratio_c, ratio_f,
          SubArray(hierarchy.ratio_array[curr_dims[2]][l]),
          SubArray(hierarchy.ratio_array[curr_dims[1]][l]),
          SubArray(hierarchy.ratio_array[curr_dims[0]][l]), dinput1, dcoarse,
          dcoeff_f, dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf, dcoeff_rc,
          dcoeff_rcf, queue_idx);
    }
  }

  if (multidim_refactoring_debug_print) { // debug
    PrintSubarray4D("after interpolation", doutput);
  } // debug

  unprocessed_idx = 0;
  // printf("reorder 1-3D\n");
  curr_dims[0] = 0;
  curr_dims[1] = 1;
  curr_dims[2] = 2;
  dinput2.project(curr_dims[0], curr_dims[1], curr_dims[2]);
  dinput1.project(curr_dims[0], curr_dims[1],
                  curr_dims[2]); // reuse input1 as temp output

  CalcCoefficientsPointers(hierarchy, curr_dims, l, dinput1, dcoarse, dcoeff_f,
                           dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf, dcoeff_rc,
                           dcoeff_rcf);

  GpkReo<D, 3, T, false, false, 1, DeviceType>().Execute(
      SubArray<1, SIZE, DeviceType>(hierarchy.shapes[l], true),
      SubArray<1, SIZE, DeviceType>(hierarchy.shapes[l + 1], true),
      hierarchy.unprocessed_n[unprocessed_idx],
      SubArray(hierarchy.unprocessed_dims[unprocessed_idx]), curr_dims[2],
      curr_dims[1], curr_dims[0],
      SubArray(hierarchy.ratio_array[curr_dims[2]][l]),
      SubArray(hierarchy.ratio_array[curr_dims[1]][l]),
      SubArray(hierarchy.ratio_array[curr_dims[0]][l]), dinput2, dcoarse,
      dcoeff_f, dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf, dcoeff_rc, dcoeff_rcf,
      queue_idx);

  DIM D_reduced = D % 2 == 0 ? D - 1 : D - 2;
  for (DIM d = 3; d < D_reduced; d += 2) {
    // copy back to input2 for reordering again

    // LwpkReo<D, T, COPY, DeviceType>().Execute(dinput1, dinput2, queue_idx);
    CopyND(dinput1, dinput2, queue_idx);

    unprocessed_idx += 2;
    // printf("reorder %u-%uD\n", d+1, d+2);
    curr_dims[0] = 0;
    curr_dims[1] = d;
    curr_dims[2] = d + 1;
    dinput2.project(curr_dims[0], curr_dims[1], curr_dims[2]);
    dinput1.project(curr_dims[0], curr_dims[1],
                    curr_dims[2]); // reuse input1 as temp output

    CalcCoefficientsPointers(hierarchy, curr_dims, l, dinput1, dcoarse,
                             dcoeff_f, dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf,
                             dcoeff_rc, dcoeff_rcf);

    GpkReo<D, 3, T, false, false, 2, DeviceType>().Execute(
        SubArray<1, SIZE, DeviceType>(hierarchy.shapes[l], true),
        SubArray<1, SIZE, DeviceType>(hierarchy.shapes[l + 1], true),
        hierarchy.unprocessed_n[unprocessed_idx],
        SubArray(hierarchy.unprocessed_dims[unprocessed_idx]), curr_dims[2],
        curr_dims[1], curr_dims[0],
        SubArray(hierarchy.ratio_array[curr_dims[2]][l]),
        SubArray(hierarchy.ratio_array[curr_dims[1]][l]),
        SubArray(hierarchy.ratio_array[curr_dims[0]][l]), dinput2, dcoarse,
        dcoeff_f, dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf, dcoeff_rc,
        dcoeff_rcf, queue_idx);
  }

  // printf("calc coeff %u-%dD\n", D_reduced+1, D_reduced+2);
  curr_dims[0] = 0;
  curr_dims[1] = D_reduced;
  curr_dims[2] = D_reduced + 1;
  dinput1.project(curr_dims[0], curr_dims[1], curr_dims[2]);
  doutput.project(curr_dims[0], curr_dims[1],
                  curr_dims[2]); // reuse input1 as temp output
  CalcCoefficientsPointers(hierarchy, curr_dims, l, doutput, dcoarse, dcoeff_f,
                           dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf, dcoeff_rc,
                           dcoeff_rcf);
  if (D - D_reduced == 1) {
    unprocessed_idx += 1;
    GpkReo<D, 2, T, false, true, 2, DeviceType>().Execute(
        SubArray<1, SIZE, DeviceType>(hierarchy.shapes[l], true),
        SubArray<1, SIZE, DeviceType>(hierarchy.shapes[l + 1], true),
        hierarchy.unprocessed_n[unprocessed_idx],
        SubArray(hierarchy.unprocessed_dims[unprocessed_idx]), curr_dims[2],
        curr_dims[1], curr_dims[0],
        SubArray(hierarchy.ratio_array[curr_dims[2]][l]),
        SubArray(hierarchy.ratio_array[curr_dims[1]][l]),
        SubArray(hierarchy.ratio_array[curr_dims[0]][l]), dinput1, dcoarse,
        dcoeff_f, dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf, dcoeff_rc,
        dcoeff_rcf, queue_idx);

  } else { // D-D_reduced == 2
    unprocessed_idx += 2;

    GpkReo<D, 3, T, false, true, 2, DeviceType>().Execute(
        SubArray<1, SIZE, DeviceType>(hierarchy.shapes[l], true),
        SubArray<1, SIZE, DeviceType>(hierarchy.shapes[l + 1], true),
        hierarchy.unprocessed_n[unprocessed_idx],
        SubArray(hierarchy.unprocessed_dims[unprocessed_idx]), curr_dims[2],
        curr_dims[1], curr_dims[0],
        SubArray(hierarchy.ratio_array[curr_dims[2]][l]),
        SubArray(hierarchy.ratio_array[curr_dims[1]][l]),
        SubArray(hierarchy.ratio_array[curr_dims[0]][l]), dinput1, dcoarse,
        dcoeff_f, dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf, dcoeff_rc,
        dcoeff_rcf, queue_idx);
  }

  if (multidim_refactoring_debug_print) { // debug
    PrintSubarray4D("after calc coeff", doutput);
  } // debug
}

} // namespace mgard_x

#endif