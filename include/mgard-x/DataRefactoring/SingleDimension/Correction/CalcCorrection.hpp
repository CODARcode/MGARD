/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../../../Hierarchy/Hierarchy.hpp"
#include "../../../RuntimeX/RuntimeX.h"

#include "../DataRefactoring.h"

#include "../../MultiDimension/Correction/IterativeProcessingKernel.hpp"
#include "../../MultiDimension/Correction/LevelwiseProcessingKernel.hpp"

#include "MassTransKernel.hpp"

#ifndef MGARD_X_DATA_REFACTORING_CALC_CORRECTION
#define MGARD_X_DATA_REFACTORING_CALC_CORRECTION

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
void CalcCorrection(Hierarchy<D, T, DeviceType> &hierarchy,
                    SubArray<D, T, DeviceType> &coeff,
                    SubArray<D, T, DeviceType> &correction, SIZE curr_dim,
                    SIZE l, int queue_idx) {

  SingleDimensionMassTrans<D, T, DeviceType>().Execute(
      curr_dim, SubArray(hierarchy.dist_array[curr_dim][l]),
      SubArray(hierarchy.ratio_array[curr_dim][l]), coeff, correction,
      queue_idx);

  if (singledim_refactoring_debug_print) {
    PrintSubarray("SingleDimensionMassTrans", correction);
  }

  DIM curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;
  if (curr_dim == 0) {
    correction.project(curr_dim_f, curr_dim_c, curr_dim_r);
    Ipk1Reo<D, T, DeviceType>().Execute(
        curr_dim_r, curr_dim_c, curr_dim_f,
        SubArray(hierarchy.am_array[curr_dim_f][l + 1]),
        SubArray(hierarchy.bm_array[curr_dim_f][l + 1]), correction, queue_idx);
    if (singledim_refactoring_debug_print) {
      PrintSubarray("Ipk1Reo", correction);
    }

  } else if (curr_dim == 1) {
    curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;
    correction.project(curr_dim_f, curr_dim_c, curr_dim_r);
    Ipk2Reo<D, T, DeviceType>().Execute(
        curr_dim_r, curr_dim_c, curr_dim_f,
        SubArray(hierarchy.am_array[curr_dim_c][l + 1]),
        SubArray(hierarchy.bm_array[curr_dim_c][l + 1]), correction, queue_idx);
    if (singledim_refactoring_debug_print) {
      PrintSubarray("Ipk2Reo", correction);
    }
  } else if (curr_dim == 2) {
    curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;
    correction.project(curr_dim_f, curr_dim_c, curr_dim_r);
    Ipk3Reo<D, T, DeviceType>().Execute(
        curr_dim_r, curr_dim_c, curr_dim_f,
        SubArray(hierarchy.am_array[curr_dim_r][l + 1]),
        SubArray(hierarchy.bm_array[curr_dim_r][l + 1]), correction, queue_idx);
    if (singledim_refactoring_debug_print) {
      PrintSubarray("Ipk3Reo", correction);
    }
  } else {
    curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = curr_dim;
    correction.project(curr_dim_f, curr_dim_c, curr_dim_r);
    Ipk3Reo<D, T, DeviceType>().Execute(
        curr_dim_r, curr_dim_c, curr_dim_f,
        SubArray(hierarchy.am_array[curr_dim_r][l + 1]),
        SubArray(hierarchy.bm_array[curr_dim_r][l + 1]), correction, queue_idx);
    if (singledim_refactoring_debug_print) {
      PrintSubarray("Ipk3Reo", correction);
    }
  }
}

} // namespace mgard_x

#endif