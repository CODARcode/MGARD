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
      curr_dim, SubArray(hierarchy.dist(hierarchy.l_target-l, curr_dim)),
      SubArray(hierarchy.ratio(hierarchy.l_target-l, curr_dim)), coeff, correction,
      queue_idx);

  if (singledim_refactoring_debug_print) {
    PrintSubarray("SingleDimensionMassTrans", correction);
  }

  DIM curr_dim_r, curr_dim_c, curr_dim_f;
  if (curr_dim == D-1) {
    curr_dim_r = D-3, curr_dim_c = D-2, curr_dim_f = D-1;
    correction.project(curr_dim_r, curr_dim_c, curr_dim_f);
    Ipk1Reo<D, T, DeviceType>().Execute(
        curr_dim_r, curr_dim_c, curr_dim_f,
        SubArray(hierarchy.am(hierarchy.l_target-l-1, D-1)),
        SubArray(hierarchy.bm(hierarchy.l_target-l-1, D-1)),
        correction, queue_idx);
    if (singledim_refactoring_debug_print) {
      PrintSubarray("Ipk1Reo", correction);
    }

  } else if (curr_dim == D-2) {
    curr_dim_r = D-3, curr_dim_c = D-2, curr_dim_f = D-1;
    correction.project(curr_dim_r, curr_dim_c, curr_dim_f);
    Ipk2Reo<D, T, DeviceType>().Execute(
        curr_dim_r, curr_dim_c, curr_dim_f,
        SubArray(hierarchy.am(hierarchy.l_target-l-1, D-2)),
        SubArray(hierarchy.bm(hierarchy.l_target-l-1, D-2)),
        correction, queue_idx);
    if (singledim_refactoring_debug_print) {
      PrintSubarray("Ipk2Reo", correction);
    }
  } else if (curr_dim == D-3) {
    curr_dim_r = D-3, curr_dim_c = D-2, curr_dim_f = D-1;
    correction.project(curr_dim_r, curr_dim_c, curr_dim_f);
    Ipk3Reo<D, T, DeviceType>().Execute(
        curr_dim_r, curr_dim_c, curr_dim_f,
        SubArray(hierarchy.am(hierarchy.l_target-l-1, D-3)),
        SubArray(hierarchy.bm(hierarchy.l_target-l-1, D-3)),
        correction, queue_idx);
    if (singledim_refactoring_debug_print) {
      PrintSubarray("Ipk3Reo", correction);
    }
  } else {
    curr_dim_r = curr_dim, curr_dim_c = D-2, curr_dim_f = D-1;
    correction.project(curr_dim_r, curr_dim_c, curr_dim_f);
    Ipk3Reo<D, T, DeviceType>().Execute(
        curr_dim_r, curr_dim_c, curr_dim_f,
        SubArray(hierarchy.am(hierarchy.l_target-l-1, curr_dim)),
        SubArray(hierarchy.bm(hierarchy.l_target-l-1, curr_dim)),
        correction, queue_idx);
    if (singledim_refactoring_debug_print) {
      PrintSubarray("Ipk3Reo", correction);
    }
  }
}

} // namespace mgard_x

#endif