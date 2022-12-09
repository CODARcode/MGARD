/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_DATA_REFACTORING
#define MGARD_X_DATA_REFACTORING

// #include "Common.h"
#include "../../Hierarchy/Hierarchy.h"
#include "../../RuntimeX/RuntimeXPublic.h"

namespace mgard_x {

static bool multidim_refactoring_store = false;
static bool multidim_refactoring_verify = false;
static bool multidim_refactoring_debug_print = false;

template <DIM D, typename T, typename DeviceType>
void CalcCoefficients3D(Hierarchy<D, T, DeviceType> &hierarchy,
                        SubArray<D, T, DeviceType> dinput,
                        SubArray<D, T, DeviceType> &doutput, SIZE l,
                        int queue_idx);

template <DIM D, typename T, typename DeviceType>
void CalcCoefficients3DWithErrorCollection(
    Hierarchy<D, T, DeviceType> &hierarchy, SubArray<D, T, DeviceType> dinput,
    SubArray<D, T, DeviceType> &doutput, SIZE l,
    SubArray<D + 1, T, DeviceType> max_coeffcient,
    SubArray<D + 1, T, DeviceType> max_coeffcient_finer, int queue_idx);

template <DIM D, typename T, typename DeviceType>
void CoefficientsRestore3D(Hierarchy<D, T, DeviceType> &hierarchy,
                           SubArray<D, T, DeviceType> dinput,
                           SubArray<D, T, DeviceType> &doutput, SIZE l,
                           int queue_idx);

template <DIM D, typename T, typename DeviceType>
void CalcCorrection3D(Hierarchy<D, T, DeviceType> &hierarchy,
                      SubArray<D, T, DeviceType> dcoeff,
                      SubArray<D, T, DeviceType> &dcorrection, SIZE l,
                      int queue_idx);

template <DIM D, typename T, typename DeviceType>
void CalcCoefficientsND(Hierarchy<D, T, DeviceType> &hierarchy,
                        SubArray<D, T, DeviceType> dinput1,
                        SubArray<D, T, DeviceType> dinput2,
                        SubArray<D, T, DeviceType> &doutput, SIZE l,
                        int queue_idx);

template <DIM D, typename T, typename DeviceType>
void CoefficientsRestoreND(Hierarchy<D, T, DeviceType> &hierarchy,
                           SubArray<D, T, DeviceType> dinput1,
                           SubArray<D, T, DeviceType> dinput2,
                           SubArray<D, T, DeviceType> &doutput, SIZE l,
                           int queue_idx);

template <DIM D, typename T, typename DeviceType>
void CalcCorrectionND(Hierarchy<D, T, DeviceType> &hierarchy,
                      SubArray<D, T, DeviceType> dcoeff,
                      SubArray<D, T, DeviceType> &dcorrection, SIZE l,
                      int queue_idx);

template <DIM D, typename T, typename DeviceType>
void CopyND(SubArray<D, T, DeviceType> dinput,
            SubArray<D, T, DeviceType> doutput, int queue_idx);

template <DIM D, typename T, typename DeviceType>
void AddND(SubArray<D, T, DeviceType> dinput,
           SubArray<D, T, DeviceType> doutput, int queue_idx);

template <DIM D, typename T, typename DeviceType>
void SubtractND(SubArray<D, T, DeviceType> dinput,
                SubArray<D, T, DeviceType> doutput, int queue_idx);

template <DIM D, typename T, typename DeviceType>
void decompose(Hierarchy<D, T, DeviceType> &hierarchy,
               SubArray<D, T, DeviceType> &v, SubArray<D, T, DeviceType> w,
               SubArray<D, T, DeviceType> b, int start_level, int stop_level,
               int queue_idx);

template <DIM D, typename T, typename DeviceType>
void recompose(Hierarchy<D, T, DeviceType> &hierarchy,
               SubArray<D, T, DeviceType> &v, SubArray<D, T, DeviceType> w,
               SubArray<D, T, DeviceType> b, int start_level, int stop_level,
               int queue_idx);

template <DIM D, typename T, typename DeviceType>
void decompose_adaptive_resolution(
    Hierarchy<D, T, DeviceType> &hierarchy, SubArray<D, T, DeviceType> &v,
    SIZE l_target, SubArray<1, T, DeviceType> level_max,
    SubArray<D + 1, T, DeviceType> *max_coefficient, int queue_idx);

template <DIM D, typename T, typename DeviceType>
Array<D, T, DeviceType> recompose_adaptive_resolution(
    Hierarchy<D, T, DeviceType> &hierarchy, SubArray<D, T, DeviceType> &v,
    SIZE l_target, T iso_value, T error_target,
    SubArray<1, T, DeviceType> level_max,
    SubArray<D + 1, T, DeviceType> *max_abs_coefficient,
    SubArray<D, SIZE, DeviceType> *refinement_flag, int queue_idx);

template <DIM D, typename T, typename DeviceType>
void LevelMax(SIZE l_target, SubArray<1, SIZE, DeviceType> ranges,
              SubArray<D, T, DeviceType> v,
              SubArray<1, T, DeviceType> level_max, int queue_idx);

template <DIM D, typename T, typename DeviceType>
void EarlyFeatureDetector(SubArray<D, T, DeviceType> v, T current_error,
                          T iso_value,
                          SubArray<D, SIZE, DeviceType> feature_flag_coarser,
                          SubArray<D, SIZE, DeviceType> feature_flag,
                          int queue_idx);

template <DIM D, typename T, typename DeviceType>
void AccuracyGuard(SIZE total_level, SIZE current_level,
                   SubArray<1, T, DeviceType> error_impact_budget,
                   SubArray<1, T, DeviceType> previous_error_impact,
                   SubArray<1, T, DeviceType> current_error_impact,
                   SubArray<D + 1, T, DeviceType> max_abs_coefficient,
                   SubArray<D, SIZE, DeviceType> refinement_flag,
                   int queue_idx);

template <DIM D, typename T, typename DeviceType>
void CoefficientRetriever(Hierarchy<D, T, DeviceType> &hierarchy,
                          SubArray<D, T, DeviceType> dinput,
                          SubArray<D, SIZE, DeviceType> refinement_flag,
                          SubArray<D, T, DeviceType> &doutput, SIZE l,
                          int queue_idx);

} // namespace mgard_x

#endif