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
            SubArray<D, T, DeviceType> &doutput, int queue_idx);

template <DIM D, typename T, typename DeviceType>
void AddND(SubArray<D, T, DeviceType> dinput,
           SubArray<D, T, DeviceType> &doutput, int queue_idx);

template <DIM D, typename T, typename DeviceType>
void SubtractND(SubArray<D, T, DeviceType> dinput,
                SubArray<D, T, DeviceType> &doutput, int queue_idx);

template <DIM D, typename T, typename DeviceType>
void decompose(Hierarchy<D, T, DeviceType> &hierarchy,
               SubArray<D, T, DeviceType> &v, SIZE l_target, int queue_idx);

template <DIM D, typename T, typename DeviceType>
void recompose(Hierarchy<D, T, DeviceType> &hierarchy,
               SubArray<D, T, DeviceType> &v, SIZE l_target, int queue_idx);

} // namespace mgard_x

#endif