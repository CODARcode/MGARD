/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_SINGLE_DIMENSION_DATA_REFACTORING
#define MGARD_X_SINGLE_DIMENSION_DATA_REFACTORING

#include "../../Hierarchy/Hierarchy.h"
#include "../../RuntimeX/RuntimeXPublic.h"

namespace mgard_x {

static bool singledim_refactoring_store = false;
static bool singledim_refactoring_verify = false;
static bool singledim_refactoring_debug_print = false;

template <DIM D, typename T, typename DeviceType>
void CalcCoefficients(DIM current_dim, SubArray<1, T, DeviceType> ratio,
                      SubArray<D, T, DeviceType> v,
                      SubArray<D, T, DeviceType> coarse,
                      SubArray<D, T, DeviceType> coeff, int queue_idx);

template <DIM D, typename T, typename DeviceType>
void CoefficientsRestore(DIM current_dim, SubArray<1, T, DeviceType> ratio,
                         SubArray<D, T, DeviceType> v,
                         SubArray<D, T, DeviceType> coarse,
                         SubArray<D, T, DeviceType> coeff, int queue_idx);

template <DIM D, typename T, typename DeviceType>
void CalcCorrection(Hierarchy<D, T, DeviceType> &hierarchy,
                    SubArray<D, T, DeviceType> &coeff,
                    SubArray<D, T, DeviceType> &correction, SIZE curr_dim,
                    SIZE l, int queue_idx);

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
void decompose_single(Hierarchy<D, T, DeviceType> &hierarchy,
                      SubArray<D, T, DeviceType> &v, int stop_level,
                      int queue_idx);

template <DIM D, typename T, typename DeviceType>
void recompose_single(Hierarchy<D, T, DeviceType> &hierarchy,
                      SubArray<D, T, DeviceType> &v, int stop_level,
                      int queue_idx);

} // namespace mgard_x

#endif