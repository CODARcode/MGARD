/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#include "../../Hierarchy.hpp"
#include "../../RuntimeX/RuntimeX.h"
// #include "SubArray.hpp"
// #include "DeviceAdapters/DeviceAdapterCuda.h"

// #include "DataRefactoring/Coefficient/GridProcessingKernel.h"
#include "../MultiDimension/Correction/LevelwiseProcessingKernel.hpp"
#include "Coefficient/CoefficientKernel.hpp"

namespace mgard_x {


template <DIM D, typename T, typename DeviceType>
void calc_coefficients_single(Hierarchy<D, T, DeviceType> &hierarchy,
                          SubArray<D, T, DeviceType> dinput,
                          SubArray<D, T, DeviceType> &doutput, SIZE curr_dim, SIZE l,
                          int queue_idx) {

  SIZE f = hierarchy.dofs[0][l];
  SIZE c = hierarchy.dofs[1][l];
  SIZE r = hierarchy.dofs[2][l];
  SIZE ff = hierarchy.dofs[0][l + 1];
  SIZE cc = hierarchy.dofs[1][l + 1];
  SIZE rr = hierarchy.dofs[2][l + 1];

  std::vector<SIZE> v_shape(D);
  std::vector<SIZE> coarse_shape(D);
  std::vector<SIZE> coeff_shape(D);
  for (DIM d = 0; d < D; d ++) {
    if (d < curr_dim) {
      v_shape[d] = hierarchy.dofs[d][l + 1];
    } else {
      v_shape[d] = hierarchy.dofs[d][l];
    }
  }

  for (DIM d = 0; d < D; d ++) {
    if (d == curr_dim) {
      coarse_shape[d] = hierarchy.dofs[d][l + 1];
      coeff_shape[d] = v_shape[d] - hierarchy.dofs[d][l + 1];
    } else {
      coarse_shape[d] = v_shape[d];
      coeff_shape[d] = v_shape[d];
    }
  }

  SubArray<D, T, DeviceType> dv = dinput;
  dv.resize(v_shape);

  SubArray<D, T, DeviceType> dcoarse = doutput;
  dcoarse.resize(coarse_shape);
  SubArray<D, T, DeviceType> dcoeff = doutput;
  dcoeff.offset(curr_dim, hierarchy.dofs[curr_dim][l + 1]);
  dcoeff.resize(coeff_shape);

  SingleDimensionCoefficient<D, T, DeviceType>().Execute(curr_dim, SubArray(hierarchy.ratio_array[0][l]), dv, dcoarse, dcoeff, queue_idx);
  DeviceRuntime<DeviceType>::SyncDevice();

  exit(0);
  // PrintSubarray("dv", dv);
  // PrintSubarray("doutput", doutput);

}


template <DIM D, typename T, typename DeviceType>
void decompose_single(Hierarchy<D, T, DeviceType> &hierarchy,
               SubArray<D, T, DeviceType> &v, SIZE l_target, int queue_idx) {
  std::vector<SIZE> workspace_shape(D);
  for (DIM d = 0; d < D; d++)
    workspace_shape[d] = hierarchy.dofs[d][0] + 2;
  std::reverse(workspace_shape.begin(), workspace_shape.end());
  Array<D, T, DeviceType> workspace(workspace_shape);
  SubArray w(workspace);

  for (int l = 0; l < l_target; ++l) {
    for (DIM curr_dim = 0; curr_dim < D; curr_dim++) {
      std::vector<SIZE> v_shape(D);
      for (DIM d = 0; d < D; d ++) {
        if (d < curr_dim) {
          v_shape[d] = hierarchy.dofs[d][l + 1];
        } else {
          v_shape[d] = hierarchy.dofs[d][l];
        }
      }

      Array<1, SIZE, DeviceType> v_shape_array({(SIZE)D});
      v_shape_array.loadData(v_shape.data());

      // PrintSubarray("v_shape_array", SubArray<1, SIZE, DeviceType>(v_shape_array));

      LwpkReo<D, T, COPY, DeviceType>().Execute(
          SubArray<1, SIZE, DeviceType>(v_shape_array, true), v, w,
          queue_idx);
      calc_coefficients_single(hierarchy, w, v, curr_dim, l, queue_idx);
    }
  }

}


template <DIM D, typename T, typename DeviceType>
void recompose_single(Hierarchy<D, T, DeviceType> &hierarchy,
               SubArray<D, T, DeviceType> &v, SIZE l_target, int queue_idx) {

}

}