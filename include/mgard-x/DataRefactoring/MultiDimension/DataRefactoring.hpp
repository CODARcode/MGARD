/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../../Hierarchy/Hierarchy.hpp"
#include "../../RuntimeX/RuntimeX.h"

#include "DataRefactoring.h"

#include <iostream>

#ifndef MGARD_X_DATA_REFACTORING_HPP
#define MGARD_X_DATA_REFACTORING_HPP

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
void decompose(Hierarchy<D, T, DeviceType> &hierarchy,
               SubArray<D, T, DeviceType> &v, SIZE l_target, int queue_idx) {

  std::string prefix = "decomp_";
  if (sizeof(T) == sizeof(double))
    prefix += "d_";
  if (sizeof(T) == sizeof(float))
    prefix += "f_";
  for (int d = 0; d < D; d++)
    prefix += std::to_string(hierarchy.shape[d]) + "_";
  // std::cout << prefix << std::endl;

  std::vector<SIZE> workspace_shape(D);
  for (DIM d = 0; d < D; d++)
    workspace_shape[d] = hierarchy.dofs[d][0] + 2;
  std::reverse(workspace_shape.begin(), workspace_shape.end());
  Array<D, T, DeviceType> workspace(workspace_shape);
  // workspace.memset(0); can cause large overhead in HIP
  SubArray w(workspace);

  SubArray<D, T, DeviceType> v_fine = v;
  SubArray<D, T, DeviceType> w_fine = w;
  SubArray<D, T, DeviceType> v_coeff = v;
  SubArray<D, T, DeviceType> w_correction = w;
  SubArray<D, T, DeviceType> v_coarse = v;

  if constexpr (D <= 3) {
    for (int l = 0; l < l_target; ++l) {
      if (multidim_refactoring_debug_print) {
        PrintSubarray("input v", v);
      }

      v_fine.resize(hierarchy.shapes_vec[l]);
      w_fine.resize(hierarchy.shapes_vec[l]);
      CopyND(v_fine, w_fine, queue_idx);

      v_coeff.resize(hierarchy.shapes_vec[l]);
      CalcCoefficients3D(hierarchy, w_fine, v_coeff, l, queue_idx);

      w_correction.resize(hierarchy.shapes_vec[l]);
      CalcCorrection3D(hierarchy, v_coeff, w_correction, l, queue_idx);

      w_correction.resize(hierarchy.shapes_vec[l + 1]);
      v_coarse.resize(hierarchy.shapes_vec[l + 1]);
      AddND(w_correction, v_coarse, queue_idx);
      if (multidim_refactoring_debug_print) {
        PrintSubarray("after add", v);
      }
    } // end of loop

    if (multidim_refactoring_debug_print) {
      PrintSubarray("output of decomposition", v);
    }
  }

  if constexpr (D > 3) {
    Array<D, T, DeviceType> workspace2(workspace_shape);
    SubArray b(workspace2);
    SubArray<D, T, DeviceType> b_fine = b;
    for (int l = 0; l < l_target; ++l) {
      if (multidim_refactoring_debug_print) { // debug
        PrintSubarray4D("before coeff", v);
      }

      v_fine.resize(hierarchy.shapes_vec[l]);
      w_fine.resize(hierarchy.shapes_vec[l]);
      CopyND(v_fine, w_fine, queue_idx);

      v_fine.resize(hierarchy.shapes_vec[l]);
      b_fine.resize(hierarchy.shapes_vec[l]);
      CopyND(v_fine, b_fine, queue_idx);

      v_coeff.resize(hierarchy.shapes_vec[l]);
      CalcCoefficientsND(hierarchy, w_fine, b_fine, v_coeff, l, queue_idx);

      if (multidim_refactoring_debug_print) { // debug
        PrintSubarray4D(format("after coeff[%d]", l), v_coeff);
      } // debug

      w_correction.resize(hierarchy.shapes_vec[l]);
      CalcCorrectionND(hierarchy, v_coeff, w_correction, l, queue_idx);

      w_correction.resize(hierarchy.shapes_vec[l + 1]);
      v_coarse.resize(hierarchy.shapes_vec[l + 1]);
      AddND(w_correction, v_coarse, queue_idx);
      if (multidim_refactoring_debug_print) { // debug
        PrintSubarray4D(format("after apply correction[%d]", l), v);
      } // debug
    }
  }
  DeviceRuntime<DeviceType>::SyncDevice();
}

template <DIM D, typename T, typename DeviceType>
void recompose(Hierarchy<D, T, DeviceType> &hierarchy,
               SubArray<D, T, DeviceType> &v, SIZE l_target, int queue_idx) {

  std::vector<SIZE> workspace_shape(D);
  for (DIM d = 0; d < D; d++)
    workspace_shape[d] = hierarchy.dofs[d][0] + 2;
  std::reverse(workspace_shape.begin(), workspace_shape.end());
  Array<D, T, DeviceType> workspace(workspace_shape);
  // workspace.memset(0); // can cause large overhead in HIP
  SubArray w(workspace);

  SubArray<D, T, DeviceType> v_fine = v;
  SubArray<D, T, DeviceType> w_fine = w;
  SubArray<D, T, DeviceType> v_coeff = v;
  SubArray<D, T, DeviceType> w_correction = w;
  SubArray<D, T, DeviceType> v_coarse = v;

  if constexpr (D <= 3) {
    if (multidim_refactoring_debug_print) {
      PrintSubarray("input of recomposition", v);
    }
    std::string prefix = "recomp_";
    if (sizeof(T) == sizeof(double))
      prefix += "d_";
    if (sizeof(T) == sizeof(float))
      prefix += "f_";
    for (int d = 0; d < D; d++)
      prefix += std::to_string(hierarchy.shape[d]) + "_";
    // std::cout << prefix << std::endl;

    for (int l = l_target - 1; l >= 0; l--) {
      v_coeff.resize(hierarchy.shapes_vec[l]);
      w_correction.resize(hierarchy.shapes_vec[l]);
      CalcCorrection3D(hierarchy, v_coeff, w_correction, l, queue_idx);

      w_correction.resize(hierarchy.shapes_vec[l + 1]);
      v_coarse.resize(hierarchy.shapes_vec[l + 1]);
      SubtractND(w_correction, v_coarse, queue_idx);

      v_coeff.resize(hierarchy.shapes_vec[l]);
      w_fine.resize(hierarchy.shapes_vec[l]);
      CoefficientsRestore3D(hierarchy, v_coeff, w_fine, l, queue_idx);

      v_fine.resize(hierarchy.shapes_vec[l]);
      CopyND(w_fine, v_fine, queue_idx);
      if (multidim_refactoring_debug_print) {
        PrintSubarray("output of recomposition", v);
      }
    }
  }
  if constexpr (D > 3) {
    Array<D, T, DeviceType> workspace2(workspace_shape);
    SubArray b(workspace2);
    SubArray<D, T, DeviceType> b_fine = b;
    for (int l = l_target - 1; l >= 0; l--) {

      if (multidim_refactoring_debug_print) { // debug
        PrintSubarray4D(format("before corection[%d]", l), v);
      }

      int curr_dim_r, curr_dim_c, curr_dim_f;
      int lddv1, lddv2;
      int lddw1, lddw2;
      int lddb1, lddb2;

      if (multidim_refactoring_debug_print) { // debug
        PrintSubarray4D(format("before subtract correction[%d]", l), v);
      } // deb

      v_coeff.resize(hierarchy.shapes_vec[l]);
      w_correction.resize(hierarchy.shapes_vec[l]);
      CalcCorrectionND(hierarchy, v_coeff, w_correction, l, queue_idx);

      w_correction.resize(hierarchy.shapes_vec[l + 1]);
      v_coarse.resize(hierarchy.shapes_vec[l + 1]);
      SubtractND(w_correction, v_coarse, queue_idx);

      if (multidim_refactoring_debug_print) { // debug
        PrintSubarray4D(format("after subtract correction[%d]", l), v);
      } // deb

      v_coeff.resize(hierarchy.shapes_vec[l]);
      w_fine.resize(hierarchy.shapes_vec[l]);
      b_fine.resize(hierarchy.shapes_vec[l]);
      CopyND(v_coeff, b_fine, queue_idx);
      CopyND(v_coeff, w_fine, queue_idx);
      v_fine.resize(hierarchy.shapes_vec[l]);
      CoefficientsRestoreND(hierarchy, w_fine, b_fine, v_fine, l, queue_idx);
    } // loop levels

    if (multidim_refactoring_debug_print) { // debug
      std::vector<SIZE> shape(hierarchy.D_padded);
      PrintSubarray4D(format("final output"), v);
    } // deb
  }   // D > 3
  DeviceRuntime<DeviceType>::SyncDevice();
}

} // namespace mgard_x

#endif