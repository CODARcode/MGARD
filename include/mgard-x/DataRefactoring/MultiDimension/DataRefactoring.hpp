/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../../Hierarchy/Hierarchy.h"
#include "../../RuntimeX/RuntimeX.h"

#include "DataRefactoring.h"

#include <iostream>

#ifndef MGARD_X_MULTI_DIMENSION_DATA_REFACTORING_HPP
#define MGARD_X_MULTI_DIMENSION_DATA_REFACTORING_HPP

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
void decompose(Hierarchy<D, T, DeviceType> &hierarchy,
               SubArray<D, T, DeviceType> &v, SubArray<D, T, DeviceType> w,
               SubArray<D, T, DeviceType> b, int stop_level, int queue_idx) {

  if (stop_level < 0 || stop_level > hierarchy.l_target()) {
    std::cout << log::log_err << "decompose: stop_level out of bound.\n";
    exit(-1);
  }

  std::string prefix = "decomp_";
  if (sizeof(T) == sizeof(double))
    prefix += "d_";
  if (sizeof(T) == sizeof(float))
    prefix += "f_";
  for (int d = 0; d < D; d++)
    prefix +=
        std::to_string(hierarchy.level_shape(hierarchy.l_target(), d)) + "_";

  Array<D, T, DeviceType> workspace;
  bool shape_pass = true;
  if (w.data() != nullptr) {
    for (DIM d = 0; d < D; d++) {
      if (w.shape(d) < hierarchy.level_shape(hierarchy.l_target())[d] + 2) {
        shape_pass = false;
      }
    }
  }
  if (w.data() == nullptr || !shape_pass) {
    log::info("decompose: allocating workspace as it is not pre-allocated.");
    std::vector<SIZE> workspace_shape =
        hierarchy.level_shape(hierarchy.l_target());
    for (DIM d = 0; d < D; d++)
      workspace_shape[d] += 2;
    workspace = Array<D, T, DeviceType>(workspace_shape);
    w = SubArray(workspace);
  }

  // Array<D, T, DeviceType> workspace(workspace_shape);
  // workspace.memset(0); can cause large overhead in HIP
  // SubArray w(workspace);

  SubArray<D, T, DeviceType> v_fine = v;
  SubArray<D, T, DeviceType> w_fine = w;
  SubArray<D, T, DeviceType> v_coeff = v;
  SubArray<D, T, DeviceType> w_correction = w;
  SubArray<D, T, DeviceType> v_coarse = v;

  if constexpr (D <= 3) {
    for (int l = hierarchy.l_target(); l > stop_level; l--) {
      if (multidim_refactoring_debug_print) {
        PrintSubarray("input v", v);
      }

      v_fine.resize(hierarchy.level_shape(l));
      w_fine.resize(hierarchy.level_shape(l));
      CopyND(v_fine, w_fine, queue_idx);

      v_coeff.resize(hierarchy.level_shape(l));
      CalcCoefficients3D(hierarchy, w_fine, v_coeff, l, queue_idx);

      w_correction.resize(hierarchy.level_shape(l));
      CalcCorrection3D(hierarchy, v_coeff, w_correction, l, queue_idx);

      w_correction.resize(hierarchy.level_shape(l - 1));
      v_coarse.resize(hierarchy.level_shape(l - 1));
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
    Array<D, T, DeviceType> workspace2;
    bool shape_pass = true;
    if (b.data() != nullptr) {
      for (DIM d = 0; d < D; d++) {
        if (b.shape(d) < hierarchy.level_shape(hierarchy.l_target())[d] + 2) {
          shape_pass = false;
        }
      }
    }
    if (b.data() == nullptr || !shape_pass) {
      log::info("decompose: allocating workspace as it is not pre-allocated.");
      std::vector<SIZE> workspace_shape =
          hierarchy.level_shape(hierarchy.l_target());
      for (DIM d = 0; d < D; d++)
        workspace_shape[d] += 2;
      workspace2 = Array<D, T, DeviceType>(workspace_shape);
      b = SubArray(workspace2);
    }
    // Array<D, T, DeviceType> workspace2(workspace_shape);
    // SubArray b(workspace2);
    SubArray<D, T, DeviceType> b_fine = b;
    for (int l = hierarchy.l_target(); l > stop_level; l--) {
      if (multidim_refactoring_debug_print) { // debug
        PrintSubarray4D("before coeff", v);
      }

      v_fine.resize(hierarchy.level_shape(l));
      w_fine.resize(hierarchy.level_shape(l));
      CopyND(v_fine, w_fine, queue_idx);

      v_fine.resize(hierarchy.level_shape(l));
      b_fine.resize(hierarchy.level_shape(l));
      CopyND(v_fine, b_fine, queue_idx);

      v_coeff.resize(hierarchy.level_shape(l));
      CalcCoefficientsND(hierarchy, w_fine, b_fine, v_coeff, l, queue_idx);

      if (multidim_refactoring_debug_print) { // debug
        PrintSubarray4D(format("after coeff[%d]", l), v_coeff);
      } // debug

      w_correction.resize(hierarchy.level_shape(l));
      CalcCorrectionND(hierarchy, v_coeff, w_correction, l, queue_idx);

      w_correction.resize(hierarchy.level_shape(l - 1));
      v_coarse.resize(hierarchy.level_shape(l - 1));
      AddND(w_correction, v_coarse, queue_idx);
      if (multidim_refactoring_debug_print) { // debug
        PrintSubarray4D(format("after apply correction[%d]", l), v);
      } // debug
    }
  }
  // DeviceRuntime<DeviceType>::SyncDevice();
}

template <DIM D, typename T, typename DeviceType>
void recompose(Hierarchy<D, T, DeviceType> &hierarchy,
               SubArray<D, T, DeviceType> &v, SubArray<D, T, DeviceType> w,
               SubArray<D, T, DeviceType> b, int stop_level, int queue_idx) {

  if (stop_level < 0 || stop_level > hierarchy.l_target()) {
    std::cout << log::log_err << "recompose: stop_level out of bound.\n";
    exit(-1);
  }

  Array<D, T, DeviceType> workspace;
  bool shape_pass = true;
  if (w.data() != nullptr) {
    for (DIM d = 0; d < D; d++) {
      if (w.shape(d) < hierarchy.level_shape(hierarchy.l_target())[d] + 2) {
        shape_pass = false;
      }
    }
  }
  if (w.data() == nullptr || !shape_pass) {
    log::info("recompose: allocating workspace as it is not pre-allocated.");
    std::vector<SIZE> workspace_shape =
        hierarchy.level_shape(hierarchy.l_target());
    for (DIM d = 0; d < D; d++)
      workspace_shape[d] += 2;
    workspace = Array<D, T, DeviceType>(workspace_shape);
    w = SubArray(workspace);
  }

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
      prefix +=
          std::to_string(hierarchy.level_shape(hierarchy.l_target(), d)) + "_";

    for (int l = 1; l <= stop_level; l++) {

      v_coeff.resize(hierarchy.level_shape(l));
      w_correction.resize(hierarchy.level_shape(l));
      CalcCorrection3D(hierarchy, v_coeff, w_correction, l, queue_idx);

      w_correction.resize(hierarchy.level_shape(l - 1));
      v_coarse.resize(hierarchy.level_shape(l - 1));
      SubtractND(w_correction, v_coarse, queue_idx);

      v_coeff.resize(hierarchy.level_shape(l));
      w_fine.resize(hierarchy.level_shape(l));
      CoefficientsRestore3D(hierarchy, v_coeff, w_fine, l, queue_idx);

      v_fine.resize(hierarchy.level_shape(l));
      CopyND(w_fine, v_fine, queue_idx);
      if (multidim_refactoring_debug_print) {
        PrintSubarray("output of recomposition", v);
      }
    }
  }
  if constexpr (D > 3) {
    Array<D, T, DeviceType> workspace2;
    bool shape_pass = true;
    if (b.data() != nullptr) {
      for (DIM d = 0; d < D; d++) {
        if (b.shape(d) < hierarchy.level_shape(hierarchy.l_target())[d] + 2) {
          shape_pass = false;
        }
      }
    }
    if (b.data() == nullptr || !shape_pass) {
      log::info("decompose: allocating workspace as it is not pre-allocated.");
      std::vector<SIZE> workspace_shape =
          hierarchy.level_shape(hierarchy.l_target());
      for (DIM d = 0; d < D; d++)
        workspace_shape[d] += 2;
      workspace2 = Array<D, T, DeviceType>(workspace_shape);
      b = SubArray(workspace2);
    }
    SubArray<D, T, DeviceType> b_fine = b;
    for (int l = 1; l <= stop_level; l++) {

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

      v_coeff.resize(hierarchy.level_shape(l));
      w_correction.resize(hierarchy.level_shape(l));
      CalcCorrectionND(hierarchy, v_coeff, w_correction, l, queue_idx);

      w_correction.resize(hierarchy.level_shape(l - 1));
      v_coarse.resize(hierarchy.level_shape(l - 1));
      SubtractND(w_correction, v_coarse, queue_idx);

      if (multidim_refactoring_debug_print) { // debug
        PrintSubarray4D(format("after subtract correction[%d]", l), v);
      } // deb

      v_coeff.resize(hierarchy.level_shape(l));
      w_fine.resize(hierarchy.level_shape(l));
      b_fine.resize(hierarchy.level_shape(l));
      CopyND(v_coeff, b_fine, queue_idx);
      CopyND(v_coeff, w_fine, queue_idx);

      v_fine.resize(hierarchy.level_shape(l));
      CoefficientsRestoreND(hierarchy, w_fine, b_fine, v_fine, l, queue_idx);
    } // loop levels

    if (multidim_refactoring_debug_print) { // debug
      PrintSubarray4D(format("final output"), v);
    } // deb
  }   // D > 3
  // DeviceRuntime<DeviceType>::SyncDevice();
}

} // namespace mgard_x

#endif