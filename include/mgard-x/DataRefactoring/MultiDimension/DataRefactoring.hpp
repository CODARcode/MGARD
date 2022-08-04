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
               SubArray<D, T, DeviceType> &v, int stop_level, 
               int queue_idx) {

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
    prefix += std::to_string(hierarchy.level_shape(hierarchy.l_target(), d)) + "_";

  std::vector<SIZE> workspace_shape = hierarchy.level_shape(hierarchy.l_target());
  for (DIM d = 0; d < D; d++) workspace_shape[d] += 2;
  Array<D, T, DeviceType> workspace(workspace_shape);
  // workspace.memset(0); can cause large overhead in HIP
  SubArray w(workspace);

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

      w_correction.resize(hierarchy.level_shape(l-1));
      v_coarse.resize(hierarchy.level_shape(l-1));
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

      w_correction.resize(hierarchy.level_shape(l-1));
      v_coarse.resize(hierarchy.level_shape(l-1));
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
               SubArray<D, T, DeviceType> &v, int stop_level, int queue_idx) {

  if (stop_level < 0 || stop_level > hierarchy.l_target()) {
    std::cout << log::log_err << "recompose: stop_level out of bound.\n";
    exit(-1);
  }
  std::vector<SIZE> workspace_shape = hierarchy.level_shape(hierarchy.l_target());
  for (DIM d = 0; d < D; d++) workspace_shape[d] += 2;
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
      prefix += std::to_string(hierarchy.level_shape(hierarchy.l_target(), d)) + "_";

    for (int l = 1; l <= stop_level; l++) {

      v_coeff.resize(hierarchy.level_shape(l));
      w_correction.resize(hierarchy.level_shape(l));
      CalcCorrection3D(hierarchy, v_coeff, w_correction, l, queue_idx);

      w_correction.resize(hierarchy.level_shape(l-1));
      v_coarse.resize(hierarchy.level_shape(l-1));
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
    Array<D, T, DeviceType> workspace2(workspace_shape);
    SubArray b(workspace2);
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

      w_correction.resize(hierarchy.level_shape(l-1));
      v_coarse.resize(hierarchy.level_shape(l-1));
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
  DeviceRuntime<DeviceType>::SyncDevice();
}

// template <DIM D, typename T, typename DeviceType>
// void decompose_adaptive_resolution(Hierarchy<D, T, DeviceType> &hierarchy,
//                SubArray<D, T, DeviceType> &v, SIZE l_target, 
//                SubArray<1, T, DeviceType> level_max, 
//                SubArray<D+1, T, DeviceType> * max_abs_coeffcient, 
//                int queue_idx) {

//   std::string prefix = "decomp_";
//   if (sizeof(T) == sizeof(double))
//     prefix += "d_";
//   if (sizeof(T) == sizeof(float))
//     prefix += "f_";
//   for (int d = 0; d < D; d++)
//     prefix += std::to_string(hierarchy.shape[d]) + "_";
//   // std::cout << prefix << std::endl;

//   std::vector<SIZE> workspace_shape(D);
//   for (DIM d = 0; d < D; d++)
//     workspace_shape[d] = hierarchy.dofs[d][0] + 2;
//   std::reverse(workspace_shape.begin(), workspace_shape.end());
//   Array<D, T, DeviceType> workspace(workspace_shape);
//   // workspace.memset(0); can cause large overhead in HIP
//   SubArray w(workspace);

//   SubArray<D, T, DeviceType> v_fine = v;
//   SubArray<D, T, DeviceType> w_fine = w;
//   SubArray<D, T, DeviceType> v_coeff = v;
//   SubArray<D, T, DeviceType> w_correction = w;
//   SubArray<D, T, DeviceType> v_coarse = v;

//   if constexpr (D <= 3) {
//     for (int l = 0; l < l_target; ++l) {
//       if (multidim_refactoring_debug_print) {
//         PrintSubarray("input v", v);
//       }

//       v_fine.resize(hierarchy.shapes_vec[l]);
//       w_fine.resize(hierarchy.shapes_vec[l]);
//       CopyND(v_fine, w_fine, queue_idx);

//       v_coeff.resize(hierarchy.shapes_vec[l]);
//       SubArray<D+1, T, DeviceType> max_abs_coeffcient_finer = l > 0 ? max_abs_coeffcient[l-1] : SubArray<D+1, T, DeviceType>();
//       CalcCoefficients3DWithErrorCollection(hierarchy, w_fine, v_coeff, l, max_abs_coeffcient[l], max_abs_coeffcient_finer, queue_idx);

//       w_correction.resize(hierarchy.shapes_vec[l]);
//       CalcCorrection3D(hierarchy, v_coeff, w_correction, l, queue_idx);

//       w_correction.resize(hierarchy.shapes_vec[l + 1]);
//       v_coarse.resize(hierarchy.shapes_vec[l + 1]);
//       AddND(w_correction, v_coarse, queue_idx);
//       if (multidim_refactoring_debug_print) {
//         PrintSubarray("after add", v);
//       }
//     } // end of loop

//     if (multidim_refactoring_debug_print) {
//       PrintSubarray("output of decomposition", v);
//     }
//   }

//   LevelMax(l_target,
//            SubArray<1, SIZE, DeviceType>(hierarchy.ranges, true),
//            v, level_max, queue_idx);

//   if constexpr (D > 3) {
//     std::cout << log::log_err << "decompose_adaptive_resolution does not support higher than 3D\n";
//   }
//   DeviceRuntime<DeviceType>::SyncDevice();
// }


// template <DIM D, typename T, typename DeviceType>
// void initialize_error_impact_budget(T error_target, SubArray<D, T, DeviceType> error_impact_budget) {
//   T max_error_sum = error_target / (1 + std::pow(std::sqrt(3) / 2, D)); 
//   SIZE total_level = error_impact_budget.getShape(0);
//   std::vector<T> error_impact_budget_host(total_level);
//   for (int l = 0; l < total_level; l++) {
//     error_impact_budget_host[l] = max_error_sum / total_level;
//   }
//   MemoryManager<DeviceType>::Copy1D(error_impact_budget.data(), error_impact_budget_host.data(),
//                                     total_level, 0);
//   DeviceRuntime<DeviceType>::SyncQueue(0);
// }

// template <DIM D, typename T, typename DeviceType>
// T error_impact_to_error(SubArray<D, T, DeviceType> error_impact) {
//   SIZE total_level = error_impact.getShape(0);
//   std::vector<T> error_impact_host(total_level);
//   MemoryManager<DeviceType>::Copy1D(error_impact_host.data(), error_impact.data(),
//                                     total_level, 0);
//   DeviceRuntime<DeviceType>::SyncQueue(0);
//   T sum = 0;
//   for (int i = 0; i < total_level; i++) {
//     sum += error_impact_host[i];
//   }
//   return sum * (1 + std::pow(std::sqrt(3) / 2, D)); 
// }

// template <DIM D, typename T, typename DeviceType>
// Array<D, T, DeviceType> recompose_adaptive_resolution(Hierarchy<D, T, DeviceType> &hierarchy,
//                SubArray<D, T, DeviceType> &v, SIZE l_target, 
//                T iso_value, T error_target,
//                SubArray<1, T, DeviceType> level_max, 
//                SubArray<D+1, T, DeviceType> * max_abs_coefficient,
//                SubArray<D, SIZE, DeviceType> * refinement_flag,
//                int queue_idx) {

//   std::vector<SIZE> workspace_shape(D);
//   for (DIM d = 0; d < D; d++)
//     workspace_shape[d] = hierarchy.dofs[d][0] + 2;
//   std::reverse(workspace_shape.begin(), workspace_shape.end());
//   Array<D, T, DeviceType> workspace(workspace_shape);
//   // workspace.memset(0); // can cause large overhead in HIP
//   SubArray w(workspace);

//   Array<D, T, DeviceType> retrieved_coefficient_array(workspace_shape);
//   SubArray retrieved_coefficient(retrieved_coefficient_array);

//   Array<1, T, DeviceType> error_impact_budget_array({l_target+1});
//   Array<1, T, DeviceType> previous_error_impact_array({l_target+1});
//   Array<1, T, DeviceType> current_error_impact_array({l_target+1});
//   Array<1, T, DeviceType> partial_recompose_error_impact_array({l_target+1});
//   partial_recompose_error_impact_array.memset(0);

//   SubArray error_impact_budget(error_impact_budget_array);
//   SubArray previous_error_impact(previous_error_impact_array);
//   SubArray current_error_impact(current_error_impact_array);
//   SubArray partial_recompose_error_impact(partial_recompose_error_impact_array);

//   initialize_error_impact_budget(error_target, error_impact_budget);

//   // PrintSubarray("error_impact_budget", error_impact_budget);

//   // Corasest level is not copied since we always assume it is recomposed
//   // So we only copy 'l_target' instead of 'l_target+1' elements here.
//   MemoryManager<DeviceType>::Copy1D(partial_recompose_error_impact.data(), level_max.data(),
//                                     l_target, queue_idx);
  

//   // Assume no coefficient loss at the beginning
//   T current_error = 0;


//   SubArray<D, T, DeviceType> v_fine = v;
//   SubArray<D, T, DeviceType> w_fine = w;
//   SubArray<D, T, DeviceType> v_coeff = v;
//   SubArray<D, T, DeviceType> w_correction = w;
//   SubArray<D, T, DeviceType> v_coarse = v;

//   T partial_recompose_error = 0;

//   if constexpr (D <= 3) {
//     if (multidim_refactoring_debug_print) {
//       PrintSubarray("input of recomposition", v);
//     }
//     std::string prefix = "recomp_";
//     if (sizeof(T) == sizeof(double))
//       prefix += "d_";
//     if (sizeof(T) == sizeof(float))
//       prefix += "f_";
//     for (int d = 0; d < D; d++)
//       prefix += std::to_string(hierarchy.shape[d]) + "_";
//     // std::cout << prefix << std::endl;

//     if (multidim_refactoring_debug_print) {
//       PrintSubarray("v_fine before resize", v_fine);
//     }

//     v_fine.resize(hierarchy.shapes_vec[l_target]);

//     // if the partial recomposed data is already accurate enough, we do early return
//     partial_recompose_error = error_impact_to_error(partial_recompose_error_impact);
//     if (partial_recompose_error < error_target) {
//       std::vector<SIZE> result_data_shape = hierarchy.shapes_vec[l_target];
//       std::reverse(result_data_shape.begin(), result_data_shape.end());
//       Array<D, T, DeviceType> result_data_array(result_data_shape);
//       SubArray result_data(result_data_array);
//       CopyND(v_fine, result_data, queue_idx);
//       return result_data_array;
//     }
    
//     for (int l = l_target - 1; l >= 0; l--) {
//       v_fine.resize(hierarchy.shapes_vec[l]);
//       v_coarse.resize(hierarchy.shapes_vec[l + 1]);
//       if (multidim_refactoring_debug_print) {
//         PrintSubarray("input of recomposition", v);
//       }
//       DeviceRuntime<DeviceType>::SyncDevice();
//       // PrintSubarray("partial_recompose_error_impact", partial_recompose_error_impact);
//       partial_recompose_error = error_impact_to_error(partial_recompose_error_impact);
//       // std::cout << "level " << l+1 << " partial_recompose_error: " << partial_recompose_error << "\n";

//       // Feature detection on coarse representation

//       // SubArray<D, SIZE, DeviceType> refinement_flag_coarser = l+2 < l_target+1 ? refinement_flag[l+2] : SubArray<D, SIZE, DeviceType>();
//       // EarlyFeatureDetector(v_coarse, partial_recompose_error, iso_value,
//       //                     refinement_flag_coarser, refinement_flag[l+1], queue_idx);

//       DeviceRuntime<DeviceType>::SyncDevice();


//       // PrintSubarray("v_coarse", v_coarse);

//       // PrintSubarray("refinement_flag[l+1]", refinement_flag[l+1]);
      

//       // Discard coefficients
//       // AccuracyGuard(l_target+1, l, error_impact_budget,
//       //               previous_error_impact, current_error_impact,
//       //               max_abs_coefficient[l+1], refinement_flag[l+1], queue_idx);

//       // PrintSubarray("refinement_flag[l+1]", refinement_flag[l+1]);

//       // Update partial recompose error
//       // MemoryManager<DeviceType>::Copy1D(partial_recompose_error_impact(l),
//       //                                   current_error_impact(l), 1, queue_idx);

//       // Retrieve necessary coefficients
//       // CoefficientRetriever(hierarchy, v_coeff,
//       //                      refinement_flag[l+1],
//       //                      retrieved_coefficient, l, queue_idx);

//       // Retrieve all coefficients
//       retrieved_coefficient = v_coeff;

//       T zero = 0;
//       MemoryManager<DeviceType>::Copy1D(partial_recompose_error_impact(l),
//                                         &zero, 1, queue_idx);

//       retrieved_coefficient.resize(hierarchy.shapes_vec[l]);
//       w_correction.resize(hierarchy.shapes_vec[l]);
//       CalcCorrection3D(hierarchy, retrieved_coefficient, w_correction, l, queue_idx);

//       w_correction.resize(hierarchy.shapes_vec[l + 1]);
//       v_coarse.resize(hierarchy.shapes_vec[l + 1]);
//       SubtractND(w_correction, v_coarse, queue_idx);

//       retrieved_coefficient.resize(hierarchy.shapes_vec[l]);
//       w_fine.resize(hierarchy.shapes_vec[l]);
//       CoefficientsRestore3D(hierarchy, retrieved_coefficient, w_fine, l, queue_idx);

//       v_fine.resize(hierarchy.shapes_vec[l]);
//       CopyND(w_fine, v_fine, queue_idx);

//       // if the partial recomposed data is already accurate enough, we do early return
//       if (partial_recompose_error < error_target) {
//         std::vector<SIZE> result_data_shape = hierarchy.shapes_vec[l_target];
//         std::reverse(result_data_shape.begin(), result_data_shape.end());
//         Array<D, T, DeviceType> result_data_array(result_data_shape);
//         SubArray result_data(result_data_array);
//         CopyND(v_fine, result_data, queue_idx);
//         return result_data_array;
//       }

//       if (multidim_refactoring_debug_print) {
//         PrintSubarray("output of recomposition", v);
//       }
//     } // end of loop

//     // printf("v_fine shape: %u %u %u\n", v_fine.getShape(0), v_fine.getShape(1), v_fine.getShape(2));   

//     // PrintSubarray("partial_recompose_error_impact", partial_recompose_error_impact);
//     partial_recompose_error = error_impact_to_error(partial_recompose_error_impact);
//     // std::cout << "level " << 0 << " partial_recompose_error: " << partial_recompose_error << "\n";
//     // EarlyFeatureDetector(v, partial_recompose_error, iso_value,
//     //                      refinement_flag[1], refinement_flag[0], queue_idx);
//     DeviceRuntime<DeviceType>::SyncDevice();
//     // PrintSubarray("v", v);
//     // PrintSubarray("refinement_flag[0]", refinement_flag[0]);

//     // If there did not do early return, we need to return the full resolution data.
//     std::vector<SIZE> result_data_shape = hierarchy.shapes_vec[0];
//     std::reverse(result_data_shape.begin(), result_data_shape.end());
//     Array<D, T, DeviceType> result_data(result_data_shape);
//     result_data.load(v_fine.data());
//     return result_data;
//   }
//   if constexpr (D > 3) {
//     std::cout << log::log_err << "recompose_adaptive_resolution does not support higher than 3D\n";
//     return Array<D, T, DeviceType>();
//   }
//   DeviceRuntime<DeviceType>::SyncDevice();
// }

} // namespace mgard_x

#endif