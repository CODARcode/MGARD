/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../../Hierarchy/Hierarchy.hpp"
#include "../../RuntimeX/RuntimeX.h"

#include "DataRefactoring.h"

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
void decompose_single(Hierarchy<D, T, DeviceType> &hierarchy,
                      SubArray<D, T, DeviceType> &v, SIZE l_target,
                      int queue_idx) {
  std::vector<SIZE> workspace_shape(D);
  for (DIM d = 0; d < D; d++)
    workspace_shape[d] = hierarchy.dofs[d][0] + 2;
  std::reverse(workspace_shape.begin(), workspace_shape.end());
  Array<D, T, DeviceType> workspace(workspace_shape);
  SubArray w(workspace);

  if (singledim_refactoring_debug_print) {
    PrintSubarray("Input", v);
  }

  for (int l = 0; l < l_target; ++l) {
    for (int curr_dim = 0; curr_dim < D; curr_dim++) {
      if (singledim_refactoring_debug_print) {
        std::cout << "l: " << l << " curr_dim: " << curr_dim << "\n";
      }
      std::vector<SIZE> fine_shape(D);
      std::vector<SIZE> coarse_shape(D);
      std::vector<SIZE> coeff_shape(D);
      for (DIM d = 0; d < D; d++) {
        if (d < curr_dim) {
          fine_shape[d] = hierarchy.dofs[d][l + 1];
        } else {
          fine_shape[d] = hierarchy.dofs[d][l];
        }
      }

      for (DIM d = 0; d < D; d++) {
        if (d == curr_dim) {
          coarse_shape[d] = hierarchy.dofs[d][l + 1];
          coeff_shape[d] = fine_shape[d] - hierarchy.dofs[d][l + 1];
        } else {
          coarse_shape[d] = fine_shape[d];
          coeff_shape[d] = fine_shape[d];
        }
      }

      Array<1, SIZE, DeviceType> fine_shape_array({(SIZE)D});
      fine_shape_array.load(fine_shape.data());
      Array<1, SIZE, DeviceType> coarse_shape_array({(SIZE)D});
      coarse_shape_array.load(coarse_shape.data());
      // PrintSubarray("v_shape_array", SubArray<1, SIZE,
      // DeviceType>(v_shape_array));
      SubArray<D, T, DeviceType> v_fine = v;
      v_fine.resize(fine_shape);
      SubArray<D, T, DeviceType> w_fine = w;
      w_fine.resize(fine_shape);
      SubArray<D, T, DeviceType> coarse = v;
      coarse.resize(coarse_shape);
      SubArray<D, T, DeviceType> coeff = v;
      coeff.offset(curr_dim, hierarchy.dofs[curr_dim][l + 1]);
      coeff.resize(coeff_shape);
      SubArray<D, T, DeviceType> correction = w;
      correction.resize(coarse_shape);

      CopyND(v_fine, w_fine, queue_idx);

      if (singledim_refactoring_debug_print) {
        PrintSubarray("COPY", w_fine);
      }

      CalcCoefficients(curr_dim, SubArray(hierarchy.ratio_array[curr_dim][l]),
                       w_fine, coarse, coeff, queue_idx);

      if (singledim_refactoring_debug_print) {
        PrintSubarray("SingleDimensionCoefficient - fine", w_fine);
        PrintSubarray("SingleDimensionCoefficient - corase", coarse);
        PrintSubarray("SingleDimensionCoefficient - coeff", coeff);
      }

      CalcCorrection(hierarchy, coeff, correction, curr_dim, l, queue_idx);

      AddND(correction, coarse, queue_idx);

      if (singledim_refactoring_debug_print) {
        PrintSubarray("ADD", coarse);
      }

    } // loop dimensions
  }   // loop levels
}

template <DIM D, typename T, typename DeviceType>
void recompose_single(Hierarchy<D, T, DeviceType> &hierarchy,
                      SubArray<D, T, DeviceType> &v, SIZE l_target,
                      int queue_idx) {
  std::vector<SIZE> workspace_shape(D);
  for (DIM d = 0; d < D; d++)
    workspace_shape[d] = hierarchy.dofs[d][0] + 2;
  std::reverse(workspace_shape.begin(), workspace_shape.end());
  Array<D, T, DeviceType> workspace(workspace_shape);
  SubArray w(workspace);

  if (singledim_refactoring_debug_print) {
    PrintSubarray("Input", v);
  }

  for (int l = l_target - 1; l >= 0; --l) {
    for (int curr_dim = D - 1; curr_dim >= 0; curr_dim--) {
      if (singledim_refactoring_debug_print) {
        std::cout << "l: " << l << " curr_dim: " << curr_dim << "\n";
      }
      std::vector<SIZE> fine_shape(D);
      std::vector<SIZE> coarse_shape(D);
      std::vector<SIZE> coeff_shape(D);
      for (DIM d = 0; d < D; d++) {
        if (d < curr_dim) {
          fine_shape[d] = hierarchy.dofs[d][l + 1];
        } else {
          fine_shape[d] = hierarchy.dofs[d][l];
        }
      }

      for (DIM d = 0; d < D; d++) {
        if (d == curr_dim) {
          coarse_shape[d] = hierarchy.dofs[d][l + 1];
          coeff_shape[d] = fine_shape[d] - hierarchy.dofs[d][l + 1];
        } else {
          coarse_shape[d] = fine_shape[d];
          coeff_shape[d] = fine_shape[d];
        }
      }

      Array<1, SIZE, DeviceType> fine_shape_array({(SIZE)D});
      fine_shape_array.load(fine_shape.data());
      Array<1, SIZE, DeviceType> coarse_shape_array({(SIZE)D});
      coarse_shape_array.load(coarse_shape.data());

      SubArray<D, T, DeviceType> v_fine = v;
      v_fine.resize(fine_shape);
      SubArray<D, T, DeviceType> w_fine = w;
      w_fine.resize(fine_shape);
      SubArray<D, T, DeviceType> coarse = v;
      coarse.resize(coarse_shape);
      SubArray<D, T, DeviceType> coeff = v;
      coeff.offset(curr_dim, hierarchy.dofs[curr_dim][l + 1]);
      coeff.resize(coeff_shape);
      SubArray<D, T, DeviceType> correction = w;
      correction.resize(coarse_shape);

      CalcCorrection(hierarchy, coeff, correction, curr_dim, l, queue_idx);

      SubtractND(correction, coarse, queue_idx);

      if (singledim_refactoring_debug_print) {
        PrintSubarray("SUBTRACT", coarse);
      }

      CoefficientsRestore(curr_dim,
                          SubArray(hierarchy.ratio_array[curr_dim][l]), w_fine,
                          coarse, coeff, queue_idx);

      if (singledim_refactoring_debug_print) {
        PrintSubarray("SingleDimensionCoefficient - fine", w_fine);
        PrintSubarray("SingleDimensionCoefficient - corase", coarse);
        PrintSubarray("SingleDimensionCoefficient - coeff", coeff);
      }

      CopyND(w_fine, v_fine, queue_idx);

      if (singledim_refactoring_debug_print) {
        PrintSubarray("COPY", v_fine);
      }

    } // loop dimensions
  }   // loop levels
}

} // namespace mgard_x