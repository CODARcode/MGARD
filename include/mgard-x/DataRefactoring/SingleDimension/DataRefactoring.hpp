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

static bool store = false;
static bool verify = false;
static bool debug_print = false;

template <DIM D, typename T, typename DeviceType>
void decompose_single(Hierarchy<D, T, DeviceType> &hierarchy,
                      SubArray<D, T, DeviceType> &v, SIZE l_target,
                      int queue_idx) {
  std::vector<SIZE> workspace_shape = hierarchy.level_shape(hierarchy.l_target);
  for (DIM d = 0; d < D; d++) workspace_shape[d] += 2;
  Array<D, T, DeviceType> workspace(workspace_shape);
  SubArray w(workspace);

  if (singledim_refactoring_debug_print) {
    PrintSubarray("Input", v);
  }

  for (int l = hierarchy.l_target; l > 0; l--) {
    for (int curr_dim = D-1; curr_dim >= 0; curr_dim--) {
      if (singledim_refactoring_debug_print) {
        std::cout << "l: " << l << " curr_dim: " << curr_dim << "\n";
      }
      std::vector<SIZE> fine_shape(D);
      std::vector<SIZE> coarse_shape(D);
      std::vector<SIZE> coeff_shape(D);
      for (int d = D-1; d >= 0; d--) {
        if (d > curr_dim) {
          fine_shape[d] = hierarchy.level_shape(l-1, d);
        } else {
          fine_shape[d] = hierarchy.level_shape(l, d);
        }
      }

      for (int d = D-1; d >= 0; d--) {
        if (d == curr_dim) {
          coarse_shape[d] = hierarchy.level_shape(l-1, d);
          coeff_shape[d] = fine_shape[d] - hierarchy.level_shape(l-1, d);
        } else {
          coarse_shape[d] = fine_shape[d];
          coeff_shape[d] = fine_shape[d];
        }
      }

      SubArray<D, T, DeviceType> v_fine = v;
      v_fine.resize(fine_shape);
      SubArray<D, T, DeviceType> w_fine = w;
      w_fine.resize(fine_shape);
      SubArray<D, T, DeviceType> coarse = v;
      coarse.resize(coarse_shape);
      SubArray<D, T, DeviceType> coeff = v;
      coeff.offset(curr_dim, hierarchy.level_shape(l-1, curr_dim));
      coeff.resize(coeff_shape);
      SubArray<D, T, DeviceType> correction = w;
      correction.resize(coarse_shape);

      CopyND(v_fine, w_fine, queue_idx);

      if (singledim_refactoring_debug_print) {
        PrintSubarray("COPY", w_fine);
      }

      CalcCoefficients(curr_dim, SubArray(hierarchy.ratio(l, curr_dim)),
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
  std::vector<SIZE> workspace_shape = hierarchy.level_shape(hierarchy.l_target);
  for (DIM d = 0; d < D; d++) workspace_shape[d] += 2;
  Array<D, T, DeviceType> workspace(workspace_shape);
  SubArray w(workspace);

  if (singledim_refactoring_debug_print) {
    PrintSubarray("Input", v);
  }

  for (int l = 1; l <= hierarchy.l_target; l++) {
    for (int curr_dim = 0; curr_dim < D; curr_dim++) {
      if (singledim_refactoring_debug_print) {
        std::cout << "l: " << l << " curr_dim: " << curr_dim << "\n";
      }
      std::vector<SIZE> fine_shape(D);
      std::vector<SIZE> coarse_shape(D);
      std::vector<SIZE> coeff_shape(D);
      for (int d = D-1; d >= 0; d--) {
        if (d > curr_dim) {
          fine_shape[d] = hierarchy.level_shape(l-1, d);
        } else {
          fine_shape[d] = hierarchy.level_shape(l, d);
        }
      }

      for (int d = D-1; d >= 0; d--) {
        if (d == curr_dim) {
          coarse_shape[d] = hierarchy.level_shape(l-1, d);
          coeff_shape[d] = fine_shape[d] - hierarchy.level_shape(l-1, d);
        } else {
          coarse_shape[d] = fine_shape[d];
          coeff_shape[d] = fine_shape[d];
        }
      }

      SubArray<D, T, DeviceType> v_fine = v;
      v_fine.resize(fine_shape);
      SubArray<D, T, DeviceType> w_fine = w;
      w_fine.resize(fine_shape);
      SubArray<D, T, DeviceType> coarse = v;
      coarse.resize(coarse_shape);
      SubArray<D, T, DeviceType> coeff = v;
      coeff.offset(curr_dim, hierarchy.level_shape(l-1, curr_dim));
      coeff.resize(coeff_shape);
      SubArray<D, T, DeviceType> correction = w;
      correction.resize(coarse_shape);

      CalcCorrection(hierarchy, coeff, correction, curr_dim, l, queue_idx);

      SubtractND(correction, coarse, queue_idx);

      if (singledim_refactoring_debug_print) {
        PrintSubarray("SUBTRACT", coarse);
      }

      CoefficientsRestore(curr_dim,
                          SubArray(hierarchy.ratio(l, curr_dim)), w_fine,
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