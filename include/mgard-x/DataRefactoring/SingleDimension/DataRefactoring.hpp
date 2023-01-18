/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../../Hierarchy/Hierarchy.h"
#include "../../RuntimeX/RuntimeX.h"

#include "DataRefactoring.h"

namespace mgard_x {

namespace data_refactoring {

namespace single_dimension {

static bool store = false;
static bool verify = false;
static bool debug_print = false;

template <DIM D, typename T, typename DeviceType>
void decompose(Hierarchy<D, T, DeviceType> &hierarchy,
               SubArray<D, T, DeviceType> &v, int start_level, int stop_level,
               int queue_idx) {

  if (stop_level < 0) {
    std::cout << log::log_err << "decompose: stop_level out of bound.\n";
    exit(-1);
  }

  std::vector<SIZE> workspace_shape =
      hierarchy.level_shape(hierarchy.l_target());
  for (DIM d = 0; d < D; d++)
    workspace_shape[d] += 2;
  Array<D, T, DeviceType> workspace(workspace_shape);
  SubArray w(workspace);

  if (singledim_refactoring_debug_print) {
    PrintSubarray("Input", v);
  }

  for (int l = start_level; l > stop_level; l--) {
    for (int curr_dim = D - 1; curr_dim >= 0; curr_dim--) {
      if (singledim_refactoring_debug_print) {
        std::cout << "l: " << l << " curr_dim: " << curr_dim << "\n";
      }
      std::vector<SIZE> fine_shape(D);
      std::vector<SIZE> coarse_shape(D);
      std::vector<SIZE> coeff_shape(D);
      for (int d = D - 1; d >= 0; d--) {
        if (d > curr_dim) {
          fine_shape[d] = hierarchy.level_shape(l - 1, d);
        } else {
          fine_shape[d] = hierarchy.level_shape(l, d);
        }
      }

      for (int d = D - 1; d >= 0; d--) {
        if (d == curr_dim) {
          coarse_shape[d] = hierarchy.level_shape(l - 1, d);
          coeff_shape[d] = fine_shape[d] - hierarchy.level_shape(l - 1, d);
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
      coeff.offset_dim(curr_dim, hierarchy.level_shape(l - 1, curr_dim));
      coeff.resize(coeff_shape);
      SubArray<D, T, DeviceType> correction = w;
      correction.resize(coarse_shape);

      multi_dimension::CopyND(v_fine, w_fine, queue_idx);

      if (singledim_refactoring_debug_print) {
        PrintSubarray("COPY", w_fine);
      }

      CalcCoefficients(curr_dim, SubArray(hierarchy.ratio(l, curr_dim)), w_fine,
                       coarse, coeff, queue_idx);

      if (singledim_refactoring_debug_print) {
        PrintSubarray("SingleDimensionCoefficient - fine", w_fine);
        PrintSubarray("SingleDimensionCoefficient - corase", coarse);
        PrintSubarray("SingleDimensionCoefficient - coeff", coeff);
      }

      CalcCorrection(hierarchy, coeff, correction, curr_dim, l, queue_idx);

      multi_dimension::AddND(correction, coarse, queue_idx);

      if (singledim_refactoring_debug_print) {
        PrintSubarray("ADD", coarse);
      }

    } // loop dimensions
  }   // loop levels
}

template <DIM D, typename T, typename DeviceType>
void recompose(Hierarchy<D, T, DeviceType> &hierarchy,
               SubArray<D, T, DeviceType> &v, int start_level, int stop_level,
               int queue_idx) {

  if (stop_level < 0 || stop_level > hierarchy.l_target()) {
    std::cout << log::log_err << "recompose: stop_level out of bound.\n";
    exit(-1);
  }

  std::vector<SIZE> workspace_shape =
      hierarchy.level_shape(hierarchy.l_target());
  for (DIM d = 0; d < D; d++)
    workspace_shape[d] += 2;
  Array<D, T, DeviceType> workspace(workspace_shape);
  SubArray w(workspace);

  if (singledim_refactoring_debug_print) {
    PrintSubarray("Input", v);
  }

  for (int l = start_level; l < stop_level; l++) {
    for (int curr_dim = 0; curr_dim < D; curr_dim++) {
      if (singledim_refactoring_debug_print) {
        std::cout << "l: " << l << " curr_dim: " << curr_dim << "\n";
      }
      std::vector<SIZE> fine_shape(D);
      std::vector<SIZE> coarse_shape(D);
      std::vector<SIZE> coeff_shape(D);
      for (int d = D - 1; d >= 0; d--) {
        if (d > curr_dim) {
          fine_shape[d] = hierarchy.level_shape(l, d);
        } else {
          fine_shape[d] = hierarchy.level_shape(l + 1, d);
        }
      }

      for (int d = D - 1; d >= 0; d--) {
        if (d == curr_dim) {
          coarse_shape[d] = hierarchy.level_shape(l, d);
          coeff_shape[d] = fine_shape[d] - hierarchy.level_shape(l, d);
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
      coeff.offset_dim(curr_dim, hierarchy.level_shape(l, curr_dim));
      coeff.resize(coeff_shape);
      SubArray<D, T, DeviceType> correction = w;
      correction.resize(coarse_shape);

      CalcCorrection(hierarchy, coeff, correction, curr_dim, l + 1, queue_idx);

      multi_dimension::SubtractND(correction, coarse, queue_idx);

      if (singledim_refactoring_debug_print) {
        PrintSubarray("SUBTRACT", coarse);
      }

      CoefficientsRestore(curr_dim, SubArray(hierarchy.ratio(l + 1, curr_dim)),
                          w_fine, coarse, coeff, queue_idx);

      if (singledim_refactoring_debug_print) {
        PrintSubarray("SingleDimensionCoefficient - fine", w_fine);
        PrintSubarray("SingleDimensionCoefficient - corase", coarse);
        PrintSubarray("SingleDimensionCoefficient - coeff", coeff);
      }

      multi_dimension::CopyND(w_fine, v_fine, queue_idx);

      if (singledim_refactoring_debug_print) {
        PrintSubarray("COPY", v_fine);
      }

    } // loop dimensions
  }   // loop levels
}

} // namespace single_dimension

} // namespace data_refactoring

} // namespace mgard_x
