/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../../Hierarchy/Hierarchy.hpp"
#include "../../RuntimeX/RuntimeX.h"

#include "../MultiDimension/Correction/IterativeProcessingKernel.hpp"
#include "../MultiDimension/Correction/LevelwiseProcessingKernel.hpp"
#include "Coefficient/CoefficientKernel.hpp"
#include "Correction/MassTransKernel.hpp"

namespace mgard_x {

static bool store = false;
static bool verify = false;
static bool debug_print = false;

template <DIM D, typename T, typename DeviceType>
void calc_correction_single(Hierarchy<D, T, DeviceType> &hierarchy,
                            SubArray<D, T, DeviceType> &coeff,
                            SubArray<D, T, DeviceType> &correction,
                            SIZE curr_dim, SIZE l, int queue_idx) {

  SingleDimensionMassTrans<D, T, DeviceType>().Execute(
      curr_dim, SubArray(hierarchy.dist_array[curr_dim][l]),
      SubArray(hierarchy.ratio_array[curr_dim][l]), coeff, correction,
      queue_idx);

  if (debug_print) {
    PrintSubarray("SingleDimensionMassTrans", correction);
  }

  DIM curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;
  if (curr_dim == 0) {
    correction.project(curr_dim_f, curr_dim_c, curr_dim_r);
    Ipk1Reo<D, T, DeviceType>().Execute(
        curr_dim_r, curr_dim_c, curr_dim_f,
        SubArray(hierarchy.am_array[curr_dim_f][l + 1]),
        SubArray(hierarchy.bm_array[curr_dim_f][l + 1]), correction, queue_idx);
    if (debug_print) {
      PrintSubarray("Ipk1Reo", correction);
    }

  } else if (curr_dim == 1) {
    curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;
    correction.project(curr_dim_f, curr_dim_c, curr_dim_r);
    Ipk2Reo<D, T, DeviceType>().Execute(
        curr_dim_r, curr_dim_c, curr_dim_f,
        SubArray(hierarchy.am_array[curr_dim_c][l + 1]),
        SubArray(hierarchy.bm_array[curr_dim_c][l + 1]), correction, queue_idx);
    if (debug_print) {
      PrintSubarray("Ipk2Reo", correction);
    }
  } else if (curr_dim == 2) {
    curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = 2;
    correction.project(curr_dim_f, curr_dim_c, curr_dim_r);
    Ipk3Reo<D, T, DeviceType>().Execute(
        curr_dim_r, curr_dim_c, curr_dim_f,
        SubArray(hierarchy.am_array[curr_dim_r][l + 1]),
        SubArray(hierarchy.bm_array[curr_dim_r][l + 1]), correction, queue_idx);
    if (debug_print) {
      PrintSubarray("Ipk3Reo", correction);
    }
  } else {
    curr_dim_f = 0, curr_dim_c = 1, curr_dim_r = curr_dim;
    correction.project(curr_dim_f, curr_dim_c, curr_dim_r);
    Ipk3Reo<D, T, DeviceType>().Execute(
        curr_dim_r, curr_dim_c, curr_dim_f,
        SubArray(hierarchy.am_array[curr_dim_r][l + 1]),
        SubArray(hierarchy.bm_array[curr_dim_r][l + 1]), correction, queue_idx);
    if (debug_print) {
      PrintSubarray("Ipk3Reo", correction);
    }
  }
}

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

  if (debug_print) {
    PrintSubarray("Input", v);
  }

  for (int l = 0; l < l_target; ++l) {
    for (int curr_dim = 0; curr_dim < D; curr_dim++) {
      if (debug_print) {
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

      LwpkReo<D, T, COPY, DeviceType>().Execute(v_fine, w_fine, queue_idx);

      if (debug_print) {
        PrintSubarray("COPY", w_fine);
      }

      SingleDimensionCoefficient<D, T, DECOMPOSE, DeviceType>().Execute(
          curr_dim, SubArray(hierarchy.ratio_array[curr_dim][l]), w_fine,
          coarse, coeff, queue_idx);

      if (debug_print) {
        PrintSubarray("SingleDimensionCoefficient - fine", w_fine);
        PrintSubarray("SingleDimensionCoefficient - corase", coarse);
        PrintSubarray("SingleDimensionCoefficient - coeff", coeff);
      }

      calc_correction_single(hierarchy, coeff, correction, curr_dim, l,
                             queue_idx);

      LwpkReo<D, T, ADD, DeviceType>().Execute(correction, coarse, queue_idx);

      if (debug_print) {
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

  if (debug_print) {
    PrintSubarray("Input", v);
  }

  for (int l = l_target - 1; l >= 0; --l) {
    for (int curr_dim = D - 1; curr_dim >= 0; curr_dim--) {
      if (debug_print) {
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

      calc_correction_single(hierarchy, coeff, correction, curr_dim, l,
                             queue_idx);

      LwpkReo<D, T, SUBTRACT, DeviceType>().Execute(correction, coarse,
                                                    queue_idx);

      if (debug_print) {
        PrintSubarray("SUBTRACT", coarse);
      }

      SingleDimensionCoefficient<D, T, RECOMPOSE, DeviceType>().Execute(
          curr_dim, SubArray(hierarchy.ratio_array[curr_dim][l]), w_fine,
          coarse, coeff, queue_idx);

      if (debug_print) {
        PrintSubarray("SingleDimensionCoefficient - fine", w_fine);
        PrintSubarray("SingleDimensionCoefficient - corase", coarse);
        PrintSubarray("SingleDimensionCoefficient - coeff", coeff);
      }

      LwpkReo<D, T, COPY, DeviceType>().Execute(w_fine, v_fine, queue_idx);

      if (debug_print) {
        PrintSubarray("COPY", v_fine);
      }

    } // loop dimensions
  }   // loop levels
}

} // namespace mgard_x