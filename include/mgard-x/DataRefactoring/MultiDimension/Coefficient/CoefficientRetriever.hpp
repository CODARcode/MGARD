/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../../../Hierarchy/Hierarchy.hpp"
#include "../../../RuntimeX/RuntimeX.h"

#include "../DataRefactoring.h"

#include "CoefficientRetrieverKernel.hpp"

#ifndef MGARD_X_DATA_REFACTORING_COEFFICIENTS_RESTORE_3D
#define MGARD_X_DATA_REFACTORING_COEFFICIENTS_RESTORE_3D

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
void CoefficientRetriever(Hierarchy<D, T, DeviceType> &hierarchy,
                           SubArray<D, T, DeviceType> dinput,
                           SubArray<D, SIZE, DeviceType> refinement_flag,
                           SubArray<D, T, DeviceType> &doutput, SIZE l,
                           int queue_idx) {

  int range_l = std::min(6, (int)std::log2(hierarchy.dofs[0][l]) - 1);
  int range_lp1 = std::min(6, (int)std::log2(hierarchy.dofs[0][l + 1]) - 1);

  std::string prefix = "decomp_";
  if (sizeof(T) == sizeof(double))
    prefix += "d_";
  if (sizeof(T) == sizeof(float))
    prefix += "f_";
  for (int d = 0; d < D; d++)
    prefix += std::to_string(hierarchy.shape[d]) + "_";

  dinput.project(0, 1, 2);
  doutput.project(0, 1, 2);

  SIZE f = hierarchy.dofs[0][l];
  SIZE c = hierarchy.dofs[1][l];
  SIZE r = hierarchy.dofs[2][l];
  SIZE ff = hierarchy.dofs[0][l + 1];
  SIZE cc = hierarchy.dofs[1][l + 1];
  SIZE rr = hierarchy.dofs[2][l + 1];

  SubArray<D, T, DeviceType> dcoarse = dinput;
  dcoarse.resize({ff, cc, rr});
  SubArray<D, T, DeviceType> dcoeff_f = dinput;
  dcoeff_f.offset({ff, 0, 0});
  dcoeff_f.resize({f - ff, cc, rr});
  SubArray<D, T, DeviceType> dcoeff_c = dinput;
  dcoeff_c.offset({0, cc, 0});
  dcoeff_c.resize({ff, c - cc, rr});
  SubArray<D, T, DeviceType> dcoeff_r = dinput;
  dcoeff_r.offset({0, 0, rr});
  dcoeff_r.resize({ff, cc, r - rr});
  SubArray<D, T, DeviceType> dcoeff_cf = dinput;
  dcoeff_cf.offset({ff, cc, 0});
  dcoeff_cf.resize({f - ff, c - cc, rr});
  SubArray<D, T, DeviceType> dcoeff_rf = dinput;
  dcoeff_rf.offset({ff, 0, rr});
  dcoeff_rf.resize({f - ff, cc, r - rr});
  SubArray<D, T, DeviceType> dcoeff_rc = dinput;
  dcoeff_rc.offset({0, cc, rr});
  dcoeff_rc.resize({ff, c - cc, r - rr});
  SubArray<D, T, DeviceType> dcoeff_rcf = dinput;
  dcoeff_rcf.offset({ff, cc, rr});
  dcoeff_rcf.resize({f - ff, c - cc, r - rr});

  SubArray<D, T, DeviceType> dcoarse_new = doutput;
  dcoarse_new.resize({ff, cc, rr});
  SubArray<D, T, DeviceType> dcoeff_f_new = doutput;
  dcoeff_f_new.offset({ff, 0, 0});
  dcoeff_f_new.resize({f - ff, cc, rr});
  SubArray<D, T, DeviceType> dcoeff_c_new = doutput;
  dcoeff_c_new.offset({0, cc, 0});
  dcoeff_c_new.resize({ff, c - cc, rr});
  SubArray<D, T, DeviceType> dcoeff_r_new = doutput;
  dcoeff_r_new.offset({0, 0, rr});
  dcoeff_r_new.resize({ff, cc, r - rr});
  SubArray<D, T, DeviceType> dcoeff_cf_new = doutput;
  dcoeff_cf_new.offset({ff, cc, 0});
  dcoeff_cf_new.resize({f - ff, c - cc, rr});
  SubArray<D, T, DeviceType> dcoeff_rf_new = doutput;
  dcoeff_rf_new.offset({ff, 0, rr});
  dcoeff_rf_new.resize({f - ff, cc, r - rr});
  SubArray<D, T, DeviceType> dcoeff_rc_new = doutput;
  dcoeff_rc_new.offset({0, cc, rr});
  dcoeff_rc_new.resize({ff, c - cc, r - rr});
  SubArray<D, T, DeviceType> dcoeff_rcf_new = doutput;
  dcoeff_rcf_new.offset({ff, cc, rr});
  dcoeff_rcf_new.resize({f - ff, c - cc, r - rr});


  CoefficientRetrieverKernel<D, T, DeviceType>().Execute(
      hierarchy.dofs[2][l], hierarchy.dofs[1][l], hierarchy.dofs[0][l],
      hierarchy.dofs[2][l + 1], hierarchy.dofs[1][l + 1],
      hierarchy.dofs[0][l + 1], 
      dcoarse, dcoeff_f, dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf, dcoeff_rc, dcoeff_rcf, 
      refinement_flag,
      dcoarse_new, dcoeff_f_new, dcoeff_c_new, dcoeff_r_new, dcoeff_cf_new, dcoeff_rf_new, dcoeff_rc_new, dcoeff_rcf_new, 
      queue_idx);

  verify_matrix_cuda(hierarchy.dofs[2][l], hierarchy.dofs[1][l],
                     hierarchy.dofs[0][l], doutput.data(), doutput.getLd(0),
                     doutput.getLd(1), doutput.getLd(0),
                     prefix + "gpk_rev_3d" + "_level_" + std::to_string(l),
                     multidim_refactoring_store, multidim_refactoring_verify);

  if (multidim_refactoring_debug_print) {
    PrintSubarray("after coeff-restore", doutput);
  }
}

} // namespace mgard_x

#endif