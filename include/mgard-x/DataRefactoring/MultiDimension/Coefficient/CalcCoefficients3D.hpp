/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../../../Hierarchy/Hierarchy.hpp"
#include "../../../RuntimeX/RuntimeX.h"

#include "../DataRefactoring.h"

#include "GridProcessingKernel3D.hpp"

#ifndef MGARD_X_DATA_REFACTORING_CALC_COEFFICIENTS_3D
#define MGARD_X_DATA_REFACTORING_CALC_COEFFICIENTS_3D

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
void CalcCoefficients3D(Hierarchy<D, T, DeviceType> &hierarchy,
                        SubArray<D, T, DeviceType> dinput,
                        SubArray<D, T, DeviceType> &doutput, SIZE l,
                        int queue_idx) {

  // int range_l = std::min(6, (int)std::log2(hierarchy.dofs[0][l]) - 1);
  // int range_lp1 = std::min(6, (int)std::log2(hierarchy.dofs[0][l + 1]) - 1);

  std::string prefix = "decomp_";
  if (sizeof(T) == sizeof(double))
    prefix += "d_";
  if (sizeof(T) == sizeof(float))
    prefix += "f_";
  for (int d = 0; d < D; d++)
    prefix += std::to_string(hierarchy.shape[d]) + "_";

  // dinput.project(0, 1, 2);
  // doutput.project(0, 1, 2);
  dinput.project2(D-3, D-2, D-1);
  doutput.project2(D-3, D-2, D-1);
  SIZE f = 1, c = 1, r = 1, ff = 1, cc = 1, rr = 1;
  // f = hierarchy.dofs[0][l];
  // c = hierarchy.dofs[1][l];
  // r = hierarchy.dofs[2][l];
  // ff = hierarchy.dofs[0][l + 1];
  // cc = hierarchy.dofs[1][l + 1];
  // rr = hierarchy.dofs[2][l + 1];
  // printf("%u %u %u %u %u %u\n", f, c, r, ff, cc, ff);

  f = hierarchy.level_shape(hierarchy.l_target-l, D-1);
  c = hierarchy.level_shape(hierarchy.l_target-l, D-2);
  r = hierarchy.level_shape(hierarchy.l_target-l, D-3);
  ff = hierarchy.level_shape(hierarchy.l_target-l-1, D-1);
  cc = hierarchy.level_shape(hierarchy.l_target-l-1, D-2);
  rr = hierarchy.level_shape(hierarchy.l_target-l-1, D-3);
  // printf("%u %u %u %u %u %u\n", f, c, r, ff, cc, ff);



  SubArray<D, T, DeviceType> dcoarse = doutput;
  // dcoarse.resize({ff, cc, rr});
  dcoarse.resize2({rr, cc, ff});
  SubArray<D, T, DeviceType> dcoeff_f = doutput;
  // dcoeff_f.offset({ff, 0, 0});
  // dcoeff_f.resize({f - ff, cc, rr});
  dcoeff_f.offset2({0, 0, ff});
  dcoeff_f.resize2({rr, cc, f - ff});
  SubArray<D, T, DeviceType> dcoeff_c = doutput;
  // dcoeff_c.offset({0, cc, 0});
  // dcoeff_c.resize({ff, c - cc, rr});
  dcoeff_c.offset2({0, cc, 0});
  dcoeff_c.resize2({rr, c - cc, ff});
  SubArray<D, T, DeviceType> dcoeff_r = doutput;
  // dcoeff_r.offset({0, 0, rr});
  // dcoeff_r.resize({ff, cc, r - rr});
  dcoeff_r.offset2({rr, 0, 0});
  dcoeff_r.resize2({r - rr, cc, ff});
  SubArray<D, T, DeviceType> dcoeff_cf = doutput;
  // dcoeff_cf.offset({ff, cc, 0});
  // dcoeff_cf.resize({f - ff, c - cc, rr});
  dcoeff_cf.offset2({0, cc, ff});
  dcoeff_cf.resize2({rr, c - cc, f - ff});
  SubArray<D, T, DeviceType> dcoeff_rf = doutput;
  // dcoeff_rf.offset({ff, 0, rr});
  // dcoeff_rf.resize({f - ff, cc, r - rr});
  dcoeff_rf.offset2({rr, 0, ff});
  dcoeff_rf.resize2({r - rr, cc, f - ff});
  SubArray<D, T, DeviceType> dcoeff_rc = doutput;
  // dcoeff_rc.offset({0, cc, rr});
  // dcoeff_rc.resize({ff, c - cc, r - rr});
  dcoeff_rc.offset2({rr, cc, 0});
  dcoeff_rc.resize2({r - rr, c - cc, ff});
  SubArray<D, T, DeviceType> dcoeff_rcf = doutput;
  // dcoeff_rcf.offset({ff, cc, rr});
  // dcoeff_rcf.resize({f - ff, c - cc, r - rr});
  dcoeff_rcf.offset2({rr, cc, ff});
  dcoeff_rcf.resize2({r - rr, c - cc, f - ff});

  SubArray ratio_f(hierarchy.ratio(hierarchy.l_target-l, D-1));
  SubArray ratio_c(hierarchy.ratio(hierarchy.l_target-l, D-2));
  SubArray ratio_r(hierarchy.ratio(hierarchy.l_target-l, D-3));

  GpkReo3D<D, T, DeviceType>().Execute(r, c, f, rr, cc, ff,
      ratio_r, ratio_c, ratio_f, dinput, dcoarse, dcoeff_f,
      dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf, dcoeff_rc, dcoeff_rcf,
      queue_idx);

  verify_matrix_cuda(r, c, f, doutput.data(), doutput.getLd(0),
                     doutput.getLd(1), doutput.getLd(0),
                     prefix + "gpk_reo_3d" + "_level_" + std::to_string(l),
                     multidim_refactoring_store, multidim_refactoring_verify);

  if (multidim_refactoring_debug_print) {
    PrintSubarray("after pi_Ql_reo", doutput);
  }
}

} // namespace mgard_x

#endif