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

  std::string prefix = "decomp_";
  if (sizeof(T) == sizeof(double))
    prefix += "d_";
  if (sizeof(T) == sizeof(float))
    prefix += "f_";
  for (int d = 0; d < D; d++)
    prefix +=
        std::to_string(hierarchy.level_shape(hierarchy.l_target(), d)) + "_";

  dinput.project(D - 3, D - 2, D - 1);
  doutput.project(D - 3, D - 2, D - 1);
  SIZE f, c, r, ff, cc, rr;

  f = hierarchy.level_shape(l, D - 1);
  c = hierarchy.level_shape(l, D - 2);
  r = hierarchy.level_shape(l, D - 3);
  ff = hierarchy.level_shape(l - 1, D - 1);
  cc = hierarchy.level_shape(l - 1, D - 2);
  rr = hierarchy.level_shape(l - 1, D - 3);

  SubArray<D, T, DeviceType> dcoarse = doutput;
  dcoarse.resize({rr, cc, ff});
  SubArray<D, T, DeviceType> dcoeff_f = doutput;
  dcoeff_f.offset({0, 0, ff});
  dcoeff_f.resize({rr, cc, f - ff});
  SubArray<D, T, DeviceType> dcoeff_c = doutput;
  dcoeff_c.offset({0, cc, 0});
  dcoeff_c.resize({rr, c - cc, ff});
  SubArray<D, T, DeviceType> dcoeff_r = doutput;
  dcoeff_r.offset({rr, 0, 0});
  dcoeff_r.resize({r - rr, cc, ff});
  SubArray<D, T, DeviceType> dcoeff_cf = doutput;
  dcoeff_cf.offset({0, cc, ff});
  dcoeff_cf.resize({rr, c - cc, f - ff});
  SubArray<D, T, DeviceType> dcoeff_rf = doutput;
  dcoeff_rf.offset({rr, 0, ff});
  dcoeff_rf.resize({r - rr, cc, f - ff});
  SubArray<D, T, DeviceType> dcoeff_rc = doutput;
  dcoeff_rc.offset({rr, cc, 0});
  dcoeff_rc.resize({r - rr, c - cc, ff});
  SubArray<D, T, DeviceType> dcoeff_rcf = doutput;
  dcoeff_rcf.offset({rr, cc, ff});
  dcoeff_rcf.resize({r - rr, c - cc, f - ff});

  SubArray ratio_f(hierarchy.ratio(l, D - 1));
  SubArray ratio_c(hierarchy.ratio(l, D - 2));
  SubArray ratio_r(hierarchy.ratio(l, D - 3));

  GpkReo3D<D, T, DeviceType>().Execute(r, c, f, rr, cc, ff, ratio_r, ratio_c,
                                       ratio_f, dinput, dcoarse, dcoeff_f,
                                       dcoeff_c, dcoeff_r, dcoeff_cf, dcoeff_rf,
                                       dcoeff_rc, dcoeff_rcf, queue_idx);

  verify_matrix_cuda(r, c, f, doutput.data(), doutput.ld(D - 1),
                     doutput.ld(D - 2), doutput.ld(D - 1),
                     prefix + "gpk_reo_3d" + "_level_" + std::to_string(l),
                     multidim_refactoring_store, multidim_refactoring_verify);

  if (multidim_refactoring_debug_print) {
    PrintSubarray("after pi_Ql_reo", doutput);
  }
}

} // namespace mgard_x

#endif