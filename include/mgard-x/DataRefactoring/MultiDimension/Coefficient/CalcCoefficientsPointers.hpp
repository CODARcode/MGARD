/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../../../Hierarchy/Hierarchy.hpp"
#include "../../../RuntimeX/RuntimeX.h"

#ifndef MGARD_X_DATA_REFACTORING_CALC_COEFFICIENT_POINTERS
#define MGARD_X_DATA_REFACTORING_CALC_COEFFICIENT_POINTERS

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
void CalcCoefficientsPointers(
    Hierarchy<D, T, DeviceType> &hierarchy, DIM curr_dims[3], DIM l,
    SubArray<D, T, DeviceType> doutput, SubArray<D, T, DeviceType> &dcoarse,
    SubArray<D, T, DeviceType> &dcoeff_f, SubArray<D, T, DeviceType> &dcoeff_c,
    SubArray<D, T, DeviceType> &dcoeff_r, SubArray<D, T, DeviceType> &dcoeff_cf,
    SubArray<D, T, DeviceType> &dcoeff_rf,
    SubArray<D, T, DeviceType> &dcoeff_rc,
    SubArray<D, T, DeviceType> &dcoeff_rcf) {

  SIZE n[3];
  SIZE nn[3];
  for (DIM d = 0; d < 3; d++) {
    n[d] = hierarchy.dofs[curr_dims[d]][l];
    nn[d] = hierarchy.dofs[curr_dims[d]][l + 1];
  }

  dcoarse = doutput;
  dcoarse.resize(curr_dims[0], nn[0]);
  dcoarse.resize(curr_dims[1], nn[1]);
  dcoarse.resize(curr_dims[2], nn[2]);

  dcoeff_f = doutput;
  dcoeff_f.offset(curr_dims[0], nn[0]);
  dcoeff_f.resize(curr_dims[0], n[0] - nn[0]);
  dcoeff_f.resize(curr_dims[1], nn[1]);
  dcoeff_f.resize(curr_dims[2], nn[2]);

  dcoeff_c = doutput;
  dcoeff_c.offset(curr_dims[1], nn[1]);
  dcoeff_c.resize(curr_dims[0], nn[0]);
  dcoeff_c.resize(curr_dims[1], n[1] - nn[1]);
  dcoeff_c.resize(curr_dims[2], nn[2]);

  dcoeff_r = doutput;
  dcoeff_r.offset(curr_dims[2], nn[2]);
  dcoeff_r.resize(curr_dims[0], nn[0]);
  dcoeff_r.resize(curr_dims[1], nn[1]);
  dcoeff_r.resize(curr_dims[2], n[2] - nn[2]);

  dcoeff_cf = doutput;
  dcoeff_cf.offset(curr_dims[0], nn[0]);
  dcoeff_cf.offset(curr_dims[1], nn[1]);
  dcoeff_cf.resize(curr_dims[0], n[0] - nn[0]);
  dcoeff_cf.resize(curr_dims[1], n[1] - nn[1]);
  dcoeff_cf.resize(curr_dims[2], nn[2]);

  dcoeff_rf = doutput;
  dcoeff_rf.offset(curr_dims[0], nn[0]);
  dcoeff_rf.offset(curr_dims[2], nn[2]);
  dcoeff_rf.resize(curr_dims[0], n[0] - nn[0]);
  dcoeff_rf.resize(curr_dims[1], nn[1]);
  dcoeff_rf.resize(curr_dims[2], n[2] - nn[2]);

  dcoeff_rc = doutput;
  dcoeff_rc.offset(curr_dims[1], nn[1]);
  dcoeff_rc.offset(curr_dims[2], nn[2]);
  dcoeff_rc.resize(curr_dims[0], nn[0]);
  dcoeff_rc.resize(curr_dims[1], n[1] - nn[1]);
  dcoeff_rc.resize(curr_dims[2], n[2] - nn[2]);

  dcoeff_rcf = doutput;
  dcoeff_rcf.offset(curr_dims[0], nn[0]);
  dcoeff_rcf.offset(curr_dims[1], nn[1]);
  dcoeff_rcf.offset(curr_dims[2], nn[2]);
  dcoeff_rcf.resize(curr_dims[0], n[0] - nn[0]);
  dcoeff_rcf.resize(curr_dims[1], n[1] - nn[1]);
  dcoeff_rcf.resize(curr_dims[2], n[2] - nn[2]);
}

} // namespace mgard_x

#endif