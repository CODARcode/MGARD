/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../../../Hierarchy/Hierarchy.h"
#include "../../../RuntimeX/RuntimeX.h"

#ifndef MGARD_X_DATA_REFACTORING_CALC_COEFFICIENT_POINTERS
#define MGARD_X_DATA_REFACTORING_CALC_COEFFICIENT_POINTERS

namespace mgard_x {

namespace data_refactoring {

namespace multi_dimension {

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
    n[d] = hierarchy.level_shape(l, curr_dims[d]);
    nn[d] = hierarchy.level_shape(l - 1, curr_dims[d]);
  }

  dcoarse = doutput;
  dcoarse.resize(curr_dims[0], nn[0]);
  dcoarse.resize(curr_dims[1], nn[1]);
  dcoarse.resize(curr_dims[2], nn[2]);

  dcoeff_r = doutput;
  dcoeff_r.offset_dim(curr_dims[0], nn[0]);
  dcoeff_r.resize(curr_dims[0], n[0] - nn[0]);
  dcoeff_r.resize(curr_dims[1], nn[1]);
  dcoeff_r.resize(curr_dims[2], nn[2]);

  dcoeff_c = doutput;
  dcoeff_c.offset_dim(curr_dims[1], nn[1]);
  dcoeff_c.resize(curr_dims[0], nn[0]);
  dcoeff_c.resize(curr_dims[1], n[1] - nn[1]);
  dcoeff_c.resize(curr_dims[2], nn[2]);

  dcoeff_f = doutput;
  dcoeff_f.offset_dim(curr_dims[2], nn[2]);
  dcoeff_f.resize(curr_dims[0], nn[0]);
  dcoeff_f.resize(curr_dims[1], nn[1]);
  dcoeff_f.resize(curr_dims[2], n[2] - nn[2]);

  dcoeff_rc = doutput;
  dcoeff_rc.offset_dim(curr_dims[0], nn[0]);
  dcoeff_rc.offset_dim(curr_dims[1], nn[1]);
  dcoeff_rc.resize(curr_dims[0], n[0] - nn[0]);
  dcoeff_rc.resize(curr_dims[1], n[1] - nn[1]);
  dcoeff_rc.resize(curr_dims[2], nn[2]);

  dcoeff_rf = doutput;
  dcoeff_rf.offset_dim(curr_dims[0], nn[0]);
  dcoeff_rf.offset_dim(curr_dims[2], nn[2]);
  dcoeff_rf.resize(curr_dims[0], n[0] - nn[0]);
  dcoeff_rf.resize(curr_dims[1], nn[1]);
  dcoeff_rf.resize(curr_dims[2], n[2] - nn[2]);

  dcoeff_cf = doutput;
  dcoeff_cf.offset_dim(curr_dims[1], nn[1]);
  dcoeff_cf.offset_dim(curr_dims[2], nn[2]);
  dcoeff_cf.resize(curr_dims[0], nn[0]);
  dcoeff_cf.resize(curr_dims[1], n[1] - nn[1]);
  dcoeff_cf.resize(curr_dims[2], n[2] - nn[2]);

  dcoeff_rcf = doutput;
  dcoeff_rcf.offset_dim(curr_dims[0], nn[0]);
  dcoeff_rcf.offset_dim(curr_dims[1], nn[1]);
  dcoeff_rcf.offset_dim(curr_dims[2], nn[2]);
  dcoeff_rcf.resize(curr_dims[0], n[0] - nn[0]);
  dcoeff_rcf.resize(curr_dims[1], n[1] - nn[1]);
  dcoeff_rcf.resize(curr_dims[2], n[2] - nn[2]);
}

} // namespace multi_dimension

} // namespace data_refactoring

} // namespace mgard_x

#endif
