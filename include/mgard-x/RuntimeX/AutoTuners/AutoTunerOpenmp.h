/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_AUTOTUNER_OPENMP_H
#define MGARD_X_AUTOTUNER_OPENMP_H

namespace mgard_x {

template <> class AutoTuningTable<OPENMP> {
public:
  static const int num_precision = 2;
  static const int num_range = 9;

  static int gpk_reo_3d[num_precision][num_range];

  static int gpk_rev_3d[num_precision][num_range];

  static int gpk_reo_nd[num_precision][num_range];

  static int gpk_rev_nd[num_precision][num_range];

  static int lpk1_3d[num_precision][num_range];

  static int lpk2_3d[num_precision][num_range];

  static int lpk3_3d[num_precision][num_range];

  static int lpk1_nd[num_precision][num_range];

  static int lpk2_nd[num_precision][num_range];

  static int lpk3_nd[num_precision][num_range];

  static int ipk1_3d[num_precision][num_range];

  static int ipk2_3d[num_precision][num_range];

  static int ipk3_3d[num_precision][num_range];

  static int ipk1_nd[num_precision][num_range];

  static int ipk2_nd[num_precision][num_range];

  static int ipk3_nd[num_precision][num_range];

  static int lwpk[num_precision][num_range];

  static int lwqzk[num_precision][num_range];

  static int lwdqzk[num_precision][num_range];

  static int llk[num_precision][num_range];
};

template <> class AutoTuner<OPENMP> {
public:
  MGARDX_CONT
  AutoTuner(){};
  static AutoTuningTable<OPENMP> autoTuningTable;
  static bool ProfileKernels;
};

} // namespace mgard_x

#endif