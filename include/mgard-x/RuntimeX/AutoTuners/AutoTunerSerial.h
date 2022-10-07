/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_AUTOTUNER_SERIAL_H
#define MGARD_X_AUTOTUNER_SERIAL_H
// clang-format off
namespace mgard_x {

template <> class AutoTuningTable<SERIAL> {
public:
  static constexpr int num_precision = 2;
  static constexpr int num_range = 9;

  static constexpr int gpk_reo_3d[num_precision][num_range] = {{5, 5, 5, 3, 3, 5, 5, 0, 0},
                                                                {3, 6, 5, 3, 3, 3, 5, 0, 0}};

  static constexpr int gpk_rev_3d[num_precision][num_range] = {{2, 4, 5, 5, 3, 5, 5, 0, 0},
                                                              {3, 6, 6, 5, 3, 5, 6, 0, 0}};

  static constexpr int gpk_reo_nd[num_precision][num_range] = {{5, 5, 5, 5, 5, 5, 5, 5, 5},
                                                              {5, 5, 5, 5, 5, 5, 5, 5, 5}};

  static constexpr int gpk_rev_nd[num_precision][num_range] = {{0, 0, 3, 4, 3, 0, 0, 0, 0},
                                                              {0, 0, 3, 4, 5, 0, 0, 0, 0}};

  static constexpr int lpk1_3d[num_precision][num_range] = {{4, 4, 1, 1, 1, 1, 1, 0, 0},
                                                            {1, 1, 1, 1, 1, 1, 1, 0, 0}};

  static constexpr int lpk2_3d[num_precision][num_range] = {{5, 4, 4, 4, 3, 3, 4, 0, 0},
                                                            {4, 1, 1, 1, 1, 1, 3, 0, 0}};

  static constexpr int lpk3_3d[num_precision][num_range] = {{4, 4, 3, 3, 2, 3, 4, 0, 0},
                                                            {1, 1, 1, 1, 1, 1, 2, 0, 0}};

  static constexpr int lpk1_nd[num_precision][num_range] = {{2, 0, 1, 1, 1, 0, 0, 0, 0},
                                                            {0, 0, 1, 1, 1, 0, 0, 0, 0}};

  static constexpr int lpk2_nd[num_precision][num_range] = {{2, 1, 3, 1, 0, 0, 0, 0, 0},
                                                            {0, 2, 1, 1, 0, 0, 0, 0, 0}};

  static constexpr int lpk3_nd[num_precision][num_range] = {{2, 3, 1, 1, 0, 0, 0, 0, 0},
                                                            {0, 2, 1, 1, 0, 0, 0, 0, 0}};

  static constexpr int ipk1_3d[num_precision][num_range] = {{3, 3, 4, 5, 5, 3, 4, 0, 0},
                                                            {3, 6, 4, 4, 3, 3, 3, 0, 0}};

  static constexpr int ipk2_3d[num_precision][num_range] = {{3, 3, 2, 2, 2, 2, 6, 0, 0},
                                                            {2, 2, 2, 2, 2, 2, 5, 0, 0}};

  static constexpr int ipk3_3d[num_precision][num_range] = {{3, 3, 2, 2, 2, 2, 1, 0, 0},
                                                            {2, 2, 2, 2, 2, 2, 6, 0, 0}};

  static constexpr int ipk1_nd[num_precision][num_range] = {{0, 2, 3, 3, 0, 0, 0, 0, 0},
                                                            {0, 3, 3, 3, 0, 0, 0, 0, 0}};

  static constexpr int ipk2_nd[num_precision][num_range] = {{0, 1, 2, 2, 0, 0, 0, 0, 0},
                                                            {0, 2, 2, 2, 0, 0, 0, 0, 0}};

  static constexpr int ipk3_nd[num_precision][num_range] = {{0, 2, 3, 2, 0, 0, 0, 0, 0},
                                                            {0, 3, 4, 2, 0, 0, 0, 0, 0}};

  static constexpr int lwpk[num_precision][num_range] = {{5, 2, 2, 1, 0, 2, 1, 0, 0},
                                                        {2, 2, 3, 3, 0, 2, 5, 0, 0}};

  static constexpr int lwqzk[num_precision][num_range] = {{0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                          {0, 0, 0, 0, 0, 0, 0, 0, 0}};

  static constexpr int lwdqzk[num_precision][num_range] = {{0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                          {0, 0, 0, 0, 0, 0, 0, 0, 0}};

  static constexpr int llk[num_precision][num_range] = {{0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                        {0, 0, 0, 0, 0, 0, 0, 0, 0}};

  static constexpr int sdck[num_precision][num_range] = {{0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                        {0, 0, 0, 0, 0, 0, 0, 0, 0}};

  static constexpr int sdmtk[num_precision][num_range] = {{0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                        {0, 0, 0, 0, 0, 0, 0, 0, 0}};
};

template <> class AutoTuner<SERIAL> {
public:
  MGARDX_CONT
  AutoTuner(){};
  static AutoTuningTable<SERIAL> autoTuningTable;
  static bool ProfileKernels;
};

} // namespace mgard_x
// clang-format on
#endif