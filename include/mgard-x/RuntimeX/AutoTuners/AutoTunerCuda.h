/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_AUTOTUNER_CUDA_H
#define MGARD_X_AUTOTUNER_CUDA_H
// clang-format off
namespace mgard_x {

template <> class AutoTuningTable<CUDA> {
public:
  static constexpr int num_types = 2;
  static constexpr int num_dims = 9;

  static constexpr int gpk_reo_3d[num_types][num_dims] = {{2, 3, 5, 0, 0, 0, 2, 0, 0},
                                                          {2, 3, 2, 0, 0, 0, 3, 0, 0}};

  static constexpr int gpk_rev_3d[num_types][num_dims] = {{3, 5, 6, 3, 3, 5, 6, 0, 0},
                                                          {2, 5, 3, 3, 4, 5, 6, 0, 0}};

  static constexpr int gpk_reo_nd[num_types][num_dims] = {{5, 3, 5, 5, 5, 5, 5, 5, 5},
                                                          {5, 3, 5, 5, 5, 5, 5, 5, 5}};

  static constexpr int gpk_rev_nd[num_types][num_dims] = {{0, 5, 3, 4, 3, 0, 0, 0, 0},
                                                          {0, 5, 3, 4, 5, 0, 0, 0, 0}};

  static constexpr int lpk1_3d[num_types][num_dims] = {{3, 5, 3, 0, 1, 1, 4, 0, 0},
                                                        {4, 3, 1, 1, 1, 1, 1, 0, 0}};

  static constexpr int lpk2_3d[num_types][num_dims] = {{4, 4, 3, 4, 1, 3, 4, 0, 0},
                                                        {0, 4, 2, 1, 1, 1, 1, 0, 0}};

  static constexpr int lpk3_3d[num_types][num_dims] = {{1, 4, 2, 1, 1, 1, 4, 0, 0},
                                                        {1, 0, 1, 1, 1, 1, 1, 0, 0}};

  static constexpr int lpk1_nd[num_types][num_dims] = {{2, 0, 1, 1, 1, 0, 0, 0, 0},
                                                        {0, 0, 1, 1, 1, 0, 0, 0, 0}};

  static constexpr int lpk2_nd[num_types][num_dims] = {{2, 1, 3, 1, 1, 0, 0, 0, 0},
                                                        {0, 2, 1, 1, 1, 0, 0, 0, 0}};

  static constexpr int lpk3_nd[num_types][num_dims] = {{2, 3, 1, 1, 1, 0, 0, 0, 0},
                                                        {0, 2, 1, 1, 2, 0, 0, 0, 0}};

  static constexpr int ipk1_3d[num_types][num_dims] = {{6, 4, 4, 4, 5, 3, 4, 0, 0},
                                                        {5, 3, 2, 3, 3, 1, 1, 0, 0}};

  static constexpr int ipk2_3d[num_types][num_dims] = {{2, 3, 6, 2, 2, 2, 6, 0, 0},
                                                        {3, 4, 5, 2, 2, 3, 4, 0, 0}};

  static constexpr int ipk3_3d[num_types][num_dims] = {{2, 2, 6, 2, 2, 2, 6, 0, 0},
                                                        {2, 2, 3, 2, 2, 2, 4, 0, 0}};

  static constexpr int ipk1_nd[num_types][num_dims] = {{0, 2, 3, 2, 1, 0, 0, 0, 0},
                                                       {0, 3, 3, 2, 1, 0, 0, 0, 0}};

  static constexpr int ipk2_nd[num_types][num_dims] = {{0, 1, 2, 2, 2, 0, 0, 0, 0},
                                                        {0, 2, 2, 2, 2, 0, 0, 0, 0}};

  static constexpr int ipk3_nd[num_types][num_dims] = {{0, 2, 3, 2, 2, 0, 0, 0, 0},
                                                       {0, 3, 4, 2, 4, 0, 0, 0, 0}};

  static constexpr int lwpk[num_types][num_dims] = {{3, 5, 5, 5, 2, 1, 2, 0, 0},
                                                    {3, 6, 4, 1, 2, 1, 1, 0, 0}};

  static constexpr int lwqzk[num_types][num_dims] = {{4, 4, 3, 3, 3, 0, 2, 0, 0},
                                                      {2, 3, 3, 3, 3, 0, 2, 0, 0}};

  static constexpr int lwdqzk[num_types][num_dims] = {{0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                      {0, 0, 0, 0, 0, 0, 0, 0, 0}};

  static constexpr int llk[num_types][num_dims] = {{0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                    {0, 0, 0, 0, 0, 0, 0, 0, 0}};

  static constexpr int sdck[num_types][num_dims] = {{0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                    {0, 0, 0, 0, 0, 0, 0, 0, 0}};

  static constexpr int sdmtk[num_types][num_dims] = {{0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                    {0, 0, 0, 0, 0, 0, 0, 0, 0}};

  static constexpr int encode[num_types][num_dims] = {{4, 4, 4, 4, 4, 4, 4, 4, 4},
                                                    {4, 4, 4, 4, 4, 4, 4, 4, 4}};

  static constexpr int deflate[num_types][num_dims] = {{0, 3, 3, 3, 3, 3, 3, 3, 3},
                                                    {3, 3, 3, 3, 3, 3, 3, 3, 3}};

  static constexpr int decode[num_types][num_dims] = {{0, 3, 3, 3, 3, 3, 3, 3, 3},
                                                    {3, 3, 3, 3, 3, 3, 3, 3, 3}};

};

template <> class AutoTuner<CUDA> {
public:
  MGARDX_CONT
  AutoTuner(){};
  static AutoTuningTable<CUDA> autoTuningTable;
  static bool ProfileKernels;
  static bool WriteToTable;
};

} // namespace mgard_x
// clang-format on
#endif