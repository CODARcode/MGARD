#include "mgard-x/RuntimeX/RuntimeX.h"
// clang-format off
namespace mgard_x {

int AutoTuningTable<HIP>::gpk_reo_3d[2][9] = {{5, 5, 5, 3, 3, 5, 5, 0, 0},
                                              {3, 6, 5, 3, 3, 3, 5, 0, 0}};

int AutoTuningTable<HIP>::gpk_rev_3d[2][9] = {{2, 4, 5, 5, 3, 5, 5, 0, 0},
                                              {3, 6, 6, 5, 3, 5, 6, 0, 0}};

int AutoTuningTable<HIP>::gpk_reo_nd[2][9] = {{0, 0, 3, 4, 3, 0, 0, 0, 0},
                                              {0, 0, 3, 4, 5, 0, 0, 0, 0}};

int AutoTuningTable<HIP>::gpk_rev_nd[2][9] = {{0, 0, 3, 4, 3, 0, 0, 0, 0},
                                              {0, 0, 3, 4, 5, 0, 0, 0, 0}};

int AutoTuningTable<HIP>::lpk1_3d[2][9] = {{4, 4, 1, 1, 1, 1, 1, 0, 0},
                                           {1, 1, 1, 1, 1, 1, 1, 0, 0}};

int AutoTuningTable<HIP>::lpk2_3d[2][9] = {{5, 4, 4, 4, 3, 3, 4, 0, 0},
                                           {4, 1, 1, 1, 1, 1, 3, 0, 0}};

int AutoTuningTable<HIP>::lpk3_3d[2][9] = {{4, 4, 3, 3, 2, 3, 4, 0, 0},
                                           {1, 1, 1, 1, 1, 1, 2, 0, 0}};

int AutoTuningTable<HIP>::lpk1_nd[2][9] = {{2, 0, 1, 1, 1, 0, 0, 0, 0},
                                           {0, 0, 1, 1, 1, 0, 0, 0, 0}};

int AutoTuningTable<HIP>::lpk2_nd[2][9] = {{2, 1, 3, 1, 0, 0, 0, 0, 0},
                                           {0, 2, 1, 1, 0, 0, 0, 0, 0}};

int AutoTuningTable<HIP>::lpk3_nd[2][9] = {{2, 3, 1, 1, 0, 0, 0, 0, 0},
                                           {0, 2, 1, 1, 0, 0, 0, 0, 0}};

int AutoTuningTable<HIP>::ipk1_3d[2][9] = {{3, 3, 4, 5, 5, 3, 4, 0, 0},
                                           {3, 6, 4, 4, 3, 3, 3, 0, 0}};

int AutoTuningTable<HIP>::ipk2_3d[2][9] = {{3, 3, 2, 2, 2, 2, 6, 0, 0},
                                           {2, 2, 2, 2, 2, 2, 5, 0, 0}};

int AutoTuningTable<HIP>::ipk3_3d[2][9] = {{3, 3, 2, 2, 2, 2, 1, 0, 0},
                                           {2, 2, 2, 2, 2, 2, 6, 0, 0}};

int AutoTuningTable<HIP>::ipk1_nd[2][9] = {{0, 2, 3, 3, 0, 0, 0, 0, 0},
                                           {0, 3, 3, 3, 0, 0, 0, 0, 0}};

int AutoTuningTable<HIP>::ipk2_nd[2][9] = {{0, 1, 2, 2, 0, 0, 0, 0, 0},
                                           {0, 2, 2, 2, 0, 0, 0, 0, 0}};

int AutoTuningTable<HIP>::ipk3_nd[2][9] = {{0, 2, 3, 2, 0, 0, 0, 0, 0},
                                           {0, 3, 4, 2, 0, 0, 0, 0, 0}};

int AutoTuningTable<HIP>::lwpk[2][9] = {{5, 2, 2, 1, 0, 2, 1, 0, 0},
                                        {2, 2, 3, 3, 0, 2, 5, 0, 0}};

int AutoTuningTable<HIP>::lwqzk[2][9] = {{5, 2, 2, 1, 0, 2, 1, 0, 0},
                                         {2, 2, 3, 3, 0, 2, 5, 0, 0}};

int AutoTuningTable<HIP>::lwdqzk[2][9] = {{5, 2, 2, 1, 0, 2, 1, 0, 0},
                                         {2, 2, 3, 3, 0, 2, 5, 0, 0}};  

int AutoTuningTable<HIP>::llk[2][9] = {{5, 2, 2, 1, 0, 2, 1, 0, 0},
                                         {2, 2, 3, 3, 0, 2, 5, 0, 0}};

template void BeginAutoTuning<HIP>();
template void EndAutoTuning<HIP>();

} // namespace mgard_x
// clang-format on
#undef MGARDX_COMPILE_HIP