#define MGARDX_COMPILE_SERIAL

#include "mgard-x/RuntimeX/RuntimeX.h"

namespace mgard_x {

int AutoTuningTable<Serial>::gpk_reo_3d[2][9] = {{5, 5, 5, 3, 3, 5, 5, 0, 0},
                                                 {3, 6, 5, 3, 3, 3, 5, 0, 0}};

int AutoTuningTable<Serial>::gpk_rev_3d[2][9] = {{2, 4, 5, 5, 3, 5, 5, 0, 0},
                                                 {3, 6, 6, 5, 3, 5, 6, 0, 0}};

int AutoTuningTable<Serial>::gpk_reo_nd[2][9] = {{0, 0, 3, 4, 3, 0, 0, 0, 0},
                                                 {0, 0, 3, 4, 5, 0, 0, 0, 0}};

int AutoTuningTable<Serial>::gpk_rev_nd[2][9] = {{0, 0, 3, 4, 3, 0, 0, 0, 0},
                                                 {0, 0, 3, 4, 5, 0, 0, 0, 0}};

int AutoTuningTable<Serial>::lpk1_3d[2][9] = {{4, 4, 1, 1, 1, 1, 1, 0, 0},
                                              {1, 1, 1, 1, 1, 1, 1, 0, 0}};

int AutoTuningTable<Serial>::lpk2_3d[2][9] = {{5, 4, 4, 4, 3, 3, 4, 0, 0},
                                              {4, 1, 1, 1, 1, 1, 3, 0, 0}};

int AutoTuningTable<Serial>::lpk3_3d[2][9] = {{4, 4, 3, 3, 2, 3, 4, 0, 0},
                                              {1, 1, 1, 1, 1, 1, 2, 0, 0}};

int AutoTuningTable<Serial>::lpk1_nd[2][9] = {{2, 0, 1, 1, 1, 0, 0, 0, 0},
                                              {0, 0, 1, 1, 1, 0, 0, 0, 0}};

int AutoTuningTable<Serial>::lpk2_nd[2][9] = {{2, 1, 3, 1, 0, 0, 0, 0, 0},
                                              {0, 2, 1, 1, 0, 0, 0, 0, 0}};

int AutoTuningTable<Serial>::lpk3_nd[2][9] = {{2, 3, 1, 1, 0, 0, 0, 0, 0},
                                              {0, 2, 1, 1, 0, 0, 0, 0, 0}};

int AutoTuningTable<Serial>::ipk1_3d[2][9] = {{3, 3, 4, 5, 5, 3, 4, 0, 0},
                                              {3, 6, 4, 4, 3, 3, 3, 0, 0}};

int AutoTuningTable<Serial>::ipk2_3d[2][9] = {{3, 3, 2, 2, 2, 2, 6, 0, 0},
                                              {2, 2, 2, 2, 2, 2, 5, 0, 0}};

int AutoTuningTable<Serial>::ipk3_3d[2][9] = {{3, 3, 2, 2, 2, 2, 1, 0, 0},
                                              {2, 2, 2, 2, 2, 2, 6, 0, 0}};

int AutoTuningTable<Serial>::ipk1_nd[2][9] = {{0, 2, 3, 3, 0, 0, 0, 0, 0},
                                              {0, 3, 3, 3, 0, 0, 0, 0, 0}};

int AutoTuningTable<Serial>::ipk2_nd[2][9] = {{0, 1, 2, 2, 0, 0, 0, 0, 0},
                                              {0, 2, 2, 2, 0, 0, 0, 0, 0}};

int AutoTuningTable<Serial>::ipk3_nd[2][9] = {{0, 2, 3, 2, 0, 0, 0, 0, 0},
                                              {0, 3, 4, 2, 0, 0, 0, 0, 0}};

int AutoTuningTable<Serial>::lwpk[2][9] = {{5, 2, 2, 1, 0, 2, 1, 0, 0},
                                           {2, 2, 3, 3, 0, 2, 5, 0, 0}};

} // namespace mgard_x

#undef MGARDX_COMPILE_SERIAL