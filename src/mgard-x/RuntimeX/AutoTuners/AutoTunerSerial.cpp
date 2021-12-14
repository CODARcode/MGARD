#define MGARDX_COMPILE_SERIAL

#include "mgard-x/RuntimeX/RuntimeX.h"

namespace mgard_x {

int AutoTuningTable<Serial>::gpk_reo_3d[2][9] = {
                    {0, 0, 1, 0, 0, 0, 0, 0, 0},
                    {0, 1, 0, 0, 0, 0, 0, 0, 0}
                  };

int AutoTuningTable<Serial>::gpk_rev_3d[2][9] = {
                    {1, 0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0, 0}
                  };

int AutoTuningTable<Serial>::gpk_reo_nd[2][9] = {
                    {0, 0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 3, 4, 0, 0, 0, 0, 0}
                  };

int AutoTuningTable<Serial>::gpk_rev_nd[2][9] = {
                    {0, 0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 3, 4, 5, 0, 0, 0, 0}
                  };

int AutoTuningTable<Serial>::lpk1_3d[2][9] = {
                    {0, 0, 0, 0, 0, 0, 1, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0, 0}
                  };

int AutoTuningTable<Serial>::lpk2_3d[2][9] = {
                    {1, 0, 0, 0, 0, 0, 1, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0, 0}
                  };

int AutoTuningTable<Serial>::lpk3_3d[2][9] = {
                    {1, 0, 0, 0, 0, 0, 1, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0, 0}
                  };

int AutoTuningTable<Serial>::lpk1_nd[2][9] = {
                    {2, 0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 1, 1, 1, 0, 0, 0, 0}
                  };

int AutoTuningTable<Serial>::lpk2_nd[2][9] = {
                    {2, 0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 2, 1, 1, 0, 0, 0, 0, 0}
                  };

int AutoTuningTable<Serial>::lpk3_nd[2][9] = {
                    {2, 0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 2, 1, 1, 0, 0, 0, 0, 0}
                  };


int AutoTuningTable<Serial>::ipk1_3d[2][9] = {
                    {0, 2, 1, 2, 2, 2, 2, 0, 0},
                    {1, 0, 1, 2, 2, 2, 2, 0, 0}
                  };

int AutoTuningTable<Serial>::ipk2_3d[2][9] = {
                    {0, 2, 1, 2, 2, 2, 2, 0, 0},
                    {1, 2, 1, 2, 2, 2, 2, 0, 0}
                  };

int AutoTuningTable<Serial>::ipk3_3d[2][9] = {
                    {0, 2, 2, 2, 2, 2, 2, 0, 0},
                    {0, 2, 1, 2, 2, 2, 2, 0, 0}
                  };

int AutoTuningTable<Serial>::ipk1_nd[2][9] = {
                    {0, 2, 1, 2, 0, 0, 0, 0, 0},
                    {0, 3, 3, 3, 0, 0, 0, 0, 0}
                  };

int AutoTuningTable<Serial>::ipk2_nd[2][9] = {
                    {0, 2, 2, 2, 0, 0, 0, 0, 0},
                    {0, 2, 2, 2, 0, 0, 0, 0, 0}
                  };

int AutoTuningTable<Serial>::ipk3_nd[2][9] = {
                    {0, 0, 1, 2, 0, 0, 0, 0, 0},
                    {0, 3, 4, 2, 0, 0, 0, 0, 0}
                  };

int AutoTuningTable<Serial>::lwpk[2][9] = {
                    {0, 0, 0, 0, 0, 0, 2, 0, 0},
                    {0, 1, 0, 0, 0, 0, 0, 0, 0}
                  };

}

#undef MGARDX_COMPILE_SERIAL