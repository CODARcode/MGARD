/*
 * Copyright 2023, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: Jan. 15, 2023
 */

namespace mgard_x {
// clang-format off

MGARDX_EXEC constexpr int offset5x5x5(SIZE z, SIZE y, SIZE x) {
  return z * 5 * 5 + y * 5 + x;
}

MGARDX_EXEC constexpr int offset5x5x5(SIZE z, SIZE y, SIZE x, SIZE ld1, SIZE ld2) {
  return z * ld1 * ld2 + y * ld1 + x;
}

static constexpr int c1d_x_5x5x5[18][3] = { {0, 0, 1}, {0, 0, 3}, // X
                                            {0, 2, 1}, {0, 2, 3},
                                            {0, 4, 1}, {0, 4, 3},
                                            {2, 0, 1}, {2, 0, 3},
                                            {2, 2, 1}, {2, 2, 3},
                                            {2, 4, 1}, {2, 4, 3},
                                            {4, 0, 1}, {4, 0, 3},
                                            {4, 2, 1}, {4, 2, 3},
                                            {4, 4, 1}, {4, 4, 3}
                                          };

static constexpr int c1d_y_5x5x5[18][3] = { {0, 1, 0}, {0, 3, 0}, // Y
                                            {0, 1, 2}, {0, 3, 2}, 
                                            {0, 1, 4}, {0, 3, 4}, 
                                            {2, 1, 0}, {2, 3, 0},
                                            {2, 1, 2}, {2, 3, 2}, 
                                            {2, 1, 4}, {2, 3, 4}, 
                                            {4, 1, 0}, {4, 3, 0},
                                            {4, 1, 2}, {4, 3, 2}, 
                                            {4, 1, 4}, {4, 3, 4}
                                            };

static constexpr int c1d_z_5x5x5[18][3] = { {1, 0, 0}, {3, 0, 0}, // Z
                                            {1, 0, 2}, {3, 0, 2},
                                            {1, 0, 4}, {3, 0, 4},
                                            {1, 2, 0}, {3, 2, 0},
                                            {1, 2, 2}, {3, 2, 2},
                                            {1, 2, 4}, {3, 2, 4},
                                            {1, 4, 0}, {3, 4, 0},
                                            {1, 4, 2}, {3, 4, 2},
                                            {1, 4, 4}, {3, 4, 4},
                                            };
static constexpr int c2d_xy_5x5x5[12][3] = {{0, 1, 1}, {0, 1, 3}, // XY
                                            {0, 3, 1}, {0, 3, 3},
                                            {2, 1, 1}, {2, 1, 3},
                                            {2, 3, 1}, {2, 3, 3},
                                            {4, 1, 1}, {4, 1, 3},
                                            {4, 3, 1}, {4, 3, 3}
                                          };

static constexpr int c2d_yz_5x5x5[12][3] = {{1, 1, 0}, {1, 3, 0}, // YZ
                                            {3, 1, 0}, {3, 3, 0},
                                            {1, 1, 2}, {1, 3, 2},
                                            {3, 1, 2}, {3, 3, 2},
                                            {1, 1, 4}, {1, 3, 4},
                                            {3, 1, 4}, {3, 3, 4}
                                          };

static constexpr int c2d_xz_5x5x5[12][3] = {{1, 0, 1}, {1, 0, 3}, // XZ
                                            {3, 0, 1}, {3, 0, 3},
                                            {1, 2, 1}, {1, 2, 3},
                                            {3, 2, 1}, {3, 2, 3},
                                            {1, 4, 1}, {1, 4, 3},
                                            {3, 4, 1}, {3, 4, 3}
                                          };

static constexpr int c3d_xyz_5x5x5[8][3] = { {1, 1, 1}, {1, 1, 3}, // XYZ
                                             {1, 3, 1}, {1, 3, 3},
                                             {3, 1, 1}, {3, 1, 3},
                                             {3, 3, 1}, {3, 3, 3}
                                          };

static constexpr int coarse_x_5x5x5[3] = {0, 2, 4};

static constexpr int coarse_y_5x5x5[3] = {0, 2, 4};

static constexpr int coarse_z_5x5x5[3] = {0, 2, 4};

MGARDX_EXEC int Coeff1D_M_Offset_5x5x5(SIZE i) {
  static constexpr int offset[54] = { offset5x5x5(c1d_x_5x5x5[0][0], c1d_x_5x5x5[0][1], c1d_x_5x5x5[0][2]), // X
                                      offset5x5x5(c1d_x_5x5x5[1][0], c1d_x_5x5x5[1][1], c1d_x_5x5x5[1][2]),
                                      offset5x5x5(c1d_x_5x5x5[2][0], c1d_x_5x5x5[2][1], c1d_x_5x5x5[2][2]),
                                      offset5x5x5(c1d_x_5x5x5[3][0], c1d_x_5x5x5[3][1], c1d_x_5x5x5[3][2]),
                                      offset5x5x5(c1d_x_5x5x5[4][0], c1d_x_5x5x5[4][1], c1d_x_5x5x5[4][2]),
                                      offset5x5x5(c1d_x_5x5x5[5][0], c1d_x_5x5x5[5][1], c1d_x_5x5x5[5][2]),
                                      offset5x5x5(c1d_x_5x5x5[6][0], c1d_x_5x5x5[6][1], c1d_x_5x5x5[6][2]),
                                      offset5x5x5(c1d_x_5x5x5[7][0], c1d_x_5x5x5[7][1], c1d_x_5x5x5[7][2]),
                                      offset5x5x5(c1d_x_5x5x5[8][0], c1d_x_5x5x5[8][1], c1d_x_5x5x5[8][2]),
                                      offset5x5x5(c1d_x_5x5x5[9][0], c1d_x_5x5x5[9][1], c1d_x_5x5x5[9][2]),
                                      offset5x5x5(c1d_x_5x5x5[10][0], c1d_x_5x5x5[10][1], c1d_x_5x5x5[10][2]),
                                      offset5x5x5(c1d_x_5x5x5[11][0], c1d_x_5x5x5[11][1], c1d_x_5x5x5[11][2]),
                                      offset5x5x5(c1d_x_5x5x5[12][0], c1d_x_5x5x5[12][1], c1d_x_5x5x5[12][2]),
                                      offset5x5x5(c1d_x_5x5x5[13][0], c1d_x_5x5x5[13][1], c1d_x_5x5x5[13][2]),
                                      offset5x5x5(c1d_x_5x5x5[14][0], c1d_x_5x5x5[14][1], c1d_x_5x5x5[14][2]),
                                      offset5x5x5(c1d_x_5x5x5[15][0], c1d_x_5x5x5[15][1], c1d_x_5x5x5[15][2]),
                                      offset5x5x5(c1d_x_5x5x5[16][0], c1d_x_5x5x5[16][1], c1d_x_5x5x5[16][2]),
                                      offset5x5x5(c1d_x_5x5x5[17][0], c1d_x_5x5x5[17][1], c1d_x_5x5x5[17][2]),

                                      offset5x5x5(c1d_y_5x5x5[0][0], c1d_y_5x5x5[0][1], c1d_y_5x5x5[0][2]), // Y
                                      offset5x5x5(c1d_y_5x5x5[1][0], c1d_y_5x5x5[1][1], c1d_y_5x5x5[1][2]),
                                      offset5x5x5(c1d_y_5x5x5[2][0], c1d_y_5x5x5[2][1], c1d_y_5x5x5[2][2]),
                                      offset5x5x5(c1d_y_5x5x5[3][0], c1d_y_5x5x5[3][1], c1d_y_5x5x5[3][2]),
                                      offset5x5x5(c1d_y_5x5x5[4][0], c1d_y_5x5x5[4][1], c1d_y_5x5x5[4][2]),
                                      offset5x5x5(c1d_y_5x5x5[5][0], c1d_y_5x5x5[5][1], c1d_y_5x5x5[5][2]),
                                      offset5x5x5(c1d_y_5x5x5[6][0], c1d_y_5x5x5[6][1], c1d_y_5x5x5[6][2]),
                                      offset5x5x5(c1d_y_5x5x5[7][0], c1d_y_5x5x5[7][1], c1d_y_5x5x5[7][2]),
                                      offset5x5x5(c1d_y_5x5x5[8][0], c1d_y_5x5x5[8][1], c1d_y_5x5x5[8][2]),
                                      offset5x5x5(c1d_y_5x5x5[9][0], c1d_y_5x5x5[9][1], c1d_y_5x5x5[9][2]),
                                      offset5x5x5(c1d_y_5x5x5[10][0], c1d_y_5x5x5[10][1], c1d_y_5x5x5[10][2]),
                                      offset5x5x5(c1d_y_5x5x5[11][0], c1d_y_5x5x5[11][1], c1d_y_5x5x5[11][2]),
                                      offset5x5x5(c1d_y_5x5x5[12][0], c1d_y_5x5x5[12][1], c1d_y_5x5x5[12][2]),
                                      offset5x5x5(c1d_y_5x5x5[13][0], c1d_y_5x5x5[13][1], c1d_y_5x5x5[13][2]),
                                      offset5x5x5(c1d_y_5x5x5[14][0], c1d_y_5x5x5[14][1], c1d_y_5x5x5[14][2]),
                                      offset5x5x5(c1d_y_5x5x5[15][0], c1d_y_5x5x5[15][1], c1d_y_5x5x5[15][2]),
                                      offset5x5x5(c1d_y_5x5x5[16][0], c1d_y_5x5x5[16][1], c1d_y_5x5x5[16][2]),
                                      offset5x5x5(c1d_y_5x5x5[17][0], c1d_y_5x5x5[17][1], c1d_y_5x5x5[17][2]),

                                      offset5x5x5(c1d_z_5x5x5[0][0], c1d_z_5x5x5[0][1], c1d_z_5x5x5[0][2]), // Z
                                      offset5x5x5(c1d_z_5x5x5[1][0], c1d_z_5x5x5[1][1], c1d_z_5x5x5[1][2]),
                                      offset5x5x5(c1d_z_5x5x5[2][0], c1d_z_5x5x5[2][1], c1d_z_5x5x5[2][2]),
                                      offset5x5x5(c1d_z_5x5x5[3][0], c1d_z_5x5x5[3][1], c1d_z_5x5x5[3][2]),
                                      offset5x5x5(c1d_z_5x5x5[4][0], c1d_z_5x5x5[4][1], c1d_z_5x5x5[4][2]),
                                      offset5x5x5(c1d_z_5x5x5[5][0], c1d_z_5x5x5[5][1], c1d_z_5x5x5[5][2]),
                                      offset5x5x5(c1d_z_5x5x5[6][0], c1d_z_5x5x5[6][1], c1d_z_5x5x5[6][2]),
                                      offset5x5x5(c1d_z_5x5x5[7][0], c1d_z_5x5x5[7][1], c1d_z_5x5x5[7][2]),
                                      offset5x5x5(c1d_z_5x5x5[8][0], c1d_z_5x5x5[8][1], c1d_z_5x5x5[8][2]),
                                      offset5x5x5(c1d_z_5x5x5[9][0], c1d_z_5x5x5[9][1], c1d_z_5x5x5[9][2]),
                                      offset5x5x5(c1d_z_5x5x5[10][0], c1d_z_5x5x5[10][1], c1d_z_5x5x5[10][2]),
                                      offset5x5x5(c1d_z_5x5x5[11][0], c1d_z_5x5x5[11][1], c1d_z_5x5x5[11][2]),
                                      offset5x5x5(c1d_z_5x5x5[12][0], c1d_z_5x5x5[12][1], c1d_z_5x5x5[12][2]),
                                      offset5x5x5(c1d_z_5x5x5[13][0], c1d_z_5x5x5[13][1], c1d_z_5x5x5[13][2]),
                                      offset5x5x5(c1d_z_5x5x5[14][0], c1d_z_5x5x5[14][1], c1d_z_5x5x5[14][2]),
                                      offset5x5x5(c1d_z_5x5x5[15][0], c1d_z_5x5x5[15][1], c1d_z_5x5x5[15][2]),
                                      offset5x5x5(c1d_z_5x5x5[16][0], c1d_z_5x5x5[16][1], c1d_z_5x5x5[16][2]),
                                      offset5x5x5(c1d_z_5x5x5[17][0], c1d_z_5x5x5[17][1], c1d_z_5x5x5[17][2])
                                    };
  return offset[i];
}

MGARDX_EXEC int Coeff1D_L_Offset_5x5x5(SIZE i) {
  static constexpr int offset[54] = { offset5x5x5(c1d_x_5x5x5[0][0], c1d_x_5x5x5[0][1], c1d_x_5x5x5[0][2]-1), // X
                                      offset5x5x5(c1d_x_5x5x5[1][0], c1d_x_5x5x5[1][1], c1d_x_5x5x5[1][2]-1),
                                      offset5x5x5(c1d_x_5x5x5[2][0], c1d_x_5x5x5[2][1], c1d_x_5x5x5[2][2]-1),
                                      offset5x5x5(c1d_x_5x5x5[3][0], c1d_x_5x5x5[3][1], c1d_x_5x5x5[3][2]-1),
                                      offset5x5x5(c1d_x_5x5x5[4][0], c1d_x_5x5x5[4][1], c1d_x_5x5x5[4][2]-1),
                                      offset5x5x5(c1d_x_5x5x5[5][0], c1d_x_5x5x5[5][1], c1d_x_5x5x5[5][2]-1),
                                      offset5x5x5(c1d_x_5x5x5[6][0], c1d_x_5x5x5[6][1], c1d_x_5x5x5[6][2]-1),
                                      offset5x5x5(c1d_x_5x5x5[7][0], c1d_x_5x5x5[7][1], c1d_x_5x5x5[7][2]-1),
                                      offset5x5x5(c1d_x_5x5x5[8][0], c1d_x_5x5x5[8][1], c1d_x_5x5x5[8][2]-1),
                                      offset5x5x5(c1d_x_5x5x5[9][0], c1d_x_5x5x5[9][1], c1d_x_5x5x5[9][2]-1),
                                      offset5x5x5(c1d_x_5x5x5[10][0], c1d_x_5x5x5[10][1], c1d_x_5x5x5[10][2]-1),
                                      offset5x5x5(c1d_x_5x5x5[11][0], c1d_x_5x5x5[11][1], c1d_x_5x5x5[11][2]-1),
                                      offset5x5x5(c1d_x_5x5x5[12][0], c1d_x_5x5x5[12][1], c1d_x_5x5x5[12][2]-1),
                                      offset5x5x5(c1d_x_5x5x5[13][0], c1d_x_5x5x5[13][1], c1d_x_5x5x5[13][2]-1),
                                      offset5x5x5(c1d_x_5x5x5[14][0], c1d_x_5x5x5[14][1], c1d_x_5x5x5[14][2]-1),
                                      offset5x5x5(c1d_x_5x5x5[15][0], c1d_x_5x5x5[15][1], c1d_x_5x5x5[15][2]-1),
                                      offset5x5x5(c1d_x_5x5x5[16][0], c1d_x_5x5x5[16][1], c1d_x_5x5x5[16][2]-1),
                                      offset5x5x5(c1d_x_5x5x5[17][0], c1d_x_5x5x5[17][1], c1d_x_5x5x5[17][2]-1),

                                      offset5x5x5(c1d_y_5x5x5[0][0], c1d_y_5x5x5[0][1]-1, c1d_y_5x5x5[0][2]), // Y
                                      offset5x5x5(c1d_y_5x5x5[1][0], c1d_y_5x5x5[1][1]-1, c1d_y_5x5x5[1][2]),
                                      offset5x5x5(c1d_y_5x5x5[2][0], c1d_y_5x5x5[2][1]-1, c1d_y_5x5x5[2][2]),
                                      offset5x5x5(c1d_y_5x5x5[3][0], c1d_y_5x5x5[3][1]-1, c1d_y_5x5x5[3][2]),
                                      offset5x5x5(c1d_y_5x5x5[4][0], c1d_y_5x5x5[4][1]-1, c1d_y_5x5x5[4][2]),
                                      offset5x5x5(c1d_y_5x5x5[5][0], c1d_y_5x5x5[5][1]-1, c1d_y_5x5x5[5][2]),
                                      offset5x5x5(c1d_y_5x5x5[6][0], c1d_y_5x5x5[6][1]-1, c1d_y_5x5x5[6][2]),
                                      offset5x5x5(c1d_y_5x5x5[7][0], c1d_y_5x5x5[7][1]-1, c1d_y_5x5x5[7][2]),
                                      offset5x5x5(c1d_y_5x5x5[8][0], c1d_y_5x5x5[8][1]-1, c1d_y_5x5x5[8][2]),
                                      offset5x5x5(c1d_y_5x5x5[9][0], c1d_y_5x5x5[9][1]-1, c1d_y_5x5x5[9][2]),
                                      offset5x5x5(c1d_y_5x5x5[10][0], c1d_y_5x5x5[10][1]-1, c1d_y_5x5x5[10][2]),
                                      offset5x5x5(c1d_y_5x5x5[11][0], c1d_y_5x5x5[11][1]-1, c1d_y_5x5x5[11][2]),
                                      offset5x5x5(c1d_y_5x5x5[12][0], c1d_y_5x5x5[12][1]-1, c1d_y_5x5x5[12][2]),
                                      offset5x5x5(c1d_y_5x5x5[13][0], c1d_y_5x5x5[13][1]-1, c1d_y_5x5x5[13][2]),
                                      offset5x5x5(c1d_y_5x5x5[14][0], c1d_y_5x5x5[14][1]-1, c1d_y_5x5x5[14][2]),
                                      offset5x5x5(c1d_y_5x5x5[15][0], c1d_y_5x5x5[15][1]-1, c1d_y_5x5x5[15][2]),
                                      offset5x5x5(c1d_y_5x5x5[16][0], c1d_y_5x5x5[16][1]-1, c1d_y_5x5x5[16][2]),
                                      offset5x5x5(c1d_y_5x5x5[17][0], c1d_y_5x5x5[17][1]-1, c1d_y_5x5x5[17][2]),

                                      offset5x5x5(c1d_z_5x5x5[0][0]-1, c1d_z_5x5x5[0][1], c1d_z_5x5x5[0][2]), // Z
                                      offset5x5x5(c1d_z_5x5x5[1][0]-1, c1d_z_5x5x5[1][1], c1d_z_5x5x5[1][2]),
                                      offset5x5x5(c1d_z_5x5x5[2][0]-1, c1d_z_5x5x5[2][1], c1d_z_5x5x5[2][2]),
                                      offset5x5x5(c1d_z_5x5x5[3][0]-1, c1d_z_5x5x5[3][1], c1d_z_5x5x5[3][2]),
                                      offset5x5x5(c1d_z_5x5x5[4][0]-1, c1d_z_5x5x5[4][1], c1d_z_5x5x5[4][2]),
                                      offset5x5x5(c1d_z_5x5x5[5][0]-1, c1d_z_5x5x5[5][1], c1d_z_5x5x5[5][2]),
                                      offset5x5x5(c1d_z_5x5x5[6][0]-1, c1d_z_5x5x5[6][1], c1d_z_5x5x5[6][2]),
                                      offset5x5x5(c1d_z_5x5x5[7][0]-1, c1d_z_5x5x5[7][1], c1d_z_5x5x5[7][2]),
                                      offset5x5x5(c1d_z_5x5x5[8][0]-1, c1d_z_5x5x5[8][1], c1d_z_5x5x5[8][2]),
                                      offset5x5x5(c1d_z_5x5x5[9][0]-1, c1d_z_5x5x5[9][1], c1d_z_5x5x5[9][2]),
                                      offset5x5x5(c1d_z_5x5x5[10][0]-1, c1d_z_5x5x5[10][1], c1d_z_5x5x5[10][2]),
                                      offset5x5x5(c1d_z_5x5x5[11][0]-1, c1d_z_5x5x5[11][1], c1d_z_5x5x5[11][2]),
                                      offset5x5x5(c1d_z_5x5x5[12][0]-1, c1d_z_5x5x5[12][1], c1d_z_5x5x5[12][2]),
                                      offset5x5x5(c1d_z_5x5x5[13][0]-1, c1d_z_5x5x5[13][1], c1d_z_5x5x5[13][2]),
                                      offset5x5x5(c1d_z_5x5x5[14][0]-1, c1d_z_5x5x5[14][1], c1d_z_5x5x5[14][2]),
                                      offset5x5x5(c1d_z_5x5x5[15][0]-1, c1d_z_5x5x5[15][1], c1d_z_5x5x5[15][2]),
                                      offset5x5x5(c1d_z_5x5x5[16][0]-1, c1d_z_5x5x5[16][1], c1d_z_5x5x5[16][2]),
                                      offset5x5x5(c1d_z_5x5x5[17][0]-1, c1d_z_5x5x5[17][1], c1d_z_5x5x5[17][2])
                                    };
  return offset[i];
}

MGARDX_EXEC int Coeff1D_R_Offset_5x5x5(SIZE i) {
  static constexpr int offset[54] = { offset5x5x5(c1d_x_5x5x5[0][0], c1d_x_5x5x5[0][1], c1d_x_5x5x5[0][2]+1), // X
                                      offset5x5x5(c1d_x_5x5x5[1][0], c1d_x_5x5x5[1][1], c1d_x_5x5x5[1][2]+1),
                                      offset5x5x5(c1d_x_5x5x5[2][0], c1d_x_5x5x5[2][1], c1d_x_5x5x5[2][2]+1),
                                      offset5x5x5(c1d_x_5x5x5[3][0], c1d_x_5x5x5[3][1], c1d_x_5x5x5[3][2]+1),
                                      offset5x5x5(c1d_x_5x5x5[4][0], c1d_x_5x5x5[4][1], c1d_x_5x5x5[4][2]+1),
                                      offset5x5x5(c1d_x_5x5x5[5][0], c1d_x_5x5x5[5][1], c1d_x_5x5x5[5][2]+1),
                                      offset5x5x5(c1d_x_5x5x5[6][0], c1d_x_5x5x5[6][1], c1d_x_5x5x5[6][2]+1),
                                      offset5x5x5(c1d_x_5x5x5[7][0], c1d_x_5x5x5[7][1], c1d_x_5x5x5[7][2]+1),
                                      offset5x5x5(c1d_x_5x5x5[8][0], c1d_x_5x5x5[8][1], c1d_x_5x5x5[8][2]+1),
                                      offset5x5x5(c1d_x_5x5x5[9][0], c1d_x_5x5x5[9][1], c1d_x_5x5x5[9][2]+1),
                                      offset5x5x5(c1d_x_5x5x5[10][0], c1d_x_5x5x5[10][1], c1d_x_5x5x5[10][2]+1),
                                      offset5x5x5(c1d_x_5x5x5[11][0], c1d_x_5x5x5[11][1], c1d_x_5x5x5[11][2]+1),
                                      offset5x5x5(c1d_x_5x5x5[12][0], c1d_x_5x5x5[12][1], c1d_x_5x5x5[12][2]+1),
                                      offset5x5x5(c1d_x_5x5x5[13][0], c1d_x_5x5x5[13][1], c1d_x_5x5x5[13][2]+1),
                                      offset5x5x5(c1d_x_5x5x5[14][0], c1d_x_5x5x5[14][1], c1d_x_5x5x5[14][2]+1),
                                      offset5x5x5(c1d_x_5x5x5[15][0], c1d_x_5x5x5[15][1], c1d_x_5x5x5[15][2]+1),
                                      offset5x5x5(c1d_x_5x5x5[16][0], c1d_x_5x5x5[16][1], c1d_x_5x5x5[16][2]+1),
                                      offset5x5x5(c1d_x_5x5x5[17][0], c1d_x_5x5x5[17][1], c1d_x_5x5x5[17][2]+1),

                                      offset5x5x5(c1d_y_5x5x5[0][0], c1d_y_5x5x5[0][1]+1, c1d_y_5x5x5[0][2]), // Y
                                      offset5x5x5(c1d_y_5x5x5[1][0], c1d_y_5x5x5[1][1]+1, c1d_y_5x5x5[1][2]),
                                      offset5x5x5(c1d_y_5x5x5[2][0], c1d_y_5x5x5[2][1]+1, c1d_y_5x5x5[2][2]),
                                      offset5x5x5(c1d_y_5x5x5[3][0], c1d_y_5x5x5[3][1]+1, c1d_y_5x5x5[3][2]),
                                      offset5x5x5(c1d_y_5x5x5[4][0], c1d_y_5x5x5[4][1]+1, c1d_y_5x5x5[4][2]),
                                      offset5x5x5(c1d_y_5x5x5[5][0], c1d_y_5x5x5[5][1]+1, c1d_y_5x5x5[5][2]),
                                      offset5x5x5(c1d_y_5x5x5[6][0], c1d_y_5x5x5[6][1]+1, c1d_y_5x5x5[6][2]),
                                      offset5x5x5(c1d_y_5x5x5[7][0], c1d_y_5x5x5[7][1]+1, c1d_y_5x5x5[7][2]),
                                      offset5x5x5(c1d_y_5x5x5[8][0], c1d_y_5x5x5[8][1]+1, c1d_y_5x5x5[8][2]),
                                      offset5x5x5(c1d_y_5x5x5[9][0], c1d_y_5x5x5[9][1]+1, c1d_y_5x5x5[9][2]),
                                      offset5x5x5(c1d_y_5x5x5[10][0], c1d_y_5x5x5[10][1]+1, c1d_y_5x5x5[10][2]),
                                      offset5x5x5(c1d_y_5x5x5[11][0], c1d_y_5x5x5[11][1]+1, c1d_y_5x5x5[11][2]),
                                      offset5x5x5(c1d_y_5x5x5[12][0], c1d_y_5x5x5[12][1]+1, c1d_y_5x5x5[12][2]),
                                      offset5x5x5(c1d_y_5x5x5[13][0], c1d_y_5x5x5[13][1]+1, c1d_y_5x5x5[13][2]),
                                      offset5x5x5(c1d_y_5x5x5[14][0], c1d_y_5x5x5[14][1]+1, c1d_y_5x5x5[14][2]),
                                      offset5x5x5(c1d_y_5x5x5[15][0], c1d_y_5x5x5[15][1]+1, c1d_y_5x5x5[15][2]),
                                      offset5x5x5(c1d_y_5x5x5[16][0], c1d_y_5x5x5[16][1]+1, c1d_y_5x5x5[16][2]),
                                      offset5x5x5(c1d_y_5x5x5[17][0], c1d_y_5x5x5[17][1]+1, c1d_y_5x5x5[17][2]),

                                      offset5x5x5(c1d_z_5x5x5[0][0]+1, c1d_z_5x5x5[0][1], c1d_z_5x5x5[0][2]), // Z
                                      offset5x5x5(c1d_z_5x5x5[1][0]+1, c1d_z_5x5x5[1][1], c1d_z_5x5x5[1][2]),
                                      offset5x5x5(c1d_z_5x5x5[2][0]+1, c1d_z_5x5x5[2][1], c1d_z_5x5x5[2][2]),
                                      offset5x5x5(c1d_z_5x5x5[3][0]+1, c1d_z_5x5x5[3][1], c1d_z_5x5x5[3][2]),
                                      offset5x5x5(c1d_z_5x5x5[4][0]+1, c1d_z_5x5x5[4][1], c1d_z_5x5x5[4][2]),
                                      offset5x5x5(c1d_z_5x5x5[5][0]+1, c1d_z_5x5x5[5][1], c1d_z_5x5x5[5][2]),
                                      offset5x5x5(c1d_z_5x5x5[6][0]+1, c1d_z_5x5x5[6][1], c1d_z_5x5x5[6][2]),
                                      offset5x5x5(c1d_z_5x5x5[7][0]+1, c1d_z_5x5x5[7][1], c1d_z_5x5x5[7][2]),
                                      offset5x5x5(c1d_z_5x5x5[8][0]+1, c1d_z_5x5x5[8][1], c1d_z_5x5x5[8][2]),
                                      offset5x5x5(c1d_z_5x5x5[9][0]+1, c1d_z_5x5x5[9][1], c1d_z_5x5x5[9][2]),
                                      offset5x5x5(c1d_z_5x5x5[10][0]+1, c1d_z_5x5x5[10][1], c1d_z_5x5x5[10][2]),
                                      offset5x5x5(c1d_z_5x5x5[11][0]+1, c1d_z_5x5x5[11][1], c1d_z_5x5x5[11][2]),
                                      offset5x5x5(c1d_z_5x5x5[12][0]+1, c1d_z_5x5x5[12][1], c1d_z_5x5x5[12][2]),
                                      offset5x5x5(c1d_z_5x5x5[13][0]+1, c1d_z_5x5x5[13][1], c1d_z_5x5x5[13][2]),
                                      offset5x5x5(c1d_z_5x5x5[14][0]+1, c1d_z_5x5x5[14][1], c1d_z_5x5x5[14][2]),
                                      offset5x5x5(c1d_z_5x5x5[15][0]+1, c1d_z_5x5x5[15][1], c1d_z_5x5x5[15][2]),
                                      offset5x5x5(c1d_z_5x5x5[16][0]+1, c1d_z_5x5x5[16][1], c1d_z_5x5x5[16][2]),
                                      offset5x5x5(c1d_z_5x5x5[17][0]+1, c1d_z_5x5x5[17][1], c1d_z_5x5x5[17][2])
                                    };
  return offset[i];
}

MGARDX_EXEC int Coeff2D_MM_Offset_5x5x5(SIZE i) {
  static constexpr int offset[36] = { offset5x5x5(c2d_xy_5x5x5[0][0], c2d_xy_5x5x5[0][1], c2d_xy_5x5x5[0][2]), // XY
                                      offset5x5x5(c2d_xy_5x5x5[1][0], c2d_xy_5x5x5[1][1], c2d_xy_5x5x5[1][2]),
                                      offset5x5x5(c2d_xy_5x5x5[2][0], c2d_xy_5x5x5[2][1], c2d_xy_5x5x5[2][2]),
                                      offset5x5x5(c2d_xy_5x5x5[3][0], c2d_xy_5x5x5[3][1], c2d_xy_5x5x5[3][2]),
                                      offset5x5x5(c2d_xy_5x5x5[4][0], c2d_xy_5x5x5[4][1], c2d_xy_5x5x5[4][2]),
                                      offset5x5x5(c2d_xy_5x5x5[5][0], c2d_xy_5x5x5[5][1], c2d_xy_5x5x5[5][2]),
                                      offset5x5x5(c2d_xy_5x5x5[6][0], c2d_xy_5x5x5[6][1], c2d_xy_5x5x5[6][2]),
                                      offset5x5x5(c2d_xy_5x5x5[7][0], c2d_xy_5x5x5[7][1], c2d_xy_5x5x5[7][2]),
                                      offset5x5x5(c2d_xy_5x5x5[8][0], c2d_xy_5x5x5[8][1], c2d_xy_5x5x5[8][2]),
                                      offset5x5x5(c2d_xy_5x5x5[9][0], c2d_xy_5x5x5[9][1], c2d_xy_5x5x5[9][2]),
                                      offset5x5x5(c2d_xy_5x5x5[10][0], c2d_xy_5x5x5[10][1], c2d_xy_5x5x5[10][2]),
                                      offset5x5x5(c2d_xy_5x5x5[11][0], c2d_xy_5x5x5[11][1], c2d_xy_5x5x5[11][2]),

                                      offset5x5x5(c2d_xz_5x5x5[0][0], c2d_xz_5x5x5[0][1], c2d_xz_5x5x5[0][2]), // XZ
                                      offset5x5x5(c2d_xz_5x5x5[1][0], c2d_xz_5x5x5[1][1], c2d_xz_5x5x5[1][2]),
                                      offset5x5x5(c2d_xz_5x5x5[2][0], c2d_xz_5x5x5[2][1], c2d_xz_5x5x5[2][2]),
                                      offset5x5x5(c2d_xz_5x5x5[3][0], c2d_xz_5x5x5[3][1], c2d_xz_5x5x5[3][2]),
                                      offset5x5x5(c2d_xz_5x5x5[4][0], c2d_xz_5x5x5[4][1], c2d_xz_5x5x5[4][2]),
                                      offset5x5x5(c2d_xz_5x5x5[5][0], c2d_xz_5x5x5[5][1], c2d_xz_5x5x5[5][2]),
                                      offset5x5x5(c2d_xz_5x5x5[6][0], c2d_xz_5x5x5[6][1], c2d_xz_5x5x5[6][2]),
                                      offset5x5x5(c2d_xz_5x5x5[7][0], c2d_xz_5x5x5[7][1], c2d_xz_5x5x5[7][2]),
                                      offset5x5x5(c2d_xz_5x5x5[8][0], c2d_xz_5x5x5[8][1], c2d_xz_5x5x5[8][2]),
                                      offset5x5x5(c2d_xz_5x5x5[9][0], c2d_xz_5x5x5[9][1], c2d_xz_5x5x5[9][2]),
                                      offset5x5x5(c2d_xz_5x5x5[10][0], c2d_xz_5x5x5[10][1], c2d_xz_5x5x5[10][2]),
                                      offset5x5x5(c2d_xz_5x5x5[11][0], c2d_xz_5x5x5[11][1], c2d_xz_5x5x5[11][2]),

                                      offset5x5x5(c2d_yz_5x5x5[0][0], c2d_yz_5x5x5[0][1], c2d_yz_5x5x5[0][2]), // YZ
                                      offset5x5x5(c2d_yz_5x5x5[1][0], c2d_yz_5x5x5[1][1], c2d_yz_5x5x5[1][2]),
                                      offset5x5x5(c2d_yz_5x5x5[2][0], c2d_yz_5x5x5[2][1], c2d_yz_5x5x5[2][2]),
                                      offset5x5x5(c2d_yz_5x5x5[3][0], c2d_yz_5x5x5[3][1], c2d_yz_5x5x5[3][2]),
                                      offset5x5x5(c2d_yz_5x5x5[4][0], c2d_yz_5x5x5[4][1], c2d_yz_5x5x5[4][2]),
                                      offset5x5x5(c2d_yz_5x5x5[5][0], c2d_yz_5x5x5[5][1], c2d_yz_5x5x5[5][2]),
                                      offset5x5x5(c2d_yz_5x5x5[6][0], c2d_yz_5x5x5[6][1], c2d_yz_5x5x5[6][2]),
                                      offset5x5x5(c2d_yz_5x5x5[7][0], c2d_yz_5x5x5[7][1], c2d_yz_5x5x5[7][2]),
                                      offset5x5x5(c2d_yz_5x5x5[8][0], c2d_yz_5x5x5[8][1], c2d_yz_5x5x5[8][2]),
                                      offset5x5x5(c2d_yz_5x5x5[9][0], c2d_yz_5x5x5[9][1], c2d_yz_5x5x5[9][2]),
                                      offset5x5x5(c2d_yz_5x5x5[10][0], c2d_yz_5x5x5[10][1], c2d_yz_5x5x5[10][2]),
                                      offset5x5x5(c2d_yz_5x5x5[11][0], c2d_yz_5x5x5[11][1], c2d_yz_5x5x5[11][2])
                                      };
  return offset[i];
}

MGARDX_EXEC int Coeff2D_LL_Offset_5x5x5(SIZE i) {
  static constexpr int offset[36] = { offset5x5x5(c2d_xy_5x5x5[0][0], c2d_xy_5x5x5[0][1]-1, c2d_xy_5x5x5[0][2]-1), // XY
                                      offset5x5x5(c2d_xy_5x5x5[1][0], c2d_xy_5x5x5[1][1]-1, c2d_xy_5x5x5[1][2]-1),
                                      offset5x5x5(c2d_xy_5x5x5[2][0], c2d_xy_5x5x5[2][1]-1, c2d_xy_5x5x5[2][2]-1),
                                      offset5x5x5(c2d_xy_5x5x5[3][0], c2d_xy_5x5x5[3][1]-1, c2d_xy_5x5x5[3][2]-1),
                                      offset5x5x5(c2d_xy_5x5x5[4][0], c2d_xy_5x5x5[4][1]-1, c2d_xy_5x5x5[4][2]-1),
                                      offset5x5x5(c2d_xy_5x5x5[5][0], c2d_xy_5x5x5[5][1]-1, c2d_xy_5x5x5[5][2]-1),
                                      offset5x5x5(c2d_xy_5x5x5[6][0], c2d_xy_5x5x5[6][1]-1, c2d_xy_5x5x5[6][2]-1),
                                      offset5x5x5(c2d_xy_5x5x5[7][0], c2d_xy_5x5x5[7][1]-1, c2d_xy_5x5x5[7][2]-1),
                                      offset5x5x5(c2d_xy_5x5x5[8][0], c2d_xy_5x5x5[8][1]-1, c2d_xy_5x5x5[8][2]-1),
                                      offset5x5x5(c2d_xy_5x5x5[9][0], c2d_xy_5x5x5[9][1]-1, c2d_xy_5x5x5[9][2]-1),
                                      offset5x5x5(c2d_xy_5x5x5[10][0], c2d_xy_5x5x5[10][1]-1, c2d_xy_5x5x5[10][2]-1),
                                      offset5x5x5(c2d_xy_5x5x5[11][0], c2d_xy_5x5x5[11][1]-1, c2d_xy_5x5x5[11][2]-1),

                                      offset5x5x5(c2d_xz_5x5x5[0][0]-1, c2d_xz_5x5x5[0][1], c2d_xz_5x5x5[0][2]-1), // XZ
                                      offset5x5x5(c2d_xz_5x5x5[1][0]-1, c2d_xz_5x5x5[1][1], c2d_xz_5x5x5[1][2]-1),
                                      offset5x5x5(c2d_xz_5x5x5[2][0]-1, c2d_xz_5x5x5[2][1], c2d_xz_5x5x5[2][2]-1),
                                      offset5x5x5(c2d_xz_5x5x5[3][0]-1, c2d_xz_5x5x5[3][1], c2d_xz_5x5x5[3][2]-1),
                                      offset5x5x5(c2d_xz_5x5x5[4][0]-1, c2d_xz_5x5x5[4][1], c2d_xz_5x5x5[4][2]-1),
                                      offset5x5x5(c2d_xz_5x5x5[5][0]-1, c2d_xz_5x5x5[5][1], c2d_xz_5x5x5[5][2]-1),
                                      offset5x5x5(c2d_xz_5x5x5[6][0]-1, c2d_xz_5x5x5[6][1], c2d_xz_5x5x5[6][2]-1),
                                      offset5x5x5(c2d_xz_5x5x5[7][0]-1, c2d_xz_5x5x5[7][1], c2d_xz_5x5x5[7][2]-1),
                                      offset5x5x5(c2d_xz_5x5x5[8][0]-1, c2d_xz_5x5x5[8][1], c2d_xz_5x5x5[8][2]-1),
                                      offset5x5x5(c2d_xz_5x5x5[9][0]-1, c2d_xz_5x5x5[9][1], c2d_xz_5x5x5[9][2]-1),
                                      offset5x5x5(c2d_xz_5x5x5[10][0]-1, c2d_xz_5x5x5[10][1], c2d_xz_5x5x5[10][2]-1),
                                      offset5x5x5(c2d_xz_5x5x5[11][0]-1, c2d_xz_5x5x5[11][1], c2d_xz_5x5x5[11][2]-1),

                                      offset5x5x5(c2d_yz_5x5x5[0][0]-1, c2d_yz_5x5x5[0][1]-1, c2d_yz_5x5x5[0][2]), // YZ
                                      offset5x5x5(c2d_yz_5x5x5[1][0]-1, c2d_yz_5x5x5[1][1]-1, c2d_yz_5x5x5[1][2]),
                                      offset5x5x5(c2d_yz_5x5x5[2][0]-1, c2d_yz_5x5x5[2][1]-1, c2d_yz_5x5x5[2][2]),
                                      offset5x5x5(c2d_yz_5x5x5[3][0]-1, c2d_yz_5x5x5[3][1]-1, c2d_yz_5x5x5[3][2]),
                                      offset5x5x5(c2d_yz_5x5x5[4][0]-1, c2d_yz_5x5x5[4][1]-1, c2d_yz_5x5x5[4][2]),
                                      offset5x5x5(c2d_yz_5x5x5[5][0]-1, c2d_yz_5x5x5[5][1]-1, c2d_yz_5x5x5[5][2]),
                                      offset5x5x5(c2d_yz_5x5x5[6][0]-1, c2d_yz_5x5x5[6][1]-1, c2d_yz_5x5x5[6][2]),
                                      offset5x5x5(c2d_yz_5x5x5[7][0]-1, c2d_yz_5x5x5[7][1]-1, c2d_yz_5x5x5[7][2]),
                                      offset5x5x5(c2d_yz_5x5x5[8][0]-1, c2d_yz_5x5x5[8][1]-1, c2d_yz_5x5x5[8][2]),
                                      offset5x5x5(c2d_yz_5x5x5[9][0]-1, c2d_yz_5x5x5[9][1]-1, c2d_yz_5x5x5[9][2]),
                                      offset5x5x5(c2d_yz_5x5x5[10][0]-1, c2d_yz_5x5x5[10][1]-1, c2d_yz_5x5x5[10][2]),
                                      offset5x5x5(c2d_yz_5x5x5[11][0]-1, c2d_yz_5x5x5[11][1]-1, c2d_yz_5x5x5[11][2])
                                      };
  return offset[i];
}

MGARDX_EXEC int Coeff2D_LR_Offset_5x5x5(SIZE i) {
  static constexpr int offset[36] = { offset5x5x5(c2d_xy_5x5x5[0][0], c2d_xy_5x5x5[0][1]-1, c2d_xy_5x5x5[0][2]+1), // XY
                                      offset5x5x5(c2d_xy_5x5x5[1][0], c2d_xy_5x5x5[1][1]-1, c2d_xy_5x5x5[1][2]+1),
                                      offset5x5x5(c2d_xy_5x5x5[2][0], c2d_xy_5x5x5[2][1]-1, c2d_xy_5x5x5[2][2]+1),
                                      offset5x5x5(c2d_xy_5x5x5[3][0], c2d_xy_5x5x5[3][1]-1, c2d_xy_5x5x5[3][2]+1),
                                      offset5x5x5(c2d_xy_5x5x5[4][0], c2d_xy_5x5x5[4][1]-1, c2d_xy_5x5x5[4][2]+1),
                                      offset5x5x5(c2d_xy_5x5x5[5][0], c2d_xy_5x5x5[5][1]-1, c2d_xy_5x5x5[5][2]+1),
                                      offset5x5x5(c2d_xy_5x5x5[6][0], c2d_xy_5x5x5[6][1]-1, c2d_xy_5x5x5[6][2]+1),
                                      offset5x5x5(c2d_xy_5x5x5[7][0], c2d_xy_5x5x5[7][1]-1, c2d_xy_5x5x5[7][2]+1),
                                      offset5x5x5(c2d_xy_5x5x5[8][0], c2d_xy_5x5x5[8][1]-1, c2d_xy_5x5x5[8][2]+1),
                                      offset5x5x5(c2d_xy_5x5x5[9][0], c2d_xy_5x5x5[9][1]-1, c2d_xy_5x5x5[9][2]+1),
                                      offset5x5x5(c2d_xy_5x5x5[10][0], c2d_xy_5x5x5[10][1]-1, c2d_xy_5x5x5[10][2]+1),
                                      offset5x5x5(c2d_xy_5x5x5[11][0], c2d_xy_5x5x5[11][1]-1, c2d_xy_5x5x5[11][2]+1),

                                      offset5x5x5(c2d_xz_5x5x5[0][0]-1, c2d_xz_5x5x5[0][1], c2d_xz_5x5x5[0][2]+1), // XZ
                                      offset5x5x5(c2d_xz_5x5x5[1][0]-1, c2d_xz_5x5x5[1][1], c2d_xz_5x5x5[1][2]+1),
                                      offset5x5x5(c2d_xz_5x5x5[2][0]-1, c2d_xz_5x5x5[2][1], c2d_xz_5x5x5[2][2]+1),
                                      offset5x5x5(c2d_xz_5x5x5[3][0]-1, c2d_xz_5x5x5[3][1], c2d_xz_5x5x5[3][2]+1),
                                      offset5x5x5(c2d_xz_5x5x5[4][0]-1, c2d_xz_5x5x5[4][1], c2d_xz_5x5x5[4][2]+1),
                                      offset5x5x5(c2d_xz_5x5x5[5][0]-1, c2d_xz_5x5x5[5][1], c2d_xz_5x5x5[5][2]+1),
                                      offset5x5x5(c2d_xz_5x5x5[6][0]-1, c2d_xz_5x5x5[6][1], c2d_xz_5x5x5[6][2]+1),
                                      offset5x5x5(c2d_xz_5x5x5[7][0]-1, c2d_xz_5x5x5[7][1], c2d_xz_5x5x5[7][2]+1),
                                      offset5x5x5(c2d_xz_5x5x5[8][0]-1, c2d_xz_5x5x5[8][1], c2d_xz_5x5x5[8][2]+1),
                                      offset5x5x5(c2d_xz_5x5x5[9][0]-1, c2d_xz_5x5x5[9][1], c2d_xz_5x5x5[9][2]+1),
                                      offset5x5x5(c2d_xz_5x5x5[10][0]-1, c2d_xz_5x5x5[10][1], c2d_xz_5x5x5[10][2]+1),
                                      offset5x5x5(c2d_xz_5x5x5[11][0]-1, c2d_xz_5x5x5[11][1], c2d_xz_5x5x5[11][2]+1),

                                      offset5x5x5(c2d_yz_5x5x5[0][0]-1, c2d_yz_5x5x5[0][1]+1, c2d_yz_5x5x5[0][2]), // YZ
                                      offset5x5x5(c2d_yz_5x5x5[1][0]-1, c2d_yz_5x5x5[1][1]+1, c2d_yz_5x5x5[1][2]),
                                      offset5x5x5(c2d_yz_5x5x5[2][0]-1, c2d_yz_5x5x5[2][1]+1, c2d_yz_5x5x5[2][2]),
                                      offset5x5x5(c2d_yz_5x5x5[3][0]-1, c2d_yz_5x5x5[3][1]+1, c2d_yz_5x5x5[3][2]),
                                      offset5x5x5(c2d_yz_5x5x5[4][0]-1, c2d_yz_5x5x5[4][1]+1, c2d_yz_5x5x5[4][2]),
                                      offset5x5x5(c2d_yz_5x5x5[5][0]-1, c2d_yz_5x5x5[5][1]+1, c2d_yz_5x5x5[5][2]),
                                      offset5x5x5(c2d_yz_5x5x5[6][0]-1, c2d_yz_5x5x5[6][1]+1, c2d_yz_5x5x5[6][2]),
                                      offset5x5x5(c2d_yz_5x5x5[7][0]-1, c2d_yz_5x5x5[7][1]+1, c2d_yz_5x5x5[7][2]),
                                      offset5x5x5(c2d_yz_5x5x5[8][0]-1, c2d_yz_5x5x5[8][1]+1, c2d_yz_5x5x5[8][2]),
                                      offset5x5x5(c2d_yz_5x5x5[9][0]-1, c2d_yz_5x5x5[9][1]+1, c2d_yz_5x5x5[9][2]),
                                      offset5x5x5(c2d_yz_5x5x5[10][0]-1, c2d_yz_5x5x5[10][1]+1, c2d_yz_5x5x5[10][2]),
                                      offset5x5x5(c2d_yz_5x5x5[11][0]-1, c2d_yz_5x5x5[11][1]+1, c2d_yz_5x5x5[11][2])
                                      };
  return offset[i];
}

MGARDX_EXEC int Coeff2D_RL_Offset_5x5x5(SIZE i) {
  static constexpr int offset[36] = { offset5x5x5(c2d_xy_5x5x5[0][0], c2d_xy_5x5x5[0][1]+1, c2d_xy_5x5x5[0][2]-1), // XY
                                      offset5x5x5(c2d_xy_5x5x5[1][0], c2d_xy_5x5x5[1][1]+1, c2d_xy_5x5x5[1][2]-1),
                                      offset5x5x5(c2d_xy_5x5x5[2][0], c2d_xy_5x5x5[2][1]+1, c2d_xy_5x5x5[2][2]-1),
                                      offset5x5x5(c2d_xy_5x5x5[3][0], c2d_xy_5x5x5[3][1]+1, c2d_xy_5x5x5[3][2]-1),
                                      offset5x5x5(c2d_xy_5x5x5[4][0], c2d_xy_5x5x5[4][1]+1, c2d_xy_5x5x5[4][2]-1),
                                      offset5x5x5(c2d_xy_5x5x5[5][0], c2d_xy_5x5x5[5][1]+1, c2d_xy_5x5x5[5][2]-1),
                                      offset5x5x5(c2d_xy_5x5x5[6][0], c2d_xy_5x5x5[6][1]+1, c2d_xy_5x5x5[6][2]-1),
                                      offset5x5x5(c2d_xy_5x5x5[7][0], c2d_xy_5x5x5[7][1]+1, c2d_xy_5x5x5[7][2]-1),
                                      offset5x5x5(c2d_xy_5x5x5[8][0], c2d_xy_5x5x5[8][1]+1, c2d_xy_5x5x5[8][2]-1),
                                      offset5x5x5(c2d_xy_5x5x5[9][0], c2d_xy_5x5x5[9][1]+1, c2d_xy_5x5x5[9][2]-1),
                                      offset5x5x5(c2d_xy_5x5x5[10][0], c2d_xy_5x5x5[10][1]+1, c2d_xy_5x5x5[10][2]-1),
                                      offset5x5x5(c2d_xy_5x5x5[11][0], c2d_xy_5x5x5[11][1]+1, c2d_xy_5x5x5[11][2]-1),

                                      offset5x5x5(c2d_xz_5x5x5[0][0]+1, c2d_xz_5x5x5[0][1], c2d_xz_5x5x5[0][2]-1), // XZ
                                      offset5x5x5(c2d_xz_5x5x5[1][0]+1, c2d_xz_5x5x5[1][1], c2d_xz_5x5x5[1][2]-1),
                                      offset5x5x5(c2d_xz_5x5x5[2][0]+1, c2d_xz_5x5x5[2][1], c2d_xz_5x5x5[2][2]-1),
                                      offset5x5x5(c2d_xz_5x5x5[3][0]+1, c2d_xz_5x5x5[3][1], c2d_xz_5x5x5[3][2]-1),
                                      offset5x5x5(c2d_xz_5x5x5[4][0]+1, c2d_xz_5x5x5[4][1], c2d_xz_5x5x5[4][2]-1),
                                      offset5x5x5(c2d_xz_5x5x5[5][0]+1, c2d_xz_5x5x5[5][1], c2d_xz_5x5x5[5][2]-1),
                                      offset5x5x5(c2d_xz_5x5x5[6][0]+1, c2d_xz_5x5x5[6][1], c2d_xz_5x5x5[6][2]-1),
                                      offset5x5x5(c2d_xz_5x5x5[7][0]+1, c2d_xz_5x5x5[7][1], c2d_xz_5x5x5[7][2]-1),
                                      offset5x5x5(c2d_xz_5x5x5[8][0]+1, c2d_xz_5x5x5[8][1], c2d_xz_5x5x5[8][2]-1),
                                      offset5x5x5(c2d_xz_5x5x5[9][0]+1, c2d_xz_5x5x5[9][1], c2d_xz_5x5x5[9][2]-1),
                                      offset5x5x5(c2d_xz_5x5x5[10][0]+1, c2d_xz_5x5x5[10][1], c2d_xz_5x5x5[10][2]-1),
                                      offset5x5x5(c2d_xz_5x5x5[11][0]+1, c2d_xz_5x5x5[11][1], c2d_xz_5x5x5[11][2]-1),

                                      offset5x5x5(c2d_yz_5x5x5[0][0]+1, c2d_yz_5x5x5[0][1]-1, c2d_yz_5x5x5[0][2]), // YZ
                                      offset5x5x5(c2d_yz_5x5x5[1][0]+1, c2d_yz_5x5x5[1][1]-1, c2d_yz_5x5x5[1][2]),
                                      offset5x5x5(c2d_yz_5x5x5[2][0]+1, c2d_yz_5x5x5[2][1]-1, c2d_yz_5x5x5[2][2]),
                                      offset5x5x5(c2d_yz_5x5x5[3][0]+1, c2d_yz_5x5x5[3][1]-1, c2d_yz_5x5x5[3][2]),
                                      offset5x5x5(c2d_yz_5x5x5[4][0]+1, c2d_yz_5x5x5[4][1]-1, c2d_yz_5x5x5[4][2]),
                                      offset5x5x5(c2d_yz_5x5x5[5][0]+1, c2d_yz_5x5x5[5][1]-1, c2d_yz_5x5x5[5][2]),
                                      offset5x5x5(c2d_yz_5x5x5[6][0]+1, c2d_yz_5x5x5[6][1]-1, c2d_yz_5x5x5[6][2]),
                                      offset5x5x5(c2d_yz_5x5x5[7][0]+1, c2d_yz_5x5x5[7][1]-1, c2d_yz_5x5x5[7][2]),
                                      offset5x5x5(c2d_yz_5x5x5[8][0]+1, c2d_yz_5x5x5[8][1]-1, c2d_yz_5x5x5[8][2]),
                                      offset5x5x5(c2d_yz_5x5x5[9][0]+1, c2d_yz_5x5x5[9][1]-1, c2d_yz_5x5x5[9][2]),
                                      offset5x5x5(c2d_yz_5x5x5[10][0]+1, c2d_yz_5x5x5[10][1]-1, c2d_yz_5x5x5[10][2]),
                                      offset5x5x5(c2d_yz_5x5x5[11][0]+1, c2d_yz_5x5x5[11][1]-1, c2d_yz_5x5x5[11][2])
                                      };
  return offset[i];
}

MGARDX_EXEC int Coeff2D_RR_Offset_5x5x5(SIZE i) {
  static constexpr int offset[36] = { offset5x5x5(c2d_xy_5x5x5[0][0], c2d_xy_5x5x5[0][1]+1, c2d_xy_5x5x5[0][2]+1), // XY
                                      offset5x5x5(c2d_xy_5x5x5[1][0], c2d_xy_5x5x5[1][1]+1, c2d_xy_5x5x5[1][2]+1),
                                      offset5x5x5(c2d_xy_5x5x5[2][0], c2d_xy_5x5x5[2][1]+1, c2d_xy_5x5x5[2][2]+1),
                                      offset5x5x5(c2d_xy_5x5x5[3][0], c2d_xy_5x5x5[3][1]+1, c2d_xy_5x5x5[3][2]+1),
                                      offset5x5x5(c2d_xy_5x5x5[4][0], c2d_xy_5x5x5[4][1]+1, c2d_xy_5x5x5[4][2]+1),
                                      offset5x5x5(c2d_xy_5x5x5[5][0], c2d_xy_5x5x5[5][1]+1, c2d_xy_5x5x5[5][2]+1),
                                      offset5x5x5(c2d_xy_5x5x5[6][0], c2d_xy_5x5x5[6][1]+1, c2d_xy_5x5x5[6][2]+1),
                                      offset5x5x5(c2d_xy_5x5x5[7][0], c2d_xy_5x5x5[7][1]+1, c2d_xy_5x5x5[7][2]+1),
                                      offset5x5x5(c2d_xy_5x5x5[8][0], c2d_xy_5x5x5[8][1]+1, c2d_xy_5x5x5[8][2]+1),
                                      offset5x5x5(c2d_xy_5x5x5[9][0], c2d_xy_5x5x5[9][1]+1, c2d_xy_5x5x5[9][2]+1),
                                      offset5x5x5(c2d_xy_5x5x5[10][0], c2d_xy_5x5x5[10][1]+1, c2d_xy_5x5x5[10][2]+1),
                                      offset5x5x5(c2d_xy_5x5x5[11][0], c2d_xy_5x5x5[11][1]+1, c2d_xy_5x5x5[11][2]+1),

                                      offset5x5x5(c2d_xz_5x5x5[0][0]+1, c2d_xz_5x5x5[0][1], c2d_xz_5x5x5[0][2]+1), // XZ
                                      offset5x5x5(c2d_xz_5x5x5[1][0]+1, c2d_xz_5x5x5[1][1], c2d_xz_5x5x5[1][2]+1),
                                      offset5x5x5(c2d_xz_5x5x5[2][0]+1, c2d_xz_5x5x5[2][1], c2d_xz_5x5x5[2][2]+1),
                                      offset5x5x5(c2d_xz_5x5x5[3][0]+1, c2d_xz_5x5x5[3][1], c2d_xz_5x5x5[3][2]+1),
                                      offset5x5x5(c2d_xz_5x5x5[4][0]+1, c2d_xz_5x5x5[4][1], c2d_xz_5x5x5[4][2]+1),
                                      offset5x5x5(c2d_xz_5x5x5[5][0]+1, c2d_xz_5x5x5[5][1], c2d_xz_5x5x5[5][2]+1),
                                      offset5x5x5(c2d_xz_5x5x5[6][0]+1, c2d_xz_5x5x5[6][1], c2d_xz_5x5x5[6][2]+1),
                                      offset5x5x5(c2d_xz_5x5x5[7][0]+1, c2d_xz_5x5x5[7][1], c2d_xz_5x5x5[7][2]+1),
                                      offset5x5x5(c2d_xz_5x5x5[8][0]+1, c2d_xz_5x5x5[8][1], c2d_xz_5x5x5[8][2]+1),
                                      offset5x5x5(c2d_xz_5x5x5[9][0]+1, c2d_xz_5x5x5[9][1], c2d_xz_5x5x5[9][2]+1),
                                      offset5x5x5(c2d_xz_5x5x5[10][0]+1, c2d_xz_5x5x5[10][1], c2d_xz_5x5x5[10][2]+1),
                                      offset5x5x5(c2d_xz_5x5x5[11][0]+1, c2d_xz_5x5x5[11][1], c2d_xz_5x5x5[11][2]+1),

                                      offset5x5x5(c2d_yz_5x5x5[0][0]+1, c2d_yz_5x5x5[0][1]+1, c2d_yz_5x5x5[0][2]), // YZ
                                      offset5x5x5(c2d_yz_5x5x5[1][0]+1, c2d_yz_5x5x5[1][1]+1, c2d_yz_5x5x5[1][2]),
                                      offset5x5x5(c2d_yz_5x5x5[2][0]+1, c2d_yz_5x5x5[2][1]+1, c2d_yz_5x5x5[2][2]),
                                      offset5x5x5(c2d_yz_5x5x5[3][0]+1, c2d_yz_5x5x5[3][1]+1, c2d_yz_5x5x5[3][2]),
                                      offset5x5x5(c2d_yz_5x5x5[4][0]+1, c2d_yz_5x5x5[4][1]+1, c2d_yz_5x5x5[4][2]),
                                      offset5x5x5(c2d_yz_5x5x5[5][0]+1, c2d_yz_5x5x5[5][1]+1, c2d_yz_5x5x5[5][2]),
                                      offset5x5x5(c2d_yz_5x5x5[6][0]+1, c2d_yz_5x5x5[6][1]+1, c2d_yz_5x5x5[6][2]),
                                      offset5x5x5(c2d_yz_5x5x5[7][0]+1, c2d_yz_5x5x5[7][1]+1, c2d_yz_5x5x5[7][2]),
                                      offset5x5x5(c2d_yz_5x5x5[8][0]+1, c2d_yz_5x5x5[8][1]+1, c2d_yz_5x5x5[8][2]),
                                      offset5x5x5(c2d_yz_5x5x5[9][0]+1, c2d_yz_5x5x5[9][1]+1, c2d_yz_5x5x5[9][2]),
                                      offset5x5x5(c2d_yz_5x5x5[10][0]+1, c2d_yz_5x5x5[10][1]+1, c2d_yz_5x5x5[10][2]),
                                      offset5x5x5(c2d_yz_5x5x5[11][0]+1, c2d_yz_5x5x5[11][1]+1, c2d_yz_5x5x5[11][2])
                                      };
  return offset[i];
}

MGARDX_EXEC int Coeff3D_MMM_Offset_5x5x5(SIZE i) {
  static constexpr int offset[8] = {offset5x5x5(c3d_xyz_5x5x5[0][0], c3d_xyz_5x5x5[0][1], c3d_xyz_5x5x5[0][2]),
                                    offset5x5x5(c3d_xyz_5x5x5[1][0], c3d_xyz_5x5x5[1][1], c3d_xyz_5x5x5[1][2]),
                                    offset5x5x5(c3d_xyz_5x5x5[2][0], c3d_xyz_5x5x5[2][1], c3d_xyz_5x5x5[2][2]),
                                    offset5x5x5(c3d_xyz_5x5x5[3][0], c3d_xyz_5x5x5[3][1], c3d_xyz_5x5x5[3][2]),
                                    offset5x5x5(c3d_xyz_5x5x5[4][0], c3d_xyz_5x5x5[4][1], c3d_xyz_5x5x5[4][2]),
                                    offset5x5x5(c3d_xyz_5x5x5[5][0], c3d_xyz_5x5x5[5][1], c3d_xyz_5x5x5[5][2]),
                                    offset5x5x5(c3d_xyz_5x5x5[6][0], c3d_xyz_5x5x5[6][1], c3d_xyz_5x5x5[6][2])};

  return offset[i];
}

MGARDX_EXEC int Coeff3D_LLL_Offset_5x5x5(SIZE i) {
  static constexpr int offset[8] = {offset5x5x5(c3d_xyz_5x5x5[0][0]-1, c3d_xyz_5x5x5[0][1]-1, c3d_xyz_5x5x5[0][2]-1), // XY
                                    offset5x5x5(c3d_xyz_5x5x5[1][0]-1, c3d_xyz_5x5x5[1][1]-1, c3d_xyz_5x5x5[1][2]-1),
                                    offset5x5x5(c3d_xyz_5x5x5[2][0]-1, c3d_xyz_5x5x5[2][1]-1, c3d_xyz_5x5x5[2][2]-1),
                                    offset5x5x5(c3d_xyz_5x5x5[3][0]-1, c3d_xyz_5x5x5[3][1]-1, c3d_xyz_5x5x5[3][2]-1),
                                    offset5x5x5(c3d_xyz_5x5x5[4][0]-1, c3d_xyz_5x5x5[4][1]-1, c3d_xyz_5x5x5[4][2]-1),
                                    offset5x5x5(c3d_xyz_5x5x5[5][0]-1, c3d_xyz_5x5x5[5][1]-1, c3d_xyz_5x5x5[5][2]-1),
                                    offset5x5x5(c3d_xyz_5x5x5[6][0]-1, c3d_xyz_5x5x5[6][1]-1, c3d_xyz_5x5x5[6][2]-1),
                                    offset5x5x5(c3d_xyz_5x5x5[7][0]-1, c3d_xyz_5x5x5[7][1]-1, c3d_xyz_5x5x5[7][2]-1)};

  return offset[i];
}

MGARDX_EXEC int Coeff3D_LLR_Offset_5x5x5(SIZE i) {
  static constexpr int offset[8] = {offset5x5x5(c3d_xyz_5x5x5[0][0]-1, c3d_xyz_5x5x5[0][1]-1, c3d_xyz_5x5x5[0][2]+1), // XY
                                    offset5x5x5(c3d_xyz_5x5x5[1][0]-1, c3d_xyz_5x5x5[1][1]-1, c3d_xyz_5x5x5[1][2]+1),
                                    offset5x5x5(c3d_xyz_5x5x5[2][0]-1, c3d_xyz_5x5x5[2][1]-1, c3d_xyz_5x5x5[2][2]+1),
                                    offset5x5x5(c3d_xyz_5x5x5[3][0]-1, c3d_xyz_5x5x5[3][1]-1, c3d_xyz_5x5x5[3][2]+1),
                                    offset5x5x5(c3d_xyz_5x5x5[4][0]-1, c3d_xyz_5x5x5[4][1]-1, c3d_xyz_5x5x5[4][2]+1),
                                    offset5x5x5(c3d_xyz_5x5x5[5][0]-1, c3d_xyz_5x5x5[5][1]-1, c3d_xyz_5x5x5[5][2]+1),
                                    offset5x5x5(c3d_xyz_5x5x5[6][0]-1, c3d_xyz_5x5x5[6][1]-1, c3d_xyz_5x5x5[6][2]+1),
                                    offset5x5x5(c3d_xyz_5x5x5[7][0]-1, c3d_xyz_5x5x5[7][1]-1, c3d_xyz_5x5x5[7][2]+1)};

  return offset[i];
}

MGARDX_EXEC int Coeff3D_LRL_Offset_5x5x5(SIZE i) {
  static constexpr int offset[8] = {offset5x5x5(c3d_xyz_5x5x5[0][0]-1, c3d_xyz_5x5x5[0][1]+1, c3d_xyz_5x5x5[0][2]-1), // XY
                                    offset5x5x5(c3d_xyz_5x5x5[1][0]-1, c3d_xyz_5x5x5[1][1]+1, c3d_xyz_5x5x5[1][2]-1),
                                    offset5x5x5(c3d_xyz_5x5x5[2][0]-1, c3d_xyz_5x5x5[2][1]+1, c3d_xyz_5x5x5[2][2]-1),
                                    offset5x5x5(c3d_xyz_5x5x5[3][0]-1, c3d_xyz_5x5x5[3][1]+1, c3d_xyz_5x5x5[3][2]-1),
                                    offset5x5x5(c3d_xyz_5x5x5[4][0]-1, c3d_xyz_5x5x5[4][1]+1, c3d_xyz_5x5x5[4][2]-1),
                                    offset5x5x5(c3d_xyz_5x5x5[5][0]-1, c3d_xyz_5x5x5[5][1]+1, c3d_xyz_5x5x5[5][2]-1),
                                    offset5x5x5(c3d_xyz_5x5x5[6][0]-1, c3d_xyz_5x5x5[6][1]+1, c3d_xyz_5x5x5[6][2]-1),
                                    offset5x5x5(c3d_xyz_5x5x5[7][0]-1, c3d_xyz_5x5x5[7][1]+1, c3d_xyz_5x5x5[7][2]-1)};

  return offset[i];
}

MGARDX_EXEC int Coeff3D_LRR_Offset_5x5x5(SIZE i) {
  static constexpr int offset[8] = {offset5x5x5(c3d_xyz_5x5x5[0][0]-1, c3d_xyz_5x5x5[0][1]+1, c3d_xyz_5x5x5[0][2]+1), // XY
                                    offset5x5x5(c3d_xyz_5x5x5[1][0]-1, c3d_xyz_5x5x5[1][1]+1, c3d_xyz_5x5x5[1][2]+1),
                                    offset5x5x5(c3d_xyz_5x5x5[2][0]-1, c3d_xyz_5x5x5[2][1]+1, c3d_xyz_5x5x5[2][2]+1),
                                    offset5x5x5(c3d_xyz_5x5x5[3][0]-1, c3d_xyz_5x5x5[3][1]+1, c3d_xyz_5x5x5[3][2]+1),
                                    offset5x5x5(c3d_xyz_5x5x5[4][0]-1, c3d_xyz_5x5x5[4][1]+1, c3d_xyz_5x5x5[4][2]+1),
                                    offset5x5x5(c3d_xyz_5x5x5[5][0]-1, c3d_xyz_5x5x5[5][1]+1, c3d_xyz_5x5x5[5][2]+1),
                                    offset5x5x5(c3d_xyz_5x5x5[6][0]-1, c3d_xyz_5x5x5[6][1]+1, c3d_xyz_5x5x5[6][2]+1),
                                    offset5x5x5(c3d_xyz_5x5x5[7][0]-1, c3d_xyz_5x5x5[7][1]+1, c3d_xyz_5x5x5[7][2]+1)};

  return offset[i];
}

MGARDX_EXEC int Coeff3D_RLL_Offset_5x5x5(SIZE i) {
  static constexpr int offset[8] = {offset5x5x5(c3d_xyz_5x5x5[0][0]+1, c3d_xyz_5x5x5[0][1]-1, c3d_xyz_5x5x5[0][2]-1), // XY
                                    offset5x5x5(c3d_xyz_5x5x5[1][0]+1, c3d_xyz_5x5x5[1][1]-1, c3d_xyz_5x5x5[1][2]-1),
                                    offset5x5x5(c3d_xyz_5x5x5[2][0]+1, c3d_xyz_5x5x5[2][1]-1, c3d_xyz_5x5x5[2][2]-1),
                                    offset5x5x5(c3d_xyz_5x5x5[3][0]+1, c3d_xyz_5x5x5[3][1]-1, c3d_xyz_5x5x5[3][2]-1),
                                    offset5x5x5(c3d_xyz_5x5x5[4][0]+1, c3d_xyz_5x5x5[4][1]-1, c3d_xyz_5x5x5[4][2]-1),
                                    offset5x5x5(c3d_xyz_5x5x5[5][0]+1, c3d_xyz_5x5x5[5][1]-1, c3d_xyz_5x5x5[5][2]-1),
                                    offset5x5x5(c3d_xyz_5x5x5[6][0]+1, c3d_xyz_5x5x5[6][1]-1, c3d_xyz_5x5x5[6][2]-1),
                                    offset5x5x5(c3d_xyz_5x5x5[7][0]+1, c3d_xyz_5x5x5[7][1]-1, c3d_xyz_5x5x5[7][2]-1)};

  return offset[i];
}

MGARDX_EXEC int Coeff3D_RLR_Offset_5x5x5(SIZE i) {
  static constexpr int offset[8] = {offset5x5x5(c3d_xyz_5x5x5[0][0]+1, c3d_xyz_5x5x5[0][1]-1, c3d_xyz_5x5x5[0][2]+1), // XY
                                    offset5x5x5(c3d_xyz_5x5x5[1][0]+1, c3d_xyz_5x5x5[1][1]-1, c3d_xyz_5x5x5[1][2]+1),
                                    offset5x5x5(c3d_xyz_5x5x5[2][0]+1, c3d_xyz_5x5x5[2][1]-1, c3d_xyz_5x5x5[2][2]+1),
                                    offset5x5x5(c3d_xyz_5x5x5[3][0]+1, c3d_xyz_5x5x5[3][1]-1, c3d_xyz_5x5x5[3][2]+1),
                                    offset5x5x5(c3d_xyz_5x5x5[4][0]+1, c3d_xyz_5x5x5[4][1]-1, c3d_xyz_5x5x5[4][2]+1),
                                    offset5x5x5(c3d_xyz_5x5x5[5][0]+1, c3d_xyz_5x5x5[5][1]-1, c3d_xyz_5x5x5[5][2]+1),
                                    offset5x5x5(c3d_xyz_5x5x5[6][0]+1, c3d_xyz_5x5x5[6][1]-1, c3d_xyz_5x5x5[6][2]+1),
                                    offset5x5x5(c3d_xyz_5x5x5[7][0]+1, c3d_xyz_5x5x5[7][1]-1, c3d_xyz_5x5x5[7][2]+1)};

  return offset[i];
}

MGARDX_EXEC int Coeff3D_RRL_Offset_5x5x5(SIZE i) {
  static constexpr int offset[8] = {offset5x5x5(c3d_xyz_5x5x5[0][0]+1, c3d_xyz_5x5x5[0][1]+1, c3d_xyz_5x5x5[0][2]-1), // XY
                                    offset5x5x5(c3d_xyz_5x5x5[1][0]+1, c3d_xyz_5x5x5[1][1]+1, c3d_xyz_5x5x5[1][2]-1),
                                    offset5x5x5(c3d_xyz_5x5x5[2][0]+1, c3d_xyz_5x5x5[2][1]+1, c3d_xyz_5x5x5[2][2]-1),
                                    offset5x5x5(c3d_xyz_5x5x5[3][0]+1, c3d_xyz_5x5x5[3][1]+1, c3d_xyz_5x5x5[3][2]-1),
                                    offset5x5x5(c3d_xyz_5x5x5[4][0]+1, c3d_xyz_5x5x5[4][1]+1, c3d_xyz_5x5x5[4][2]-1),
                                    offset5x5x5(c3d_xyz_5x5x5[5][0]+1, c3d_xyz_5x5x5[5][1]+1, c3d_xyz_5x5x5[5][2]-1),
                                    offset5x5x5(c3d_xyz_5x5x5[6][0]+1, c3d_xyz_5x5x5[6][1]+1, c3d_xyz_5x5x5[6][2]-1),
                                    offset5x5x5(c3d_xyz_5x5x5[7][0]+1, c3d_xyz_5x5x5[7][1]+1, c3d_xyz_5x5x5[7][2]-1)};

  return offset[i];
}

MGARDX_EXEC int Coeff3D_RRR_Offset_5x5x5(SIZE i) {
  static constexpr int offset[8] = {offset5x5x5(c3d_xyz_5x5x5[0][0]+1, c3d_xyz_5x5x5[0][1]+1, c3d_xyz_5x5x5[0][2]+1), // XY
                                    offset5x5x5(c3d_xyz_5x5x5[1][0]+1, c3d_xyz_5x5x5[1][1]+1, c3d_xyz_5x5x5[1][2]+1),
                                    offset5x5x5(c3d_xyz_5x5x5[2][0]+1, c3d_xyz_5x5x5[2][1]+1, c3d_xyz_5x5x5[2][2]+1),
                                    offset5x5x5(c3d_xyz_5x5x5[3][0]+1, c3d_xyz_5x5x5[3][1]+1, c3d_xyz_5x5x5[3][2]+1),
                                    offset5x5x5(c3d_xyz_5x5x5[4][0]+1, c3d_xyz_5x5x5[4][1]+1, c3d_xyz_5x5x5[4][2]+1),
                                    offset5x5x5(c3d_xyz_5x5x5[5][0]+1, c3d_xyz_5x5x5[5][1]+1, c3d_xyz_5x5x5[5][2]+1),
                                    offset5x5x5(c3d_xyz_5x5x5[6][0]+1, c3d_xyz_5x5x5[6][1]+1, c3d_xyz_5x5x5[6][2]+1),
                                    offset5x5x5(c3d_xyz_5x5x5[7][0]+1, c3d_xyz_5x5x5[7][1]+1, c3d_xyz_5x5x5[7][2]+1)};

  return offset[i];
}

MGARDX_EXEC int const *MassTrans_X_Offset_5x5x5(SIZE i) {
  static constexpr int zero_offset = 0;
  #define OFFSET(Z, Y)                             \
  {                                                \
    zero_offset,                                   \
    zero_offset,                                   \
    offset5x5x5(Z, Y, coarse_x_5x5x5[0],   5, 5),  \
    offset5x5x5(Z, Y, coarse_x_5x5x5[0]+1, 5, 5),  \
    offset5x5x5(Z, Y, coarse_x_5x5x5[0]+2, 5, 5),  \
    offset5x5x5(Z, Y, coarse_x_5x5x5[0],   3, 5)   \
  },                                               \
  {                                                \
    offset5x5x5(Z, Y, coarse_x_5x5x5[1]-2, 5, 5),  \
    offset5x5x5(Z, Y, coarse_x_5x5x5[1]-1, 5, 5),  \
    offset5x5x5(Z, Y, coarse_x_5x5x5[1],   5, 5),  \
    offset5x5x5(Z, Y, coarse_x_5x5x5[1]+1, 5, 5),  \
    offset5x5x5(Z, Y, coarse_x_5x5x5[1]+2, 5, 5),  \
    offset5x5x5(Z, Y, coarse_x_5x5x5[1],   3, 5)   \
  },                                               \
  {                                                \
    offset5x5x5(Z, Y, coarse_x_5x5x5[2]-2, 5, 5),  \
    offset5x5x5(Z, Y, coarse_x_5x5x5[2]-1, 5, 5),  \
    offset5x5x5(Z, Y, coarse_x_5x5x5[2],   5, 5),  \
    zero_offset,                                   \
    zero_offset,                                   \
    offset5x5x5(Z, Y, coarse_x_5x5x5[2],   3, 5)   \
  }

  static constexpr int offset[75][6] = {
    OFFSET(0, 0), OFFSET(0, 1), OFFSET(0, 2), OFFSET(0, 3), OFFSET(0, 4),
    OFFSET(1, 0), OFFSET(1, 1), OFFSET(1, 2), OFFSET(1, 3), OFFSET(1, 4),
    OFFSET(2, 0), OFFSET(2, 1), OFFSET(2, 2), OFFSET(2, 3), OFFSET(2, 4),
    OFFSET(3, 0), OFFSET(3, 1), OFFSET(3, 2), OFFSET(3, 3), OFFSET(3, 4),
    OFFSET(4, 0), OFFSET(4, 1), OFFSET(4, 2), OFFSET(4, 3), OFFSET(4, 4)
  };
  #undef OFFSET
  return offset[i];
}

MGARDX_EXEC int const *MassTrans_Y_Offset_5x5x5(SIZE i) {
  static constexpr int zero_offset = 0;
  #define OFFSET(Z, X)                             \
  {                                                \
    zero_offset,                                   \
    zero_offset,                                   \
    offset5x5x5(Z, coarse_y_5x5x5[0],   X, 3, 5),  \
    offset5x5x5(Z, coarse_y_5x5x5[0]+1, X, 3, 5),  \
    offset5x5x5(Z, coarse_y_5x5x5[0]+2, X, 3, 5),  \
    offset5x5x5(Z, coarse_y_5x5x5[0],   X, 3, 3)   \
  },                                               \
  {                                                \
    offset5x5x5(Z, coarse_y_5x5x5[1]-2, X, 3, 5),  \
    offset5x5x5(Z, coarse_y_5x5x5[1]-1, X, 3, 5),  \
    offset5x5x5(Z, coarse_y_5x5x5[1],   X, 3, 5),  \
    offset5x5x5(Z, coarse_y_5x5x5[1]+1, X, 3, 5),  \
    offset5x5x5(Z, coarse_y_5x5x5[1]+2, X, 3, 5),  \
    offset5x5x5(Z, coarse_y_5x5x5[1],   X, 3, 3)   \
  },                                               \
  {                                                \
    offset5x5x5(Z, coarse_y_5x5x5[2]-2, X, 3, 5),  \
    offset5x5x5(Z, coarse_y_5x5x5[2]-1, X, 3, 5),  \
    offset5x5x5(Z, coarse_y_5x5x5[2],   X, 3, 5),  \
    zero_offset,                                   \
    zero_offset,                                   \
    offset5x5x5(Z, coarse_y_5x5x5[2],   X, 3, 3)   \
  }

  static constexpr int offset[45][6] = {
    OFFSET(0, 0), OFFSET(0, 1), OFFSET(0, 2), 
    OFFSET(1, 0), OFFSET(1, 1), OFFSET(1, 2), 
    OFFSET(2, 0), OFFSET(2, 1), OFFSET(2, 2), 
    OFFSET(3, 0), OFFSET(3, 1), OFFSET(3, 2), 
    OFFSET(4, 0), OFFSET(4, 1), OFFSET(4, 2)
  };
  #undef OFFSET
  return offset[i];
}

MGARDX_EXEC int const *MassTrans_Z_Offset_5x5x5(SIZE i) {
  static constexpr int zero_offset = 0;
  #define OFFSET(Y, X)                             \
  {                                                \
    zero_offset,                                   \
    zero_offset,                                   \
    offset5x5x5(coarse_z_5x5x5[0],   Y, X, 3, 3),  \
    offset5x5x5(coarse_z_5x5x5[0]+1, Y, X, 3, 3),  \
    offset5x5x5(coarse_z_5x5x5[0]+2, Y, X, 3, 3),  \
    offset5x5x5(coarse_z_5x5x5[0],   Y, X, 3, 3)   \
  },                                               \
  {                                                \
    offset5x5x5(coarse_z_5x5x5[1]-2, Y, X, 3, 3),  \
    offset5x5x5(coarse_z_5x5x5[1]-1, Y, X, 3, 3),  \
    offset5x5x5(coarse_z_5x5x5[1],   Y, X, 3, 3),  \
    offset5x5x5(coarse_z_5x5x5[1]+1, Y, X, 3, 3),  \
    offset5x5x5(coarse_z_5x5x5[1]+2, Y, X, 3, 3),  \
    offset5x5x5(coarse_z_5x5x5[1],   Y, X, 3, 3)   \
  },                                               \
  {                                                \
    offset5x5x5(coarse_z_5x5x5[2]-2, Y, X, 3, 3),  \
    offset5x5x5(coarse_z_5x5x5[2]-1, Y, X, 3, 3),  \
    offset5x5x5(coarse_z_5x5x5[2],   Y, X, 3, 3),  \
    zero_offset,                                   \
    zero_offset,                                   \
    offset5x5x5(coarse_z_5x5x5[2],   Y, X, 3, 3)   \
  }

  static constexpr int offset[27][6] = {
    OFFSET(0, 0), OFFSET(0, 1), OFFSET(0, 2),
    OFFSET(1, 0), OFFSET(1, 1), OFFSET(1, 2),
    OFFSET(2, 0), OFFSET(2, 1), OFFSET(2, 2)
  };
  #undef OFFSET
  return offset[i];
}

MGARDX_EXEC int const *TriDiag_X_Offset_5x5x5(SIZE i) {
  static constexpr int offset[9][3] = {
    {offset5x5x5(0, 0, 0, 3, 3), offset5x5x5(0, 0, 1, 3, 3), offset5x5x5(0, 0, 2, 3, 3)},
    {offset5x5x5(0, 1, 0, 3, 3), offset5x5x5(0, 1, 1, 3, 3), offset5x5x5(0, 1, 2, 3, 3)},
    {offset5x5x5(0, 2, 0, 3, 3), offset5x5x5(0, 2, 1, 3, 3), offset5x5x5(0, 2, 2, 3, 3)},

    {offset5x5x5(1, 0, 0, 3, 3), offset5x5x5(1, 0, 1, 3, 3), offset5x5x5(1, 0, 2, 3, 3)},
    {offset5x5x5(1, 1, 0, 3, 3), offset5x5x5(1, 1, 1, 3, 3), offset5x5x5(1, 1, 2, 3, 3)},
    {offset5x5x5(1, 2, 0, 3, 3), offset5x5x5(1, 2, 1, 3, 3), offset5x5x5(1, 2, 2, 3, 3)},

    {offset5x5x5(2, 0, 0, 3, 3), offset5x5x5(2, 0, 1, 3, 3), offset5x5x5(2, 0, 2, 3, 3)},
    {offset5x5x5(2, 1, 0, 3, 3), offset5x5x5(2, 1, 1, 3, 3), offset5x5x5(2, 1, 2, 3, 3)},
    {offset5x5x5(2, 2, 0, 3, 3), offset5x5x5(2, 2, 1, 3, 3), offset5x5x5(2, 2, 2, 3, 3)}
  };
  return offset[i];
}

MGARDX_EXEC int const *TriDiag_Y_Offset_5x5x5(SIZE i) {
  static constexpr int offset[9][3] = {
    {offset5x5x5(0, 0, 0, 3, 3), offset5x5x5(0, 1, 0, 3, 3), offset5x5x5(0, 2, 0, 3, 3)},
    {offset5x5x5(0, 0, 1, 3, 3), offset5x5x5(0, 1, 1, 3, 3), offset5x5x5(0, 2, 1, 3, 3)},
    {offset5x5x5(0, 0, 2, 3, 3), offset5x5x5(0, 1, 2, 3, 3), offset5x5x5(0, 2, 2, 3, 3)},

    {offset5x5x5(1, 0, 0, 3, 3), offset5x5x5(1, 1, 0, 3, 3), offset5x5x5(1, 2, 0, 3, 3)},
    {offset5x5x5(1, 0, 1, 3, 3), offset5x5x5(1, 1, 1, 3, 3), offset5x5x5(1, 2, 1, 3, 3)},
    {offset5x5x5(1, 0, 2, 3, 3), offset5x5x5(1, 1, 2, 3, 3), offset5x5x5(1, 2, 2, 3, 3)},

    {offset5x5x5(2, 0, 0, 3, 3), offset5x5x5(2, 1, 0, 3, 3), offset5x5x5(2, 2, 0, 3, 3)},
    {offset5x5x5(2, 0, 1, 3, 3), offset5x5x5(2, 1, 1, 3, 3), offset5x5x5(2, 2, 1, 3, 3)},
    {offset5x5x5(2, 0, 2, 3, 3), offset5x5x5(2, 1, 2, 3, 3), offset5x5x5(2, 2, 2, 3, 3)}
  };
  return offset[i];
}

MGARDX_EXEC int const *TriDiag_Z_Offset_5x5x5(SIZE i) {
  static constexpr int offset[9][3] = {
    {offset5x5x5(0, 0, 0, 3, 3), offset5x5x5(1, 0, 0, 3, 3), offset5x5x5(2, 0, 0, 3, 3)},
    {offset5x5x5(0, 0, 1, 3, 3), offset5x5x5(1, 0, 1, 3, 3), offset5x5x5(2, 0, 1, 3, 3)},
    {offset5x5x5(0, 0, 2, 3, 3), offset5x5x5(1, 0, 2, 3, 3), offset5x5x5(2, 0, 2, 3, 3)},

    {offset5x5x5(0, 1, 0, 3, 3), offset5x5x5(1, 1, 0, 3, 3), offset5x5x5(2, 1, 0, 3, 3)},
    {offset5x5x5(0, 1, 1, 3, 3), offset5x5x5(1, 1, 1, 3, 3), offset5x5x5(2, 1, 1, 3, 3)},
    {offset5x5x5(0, 1, 2, 3, 3), offset5x5x5(1, 1, 2, 3, 3), offset5x5x5(2, 1, 2, 3, 3)},

    {offset5x5x5(0, 2, 0, 3, 3), offset5x5x5(1, 2, 0, 3, 3), offset5x5x5(2, 2, 0, 3, 3)},
    {offset5x5x5(0, 2, 1, 3, 3), offset5x5x5(1, 2, 1, 3, 3), offset5x5x5(2, 2, 1, 3, 3)},
    {offset5x5x5(0, 2, 2, 3, 3), offset5x5x5(1, 2, 2, 3, 3), offset5x5x5(2, 2, 2, 3, 3)}
  };
  return offset[i];
}

MGARDX_EXEC int Coarse_Offset_5x5x5(SIZE i) {
  static constexpr int offset[27] = {
    offset8x8x8(0, 0, 0, 5, 5), offset8x8x8(0, 0, 2, 5, 5), offset8x8x8(0, 0, 4, 5, 5), 
    offset8x8x8(0, 2, 0, 5, 5), offset8x8x8(0, 2, 2, 5, 5), offset8x8x8(0, 2, 4, 5, 5), 
    offset8x8x8(0, 4, 0, 5, 5), offset8x8x8(0, 4, 2, 5, 5), offset8x8x8(0, 4, 4, 5, 5), 

    offset8x8x8(2, 0, 0, 5, 5), offset8x8x8(2, 0, 2, 5, 5), offset8x8x8(2, 0, 4, 5, 5), 
    offset8x8x8(2, 2, 0, 5, 5), offset8x8x8(2, 2, 2, 5, 5), offset8x8x8(2, 2, 4, 5, 5), 
    offset8x8x8(2, 4, 0, 5, 5), offset8x8x8(2, 4, 2, 5, 5), offset8x8x8(2, 4, 4, 5, 5), 

    offset8x8x8(4, 0, 0, 5, 5), offset8x8x8(4, 0, 2, 5, 5), offset8x8x8(4, 0, 4, 5, 5), 
    offset8x8x8(4, 2, 0, 5, 5), offset8x8x8(4, 2, 2, 5, 5), offset8x8x8(4, 2, 4, 5, 5), 
    offset8x8x8(4, 4, 0, 5, 5), offset8x8x8(4, 4, 2, 5, 5), offset8x8x8(4, 4, 4, 5, 5), 

  };
  return offset[i];
}

// clang-format on
} // namespace mgard_x