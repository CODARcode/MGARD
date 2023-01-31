/*
 * Copyright 2023, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: Jan. 15, 2023
 */

namespace mgard_x {
// clang-format off

MGARDX_EXEC constexpr int offset3x3x3(SIZE z, SIZE y, SIZE x) {
  return z * 3 * 3 + y * 3 + x;
}

MGARDX_EXEC constexpr int offset3x3x3(SIZE z, SIZE y, SIZE x, SIZE ld1, SIZE ld2) {
  return z * ld1 * ld2 + y * ld1 + x;
}

static constexpr int c1d_x_3x3x3[4][3] = {  {0, 0, 1}, // X
                                            {0, 2, 1},
                                            {2, 0, 1},
                                            {2, 2, 1}
                                          };

static constexpr int c1d_y_3x3x3[4][3] = {  {0, 1, 0},// Y
                                            {0, 1, 2},  
                                            {2, 1, 0},
                                            {2, 1, 2}
                                          };

static constexpr int c1d_z_3x3x3[4][3] = {  {1, 0, 0}, // Z
                                            {1, 0, 2},
                                            {1, 2, 0},
                                            {1, 2, 2}
                                          };

static constexpr int c2d_xy_3x3x3[2][3] = { {0, 1, 1}, // XY
                                            {2, 1, 1}
                                          };

static constexpr int c2d_yz_3x3x3[2][3] = { {1, 1, 0}, // YZ
                                            {1, 1, 2}
                                          };

static constexpr int c2d_xz_3x3x3[2][3] = { {1, 0, 1}, // XZ
                                            {1, 2, 1}
                                          };

static constexpr int c3d_xyz_3x3x3[1][3] = { {1, 1, 1}// XYZ
                                          };

static constexpr int coarse_x_3x3x3[2] = {0, 2};

static constexpr int coarse_y_3x3x3[2] = {0, 2};

static constexpr int coarse_z_3x3x3[2] = {0, 2};

MGARDX_EXEC int Coeff1D_M_Offset_3x3x3(SIZE i) {
  static constexpr int offset[12] = { offset3x3x3(c1d_x_3x3x3[0][0], c1d_x_3x3x3[0][1], c1d_x_3x3x3[0][2]), // X
                                      offset3x3x3(c1d_x_3x3x3[1][0], c1d_x_3x3x3[1][1], c1d_x_3x3x3[1][2]),
                                      offset3x3x3(c1d_x_3x3x3[2][0], c1d_x_3x3x3[2][1], c1d_x_3x3x3[2][2]),
                                      offset3x3x3(c1d_x_3x3x3[3][0], c1d_x_3x3x3[3][1], c1d_x_3x3x3[3][2]),

                                      offset3x3x3(c1d_y_3x3x3[0][0], c1d_y_3x3x3[0][1], c1d_y_3x3x3[0][2]), // Y
                                      offset3x3x3(c1d_y_3x3x3[1][0], c1d_y_3x3x3[1][1], c1d_y_3x3x3[1][2]),
                                      offset3x3x3(c1d_y_3x3x3[2][0], c1d_y_3x3x3[2][1], c1d_y_3x3x3[2][2]),
                                      offset3x3x3(c1d_y_3x3x3[3][0], c1d_y_3x3x3[3][1], c1d_y_3x3x3[3][2]),

                                      offset3x3x3(c1d_z_3x3x3[0][0], c1d_z_3x3x3[0][1], c1d_z_3x3x3[0][2]), // Z
                                      offset3x3x3(c1d_z_3x3x3[1][0], c1d_z_3x3x3[1][1], c1d_z_3x3x3[1][2]),
                                      offset3x3x3(c1d_z_3x3x3[2][0], c1d_z_3x3x3[2][1], c1d_z_3x3x3[2][2]),
                                      offset3x3x3(c1d_z_3x3x3[3][0], c1d_z_3x3x3[3][1], c1d_z_3x3x3[3][2])
                                    };
  return offset[i];
}

MGARDX_EXEC int Coeff1D_L_Offset_3x3x3(SIZE i) {
  static constexpr int offset[12] = { offset3x3x3(c1d_x_3x3x3[0][0], c1d_x_3x3x3[0][1], c1d_x_3x3x3[0][2]-1), // X
                                      offset3x3x3(c1d_x_3x3x3[1][0], c1d_x_3x3x3[1][1], c1d_x_3x3x3[1][2]-1),
                                      offset3x3x3(c1d_x_3x3x3[2][0], c1d_x_3x3x3[2][1], c1d_x_3x3x3[2][2]-1),
                                      offset3x3x3(c1d_x_3x3x3[3][0], c1d_x_3x3x3[3][1], c1d_x_3x3x3[3][2]-1),

                                      offset3x3x3(c1d_y_3x3x3[0][0], c1d_y_3x3x3[0][1]-1, c1d_y_3x3x3[0][2]), // Y
                                      offset3x3x3(c1d_y_3x3x3[1][0], c1d_y_3x3x3[1][1]-1, c1d_y_3x3x3[1][2]),
                                      offset3x3x3(c1d_y_3x3x3[2][0], c1d_y_3x3x3[2][1]-1, c1d_y_3x3x3[2][2]),
                                      offset3x3x3(c1d_y_3x3x3[3][0], c1d_y_3x3x3[3][1]-1, c1d_y_3x3x3[3][2]),

                                      offset3x3x3(c1d_z_3x3x3[0][0]-1, c1d_z_3x3x3[0][1], c1d_z_3x3x3[0][2]), // Z
                                      offset3x3x3(c1d_z_3x3x3[1][0]-1, c1d_z_3x3x3[1][1], c1d_z_3x3x3[1][2]),
                                      offset3x3x3(c1d_z_3x3x3[2][0]-1, c1d_z_3x3x3[2][1], c1d_z_3x3x3[2][2]),
                                      offset3x3x3(c1d_z_3x3x3[3][0]-1, c1d_z_3x3x3[3][1], c1d_z_3x3x3[3][2])
                                    };
  return offset[i];
}

MGARDX_EXEC int Coeff1D_R_Offset_3x3x3(SIZE i) {
  static constexpr int offset[12] = { offset3x3x3(c1d_x_3x3x3[0][0], c1d_x_3x3x3[0][1], c1d_x_3x3x3[0][2]+1), // X
                                      offset3x3x3(c1d_x_3x3x3[1][0], c1d_x_3x3x3[1][1], c1d_x_3x3x3[1][2]+1),
                                      offset3x3x3(c1d_x_3x3x3[2][0], c1d_x_3x3x3[2][1], c1d_x_3x3x3[2][2]+1),
                                      offset3x3x3(c1d_x_3x3x3[3][0], c1d_x_3x3x3[3][1], c1d_x_3x3x3[3][2]+1),

                                      offset3x3x3(c1d_y_3x3x3[0][0], c1d_y_3x3x3[0][1]+1, c1d_y_3x3x3[0][2]), // Y
                                      offset3x3x3(c1d_y_3x3x3[1][0], c1d_y_3x3x3[1][1]+1, c1d_y_3x3x3[1][2]),
                                      offset3x3x3(c1d_y_3x3x3[2][0], c1d_y_3x3x3[2][1]+1, c1d_y_3x3x3[2][2]),
                                      offset3x3x3(c1d_y_3x3x3[3][0], c1d_y_3x3x3[3][1]+1, c1d_y_3x3x3[3][2]),

                                      offset3x3x3(c1d_z_3x3x3[0][0]+1, c1d_z_3x3x3[0][1], c1d_z_3x3x3[0][2]), // Z
                                      offset3x3x3(c1d_z_3x3x3[1][0]+1, c1d_z_3x3x3[1][1], c1d_z_3x3x3[1][2]),
                                      offset3x3x3(c1d_z_3x3x3[2][0]+1, c1d_z_3x3x3[2][1], c1d_z_3x3x3[2][2]),
                                      offset3x3x3(c1d_z_3x3x3[3][0]+1, c1d_z_3x3x3[3][1], c1d_z_3x3x3[3][2])
                                    };
  return offset[i];
}

MGARDX_EXEC int Coeff2D_MM_Offset_3x3x3(SIZE i) {
  static constexpr int offset[6] = {  offset3x3x3(c2d_xy_3x3x3[0][0], c2d_xy_3x3x3[0][1], c2d_xy_3x3x3[0][2]), // XY
                                      offset3x3x3(c2d_xy_3x3x3[1][0], c2d_xy_3x3x3[1][1], c2d_xy_3x3x3[1][2]),

                                      offset3x3x3(c2d_xz_3x3x3[0][0], c2d_xz_3x3x3[0][1], c2d_xz_3x3x3[0][2]), // XZ
                                      offset3x3x3(c2d_xz_3x3x3[1][0], c2d_xz_3x3x3[1][1], c2d_xz_3x3x3[1][2]),

                                      offset3x3x3(c2d_yz_3x3x3[0][0], c2d_yz_3x3x3[0][1], c2d_yz_3x3x3[0][2]), // YZ
                                      offset3x3x3(c2d_yz_3x3x3[1][0], c2d_yz_3x3x3[1][1], c2d_yz_3x3x3[1][2])
                                      };
  return offset[i];
}

MGARDX_EXEC int Coeff2D_LL_Offset_3x3x3(SIZE i) {
  static constexpr int offset[6] = {  offset3x3x3(c2d_xy_3x3x3[0][0], c2d_xy_3x3x3[0][1]-1, c2d_xy_3x3x3[0][2]-1), // XY
                                      offset3x3x3(c2d_xy_3x3x3[1][0], c2d_xy_3x3x3[1][1]-1, c2d_xy_3x3x3[1][2]-1),

                                      offset3x3x3(c2d_xz_3x3x3[0][0]-1, c2d_xz_3x3x3[0][1], c2d_xz_3x3x3[0][2]-1), // XZ
                                      offset3x3x3(c2d_xz_3x3x3[1][0]-1, c2d_xz_3x3x3[1][1], c2d_xz_3x3x3[1][2]-1),

                                      offset3x3x3(c2d_yz_3x3x3[0][0]-1, c2d_yz_3x3x3[0][1]-1, c2d_yz_3x3x3[0][2]), // YZ
                                      offset3x3x3(c2d_yz_3x3x3[1][0]-1, c2d_yz_3x3x3[1][1]-1, c2d_yz_3x3x3[1][2])
                                      };
  return offset[i];
}

MGARDX_EXEC int Coeff2D_LR_Offset_3x3x3(SIZE i) {
  static constexpr int offset[6] = {  offset3x3x3(c2d_xy_3x3x3[0][0], c2d_xy_3x3x3[0][1]-1, c2d_xy_3x3x3[0][2]+1), // XY
                                      offset3x3x3(c2d_xy_3x3x3[1][0], c2d_xy_3x3x3[1][1]-1, c2d_xy_3x3x3[1][2]+1),

                                      offset3x3x3(c2d_xz_3x3x3[0][0]-1, c2d_xz_3x3x3[0][1], c2d_xz_3x3x3[0][2]+1), // XZ
                                      offset3x3x3(c2d_xz_3x3x3[1][0]-1, c2d_xz_3x3x3[1][1], c2d_xz_3x3x3[1][2]+1),

                                      offset3x3x3(c2d_yz_3x3x3[0][0]-1, c2d_yz_3x3x3[0][1]+1, c2d_yz_3x3x3[0][2]), // YZ
                                      offset3x3x3(c2d_yz_3x3x3[1][0]-1, c2d_yz_3x3x3[1][1]+1, c2d_yz_3x3x3[1][2])
                                      };
  return offset[i];
}

MGARDX_EXEC int Coeff2D_RL_Offset_3x3x3(SIZE i) {
  static constexpr int offset[6] = {  offset3x3x3(c2d_xy_3x3x3[0][0], c2d_xy_3x3x3[0][1]+1, c2d_xy_3x3x3[0][2]-1), // XY
                                      offset3x3x3(c2d_xy_3x3x3[1][0], c2d_xy_3x3x3[1][1]+1, c2d_xy_3x3x3[1][2]-1),

                                      offset3x3x3(c2d_xz_3x3x3[0][0]+1, c2d_xz_3x3x3[0][1], c2d_xz_3x3x3[0][2]-1), // XZ
                                      offset3x3x3(c2d_xz_3x3x3[1][0]+1, c2d_xz_3x3x3[1][1], c2d_xz_3x3x3[1][2]-1),

                                      offset3x3x3(c2d_yz_3x3x3[0][0]+1, c2d_yz_3x3x3[0][1]-1, c2d_yz_3x3x3[0][2]), // YZ
                                      offset3x3x3(c2d_yz_3x3x3[1][0]+1, c2d_yz_3x3x3[1][1]-1, c2d_yz_3x3x3[1][2]),
                                      };
  return offset[i];
}

MGARDX_EXEC int Coeff2D_RR_Offset_3x3x3(SIZE i) {
  static constexpr int offset[6] = {  offset3x3x3(c2d_xy_3x3x3[0][0], c2d_xy_3x3x3[0][1]+1, c2d_xy_3x3x3[0][2]+1), // XY
                                      offset3x3x3(c2d_xy_3x3x3[1][0], c2d_xy_3x3x3[1][1]+1, c2d_xy_3x3x3[1][2]+1),

                                      offset3x3x3(c2d_xz_3x3x3[0][0]+1, c2d_xz_3x3x3[0][1], c2d_xz_3x3x3[0][2]+1), // XZ
                                      offset3x3x3(c2d_xz_3x3x3[1][0]+1, c2d_xz_3x3x3[1][1], c2d_xz_3x3x3[1][2]+1),

                                      offset3x3x3(c2d_yz_3x3x3[0][0]+1, c2d_yz_3x3x3[0][1]+1, c2d_yz_3x3x3[0][2]), // YZ
                                      offset3x3x3(c2d_yz_3x3x3[1][0]+1, c2d_yz_3x3x3[1][1]+1, c2d_yz_3x3x3[1][2])
                                      };
  return offset[i];
}

MGARDX_EXEC int Coeff3D_MMM_Offset_3x3x3(SIZE i) {
  static constexpr int offset[1] = {offset3x3x3(c3d_xyz_3x3x3[0][0], c3d_xyz_3x3x3[0][1], c3d_xyz_3x3x3[0][2])};

  return offset[i];
}

MGARDX_EXEC int Coeff3D_LLL_Offset_3x3x3(SIZE i) {
  static constexpr int offset[1] = {offset3x3x3(c3d_xyz_3x3x3[0][0]-1, c3d_xyz_3x3x3[0][1]-1, c3d_xyz_3x3x3[0][2]-1)};

  return offset[i];
}

MGARDX_EXEC int Coeff3D_LLR_Offset_3x3x3(SIZE i) {
  static constexpr int offset[1] = {offset3x3x3(c3d_xyz_3x3x3[0][0]-1, c3d_xyz_3x3x3[0][1]-1, c3d_xyz_3x3x3[0][2]+1)};

  return offset[i];
}

MGARDX_EXEC int Coeff3D_LRL_Offset_3x3x3(SIZE i) {
  static constexpr int offset[1] = {offset3x3x3(c3d_xyz_3x3x3[0][0]-1, c3d_xyz_3x3x3[0][1]+1, c3d_xyz_3x3x3[0][2]-1)};

  return offset[i];
}

MGARDX_EXEC int Coeff3D_LRR_Offset_3x3x3(SIZE i) {
  static constexpr int offset[1] = {offset3x3x3(c3d_xyz_3x3x3[0][0]-1, c3d_xyz_3x3x3[0][1]+1, c3d_xyz_3x3x3[0][2]+1)};

  return offset[i];
}

MGARDX_EXEC int Coeff3D_RLL_Offset_3x3x3(SIZE i) {
  static constexpr int offset[1] = {offset3x3x3(c3d_xyz_3x3x3[0][0]+1, c3d_xyz_3x3x3[0][1]-1, c3d_xyz_3x3x3[0][2]-1)};

  return offset[i];
}

MGARDX_EXEC int Coeff3D_RLR_Offset_3x3x3(SIZE i) {
  static constexpr int offset[1] = {offset3x3x3(c3d_xyz_3x3x3[0][0]+1, c3d_xyz_3x3x3[0][1]-1, c3d_xyz_3x3x3[0][2]+1)};

  return offset[i];
}

MGARDX_EXEC int Coeff3D_RRL_Offset_3x3x3(SIZE i) {
  static constexpr int offset[1] = {offset3x3x3(c3d_xyz_3x3x3[0][0]+1, c3d_xyz_3x3x3[0][1]+1, c3d_xyz_3x3x3[0][2]-1)};

  return offset[i];
}

MGARDX_EXEC int Coeff3D_RRR_Offset_3x3x3(SIZE i) {
  static constexpr int offset[1] = {offset3x3x3(c3d_xyz_3x3x3[0][0]+1, c3d_xyz_3x3x3[0][1]+1, c3d_xyz_3x3x3[0][2]+1)};

  return offset[i];
}

MGARDX_EXEC int const *MassTrans_X_Offset_3x3x3(SIZE i) {
  static constexpr int zero_offset = 0;
  #define OFFSET(Z, Y)                             \
  {                                                \
    zero_offset,                                   \
    zero_offset,                                   \
    offset3x3x3(Z, Y, coarse_x_3x3x3[0],   3, 3),  \
    offset3x3x3(Z, Y, coarse_x_3x3x3[0]+1, 3, 3),  \
    offset3x3x3(Z, Y, coarse_x_3x3x3[0]+2, 3, 3),  \
    offset3x3x3(Z, Y, coarse_x_3x3x3[0],   2, 3)   \
  },                                               \
  {                                                \
    offset3x3x3(Z, Y, coarse_x_3x3x3[1]-2, 3, 3),  \
    offset3x3x3(Z, Y, coarse_x_3x3x3[1]-1, 3, 3),  \
    offset3x3x3(Z, Y, coarse_x_3x3x3[1],   3, 3),  \
    zero_offset,                                   \
    zero_offset,                                   \
    offset3x3x3(Z, Y, coarse_x_3x3x3[1],   2, 3)   \
  }

  static constexpr int offset[18][6] = {
    OFFSET(0, 0), OFFSET(0, 1), OFFSET(0, 2), 
    OFFSET(1, 0), OFFSET(1, 1), OFFSET(1, 2), 
    OFFSET(2, 0), OFFSET(2, 1), OFFSET(2, 2)
  };
  #undef OFFSET
  return offset[i];
}

MGARDX_EXEC int const *MassTrans_Y_Offset_3x3x3(SIZE i) {
  static constexpr int zero_offset = 0;
  #define OFFSET(Z, X)                             \
  {                                                \
    zero_offset,                                   \
    zero_offset,                                   \
    offset3x3x3(Z, coarse_y_3x3x3[0],   X, 2, 3),  \
    offset3x3x3(Z, coarse_y_3x3x3[0]+1, X, 2, 3),  \
    offset3x3x3(Z, coarse_y_3x3x3[0]+2, X, 2, 3),  \
    offset3x3x3(Z, coarse_y_3x3x3[0],   X, 2, 2)   \
  },                                               \
  {                                                \
    offset3x3x3(Z, coarse_y_3x3x3[1]-2, X, 3, 3),  \
    offset3x3x3(Z, coarse_y_3x3x3[1]-1, X, 3, 3),  \
    offset3x3x3(Z, coarse_y_3x3x3[1],   X, 3, 3),  \
    zero_offset,                                   \
    zero_offset,                                   \
    offset3x3x3(Z, coarse_y_3x3x3[1],   X, 2, 2)   \
  }

  static constexpr int offset[12][6] = {
    OFFSET(0, 0), OFFSET(0, 1),
    OFFSET(1, 0), OFFSET(1, 1),
    OFFSET(2, 0), OFFSET(2, 1)
  };
  #undef OFFSET
  return offset[i];
}

MGARDX_EXEC int const *MassTrans_Z_Offset_3x3x3(SIZE i) {
  static constexpr int zero_offset = 0;
  #define OFFSET(Y, X)                             \
  {                                                \
    zero_offset,                                   \
    zero_offset,                                   \
    offset3x3x3(coarse_z_3x3x3[0],   Y, X, 2, 2),  \
    offset3x3x3(coarse_z_3x3x3[0]+1, Y, X, 2, 2),  \
    offset3x3x3(coarse_z_3x3x3[0]+2, Y, X, 2, 2),  \
    offset3x3x3(coarse_z_3x3x3[0],   Y, X, 2, 2)   \
  },                                               \
  {                                                \
    offset3x3x3(coarse_z_3x3x3[1]-2, Y, X, 2, 2),  \
    offset3x3x3(coarse_z_3x3x3[1]-1, Y, X, 2, 2),  \
    offset3x3x3(coarse_z_3x3x3[1],   Y, X, 2, 2),  \
    zero_offset,                                   \
    zero_offset,                                   \
    offset3x3x3(coarse_z_3x3x3[1],   Y, X, 2, 2)   \
  }

  static constexpr int offset[8][6] = {
    OFFSET(0, 0), OFFSET(0, 1),
    OFFSET(1, 0), OFFSET(1, 1)
  };
  #undef OFFSET
  return offset[i];
}

MGARDX_EXEC int const *TriDiag_X_Offset_3x3x3(SIZE i) {
  static constexpr int offset[4][3] = {
    {offset3x3x3(0, 0, 0, 2, 2), offset3x3x3(0, 0, 1, 2, 2)},
    {offset3x3x3(0, 1, 0, 2, 2), offset3x3x3(0, 1, 1, 2, 2)},

    {offset3x3x3(1, 0, 0, 2, 2), offset3x3x3(1, 0, 1, 2, 2)},
    {offset3x3x3(1, 1, 0, 2, 2), offset3x3x3(1, 1, 1, 2, 2)}
  };
  return offset[i];
}

MGARDX_EXEC int const *TriDiag_Y_Offset_3x3x3(SIZE i) {
  static constexpr int offset[4][3] = {
    {offset3x3x3(0, 0, 0, 2, 2), offset3x3x3(0, 1, 0, 2, 2)},
    {offset3x3x3(0, 0, 1, 2, 2), offset3x3x3(0, 1, 1, 2, 2)},

    {offset3x3x3(1, 0, 0, 2, 2), offset3x3x3(1, 1, 0, 2, 2)},
    {offset3x3x3(1, 0, 1, 2, 2), offset3x3x3(1, 1, 1, 2, 2)}

  };
  return offset[i];
}

MGARDX_EXEC int const *TriDiag_Z_Offset_3x3x3(SIZE i) {
  static constexpr int offset[4][3] = {
    {offset3x3x3(0, 0, 0, 2, 2), offset3x3x3(1, 0, 0, 2, 2)},
    {offset3x3x3(0, 0, 1, 2, 2), offset3x3x3(1, 0, 1, 2, 2)},

    {offset3x3x3(0, 1, 0, 2, 2), offset3x3x3(1, 1, 0, 2, 2)},
    {offset3x3x3(0, 1, 1, 2, 2), offset3x3x3(1, 1, 1, 2, 2)}
  };
  return offset[i];
}

MGARDX_EXEC int Coarse_Offset_3x3x3(SIZE i) {
  static constexpr int offset[8] = {
    offset3x3x3(0, 0, 0, 3, 3), offset3x3x3(0, 0, 2, 3, 3),
    offset3x3x3(0, 2, 0, 3, 3), offset3x3x3(0, 2, 2, 3, 3),

    offset3x3x3(2, 0, 0, 3, 3), offset3x3x3(2, 0, 2, 3, 3),
    offset3x3x3(2, 2, 0, 3, 3), offset3x3x3(2, 2, 2, 3, 3)
  };
  return offset[i];
}

// clang-format on
} // namespace mgard_x