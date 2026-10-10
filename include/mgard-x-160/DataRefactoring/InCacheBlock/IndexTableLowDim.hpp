/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 */

#ifndef MGARD_X_160_INDEX_TABLE_LOW_DIM_HPP
#define MGARD_X_160_INDEX_TABLE_LOW_DIM_HPP

#include "IndexTable8x8x8.hpp"

namespace mgard_x_160 {

// ---------------------------------------------------------------------------
// Index tables for the 1D (8) and 2D (8x8) in-cache block transforms.
//
// The block geometry is the same in every dimension as the 3D block: an
// 8-wide block coarsens to the 5 nodes {0, 2, 4, 6, 7}, and the trailing cell
// [6, 7] is treated as two half cells around a phantom node at 6.5. So the
// coarse-grid spacing is {2, 2, 2, 1} and the mass-transform weight cases are
// identical to the 3D block's -- am_8x8x8 / bm_8x8x8 /
// MassTrans_Weights_8x8x8 are reused here verbatim rather than duplicated.
//
// IndexTable8x8x8.hpp spells out every table entry by hand. The low-dimension
// tables are two orders of magnitude smaller, so they are built with constexpr
// loops instead: the geometry rules are stated once below and the compiler
// expands them into the same kind of `static constexpr` lookup tables.
// ---------------------------------------------------------------------------

// Minimal constexpr table containers. Not std::array: these are read from
// device code, where std::array's constexpr accessors are not portably usable.
template <int N, int K> struct LowDimTable { int v[N][K]; };

static constexpr int LowDim_Block = 8;   // fine nodes per block per dim
static constexpr int LowDim_Coarse = 5;  // coarse nodes per block per dim
static constexpr int LowDim_Coeff1D = 3; // coefficient nodes per dim

static constexpr int lowdim_coarse_idx[LowDim_Coarse] = {0, 2, 4, 6, 7};
static constexpr int lowdim_coeff_idx[LowDim_Coeff1D] = {1, 3, 5};

MGARDX_CONT_EXEC constexpr bool lowdim_is_coarse(int i) {
  return i == 0 || i == 2 || i == 4 || i == 6 || i == 7;
}

// Fine-grid position feeding stencil slot `slot` (0..4, i.e. -2..+2 half-cell
// steps) of coarse node `j`, or -1 when that slot has no node -- either
// outside the block or the phantom node at 6.5.
//
// `masked` additionally drops fine nodes that are themselves coarse nodes.
// The correction is the mass matrix applied to the *coefficient* field, which
// is zero at coarse nodes; along a line whose other coordinates are all
// coarse, the fine nodes at coarse positions therefore contribute nothing.
// This mirrors the OFFSET1 (masked) / OFFSET2 (full) split of
// MassTrans_X_Offset_8x8x8.
MGARDX_CONT_EXEC constexpr int lowdim_mt_pos(int j, int slot, bool masked) {
  int p = -1;
  if (j == 3) {
    // Coarse node 6: slot 3 is the phantom node at 6.5, slot 4 is node 7.
    const int tail[5] = {4, 5, 6, -1, 7};
    p = tail[slot];
  } else if (j == 4) {
    // Coarse node 7: slot 1 is the phantom node at 6.5, nothing to the right.
    const int tail[5] = {6, -1, 7, -1, -1};
    p = tail[slot];
  } else {
    p = lowdim_coarse_idx[j] + slot - 2;
    if (p < 0 || p >= LowDim_Block) {
      p = -1;
    }
  }
  if (masked && p >= 0 && lowdim_is_coarse(p)) {
    p = -1;
  }
  return p;
}

// ---------------------------------------------------------------------------
// 2D: 8x8 block
//
// Shared memory, in one allocation, offsets below relative to sm_v:
//   sm_v [8][9]  the block itself; the x-row length is padded from 8 to 9 so
//                the power-of-two coefficient strides scatter across the 32
//                banks, same rationale as SMV_LDY_8x8x8
//   sm_x [8][5]  after the x mass transform
//   sm_y [5][5]  after the y mass transform, then the tridiagonal solves
//   one trailing slot holding a constant zero, read by stencil slots that
//   have no node.
// ---------------------------------------------------------------------------
static constexpr int SMV_LDY_8x8 = 9;
static constexpr int SMV_SIZE_8x8 = 7 * SMV_LDY_8x8 + 7 + 1; // 71
static constexpr int SMX_SIZE_8x8 = LowDim_Block * LowDim_Coarse;
static constexpr int SMY_SIZE_8x8 = LowDim_Coarse * LowDim_Coarse;
static constexpr int SM_SIZE_8x8 =
    SMV_SIZE_8x8 + SMX_SIZE_8x8 + SMY_SIZE_8x8 + 1;
// The one zero slot, addressed relative to sm_v and to sm_x respectively.
static constexpr int ZERO_V_8x8 = SMV_SIZE_8x8 + SMX_SIZE_8x8 + SMY_SIZE_8x8;
static constexpr int ZERO_X_8x8 = SMX_SIZE_8x8 + SMY_SIZE_8x8;

static constexpr int NumCoarse_8x8 = LowDim_Coarse * LowDim_Coarse; // 25
static constexpr int NumCoeff_8x8 =
    LowDim_Block * LowDim_Block - NumCoarse_8x8;                          // 39
static constexpr int NumCoeff1D_8x8 = 2 * LowDim_Coarse * LowDim_Coeff1D; // 30
static constexpr int NumCoeff2D_8x8 = LowDim_Coeff1D * LowDim_Coeff1D;    // 9
static constexpr int NumMassTransX_8x8 = LowDim_Block * LowDim_Coarse;    // 40
static constexpr int NumMassTransY_8x8 = LowDim_Coarse * LowDim_Coarse;   // 25

MGARDX_CONT_EXEC constexpr int offset8x8(int y, int x) {
  return y * SMV_LDY_8x8 + x;
}

// Interpolation, 1D coefficients: {middle, left, right} sm_v offsets.
// First the 15 x-direction coefficients (on coarse rows), then the 15
// y-direction ones (on coarse columns).
MGARDX_CONT_EXEC constexpr LowDimTable<NumCoeff1D_8x8, 3> make_coeff1d_8x8() {
  LowDimTable<NumCoeff1D_8x8, 3> t{};
  int n = 0;
  for (int iy = 0; iy < LowDim_Coarse; iy++) {
    for (int j = 0; j < LowDim_Coeff1D; j++) {
      int y = lowdim_coarse_idx[iy];
      int x = lowdim_coeff_idx[j];
      t.v[n][0] = offset8x8(y, x);
      t.v[n][1] = offset8x8(y, x - 1);
      t.v[n][2] = offset8x8(y, x + 1);
      n++;
    }
  }
  for (int j = 0; j < LowDim_Coeff1D; j++) {
    for (int ix = 0; ix < LowDim_Coarse; ix++) {
      int y = lowdim_coeff_idx[j];
      int x = lowdim_coarse_idx[ix];
      t.v[n][0] = offset8x8(y, x);
      t.v[n][1] = offset8x8(y - 1, x);
      t.v[n][2] = offset8x8(y + 1, x);
      n++;
    }
  }
  return t;
}

MGARDX_EXEC int const *Coeff1D_Offset_8x8(SIZE i) {
  static constexpr LowDimTable<NumCoeff1D_8x8, 3> t = make_coeff1d_8x8();
  return t.v[i];
}

// Interpolation, 2D coefficients: {middle, and the four surrounding corners}.
MGARDX_CONT_EXEC constexpr LowDimTable<NumCoeff2D_8x8, 5> make_coeff2d_8x8() {
  LowDimTable<NumCoeff2D_8x8, 5> t{};
  int n = 0;
  for (int jy = 0; jy < LowDim_Coeff1D; jy++) {
    for (int jx = 0; jx < LowDim_Coeff1D; jx++) {
      int y = lowdim_coeff_idx[jy];
      int x = lowdim_coeff_idx[jx];
      t.v[n][0] = offset8x8(y, x);
      t.v[n][1] = offset8x8(y - 1, x - 1);
      t.v[n][2] = offset8x8(y - 1, x + 1);
      t.v[n][3] = offset8x8(y + 1, x - 1);
      t.v[n][4] = offset8x8(y + 1, x + 1);
      n++;
    }
  }
  return t;
}

MGARDX_EXEC int const *Coeff2D_Offset_8x8(SIZE i) {
  static constexpr LowDimTable<NumCoeff2D_8x8, 5> t = make_coeff2d_8x8();
  return t.v[i];
}

// X mass transform: {5 sm_v inputs, sm_x output, weight case}, one row per
// (fine y, coarse x). Rows on coarse y are masked, as explained above.
MGARDX_CONT_EXEC constexpr LowDimTable<NumMassTransX_8x8, 7>
make_masstrans_x_8x8() {
  LowDimTable<NumMassTransX_8x8, 7> t{};
  for (int y = 0; y < LowDim_Block; y++) {
    bool masked = lowdim_is_coarse(y);
    for (int j = 0; j < LowDim_Coarse; j++) {
      int n = y * LowDim_Coarse + j;
      for (int s = 0; s < 5; s++) {
        int p = lowdim_mt_pos(j, s, masked);
        t.v[n][s] = (p < 0) ? ZERO_V_8x8 : offset8x8(y, p);
      }
      t.v[n][5] = y * LowDim_Coarse + j;
      t.v[n][6] = j;
    }
  }
  return t;
}

MGARDX_EXEC int const *MassTrans_X_Offset_8x8(SIZE i) {
  static constexpr LowDimTable<NumMassTransX_8x8, 7> t = make_masstrans_x_8x8();
  return t.v[i];
}

// Y mass transform: {5 sm_x inputs, sm_y output, weight case}, one row per
// (coarse y, coarse x). Never masked: the x pass already zeroed the coarse
// nodes' contribution, and the remaining passes are plain 1D mass transforms.
MGARDX_CONT_EXEC constexpr LowDimTable<NumMassTransY_8x8, 7>
make_masstrans_y_8x8() {
  LowDimTable<NumMassTransY_8x8, 7> t{};
  for (int jy = 0; jy < LowDim_Coarse; jy++) {
    for (int jx = 0; jx < LowDim_Coarse; jx++) {
      int n = jy * LowDim_Coarse + jx;
      for (int s = 0; s < 5; s++) {
        int p = lowdim_mt_pos(jy, s, false);
        t.v[n][s] = (p < 0) ? ZERO_X_8x8 : (p * LowDim_Coarse + jx);
      }
      t.v[n][5] = jy * LowDim_Coarse + jx;
      t.v[n][6] = jy;
    }
  }
  return t;
}

MGARDX_EXEC int const *MassTrans_Y_Offset_8x8(SIZE i) {
  static constexpr LowDimTable<NumMassTransY_8x8, 7> t = make_masstrans_y_8x8();
  return t.v[i];
}

// Tridiagonal solves over sm_y: one line per row (x solve) or column (y solve).
MGARDX_CONT_EXEC constexpr LowDimTable<LowDim_Coarse, LowDim_Coarse>
make_tridiag_x_8x8() {
  LowDimTable<LowDim_Coarse, LowDim_Coarse> t{};
  for (int jy = 0; jy < LowDim_Coarse; jy++) {
    for (int k = 0; k < LowDim_Coarse; k++) {
      t.v[jy][k] = jy * LowDim_Coarse + k;
    }
  }
  return t;
}

MGARDX_EXEC int const *TriDiag_X_Offset_8x8(SIZE i) {
  static constexpr LowDimTable<LowDim_Coarse, LowDim_Coarse> t =
      make_tridiag_x_8x8();
  return t.v[i];
}

MGARDX_CONT_EXEC constexpr LowDimTable<LowDim_Coarse, LowDim_Coarse>
make_tridiag_y_8x8() {
  LowDimTable<LowDim_Coarse, LowDim_Coarse> t{};
  for (int jx = 0; jx < LowDim_Coarse; jx++) {
    for (int k = 0; k < LowDim_Coarse; k++) {
      t.v[jx][k] = k * LowDim_Coarse + jx;
    }
  }
  return t;
}

MGARDX_EXEC int const *TriDiag_Y_Offset_8x8(SIZE i) {
  static constexpr LowDimTable<LowDim_Coarse, LowDim_Coarse> t =
      make_tridiag_y_8x8();
  return t.v[i];
}

// sm_v offsets of the 25 coarse nodes, in the same row-major order as sm_y,
// so the correction at sm_y[i] belongs to Coarse_Offset_8x8(i).
MGARDX_CONT_EXEC constexpr LowDimTable<NumCoarse_8x8, 1> make_coarse_8x8() {
  LowDimTable<NumCoarse_8x8, 1> t{};
  for (int jy = 0; jy < LowDim_Coarse; jy++) {
    for (int jx = 0; jx < LowDim_Coarse; jx++) {
      t.v[jy * LowDim_Coarse + jx][0] =
          offset8x8(lowdim_coarse_idx[jy], lowdim_coarse_idx[jx]);
    }
  }
  return t;
}

MGARDX_EXEC int Coarse_Offset_8x8(SIZE i) {
  static constexpr LowDimTable<NumCoarse_8x8, 1> t = make_coarse_8x8();
  return t.v[i][0];
}

// sm_v offsets of the 39 coefficient nodes, enumerated row-major over the
// non-coarse positions -- the 2D analogue of Coeff_Offset_8x8x8's ordering.
// This ordering is part of the compressed layout: it is what the ROI
// per-block quantizer indexing and the recompose kernel both assume.
MGARDX_CONT_EXEC constexpr LowDimTable<NumCoeff_8x8, 1> make_coeff_8x8() {
  LowDimTable<NumCoeff_8x8, 1> t{};
  int n = 0;
  for (int y = 0; y < LowDim_Block; y++) {
    for (int x = 0; x < LowDim_Block; x++) {
      if (lowdim_is_coarse(y) && lowdim_is_coarse(x)) {
        continue;
      }
      t.v[n][0] = offset8x8(y, x);
      n++;
    }
  }
  return t;
}

MGARDX_EXEC int Coeff_Offset_8x8(SIZE i) {
  static constexpr LowDimTable<NumCoeff_8x8, 1> t = make_coeff_8x8();
  return t.v[i][0];
}

// ---------------------------------------------------------------------------
// 1D: 8-element block
//
// A single 8-element block is far too little work for a thread block, so the
// 1D kernels give each thread block LowDim_Tiles_1D independent tiles, laid
// out side by side in shared memory:
//   sm_v [tiles][9]  8 values plus a per-tile zero slot at index 8
//   sm_x [tiles][5]  after the mass transform, then the tridiagonal solve
// A per-tile zero slot (rather than one shared slot) keeps every table entry
// a pure within-tile offset, so the kernels can add tile * stride uniformly.
// ---------------------------------------------------------------------------
static constexpr SIZE LowDim_Tiles_1D = 32;
static constexpr int SMV_STRIDE_8 = LowDim_Block + 1; // 9
static constexpr int SMV_ZERO_8 = LowDim_Block;       // per-tile zero slot
static constexpr int SMX_STRIDE_8 = LowDim_Coarse;    // 5
static constexpr SIZE SM_SIZE_8 =
    LowDim_Tiles_1D * (SMV_STRIDE_8 + SMX_STRIDE_8);

static constexpr int NumCoarse_8 = LowDim_Coarse;               // 5
static constexpr int NumCoeff_8 = LowDim_Block - LowDim_Coarse; // 3
static constexpr int NumMassTransX_8 = LowDim_Coarse;           // 5

// Interpolation: {middle, left, right} within-tile sm_v offsets.
MGARDX_CONT_EXEC constexpr LowDimTable<NumCoeff_8, 3> make_coeff1d_8() {
  LowDimTable<NumCoeff_8, 3> t{};
  for (int j = 0; j < NumCoeff_8; j++) {
    int x = lowdim_coeff_idx[j];
    t.v[j][0] = x;
    t.v[j][1] = x - 1;
    t.v[j][2] = x + 1;
  }
  return t;
}

MGARDX_EXEC int const *Coeff1D_Offset_8(SIZE i) {
  static constexpr LowDimTable<NumCoeff_8, 3> t = make_coeff1d_8();
  return t.v[i];
}

// Mass transform: {5 sm_v inputs, sm_x output, weight case}. Always masked --
// in 1D every line is a line of coarse nodes.
MGARDX_CONT_EXEC constexpr LowDimTable<NumMassTransX_8, 7>
make_masstrans_x_8() {
  LowDimTable<NumMassTransX_8, 7> t{};
  for (int j = 0; j < LowDim_Coarse; j++) {
    for (int s = 0; s < 5; s++) {
      int p = lowdim_mt_pos(j, s, true);
      t.v[j][s] = (p < 0) ? SMV_ZERO_8 : p;
    }
    t.v[j][5] = j;
    t.v[j][6] = j;
  }
  return t;
}

MGARDX_EXEC int const *MassTrans_X_Offset_8(SIZE i) {
  static constexpr LowDimTable<NumMassTransX_8, 7> t = make_masstrans_x_8();
  return t.v[i];
}

MGARDX_EXEC int Coarse_Offset_8(SIZE i) {
  static constexpr LowDimTable<NumCoarse_8, 1> t = {{{lowdim_coarse_idx[0]},
                                                     {lowdim_coarse_idx[1]},
                                                     {lowdim_coarse_idx[2]},
                                                     {lowdim_coarse_idx[3]},
                                                     {lowdim_coarse_idx[4]}}};
  return t.v[i][0];
}

MGARDX_EXEC int Coeff_Offset_8(SIZE i) {
  static constexpr LowDimTable<NumCoeff_8, 1> t = {
      {{lowdim_coeff_idx[0]}, {lowdim_coeff_idx[1]}, {lowdim_coeff_idx[2]}}};
  return t.v[i][0];
}

} // namespace mgard_x_160

#endif
