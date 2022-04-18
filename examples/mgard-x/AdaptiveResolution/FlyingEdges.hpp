/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_FLYING_EDGES_HPP
#define MGARD_X_FLYING_EDGES_HPP

#include "mgard/mgard-x/RuntimeX/RuntimeX.h"

namespace mgard_x {

#define MGARD_Below 0
#define MGARD_LeftAbove 1
#define MGARD_RightAbove 2

#define MGARD_Interior 0
#define MGARD_MinBoundary 1
#define MGARD_MaxBoundary 2

template <typename DeviceType>
MGARDX_EXEC bool computeTrimBounds(SIZE r, SIZE f, SIZE rightMax,
                                   SubArray<3, SIZE, DeviceType> &edges,
                                   SubArray<2, SIZE, DeviceType> &axis_min,
                                   SubArray<2, SIZE, DeviceType> &axis_max,
                                   SIZE &left, SIZE &right) {
  SIZE axis_mins[4] = {*axis_min(r, f), *axis_min(r, f + 1),
                       *axis_min(r + 1, f + 1), *axis_min(r + 1, f)};
  SIZE axis_maxs[4] = {*axis_max(r, f), *axis_max(r, f + 1),
                       *axis_max(r + 1, f + 1), *axis_max(r + 1, f)};

  left = min(axis_mins[0], axis_mins[1]);
  left = min(left, axis_mins[2]);
  left = min(left, axis_mins[3]);

  right = max(axis_maxs[0], axis_maxs[1]);
  right = max(right, axis_maxs[2]);
  right = max(right, axis_maxs[3]);

  if (left > rightMax && right == 0) {
    // verify that we have nothing to generate and early terminate.
    bool mins_same =
        (axis_mins[0] == axis_mins[1] && axis_mins[0] == axis_mins[2] &&
         axis_mins[0] == axis_mins[3]);
    bool maxs_same =
        (axis_maxs[0] == axis_maxs[1] && axis_maxs[0] == axis_maxs[2] &&
         axis_maxs[0] == axis_maxs[3]);

    left = 0;
    right = rightMax;
    if (mins_same && maxs_same) {
      SIZE e0 = *edges(r, 0, f);
      SIZE e1 = *edges(r, 0, f + 1);
      SIZE e2 = *edges(r + 1, 0, f + 1);
      SIZE e3 = *edges(r + 1, 0, f);
      if (e0 == e1 && e1 == e2 && e2 == e3) {
        // We have nothing to process in this row
        return false;
      }
    }
  } else {

    SIZE e0 = *edges(r, left, f);
    SIZE e1 = *edges(r, left, f + 1);
    SIZE e2 = *edges(r + 1, left, f + 1);
    SIZE e3 = *edges(r + 1, left, f);
    if ((e0 & 0x1) != (e1 & 0x1) || (e1 & 0x1) != (e2 & 0x1) ||
        (e2 & 0x1) != (e3 & 0x1)) {
      left = 0;
    }

    e0 = *edges(r, right, f);
    e1 = *edges(r, right, f + 1);
    e2 = *edges(r + 1, right, f + 1);
    e3 = *edges(r + 1, right, f);
    if ((e0 & 0x2) != (e1 & 0x2) || (e1 & 0x2) != (e2 & 0x2) ||
        (e2 & 0x2) != (e3 & 0x2)) {
      right = rightMax;
    }
  }
  return true;
}

template <typename DeviceType>
MGARDX_EXEC SIZE getEdgeCase(SIZE r, SIZE c, SIZE f,
                             SubArray<3, SIZE, DeviceType> &edges) {
  SIZE e0 = *edges(r, c, f);
  SIZE e1 = *edges(r, c, f + 1);
  SIZE e2 = *edges(r + 1, c, f);
  SIZE e3 = *edges(r + 1, c, f + 1);
  SIZE edgeCase = (e0 | (e1 << 2) | (e2 << 4) | (e3 << 6));
  return edgeCase;
}

MGARDX_EXEC SIZE GetNumberOfPrimitives(SIZE edgeCase) {
  // if (edgeCase >= 256) { printf("GetNumberOfPrimitives out of range\n");
  // edgeCase = 0; }
  static constexpr SIZE numTris[256] = {
      0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 2, 1, 2, 2, 3, 2, 3, 3, 4,
      2, 3, 3, 4, 3, 4, 4, 3, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3,
      2, 3, 3, 2, 3, 4, 4, 3, 3, 4, 4, 3, 4, 5, 5, 2, 1, 2, 2, 3, 2, 3, 3, 4,
      2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 4, 3, 2, 4, 3, 3, 4, 4, 5, 4, 3, 5, 2,
      2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 4, 3, 4, 4, 3, 4, 3, 5, 2,
      4, 5, 5, 4, 5, 4, 2, 1, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3,
      2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 4, 2, 3, 3, 4, 3, 4, 4, 5,
      3, 4, 2, 3, 4, 5, 3, 2, 3, 4, 4, 3, 4, 5, 5, 4, 4, 5, 3, 2, 5, 2, 4, 1,
      2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 2, 3, 3, 2, 3, 4, 4, 5, 4, 3, 5, 4,
      4, 5, 5, 2, 3, 2, 4, 1, 3, 4, 4, 5, 4, 5, 5, 2, 4, 5, 3, 4, 3, 4, 2, 1,
      2, 3, 3, 2, 3, 2, 4, 1, 3, 4, 2, 1, 2, 1, 1, 0};

  return numTris[edgeCase];
}

MGARDX_EXEC SIZE const *GetEdgeUses(SIZE edgeCase) {

  // if (edgeCase >= 256) { printf("GetEdgeUses out of range\n"); edgeCase = 0;
  // }
  static constexpr SIZE edgeUses[128][12] = {
      // This is [128][12] as idx 0 == idx 254, idx 1 == 253...
      //
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0},
      {1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0},
      {0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0},
      {0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0},
      {1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0},
      {1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0},
      {0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0},
      {0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1},
      {1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1},
      {1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1},
      {0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1},
      {0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1},
      {1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1},
      {1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1},
      {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1},
      {0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0},
      {1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0},
      {1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0},
      {0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0},
      {0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
      {1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0},
      {1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0},
      {0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0},
      {0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1},
      {1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1},
      {1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1},
      {0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1},
      {0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1},
      {1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1},
      {1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1},
      {0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1},
      {0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0},
      {1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0},
      {1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0},
      {0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0},
      {0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0},
      {1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0},
      {1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0},
      {0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0},
      {0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1},
      {1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1},
      {1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1},
      {0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1},
      {0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1},
      {1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1},
      {1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1},
      {0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1},
      {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0},
      {1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0},
      {1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0},
      {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0},
      {0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0},
      {1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0},
      {1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0},
      {0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0},
      {0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1},
      {1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1},
      {1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1},
      {0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1},
      {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
      {1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1},
      {1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1},
      {0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1},
      {0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0},
      {1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0},
      {1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0},
      {0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0},
      {0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0},
      {1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0},
      {1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0},
      {0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0},
      {0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1},
      {1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1},
      {1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1},
      {0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1},
      {0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1},
      {1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1},
      {1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1},
      {0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1},
      {0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0},
      {1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0},
      {1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0},
      {0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0},
      {0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0},
      {1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
      {1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0},
      {0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0},
      {0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1},
      {1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1},
      {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1},
      {0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1},
      {0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1},
      {1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1},
      {1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1},
      {0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1},
      {0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0},
      {1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0},
      {1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0},
      {0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0},
      {0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0},
      {1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0},
      {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
      {0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0},
      {0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      {1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1},
      {0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1},
      {0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1},
      {1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1},
      {1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1},
      {0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1},
      {0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0},
      {1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0},
      {1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0},
      {0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0},
      {0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0},
      {1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0},
      {1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0},
      {0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0},
      {0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1},
      {1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1},
      {1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1},
      {0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1},
      {0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1},
      {1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
      {1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1},
      {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1},
  };
  return edgeCase < 128 ? edgeUses[edgeCase] : edgeUses[127 - (edgeCase - 128)];
}

MGARDX_EXEC void CountBoundaryEdgeUses(bool *onBoundary, SIZE const *edgeUses,
                                       SIZE *_axis_sum, SIZE *adj_row_sum,
                                       SIZE *adj_col_sum) {
  if (onBoundary[1]) //+x boundary
  {
    _axis_sum[0] += edgeUses[5];
    _axis_sum[2] += edgeUses[9];
    if (onBoundary[0]) //+x +y
    {
      adj_row_sum[2] += edgeUses[11];
    }
    if (onBoundary[2]) //+x +z
    {
      adj_col_sum[0] += edgeUses[7];
    }
  }
  if (onBoundary[0]) //+y boundary
  {
    adj_row_sum[2] += edgeUses[10];
  }
  if (onBoundary[2]) //+z boundary
  {
    adj_col_sum[0] += edgeUses[6];
  }
}

MGARDX_EXEC SIZE const *GetTriEdgeCases(SIZE edgecase) {

  // if (edgecase >= 256) { printf("GetTriEdgeCases out of range\n"); edgecase =
  // 0; }

  static constexpr SIZE edgeCases[256][16] = {
      // I expect we have some form on symmetry in this table
      // that we can exploit to make it smaller
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {1, 0, 4, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {1, 0, 9, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {2, 5, 4, 8, 9, 5, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {1, 4, 1, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {2, 0, 1, 10, 8, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {2, 5, 0, 9, 1, 10, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 5, 1, 10, 5, 10, 9, 9, 10, 8, 0, 0, 0, 0, 0, 0},
      {1, 5, 11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {2, 0, 4, 8, 5, 11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {2, 9, 11, 1, 0, 9, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 1, 4, 8, 1, 8, 11, 11, 8, 9, 0, 0, 0, 0, 0, 0},
      {2, 4, 5, 11, 10, 4, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 0, 5, 11, 0, 11, 8, 8, 11, 10, 0, 0, 0, 0, 0, 0},
      {3, 4, 0, 9, 4, 9, 10, 10, 9, 11, 0, 0, 0, 0, 0, 0},
      {2, 9, 11, 8, 11, 10, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {1, 2, 8, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {2, 2, 0, 4, 6, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {2, 0, 9, 5, 8, 6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 2, 9, 5, 2, 5, 6, 6, 5, 4, 0, 0, 0, 0, 0, 0},
      {2, 8, 6, 2, 4, 1, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 10, 6, 2, 10, 2, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0},
      {3, 9, 5, 0, 8, 6, 2, 1, 10, 4, 0, 0, 0, 0, 0, 0},
      {4, 2, 10, 6, 9, 10, 2, 9, 1, 10, 9, 5, 1, 0, 0, 0},
      {2, 5, 11, 1, 8, 6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 4, 6, 2, 4, 2, 0, 5, 11, 1, 0, 0, 0, 0, 0, 0},
      {3, 9, 11, 1, 9, 1, 0, 8, 6, 2, 0, 0, 0, 0, 0, 0},
      {4, 1, 9, 11, 1, 6, 9, 1, 4, 6, 6, 2, 9, 0, 0, 0},
      {3, 4, 5, 11, 4, 11, 10, 6, 2, 8, 0, 0, 0, 0, 0, 0},
      {4, 5, 11, 10, 5, 10, 2, 5, 2, 0, 6, 2, 10, 0, 0, 0},
      {4, 2, 8, 6, 9, 10, 0, 9, 11, 10, 10, 4, 0, 0, 0, 0},
      {3, 2, 10, 6, 2, 9, 10, 9, 11, 10, 0, 0, 0, 0, 0, 0},
      {1, 9, 2, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {2, 9, 2, 7, 0, 4, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {2, 0, 2, 7, 5, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 8, 2, 7, 8, 7, 4, 4, 7, 5, 0, 0, 0, 0, 0, 0},
      {2, 9, 2, 7, 1, 10, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 0, 1, 10, 0, 10, 8, 2, 7, 9, 0, 0, 0, 0, 0, 0},
      {3, 0, 2, 7, 0, 7, 5, 1, 10, 4, 0, 0, 0, 0, 0, 0},
      {4, 1, 7, 5, 1, 8, 7, 1, 10, 8, 2, 7, 8, 0, 0, 0},
      {2, 5, 11, 1, 9, 2, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 4, 8, 0, 5, 11, 1, 2, 7, 9, 0, 0, 0, 0, 0, 0},
      {3, 7, 11, 1, 7, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0},
      {4, 1, 7, 11, 4, 7, 1, 4, 2, 7, 4, 8, 2, 0, 0, 0},
      {3, 11, 10, 4, 11, 4, 5, 9, 2, 7, 0, 0, 0, 0, 0, 0},
      {4, 2, 7, 9, 0, 5, 8, 8, 5, 11, 8, 11, 10, 0, 0, 0},
      {4, 7, 0, 2, 7, 10, 0, 7, 11, 10, 10, 4, 0, 0, 0, 0},
      {3, 7, 8, 2, 7, 11, 8, 11, 10, 8, 0, 0, 0, 0, 0, 0},
      {2, 9, 8, 6, 7, 9, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 9, 0, 4, 9, 4, 7, 7, 4, 6, 0, 0, 0, 0, 0, 0},
      {3, 0, 8, 6, 0, 6, 5, 5, 6, 7, 0, 0, 0, 0, 0, 0},
      {2, 5, 4, 7, 4, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 6, 7, 9, 6, 9, 8, 4, 1, 10, 0, 0, 0, 0, 0, 0},
      {4, 9, 6, 7, 9, 1, 6, 9, 0, 1, 1, 10, 6, 0, 0, 0},
      {4, 1, 10, 4, 0, 8, 5, 5, 8, 6, 5, 6, 7, 0, 0, 0},
      {3, 10, 5, 1, 10, 6, 5, 6, 7, 5, 0, 0, 0, 0, 0, 0},
      {3, 9, 8, 6, 9, 6, 7, 11, 1, 5, 0, 0, 0, 0, 0, 0},
      {4, 11, 1, 5, 9, 0, 7, 7, 0, 4, 7, 4, 6, 0, 0, 0},
      {4, 8, 1, 0, 8, 7, 1, 8, 6, 7, 11, 1, 7, 0, 0, 0},
      {3, 1, 7, 11, 1, 4, 7, 4, 6, 7, 0, 0, 0, 0, 0, 0},
      {4, 9, 8, 7, 8, 6, 7, 11, 4, 5, 11, 10, 4, 0, 0, 0},
      {5, 7, 0, 6, 7, 9, 0, 6, 0, 10, 5, 11, 0, 10, 0, 11},
      {5, 10, 0, 11, 10, 4, 0, 11, 0, 7, 8, 6, 0, 7, 0, 6},
      {2, 10, 7, 11, 6, 7, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {1, 6, 10, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {2, 4, 8, 0, 10, 3, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {2, 0, 9, 5, 10, 3, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 8, 9, 5, 8, 5, 4, 10, 3, 6, 0, 0, 0, 0, 0, 0},
      {2, 6, 4, 1, 3, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 6, 8, 0, 6, 0, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0},
      {3, 1, 3, 6, 1, 6, 4, 0, 9, 5, 0, 0, 0, 0, 0, 0},
      {4, 5, 1, 3, 5, 3, 8, 5, 8, 9, 8, 3, 6, 0, 0, 0},
      {2, 11, 1, 5, 3, 6, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 5, 11, 1, 4, 8, 0, 3, 6, 10, 0, 0, 0, 0, 0, 0},
      {3, 1, 0, 9, 1, 9, 11, 3, 6, 10, 0, 0, 0, 0, 0, 0},
      {4, 3, 6, 10, 1, 4, 11, 11, 4, 8, 11, 8, 9, 0, 0, 0},
      {3, 11, 3, 6, 11, 6, 5, 5, 6, 4, 0, 0, 0, 0, 0, 0},
      {4, 11, 3, 6, 5, 11, 6, 5, 6, 8, 5, 8, 0, 0, 0, 0},
      {4, 0, 6, 4, 0, 11, 6, 0, 9, 11, 3, 6, 11, 0, 0, 0},
      {3, 6, 11, 3, 6, 8, 11, 8, 9, 11, 0, 0, 0, 0, 0, 0},
      {2, 3, 2, 8, 10, 3, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 4, 10, 3, 4, 3, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0},
      {3, 8, 10, 3, 8, 3, 2, 9, 5, 0, 0, 0, 0, 0, 0, 0},
      {4, 9, 3, 2, 9, 4, 3, 9, 5, 4, 10, 3, 4, 0, 0, 0},
      {3, 8, 4, 1, 8, 1, 2, 2, 1, 3, 0, 0, 0, 0, 0, 0},
      {2, 0, 1, 2, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {4, 5, 0, 9, 1, 2, 4, 1, 3, 2, 2, 8, 4, 0, 0, 0},
      {3, 5, 2, 9, 5, 1, 2, 1, 3, 2, 0, 0, 0, 0, 0, 0},
      {3, 3, 2, 8, 3, 8, 10, 1, 5, 11, 0, 0, 0, 0, 0, 0},
      {4, 5, 11, 1, 4, 10, 0, 0, 10, 3, 0, 3, 2, 0, 0, 0},
      {4, 2, 8, 10, 2, 10, 3, 0, 9, 1, 1, 9, 11, 0, 0, 0},
      {5, 11, 4, 9, 11, 1, 4, 9, 4, 2, 10, 3, 4, 2, 4, 3},
      {4, 8, 4, 5, 8, 5, 3, 8, 3, 2, 3, 5, 11, 0, 0, 0},
      {3, 11, 0, 5, 11, 3, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0},
      {5, 2, 4, 3, 2, 8, 4, 3, 4, 11, 0, 9, 4, 11, 4, 9},
      {2, 11, 2, 9, 3, 2, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {2, 2, 7, 9, 6, 10, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 0, 4, 8, 2, 7, 9, 10, 3, 6, 0, 0, 0, 0, 0, 0},
      {3, 7, 5, 0, 7, 0, 2, 6, 10, 3, 0, 0, 0, 0, 0, 0},
      {4, 10, 3, 6, 8, 2, 4, 4, 2, 7, 4, 7, 5, 0, 0, 0},
      {3, 6, 4, 1, 6, 1, 3, 7, 9, 2, 0, 0, 0, 0, 0, 0},
      {4, 9, 2, 7, 0, 3, 8, 0, 1, 3, 3, 6, 8, 0, 0, 0},
      {4, 4, 1, 3, 4, 3, 6, 5, 0, 7, 7, 0, 2, 0, 0, 0},
      {5, 3, 8, 1, 3, 6, 8, 1, 8, 5, 2, 7, 8, 5, 8, 7},
      {3, 9, 2, 7, 11, 1, 5, 6, 10, 3, 0, 0, 0, 0, 0, 0},
      {4, 3, 6, 10, 5, 11, 1, 0, 4, 8, 2, 7, 9, 0, 0, 0},
      {4, 6, 10, 3, 7, 11, 2, 2, 11, 1, 2, 1, 0, 0, 0, 0},
      {5, 4, 8, 2, 4, 2, 7, 4, 7, 1, 11, 1, 7, 10, 3, 6},
      {4, 9, 2, 7, 11, 3, 5, 5, 3, 6, 5, 6, 4, 0, 0, 0},
      {5, 5, 11, 3, 5, 3, 6, 5, 6, 0, 8, 0, 6, 9, 2, 7},
      {5, 2, 11, 0, 2, 7, 11, 0, 11, 4, 3, 6, 11, 4, 11, 6},
      {4, 6, 11, 3, 6, 8, 11, 7, 11, 2, 2, 11, 8, 0, 0, 0},
      {3, 3, 7, 9, 3, 9, 10, 10, 9, 8, 0, 0, 0, 0, 0, 0},
      {4, 4, 10, 3, 0, 4, 3, 0, 3, 7, 0, 7, 9, 0, 0, 0},
      {4, 0, 8, 10, 0, 10, 7, 0, 7, 5, 7, 10, 3, 0, 0, 0},
      {3, 3, 4, 10, 3, 7, 4, 7, 5, 4, 0, 0, 0, 0, 0, 0},
      {4, 7, 9, 8, 7, 8, 1, 7, 1, 3, 4, 1, 8, 0, 0, 0},
      {3, 9, 3, 7, 9, 0, 3, 0, 1, 3, 0, 0, 0, 0, 0, 0},
      {5, 5, 8, 7, 5, 0, 8, 7, 8, 3, 4, 1, 8, 3, 8, 1},
      {2, 5, 3, 7, 1, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {4, 5, 11, 1, 9, 10, 7, 9, 8, 10, 10, 3, 7, 0, 0, 0},
      {5, 0, 4, 10, 0, 10, 3, 0, 3, 9, 7, 9, 3, 5, 11, 1},
      {5, 10, 7, 8, 10, 3, 7, 8, 7, 0, 11, 1, 7, 0, 7, 1},
      {4, 3, 4, 10, 3, 7, 4, 1, 4, 11, 11, 4, 7, 0, 0, 0},
      {5, 5, 3, 4, 5, 11, 3, 4, 3, 8, 7, 9, 3, 8, 3, 9},
      {4, 11, 0, 5, 11, 3, 0, 9, 0, 7, 7, 0, 3, 0, 0, 0},
      {2, 0, 8, 4, 7, 11, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {1, 11, 3, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {1, 11, 7, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {2, 0, 4, 8, 7, 3, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {2, 9, 5, 0, 7, 3, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 5, 4, 8, 5, 8, 9, 7, 3, 11, 0, 0, 0, 0, 0, 0},
      {2, 1, 10, 4, 11, 7, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 10, 8, 0, 10, 0, 1, 11, 7, 3, 0, 0, 0, 0, 0, 0},
      {3, 0, 9, 5, 1, 10, 4, 7, 3, 11, 0, 0, 0, 0, 0, 0},
      {4, 7, 3, 11, 5, 1, 9, 9, 1, 10, 9, 10, 8, 0, 0, 0},
      {2, 5, 7, 3, 1, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 5, 7, 3, 5, 3, 1, 4, 8, 0, 0, 0, 0, 0, 0, 0},
      {3, 9, 7, 3, 9, 3, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0},
      {4, 7, 8, 9, 7, 1, 8, 7, 3, 1, 4, 8, 1, 0, 0, 0},
      {3, 3, 10, 4, 3, 4, 7, 7, 4, 5, 0, 0, 0, 0, 0, 0},
      {4, 0, 10, 8, 0, 7, 10, 0, 5, 7, 7, 3, 10, 0, 0, 0},
      {4, 4, 3, 10, 0, 3, 4, 0, 7, 3, 0, 9, 7, 0, 0, 0},
      {3, 3, 9, 7, 3, 10, 9, 10, 8, 9, 0, 0, 0, 0, 0, 0},
      {2, 7, 3, 11, 2, 8, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 2, 0, 4, 2, 4, 6, 3, 11, 7, 0, 0, 0, 0, 0, 0},
      {3, 5, 0, 9, 7, 3, 11, 8, 6, 2, 0, 0, 0, 0, 0, 0},
      {4, 11, 7, 3, 5, 6, 9, 5, 4, 6, 6, 2, 9, 0, 0, 0},
      {3, 4, 1, 10, 6, 2, 8, 11, 7, 3, 0, 0, 0, 0, 0, 0},
      {4, 7, 3, 11, 2, 1, 6, 2, 0, 1, 1, 10, 6, 0, 0, 0},
      {4, 0, 9, 5, 2, 8, 6, 1, 10, 4, 7, 3, 11, 0, 0, 0},
      {5, 9, 5, 1, 9, 1, 10, 9, 10, 2, 6, 2, 10, 7, 3, 11},
      {3, 3, 1, 5, 3, 5, 7, 2, 8, 6, 0, 0, 0, 0, 0, 0},
      {4, 5, 7, 1, 7, 3, 1, 4, 2, 0, 4, 6, 2, 0, 0, 0},
      {4, 8, 6, 2, 9, 7, 0, 0, 7, 3, 0, 3, 1, 0, 0, 0},
      {5, 6, 9, 4, 6, 2, 9, 4, 9, 1, 7, 3, 9, 1, 9, 3},
      {4, 8, 6, 2, 4, 7, 10, 4, 5, 7, 7, 3, 10, 0, 0, 0},
      {5, 7, 10, 5, 7, 3, 10, 5, 10, 0, 6, 2, 10, 0, 10, 2},
      {5, 0, 9, 7, 0, 7, 3, 0, 3, 4, 10, 4, 3, 8, 6, 2},
      {4, 3, 9, 7, 3, 10, 9, 2, 9, 6, 6, 9, 10, 0, 0, 0},
      {2, 11, 9, 2, 3, 11, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 2, 3, 11, 2, 11, 9, 0, 4, 8, 0, 0, 0, 0, 0, 0},
      {3, 11, 5, 0, 11, 0, 3, 3, 0, 2, 0, 0, 0, 0, 0, 0},
      {4, 8, 5, 4, 8, 3, 5, 8, 2, 3, 3, 11, 5, 0, 0, 0},
      {3, 11, 9, 2, 11, 2, 3, 10, 4, 1, 0, 0, 0, 0, 0, 0},
      {4, 0, 1, 8, 1, 10, 8, 2, 11, 9, 2, 3, 11, 0, 0, 0},
      {4, 4, 1, 10, 0, 3, 5, 0, 2, 3, 3, 11, 5, 0, 0, 0},
      {5, 3, 5, 2, 3, 11, 5, 2, 5, 8, 1, 10, 5, 8, 5, 10},
      {3, 5, 9, 2, 5, 2, 1, 1, 2, 3, 0, 0, 0, 0, 0, 0},
      {4, 4, 8, 0, 5, 9, 1, 1, 9, 2, 1, 2, 3, 0, 0, 0},
      {2, 0, 2, 1, 2, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 8, 1, 4, 8, 2, 1, 2, 3, 1, 0, 0, 0, 0, 0, 0},
      {4, 9, 2, 3, 9, 3, 4, 9, 4, 5, 10, 4, 3, 0, 0, 0},
      {5, 8, 5, 10, 8, 0, 5, 10, 5, 3, 9, 2, 5, 3, 5, 2},
      {3, 4, 3, 10, 4, 0, 3, 0, 2, 3, 0, 0, 0, 0, 0, 0},
      {2, 3, 8, 2, 10, 8, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 6, 3, 11, 6, 11, 8, 8, 11, 9, 0, 0, 0, 0, 0, 0},
      {4, 0, 4, 6, 0, 6, 11, 0, 11, 9, 3, 11, 6, 0, 0, 0},
      {4, 11, 6, 3, 5, 6, 11, 5, 8, 6, 5, 0, 8, 0, 0, 0},
      {3, 11, 6, 3, 11, 5, 6, 5, 4, 6, 0, 0, 0, 0, 0, 0},
      {4, 1, 10, 4, 11, 8, 3, 11, 9, 8, 8, 6, 3, 0, 0, 0},
      {5, 1, 6, 0, 1, 10, 6, 0, 6, 9, 3, 11, 6, 9, 6, 11},
      {5, 5, 0, 8, 5, 8, 6, 5, 6, 11, 3, 11, 6, 1, 10, 4},
      {4, 10, 5, 1, 10, 6, 5, 11, 5, 3, 3, 5, 6, 0, 0, 0},
      {4, 5, 3, 1, 5, 8, 3, 5, 9, 8, 8, 6, 3, 0, 0, 0},
      {5, 1, 9, 3, 1, 5, 9, 3, 9, 6, 0, 4, 9, 6, 9, 4},
      {3, 6, 0, 8, 6, 3, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0},
      {2, 6, 1, 4, 3, 1, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {5, 8, 3, 9, 8, 6, 3, 9, 3, 5, 10, 4, 3, 5, 3, 4},
      {2, 0, 5, 9, 10, 6, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {4, 6, 0, 8, 6, 3, 0, 4, 0, 10, 10, 0, 3, 0, 0, 0},
      {1, 6, 3, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {2, 10, 11, 7, 6, 10, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 10, 11, 7, 10, 7, 6, 8, 0, 4, 0, 0, 0, 0, 0, 0},
      {3, 7, 6, 10, 7, 10, 11, 5, 0, 9, 0, 0, 0, 0, 0, 0},
      {4, 11, 7, 6, 11, 6, 10, 9, 5, 8, 8, 5, 4, 0, 0, 0},
      {3, 1, 11, 7, 1, 7, 4, 4, 7, 6, 0, 0, 0, 0, 0, 0},
      {4, 8, 0, 1, 8, 1, 7, 8, 7, 6, 11, 7, 1, 0, 0, 0},
      {4, 9, 5, 0, 7, 4, 11, 7, 6, 4, 4, 1, 11, 0, 0, 0},
      {5, 9, 1, 8, 9, 5, 1, 8, 1, 6, 11, 7, 1, 6, 1, 7},
      {3, 10, 1, 5, 10, 5, 6, 6, 5, 7, 0, 0, 0, 0, 0, 0},
      {4, 0, 4, 8, 5, 6, 1, 5, 7, 6, 6, 10, 1, 0, 0, 0},
      {4, 9, 7, 6, 9, 6, 1, 9, 1, 0, 1, 6, 10, 0, 0, 0},
      {5, 6, 1, 7, 6, 10, 1, 7, 1, 9, 4, 8, 1, 9, 1, 8},
      {2, 5, 7, 4, 4, 7, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 0, 6, 8, 0, 5, 6, 5, 7, 6, 0, 0, 0, 0, 0, 0},
      {3, 9, 4, 0, 9, 7, 4, 7, 6, 4, 0, 0, 0, 0, 0, 0},
      {2, 9, 6, 8, 7, 6, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 7, 2, 8, 7, 8, 11, 11, 8, 10, 0, 0, 0, 0, 0, 0},
      {4, 7, 2, 0, 7, 0, 10, 7, 10, 11, 10, 0, 4, 0, 0, 0},
      {4, 0, 9, 5, 8, 11, 2, 8, 10, 11, 11, 7, 2, 0, 0, 0},
      {5, 11, 2, 10, 11, 7, 2, 10, 2, 4, 9, 5, 2, 4, 2, 5},
      {4, 1, 11, 7, 4, 1, 7, 4, 7, 2, 4, 2, 8, 0, 0, 0},
      {3, 7, 1, 11, 7, 2, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0},
      {5, 4, 1, 11, 4, 11, 7, 4, 7, 8, 2, 8, 7, 0, 9, 5},
      {4, 7, 1, 11, 7, 2, 1, 5, 1, 9, 9, 1, 2, 0, 0, 0},
      {4, 1, 5, 7, 1, 7, 8, 1, 8, 10, 2, 8, 7, 0, 0, 0},
      {5, 0, 10, 2, 0, 4, 10, 2, 10, 7, 1, 5, 10, 7, 10, 5},
      {5, 0, 7, 1, 0, 9, 7, 1, 7, 10, 2, 8, 7, 10, 7, 8},
      {2, 9, 7, 2, 1, 4, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 8, 7, 2, 8, 4, 7, 4, 5, 7, 0, 0, 0, 0, 0, 0},
      {2, 0, 7, 2, 5, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {4, 8, 7, 2, 8, 4, 7, 9, 7, 0, 0, 7, 4, 0, 0, 0},
      {1, 9, 7, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 2, 6, 10, 2, 10, 9, 9, 10, 11, 0, 0, 0, 0, 0, 0},
      {4, 0, 4, 8, 2, 6, 9, 9, 6, 10, 9, 10, 11, 0, 0, 0},
      {4, 5, 10, 11, 5, 2, 10, 5, 0, 2, 6, 10, 2, 0, 0, 0},
      {5, 4, 2, 5, 4, 8, 2, 5, 2, 11, 6, 10, 2, 11, 2, 10},
      {4, 1, 11, 9, 1, 9, 6, 1, 6, 4, 6, 9, 2, 0, 0, 0},
      {5, 9, 6, 11, 9, 2, 6, 11, 6, 1, 8, 0, 6, 1, 6, 0},
      {5, 4, 11, 6, 4, 1, 11, 6, 11, 2, 5, 0, 11, 2, 11, 0},
      {2, 5, 1, 11, 8, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {4, 2, 6, 10, 9, 2, 10, 9, 10, 1, 9, 1, 5, 0, 0, 0},
      {5, 9, 2, 6, 9, 6, 10, 9, 10, 5, 1, 5, 10, 0, 4, 8},
      {3, 10, 2, 6, 10, 1, 2, 1, 0, 2, 0, 0, 0, 0, 0, 0},
      {4, 10, 2, 6, 10, 1, 2, 8, 2, 4, 4, 2, 1, 0, 0, 0},
      {3, 2, 5, 9, 2, 6, 5, 6, 4, 5, 0, 0, 0, 0, 0, 0},
      {4, 2, 5, 9, 2, 6, 5, 0, 5, 8, 8, 5, 6, 0, 0, 0},
      {2, 2, 4, 0, 6, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {1, 2, 6, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {2, 9, 8, 11, 11, 8, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 4, 9, 0, 4, 10, 9, 10, 11, 9, 0, 0, 0, 0, 0, 0},
      {3, 0, 11, 5, 0, 8, 11, 8, 10, 11, 0, 0, 0, 0, 0, 0},
      {2, 4, 11, 5, 10, 11, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 1, 8, 4, 1, 11, 8, 11, 9, 8, 0, 0, 0, 0, 0, 0},
      {2, 9, 1, 11, 0, 1, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {4, 1, 8, 4, 1, 11, 8, 0, 8, 5, 5, 8, 11, 0, 0, 0},
      {1, 5, 1, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {3, 5, 10, 1, 5, 9, 10, 9, 8, 10, 0, 0, 0, 0, 0, 0},
      {4, 4, 9, 0, 4, 10, 9, 5, 9, 1, 1, 9, 10, 0, 0, 0},
      {2, 0, 10, 1, 8, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {1, 4, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {2, 5, 8, 4, 9, 8, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {1, 0, 5, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {1, 0, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  };
  return edgeCases[edgecase];
}

template <typename DeviceType> struct Pass4TrimState {
  SIZE left, right;
  SIZE r, c, f;
  SIZE nr, nc, nf;
  // vtkm::Id3 ijk;
  // vtkm::Id4 startPos;
  // SIZE cellId;
  // vtkm::Id axis_inc;
  SIZE boundaryStatus[3];
  bool hasWork = true;

  MGARDX_EXEC
  Pass4TrimState(SIZE r, SIZE f, SIZE nf, SIZE nc, SIZE nr,
                 SubArray<2, SIZE, DeviceType> &axis_min,
                 SubArray<2, SIZE, DeviceType> &axis_max,
                 SubArray<3, SIZE, DeviceType> &edges)
      : r(r), f(f), nf(nf), nc(nc), nr(nr) {
    // ijk = compute_ijk(AxisToSum{}, threadIndices.GetInputIndex3D());

    // startPos = compute_neighbor_starts(AxisToSum{}, ijk, pdims);
    // axis_inc = compute_inc(AxisToSum{}, pdims);

    // Compute the subset (start and end) of the row that we need
    // to iterate to generate triangles for the iso-surface
    hasWork =
        computeTrimBounds(r, f, nc, edges, axis_min, axis_max, left, right);
    hasWork = hasWork && left != right;
    if (!hasWork) {
      return;
    }

    // cellId = compute_start(AxisToSum{}, ijk, pdims - vtkm::Id3{ 1, 1, 1 });

    // update our ijk

    c = left;

    boundaryStatus[0] = MGARD_Interior;
    boundaryStatus[1] = MGARD_Interior;
    boundaryStatus[2] = MGARD_Interior;

    if (c < 1) {
      boundaryStatus[1] += MGARD_MinBoundary;
    }
    if (c >= (nc - 2)) {
      boundaryStatus[1] += MGARD_MaxBoundary;
    }
    if (f < 1) {
      boundaryStatus[0] += MGARD_MinBoundary;
    }
    if (f >= (nf - 2)) {
      boundaryStatus[0] += MGARD_MaxBoundary;
    }
    if (r < 1) {
      boundaryStatus[2] += MGARD_MinBoundary;
    }
    if (r >= (nr - 2)) {
      boundaryStatus[2] += MGARD_MaxBoundary;
    }
  }

  MGARDX_EXEC void increment() {
    // compute what the current cellId is
    // cellId = increment_cellId(AxisToSum{}, cellId, axis_inc);

    // compute what the current ijk is
    c++;

    // compute what the current boundary state is
    // can never be on the MGARD_MinBoundary after we increment
    if (c >= (nc - 2)) {
      boundaryStatus[1] = MGARD_MaxBoundary;
    } else {
      boundaryStatus[1] = MGARD_Interior;
    }
  }
};

template <typename DeviceType>
MGARDX_EXEC void init_voxelIds(SIZE r, SIZE f, SIZE edgeCase,
                               SubArray<3, SIZE, DeviceType> &axis_sum,
                               SIZE *edgeIds) {
  SIZE const *edgeUses = GetEdgeUses(edgeCase);
  edgeIds[0] = *axis_sum(r, f, 1); // x-edges
  edgeIds[1] = *axis_sum(r, f + 1, 1);
  edgeIds[2] = *axis_sum(r + 1, f, 1);
  edgeIds[3] = *axis_sum(r + 1, f + 1, 1);

  edgeIds[4] = *axis_sum(r, f, 0); // y-edges
  edgeIds[5] = edgeIds[4] + edgeUses[4];
  edgeIds[6] = *axis_sum(r + 1, f, 0);
  edgeIds[7] = edgeIds[6] + edgeUses[6];

  edgeIds[8] = *axis_sum(r, f, 2); // z-edges
  edgeIds[9] = edgeIds[8] + edgeUses[8];
  edgeIds[10] = *axis_sum(r, f + 1, 2);
  edgeIds[11] = edgeIds[10] + edgeUses[10];
}

MGARDX_EXEC void advance_voxelIds(SIZE const *edgeUses, SIZE *edgeIds) {
  edgeIds[0] += edgeUses[0]; // x-edges
  edgeIds[1] += edgeUses[1];
  edgeIds[2] += edgeUses[2];
  edgeIds[3] += edgeUses[3];
  edgeIds[4] += edgeUses[4]; // y-edges
  edgeIds[5] = edgeIds[4] + edgeUses[5];
  edgeIds[6] += edgeUses[6];
  edgeIds[7] = edgeIds[6] + edgeUses[7];
  edgeIds[8] += edgeUses[8]; // z-edges
  edgeIds[9] = edgeIds[8] + edgeUses[9];
  edgeIds[10] += edgeUses[10];
  edgeIds[11] = edgeIds[10] + edgeUses[11];
}

template <typename DeviceType>
MGARDX_EXEC void
generate_tris(SIZE edgeCase, SIZE numTris, SIZE *edgeIds, SIZE &cell_tri_offset,
              SubArray<1, SIZE, DeviceType> &triangle_topology) {

  // printf("edgeCase: %u\n", edgeCase);
  SIZE const *edges = GetTriEdgeCases(edgeCase);
  SIZE edgeIndex = 1;
  SIZE index = cell_tri_offset * 3;
  // printf("generate_tris: (%u %u %u) \n",
  //     index, index+1, index+2);
  for (SIZE i = 0; i < numTris; ++i) {
    // This keeps the same winding for the triangles that marching cells
    // produced. By keeping the winding the same we make sure
    // that 'fast' normals are consistent with the marching
    // cells version

    // printf("generate_tris: (%u %u %u) <- (%u %u %u)\n",
    //   index, index+1, index+2,
    //   edgeIndex,
    //   edgeIndex + 2,
    //   edgeIndex + 1);

    // if (cell_tri_offset == 6) {
    //   printf("edgeIds: %u %u %u %u %u %u %u %u %u %u %u %u\n",
    //           edgeIds[0], edgeIds[1], edgeIds[2], edgeIds[3], edgeIds[4],
    //           edgeIds[5], edgeIds[6], edgeIds[7], edgeIds[8], edgeIds[9],
    //           edgeIds[10], edgeIds[11]);
    //   printf("edgeCase: %u, tri (%u)%u (%u)%u (%u)%u\n", edgeCase,
    //   edges[edgeIndex], edgeIds[edges[edgeIndex]],
    //                           edges[edgeIndex + 2], edgeIds[edges[edgeIndex +
    //                           2]], edges[edgeIndex + 1],
    //                           edgeIds[edges[edgeIndex + 1]]);
    // }

    // if (edgeCase >= 256 || index + 2 >= 3187914*3)
    //   printf("edgeCase: %u, index: %u %u %u\n", edgeCase, index, index + 1,
    //   index + 2);
    // if (edges[edgeIndex]>=12 || edges[edgeIndex + 2] >= 12 || edges[edgeIndex
    // + 1] >= 12) {
    //   printf("edges: %u %u %u\n", edges[edgeIndex], edges[edgeIndex+2],
    //   edges[edgeIndex+1]);
    // }
    // if (edgeIndex+2 >= 16) {
    //   printf("edgeIndex: %u %u %u\n", edgeIndex, edgeIndex+2, edgeIndex+1);
    // }
    // index = 0;
    *triangle_topology(index) = edgeIds[edges[edgeIndex]];
    *triangle_topology(index + 1) = edgeIds[edges[edgeIndex + 2]];
    *triangle_topology(index + 2) = edgeIds[edges[edgeIndex + 1]];
    index += 3;
    edgeIndex += 3;
  }
  cell_tri_offset += numTris;
}

MGARDX_EXEC bool fully_interior(const SIZE *boundaryStatus) {
  return boundaryStatus[0] == MGARD_Interior &&
         boundaryStatus[1] == MGARD_Interior &&
         boundaryStatus[2] == MGARD_Interior;
}

MGARDX_EXEC bool case_includes_axes(const SIZE *const edgeUses) {

  return (edgeUses[0] != 0 || edgeUses[4] != 0 || edgeUses[8] != 0);
}

MGARDX_EXEC SIZE const *GetVertMap(SIZE index) {
  static constexpr SIZE vertMap[12][2] = {
      {0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3},
      {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7},
  };
  return vertMap[index];
}

MGARDX_EXEC SIZE const *GetVertOffsets(SIZE Vert) {
  static constexpr SIZE vertMap[8][3] = {
      {0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 0},
      {0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1},
  };
  return vertMap[Vert];
}

template <typename T, typename DeviceType>
MGARDX_EXEC void InterpolateEdge(SIZE edgeNum, SIZE f, SIZE c, SIZE r,
                                 SIZE const *edgeUses, SIZE *edgeIds,
                                 T iso_value, SubArray<3, T, DeviceType> &v,
                                 SubArray<1, T, DeviceType> &points) {
  if (!edgeUses[edgeNum]) {
    return;
  }

  SIZE writeIndex = edgeIds[edgeNum] * 3;
  SIZE const *verts = GetVertMap(edgeNum);
  SIZE const *offsets0 = GetVertOffsets(verts[0]);
  SIZE const *offsets1 = GetVertOffsets(verts[1]);

  SIZE r0 = r + offsets0[2];
  SIZE c0 = c + offsets0[1];
  SIZE f0 = f + offsets0[0];

  SIZE r1 = r + offsets1[2];
  SIZE c1 = c + offsets1[1];
  SIZE f1 = f + offsets1[0];

  T s0 = *v(r0, c0, f0);
  T s1 = *v(r1, c1, f1);

  T w = (iso_value - s0) / (s1 - s0);

  // if (writeIndex == 0) {
  //   printf("iso_value: %f, s0: %f, s1: %f, %f %f w: %f\n",
  //             iso_value, s0, s1, (iso_value - s0), (s1 - s0), w);
  // }

  *points(writeIndex) = (1 - w) * f0 + w * f1;
  *points(writeIndex + 1) = (1 - w) * c0 + w * c1;
  *points(writeIndex + 2) = (1 - w) * r0 + w * r1;
}

template <typename T, typename DeviceType>
MGARDX_EXEC void Generate(SIZE f, SIZE c, SIZE r, SIZE *boundaryStatus,
                          SIZE const *edgeUses, SIZE *edgeIds, T iso_value,
                          SubArray<3, T, DeviceType> &v,
                          SubArray<1, T, DeviceType> &points) {

  InterpolateEdge(0, f, c, r, edgeUses, edgeIds, iso_value, v, points);
  InterpolateEdge(4, f, c, r, edgeUses, edgeIds, iso_value, v, points);
  InterpolateEdge(8, f, c, r, edgeUses, edgeIds, iso_value, v, points);

  const bool onX = boundaryStatus[1] & MGARD_MaxBoundary;
  const bool onY = boundaryStatus[0] & MGARD_MaxBoundary;
  const bool onZ = boundaryStatus[2] & MGARD_MaxBoundary;

  if (onX) //+x boundary
  {
    InterpolateEdge(5, f, c, r, edgeUses, edgeIds, iso_value, v, points);
    InterpolateEdge(9, f, c, r, edgeUses, edgeIds, iso_value, v, points);
    if (onY) //+y boundary
    {
      InterpolateEdge(11, f, c, r, edgeUses, edgeIds, iso_value, v, points);
    }
    if (onZ) //+z boundary
    {
      InterpolateEdge(7, f, c, r, edgeUses, edgeIds, iso_value, v, points);
    }
  }

  if (onY) //+y boundary
  {
    InterpolateEdge(1, f, c, r, edgeUses, edgeIds, iso_value, v, points);
    InterpolateEdge(10, f, c, r, edgeUses, edgeIds, iso_value, v, points);
    if (onZ) //+z boundary
    {
      InterpolateEdge(3, f, c, r, edgeUses, edgeIds, iso_value, v, points);
    }
  }

  if (onZ) //+z boundary
  {
    InterpolateEdge(2, f, c, r, edgeUses, edgeIds, iso_value, v, points);
    InterpolateEdge(6, f, c, r, edgeUses, edgeIds, iso_value, v, points);
  }
}

template <typename T, typename DeviceType>
class Pass1Functor : public Functor<DeviceType> {
public:
  MGARDX_CONT Pass1Functor(SIZE nr, SIZE nc, SIZE nf,
                           SubArray<3, T, DeviceType> v, T iso_value,
                           SubArray<3, SIZE, DeviceType> axis_sum,
                           SubArray<2, SIZE, DeviceType> axis_min,
                           SubArray<2, SIZE, DeviceType> axis_max,
                           SubArray<3, SIZE, DeviceType> edges)
      : nr(nr), nc(nc), nf(nf), v(v), iso_value(iso_value), axis_sum(axis_sum),
        axis_min(axis_min), axis_max(axis_max), edges(edges) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE f = FunctorBase<DeviceType>::GetBlockIdX() *
                 FunctorBase<DeviceType>::GetBlockDimX() +
             FunctorBase<DeviceType>::GetThreadIdX();
    SIZE r = FunctorBase<DeviceType>::GetBlockIdY() *
                 FunctorBase<DeviceType>::GetBlockDimY() +
             FunctorBase<DeviceType>::GetThreadIdY();

    SIZE _axis_sum = 0;
    SIZE _axis_min = nc;
    SIZE _axis_max = 0;

    T s1 = *v(r, 0, f);
    T s0 = s1;

    if (f < nf && r < nr) {
      for (SIZE c = 0; c < nc - 1; c++) {
        SIZE edgeCase = MGARD_Below;
        s0 = s1;
        s1 = *v(r, c + 1, f);
        if (s0 >= iso_value)
          edgeCase = MGARD_LeftAbove;
        if (s1 >= iso_value)
          edgeCase |= MGARD_RightAbove;

        *edges(r, c, f) = edgeCase;

        if (edgeCase == MGARD_LeftAbove || edgeCase == MGARD_RightAbove) {
          _axis_sum += 1; // increment number of intersections along axis
          _axis_max = c + 1;
          if (_axis_min == nc) {
            _axis_min = c;
          }
        }
        // if (r == 0 && f == 0) {
        //   printf("s0: %f, s1: %f\n",
        //           s0, s1);
        // }
      }

      *axis_sum(r, f, 1) = _axis_sum;
      *axis_min(r, f) = _axis_min;
      *axis_max(r, f) = _axis_max;
    }
  }

  MGARDX_EXEC void Operation2() {}

  MGARDX_EXEC void Operation3() {}

  MGARDX_EXEC void Operation4() {}

  MGARDX_EXEC void Operation5() {}

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    return size;
  }

private:
  SIZE nr, nc, nf;
  SubArray<3, T, DeviceType> v;
  T iso_value;
  SubArray<3, SIZE, DeviceType> axis_sum;
  SubArray<2, SIZE, DeviceType> axis_min;
  SubArray<2, SIZE, DeviceType> axis_max;
  SubArray<3, SIZE, DeviceType> edges;
};

template <typename T, typename DeviceType>
class Pass2Functor : public Functor<DeviceType> {
public:
  MGARDX_CONT Pass2Functor(SIZE nr, SIZE nc, SIZE nf,
                           SubArray<3, SIZE, DeviceType> axis_sum,
                           SubArray<2, SIZE, DeviceType> axis_min,
                           SubArray<2, SIZE, DeviceType> axis_max,
                           SubArray<3, SIZE, DeviceType> edges,
                           SubArray<2, SIZE, DeviceType> cell_tri_count)
      : nr(nr), nc(nc), nf(nf), v(v), iso_value(iso_value), axis_sum(axis_sum),
        axis_min(axis_min), axis_max(axis_max), edges(edges),
        cell_tri_count(cell_tri_count) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE f = FunctorBase<DeviceType>::GetBlockIdX() *
                 FunctorBase<DeviceType>::GetBlockDimX() +
             FunctorBase<DeviceType>::GetThreadIdX();
    SIZE r = FunctorBase<DeviceType>::GetBlockIdY() *
                 FunctorBase<DeviceType>::GetBlockDimY() +
             FunctorBase<DeviceType>::GetThreadIdY();

    if (f >= nf - 1 || r >= nr - 1) {
      return;
    }

    // compute trim blounds
    SIZE left, right;
    bool hasWork =
        computeTrimBounds(r, f, nc, edges, axis_min, axis_max, left, right);
    if (!hasWork) {
      return;
    }

    bool onBoundary[3]; // f, c, r
    onBoundary[0] = f >= nf - 2;
    onBoundary[2] = r >= nr - 2;
    SIZE _cell_tri_count = 0;

    SIZE _axis_sum[3];
    _axis_sum[1] = *axis_sum(r, f, 1); //*axis_sum_c(r, f);
    _axis_sum[0] = 0;
    _axis_sum[2] = 0;

    SIZE adj_row_sum[3] = {0, 0, 0};
    SIZE adj_col_sum[3] = {0, 0, 0};

    if (onBoundary[0]) {
      adj_row_sum[1] = *axis_sum(r, f + 1, 1); //*axis_sum_c(r, f+1);
    }

    if (onBoundary[2]) {
      adj_col_sum[1] = *axis_sum(r + 1, f, 1); //*axis_sum_c(r+1, f);
    }

    for (SIZE c = left; c < right; c++) {
      SIZE edgeCase = getEdgeCase(r, c, f, edges);
      SIZE numTris = GetNumberOfPrimitives(edgeCase);
      if (numTris > 0) {
        _cell_tri_count += numTris;
        SIZE const *edgeUses = GetEdgeUses(edgeCase);

        onBoundary[1] = c >= nc - 2;

        _axis_sum[0] += edgeUses[4];
        _axis_sum[2] += edgeUses[8];

        CountBoundaryEdgeUses(onBoundary, edgeUses, _axis_sum, adj_row_sum,
                              adj_col_sum);
      }
    }

    *cell_tri_count(r, f) = _cell_tri_count;

    *axis_sum(r, f, 1) = _axis_sum[1];
    *axis_sum(r, f, 0) = _axis_sum[0];
    *axis_sum(r, f, 2) = _axis_sum[2];

    if (onBoundary[0]) {
      *axis_sum(r, f + 1, 1) = adj_row_sum[1];
      *axis_sum(r, f + 1, 0) = adj_row_sum[0];
      *axis_sum(r, f + 1, 2) = adj_row_sum[2];
    }

    if (onBoundary[2]) {

      *axis_sum(r + 1, f, 1) = adj_col_sum[1];
      *axis_sum(r + 1, f, 0) = adj_col_sum[0];
      *axis_sum(r + 1, f, 2) = adj_col_sum[2];
    }
  }

  MGARDX_EXEC void Operation2() {}

  MGARDX_EXEC void Operation3() {}

  MGARDX_EXEC void Operation4() {}

  MGARDX_EXEC void Operation5() {}

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    return size;
  }

private:
  SIZE nr, nc, nf;
  SubArray<3, T, DeviceType> v;
  T iso_value;
  SubArray<3, SIZE, DeviceType> axis_sum;
  SubArray<2, SIZE, DeviceType> axis_min;
  SubArray<2, SIZE, DeviceType> axis_max;
  SubArray<3, SIZE, DeviceType> edges;
  SubArray<2, SIZE, DeviceType> cell_tri_count;
};

template <typename T, typename DeviceType>
class Pass4Functor : public Functor<DeviceType> {
public:
  MGARDX_CONT Pass4Functor(SIZE nr, SIZE nc, SIZE nf,
                           SubArray<3, T, DeviceType> v, T iso_value,
                           SubArray<3, SIZE, DeviceType> axis_sum,
                           SubArray<2, SIZE, DeviceType> axis_min,
                           SubArray<2, SIZE, DeviceType> axis_max,
                           SubArray<1, SIZE, DeviceType> cell_tri_count_scan,
                           SubArray<3, SIZE, DeviceType> edges,
                           SubArray<1, SIZE, DeviceType> triangle_topology,
                           SubArray<1, T, DeviceType> points)
      : nr(nr), nc(nc), nf(nf), v(v), iso_value(iso_value), axis_sum(axis_sum),
        axis_min(axis_min), axis_max(axis_max),
        cell_tri_count_scan(cell_tri_count_scan), edges(edges),
        triangle_topology(triangle_topology), points(points) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE f = FunctorBase<DeviceType>::GetBlockIdX() *
                 FunctorBase<DeviceType>::GetBlockDimX() +
             FunctorBase<DeviceType>::GetThreadIdX();
    SIZE r = FunctorBase<DeviceType>::GetBlockIdY() *
                 FunctorBase<DeviceType>::GetBlockDimY() +
             FunctorBase<DeviceType>::GetThreadIdY();

    if (f >= nf - 1 || r >= nr - 1) {
      return;
    }

    // printf("offset: %u\n", r * (nf-1) + f);
    SIZE cell_tri_offset = *cell_tri_count_scan(r * (nf - 1) + f);
    SIZE next_tri_offset = *cell_tri_count_scan(r * (nf - 1) + f + 1);

    // printf("cell_tri_offset: %u\n", cell_tri_offset);

    Pass4TrimState state(r, f, nf, nc, nr, axis_min, axis_max, edges);
    if (!state.hasWork) {
      return;
    }

    SIZE edgeIds[12];
    SIZE edgeCase = getEdgeCase(r, state.left, f, edges);

    init_voxelIds(r, f, edgeCase, axis_sum, edgeIds);
    for (SIZE i = state.left; i < state.right;
         ++i) // run along the trimmed voxels
    {
      edgeCase = getEdgeCase(r, i, f, edges);
      SIZE numTris = GetNumberOfPrimitives(edgeCase);
      if (numTris > 0) {
        generate_tris(edgeCase, numTris, edgeIds, cell_tri_offset,
                      triangle_topology);

        SIZE const *edgeUses = GetEdgeUses(edgeCase);
        if (!fully_interior(state.boundaryStatus) ||
            case_includes_axes(edgeUses)) {
          Generate(f, i, r, state.boundaryStatus, edgeUses, edgeIds, iso_value,
                   v, points);
        }
        advance_voxelIds(edgeUses, edgeIds);
      }
      state.increment();
    }
  }

  MGARDX_EXEC void Operation2() {}

  MGARDX_EXEC void Operation3() {}

  MGARDX_EXEC void Operation4() {}

  MGARDX_EXEC void Operation5() {}

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    return size;
  }

private:
  SIZE nr, nc, nf;
  SubArray<3, T, DeviceType> v;
  T iso_value;
  SubArray<3, SIZE, DeviceType> axis_sum;
  SubArray<2, SIZE, DeviceType> axis_min;
  SubArray<2, SIZE, DeviceType> axis_max;
  SubArray<3, SIZE, DeviceType> edges;
  SubArray<1, SIZE, DeviceType> cell_tri_count_scan;
  SubArray<1, SIZE, DeviceType> triangle_topology;
  SubArray<1, T, DeviceType> points;
};

template <typename T, typename DeviceType>
class FlyingEdges : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  FlyingEdges() : AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<Pass1Functor<T, DeviceType>>
  GenTask1(SIZE nr, SIZE nc, SIZE nf, SubArray<3, T, DeviceType> v, T iso_value,
           SubArray<3, SIZE, DeviceType> axis_sum,
           SubArray<2, SIZE, DeviceType> axis_min,
           SubArray<2, SIZE, DeviceType> axis_max,
           SubArray<3, SIZE, DeviceType> edges, int queue_idx) {
    using FunctorType = Pass1Functor<T, DeviceType>;
    FunctorType functor(nr, nc, nf, v, iso_value, axis_sum, axis_min, axis_max,
                        edges);

    SIZE total_thread_z = 1;
    SIZE total_thread_y = nr;
    SIZE total_thread_x = nf;

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);

    // printf("%u %u %u\n", shape.dataHost()[2], shape.dataHost()[1],
    // shape.dataHost()[0]); PrintSubarray("shape", shape);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size,
                queue_idx);
  }

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<Pass2Functor<T, DeviceType>>
  GenTask2(SIZE nr, SIZE nc, SIZE nf, SubArray<3, SIZE, DeviceType> axis_sum,
           SubArray<2, SIZE, DeviceType> axis_min,
           SubArray<2, SIZE, DeviceType> axis_max,
           SubArray<3, SIZE, DeviceType> edges,
           SubArray<2, SIZE, DeviceType> cell_tri_count, int queue_idx) {
    using FunctorType = Pass2Functor<T, DeviceType>;
    FunctorType functor(nr, nc, nf, axis_sum, axis_min, axis_max, edges,
                        cell_tri_count);

    SIZE total_thread_z = 1;
    SIZE total_thread_y = nr - 1;
    SIZE total_thread_x = nf - 1;

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);

    // printf("%u %u %u\n", shape.dataHost()[2], shape.dataHost()[1],
    // shape.dataHost()[0]); PrintSubarray("shape", shape);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size,
                queue_idx);
  }

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<Pass4Functor<T, DeviceType>>
  GenTask4(SIZE nr, SIZE nc, SIZE nf, SubArray<3, T, DeviceType> v, T iso_value,
           SubArray<3, SIZE, DeviceType> axis_sum,
           SubArray<2, SIZE, DeviceType> axis_min,
           SubArray<2, SIZE, DeviceType> axis_max,
           SubArray<1, SIZE, DeviceType> cell_tri_count_scan,
           SubArray<3, SIZE, DeviceType> edges,
           SubArray<1, SIZE, DeviceType> triangle_topology,
           SubArray<1, T, DeviceType> points, int queue_idx) {
    using FunctorType = Pass4Functor<T, DeviceType>;
    FunctorType functor(nr, nc, nf, v, iso_value, axis_sum, axis_min, axis_max,
                        cell_tri_count_scan, edges, triangle_topology, points);

    SIZE total_thread_z = 1;
    SIZE total_thread_y = nr - 1;
    SIZE total_thread_x = nf - 1;

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);

    // printf("%u %u %u\n", shape.dataHost()[2], shape.dataHost()[1],
    // shape.dataHost()[0]); PrintSubarray("shape", shape);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size,
                queue_idx);
  }

  MGARDX_CONT
  void Execute(SIZE nr, SIZE nc, SIZE nf, SubArray<3, T, DeviceType> v,
               T iso_value, Array<1, SIZE, DeviceType> &Triangles,
               Array<1, T, DeviceType> &Points, int queue_idx) {

    Timer t;

    const bool pitched = false;

    Array<3, SIZE, DeviceType> axis_sum_array({nr, nf, 3}, pitched);
    Array<2, SIZE, DeviceType> axis_min_array({nr, nf}, pitched);
    Array<2, SIZE, DeviceType> axis_max_array({nr, nf}, pitched);
    Array<3, SIZE, DeviceType> edges_array({nr, nc, nf}, pitched);
    Array<2, SIZE, DeviceType> cell_tri_count_array({nr - 1, nf - 1}, pitched);
    Array<1, SIZE, DeviceType> cell_tri_count_array_scan(
        {(nr - 1) * (nf - 1) + 1}, pitched);

    SubArray axis_sum(axis_sum_array);
    SubArray axis_min(axis_min_array);
    SubArray axis_max(axis_max_array);
    SubArray edges(edges_array);
    SubArray cell_tri_count(cell_tri_count_array);
    SubArray cell_tri_count_scan(cell_tri_count_array_scan);

    using FunctorType1 = Pass1Functor<T, DeviceType>;
    using TaskType1 = Task<FunctorType1>;
    TaskType1 task1 = GenTask1<1, 8, 8>(nr, nc, nf, v, iso_value, axis_sum,
                                        axis_min, axis_max, edges, queue_idx);

    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    t.start();
    DeviceAdapter<TaskType1, DeviceType>().Execute(task1);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    t.end();
    t.print("Pass 1");
    t.clear();

    // printf("After pass1\n");
    // PrintSubarray("v", SubArray(v));
    // PrintSubarray("axis_sum", SubArray(axis_sum).Linearize());
    // PrintSubarray("axis_min", SubArray(axis_min));
    // PrintSubarray("axis_max", SubArray(axis_max));
    // PrintSubarray("edges", SubArray(edges));

    using FunctorType2 = Pass2Functor<T, DeviceType>;
    using TaskType2 = Task<FunctorType2>;
    TaskType2 task2 =
        GenTask2<1, 8, 8>(nr, nc, nf, axis_sum, axis_min, axis_max, edges,
                          cell_tri_count, queue_idx);

    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    t.start();
    DeviceAdapter<TaskType2, DeviceType>().Execute(task2);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    t.end();
    t.print("Pass 2");
    t.clear();

    // printf("After pass2\n");
    // PrintSubarray("axis_sum", SubArray(axis_sum).Linearize());
    // PrintSubarray("cell_tri_count", SubArray(cell_tri_count));

    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    t.start();

    SubArray<1, SIZE, DeviceType> cell_tri_count_liearized =
        cell_tri_count.Linearize();

    DeviceCollective<DeviceType>::ScanSumExtended((nr - 1) * (nf - 1), cell_tri_count_liearized,
                               cell_tri_count_scan, queue_idx);

    SIZE numTris = 0;
    MemoryManager<DeviceType>().Copy1D(
        &numTris, cell_tri_count_scan((nr - 1) * (nf - 1)), 1, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    SubArray<1, SIZE, DeviceType> axis_sum_liearized = axis_sum.Linearize();

    Array<1, SIZE, DeviceType> newPointSize_array({1});
    SubArray<1, SIZE, DeviceType> newPointSize_subarray(newPointSize_array);

    DeviceCollective<DeviceType>::Sum(nr * nf * 3, axis_sum_liearized, newPointSize_subarray,
                   queue_idx);

    SIZE newPointSize = *(newPointSize_array.hostCopy());

    DeviceCollective<DeviceType>::ScanSumExclusive(nr * nf * 3, axis_sum_liearized,
                                axis_sum_liearized, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    t.end();
    t.print("Pass 3");
    t.clear();

    // printf("After pass3\n");
    std::cout << "numTris: " << numTris << "\n";
    // PrintSubarray("cell_tri_count_scan", SubArray(cell_tri_count_scan));
    // std::cout << "newPointSize: " << newPointSize << "\n";
    // PrintSubarray("axis_sum_liearized", axis_sum_liearized);

    Triangles = Array<1, SIZE, DeviceType>({numTris * 3}, pitched);
    Points = Array<1, T, DeviceType>({newPointSize * 3}, pitched);

    if (numTris == 0 || newPointSize == 0) {
      printf("returing 0 from FlyingEdges\n");
      return;
    }

    SubArray<1, SIZE, DeviceType> triangle_topology(Triangles);
    SubArray<1, T, DeviceType> points(Points);

    using FunctorType4 = Pass4Functor<T, DeviceType>;
    using TaskType4 = Task<FunctorType4>;
    TaskType4 task4 = GenTask4<1, 16, 16>(
        nr, nc, nf, v, iso_value, axis_sum, axis_min, axis_max,
        cell_tri_count_scan, edges, triangle_topology, points, queue_idx);

    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    t.start();
    DeviceAdapter<TaskType4, DeviceType>().Execute(task4);

    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    t.end();
    t.print("Pass 4");
    t.clear();

    // printf("After pass4\n");
    // PrintSubarray("triangle_topology", triangle_topology);
    // PrintSubarray("points", points);
  }
};

} // namespace mgard_x

#endif