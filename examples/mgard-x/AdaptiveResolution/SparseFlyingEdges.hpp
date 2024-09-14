/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_SPARSE_FLYING_EDGES_HPP
#define MGARD_X_SPARSE_FLYING_EDGES_HPP

#include "mgard/mgard-x/RuntimeX/RuntimeX.h"

// avoid copying device functions
#include "FlyingEdges.hpp"

namespace mgard_x {

template <typename T, typename DeviceType>
class SFE_Pass1Functor : public Functor<DeviceType> {
public:
  MGARDX_CONT SFE_Pass1Functor(SIZE nr, SIZE nc, SIZE nf,
                               SubArray<1, T, DeviceType> *start_value,
                               SubArray<1, T, DeviceType> *end_value,
                               T iso_value,
                               SubArray<1, SIZE, DeviceType> *edge_cases)
      : nr(nr), nc(nc), nf(nf), start_value(start_value), end_value(end_value),
        iso_value(iso_value), edge_cases(edge_cases) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE f = FunctorBase<DeviceType>::GetBlockIdX() *
                 FunctorBase<DeviceType>::GetBlockDimX() +
             FunctorBase<DeviceType>::GetThreadIdX();
    SIZE c = FunctorBase<DeviceType>::GetBlockIdY() *
                 FunctorBase<DeviceType>::GetBlockDimY() +
             FunctorBase<DeviceType>::GetThreadIdY();
    SIZE r = FunctorBase<DeviceType>::GetBlockIdZ() *
                 FunctorBase<DeviceType>::GetBlockDimZ() +
             FunctorBase<DeviceType>::GetThreadIdZ();

    if (c >= nc || r >= nr || f >= start_value[r * nc + c].getShape(0)) {
      return;
    }
    T start = *start_value[r * nc + c](f);
    T end = *end_value[r * nc + c](f);
    SIZE edge_case = MGARD_Below;
    if (start >= iso_value)
      edge_case = MGARD_LeftAbove;
    if (end >= iso_value)
      edge_case |= MGARD_RightAbove;
    *edge_cases[r * nc + c](f) = edge_case;

    // printf("start: %f, end: %f, edge_case: %u\n",
    //         start, end, edge_case);
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    return size;
  }

private:
  SIZE nr, nc, nf;
  SubArray<1, T, DeviceType> *start_value;
  SubArray<1, T, DeviceType> *end_value;
  SubArray<1, SIZE, DeviceType> *edge_cases;
  T iso_value;
};

template <typename T, typename DeviceType>
class SFE_Pass2Functor : public Functor<DeviceType> {
public:
  MGARDX_CONT SFE_Pass2Functor(SIZE nr, SIZE nc, SIZE nf,
                               SubArray<1, SIZE, DeviceType> *role,
                               SubArray<1, SIZE, DeviceType> *pZ_index,
                               SubArray<1, SIZE, DeviceType> *neighbor,
                               SubArray<1, SIZE, DeviceType> *edge_cases,
                               SubArray<1, SIZE, DeviceType> *cell_ids,
                               SubArray<1, SIZE, DeviceType> cell_cases,
                               SubArray<1, SIZE, DeviceType> point_count,
                               SubArray<1, SIZE, DeviceType> tri_count)
      : nr(nr), nc(nc), nf(nf), role(role), pZ_index(pZ_index),
        neighbor(neighbor), edge_cases(edge_cases), cell_ids(cell_ids),
        cell_cases(cell_cases), point_count(point_count), tri_count(tri_count) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE f = FunctorBase<DeviceType>::GetBlockIdX() *
                 FunctorBase<DeviceType>::GetBlockDimX() +
             FunctorBase<DeviceType>::GetThreadIdX();
    SIZE c = FunctorBase<DeviceType>::GetBlockIdY() *
                 FunctorBase<DeviceType>::GetBlockDimY() +
             FunctorBase<DeviceType>::GetThreadIdY();
    SIZE r = FunctorBase<DeviceType>::GetBlockIdZ() *
                 FunctorBase<DeviceType>::GetBlockDimZ() +
             FunctorBase<DeviceType>::GetThreadIdZ();

    if (c >= nc || r >= nr || f >= role[r * nc + c].getShape(0)) {
      return;
    }

    // local cell
    if (*role[r * nc + c](f) == LEAD) {
      SIZE local_cell_id = *cell_ids[r * nc + c](f);
      SIZE local_pZ_index = *pZ_index[r * nc + c](f);

      SIZE e0 = *edge_cases[r * nc + c](f);
      SIZE e1 = *edge_cases[r * nc + c](f + 1);
      SIZE e2 = *edge_cases[(r + 1) * nc + c](local_pZ_index);
      SIZE e3 = *edge_cases[(r + 1) * nc + c](local_pZ_index + 1);

      SIZE local_cell_case = (e0 | (e1 << 2) | (e2 << 4) | (e3 << 6));
      // printf("edge case: %u %u %u %u, cell_case: %u\n",
      //         e0, e1, e2, e3, local_cell_case);

      // printf("cell_cases(%u %u %u): %u\n", r, c, f, local_cell_case);
      *cell_cases(local_cell_id) = local_cell_case;
      SIZE local_tri_count =
          flying_edges::GetNumberOfPrimitives(local_cell_case);
      *tri_count(local_cell_id) = local_tri_count;

      if (local_tri_count > 0) {
        SIZE const *edgeUses = flying_edges::GetEdgeUses(local_cell_case);
        SIZE local_neighbor = *neighbor[r * nc + c](f);
        SIZE local_edge_count = 0;

        // 1st priority
        local_edge_count += edgeUses[0];
        local_edge_count += edgeUses[8];
        local_edge_count += edgeUses[4];

        // 2nd priority
        if (!(local_neighbor & pX)) {
          local_edge_count += edgeUses[1];
        }
        if (!(local_neighbor & pY)) {
          local_edge_count += edgeUses[5];
        }
        if (!(local_neighbor & pY)) {
          local_edge_count += edgeUses[9];
        }

        // 3rd priority
        if (!(local_neighbor & pZ) && !(local_neighbor & nXpZ)) {
          local_edge_count += edgeUses[2];
        }
        if (!(local_neighbor & pZ) && !(local_neighbor & nYpZ)) {
          local_edge_count += edgeUses[6];
        }
        if (!(local_neighbor & pX) && !(local_neighbor & pXnY)) {
          local_edge_count += edgeUses[10];
        }

        // 4th priority
        if (!(local_neighbor & pXpZ) && !(local_neighbor & pZ) &&
            !(local_neighbor & pX)) {
          local_edge_count += edgeUses[3];
        }
        if (!(local_neighbor & pYpZ) && !(local_neighbor & pZ) &&
            !(local_neighbor & pY)) {
          local_edge_count += edgeUses[7];
        }
        if (!(local_neighbor & pXpY) && !(local_neighbor & pX) &&
            !(local_neighbor & pY)) {
          local_edge_count += edgeUses[11];
        }

        *point_count(local_cell_id) = local_edge_count;
        // printf("local_cell_id: %u local_edge_count: %u\n", local_cell_id,
        // local_edge_count);
      } else {
        *point_count(local_cell_id) = 0;
      }
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    return size;
  }

private:
  SIZE nr, nc, nf;
  SubArray<1, SIZE, DeviceType> *role;
  SubArray<1, SIZE, DeviceType> *pZ_index;
  SubArray<1, SIZE, DeviceType> *neighbor;
  SubArray<1, SIZE, DeviceType> *edge_cases;
  SubArray<1, SIZE, DeviceType> *cell_ids;
  SubArray<1, SIZE, DeviceType> cell_cases;
  SubArray<1, SIZE, DeviceType> point_count;
  SubArray<1, SIZE, DeviceType> tri_count;
};

template <typename T, typename DeviceType>
class SFE_Pass3Functor : public Functor<DeviceType> {
public:
  MGARDX_CONT SFE_Pass3Functor(SIZE nr, SIZE nc, SIZE nf,
                               SubArray<1, SIZE, DeviceType> *role,
                               SubArray<1, SIZE, DeviceType> *pY_index,
                               SubArray<1, SIZE, DeviceType> *pZ_index,
                               SubArray<1, SIZE, DeviceType> *neighbor,
                               SubArray<1, SIZE, DeviceType> *cell_ids,
                               SubArray<1, SIZE, DeviceType> cell_cases,
                               SubArray<1, SIZE, DeviceType> point_count_scan,
                               SubArray<1, SIZE, DeviceType> tri_count_scan,
                               SubArray<1, SIZE, DeviceType> *edgeX_ids,
                               SubArray<1, SIZE, DeviceType> *edgeY_ids,
                               SubArray<1, SIZE, DeviceType> *edgeZ_ids)
      : nr(nr), nc(nc), nf(nf), role(role), pY_index(pY_index),
        pZ_index(pZ_index), neighbor(neighbor), cell_ids(cell_ids),
        cell_cases(cell_cases), point_count_scan(point_count_scan),
        tri_count_scan(tri_count_scan), edgeX_ids(edgeX_ids),
        edgeY_ids(edgeY_ids), edgeZ_ids(edgeZ_ids) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE f = FunctorBase<DeviceType>::GetBlockIdX() *
                 FunctorBase<DeviceType>::GetBlockDimX() +
             FunctorBase<DeviceType>::GetThreadIdX();
    SIZE c = FunctorBase<DeviceType>::GetBlockIdY() *
                 FunctorBase<DeviceType>::GetBlockDimY() +
             FunctorBase<DeviceType>::GetThreadIdY();
    SIZE r = FunctorBase<DeviceType>::GetBlockIdZ() *
                 FunctorBase<DeviceType>::GetBlockDimZ() +
             FunctorBase<DeviceType>::GetThreadIdZ();

    if (c >= nc || r >= nr || f >= role[r * nc + c].getShape(0)) {
      return;
    }

    // local cell
    if (*role[r * nc + c](f) == LEAD) {
      SIZE local_cell_id = *cell_ids[r * nc + c](f);
      SIZE local_pZ_index = *pZ_index[r * nc + c](f);
      SIZE local_pY_index = *pY_index[r * nc + c](f);
      if (c == nc - 1) {
        local_pY_index = f;
      }

      // printf("local_pY_index: %u\n", local_pY_index);

      // printf("rcf: %u %u %u, local_pY_index: %u\n", r, c, f, local_pY_index);
      SIZE prev_point_count = *point_count_scan(local_cell_id);
      SIZE local_cell_case = *cell_cases(local_cell_id);
      SIZE local_tri_count =
          flying_edges::GetNumberOfPrimitives(local_cell_case);

      if (local_tri_count > 0) {
        SIZE const *edgeUses = flying_edges::GetEdgeUses(local_cell_case);
        SIZE local_neighbor = *neighbor[r * nc + c](f);

        // printf("rcf: %u %u %u, edge use: %u %u %u\n",
        //         r, c, f, edgeUses[0], edgeUses[4], edgeUses[8]);

        // 1st priority
        if (edgeUses[4]) {
          // if (1) {
          //   printf("edgeUses[4]: %u\n", prev_point_count);
          // }
          *edgeX_ids[r * (nc + 1) + c](f) = prev_point_count;
          prev_point_count++;
        }
        if (edgeUses[0]) {
          // if (1) {
          //   printf("edgeUses[0]: %u\n", prev_point_count);
          // }
          *edgeY_ids[r * (nc + 1) + c](f) = prev_point_count;
          prev_point_count++;
        }
        if (edgeUses[8]) {
          // if (1) {
          //   printf("edgeUses[8]: %u\n", prev_point_count);
          // }
          *edgeZ_ids[r * (nc + 1) + c](f) = prev_point_count;
          prev_point_count++;
        }

        // // 2nd priority
        if (edgeUses[1] && !(local_neighbor & pX)) {
          // local_edge_count += edgeUses[1];
          // if (1) {
          //   printf("edgeUses[1]: %u\n", prev_point_count);
          // }
          *edgeY_ids[r * (nc + 1) + c](f + 1) = prev_point_count;
          prev_point_count++;
        }
        if (edgeUses[5] && !(local_neighbor & pY)) {
          // local_edge_count += edgeUses[5];
          // if (1) {
          // printf("edgeUses[5]: %u\n", prev_point_count);
          // printf("local_pY_index: %u, local_neighbor: %u\n", local_pY_index,
          // local_neighbor);
          // }
          if (r == 1 & c == 6 && f == 2) {
            printf("handle 5: prev_point_count = %u\n", prev_point_count);
          }

          *edgeX_ids[r * (nc + 1) + c + 1](local_pY_index) = prev_point_count;
          prev_point_count++;
        }
        if (edgeUses[9] && !(local_neighbor & pY)) {
          // local_edge_count += edgeUses[9];
          // if (1) {
          //   printf("cell: %u rcf: %u %u %u edgeUses[9]: %u\n",local_cell_id,
          //   r, c, local_pY_index, prev_point_count);
          // }
          *edgeZ_ids[r * (nc + 1) + c + 1](local_pY_index) = prev_point_count;
          prev_point_count++;
        }

        // 3rd priority
        if (edgeUses[2] && !(local_neighbor & pZ) && !(local_neighbor & nXpZ)) {
          // local_edge_count += edgeUses[2];
          // if (1) {
          //   printf("edgeUses[2]: %u\n", prev_point_count);
          // }
          *edgeY_ids[(r + 1) * (nc + 1) + c](local_pZ_index) = prev_point_count;
          prev_point_count++;
        }
        if (edgeUses[6] && !(local_neighbor & pZ) && !(local_neighbor & nYpZ)) {
          // local_edge_count += edgeUses[6];
          // if (1) {
          //   printf("edgeUses[6]: %u\n", prev_point_count);
          // }
          *edgeX_ids[(r + 1) * (nc + 1) + c](local_pZ_index) = prev_point_count;
          prev_point_count++;
        }
        if (edgeUses[10] && !(local_neighbor & pX) &&
            !(local_neighbor & pXnY)) {
          // local_edge_count += edgeUses[10];
          // if (1) {
          //   printf("edgeUses[10]: %u\n", prev_point_count);
          // }
          *edgeZ_ids[r * (nc + 1) + c](f + 1) = prev_point_count;
          prev_point_count++;
        }

        // 4th priority
        if (edgeUses[3] && !(local_neighbor & pXpZ) && !(local_neighbor & pZ) &&
            !(local_neighbor & pX)) {
          // local_edge_count += edgeUses[3];
          // if (1) {
          //   printf("edgeUses[3]: %u\n", prev_point_count);
          // }
          SIZE local_pXpZ_index = local_pZ_index + 1;
          *edgeY_ids[(r + 1) * (nc + 1) + c](local_pXpZ_index) =
              prev_point_count;
          prev_point_count++;
        }
        if (edgeUses[7] && !(local_neighbor & pYpZ) && !(local_neighbor & pZ) &&
            !(local_neighbor & pY)) {
          // local_edge_count += edgeUses[7];
          // if (1) {
          //   printf("edgeUses[7]: %u\n", prev_point_count);
          // }
          SIZE local_pYpZ_index = *pZ_index[r * nc + c + 1](local_pY_index);
          *edgeX_ids[(r + 1) * (nc + 1) + (c + 1)](local_pYpZ_index) =
              prev_point_count;
          prev_point_count++;
        }
        if (edgeUses[11] && !(local_neighbor & pXpY) &&
            !(local_neighbor & pX) && !(local_neighbor & pY)) {
          // local_edge_count += edgeUses[11];
          // if (1) {
          //   printf("edgeUses[11]: %u\n", prev_point_count);
          // }
          SIZE local_pXpY_index = local_pY_index + 1;
          *edgeZ_ids[r * (nc + 1) + (c + 1)](local_pXpY_index) =
              prev_point_count;
          prev_point_count++;
        }
      }
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    return size;
  }

private:
  SIZE nr, nc, nf;
  SubArray<1, SIZE, DeviceType> *role;
  SubArray<1, SIZE, DeviceType> *pY_index;
  SubArray<1, SIZE, DeviceType> *pZ_index;
  SubArray<1, SIZE, DeviceType> *neighbor;
  SubArray<1, SIZE, DeviceType> *cell_ids;
  SubArray<1, SIZE, DeviceType> cell_cases;
  SubArray<1, SIZE, DeviceType> point_count_scan;
  SubArray<1, SIZE, DeviceType> tri_count_scan;
  SubArray<1, SIZE, DeviceType> *edgeX_ids;
  SubArray<1, SIZE, DeviceType> *edgeY_ids;
  SubArray<1, SIZE, DeviceType> *edgeZ_ids;
};

template <typename T, typename DeviceType>
MGARDX_EXEC void InterpolateEdge(SIZE edgeNum, SIZE x, SIZE y, SIZE z,
                                 SIZE const *edgeUses, SIZE *edgeIds,
                                 T iso_value, T *v,
                                 SubArray<1, T, DeviceType> &points) {
  if (!edgeUses[edgeNum]) {
    return;
  }

  // printf("writeIndex: %u\n", writeIndex);
  SIZE writeIndex = edgeIds[edgeNum] * 3;
  // printf("writeIndex: %u\n", writeIndex);

  SIZE const *verts = flying_edges::GetVertMap(edgeNum);
  SIZE const *offsets0 = flying_edges::GetVertOffsets(verts[0]);
  SIZE const *offsets1 = flying_edges::GetVertOffsets(verts[1]);

  SIZE z0 = z + offsets0[2];
  SIZE y0 = y + offsets0[1];
  SIZE x0 = x + offsets0[0];

  SIZE z1 = z + offsets1[2];
  SIZE y1 = y + offsets1[1];
  SIZE x1 = x + offsets1[0];

  T s0 = v[verts[0]];
  T s1 = v[verts[1]];

  // printf("s: %f %f\n", s0, s1);

  T w = (iso_value - s0) / (s1 - s0);

  // if (writeIndex == 0) {
  // printf("x: %u %u %u\n", x, y, z);
  // printf("x0: %u %u %u\n", x0, y0, z0);
  // printf("x1: %u %u %u\n", x1, y1, z1);
  //   printf("iso_value: %f, s0: %f, s1: %f, %f %f w: %f, writeIndex: %u\n",
  //             iso_value, s0, s1, (iso_value - s0), (s1 - s0), w, writeIndex);
  //   printf("point: %f %f %f, writeIndex: %u\n",
  //            (1 - w) * x0 + w * x1, (1 - w) * y0 + w * y1, (1 - w) * z0 + w *
  //            z1, writeIndex);
  // }

  // printf("index: %u %u %u, edgeNum: %u, point %f %f %f, writeIndex: %u\n",
  //         z, y, x, edgeNum,
  //         (1 - w) * x0 + w * x1,
  //         (1 - w) * y0 + w * y1,
  //         (1 - w) * z0 + w * z1,
  //         writeIndex);

  *points(writeIndex) = (1 - w) * x0 + w * x1;
  *points(writeIndex + 1) = (1 - w) * y0 + w * y1;
  *points(writeIndex + 2) = (1 - w) * z0 + w * z1;
}

template <typename T, typename DeviceType>
class SFE_Pass4Functor : public Functor<DeviceType> {
public:
  MGARDX_CONT SFE_Pass4Functor(SIZE nr, SIZE nc, SIZE nf,
                               SubArray<1, T, DeviceType> *start_value,
                               SubArray<1, T, DeviceType> *end_value,
                               SubArray<1, SIZE, DeviceType> *index,
                               SubArray<1, SIZE, DeviceType> *role,
                               SubArray<1, SIZE, DeviceType> *pY_index,
                               SubArray<1, SIZE, DeviceType> *pZ_index,
                               SubArray<1, SIZE, DeviceType> *neighbor,
                               SubArray<1, SIZE, DeviceType> *cell_ids,
                               SubArray<1, SIZE, DeviceType> *level_index,
                               T iso_value,
                               SubArray<1, SIZE, DeviceType> cell_cases,
                               SubArray<1, SIZE, DeviceType> *edgeX_ids,
                               SubArray<1, SIZE, DeviceType> *edgeY_ids,
                               SubArray<1, SIZE, DeviceType> *edgeZ_ids,
                               SubArray<1, SIZE, DeviceType> tri_count_scan,
                               SubArray<1, T, DeviceType> points,
                               SubArray<1, SIZE, DeviceType> triangles)
      : nr(nr), nc(nc), nf(nf), start_value(start_value), end_value(end_value),
        index(index), role(role), pY_index(pY_index), pZ_index(pZ_index),
        neighbor(neighbor), cell_ids(cell_ids), level_index(level_index),
        iso_value(iso_value), cell_cases(cell_cases), edgeX_ids(edgeX_ids),
        edgeY_ids(edgeY_ids), edgeZ_ids(edgeZ_ids),
        tri_count_scan(tri_count_scan), points(points), triangles(triangles) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE f = FunctorBase<DeviceType>::GetBlockIdX() *
                 FunctorBase<DeviceType>::GetBlockDimX() +
             FunctorBase<DeviceType>::GetThreadIdX();
    SIZE c = FunctorBase<DeviceType>::GetBlockIdY() *
                 FunctorBase<DeviceType>::GetBlockDimY() +
             FunctorBase<DeviceType>::GetThreadIdY();
    SIZE r = FunctorBase<DeviceType>::GetBlockIdZ() *
                 FunctorBase<DeviceType>::GetBlockDimZ() +
             FunctorBase<DeviceType>::GetThreadIdZ();

    if (c >= nc || r >= nr || f >= role[r * nc + c].getShape(0)) {
      return;
    }

    // local cell
    if (*role[r * nc + c](f) == LEAD) {
      SIZE local_cell_id = *cell_ids[r * nc + c](f);
      SIZE local_pZ_index = *pZ_index[r * nc + c](f);
      SIZE local_pY_index = 0;
      if (c < nc - 1) {
        local_pY_index = *pY_index[r * nc + c](f);
      } else {
        local_pY_index = f;
      }

      // if (c < nc - 1) {
      //   printf("r * nc + c+1: %u local_pY_index: %u, shape: %u\n", r * nc +
      //   c+1, local_pY_index, pZ_index[r * nc + c+1].getShape()[0]);
      // }

      SIZE local_pXpZ_index = local_pZ_index + 1;
      SIZE local_pYpZ_index = 0;
      if (c < nc - 1 &&
          local_pY_index < pZ_index[r * nc + c + 1].getShape()[0]) {
        local_pYpZ_index = *pZ_index[r * nc + c + 1](local_pY_index);
      } else {
        local_pYpZ_index = f;
      }
      SIZE local_pXpY_index = local_pY_index + 1;

      // printf("local_pYpZ_index: %u\n", local_pYpZ_index);

      T v[8];

      // printf("local_pY_index: %u %u %u %u %u\n", local_pY_index,
      // local_pZ_index, local_pXpZ_index, local_pYpZ_index, local_pXpY_index);

      // Get edge case
      v[0] = *start_value[r * nc + c](f);
      v[1] = *end_value[r * nc + c](f);
      v[2] = *start_value[r * nc + c](f + 1);
      v[3] = *end_value[r * nc + c](f + 1);
      v[4] = *start_value[(r + 1) * nc + c](local_pZ_index);
      v[5] = *end_value[(r + 1) * nc + c](local_pZ_index);
      v[6] = *start_value[(r + 1) * nc + c](local_pZ_index + 1);
      v[7] = *end_value[(r + 1) * nc + c](local_pZ_index + 1);

      SIZE x = *level_index[0](*index[r * nc + c](f));
      SIZE y = *level_index[1](c);
      SIZE z = *level_index[2](r);

      SIZE edgeIds[12];
      edgeIds[0] = *edgeY_ids[r * (nc + 1) + c](f);
      edgeIds[4] = *edgeX_ids[r * (nc + 1) + c](f);
      edgeIds[8] = *edgeZ_ids[r * (nc + 1) + c](f);
      edgeIds[1] = *edgeY_ids[r * (nc + 1) + c](f + 1);
      edgeIds[5] = *edgeX_ids[r * (nc + 1) + c + 1](local_pY_index);
      edgeIds[9] = *edgeZ_ids[r * (nc + 1) + c + 1](local_pY_index);
      edgeIds[2] = *edgeY_ids[(r + 1) * (nc + 1) + c](local_pZ_index);
      edgeIds[6] = *edgeX_ids[(r + 1) * (nc + 1) + c](local_pZ_index);
      edgeIds[10] = *edgeZ_ids[r * (nc + 1) + c](f + 1);
      edgeIds[3] = *edgeY_ids[(r + 1) * (nc + 1) + c](local_pXpZ_index);
      edgeIds[7] = *edgeX_ids[(r + 1) * (nc + 1) + (c + 1)](local_pYpZ_index);
      edgeIds[11] = *edgeZ_ids[r * (nc + 1) + (c + 1)](local_pXpY_index);

      // for (int i = 7; i < 8; i++)
      //   printf("edgeIds[%d]: %u\n", i, edgeIds[i]);

      // printf("data: %llu, r: %u x: %u %u %u\n", level_index, r, x, y, z);

      // printf("level_index: %llu, %llu, %llu\n",
      //     level_index[0].data(), level_index[1].data(),
      //     level_index[2].data());

      // printf("level_index: %u, %u, %u\n",
      //     *level_index[0](1), *level_index[1](1), *level_index[2](1));

      SIZE local_cell_case = *cell_cases(local_cell_id);
      SIZE local_tri_count =
          flying_edges::GetNumberOfPrimitives(local_cell_case);
      SIZE prev_tri_count = *tri_count_scan(local_cell_id);

      // printf("cell id: %u, local_tri_count: %u\n",
      //         local_cell_id, prev_tri_count);

      if (local_tri_count > 0) {
        // Sum of points in cells before me
        SIZE const *edgeUses = flying_edges::GetEdgeUses(local_cell_case);
        SIZE local_neighbor = *neighbor[r * nc + c](f);

        // printf("index: %u %u %u, neighbor: %u\n", z, y, x, local_neighbor);

        // printf("rcf: %u %u %u, edge use: %u %u %u\n",
        //         r, c, f, edgeUses[0], edgeUses[4], edgeUses[8]);

        // if (r == 2 && c == 2 && f == 1) {
        //   printf("[neighbor]rcf: %u %u %u, edge use: %u\n", r, c, f,
        //   edgeUses[4]);
        // }

        // printf("rcf: %u %u %u, edge use: %u\n",
        //         r, c, f, edgeUses[5]);

        InterpolateEdge(0, x, y, z, edgeUses, edgeIds, iso_value, v, points);
        InterpolateEdge(4, x, y, z, edgeUses, edgeIds, iso_value, v, points);
        InterpolateEdge(8, x, y, z, edgeUses, edgeIds, iso_value, v, points);

        // 2nd priority
        if (!(local_neighbor & pX)) {
          InterpolateEdge(1, x, y, z, edgeUses, edgeIds, iso_value, v, points);
        }

        if (r == 2 & c == 1 && f == 4) {
          printf("neighbor: %u \n", !(local_neighbor & pY));
        }

        if (!(local_neighbor & pY)) {
          // if (r == 1 & c == 6 && f == 2) {
          //   printf("handle 5\n");
          // }
          InterpolateEdge(5, x, y, z, edgeUses, edgeIds, iso_value, v, points);
        }
        if (!(local_neighbor & pY)) {
          InterpolateEdge(9, x, y, z, edgeUses, edgeIds, iso_value, v, points);
        }

        // 3rd priority
        if (!(local_neighbor & pZ) && !(local_neighbor & nXpZ)) {
          InterpolateEdge(2, x, y, z, edgeUses, edgeIds, iso_value, v, points);
        }
        if (!(local_neighbor & pZ) && !(local_neighbor & nYpZ)) {
          InterpolateEdge(6, x, y, z, edgeUses, edgeIds, iso_value, v, points);
        }
        if (!(local_neighbor & pX) && !(local_neighbor & pXnY)) {
          InterpolateEdge(10, x, y, z, edgeUses, edgeIds, iso_value, v, points);
        }

        // 4th priority
        if (!(local_neighbor & pXpZ) && !(local_neighbor & pZ) &&
            !(local_neighbor & pX)) {
          InterpolateEdge(3, x, y, z, edgeUses, edgeIds, iso_value, v, points);
        }

        // if (r == 0 & c == 6 && f == 2) {
        //     printf("neighbor: %u %u %u\n", !(local_neighbor & pYpZ),
        //     !(local_neighbor & pZ), !(local_neighbor & pY));
        //   }

        if (!(local_neighbor & pYpZ) && !(local_neighbor & pZ) &&
            !(local_neighbor & pY)) {
          InterpolateEdge(7, x, y, z, edgeUses, edgeIds, iso_value, v, points);
        }
        if (!(local_neighbor & pXpY) && !(local_neighbor & pX) &&
            !(local_neighbor & pY)) {
          InterpolateEdge(11, x, y, z, edgeUses, edgeIds, iso_value, v, points);
        }

        SIZE const *edges = flying_edges::GetTriEdgeCases(local_cell_case);
        SIZE edgeIndex = 1;
        for (SIZE i = 0; i < local_tri_count; ++i) {
          if (edgeIds[edges[edgeIndex]] == 2139062143) {
            printf("edgeIndex too large\n");
          }
          if (edgeIds[edges[edgeIndex + 1]] == 2139062143) {
            SIZE local_cell_id2 = *cell_ids[r * nc + c + 1](local_pY_index);
            SIZE local_cell_case2 = *cell_cases(local_cell_id2);
            SIZE const *edgeUses2 = flying_edges::GetEdgeUses(local_cell_case2);
            printf("edgeIndex+1 too large, edges[edgeIndex+1]: %u, edgeIds[5]: "
                   "%u, local_pY_index: %u, !(local_neighbor & pY): %u\n",
                   edges[edgeIndex + 1], edgeIds[5], local_pY_index,
                   !(local_neighbor & pY));
            printf("rcf: %u %u %u, edge use: %u - %u, local_cell_id: %u, "
                   "local_cell_id2: %u \n",
                   r, c, f, edgeUses[5], edgeUses2[4], local_cell_id,
                   local_cell_id2);
          }
          if (edgeIds[edges[edgeIndex + 2]] == 2139062143) {
            printf("rcf: %u %u %u edgeIndex+2 too large, edges[edgeIndex+2]: "
                   "%u, local_cell_id: %u\n",
                   r, c, f, edges[edgeIndex + 2], local_cell_id);
            SIZE local_cell_id2 =
                *cell_ids[(r + 1) * nc + c + 1](local_pYpZ_index);
            SIZE local_cell_case2 = *cell_cases(local_cell_id2);
            SIZE const *edgeUses2 = flying_edges::GetEdgeUses(local_cell_case2);
            printf("neighbor local_pYpZ_index: %u, local_cell_id2: %u\n",
                   local_pYpZ_index, local_cell_id2);
            // printf("edgeIndex+1 too large, edges[edgeIndex+1]: %u,
            // edgeIds[5]: %u, local_pY_index: %u, !(local_neighbor & pY):
            // %u\n", edges[edgeIndex+1], edgeIds[5], local_pY_index,
            // !(local_neighbor & pY));
            printf("rcf: %u %u %u, nc: %u, neighbor: %u, edge use: %u - %u\n",
                   r, c, f, nc, local_neighbor, edgeUses[7], edgeUses2[4]);
          }
          edgeIndex += 3;
        }

        generate_tris(local_cell_case, local_tri_count, edgeIds, prev_tri_count,
                      triangles);
      }
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    return size;
  }

private:
  SIZE nr, nc, nf;
  SubArray<1, T, DeviceType> *start_value;
  SubArray<1, T, DeviceType> *end_value;
  SubArray<1, SIZE, DeviceType> *index;
  SubArray<1, SIZE, DeviceType> *role;
  SubArray<1, SIZE, DeviceType> *pY_index;
  SubArray<1, SIZE, DeviceType> *pZ_index;
  SubArray<1, SIZE, DeviceType> *neighbor;
  SubArray<1, SIZE, DeviceType> *cell_ids;
  SubArray<1, SIZE, DeviceType> *level_index;
  T iso_value;
  SubArray<1, SIZE, DeviceType> cell_cases;
  SubArray<1, SIZE, DeviceType> *edgeX_ids;
  SubArray<1, SIZE, DeviceType> *edgeY_ids;
  SubArray<1, SIZE, DeviceType> *edgeZ_ids;
  SubArray<1, SIZE, DeviceType> tri_count_scan;
  SubArray<1, T, DeviceType> points;
  SubArray<1, SIZE, DeviceType> triangles;
};

template <DIM D, typename T, typename DeviceType>
class SparseFlyingEdges : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  SparseFlyingEdges() : AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<SFE_Pass1Functor<T, DeviceType>>
  GenTask1(SIZE nr, SIZE nc, SIZE nf, SubArray<1, T, DeviceType> *start_value,
           SubArray<1, T, DeviceType> *end_value, T iso_value,
           SubArray<1, SIZE, DeviceType> *edge_cases, int queue_idx) {
    using FunctorType = SFE_Pass1Functor<T, DeviceType>;
    FunctorType functor(nr, nc, nf, start_value, end_value, iso_value,
                        edge_cases);

    SIZE total_thread_z = nr;
    SIZE total_thread_y = nc;
    SIZE total_thread_x = nf;

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((double)total_thread_z / tbz);
    gridy = ceil((double)total_thread_y / tby);
    gridx = ceil((double)total_thread_x / tbx);

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size,
                queue_idx);
  }

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<SFE_Pass2Functor<T, DeviceType>>
  GenTask2(SIZE nr, SIZE nc, SIZE nf, SubArray<1, SIZE, DeviceType> *role,
           SubArray<1, SIZE, DeviceType> *pZ_index,
           SubArray<1, SIZE, DeviceType> *neighbor,
           SubArray<1, SIZE, DeviceType> *edge_cases,
           SubArray<1, SIZE, DeviceType> *cell_ids,
           SubArray<1, SIZE, DeviceType> cell_cases,
           SubArray<1, SIZE, DeviceType> point_count,
           SubArray<1, SIZE, DeviceType> tri_count, int queue_idx) {
    using FunctorType = SFE_Pass2Functor<T, DeviceType>;
    FunctorType functor(nr, nc, nf, role, pZ_index, neighbor, edge_cases,
                        cell_ids, cell_cases, point_count, tri_count);

    SIZE total_thread_z = nr;
    SIZE total_thread_y = nc;
    SIZE total_thread_x = nf;

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((double)total_thread_z / tbz);
    gridy = ceil((double)total_thread_y / tby);
    gridx = ceil((double)total_thread_x / tbx);

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size,
                queue_idx);
  }

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<SFE_Pass3Functor<T, DeviceType>>
  GenTask3(SIZE nr, SIZE nc, SIZE nf, SubArray<1, SIZE, DeviceType> *role,
           SubArray<1, SIZE, DeviceType> *pY_index,
           SubArray<1, SIZE, DeviceType> *pZ_index,
           SubArray<1, SIZE, DeviceType> *neighbor,
           SubArray<1, SIZE, DeviceType> *cell_ids,
           SubArray<1, SIZE, DeviceType> cell_cases,
           SubArray<1, SIZE, DeviceType> point_count_scan,
           SubArray<1, SIZE, DeviceType> tri_count_scan,
           SubArray<1, SIZE, DeviceType> *edgeX_ids,
           SubArray<1, SIZE, DeviceType> *edgeY_ids,
           SubArray<1, SIZE, DeviceType> *edgeZ_ids, int queue_idx) {
    using FunctorType = SFE_Pass3Functor<T, DeviceType>;
    FunctorType functor(nr, nc, nf, role, pY_index, pZ_index, neighbor,
                        cell_ids, cell_cases, point_count_scan, tri_count_scan,
                        edgeX_ids, edgeY_ids, edgeZ_ids);

    SIZE total_thread_z = nr;
    SIZE total_thread_y = nc;
    SIZE total_thread_x = nf;

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((double)total_thread_z / tbz);
    gridy = ceil((double)total_thread_y / tby);
    gridx = ceil((double)total_thread_x / tbx);

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size,
                queue_idx);
  }

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<SFE_Pass4Functor<T, DeviceType>>
  GenTask4(SIZE nr, SIZE nc, SIZE nf, SubArray<1, T, DeviceType> *start_value,
           SubArray<1, T, DeviceType> *end_value,
           SubArray<1, SIZE, DeviceType> *index,
           SubArray<1, SIZE, DeviceType> *role,
           SubArray<1, SIZE, DeviceType> *pY_index,
           SubArray<1, SIZE, DeviceType> *pZ_index,
           SubArray<1, SIZE, DeviceType> *neighbor,
           SubArray<1, SIZE, DeviceType> *cell_ids,
           SubArray<1, SIZE, DeviceType> *level_index, T iso_value,
           SubArray<1, SIZE, DeviceType> cell_cases,
           SubArray<1, SIZE, DeviceType> *edgeX_ids,
           SubArray<1, SIZE, DeviceType> *edgeY_ids,
           SubArray<1, SIZE, DeviceType> *edgeZ_ids,
           SubArray<1, SIZE, DeviceType> tri_count_scan,
           SubArray<1, T, DeviceType> points,
           SubArray<1, SIZE, DeviceType> triangles, int queue_idx) {
    using FunctorType = SFE_Pass4Functor<T, DeviceType>;
    FunctorType functor(nr, nc, nf, start_value, end_value, index, role,
                        pY_index, pZ_index, neighbor, cell_ids, level_index,
                        iso_value, cell_cases, edgeX_ids, edgeY_ids, edgeZ_ids,
                        tri_count_scan, points, triangles);

    // PrintSubarray("tri_count_scan", tri_count_scan);

    SIZE total_thread_z = nr;
    SIZE total_thread_y = nc;
    SIZE total_thread_x = nf;

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((double)total_thread_z / tbz);
    gridy = ceil((double)total_thread_y / tby);
    gridx = ceil((double)total_thread_x / tbx);

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size,
                queue_idx);
  }

  MGARDX_CONT
  void Execute(CompressedSparseEdge<D, T, DeviceType> &cse, T iso_value,
               Array<1, SIZE, DeviceType> &Triangles,
               Array<1, T, DeviceType> &Points, int queue_idx) {
    using Mem = MemoryManager<DeviceType>;

    SubArray<1, T, DeviceType> *start_value = NULL;
    SubArray<1, T, DeviceType> *end_value = NULL;
    SubArray<1, SIZE, DeviceType> *index = NULL;
    SubArray<1, SIZE, DeviceType> *role = NULL;
    SubArray<1, SIZE, DeviceType> *pY_index = NULL;
    SubArray<1, SIZE, DeviceType> *nY_index = NULL;
    SubArray<1, SIZE, DeviceType> *pZ_index = NULL;
    SubArray<1, SIZE, DeviceType> *nZ_index = NULL;
    SubArray<1, SIZE, DeviceType> *neighbor = NULL;
    SubArray<1, SIZE, DeviceType> *cell_ids = NULL;
    SubArray<1, SIZE, DeviceType> *level_index = NULL;

    Mem::Malloc1D(start_value, cse.shape[2] * cse.shape[1], queue_idx);
    Mem::Malloc1D(end_value, cse.shape[2] * cse.shape[1], queue_idx);
    Mem::Malloc1D(index, cse.shape[2] * cse.shape[1], queue_idx);
    Mem::Malloc1D(role, cse.shape[2] * cse.shape[1], queue_idx);
    Mem::Malloc1D(pY_index, cse.shape[2] * cse.shape[1], queue_idx);
    Mem::Malloc1D(nY_index, cse.shape[2] * cse.shape[1], queue_idx);
    Mem::Malloc1D(pZ_index, cse.shape[2] * cse.shape[1], queue_idx);
    Mem::Malloc1D(nZ_index, cse.shape[2] * cse.shape[1], queue_idx);
    Mem::Malloc1D(neighbor, cse.shape[2] * cse.shape[1], queue_idx);
    Mem::Malloc1D(cell_ids, cse.shape[2] * cse.shape[1], queue_idx);
    Mem::Malloc1D(level_index, D, queue_idx);

    printf("cse.shape[2]*cse.shape[1]: %u\n", cse.shape[2] * cse.shape[1]);

    Mem::Copy1D(start_value, cse.start_value, cse.shape[2] * cse.shape[1],
                queue_idx);
    Mem::Copy1D(end_value, cse.end_value, cse.shape[2] * cse.shape[1],
                queue_idx);
    Mem::Copy1D(index, cse.index, cse.shape[2] * cse.shape[1], queue_idx);
    Mem::Copy1D(role, cse.role, cse.shape[2] * cse.shape[1], queue_idx);
    Mem::Copy1D(pY_index, cse.pY_index, cse.shape[2] * cse.shape[1], queue_idx);
    Mem::Copy1D(nY_index, cse.nY_index, cse.shape[2] * cse.shape[1], queue_idx);
    Mem::Copy1D(pZ_index, cse.pZ_index, cse.shape[2] * cse.shape[1], queue_idx);
    Mem::Copy1D(nZ_index, cse.nZ_index, cse.shape[2] * cse.shape[1], queue_idx);
    Mem::Copy1D(neighbor, cse.neighbor, cse.shape[2] * cse.shape[1], queue_idx);
    Mem::Copy1D(cell_ids, cse.cell_ids, cse.shape[2] * cse.shape[1], queue_idx);
    Mem::Copy1D(level_index, cse.level_index, D, queue_idx);

    PrintSubarray("cse.role[2][1]", cse.role[2 * cse.shape[1] + 1]);
    PrintSubarray("cse.cell_ids[2][1]", cse.cell_ids[2 * cse.shape[1] + 1]);
    PrintSubarray("cse.neighbor[2][1]", cse.neighbor[2 * cse.shape[1] + 1]);
    PrintSubarray("cse.pY_index[2][1]", cse.pY_index[2 * cse.shape[1] + 1]);
    PrintSubarray("cse.role[2][2]", cse.role[2 * cse.shape[1] + 2]);
    PrintSubarray("cse.cell_ids[2][2]", cse.cell_ids[2 * cse.shape[1] + 2]);
    // PrintSubarray("cse.role[1][6]", cse.role[1*cse.shape[1]+6]);
    // PrintSubarray("cse.cell_ids[1][6]", cse.cell_ids[1*cse.shape[1]+6]);
    // PrintSubarray("cse.role[1][7]", cse.role[1*cse.shape[1]+7]);
    // PrintSubarray("cse.cell_ids[1][7]", cse.cell_ids[1*cse.shape[1]+7]);

    Array<1, SIZE, DeviceType> *edge_cases_array =
        new Array<1, SIZE, DeviceType>[cse.shape[2] * cse.shape[1]];
    SubArray<1, SIZE, DeviceType> *edge_cases_h =
        new SubArray<1, SIZE, DeviceType>[cse.shape[2] * cse.shape[1]];
    ;
    SubArray<1, SIZE, DeviceType> *edge_cases = NULL;
    Mem::Malloc1D(edge_cases, cse.shape[2] * cse.shape[1], queue_idx);
    for (SIZE i = 0; i < cse.shape[2] * cse.shape[1]; i++) {
      edge_cases_array[i] =
          Array<1, SIZE, DeviceType>({cse.role[i].getShape()[0]});
      edge_cases_array[i].memset(0);
      edge_cases_h[i] = SubArray(edge_cases_array[i]);
    }
    Mem::Copy1D(edge_cases, edge_cases_h, cse.shape[2] * cse.shape[1],
                queue_idx);

    Array<1, SIZE, DeviceType> *edgeX_ids_array =
        new Array<1, SIZE, DeviceType>[cse.shape[2] * (cse.shape[1] + 1)];
    Array<1, SIZE, DeviceType> *edgeY_ids_array =
        new Array<1, SIZE, DeviceType>[cse.shape[2] * (cse.shape[1] + 1)];
    Array<1, SIZE, DeviceType> *edgeZ_ids_array =
        new Array<1, SIZE, DeviceType>[cse.shape[2] * (cse.shape[1] + 1)];
    SubArray<1, SIZE, DeviceType> *edgeX_ids_h =
        new SubArray<1, SIZE, DeviceType>[cse.shape[2] * (cse.shape[1] + 1)];
    SubArray<1, SIZE, DeviceType> *edgeY_ids_h =
        new SubArray<1, SIZE, DeviceType>[cse.shape[2] * (cse.shape[1] + 1)];
    SubArray<1, SIZE, DeviceType> *edgeZ_ids_h =
        new SubArray<1, SIZE, DeviceType>[cse.shape[2] * (cse.shape[1] + 1)];
    SubArray<1, SIZE, DeviceType> *edgeX_ids = NULL;
    SubArray<1, SIZE, DeviceType> *edgeY_ids = NULL;
    SubArray<1, SIZE, DeviceType> *edgeZ_ids = NULL;
    Mem::Malloc1D(edgeX_ids, cse.shape[2] * (cse.shape[1] + 1), queue_idx);
    Mem::Malloc1D(edgeY_ids, cse.shape[2] * (cse.shape[1] + 1), queue_idx);
    Mem::Malloc1D(edgeZ_ids, cse.shape[2] * (cse.shape[1] + 1), queue_idx);
    for (SIZE i = 0; i < cse.shape[2]; i++) {
      for (SIZE j = 0; j < cse.shape[1] + 1; j++) {
        SIZE ij_vertex = i * (cse.shape[1] + 1) + j;
        // initialize last row since this is for per vertex
        SIZE ij_edge = i * (cse.shape[1]) + j;
        SIZE ij_edge_last_row = i * (cse.shape[1]) + j - 1;
        SIZE size = 0, size_last_row = 0;
        if (j < cse.shape[1]) {
          size = cse.role[ij_edge].getShape()[0];
        }
        if (j > 0) {
          size_last_row = cse.role[ij_edge_last_row].getShape()[0];
        }
        if (j == cse.shape[1] || size == 0 && size_last_row != 0) {
          size = size_last_row;
        }
        // printf("i %u j %u: size: %u\n", i, j, size);
        // std::cout << write_ij << " "  << read_ij << " Array size: " <<
        // cse.role[read_ij].getShape()[0] << "\n";
        edgeX_ids_array[ij_vertex] = Array<1, SIZE, DeviceType>({size});
        edgeY_ids_array[ij_vertex] = Array<1, SIZE, DeviceType>({size});
        edgeZ_ids_array[ij_vertex] = Array<1, SIZE, DeviceType>({size});
        edgeX_ids_array[ij_vertex].memset(9999999);
        edgeY_ids_array[ij_vertex].memset(9999999);
        edgeZ_ids_array[ij_vertex].memset(9999999);
        edgeX_ids_h[ij_vertex] = SubArray(edgeX_ids_array[ij_vertex]);
        edgeY_ids_h[ij_vertex] = SubArray(edgeY_ids_array[ij_vertex]);
        edgeZ_ids_h[ij_vertex] = SubArray(edgeZ_ids_array[ij_vertex]);
      }
    }
    Mem::Copy1D(edgeX_ids, edgeX_ids_h, cse.shape[2] * (cse.shape[1] + 1),
                queue_idx);
    Mem::Copy1D(edgeY_ids, edgeY_ids_h, cse.shape[2] * (cse.shape[1] + 1),
                queue_idx);
    Mem::Copy1D(edgeZ_ids, edgeZ_ids_h, cse.shape[2] * (cse.shape[1] + 1),
                queue_idx);

    std::cout << "Done allocation\n";

    // PrintSubarray("cell_ids", cse.cell_ids[0]);
    // PrintSubarray("neighbor", cse.neighbor[0]);
    // for (int i = 0; i < 25; i++) printf("y_neighbor[%d].data = %llu\n", i,
    // cse.y_neighbor[i].data());

    // PrintSubarray("cse.level_index[0]", cse.level_index[0]);
    // PrintSubarray("cse.level_index[1]", cse.level_index[1]);
    // PrintSubarray("cse.level_index[2]", cse.level_index[2]);

    const bool pitched = false;
    Array<1, SIZE, DeviceType> cell_cases_array({cse.cell_count}, pitched);
    Array<1, SIZE, DeviceType> tri_count_array({cse.cell_count}, pitched);
    Array<1, SIZE, DeviceType> point_count_array({cse.cell_count}, pitched);
    Array<1, SIZE, DeviceType> tri_count_scan_array({cse.cell_count + 1},
                                                    pitched);
    Array<1, SIZE, DeviceType> point_count_scan_array({cse.cell_count + 1},
                                                      pitched);
    // Array<1, SIZE, DeviceType> edge_cases_array( {cse.cell_count});
    Array<1, SIZE, DeviceType> num_point_array({1}, pitched);

    SubArray cell_cases(cell_cases_array);
    SubArray tri_count(tri_count_array);
    SubArray point_count(point_count_array);
    SubArray tri_count_scan(tri_count_scan_array);
    SubArray point_count_scan(point_count_scan_array);
    // SubArray edge_cases(edge_cases_array);
    SubArray num_point(num_point_array);

    using FunctorType1 = SFE_Pass1Functor<T, DeviceType>;
    using TaskType1 = Task<FunctorType1>;

    // printf("cse.cell_count: %u\n", cse.cell_count);
    // printf("edge_cases_array: %llu\n", edge_cases_array.data());
    // TaskType1 task1 = GenTask1<4, 8, 8>(cse.shape[2], cse.shape[1],
    // cse.shape[0], start_value, end_value, index, role, y_neighbor,
    // z_neighbor, neighbor,
    //                                     cell_ids, level_index, iso_value,
    //                                     edge_cases, tri_count, point_count,
    //                                     queue_idx);

    std::cout << "Pass1 start\n";
    TaskType1 task1 =
        GenTask1<4, 8, 8>(cse.shape[2], cse.shape[1], cse.shape[0], start_value,
                          end_value, iso_value, edge_cases, queue_idx);
    DeviceAdapter<TaskType1, DeviceType>().Execute(task1);
    DeviceRuntime<DeviceType>::SyncDevice();
    std::cout << "Pass1 done\n";

    // for (SIZE i = 0; i < cse.shape[2] * cse.shape[1]; i++) {
    //   std::cout << "edge_cases " << i << "\n";
    //   PrintSubarray("edge_cases_h", edge_cases_h[i]);
    // }

    using FunctorType2 = SFE_Pass2Functor<T, DeviceType>;
    using TaskType2 = Task<FunctorType2>;

    std::cout << "Pass2 start\n";
    TaskType2 task2 = GenTask2<4, 8, 8>(
        cse.shape[2], cse.shape[1], cse.shape[0], role, pZ_index, neighbor,
        edge_cases, cell_ids, cell_cases, point_count, tri_count, queue_idx);
    DeviceAdapter<TaskType2, DeviceType>().Execute(task2);
    DeviceRuntime<DeviceType>::SyncDevice();
    std::cout << "Pass2 done\n";

    // PrintSubarray("cell_cases", cell_cases);
    // PrintSubarray("point_count", point_count);
    // PrintSubarray("tri_count", tri_count);

    SubArray<1, SIZE, DeviceType> tri_count_liearized = tri_count.Linearize();
    SubArray<1, SIZE, DeviceType> point_count_liearized =
        point_count.Linearize();

    DeviceCollective<DeviceType>::ScanSumExtended(
        cse.cell_count, tri_count_liearized, tri_count_scan, queue_idx);
    DeviceCollective<DeviceType>::ScanSumExtended(
        cse.cell_count, point_count_liearized, point_count_scan, queue_idx);

    SIZE numTris = 0;
    MemoryManager<DeviceType>().Copy1D(&numTris, tri_count_scan(cse.cell_count),
                                       1, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    SIZE numPoints = 0;
    MemoryManager<DeviceType>().Copy1D(
        &numPoints, point_count_scan(cse.cell_count), 1, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    // PrintSubarray("point_count_scan", point_count_scan);
    // PrintSubarray("tri_count_scan", tri_count_scan);

    std::cout << "numPoints: " << numPoints << "\n";
    std::cout << "numTris: " << numTris << "\n";

    using FunctorType3 = SFE_Pass3Functor<T, DeviceType>;
    using TaskType3 = Task<FunctorType3>;

    std::cout << "Pass3 start\n";
    TaskType3 task3 = GenTask3<4, 8, 8>(
        cse.shape[2], cse.shape[1], cse.shape[0], role, pY_index, pZ_index,
        neighbor, cell_ids, cell_cases, point_count_scan, tri_count_scan,
        edgeX_ids, edgeY_ids, edgeZ_ids, queue_idx);
    DeviceAdapter<TaskType3, DeviceType>().Execute(task3);
    DeviceRuntime<DeviceType>::SyncDevice();
    std::cout << "Pass3 done\n";

    for (SIZE i = 0; i < cse.shape[2]; i++) {
      for (SIZE j = 0; j < cse.shape[1] + 1; j++) {
        SIZE ij_vertex = i * (cse.shape[1] + 1) + j;

        std::cout << "edge_ids " << i << ", " << j << "\n";
        PrintSubarray("edgeX_ids_h", edgeX_ids_h[ij_vertex]);
        // PrintSubarray("edgeY_ids_h", edgeY_ids_h[ij_vertex]);
        // PrintSubarray("edgeZ_ids_h", edgeZ_ids_h[ij_vertex]);
      }
    }

    Triangles = Array<1, SIZE, DeviceType>({numTris * 3});
    Points = Array<1, T, DeviceType>({numPoints * 3});
    SubArray triangles(Triangles);
    SubArray points(Points);

    // printf("level_index: %llu, %llu, %llu\n",
    //         cse.level_index[0].data(), cse.level_index[1].data(),
    //         cse.level_index[2].data());

    using FunctorType4 = SFE_Pass4Functor<T, DeviceType>;
    using TaskType4 = Task<FunctorType4>;

    std::cout << "Pass4 start\n";
    TaskType4 task4 = GenTask4<4, 8, 8>(
        cse.shape[2], cse.shape[1], cse.shape[0], start_value, end_value, index,
        role, pY_index, pZ_index, neighbor, cell_ids, level_index, iso_value,
        cell_cases, edgeX_ids, edgeY_ids, edgeZ_ids, tri_count_scan, points,
        triangles, queue_idx);
    DeviceAdapter<TaskType4, DeviceType>().Execute(task4);
    DeviceRuntime<DeviceType>::SyncDevice();
    std::cout << "Pass4 done\n";

    PrintSubarray("points", points);
    PrintSubarray("triangles", triangles);

    DeviceRuntime<DeviceType>::SyncDevice();
  }
};

} // namespace mgard_x

#endif