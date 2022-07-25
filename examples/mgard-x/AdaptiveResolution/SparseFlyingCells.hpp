/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_SPARSE_FLYING_CELLS_HPP
#define MGARD_X_SPARSE_FLYING_CELLS_HPP

#include "mgard/mgard-x/RuntimeX/RuntimeX.h"

// avoid copying device functions
#include "FlyingEdges.hpp"

namespace mgard_x {


template <typename T, typename DeviceType>
MGARDX_EXEC void InterpolateEdge(SIZE edgeNum, 
                                 SIZE index_x, SIZE index_y, SIZE index_z,
                                 SIZE size_x, SIZE size_y, SIZE size_z,
                                 SIZE const *edgeUses, SIZE *edgeIds,
                                 T iso_value, T * v,
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

  SIZE z0 = index_z + offsets0[2] * size_z;
  SIZE y0 = index_y + offsets0[1] * size_y;
  SIZE x0 = index_x + offsets0[0] * size_x;

  SIZE z1 = index_z + offsets1[2] * size_z;
  SIZE y1 = index_y + offsets1[1] * size_y;
  SIZE x1 = index_x + offsets1[0] * size_x;

  T s0 = v[verts[0]];
  T s1 = v[verts[1]];

  // printf("s: %f %f\n", s0, s1);

  T w = (iso_value - s0) / (s1 - s0);

  *points(writeIndex) =     (1 - w) * x0 + w * x1;
  *points(writeIndex + 1) = (1 - w) * y0 + w * y1;
  *points(writeIndex + 2) = (1 - w) * z0 + w * z1;
}

template <typename T, typename DeviceType>
class SFC_Pass1Functor : public Functor<DeviceType> {
public:
  MGARDX_CONT SFC_Pass1Functor(CompressedSparseCell<T, DeviceType> csc,
                               T iso_value, 
                               SubArray<1, SIZE, DeviceType> tri_count,
                               SubArray<1, SIZE, DeviceType> point_count)
      : csc(csc), iso_value(iso_value), tri_count(tri_count), point_count(point_count) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE idx = FunctorBase<DeviceType>::GetBlockIdX() *
                 FunctorBase<DeviceType>::GetBlockDimX() +
             FunctorBase<DeviceType>::GetThreadIdX();
    
 
    if (idx >= csc.num_cell) {
      return;
    }

    T v[8];
    for (int i = 0; i < 8; i++) {
      v[i] = *csc.value[i](idx);
    }

    SIZE edge_case_0 = MGARD_Below;
    SIZE edge_case_1 = MGARD_Below;
    SIZE edge_case_2 = MGARD_Below;
    SIZE edge_case_3 = MGARD_Below;

    if (v[0] >= iso_value) edge_case_0 = MGARD_LeftAbove;
    if (v[1] >= iso_value) edge_case_0 |= MGARD_RightAbove;

    if (v[2] >= iso_value) edge_case_1 = MGARD_LeftAbove;
    if (v[3] >= iso_value) edge_case_1 |= MGARD_RightAbove;

    if (v[4] >= iso_value) edge_case_2 = MGARD_LeftAbove;
    if (v[5] >= iso_value) edge_case_2 |= MGARD_RightAbove;

    if (v[6] >= iso_value) edge_case_3 = MGARD_LeftAbove;
    if (v[7] >= iso_value) edge_case_3 |= MGARD_RightAbove;

    SIZE local_cell_case = (edge_case_0 | (edge_case_1 << 2) | (edge_case_2 << 4) | (edge_case_3 << 6));
    SIZE local_tri_count = flying_edges::GetNumberOfPrimitives(local_cell_case);
    if (local_tri_count == 0) {
      // printf("idx: %u v: %f %f %f %f %f %f %f %f\n", idx, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
    }
    SIZE const *edgeUses = flying_edges::GetEdgeUses(local_cell_case);
    SIZE local_point_count = 0;
    for (int i = 0; i < 12; i++) {
      local_point_count += edgeUses[i];
    }
    *tri_count(idx) = local_tri_count;
    *point_count(idx) = local_point_count;
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    return size;
  }

private:
  CompressedSparseCell<T, DeviceType> csc;
  T iso_value;
  SubArray<1, SIZE, DeviceType> tri_count;
  SubArray<1, SIZE, DeviceType> point_count;
};

template <typename T, typename DeviceType>
class SFC_Pass2Functor : public Functor<DeviceType> {
public:
  MGARDX_CONT SFC_Pass2Functor(CompressedSparseCell<T, DeviceType> csc,
                               T iso_value, 
                               SubArray<1, SIZE, DeviceType> tri_count_scan,
                               SubArray<1, SIZE, DeviceType> point_count_scan,
                               SubArray<1, T, DeviceType> points,
                               SubArray<1, SIZE, DeviceType> triangles)
      : csc(csc), iso_value(iso_value), tri_count_scan(tri_count_scan), point_count_scan(point_count_scan),
        points(points), triangles(triangles) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE idx = FunctorBase<DeviceType>::GetBlockIdX() *
                 FunctorBase<DeviceType>::GetBlockDimX() +
             FunctorBase<DeviceType>::GetThreadIdX();
    
 
    if (idx >= csc.num_cell) {
      return;
    }

    T v[8];
    for (int i = 0; i < 8; i++) {
      v[i] = *csc.value[i](idx);
    }

    SIZE edge_case_0 = MGARD_Below;
    SIZE edge_case_1 = MGARD_Below;
    SIZE edge_case_2 = MGARD_Below;
    SIZE edge_case_3 = MGARD_Below;

    if (v[0] >= iso_value) edge_case_0 = MGARD_LeftAbove;
    if (v[1] >= iso_value) edge_case_0 |= MGARD_RightAbove;

    if (v[2] >= iso_value) edge_case_1 = MGARD_LeftAbove;
    if (v[3] >= iso_value) edge_case_1 |= MGARD_RightAbove;

    if (v[4] >= iso_value) edge_case_2 = MGARD_LeftAbove;
    if (v[5] >= iso_value) edge_case_2 |= MGARD_RightAbove;

    if (v[6] >= iso_value) edge_case_3 = MGARD_LeftAbove;
    if (v[7] >= iso_value) edge_case_3 |= MGARD_RightAbove;

    SIZE local_cell_case = (edge_case_0 | (edge_case_1 << 2) | (edge_case_2 << 4) | (edge_case_3 << 6));
    SIZE local_tri_count = flying_edges::GetNumberOfPrimitives(local_cell_case);
    SIZE const *edgeUses = flying_edges::GetEdgeUses(local_cell_case);

    SIZE prev_tri_count = *tri_count_scan(idx);
    SIZE prev_point_count = *point_count_scan(idx);

    SIZE edgeIds[12];
    for (int i = 0; i < 12; i++) {
      edgeIds[i] = prev_point_count;
      prev_point_count += edgeUses[i];
    }
    

    SIZE index_x = *csc.index[0](idx);
    SIZE index_y = *csc.index[1](idx);
    SIZE index_z = *csc.index[2](idx);
    SIZE size_x = *csc.size[0](idx);
    SIZE size_y = *csc.size[1](idx);
    SIZE size_z = *csc.size[2](idx);

    for (int i = 0; i < 12; i++) {
      InterpolateEdge(i, index_x, index_y, index_z, size_x, size_y, size_z,
                      edgeUses, edgeIds, iso_value, v, points);
    }

    generate_tris(local_cell_case, local_tri_count, edgeIds, prev_tri_count, triangles);
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    return size;
  }

private:
  CompressedSparseCell<T, DeviceType> csc;
  T iso_value;
  SubArray<1, SIZE, DeviceType> tri_count_scan;
  SubArray<1, SIZE, DeviceType> point_count_scan;
  SubArray<1, T, DeviceType> points;
  SubArray<1, SIZE, DeviceType> triangles;
};



template <DIM D, typename T, typename DeviceType>
class SparseFlyingCells : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  SparseFlyingCells() : AutoTuner<DeviceType>() {}

  template <SIZE F>
  MGARDX_CONT Task<SFC_Pass1Functor<T, DeviceType>>
  GenTask1(CompressedSparseCell<T, DeviceType> csc,
           T iso_value, 
           SubArray<1, SIZE, DeviceType> tri_count,
           SubArray<1, SIZE, DeviceType> point_count,
           int queue_idx) {
    using FunctorType = SFC_Pass1Functor<T, DeviceType>;
    FunctorType functor(csc, iso_value, tri_count, point_count);

    SIZE total_thread_z = 1;
    SIZE total_thread_y = 1;
    SIZE total_thread_x = csc.num_cell;

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = F;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size,
                queue_idx);
  }

  template <SIZE F>
  MGARDX_CONT Task<SFC_Pass2Functor<T, DeviceType>>
  GenTask2(CompressedSparseCell<T, DeviceType> csc,
           T iso_value, 
           SubArray<1, SIZE, DeviceType> tri_count_scan,
           SubArray<1, SIZE, DeviceType> point_count_scan,
           SubArray<1, T, DeviceType> points,
           SubArray<1, SIZE, DeviceType> triangles,
           int queue_idx) {
    using FunctorType = SFC_Pass2Functor<T, DeviceType>;
    FunctorType functor(csc, iso_value, tri_count_scan, point_count_scan,
                        points, triangles);

    SIZE total_thread_z = 1;
    SIZE total_thread_y = 1;
    SIZE total_thread_x = csc.num_cell;

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = F;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size,
                queue_idx);
  }


  MGARDX_CONT
  void Execute(CompressedSparseCell<T, DeviceType> &csc,
               T iso_value, Array<1, SIZE, DeviceType> &Triangles,
               Array<1, T, DeviceType> &Points, int queue_idx) {
    using Mem = MemoryManager<DeviceType>;
    Timer t;

    const bool pitched = false;
    Array<1, SIZE, DeviceType> tri_count_array({csc.num_cell}, pitched);
    Array<1, SIZE, DeviceType> point_count_array({csc.num_cell}, pitched);
    Array<1, SIZE, DeviceType> tri_count_scan_array( {csc.num_cell + 1}, pitched);
    Array<1, SIZE, DeviceType> point_count_scan_array( {csc.num_cell + 1}, pitched);
    SubArray tri_count(tri_count_array);
    SubArray point_count(point_count_array);
    SubArray tri_count_scan(tri_count_scan_array);
    SubArray point_count_scan(point_count_scan_array);

    std::cout << "Pass1 start. Num cell: " << csc.num_cell << "\n";
    using FunctorType1 = SFC_Pass1Functor<T, DeviceType>;
    using TaskType1 = Task<FunctorType1>;
    TaskType1 task1 = GenTask1<128>(csc, iso_value, tri_count, point_count, queue_idx);

    t.start();
    DeviceAdapter<TaskType1, DeviceType>().Execute(task1);
    DeviceRuntime<DeviceType>::SyncDevice();
    t.end();
    t.print("Pass 1");
    t.clear();

    SubArray<1, SIZE, DeviceType> tri_count_liearized = tri_count.Linearize();
    SubArray<1, SIZE, DeviceType> point_count_liearized = point_count.Linearize();

    DeviceCollective<DeviceType>::ScanSumExtended(csc.num_cell, tri_count_liearized, tri_count_scan, queue_idx);
    DeviceCollective<DeviceType>::ScanSumExtended(csc.num_cell, point_count_liearized, point_count_scan, queue_idx);

    SIZE numTris = 0;
    MemoryManager<DeviceType>().Copy1D(&numTris, tri_count_scan(csc.num_cell), 1, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    SIZE numPoints = 0;
    MemoryManager<DeviceType>().Copy1D(&numPoints, point_count_scan(csc.num_cell), 1, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    // PrintSubarray("point_count_scan", point_count_scan);
    // PrintSubarray("tri_count_scan", tri_count_scan);

    
    std::cout << "numPoints: " << numPoints << "\n";
    std::cout << "numTris: " << numTris << "\n";


    Triangles = Array<1, SIZE, DeviceType>({numTris * 3});
    Points = Array<1, T, DeviceType>({numPoints * 3});
    SubArray triangles(Triangles);
    SubArray points(Points);

    std::cout << "Pass2 start Num cell: " << csc.num_cell << "\n";
    using FunctorType2 = SFC_Pass2Functor<T, DeviceType>;
    using TaskType2 = Task<FunctorType2>;
    TaskType2 task2 = GenTask2<128>(csc, iso_value, tri_count_scan, point_count_scan, points, triangles, queue_idx);
    t.start();
    DeviceAdapter<TaskType2, DeviceType>().Execute(task2);
    DeviceRuntime<DeviceType>::SyncDevice();
    t.end();
    t.print("Pass 2");
    t.clear();

    DeviceRuntime<DeviceType>::SyncDevice();
  }



};

} // namespace mgard_x

#endif