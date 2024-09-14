/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_FLYING_EDGES_BATCHED_HPP
#define MGARD_X_FLYING_EDGES_BATCHED_HPP

#include "FlyingEdges.hpp"
#include "mgard/mgard-x/RuntimeX/RuntimeX.h"

namespace mgard_x {

namespace flying_edges_batched {

template <typename T, typename DeviceType>
class Pass1Functor : public Functor<DeviceType> {
public:
  MGARDX_CONT Pass1Functor() {}
  MGARDX_CONT Pass1Functor(
      SIZE nr, SIZE nc, SIZE nf,
      SubArray<1, SubArray<3, T, DeviceType>, DeviceType> v_batch, T iso_value,
      SubArray<1, SubArray<3, SIZE, DeviceType>, DeviceType> axis_sum_batch,
      SubArray<1, SubArray<2, SIZE, DeviceType>, DeviceType> axis_min_batch,
      SubArray<1, SubArray<2, SIZE, DeviceType>, DeviceType> axis_max_batch,
      SubArray<1, SubArray<3, SIZE, DeviceType>, DeviceType> edges_batch)
      : nr(nr), nc(nc), nf(nf), v_batch(v_batch), iso_value(iso_value),
        axis_sum_batch(axis_sum_batch), axis_min_batch(axis_min_batch),
        axis_max_batch(axis_max_batch), edges_batch(edges_batch) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE f = FunctorBase<DeviceType>::GetBlockIdX() *
                 FunctorBase<DeviceType>::GetBlockDimX() +
             FunctorBase<DeviceType>::GetThreadIdX();
    SIZE r = FunctorBase<DeviceType>::GetBlockIdY() *
                 FunctorBase<DeviceType>::GetBlockDimY() +
             FunctorBase<DeviceType>::GetThreadIdY();

    SIZE batch_id = FunctorBase<DeviceType>::GetBlockIdZ();
    v = *v_batch(batch_id);
    axis_sum = *axis_sum_batch(batch_id);
    axis_min = *axis_min_batch(batch_id);
    axis_max = *axis_max_batch(batch_id);
    edges = *edges_batch(batch_id);

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
  T iso_value;
  SubArray<3, T, DeviceType> v;
  SubArray<3, SIZE, DeviceType> axis_sum;
  SubArray<2, SIZE, DeviceType> axis_min;
  SubArray<2, SIZE, DeviceType> axis_max;
  SubArray<3, SIZE, DeviceType> edges;

  SubArray<1, SubArray<3, T, DeviceType>, DeviceType> v_batch;
  SubArray<1, SubArray<3, SIZE, DeviceType>, DeviceType> axis_sum_batch;
  SubArray<1, SubArray<2, SIZE, DeviceType>, DeviceType> axis_min_batch;
  SubArray<1, SubArray<2, SIZE, DeviceType>, DeviceType> axis_max_batch;
  SubArray<1, SubArray<3, SIZE, DeviceType>, DeviceType> edges_batch;
};

template <typename T, typename DeviceType>
class Pass2Functor : public Functor<DeviceType> {
public:
  MGARDX_CONT Pass2Functor() {}
  MGARDX_CONT Pass2Functor(
      SIZE nr, SIZE nc, SIZE nf,
      SubArray<1, SubArray<3, SIZE, DeviceType>, DeviceType> axis_sum_batch,
      SubArray<1, SubArray<2, SIZE, DeviceType>, DeviceType> axis_min_batch,
      SubArray<1, SubArray<2, SIZE, DeviceType>, DeviceType> axis_max_batch,
      SubArray<1, SubArray<3, SIZE, DeviceType>, DeviceType> edges_batch,
      SubArray<1, SubArray<2, SIZE, DeviceType>, DeviceType>
          cell_tri_count_batch)
      : nr(nr), nc(nc), nf(nf), axis_sum_batch(axis_sum_batch),
        axis_min_batch(axis_min_batch), axis_max_batch(axis_max_batch),
        edges_batch(edges_batch), cell_tri_count_batch(cell_tri_count_batch) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE f = FunctorBase<DeviceType>::GetBlockIdX() *
                 FunctorBase<DeviceType>::GetBlockDimX() +
             FunctorBase<DeviceType>::GetThreadIdX();
    SIZE r = FunctorBase<DeviceType>::GetBlockIdY() *
                 FunctorBase<DeviceType>::GetBlockDimY() +
             FunctorBase<DeviceType>::GetThreadIdY();

    SIZE batch_id = FunctorBase<DeviceType>::GetBlockIdZ();
    axis_sum = *axis_sum_batch(batch_id);
    axis_min = *axis_min_batch(batch_id);
    axis_max = *axis_max_batch(batch_id);
    edges = *edges_batch(batch_id);
    cell_tri_count = *cell_tri_count_batch(batch_id);

    if (f >= nf - 1 || r >= nr - 1) {
      return;
    }

    // compute trim blounds
    // Max right should be nc - 1
    SIZE left, right;
    bool hasWork = flying_edges::computeTrimBounds(
        r, f, nc - 1, edges, axis_min, axis_max, left, right);

    // printf("computeTrimBounds: rf: %u %u, left: %u, right: %u\n",
    // r, f, left, right);
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
      SIZE edgeCase = flying_edges::getEdgeCase(r, c, f, edges);
      // printf("cell case: %u\n", edgeCase);
      SIZE numTris = flying_edges::GetNumberOfPrimitives(edgeCase);
      if (numTris > 0) {
        _cell_tri_count += numTris;
        SIZE const *edgeUses = flying_edges::GetEdgeUses(edgeCase);

        onBoundary[1] = c >= nc - 2;

        _axis_sum[0] += edgeUses[4];
        _axis_sum[2] += edgeUses[8];

        flying_edges::CountBoundaryEdgeUses(onBoundary, edgeUses, _axis_sum,
                                            adj_row_sum, adj_col_sum);
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
  SubArray<1, SubArray<3, SIZE, DeviceType>, DeviceType> axis_sum_batch;
  SubArray<1, SubArray<2, SIZE, DeviceType>, DeviceType> axis_min_batch;
  SubArray<1, SubArray<2, SIZE, DeviceType>, DeviceType> axis_max_batch;
  SubArray<1, SubArray<3, SIZE, DeviceType>, DeviceType> edges_batch;
  SubArray<1, SubArray<2, SIZE, DeviceType>, DeviceType> cell_tri_count_batch;

  SubArray<3, SIZE, DeviceType> axis_sum;
  SubArray<2, SIZE, DeviceType> axis_min;
  SubArray<2, SIZE, DeviceType> axis_max;
  SubArray<3, SIZE, DeviceType> edges;
  SubArray<2, SIZE, DeviceType> cell_tri_count;
};

template <typename T, typename DeviceType>
class Pass4Functor : public Functor<DeviceType> {
public:
  MGARDX_CONT Pass4Functor() {}
  MGARDX_CONT Pass4Functor(
      SIZE nr, SIZE nc, SIZE nf,
      SubArray<1, SubArray<3, T, DeviceType>, DeviceType> v_batch, T iso_value,
      SubArray<1, SubArray<1, SIZE, DeviceType>, DeviceType>
          axis_sum_scan_batch,
      SubArray<1, SIZE, DeviceType> axis_sum_scan_offset_batch,
      SubArray<1, SubArray<2, SIZE, DeviceType>, DeviceType> axis_min_batch,
      SubArray<1, SubArray<2, SIZE, DeviceType>, DeviceType> axis_max_batch,
      SubArray<1, SubArray<1, SIZE, DeviceType>, DeviceType>
          cell_tri_count_scan_batch,
      SubArray<1, SIZE, DeviceType> cell_tri_count_scan_offset_batch,
      SubArray<1, SubArray<3, SIZE, DeviceType>, DeviceType> edges_batch,
      SubArray<1, SubArray<1, SIZE, DeviceType>, DeviceType>
          triangle_topology_batch,
      SubArray<1, SubArray<1, T, DeviceType>, DeviceType> points_batch)
      : nr(nr), nc(nc), nf(nf), v_batch(v_batch), iso_value(iso_value),
        axis_sum_scan_batch(axis_sum_scan_batch),
        axis_sum_scan_offset_batch(axis_sum_scan_offset_batch),
        axis_min_batch(axis_min_batch), axis_max_batch(axis_max_batch),
        cell_tri_count_scan_batch(cell_tri_count_scan_batch),
        cell_tri_count_scan_offset_batch(cell_tri_count_scan_offset_batch),
        edges_batch(edges_batch),
        triangle_topology_batch(triangle_topology_batch),
        points_batch(points_batch) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE f = FunctorBase<DeviceType>::GetBlockIdX() *
                 FunctorBase<DeviceType>::GetBlockDimX() +
             FunctorBase<DeviceType>::GetThreadIdX();
    SIZE r = FunctorBase<DeviceType>::GetBlockIdY() *
                 FunctorBase<DeviceType>::GetBlockDimY() +
             FunctorBase<DeviceType>::GetThreadIdY();

    SIZE batch_id = FunctorBase<DeviceType>::GetBlockIdZ();
    v = *v_batch(batch_id);
    axis_sum_scan = *axis_sum_scan_batch(batch_id);
    axis_sum_scan_curr_offset = *axis_sum_scan_offset_batch(batch_id);
    axis_sum_scan_prev_offset =
        batch_id == 0 ? 0 : *axis_sum_scan_offset_batch(batch_id - 1);
    axis_min = *axis_min_batch(batch_id);
    axis_max = *axis_max_batch(batch_id);
    cell_tri_count_scan = *cell_tri_count_scan_batch(batch_id);
    cell_tri_count_scan_curr_offset =
        *cell_tri_count_scan_offset_batch(batch_id);
    cell_tri_count_scan_prev_offset =
        batch_id == 0 ? 0 : *cell_tri_count_scan_offset_batch(batch_id - 1);
    edges = *edges_batch(batch_id);
    triangle_topology = *triangle_topology_batch(batch_id);
    points = *points_batch(batch_id);

    if (f >= nf - 1 || r >= nr - 1) {
      return;
    }

    // Check if current batch has work to do
    SIZE numPts = axis_sum_scan_curr_offset - axis_sum_scan_prev_offset;
    SIZE numTris =
        cell_tri_count_scan_curr_offset - cell_tri_count_scan_prev_offset;
    if (numPts == 0 || numTris == 0) {
      return;
    }

    // printf("offset: %u\n", r * (nf-1) + f);
    // Subtracting 'cell_tri_count_scan_prev_offset' to get the current offset
    SIZE cell_tri_offset = *cell_tri_count_scan(r * (nf - 1) + f) -
                           cell_tri_count_scan_prev_offset;
    SIZE next_tri_offset = *cell_tri_count_scan(r * (nf - 1) + f + 1) -
                           cell_tri_count_scan_prev_offset;

    flying_edges::Pass4TrimState state(r, f, nf, nc, nr, axis_min, axis_max,
                                       edges);
    if (!state.hasWork) {
      return;
    }

    SIZE edgeIds[12];
    SIZE edgeCase = flying_edges::getEdgeCase(r, state.left, f, edges);

    // Subtracting 'axis_sum_scan_prev_offset' to get the current offset
    flying_edges::init_voxelIds(nr, nf, r, f, edgeCase, axis_sum_scan, edgeIds,
                                axis_sum_scan_prev_offset);

    // run along the trimmed voxels
    // need state.right-1 since we are iterating through cells
    for (SIZE i = state.left; i < state.right - 1; ++i) {
      edgeCase = flying_edges::getEdgeCase(r, i, f, edges);
      SIZE numTris = flying_edges::GetNumberOfPrimitives(edgeCase);
      if (numTris > 0) {
        flying_edges::generate_tris(edgeCase, numTris, edgeIds, cell_tri_offset,
                                    triangle_topology);

        SIZE const *edgeUses = flying_edges::GetEdgeUses(edgeCase);
        if (!flying_edges::fully_interior(state.boundaryStatus) ||
            flying_edges::case_includes_axes(edgeUses)) {
          flying_edges::Generate(f, i, r, state.boundaryStatus, edgeUses,
                                 edgeIds, iso_value, v, points);
        }
        flying_edges::advance_voxelIds(edgeUses, edgeIds);
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
  T iso_value;
  SubArray<1, SubArray<3, T, DeviceType>, DeviceType> v_batch;
  SubArray<1, SubArray<1, SIZE, DeviceType>, DeviceType> axis_sum_scan_batch;
  SubArray<1, SIZE, DeviceType> axis_sum_scan_offset_batch;
  SubArray<1, SubArray<2, SIZE, DeviceType>, DeviceType> axis_min_batch;
  SubArray<1, SubArray<2, SIZE, DeviceType>, DeviceType> axis_max_batch;
  SubArray<1, SubArray<1, SIZE, DeviceType>, DeviceType>
      cell_tri_count_scan_batch;
  SubArray<1, SIZE, DeviceType> cell_tri_count_scan_offset_batch;
  SubArray<1, SubArray<3, SIZE, DeviceType>, DeviceType> edges_batch;
  SubArray<1, SubArray<1, SIZE, DeviceType>, DeviceType>
      triangle_topology_batch;
  SubArray<1, SubArray<1, T, DeviceType>, DeviceType> points_batch;

  SubArray<3, T, DeviceType> v;
  SubArray<1, SIZE, DeviceType> axis_sum_scan;
  SIZE axis_sum_scan_curr_offset;
  SIZE axis_sum_scan_prev_offset;
  SubArray<2, SIZE, DeviceType> axis_min;
  SubArray<2, SIZE, DeviceType> axis_max;
  SubArray<3, SIZE, DeviceType> edges;
  SubArray<1, SIZE, DeviceType> cell_tri_count_scan;
  SIZE cell_tri_count_scan_curr_offset;
  SIZE cell_tri_count_scan_prev_offset;
  SubArray<1, SIZE, DeviceType> triangle_topology;
  SubArray<1, T, DeviceType> points;
};

template <typename T, typename DeviceType>
class FlyingEdgesBatched : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  FlyingEdgesBatched() : AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<Pass1Functor<T, DeviceType>>
  GenTask1(SIZE nr, SIZE nc, SIZE nf,
           SubArray<1, SubArray<3, T, DeviceType>, DeviceType> v, T iso_value,
           SubArray<1, SubArray<3, SIZE, DeviceType>, DeviceType> axis_sum,
           SubArray<1, SubArray<2, SIZE, DeviceType>, DeviceType> axis_min,
           SubArray<1, SubArray<2, SIZE, DeviceType>, DeviceType> axis_max,
           SubArray<1, SubArray<3, SIZE, DeviceType>, DeviceType> edges,
           int queue_idx) {
    using FunctorType = Pass1Functor<T, DeviceType>;
    FunctorType functor(nr, nc, nf, v, iso_value, axis_sum, axis_min, axis_max,
                        edges);

    SIZE num_batches = v.shape(0);
    SIZE total_thread_z = num_batches;
    SIZE total_thread_y = nr;
    SIZE total_thread_x = nf;

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((double)total_thread_z / tbz);
    gridy = ceil((double)total_thread_y / tby);
    gridx = ceil((double)total_thread_x / tbx);

    // printf("%u %u %u\n", shape.dataHost()[2], shape.dataHost()[1],
    // shape.dataHost()[0]); PrintSubarray("shape", shape);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "FlyingEdges::Pass1");
  }

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<Pass2Functor<T, DeviceType>> GenTask2(
      SIZE nr, SIZE nc, SIZE nf,
      SubArray<1, SubArray<3, SIZE, DeviceType>, DeviceType> axis_sum,
      SubArray<1, SubArray<2, SIZE, DeviceType>, DeviceType> axis_min,
      SubArray<1, SubArray<2, SIZE, DeviceType>, DeviceType> axis_max,
      SubArray<1, SubArray<3, SIZE, DeviceType>, DeviceType> edges,
      SubArray<1, SubArray<2, SIZE, DeviceType>, DeviceType> cell_tri_count,
      int queue_idx) {
    using FunctorType = Pass2Functor<T, DeviceType>;
    FunctorType functor(nr, nc, nf, axis_sum, axis_min, axis_max, edges,
                        cell_tri_count);

    SIZE num_batches = axis_sum.shape(0);
    SIZE total_thread_z = num_batches;
    SIZE total_thread_y = nr - 1;
    SIZE total_thread_x = nf - 1;

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((double)total_thread_z / tbz);
    gridy = ceil((double)total_thread_y / tby);
    gridx = ceil((double)total_thread_x / tbx);

    // printf("%u %u %u\n", shape.dataHost()[2], shape.dataHost()[1],
    // shape.dataHost()[0]); PrintSubarray("shape", shape);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "FlyingEdges::Pass2");
  }

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<Pass4Functor<T, DeviceType>> GenTask4(
      SIZE nr, SIZE nc, SIZE nf,
      SubArray<1, SubArray<3, T, DeviceType>, DeviceType> v, T iso_value,
      SubArray<1, SubArray<1, SIZE, DeviceType>, DeviceType> axis_sum_scan,
      SubArray<1, SIZE, DeviceType> axis_sum_scan_offset,
      SubArray<1, SubArray<2, SIZE, DeviceType>, DeviceType> axis_min,
      SubArray<1, SubArray<2, SIZE, DeviceType>, DeviceType> axis_max,
      SubArray<1, SubArray<1, SIZE, DeviceType>, DeviceType>
          cell_tri_count_scan,
      SubArray<1, SIZE, DeviceType> cell_tri_count_scan_offset,
      SubArray<1, SubArray<3, SIZE, DeviceType>, DeviceType> edges,
      SubArray<1, SubArray<1, SIZE, DeviceType>, DeviceType> triangle_topology,
      SubArray<1, SubArray<1, T, DeviceType>, DeviceType> points,
      int queue_idx) {
    using FunctorType = Pass4Functor<T, DeviceType>;
    FunctorType functor(nr, nc, nf, v, iso_value, axis_sum_scan,
                        axis_sum_scan_offset, axis_min, axis_max,
                        cell_tri_count_scan, cell_tri_count_scan_offset, edges,
                        triangle_topology, points);

    SIZE num_batches = v.shape(0);
    SIZE total_thread_z = num_batches;
    SIZE total_thread_y = nr - 1;
    SIZE total_thread_x = nf - 1;

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((double)total_thread_z / tbz);
    gridy = ceil((double)total_thread_y / tby);
    gridx = ceil((double)total_thread_x / tbx);

    // printf("%u %u %u\n", shape.dataHost()[2], shape.dataHost()[1],
    // shape.dataHost()[0]); PrintSubarray("shape", shape);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "FlyingEdges::Pass4");
  }

  MGARDX_CONT
  void Execute(SIZE nr, SIZE nc, SIZE nf,
               SubArray<1, SubArray<3, T, DeviceType>, DeviceType> v,
               T iso_value,
               Array<1, Array<1, SIZE, DeviceType>, DeviceType> &Triangles,
               Array<1, Array<1, T, DeviceType>, DeviceType> &Points,
               double &pass1_time, double &pass2_time, double &pass3_time,
               double &pass4_time, int queue_idx) {

    Timer t1, t2, t3, t4;

    const bool pitched = false;
    const bool managed = true;

    SIZE num_batches = v.shape(0);

    Array<1, Array<3, SIZE, DeviceType>, DeviceType> axis_sum_array(
        {num_batches}, pitched, managed);
    Array<1, SubArray<3, SIZE, DeviceType>, DeviceType> axis_sum_subarray(
        {num_batches}, pitched, managed);
    for (SIZE i = 0; i < num_batches; i++) {
      SubArray subarray_of_array(axis_sum_array);
      SubArray subarray_of_subarray(axis_sum_subarray);
      // Initilize a new array and assign it to subarray_of_array.
      // We can access it from the host since axis_sum_array is managed memory
      *subarray_of_array(i) = Array<3, SIZE, DeviceType>({nr, nf, 3}, pitched);
      // Get its SubArray
      *subarray_of_subarray(i) = SubArray(*subarray_of_array(i));
      subarray_of_array(i)->memset(0);
    }

    Array<1, Array<1, SIZE, DeviceType>, DeviceType> axis_sum_scan_array(
        {num_batches}, pitched, managed);
    Array<1, SubArray<1, SIZE, DeviceType>, DeviceType> axis_sum_scan_subarray(
        {num_batches}, pitched, managed);
    for (SIZE i = 0; i < num_batches; i++) {
      SubArray subarray_of_array(axis_sum_scan_array);
      SubArray subarray_of_subarray(axis_sum_scan_subarray);
      *subarray_of_array(i) =
          Array<1, SIZE, DeviceType>({nr * nf * 3 + 1}, pitched);
      *subarray_of_subarray(i) = SubArray(*subarray_of_array(i));
      subarray_of_array(i)->memset(0);
    }

    Array<1, Array<2, SIZE, DeviceType>, DeviceType> axis_min_array(
        {num_batches}, pitched, managed);
    Array<1, SubArray<2, SIZE, DeviceType>, DeviceType> axis_min_subarray(
        {num_batches}, pitched, managed);
    for (SIZE i = 0; i < num_batches; i++) {
      SubArray subarray_of_array(axis_min_array);
      SubArray subarray_of_subarray(axis_min_subarray);
      *subarray_of_array(i) = Array<2, SIZE, DeviceType>({nr, nf}, pitched);
      *subarray_of_subarray(i) = SubArray(*subarray_of_array(i));
      subarray_of_array(i)->memset(0);
    }

    Array<1, Array<2, SIZE, DeviceType>, DeviceType> axis_max_array(
        {num_batches}, pitched, managed);
    Array<1, SubArray<2, SIZE, DeviceType>, DeviceType> axis_max_subarray(
        {num_batches}, pitched, managed);
    for (SIZE i = 0; i < num_batches; i++) {
      SubArray subarray_of_array(axis_max_array);
      SubArray subarray_of_subarray(axis_max_subarray);
      *subarray_of_array(i) = Array<2, SIZE, DeviceType>({nr, nf}, pitched);
      *subarray_of_subarray(i) = SubArray(*subarray_of_array(i));
      subarray_of_array(i)->memset(0);
    }

    Array<1, Array<3, SIZE, DeviceType>, DeviceType> edges_array(
        {num_batches}, pitched, managed);
    Array<1, SubArray<3, SIZE, DeviceType>, DeviceType> edges_subarray(
        {num_batches}, pitched, managed);
    for (SIZE i = 0; i < num_batches; i++) {
      SubArray subarray_of_array(edges_array);
      SubArray subarray_of_subarray(edges_subarray);
      *subarray_of_array(i) = Array<3, SIZE, DeviceType>({nr, nc, nf}, pitched);
      *subarray_of_subarray(i) = SubArray(*subarray_of_array(i));
      subarray_of_array(i)->memset(0);
    }

    Array<1, Array<2, SIZE, DeviceType>, DeviceType> cell_tri_count_array(
        {num_batches}, pitched, managed);
    Array<1, SubArray<2, SIZE, DeviceType>, DeviceType> cell_tri_count_subarray(
        {num_batches}, pitched, managed);
    for (SIZE i = 0; i < num_batches; i++) {
      SubArray subarray_of_array(cell_tri_count_array);
      SubArray subarray_of_subarray(cell_tri_count_subarray);
      *subarray_of_array(i) =
          Array<2, SIZE, DeviceType>({nr - 1, nf - 1}, pitched);
      *subarray_of_subarray(i) = SubArray(*subarray_of_array(i));
      subarray_of_array(i)->memset(0);
    }

    Array<1, Array<1, SIZE, DeviceType>, DeviceType> cell_tri_count_scan_array(
        {num_batches}, pitched, managed);
    Array<1, SubArray<1, SIZE, DeviceType>, DeviceType>
        cell_tri_count_scan_subarray({num_batches}, pitched, managed);
    for (SIZE i = 0; i < num_batches; i++) {
      SubArray subarray_of_array(cell_tri_count_scan_array);
      SubArray subarray_of_subarray(cell_tri_count_scan_subarray);
      *subarray_of_array(i) =
          Array<1, SIZE, DeviceType>({(nr - 1) * (nf - 1) + 1}, pitched);
      *subarray_of_subarray(i) = SubArray(*subarray_of_array(i));
      subarray_of_array(i)->memset(0);
    }

    SIZE *numTris = new SIZE[num_batches];
    SIZE *numPts = new SIZE[num_batches];
    for (SIZE i = 0; i < num_batches; i++) {
      numTris[i] = 0;
      numPts[i] = 0;
    }

    Triangles = Array<1, Array<1, SIZE, DeviceType>, DeviceType>(
        {num_batches}, pitched, managed);
    Array<1, SubArray<1, SIZE, DeviceType>, DeviceType> Triangles_subarray(
        {num_batches}, pitched, managed);

    Points = Array<1, Array<1, T, DeviceType>, DeviceType>({num_batches},
                                                           pitched, managed);
    Array<1, SubArray<1, T, DeviceType>, DeviceType> Points_subarray(
        {num_batches}, pitched, managed);

    SubArray axis_sum(axis_sum_subarray);
    SubArray axis_sum_scan(axis_sum_scan_subarray);
    SubArray axis_min(axis_min_subarray);
    SubArray axis_max(axis_max_subarray);
    SubArray edges(edges_subarray);
    SubArray cell_tri_count(cell_tri_count_subarray);
    SubArray cell_tri_count_scan(cell_tri_count_scan_subarray);

    using FunctorType1 = Pass1Functor<T, DeviceType>;
    using TaskType1 = Task<FunctorType1>;
    TaskType1 task1 = GenTask1<1, 8, 8>(nr, nc, nf, v, iso_value, axis_sum,
                                        axis_min, axis_max, edges, queue_idx);

    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    t1.start();
    DeviceAdapter<TaskType1, DeviceType>().Execute(task1);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    t1.end();
    pass1_time = t1.get();
    t1.clear();

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
    t2.start();
    DeviceAdapter<TaskType2, DeviceType>().Execute(task2);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    t2.end();
    pass2_time = t2.get();
    t2.clear();

    // printf("After pass2\n");
    // PrintSubarray("axis_sum", SubArray(axis_sum).Linearize());
    // PrintSubarray("cell_tri_count", SubArray(cell_tri_count));
    Array<1, SubArray<1, SIZE, DeviceType>, DeviceType>
        axis_sum_liearized_array({num_batches}, pitched, managed);
    SubArray<1, SubArray<1, SIZE, DeviceType>, DeviceType> axis_sum_liearized(
        axis_sum_liearized_array);
    for (SIZE i = 0; i < num_batches; i++) {
      *axis_sum_liearized(i) = axis_sum(i)->Linearize();
    }

    Array<1, SubArray<1, SIZE, DeviceType>, DeviceType>
        cell_tri_count_liearized_array({num_batches}, pitched, managed);
    SubArray<1, SubArray<1, SIZE, DeviceType>, DeviceType>
        cell_tri_count_liearized(cell_tri_count_liearized_array);
    for (SIZE i = 0; i < num_batches; i++) {
      *cell_tri_count_liearized(i) = cell_tri_count(i)->Linearize();
    }

    // SubArray<1, SIZE, DeviceType> axis_sum_liearized = axis_sum.Linearize();
    // SubArray<1, SIZE, DeviceType> cell_tri_count_liearized =
    // cell_tri_count.Linearize(); Array<1, SIZE, DeviceType>
    // newPointSize_array({1}); SubArray<1, SIZE, DeviceType>
    // newPointSize_subarray(newPointSize_array);

    t3.start();
    // non-batched
    // for (SIZE i = 0; i < num_batches; i++) {
    //   DeviceCollective<DeviceType>::ScanSumExtended(nr * nf * 3,
    //   *axis_sum_liearized(i), *axis_sum_scan(i), queue_idx);
    //   DeviceCollective<DeviceType>::ScanSumExtended((nr - 1) * (nf - 1),
    //   *cell_tri_count_liearized(i), *cell_tri_count_scan(i), queue_idx);
    // }

    // batched
    Array<1, SIZE, DeviceType> axis_sum_liearized_concat_array(
        {nr * nf * 3 * num_batches});
    Array<1, SIZE, DeviceType> cell_tri_count_liearized_concat_array(
        {(nr - 1) * (nf - 1) * num_batches});
    Array<1, SIZE, DeviceType> axis_sum_scan_offset_array({num_batches},
                                                          pitched, managed);
    Array<1, SIZE, DeviceType> cell_tri_count_scan_offset_array(
        {num_batches}, pitched, managed);
    axis_sum_scan_offset_array.memset(0);
    cell_tri_count_scan_offset_array.memset(0);

    SubArray axis_sum_liearized_concat(axis_sum_liearized_concat_array);
    SubArray cell_tri_count_liearized_concat(
        cell_tri_count_liearized_concat_array);
    SubArray axis_sum_scan_offset(axis_sum_scan_offset_array);
    SubArray cell_tri_count_scan_offset(cell_tri_count_scan_offset_array);

    for (SIZE i = 0; i < num_batches; i++) {
      SIZE offset = i * nr * nf * 3;
      MemoryManager<DeviceType>::Copy1D(
          axis_sum_liearized_concat.data() + offset,
          axis_sum_liearized(i)->data(), nr * nf * 3,
          i % MGARDX_NUM_ASYNC_QUEUES);
    }
    for (SIZE i = 0; i < num_batches; i++) {
      SIZE offset = i * (nr - 1) * (nf - 1);
      MemoryManager<DeviceType>::Copy1D(
          cell_tri_count_liearized_concat.data() + offset,
          cell_tri_count_liearized(i)->data(), (nr - 1) * (nf - 1),
          i % MGARDX_NUM_ASYNC_QUEUES);
    }
    DeviceRuntime<DeviceType>::SyncDevice();
    DeviceCollective<DeviceType>::ScanSumInclusive(
        nr * nf * 3 * num_batches, axis_sum_liearized_concat,
        axis_sum_liearized_concat, queue_idx);
    DeviceCollective<DeviceType>::ScanSumInclusive(
        (nr - 1) * (nf - 1) * num_batches, cell_tri_count_liearized_concat,
        cell_tri_count_liearized_concat, queue_idx);
    DeviceRuntime<DeviceType>::SyncDevice();
    for (SIZE i = 0; i < num_batches; i++) {
      SIZE offset = i * nr * nf * 3;
      // +1 is for making the first element 0
      if (i == 0) {
        MemoryManager<DeviceType>::Copy1D(
            axis_sum_scan(i)->data() + 1,
            axis_sum_liearized_concat.data() + offset, nr * nf * 3,
            i % MGARDX_NUM_ASYNC_QUEUES);
      } else {
        MemoryManager<DeviceType>::Copy1D(
            axis_sum_scan(i)->data(),
            axis_sum_liearized_concat.data() + offset - 1, nr * nf * 3 + 1,
            i % MGARDX_NUM_ASYNC_QUEUES);
      }

      // last element in the current batch is the 'scan offset' of the next
      // betch
      MemoryManager<DeviceType>::Copy1D(axis_sum_scan_offset.data() + i,
                                        axis_sum_liearized_concat.data() +
                                            offset + nr * nf * 3 - 1,
                                        1, i % MGARDX_NUM_ASYNC_QUEUES);
    }
    for (SIZE i = 0; i < num_batches; i++) {
      SIZE offset = i * (nr - 1) * (nf - 1);
      if (i == 0) {
        MemoryManager<DeviceType>::Copy1D(
            cell_tri_count_scan(i)->data() + 1,
            cell_tri_count_liearized_concat.data() + offset,
            (nr - 1) * (nf - 1), i % MGARDX_NUM_ASYNC_QUEUES);
      } else {
        MemoryManager<DeviceType>::Copy1D(
            cell_tri_count_scan(i)->data(),
            cell_tri_count_liearized_concat.data() + offset - 1,
            (nr - 1) * (nf - 1) + 1, i % MGARDX_NUM_ASYNC_QUEUES);
      }
      // last element in the current batch is the 'scan offset' of the next
      // betch
      MemoryManager<DeviceType>::Copy1D(cell_tri_count_scan_offset.data() + i,
                                        cell_tri_count_liearized_concat.data() +
                                            offset + (nr - 1) * (nf - 1) - 1,
                                        1, i % MGARDX_NUM_ASYNC_QUEUES);
    }
    DeviceRuntime<DeviceType>::SyncDevice();

    t3.end();
    pass3_time = t3.get();
    t3.clear();

    // printf("After pass3\n");
    // PrintSubarray("cell_tri_count_scan_offset", cell_tri_count_scan_offset);
    // PrintSubarray("cell_tri_count_liearized(0)",
    // *cell_tri_count_liearized((IDX)0)); PrintSubarray("axis_sum_liearized",
    // axis_sum_liearized);

    // MemoryManager<DeviceType>().Copy1D(&newPointSize, axis_sum_scan(nr * nf *
    // 3), 1, queue_idx); MemoryManager<DeviceType>().Copy1D(&numTris,
    // cell_tri_count_scan((nr - 1) * (nf - 1)), 1, queue_idx);
    for (SIZE i = 0; i < num_batches; i++) {
      SIZE axis_sum_scan_curr_offset = *axis_sum_scan_offset(i);
      SIZE axis_sum_scan_prev_offset =
          i == 0 ? 0 : *axis_sum_scan_offset(i - 1);
      SIZE cell_tri_count_scan_curr_offset = *cell_tri_count_scan_offset(i);
      SIZE cell_tri_count_scan_prev_offset =
          i == 0 ? 0 : *cell_tri_count_scan_offset(i - 1);
      // Check if current batch has work to do
      numPts[i] = axis_sum_scan_curr_offset - axis_sum_scan_prev_offset;
      numTris[i] =
          cell_tri_count_scan_curr_offset - cell_tri_count_scan_prev_offset;
    }

    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    // Triangles = Array<1, SIZE, DeviceType>({numTris * 3}, pitched);
    // Points = Array<1, T, DeviceType>({newPointSize * 3}, pitched);

    for (SIZE i = 0; i < num_batches; i++) {
      // std::cout << "numTris: " << numTris[i] << "\n";
      SubArray subarray_of_array(Triangles);
      SubArray subarray_of_subarray(Triangles_subarray);
      *subarray_of_array(i) =
          Array<1, SIZE, DeviceType>({numTris[i] * 3}, pitched);
      *subarray_of_subarray(i) = SubArray(*subarray_of_array(i));
      // if (numTris[i] > 0) {
      //   PrintSubarray("cell_tri_count_scan", *cell_tri_count_scan(i));
      // }
    }

    for (SIZE i = 0; i < num_batches; i++) {
      // std::cout << "numPts: " << numPts[i] << "\n";
      SubArray subarray_of_array(Points);
      SubArray subarray_of_subarray(Points_subarray);
      *subarray_of_array(i) = Array<1, T, DeviceType>({numPts[i] * 3}, pitched);
      *subarray_of_subarray(i) = SubArray(*subarray_of_array(i));
    }
    SubArray triangle_topology(Triangles_subarray);
    SubArray points(Points_subarray);

    DeviceRuntime<DeviceType>::SyncDevice();

    using FunctorType4 = Pass4Functor<T, DeviceType>;
    using TaskType4 = Task<FunctorType4>;
    TaskType4 task4 = GenTask4<1, 16, 16>(
        nr, nc, nf, v, iso_value, axis_sum_scan, axis_sum_scan_offset, axis_min,
        axis_max, cell_tri_count_scan, cell_tri_count_scan_offset, edges,
        triangle_topology, points, queue_idx);

    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    t4.start();
    DeviceAdapter<TaskType4, DeviceType>().Execute(task4);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    t4.end();
    pass4_time = t4.get();
    t4.clear();

    // printf("After pass4\n");
    // PrintSubarray("triangle_topology", triangle_topology);
    // PrintSubarray("points", points);
  }
};

} // namespace flying_edges_batched
} // namespace mgard_x
#endif