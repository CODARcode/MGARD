#ifndef PAR_MERGE_CUH
#define PAR_MERGE_CUH

#include <cuda.h>
#include <float.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/merge.h>
#include <thrust/sort.h>

#include <cooperative_groups.h>

using namespace cooperative_groups;

// Partition array
template <typename F>
__device__ void
cudaWorkloadDiagonals(F *copyFreq, int *copyIndex, int *copyIsLeaf, int cStart,
                      int cEnd, F *iNodesFreq, int iStart, int iEnd,
                      int iNodesCap, uint32_t *diagonal_path_intersections,
                      /* Shared Memory */
                      int32_t &x_top, int32_t &y_top, int32_t &x_bottom,
                      int32_t &y_bottom, int32_t &found, int32_t *oneorzero);

// Merge partitions
template <typename F>
__device__ void
cudaMergeSinglePath(F *copyFreq, int *copyIndex, int *copyIsLeaf, int cStart,
                    int cEnd, F *iNodesFreq, int iStart, int iEnd,
                    int iNodesCap, uint32_t *diagonal_path_intersections,
                    F *tempFreq, int *tempIndex, int *tempIsLeaf,
                    int tempLength);

template <typename F>
__device__ void
parMerge(F *copyFreq, int *copyIndex, int *copyIsLeaf, int cStart, int cEnd,
         F *iNodesFreq, int iStart, int iEnd, int iNodesCap, F *tempFreq,
         int *tempIndex, int *tempIsLeaf, int &tempLength,
         uint32_t *diagonal_path_intersections, int blocks, int threads,
         /* Shared Memory */
         int32_t &x_top, int32_t &y_top, int32_t &x_bottom, int32_t &y_bottom,
         int32_t &found, int32_t *oneorzero);

template <typename F>
__device__ void merge(F *copyFreq, int *copyIndex, int *copyIsLeaf, int cStart,
                      int cEnd, F *iNodesFreq, int iStart, int iEnd,
                      int iNodesCap, F *tempFreq, int *tempIndex,
                      int *tempIsLeaf, int &tempLength);

#endif
