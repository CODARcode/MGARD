/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction oT Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_GENERATE_CL_TEMPLATE_HPP
#define MGARD_X_GENERATE_CL_TEMPLATE_HPP

#include "../../RuntimeX/RuntimeX.h"

namespace mgard_x {

#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MOD(a, b) ((((a) % (b)) + (b)) % (b))

static MGARDX_MANAGED int iNodesFront = 0;
static MGARDX_MANAGED int iNodesRear = 0;
static MGARDX_MANAGED int lNodesCur = 0;

static MGARDX_MANAGED int iNodesSize = 0;
static MGARDX_MANAGED int curLeavesNum;

static MGARDX_MANAGED int minFreq;

static MGARDX_MANAGED int tempLength;

static MGARDX_MANAGED int mergeFront;
static MGARDX_MANAGED int mergeRear;

static MGARDX_MANAGED int lNodesIndex;


static MGARDX_MANAGED int cStart;
static MGARDX_MANAGED int cEnd;
static MGARDX_MANAGED int iStart;
static MGARDX_MANAGED int iEnd;
static MGARDX_MANAGED int iNodesCap;

static MGARDX_MANAGED bool HuffmanCL_loop_condition1;

template <typename T, typename DeviceType>
class GenerateCLFunctor: public HuffmanCLCustomizedFunctor<DeviceType> {
  public:
  MGARDX_CONT GenerateCLFunctor(){}
  MGARDX_CONT GenerateCLFunctor(
    SubArray<1, T, DeviceType> histogram,  SubArray<1, T, DeviceType> CL,  int size,
    /* Global Arrays */
    SubArray<1, T, DeviceType> lNodesFreq,  SubArray<1, int, DeviceType> lNodesLeader,
    SubArray<1, T, DeviceType> iNodesFreq,  SubArray<1, int, DeviceType> iNodesLeader,
    SubArray<1, T, DeviceType> tempFreq,    SubArray<1, int, DeviceType> tempIsLeaf,    SubArray<1, int, DeviceType> tempIndex,
    SubArray<1, T, DeviceType> copyFreq,    SubArray<1, int, DeviceType> copyIsLeaf,    SubArray<1, int, DeviceType> copyIndex,
    SubArray<1, uint32_t, DeviceType> diagonal_path_intersections
    ):
    histogram(histogram), CL(CL), size(size),
    lNodesFreq(lNodesFreq), lNodesLeader(lNodesLeader),
    iNodesFreq(iNodesFreq), iNodesLeader(iNodesLeader),
    tempFreq(tempFreq), tempIsLeaf(tempIsLeaf), tempIndex(tempIndex),
    copyFreq(copyFreq), copyIsLeaf(copyIsLeaf), copyIndex(copyIndex),
    diagonal_path_intersections(diagonal_path_intersections)
    {
    HuffmanCLCustomizedFunctor<DeviceType>();                  
  }

  MGARDX_EXEC void
  Operation1() {
    i = (FunctorBase<DeviceType>::GetBlockIdX() * FunctorBase<DeviceType>::GetBlockDimX()) + FunctorBase<DeviceType>::GetThreadIdX();
    // i = thread; // Adaptation for easier porting

    /* Initialization */
    if (i < size) {
      *lNodesLeader((IDX)i) = -1;
      *CL((IDX)i) = 0;
    }

    if (i == 0) {
      iNodesFront = 0;
      iNodesRear = 0;
      lNodesCur = 0;

      iNodesSize = 0;
    }
  }

  MGARDX_CONT_EXEC bool
  LoopCondition1() {
    // printf("LoopCondition1 %d %u %d\n", lNodesCur, size, iNodesSize);
    HuffmanCL_loop_condition1 = lNodesCur < size || iNodesSize > 1;
    return HuffmanCL_loop_condition1;
  }

  MGARDX_EXEC void
  Operation2() {
    i = (FunctorBase<DeviceType>::GetBlockIdX() * FunctorBase<DeviceType>::GetBlockDimX()) + FunctorBase<DeviceType>::GetThreadIdX();
    /* Combine two most frequent nodes on same level */
    if (i == 0) {
      T midFreq[4];
      int midIsLeaf[4];
      for (int j = 0; j < 4; ++j) {
        midFreq[j] = UINT_MAX;
        midIsLeaf[j] = 0;
      }

      if (lNodesCur < size) {
        midFreq[0] = *lNodesFreq(lNodesCur);
        midIsLeaf[0] = 1;
      }
      if (lNodesCur < size - 1) {
        midFreq[1] = *lNodesFreq(lNodesCur + 1);
        midIsLeaf[1] = 1;
      }
      if (iNodesSize >= 1) {
        midFreq[2] = *iNodesFreq(iNodesFront);
        midIsLeaf[2] = 0;
      }
      if (iNodesSize >= 2) {
        midFreq[3] = *iNodesFreq(MOD(iNodesFront + 1, size));
        midIsLeaf[3] = 0;
      }

      // printf("midIsLeaf: %d %d %d %d\n", midIsLeaf[0], midIsLeaf[1], midIsLeaf[2], midIsLeaf[3]);

      /* Select the minimum of minimums - 4elt sorting network */
      /* TODO There's likely a good 1-warp faster way to do this */
      {
        T tempFreq;
        int tempIsLeaf;
        if (midFreq[1] > midFreq[3]) {
          tempFreq = midFreq[1];
          midFreq[1] = midFreq[3];
          midFreq[3] = tempFreq;
          tempIsLeaf = midIsLeaf[1];
          midIsLeaf[1] = midIsLeaf[3];
          midIsLeaf[3] = tempIsLeaf;
        }
        if (midFreq[0] > midFreq[2]) {
          tempFreq = midFreq[0];
          midFreq[0] = midFreq[2];
          midFreq[2] = tempFreq;
          tempIsLeaf = midIsLeaf[0];
          midIsLeaf[0] = midIsLeaf[2];
          midIsLeaf[2] = tempIsLeaf;
        }
        if (midFreq[0] > midFreq[1]) {
          tempFreq = midFreq[0];
          midFreq[0] = midFreq[1];
          midFreq[1] = tempFreq;
          tempIsLeaf = midIsLeaf[0];
          midIsLeaf[0] = midIsLeaf[1];
          midIsLeaf[1] = tempIsLeaf;
        }
        if (midFreq[2] > midFreq[3]) {
          tempFreq = midFreq[2];
          midFreq[2] = midFreq[3];
          midFreq[3] = tempFreq;
          tempIsLeaf = midIsLeaf[2];
          midIsLeaf[2] = midIsLeaf[3];
          midIsLeaf[3] = tempIsLeaf;
        }
        if (midFreq[1] > midFreq[2]) {
          tempFreq = midFreq[1];
          midFreq[1] = midFreq[2];
          midFreq[2] = tempFreq;
          tempIsLeaf = midIsLeaf[1];
          midIsLeaf[1] = midIsLeaf[2];
          midIsLeaf[2] = tempIsLeaf;
        }
      }

      // printf("mine: lNodesCur: %u, iNodesSize: %u\n", lNodesCur, iNodesSize);
      // printf("mine: midFreq[0]: %u, midFreq[1]: %u\n", midFreq[0], midFreq[1]);
      // printf("mine: midIsLeaf[0]: %u, midIsLeaf[1]: %u\n", midIsLeaf[0], midIsLeaf[1]);

      minFreq = midFreq[0];
      if (midFreq[1] < UINT_MAX) {
        minFreq += midFreq[1];
      }
      *iNodesFreq((IDX)iNodesRear) = minFreq;
      *iNodesLeader((IDX)iNodesRear) = -1;

      // printf("mine: iNodesLeader(0.leader) = %d, iNodesRear: %u\n", *iNodesLeader(IDX(0)), iNodesRear);

      /* If is leaf */
      if (midIsLeaf[0]) {
        *lNodesLeader((IDX)lNodesCur) = iNodesRear;
        ++(*CL((IDX)lNodesCur)), ++lNodesCur;
        // printf("update CL(%d) = %u\n", lNodesCur-1, *CL(lNodesCur-1));
      } else {
        *iNodesLeader((IDX)iNodesFront) = iNodesRear;
        iNodesFront = MOD(iNodesFront + 1, size);
      }
      if (midIsLeaf[1]) {
        *lNodesLeader((IDX)lNodesCur) = iNodesRear;
        ++(*CL((IDX)lNodesCur)), ++lNodesCur;
      } else {
        *iNodesLeader((IDX)iNodesFront) = iNodesRear;
        // printf("*iNodesLeader(%d): %d\n", iNodesFront, *iNodesLeader(iNodesFront));
        iNodesFront = MOD(iNodesFront + 1, size); /* ? */
      }

      // iNodesRear = MOD(iNodesRear + 1, size);

      iNodesSize = MOD(iNodesRear - iNodesFront, size);

      // printf("mine: iNodesLeader(0.leader) = %d, iNodesRear: %u\n", *iNodesLeader(IDX(0)), iNodesRear);
    }

    // int curLeavesNum;
    /* Select elements to copy -- parallelized */
    curLeavesNum = 0;


  }

  MGARDX_EXEC void
  Operation3() {
    i = (FunctorBase<DeviceType>::GetBlockIdX() * FunctorBase<DeviceType>::GetBlockDimX()) + FunctorBase<DeviceType>::GetThreadIdX();
    /* Select elements to copy -- parallelized */
    if (i >= lNodesCur && i < size) {
      // Parallel component
      int threadCurLeavesNum;
      if (*lNodesFreq((IDX)i) <= minFreq) {
        threadCurLeavesNum = i - lNodesCur + 1;
        // Atomic max -- Largest valid index
        Atomic<DeviceType>::Max(&curLeavesNum, threadCurLeavesNum);
      }

      if (i - lNodesCur < curLeavesNum) {
        *copyFreq((IDX)i - lNodesCur) = *lNodesFreq((IDX)i);
        *copyIndex((IDX)i - lNodesCur) = i;
        *copyIsLeaf((IDX)i - lNodesCur) = 1;
      }
    }
  }

  MGARDX_EXEC void
  Operation4() {
    i = (FunctorBase<DeviceType>::GetBlockIdX() * FunctorBase<DeviceType>::GetBlockDimX()) + FunctorBase<DeviceType>::GetThreadIdX();
    // if (!thread) {
    //   printf("curLeavesNum: %d\n", curLeavesNum);
    // }

    /* Updates Iterators */
    if (i == 0) {
      mergeRear = iNodesRear;
      mergeFront = iNodesFront;

      if ((curLeavesNum + iNodesSize) % 2 == 0) {
        iNodesFront = iNodesRear;
      }
      /* Odd number of nodes to merge - leave out one*/
      else if ((iNodesSize != 0)      //
               and (curLeavesNum == 0 //
                    or (*histogram((IDX)lNodesCur + curLeavesNum) <=
                        *iNodesFreq((IDX)MOD(iNodesRear - 1, size)))) //
      ) {
        mergeRear = MOD(mergeRear - 1, size);
        iNodesFront = MOD(iNodesRear - 1, size);
      } else {
        iNodesFront = iNodesRear;
        --curLeavesNum;
      }

      lNodesCur = lNodesCur + curLeavesNum;
      iNodesRear = MOD(iNodesRear + 1, size);
    }
  }

  MGARDX_CONT_EXEC bool
  BranchCondition1() {
    cStart = 0;
    cEnd = curLeavesNum;
    iStart = mergeFront;
    iEnd = mergeRear;
    iNodesCap = size;
    tempLength = (cEnd - cStart) + MOD(iEnd - iStart, iNodesCap);
    return tempLength > 0;
  }

  MGARDX_EXEC void
  Operation5() {

    int32_t * sm = (int32_t*)FunctorBase<DeviceType>::GetSharedMemory();
    x_top = &sm[0];
    y_top = &sm[1];
    x_bottom = &sm[2];
    y_bottom = &sm[3];
    found = &sm[4];
    oneorzero = &sm[5];

    cStart = 0;
    cEnd = curLeavesNum;
    iStart = mergeFront;
    iEnd = mergeRear;
    iNodesCap = size;
    // blocks = mblocks;
    // threads = mthreads;
    
    // tempLength = (cEnd - cStart) + MOD(iEnd - iStart, iNodesCap);
    // if (tempLength == 0) return;
    A_length = cEnd - cStart;
    B_length = MOD(iEnd - iStart, iNodesCap);

    // if (!thread) {
    //   printf("A_length: %d, B_length: %d, tempLength: %d\n", A_length, B_length, tempLength);
    // }
    // Calculate combined index around the MergePath "matrix"
    combinedIndex =
        ((uint64_t)FunctorBase<DeviceType>::GetBlockIdX() * ((uint64_t)A_length + (uint64_t)B_length)) /
        (uint64_t)FunctorBase<DeviceType>::GetGridDimX();

    // if (!FunctorBase<DeviceType>::GetThreadIdX()) {
    //   printf("A_length: %d, B_length: %d, tempLength: %d, combinedIndex: %d\n", A_length, B_length, tempLength, combinedIndex);
    // }
    threadOffset = FunctorBase<DeviceType>::GetThreadIdX() - MGARDX_WARP_SIZE/2;

    if (FunctorBase<DeviceType>::GetThreadIdX() < MGARDX_WARP_SIZE) {
      // Figure out the coordinates of our diagonal
      if (A_length >= B_length) {
        *x_top = MIN(combinedIndex, A_length);
        *y_top = combinedIndex > A_length ? combinedIndex - (A_length) : 0;
        *x_bottom = *y_top;
        *y_bottom = *x_top;
      } else {
        *y_bottom = MIN(combinedIndex, B_length);
        *x_bottom = combinedIndex > B_length ? combinedIndex - (B_length) : 0;
        *y_top = *x_bottom;
        *x_top = *y_bottom;
      }
    }
    *found = 0;
  }


  MGARDX_CONT_EXEC bool
  LoopCondition2() {
    // printf("%u LoopCondition2\n", FunctorBase<DeviceType>::GetBlockIdX());
    return !(*found);
  }

  MGARDX_EXEC void
  Operation6() {
    // Update our coordinates within the 32-wide section of the diagonal
    // if (!FunctorBase<DeviceType>::GetThreadIdX()) {
    //   printf("x %d %d y: %d %d\n", *x_top, *x_bottom, *y_top, *y_bottom);
    // }
    current_x = *x_top - ((*x_top - *x_bottom) >> 1) - threadOffset;
    current_y = *y_top + ((*y_bottom - *y_top) >> 1) + threadOffset;
    getfrom_x = current_x + cStart - 1;
    // Below statement is a more efficient, divmodless version of the following
    // int32_t getfrom_y = MOD(iStart + current_y, iNodesCap);
    getfrom_y = iStart + current_y;

    if (FunctorBase<DeviceType>::GetThreadIdX() < MGARDX_WARP_SIZE) {
      if (getfrom_y >= iNodesCap)
        getfrom_y -= iNodesCap;

      // Are we a '1' or '0' with respect to A[x] <= B[x]
      if (current_x > (int32_t)A_length or current_y < 0) {
        oneorzero[FunctorBase<DeviceType>::GetThreadIdX()] = 0;
      } else if (current_y >= (int32_t)B_length || current_x < 1) {
        oneorzero[FunctorBase<DeviceType>::GetThreadIdX()] = 1;
      } else {
        oneorzero[FunctorBase<DeviceType>::GetThreadIdX()] =
            (*copyFreq(getfrom_x) <= *iNodesFreq(getfrom_y)) ? 1 : 0;
      }
    }
  }

  MGARDX_EXEC void
  Operation7() {
    //     if (!FunctorBase<DeviceType>::GetThreadIdX())
    //     printf("%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d \n\
    // %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d \n\n",
    //             oneorzero[0], oneorzero[1], oneorzero[2], oneorzero[3],
    //             oneorzero[4], oneorzero[5], oneorzero[6], oneorzero[7],
    //             oneorzero[8], oneorzero[9], oneorzero[10], oneorzero[11],
    //             oneorzero[12], oneorzero[13], oneorzero[14], oneorzero[15],
    //             oneorzero[16], oneorzero[17], oneorzero[18], oneorzero[19],
    //             oneorzero[20], oneorzero[21], oneorzero[22], oneorzero[23],
    //             oneorzero[24], oneorzero[25], oneorzero[26], oneorzero[27],
    //             oneorzero[28], oneorzero[29], oneorzero[30], oneorzero[31]);
    // If we find the meeting of the '1's and '0's, we found the
    // intersection of the path and diagonal
    if (FunctorBase<DeviceType>::GetThreadIdX() > 0 and                                    //
        FunctorBase<DeviceType>::GetThreadIdX() < MGARDX_WARP_SIZE and                                   //
        (oneorzero[FunctorBase<DeviceType>::GetThreadIdX()] != oneorzero[FunctorBase<DeviceType>::GetThreadIdX() - 1]) //
    ) {
      // printf("found\n");
      *found = 1;

      *diagonal_path_intersections((IDX)FunctorBase<DeviceType>::GetBlockIdX()) = current_x;
      *diagonal_path_intersections((IDX)FunctorBase<DeviceType>::GetBlockIdX() + FunctorBase<DeviceType>::GetGridDimX() + 1) = current_y;
    }
  }

  MGARDX_EXEC void
  Operation8() {
    // Adjust the search window on the diagonal
    if (FunctorBase<DeviceType>::GetThreadIdX() == 16) {
      if (oneorzero[31] != 0) {
        *x_bottom = current_x;
        *y_bottom = current_y;
      } else {
        *x_top = current_x;
        *y_top = current_y;
      }
    }
  }

  //end of loop 2

  MGARDX_EXEC void
  Operation9() {
    // Set the boundary diagonals (through 0,0 and A_length,B_length)
    if (FunctorBase<DeviceType>::GetThreadIdX() == 0 && FunctorBase<DeviceType>::GetBlockIdX() == 0) {
      *diagonal_path_intersections((IDX)0) = 0;
      *diagonal_path_intersections((IDX)FunctorBase<DeviceType>::GetGridDimX() + 1) = 0;
      *diagonal_path_intersections((IDX)FunctorBase<DeviceType>::GetGridDimX()) = A_length;
      *diagonal_path_intersections((IDX)FunctorBase<DeviceType>::GetGridDimX() + FunctorBase<DeviceType>::GetGridDimX() + 1) = B_length;
    }
  }

  MGARDX_EXEC void
  Operation10() {
    if (FunctorBase<DeviceType>::GetThreadIdX() == 0) {
      // Boundaries
      int x_block_top = *diagonal_path_intersections((IDX)FunctorBase<DeviceType>::GetBlockIdX());
      int y_block_top = *diagonal_path_intersections((IDX)FunctorBase<DeviceType>::GetBlockIdX() + FunctorBase<DeviceType>::GetGridDimX() + 1);
      int x_block_stop = *diagonal_path_intersections((IDX)FunctorBase<DeviceType>::GetBlockIdX() + 1);
      int y_block_stop = *diagonal_path_intersections((IDX)FunctorBase<DeviceType>::GetBlockIdX() + FunctorBase<DeviceType>::GetGridDimX() + 2);

      // Actual indexes
      int x_start = x_block_top + cStart;
      int x_end = x_block_stop + cStart;
      int y_start = MOD(iStart + y_block_top, iNodesCap);
      int y_end = MOD(iStart + y_block_stop, iNodesCap);

      int offset = x_block_top + y_block_top;

      // printf("x_block_top: %d y_block_top: %d, offset: %d\n", x_block_top, y_block_top, offset);

      int dummy; // Unused result
      // TODO optimize serial merging of each partition
      int len = 0;

      int iterCopy = x_start, iterINodes = y_start;

      while (iterCopy < x_end && MOD(y_end - iterINodes, iNodesCap) > 0) {
        if (*copyFreq((IDX)iterCopy) <= *iNodesFreq((IDX)iterINodes)) {
          *tempFreq((IDX)offset+len) = *copyFreq((IDX)iterCopy);
          *tempIndex((IDX)offset+len) = *copyIndex((IDX)iterCopy);
          *tempIsLeaf((IDX)offset+len) = *copyIsLeaf((IDX)iterCopy);
          ++iterCopy;
        } else {
          *tempFreq((IDX)offset+len) = *iNodesFreq((IDX)iterINodes);
          *tempIndex((IDX)offset+len) = iterINodes;
          *tempIsLeaf((IDX)offset+len) = 0;
          iterINodes = MOD(iterINodes + 1, iNodesCap);
        }
        ++len;
      }

      while (iterCopy < x_end) {
        *tempFreq((IDX)offset+len) = *copyFreq((IDX)iterCopy);
        *tempIndex((IDX)offset+len) = *copyIndex((IDX)iterCopy);
        *tempIsLeaf((IDX)offset+len) = *copyIsLeaf((IDX)iterCopy);
        ++iterCopy;
        ++len;
      }
      while (MOD(y_end - iterINodes, iNodesCap) > 0) {
        *tempFreq((IDX)offset+len) = *iNodesFreq((IDX)iterINodes);
        *tempIndex((IDX)offset+len) = iterINodes;
        *tempIsLeaf((IDX)offset+len) = 0;
        iterINodes = MOD(iterINodes + 1, iNodesCap);
        ++len;
      }

      // for (int i = 0; i < len; i++) {
      //   if (*tempIsLeaf((IDX)offset+i) == 2 ) {
      //     printf("*copyIsLeaf((IDX)iterCopy) = 2\n");
      //   }
      // }

      // if (FunctorBase<DeviceType>::GetThreadIdX() == 0) {
      //   printf("FunctorBase<DeviceType>::GetGridDimX(): %llu, offset: %d, len: %d\n", FunctorBase<DeviceType>::GetGridDimX(), offset, len);
      //   printf("leaf: %d %d\n", *tempIsLeaf((IDX)2 * 4), *tempIsLeaf((IDX)2 * 4 + 1));
      // }
    }

    // cg::this_grid().sync();
    // if (FunctorBase<DeviceType>::GetThreadIdX() == 0) {
    //   // printf("FunctorBase<DeviceType>::GetGridDimX(): %u, offset: %d, len: %d\n", FunctorBase<DeviceType>::GetGridDimX(), offset, len);
    //   printf("leaf: %d %d\n", *tempIsLeaf((IDX)2 * 4), *tempIsLeaf((IDX)2 * 4 + 1));
    // }

  }
  
  // end of parallel merge

  MGARDX_EXEC void
  Operation11() {
    i = (FunctorBase<DeviceType>::GetBlockIdX() * FunctorBase<DeviceType>::GetBlockDimX()) + FunctorBase<DeviceType>::GetThreadIdX();
    // if (thread == 0) {
    //   printf("leaf: %d %d\n", *tempIsLeaf((IDX)2 * 4), *tempIsLeaf((IDX)2 * 4 + 1));
    // }
    /* Melding phase -- New */
    if (i < tempLength / 2) {
      int ind = MOD(iNodesRear + i, size);
      // printf("Melding(i=%d): %u(%d) %u(%d)\n", i, *tempFreq((IDX)2 * i), *tempIsLeaf((IDX)2 * i), *tempFreq((IDX)2 * i + 1), *tempIsLeaf((IDX)2 * i + 1));
      *iNodesFreq((IDX)ind) = *tempFreq((IDX)2 * i) + *tempFreq((IDX)2 * i + 1);
      *iNodesLeader((IDX)ind) = -1;

      if (*tempIsLeaf((IDX)2 * i)) {
        *lNodesLeader((IDX)*tempIndex((IDX)2 * i)) = ind;
        ++(*CL(*tempIndex(2 * i)));
      } else {
        *iNodesLeader((IDX)*tempIndex((IDX)2 * i)) = ind;
      }
      if (*tempIsLeaf((IDX)2 * i + 1)) {
        *lNodesLeader((IDX)*tempIndex((IDX)2 * i + 1)) = ind;
        ++(*CL(*tempIndex(2 * i + 1)));
      } else {
        *iNodesLeader((IDX)*tempIndex((IDX)2 * i + 1)) = ind;
      }
    }
  }

  MGARDX_EXEC void
  Operation12() {
    i = (FunctorBase<DeviceType>::GetBlockIdX() * FunctorBase<DeviceType>::GetBlockDimX()) + FunctorBase<DeviceType>::GetThreadIdX();
    if (i == 0) {
      iNodesRear = MOD(iNodesRear + (tempLength / 2), size);
    }
  }

  MGARDX_EXEC void
  Operation13() {
    i = (FunctorBase<DeviceType>::GetBlockIdX() * FunctorBase<DeviceType>::GetBlockDimX()) + FunctorBase<DeviceType>::GetThreadIdX();
    /* Update leaders */
    // if (thread == 0) {
    //   printf("mine: iNodesLeader(0.leader) = %d, iNodesRear: %u\n", *iNodesLeader(IDX(0)), iNodesRear);
    // }
    if (i < size) {
      if (*lNodesLeader((IDX)i) != -1) {
        if (*iNodesLeader((IDX)*lNodesLeader((IDX)i)) != -1) {
          *lNodesLeader((IDX)i) = *iNodesLeader((IDX)*lNodesLeader((IDX)i));
          ++(*CL((IDX)i));
          // printf("update CL(%d):%d\n", i, *CL(i));
        }
      }
    }
  }

  MGARDX_EXEC void
  Operation14() {
    i = (FunctorBase<DeviceType>::GetBlockIdX() * FunctorBase<DeviceType>::GetBlockDimX()) + FunctorBase<DeviceType>::GetThreadIdX();
    if (i == 0) {
      iNodesSize = MOD(iNodesRear - iNodesFront, size);
    }
  }

  MGARDX_CONT size_t
  shared_memory_size() { 
    size_t sm_size = 0;
    sm_size += 5 * sizeof(int32_t);
    sm_size += 32 * sizeof(int32_t); 
    return sm_size;
  }

  private:
  SubArray<1, T, DeviceType> histogram;
  SubArray<1, T, DeviceType> CL;
  int size;
  /* Global Arrays */
  SubArray<1, T, DeviceType> lNodesFreq;
  SubArray<1, int, DeviceType> lNodesLeader;
  SubArray<1, T, DeviceType> iNodesFreq;
  SubArray<1, int, DeviceType> iNodesLeader;
  SubArray<1, T, DeviceType> tempFreq;
  SubArray<1, int, DeviceType> tempIsLeaf;
  SubArray<1, int, DeviceType> tempIndex;
  SubArray<1, T, DeviceType> copyFreq;
  SubArray<1, int, DeviceType> copyIsLeaf;
  SubArray<1, int, DeviceType> copyIndex;
  SubArray<1, uint32_t, DeviceType> diagonal_path_intersections;
  // Array<1, uint32_t, DeviceType> diagonal_path_intersections_array;
  // int mblocks;
  // int mthreads;
  // SubArray<1, int, DeviceType> *iNodesFront.data();
  // SubArray<1, int, DeviceType> iNodesRear;
  // SubArray<1, int, DeviceType> iNodesSize; 
  // SubArray<1, int, DeviceType> *lNodesCur.data();
  // SubArray<1, int, DeviceType> curLeavesNum;
  // SubArray<1, int, DeviceType> minFreq;
  // SubArray<1, int, DeviceType> tempLength;
  // SubArray<1, int, DeviceType> mergeFront;
  // SubArray<1, int, DeviceType> mergeRear; 
  // SubArray<1, int, DeviceType> lNodesIndex;
  // SubArray<1, int, DeviceType> CCL;
  // SubArray<1, int, DeviceType> CDPI;
  // SubArray<1, int, DeviceType> newCDPI;

  int32_t *x_top;
  int32_t *y_top;
  int32_t *x_bottom;
  int32_t *y_bottom;
  int32_t *found;
  int32_t *oneorzero;

  // unsigned int thread;
  unsigned int i;

  // int cStart;
  // int cEnd;
  // int iStart;
  // int iEnd;
  // int iNodesCap;
  // int blocks;
  // int threads;
  int threadOffset;
  uint32_t A_length;
  uint32_t B_length;
  int32_t combinedIndex;
  int32_t current_x;
  int32_t current_y;
  int32_t getfrom_x;
  int32_t getfrom_y;
};


template <typename T, typename DeviceType>
class GenerateCL: public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  GenerateCL():AutoTuner<DeviceType>() {}

  MGARDX_CONT
  Task<GenerateCLFunctor<T, DeviceType> > 
  GenTask(SubArray<1, T, DeviceType> histogram,  SubArray<1, T, DeviceType> CL, int dict_size,
    /* Global Arrays */
    SubArray<1, T, DeviceType> lNodesFreq,  SubArray<1, int, DeviceType> lNodesLeader,
    SubArray<1, T, DeviceType> iNodesFreq,  SubArray<1, int, DeviceType> iNodesLeader,
    SubArray<1, T, DeviceType> tempFreq,    SubArray<1, int, DeviceType> tempIsLeaf,    SubArray<1, int, DeviceType> tempIndex,
    SubArray<1, T, DeviceType> copyFreq,    SubArray<1, int, DeviceType> copyIsLeaf,    SubArray<1, int, DeviceType> copyIndex,
    SubArray<1, uint32_t, DeviceType> diagonal_path_intersections, int queue_idx) {
    using FunctorType = GenerateCLFunctor<T, DeviceType>;
    FunctorType Functor(histogram, CL, dict_size, 
                        lNodesFreq, lNodesLeader,
                        iNodesFreq, iNodesLeader,
                        tempFreq, tempIsLeaf, tempIndex,
                        copyFreq, copyIsLeaf, copyIndex,
                        diagonal_path_intersections);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = Functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = DeviceRuntime<DeviceType>::GetWarpSize();

    int cg_blocks_sm = DeviceRuntime<DeviceType>::GetOccupancyMaxActiveBlocksPerSM(Functor, tbx, sm_size);
    int cg_mblocks = cg_blocks_sm * DeviceRuntime<DeviceType>::GetNumSMs();
    int ELTS_PER_SEQ_MERGE = 16;
    int mblocks = cg_mblocks; //std::min(cg_mblocks, (dict_size / ELTS_PER_SEQ_MERGE) + 1);

    gridz = 1;
    gridy = 1;
    gridx = mblocks;

    int tthreads = tbx * gridx;
    // if (tthreads < dict_size) {
    //   std::cout << log::log_err << "Insufficient on-device parallelism to construct a "
    //        << dict_size << " non-zero item codebook" << std::endl;
    //   std::cout << log::log_err << "Provided parallelism: " << gridx << " blocks, "
    //        << tbx << " threads, " << tthreads << " total" << std::endl
    //        << std::endl;
    //   exit(1);
    // }
    if (tthreads >= dict_size) {
      if (DeviceRuntime<DeviceType>::PrintKernelConfig) {
        std::cout << log::log_info << "GenerateCL: using Cooperative Groups\n";
      }
      Functor.use_CG = true;
    } else {
      if (DeviceRuntime<DeviceType>::PrintKernelConfig) {
        std::cout << log::log_info << "GenerateCL: not using Cooperative Groups\n";
      }
      Functor.use_CG = false;
      gridx = (dict_size - 1) / tbx + 1;
    }

    return Task(Functor, gridz, gridy, gridx, 
                tbz, tby, tbx, sm_size, queue_idx, "GenerateCL"); 
  }

  MGARDX_CONT
  void Execute(SubArray<1, T, DeviceType> histogram,  SubArray<1, T, DeviceType> CL, int dict_size,
    /* Global Arrays */
    SubArray<1, T, DeviceType> lNodesFreq,  SubArray<1, int, DeviceType> lNodesLeader,
    SubArray<1, T, DeviceType> iNodesFreq,  SubArray<1, int, DeviceType> iNodesLeader,
    SubArray<1, T, DeviceType> tempFreq,    SubArray<1, int, DeviceType> tempIsLeaf,    SubArray<1, int, DeviceType> tempIndex,
    SubArray<1, T, DeviceType> copyFreq,    SubArray<1, int, DeviceType> copyIsLeaf,    SubArray<1, int, DeviceType> copyIndex,
    SubArray<1, uint32_t, DeviceType> diagonal_path_intersections, int queue_idx) {
    using FunctorType = GenerateCLFunctor<T, DeviceType>;
    using TaskType = Task<FunctorType>;
    TaskType task = GenTask(histogram, CL, dict_size, 
                        lNodesFreq, lNodesLeader,
                        iNodesFreq, iNodesLeader,
                        tempFreq, tempIsLeaf, tempIndex,
                        copyFreq, copyIsLeaf, copyIndex,
                        diagonal_path_intersections, queue_idx); 
    DeviceAdapter<TaskType, DeviceType> adapter; 


    adapter.Execute(task);

    // DeviceRuntime<DeviceType>::SyncDevice();
    // std::cout << "iNodesRear: " << iNodesRear << "\n";
  }

  
};

#undef MOD
#undef MIN
#undef MAX

}



#endif