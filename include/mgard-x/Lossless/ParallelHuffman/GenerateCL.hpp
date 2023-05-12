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

#define _iNodesFront 0
#define _iNodesRear 1
#define _lNodesCur 2
#define _iNodesSize 3
#define _curLeavesNum 4
#define _minFreq 5
#define _tempLength 6
#define _mergeFront 7
#define _mergeRear 8

template <typename T, typename DeviceType>
class GenerateCLFunctor : public HuffmanCLCustomizedFunctor<DeviceType> {
public:
  MGARDX_CONT GenerateCLFunctor() {}
  MGARDX_CONT GenerateCLFunctor(
      SubArray<1, T, DeviceType> histogram, SubArray<1, T, DeviceType> CL,
      int dict_size,
      /* Global Arrays */
      SubArray<1, T, DeviceType> lNodesFreq,
      SubArray<1, int, DeviceType> lNodesLeader,
      SubArray<1, T, DeviceType> iNodesFreq,
      SubArray<1, int, DeviceType> iNodesLeader,
      SubArray<1, T, DeviceType> tempFreq,
      SubArray<1, int, DeviceType> tempIsLeaf,
      SubArray<1, int, DeviceType> tempIndex,
      SubArray<1, T, DeviceType> copyFreq,
      SubArray<1, int, DeviceType> copyIsLeaf,
      SubArray<1, int, DeviceType> copyIndex,
      SubArray<1, uint32_t, DeviceType> diagonal_path_intersections,
      SubArray<1, int, DeviceType> status)
      : histogram(histogram), CL(CL), dict_size(dict_size),
        lNodesFreq(lNodesFreq), lNodesLeader(lNodesLeader),
        iNodesFreq(iNodesFreq), iNodesLeader(iNodesLeader), tempFreq(tempFreq),
        tempIsLeaf(tempIsLeaf), tempIndex(tempIndex), copyFreq(copyFreq),
        copyIsLeaf(copyIsLeaf), copyIndex(copyIndex),
        diagonal_path_intersections(diagonal_path_intersections),
        status(status) {
    HuffmanCLCustomizedFunctor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    i = (FunctorBase<DeviceType>::GetBlockIdX() *
         FunctorBase<DeviceType>::GetBlockDimX()) +
        FunctorBase<DeviceType>::GetThreadIdX();

    /* Initialization */
    // All leaf nodes have no leader at beginning
    if (i < dict_size) {
      *lNodesLeader((IDX)i) = -1;
      *CL((IDX)i) = 0;
    }

    if (i == 0) {
      // Front of internal node list
      (*status((IDX)_iNodesFront)) = 0;
      // End of internal node list
      (*status((IDX)_iNodesRear)) = 0;
      // Front of unprocessed leaf node list
      (*status((IDX)_lNodesCur)) = 0;
      // Size of internal node list
      (*status((IDX)_iNodesSize)) = 0;
    }
  }

  MGARDX_CONT_EXEC bool LoopCondition1() {

    // if (i == 0) {
    //   printf("************LoopCondition1 _lNodesCur(%d) (dict_size)%u
    //   (_iNodesSize)%d\n", (*status((IDX)_lNodesCur)), dict_size,
    //   (*status((IDX)_iNodesSize)));
    // }
    // Continue if we have not processed all leaf nodes OR
    // if unmeraged internal nodes is greater than 1 (more that just root)
    return (*status((IDX)_lNodesCur)) < dict_size ||
           (*status((IDX)_iNodesSize)) > 1;
  }

  MGARDX_EXEC void Operation2() {
    i = (FunctorBase<DeviceType>::GetBlockIdX() *
         FunctorBase<DeviceType>::GetBlockDimX()) +
        FunctorBase<DeviceType>::GetThreadIdX();
    /* Combine two least frequent nodes on same level */
    if (i == 0) {
      T midFreq[4];
      int midIsLeaf[4];
      for (int j = 0; j < 4; ++j) {
        midFreq[j] = UINT_MAX;
        midIsLeaf[j] = 0;
      }

      // Get two nodes from leaf node if possible
      if ((*status((IDX)_lNodesCur)) < dict_size) {
        midFreq[0] = *lNodesFreq((*status((IDX)_lNodesCur)));
        midIsLeaf[0] = 1;
      }
      if ((*status((IDX)_lNodesCur)) < dict_size - 1) {
        midFreq[1] = *lNodesFreq((*status((IDX)_lNodesCur)) + 1);
        midIsLeaf[1] = 1;
      }

      // Get two nodes from internal node if possible
      if ((*status((IDX)_iNodesSize)) >= 1) {
        midFreq[2] = *iNodesFreq((*status((IDX)_iNodesFront)));
        midIsLeaf[2] = 0;
      }
      if ((*status((IDX)_iNodesSize)) >= 2) {
        midFreq[3] =
            *iNodesFreq(MOD((*status((IDX)_iNodesFront)) + 1, dict_size));
        midIsLeaf[3] = 0;
      }

      // printf("midFreq: %u %u %u %u\n", midFreq[0], midFreq[1],
      // midFreq[2], midFreq[3]);
      // printf("midIsLeaf: %d %d %d %d\n", midIsLeaf[0], midIsLeaf[1],
      // midIsLeaf[2], midIsLeaf[3]);

      /* Select the minimum of minimums - sorting network */
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

      // printf("mine: (*status((IDX)_lNodesCur)): %u,
      // (*status((IDX)_iNodesSize)): %u\n", (*status((IDX)_lNodesCur)),
      // (*status((IDX)_iNodesSize))); printf("mine: midFreq[0]: %u, midFreq[1]:
      // %u\n", midFreq[0], midFreq[1]); printf("mine: midIsLeaf[0]: %u,
      // midIsLeaf[1]: %u\n", midIsLeaf[0], midIsLeaf[1]);

      // Calculate the merged frequency.
      // If we have only 1 node then do not merge
      (*status((IDX)_minFreq)) = midFreq[0];
      if (midFreq[1] < UINT_MAX) {
        (*status((IDX)_minFreq)) += midFreq[1];
      }

      // printf("merged freq: %u\n", *status((IDX)_minFreq));

      // Create a internal node in the end
      // Assign merged frequency
      // Assign no leader at beginning
      // _iNodesRear needs to be increased by 1 later
      *iNodesFreq((IDX)(*status((IDX)_iNodesRear))) = (*status((IDX)_minFreq));
      *iNodesLeader((IDX)(*status((IDX)_iNodesRear))) = -1;

      // printf("new internal node : %d\n", *status((IDX)_iNodesRear));

      // printf("mine: iNodesLeader(0.leader) = %d, (*status((IDX)_iNodesRear)):
      // %u\n", *iNodesLeader(IDX(0)), (*status((IDX)_iNodesRear)));

      /* If the smallest node is leaf */
      if (midFreq[0] < UINT_MAX) {
        if (midIsLeaf[0]) {
          // printf("mid[0] is leaf\n");
          *lNodesLeader((IDX)(*status((IDX)_lNodesCur))) =
              (*status((IDX)_iNodesRear));
          // printf("update leader of leaf %d to just created internal node
          // %d\n", *status((IDX)_lNodesCur),
          // *lNodesLeader((IDX)(*status((IDX)_lNodesCur))));
          ++(*CL((IDX)(*status((IDX)_lNodesCur)))),
              ++(*status((IDX)_lNodesCur));
          // printf("remove mid[0] from unprocessed leaf node list. _lNodesCur =
          // %d\n", *status((IDX)_lNodesCur)); printf("update CL(%d) = %u\n",
          // *status((IDX)_lNodesCur-1),
          //                   *CL((IDX)(*status((IDX)_lNodesCur-1))));
        } else {
          // printf("mid[0] is internal node\n");
          // Update its leader to the internal node we just created
          *iNodesLeader((IDX)(*status((IDX)_iNodesFront))) =
              (*status((IDX)_iNodesRear));
          // printf("update leader of internal node %d to just created internal
          // node %d\n", *status((IDX)_iNodesFront),
          // *iNodesLeader((IDX)(*status((IDX)_iNodesFront))));
          (*status((IDX)_iNodesFront)) =
              MOD((*status((IDX)_iNodesFront)) + 1, dict_size);
          // printf("remove mid[0] from unprocessed internal node list.
          // _iNodesFront = %d\n", *status((IDX)_iNodesFront));
        }
      }

      if (midFreq[1] < UINT_MAX) {
        /* If the second smallest node is leaf */
        if (midIsLeaf[1]) {
          // printf("mid[1] is leaf\n");
          *lNodesLeader((IDX)(*status((IDX)_lNodesCur))) =
              (*status((IDX)_iNodesRear));
          // printf("update leader of leaf %d to just created internal node
          // %d\n",
          //        *status((IDX)_lNodesCur),
          //        *lNodesLeader((IDX)(*status((IDX)_lNodesCur))));
          ++(*CL((IDX)(*status((IDX)_lNodesCur)))),
              ++(*status((IDX)_lNodesCur));
          // printf("remove mid[1] from unprocessed leaf node list. _lNodesCur =
          // %d\n", *status((IDX)_lNodesCur)); printf("update CL(%d) = %u\n",
          // *status((IDX)_lNodesCur-1),
          //                   *CL((IDX)(*status((IDX)_lNodesCur-1))));
        } else {
          // printf("mid[1] is internal node\n");
          *iNodesLeader((IDX)(*status((IDX)_iNodesFront))) =
              (*status((IDX)_iNodesRear));
          // printf("update leader of internal node %d to just created internal
          // node %d\n",
          //        *status((IDX)_iNodesFront),
          //        *iNodesLeader((IDX)(*status((IDX)_iNodesFront))));
          // printf("remove mid[1] from unprocessed internal node list.
          // _iNodesFront = %d\n", *status((IDX)_iNodesFront));
          (*status((IDX)_iNodesFront)) =
              MOD((*status((IDX)_iNodesFront)) + 1, dict_size);
          // printf("remove mid[1] from unprocessed internal node list.
          // _iNodesFront = %d\n", *status((IDX)_iNodesFront));
        }
      }

      // (*status((IDX)_iNodesRear)) = MOD((*status((IDX)_iNodesRear)) + 1,
      // dict_size);

      // Update internal node list size
      (*status((IDX)_iNodesSize)) =
          MOD((*status((IDX)_iNodesRear)) - (*status((IDX)_iNodesFront)),
              dict_size);

      // printf("update number of unprocessed internal node (not including just
      // created) _iNodesSize: %d\n", *status((IDX)_iNodesSize));

      // printf("mine: iNodesLeader(0.leader) = %d, (*status((IDX)_iNodesRear)):
      // %u\n", *iNodesLeader(IDX(0)), (*status((IDX)_iNodesRear)));

      // Reset total number of leaf nodes to be parallel merged
      (*status((IDX)_curLeavesNum)) = 0;
    }
  }

  MGARDX_EXEC void Operation3() {
    i = (FunctorBase<DeviceType>::GetBlockIdX() *
         FunctorBase<DeviceType>::GetBlockDimX()) +
        FunctorBase<DeviceType>::GetThreadIdX();
    /* Select elements to copy -- parallelized */
    // For all unprocessed leaf node
    if (i >= (*status((IDX)_lNodesCur)) && i < dict_size) {
      // Parallel component
      int threadCurLeavesNum;
      // If the frequency of a leaf node <= the frequency of just two merged
      // nodes
      if (*lNodesFreq((IDX)i) <= (*status((IDX)_minFreq))) {
        // Make it start from 1
        threadCurLeavesNum = i - (*status((IDX)_lNodesCur)) + 1;
        // Atomic max -- Largest valid index
        // Find the maximum threadCurLeavesNum that satisify our condition
        // Now we find the number of leaf nodes to be parallel merged
        Atomic<int, AtomicGlobalMemory, AtomicDeviceScope, DeviceType>::Max(
            status((IDX)_curLeavesNum), threadCurLeavesNum);
      }
    }
  }

  MGARDX_EXEC void Operation4() {
    // Copy all leaf nodes to be parallel merged in a temp buffer
    if (i - (*status((IDX)_lNodesCur)) < (*status((IDX)_curLeavesNum))) {
      *copyFreq((IDX)i - (*status((IDX)_lNodesCur))) = *lNodesFreq((IDX)i);
      *copyIndex((IDX)i - (*status((IDX)_lNodesCur))) = i;
      *copyIsLeaf((IDX)i - (*status((IDX)_lNodesCur))) = 1;
    }
    // if (i == 0) {
    //   printf("prepare parallel merge. copy %d leaf nodes\n",
    //   *status((IDX)_curLeavesNum));
    // }
  }

  MGARDX_EXEC void Operation5() {
    i = (FunctorBase<DeviceType>::GetBlockIdX() *
         FunctorBase<DeviceType>::GetBlockDimX()) +
        FunctorBase<DeviceType>::GetThreadIdX();
    // if (!thread) {
    //   printf("(*status((IDX)_curLeavesNum)): %d\n",
    //   (*status((IDX)_curLeavesNum)));
    // }

    /* Updates Iterators */
    if (i == 0) {
      // All internal nodes participate in parallel merge
      (*status((IDX)_mergeRear)) = (*status((IDX)_iNodesRear));
      (*status((IDX)_mergeFront)) = (*status((IDX)_iNodesFront));

      // If total number of node to be merged are even
      // printf("total nodes to be merged: leaf(%d), internal(%d)\n",
      //         *status((IDX)_curLeavesNum), *status((IDX)_iNodesSize));
      if (((*status((IDX)_curLeavesNum)) + (*status((IDX)_iNodesSize))) % 2 ==
          0) {
        // printf("total number of nodes is even, so empty internal list since
        // all will be merged\n");
        (*status((IDX)_iNodesFront)) = (*status((IDX)_iNodesRear));
      }
      /* Odd number of nodes to merge - leave out one*/
      // If number of participating leaf node is zero OR
      // The highest frequency leaf node is less than the
      // highest frequency internal node
      else if (((*status((IDX)_iNodesSize)) != 0) and
               ((*status((IDX)_curLeavesNum)) == 0 or
                (*histogram((IDX)(*status((IDX)_lNodesCur)) +
                            (*status((IDX)_curLeavesNum))) <=
                 *iNodesFreq(
                     (IDX)MOD((*status((IDX)_iNodesRear)) - 1, dict_size))))) {
        (*status((IDX)_mergeRear)) =
            MOD((*status((IDX)_mergeRear)) - 1, dict_size);
        (*status((IDX)_iNodesFront)) =
            MOD((*status((IDX)_iNodesRear)) - 1, dict_size);
        // printf("remove one internal node from merging list. _mergeRear =
        // %d\n", *status((IDX)_mergeRear)); printf("add one to the internal
        // node list. _iNodesFront = %d\n", *status((IDX)_iNodesFront));
      } else {
        (*status((IDX)_iNodesFront)) = (*status((IDX)_iNodesRear));
        --(*status((IDX)_curLeavesNum));
        // printf("empty internal list since all will be merged\n");
        // printf("remove highest frequency leaf node from merge list.
        // _curLeavesNum = %d\n", *status((IDX)_curLeavesNum));
      }

      // Exclude all participating leaf nodes from unprocessed leaf node list
      (*status((IDX)_lNodesCur)) =
          (*status((IDX)_lNodesCur)) + (*status((IDX)_curLeavesNum));

      // printf("exclude all participating leaf nodes from unprocessed leaf node
      // list. _lNodesCur = %d\n", *status((IDX)_lNodesCur));

      // Update _iNodesRear for the new internal node (in Operation2)
      (*status((IDX)_iNodesRear)) =
          MOD((*status((IDX)_iNodesRear)) + 1, dict_size);
      // printf("include the new internal node. _iNodesRear = %d\n",
      // *status((IDX)_iNodesRear));

      // Calculate number of node participating parallel merge
      (*status((IDX)_tempLength)) =
          ((*status((IDX)_curLeavesNum)) - 0) +
          MOD((*status((IDX)_mergeRear)) - (*status((IDX)_mergeFront)),
              dict_size);
      // printf("final number of nodes for parallel merge = %d\n",
      // *status((IDX)_tempLength));
    }
  }

  MGARDX_CONT_EXEC bool BranchCondition1() {
    // If we have nodes to be parallel merged
    return (*status((IDX)_tempLength)) > 0;
  }

  MGARDX_EXEC void Operation6() {

    int32_t *sm = (int32_t *)FunctorBase<DeviceType>::GetSharedMemory();
    x_top = &sm[0];
    y_top = &sm[1];
    x_bottom = &sm[2];
    y_bottom = &sm[3];
    found = &sm[4];
    oneorzero = &sm[5];

    // (*status((IDX)_tempLength)) = (cEnd - 0) + MOD((*status((IDX)_mergeRear))
    // - (*status((IDX)_mergeFront)), dict_size); if
    // ((*status((IDX)_tempLength)) == 0) return;
    A_length = (*status((IDX)_curLeavesNum)) - 0;
    B_length = MOD((*status((IDX)_mergeRear)) - (*status((IDX)_mergeFront)),
                   dict_size);

    // if (!thread) {
    //   printf("A_length: %d, B_length: %d, (*status((IDX)_tempLength)): %d\n",
    //   A_length, B_length, (*status((IDX)_tempLength)));
    // }
    // Calculate combined index around the MergePath "matrix"
    combinedIndex = ((uint64_t)FunctorBase<DeviceType>::GetBlockIdX() *
                     ((uint64_t)A_length + (uint64_t)B_length)) /
                    (uint64_t)FunctorBase<DeviceType>::GetGridDimX();

    // if (!FunctorBase<DeviceType>::GetThreadIdX()) {
    //   printf("A_length: %d, B_length: %d, (*status((IDX)_tempLength)): %d,
    //   combinedIndex: %d\n", A_length, B_length, (*status((IDX)_tempLength)),
    //   combinedIndex);
    // }
    threadOffset =
        FunctorBase<DeviceType>::GetThreadIdX() - MGARDX_WARP_SIZE / 2;

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

  MGARDX_EXEC bool LoopCondition2() {
    // printf("%u LoopCondition2\n", FunctorBase<DeviceType>::GetBlockIdX());
    return !(*found);
  }

  MGARDX_EXEC void Operation7() {
    // Update our coordinates within the 32-wide section of the diagonal
    // if (!FunctorBase<DeviceType>::GetThreadIdX()) {
    //   printf("x %d %d y: %d %d\n", *x_top, *x_bottom, *y_top, *y_bottom);
    // }
    current_x = *x_top - ((*x_top - *x_bottom) >> 1) - threadOffset;
    current_y = *y_top + ((*y_bottom - *y_top) >> 1) + threadOffset;
    getfrom_x = current_x + 0 - 1;
    // Below statement is a more efficient, divmodless version of the following
    // int32_t getfrom_y = MOD((*status((IDX)_mergeFront)) + current_y,
    // dict_size);
    getfrom_y = (*status((IDX)_mergeFront)) + current_y;

    if (FunctorBase<DeviceType>::GetThreadIdX() < MGARDX_WARP_SIZE) {
      if (getfrom_y >= dict_size)
        getfrom_y -= dict_size;

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

  MGARDX_EXEC void Operation8() {
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
    if (FunctorBase<DeviceType>::GetThreadIdX() > 0 and                //
        FunctorBase<DeviceType>::GetThreadIdX() < MGARDX_WARP_SIZE and //
        (oneorzero[FunctorBase<DeviceType>::GetThreadIdX()] !=
         oneorzero[FunctorBase<DeviceType>::GetThreadIdX() - 1]) //
    ) {
      // printf("found\n");
      *found = 1;

      *diagonal_path_intersections(
          (IDX)FunctorBase<DeviceType>::GetBlockIdX()) = current_x;
      *diagonal_path_intersections((IDX)FunctorBase<DeviceType>::GetBlockIdX() +
                                   FunctorBase<DeviceType>::GetGridDimX() + 1) =
          current_y;
    }
  }

  MGARDX_EXEC void Operation9() {
    // Adjust the search window on the diagonal
    if (FunctorBase<DeviceType>::GetThreadIdX() == MGARDX_WARP_SIZE / 2) {
      if (oneorzero[MGARDX_WARP_SIZE - 1] != 0) {
        *x_bottom = current_x;
        *y_bottom = current_y;
      } else {
        *x_top = current_x;
        *y_top = current_y;
      }
    }
  }

  // end of loop 2

  MGARDX_EXEC void Operation10() {
    // Set the boundary diagonals (through 0,0 and A_length,B_length)
    if (FunctorBase<DeviceType>::GetThreadIdX() == 0 &&
        FunctorBase<DeviceType>::GetBlockIdX() == 0) {
      *diagonal_path_intersections((IDX)0) = 0;
      *diagonal_path_intersections((IDX)FunctorBase<DeviceType>::GetGridDimX() +
                                   1) = 0;
      *diagonal_path_intersections(
          (IDX)FunctorBase<DeviceType>::GetGridDimX()) = A_length;
      *diagonal_path_intersections((IDX)FunctorBase<DeviceType>::GetGridDimX() +
                                   FunctorBase<DeviceType>::GetGridDimX() + 1) =
          B_length;
    }
  }

  MGARDX_EXEC void Operation11() {
    if (FunctorBase<DeviceType>::GetThreadIdX() == 0) {
      // Boundaries
      int x_block_top = *diagonal_path_intersections(
          (IDX)FunctorBase<DeviceType>::GetBlockIdX());
      int y_block_top = *diagonal_path_intersections(
          (IDX)FunctorBase<DeviceType>::GetBlockIdX() +
          FunctorBase<DeviceType>::GetGridDimX() + 1);
      int x_block_stop = *diagonal_path_intersections(
          (IDX)FunctorBase<DeviceType>::GetBlockIdX() + 1);
      int y_block_stop = *diagonal_path_intersections(
          (IDX)FunctorBase<DeviceType>::GetBlockIdX() +
          FunctorBase<DeviceType>::GetGridDimX() + 2);

      // Actual indexes
      int x_start = x_block_top + 0;
      int x_end = x_block_stop + 0;
      int y_start = MOD((*status((IDX)_mergeFront)) + y_block_top, dict_size);
      int y_end = MOD((*status((IDX)_mergeFront)) + y_block_stop, dict_size);

      int offset = x_block_top + y_block_top;

      // printf("x_block_top: %d y_block_top: %d, offset: %d\n", x_block_top,
      // y_block_top, offset);

      int dummy; // Unused result
      // TODO optimize serial merging of each partition
      int len = 0;

      int iterCopy = x_start, iterINodes = y_start;

      while (iterCopy < x_end && MOD(y_end - iterINodes, dict_size) > 0) {
        if (*copyFreq((IDX)iterCopy) <= *iNodesFreq((IDX)iterINodes)) {
          *tempFreq((IDX)offset + len) = *copyFreq((IDX)iterCopy);
          *tempIndex((IDX)offset + len) = *copyIndex((IDX)iterCopy);
          *tempIsLeaf((IDX)offset + len) = *copyIsLeaf((IDX)iterCopy);
          ++iterCopy;
        } else {
          *tempFreq((IDX)offset + len) = *iNodesFreq((IDX)iterINodes);
          *tempIndex((IDX)offset + len) = iterINodes;
          *tempIsLeaf((IDX)offset + len) = 0;
          iterINodes = MOD(iterINodes + 1, dict_size);
        }
        ++len;
      }

      while (iterCopy < x_end) {
        *tempFreq((IDX)offset + len) = *copyFreq((IDX)iterCopy);
        *tempIndex((IDX)offset + len) = *copyIndex((IDX)iterCopy);
        *tempIsLeaf((IDX)offset + len) = *copyIsLeaf((IDX)iterCopy);
        ++iterCopy;
        ++len;
      }
      while (MOD(y_end - iterINodes, dict_size) > 0) {
        *tempFreq((IDX)offset + len) = *iNodesFreq((IDX)iterINodes);
        *tempIndex((IDX)offset + len) = iterINodes;
        *tempIsLeaf((IDX)offset + len) = 0;
        iterINodes = MOD(iterINodes + 1, dict_size);
        ++len;
      }

      // for (int i = 0; i < len; i++) {
      //   if (*tempIsLeaf((IDX)offset+i) == 2 ) {
      //     printf("*copyIsLeaf((IDX)iterCopy) = 2\n");
      //   }
      // }

      // if (FunctorBase<DeviceType>::GetThreadIdX() == 0) {
      //   printf("FunctorBase<DeviceType>::GetGridDimX(): %llu, offset: %d,
      //   len: %d\n", FunctorBase<DeviceType>::GetGridDimX(), offset, len);
      //   printf("leaf: %d %d\n", *tempIsLeaf((IDX)2 * 4), *tempIsLeaf((IDX)2 *
      //   4 + 1));
      // }
    }

    // cg::this_grid().sync();
    // if (FunctorBase<DeviceType>::GetThreadIdX() == 0) {
    //   // printf("FunctorBase<DeviceType>::GetGridDimX(): %u, offset: %d, len:
    //   %d\n", FunctorBase<DeviceType>::GetGridDimX(), offset, len);
    //   printf("leaf: %d %d\n", *tempIsLeaf((IDX)2 * 4), *tempIsLeaf((IDX)2 * 4
    //   + 1));
    // }
  }

  // end of parallel merge

  MGARDX_EXEC void Operation12() {
    i = (FunctorBase<DeviceType>::GetBlockIdX() *
         FunctorBase<DeviceType>::GetBlockDimX()) +
        FunctorBase<DeviceType>::GetThreadIdX();
    // if (thread == 0) {
    //   printf("leaf: %d %d\n", *tempIsLeaf((IDX)2 * 4), *tempIsLeaf((IDX)2 * 4
    //   + 1));
    // }
    /* Melding phase -- New */
    // if (i == 0) {
    //   printf("parallel merge\n");
    // }
    if (i < (*status((IDX)_tempLength)) / 2) {
      // Index of new internal nodes
      int ind = MOD((*status((IDX)_iNodesRear)) + i, dict_size);
      // printf("Melding(i=%d): %u(%d) %u(%d)\n", i, *tempFreq((IDX)2 * i),
      // *tempIsLeaf((IDX)2 * i), *tempFreq((IDX)2 * i + 1), *tempIsLeaf((IDX)2
      // * i + 1));
      // Calculate the combined frequency of the new internel node
      *iNodesFreq((IDX)ind) = *tempFreq((IDX)2 * i) + *tempFreq((IDX)2 * i + 1);
      // New internel node should not have leader
      *iNodesLeader((IDX)ind) = -1;

      // If a merge node is leaf node, increase CL by 1
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

  MGARDX_EXEC void Operation13() {
    i = (FunctorBase<DeviceType>::GetBlockIdX() *
         FunctorBase<DeviceType>::GetBlockDimX()) +
        FunctorBase<DeviceType>::GetThreadIdX();
    if (i == 0) {
      // printf("update _iNodesRear to include new internel nodes\n");
      (*status((IDX)_iNodesRear)) =
          MOD((*status((IDX)_iNodesRear)) + ((*status((IDX)_tempLength)) / 2),
              dict_size);
    }
  }

  MGARDX_EXEC void Operation14() {
    i = (FunctorBase<DeviceType>::GetBlockIdX() *
         FunctorBase<DeviceType>::GetBlockDimX()) +
        FunctorBase<DeviceType>::GetThreadIdX();
    /* Update leaders */
    // if (thread == 0) {
    //   printf("mine: iNodesLeader(0.leader) = %d, (*status((IDX)_iNodesRear)):
    //   %u\n", *iNodesLeader(IDX(0)), (*status((IDX)_iNodesRear)));
    // }
    if (i < dict_size) {
      // For all leaf node that is already merged, we check if we need to update
      // their leader and CL
      if (*lNodesLeader((IDX)i) != -1) {
        // If the previous leader had leader (new highest leader)
        if (*iNodesLeader((IDX)*lNodesLeader((IDX)i)) != -1) {
          // We keep doing this to make sure to track highest leader
          // After this '*lNodesLeader((IDX)i)' should always points to the
          // highest leader
          *lNodesLeader((IDX)i) = *iNodesLeader((IDX)*lNodesLeader((IDX)i));
          // If we  update leader, it means we have a new leader result from a
          // merge so, we need to update CL
          ++(*CL((IDX)i));
          // printf("update CL(%d):%d\n", i, *CL(i));
        }
      }
    }
  }

  MGARDX_EXEC void Operation15() {
    i = (FunctorBase<DeviceType>::GetBlockIdX() *
         FunctorBase<DeviceType>::GetBlockDimX()) +
        FunctorBase<DeviceType>::GetThreadIdX();
    if (i == 0) {
      (*status((IDX)_iNodesSize)) =
          MOD((*status((IDX)_iNodesRear)) - (*status((IDX)_iNodesFront)),
              dict_size);
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t sm_size = 0;
    sm_size += 5 * sizeof(int32_t);
    sm_size += DeviceRuntime<DeviceType>::GetWarpSize() * sizeof(int32_t);
    return sm_size;
  }

private:
  SubArray<1, T, DeviceType> histogram;
  SubArray<1, T, DeviceType> CL;
  int dict_size;
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
  SubArray<1, int, DeviceType> status;

  int32_t *x_top;
  int32_t *y_top;
  int32_t *x_bottom;
  int32_t *y_bottom;
  int32_t *found;
  int32_t *oneorzero;

  unsigned int i;

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
class GenerateCLKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "generate codeword length";
  MGARDX_CONT
  GenerateCLKernel(
      SubArray<1, T, DeviceType> histogram, SubArray<1, T, DeviceType> CL,
      int dict_size,
      /* Global Arrays */
      SubArray<1, T, DeviceType> lNodesFreq,
      SubArray<1, int, DeviceType> lNodesLeader,
      SubArray<1, T, DeviceType> iNodesFreq,
      SubArray<1, int, DeviceType> iNodesLeader,
      SubArray<1, T, DeviceType> tempFreq,
      SubArray<1, int, DeviceType> tempIsLeaf,
      SubArray<1, int, DeviceType> tempIndex,
      SubArray<1, T, DeviceType> copyFreq,
      SubArray<1, int, DeviceType> copyIsLeaf,
      SubArray<1, int, DeviceType> copyIndex,
      SubArray<1, uint32_t, DeviceType> diagonal_path_intersections,
      SubArray<1, int, DeviceType> status)
      : histogram(histogram), CL(CL), dict_size(dict_size),
        lNodesFreq(lNodesFreq), lNodesLeader(lNodesLeader),
        iNodesFreq(iNodesFreq), iNodesLeader(iNodesLeader), tempFreq(tempFreq),
        tempIsLeaf(tempIsLeaf), tempIndex(tempIndex), copyFreq(copyFreq),
        copyIsLeaf(copyIsLeaf), copyIndex(copyIndex),
        diagonal_path_intersections(diagonal_path_intersections),
        status(status) {}

  MGARDX_CONT
  Task<GenerateCLFunctor<T, DeviceType>> GenTask(int queue_idx) {
    using FunctorType = GenerateCLFunctor<T, DeviceType>;
    FunctorType Functor(histogram, CL, dict_size, lNodesFreq, lNodesLeader,
                        iNodesFreq, iNodesLeader, tempFreq, tempIsLeaf,
                        tempIndex, copyFreq, copyIsLeaf, copyIndex,
                        diagonal_path_intersections, status);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = Functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = DeviceRuntime<DeviceType>::GetWarpSize();

    int cg_blocks_sm =
        DeviceRuntime<DeviceType>::GetOccupancyMaxActiveBlocksPerSM(
            Functor, tbx, sm_size);
    int cg_mblocks = cg_blocks_sm * DeviceRuntime<DeviceType>::GetNumSMs();

    gridz = 1;
    gridy = 1;
    gridx = cg_mblocks;

    int tthreads = tbx * gridx;
    if (tthreads >= dict_size) {
      if (DeviceRuntime<DeviceType>::PrintKernelConfig) {
        std::cout << log::log_info << "GenerateCL: using Cooperative Groups\n";
      }
      Functor.use_CG = true;
    } else {
      if (DeviceRuntime<DeviceType>::PrintKernelConfig) {
        std::cout << log::log_info
                  << "GenerateCL: not using Cooperative Groups\n";
      }
      Functor.use_CG = false;
    }
    gridx = (dict_size - 1) / tbx + 1;

    return Task(Functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<1, T, DeviceType> histogram;
  SubArray<1, T, DeviceType> CL;
  int dict_size;
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
  SubArray<1, int, DeviceType> status;
};

#undef MOD
#undef MIN
#undef MAX

} // namespace mgard_x

#endif
