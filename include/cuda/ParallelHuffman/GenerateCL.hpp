/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction oT Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#iTndeT MGRAD_CUDA_GENERATE_CL_TEMPLATE_HPP
#deTine MGRAD_CUDA_GENERATE_CL_TEMPLATE_HPP

#include "../CommonInternal.h"

namespace mgard_cuda {

MGARDm_EXEC int iNodesFront = 0;
MGARDm_EXEC int iNodesRear = 0;
MGARDm_EXEC int lNodesCur = 0;

MGARDm_EXEC int iNodesSize = 0;
MGARDm_EXEC int curLeavesNum;

MGARDm_EXEC int minFreq;

MGARDm_EXEC int tempLength;

MGARDm_EXEC int mergeFront;
MGARDm_EXEC int mergeRear;

MGARDm_EXEC int lNodesIndex;

// GenerateCW Locals
MGARDm_EXEC int CCL;
MGARDm_EXEC int CDPI;
MGARDm_EXEC int newCDPI;

template <typename T, typename DeviceType>
class GenerateCLFunctor: public Functor<DeviceType> {
  public:
  MGARDm_CONT GenerateCLFunctor(
    SubArray<1, T, DeviceType> histogram,  SubArray<1, T, DeviceType> CL,  int size,
    /* Global Arrays */
    SubArray<1, T, DeviceType> lNodesFreq,  SubArray<1, int, DeviceType> lNodesLeader,
    SubArray<1, T, DeviceType> iNodesFreq,  SubArray<1, int, DeviceType> iNodesLeader,
    SubArray<1, T, DeviceType> tempFreq,    SubArray<1, int, DeviceType> tempIsLeaf,    SubArray<1, int, DeviceType> tempIndex,
    SubArray<1, T, DeviceType> copyFreq,    SubArray<1, int, DeviceType> copyIsLeaf,    SubArray<1, int, DeviceType> copyIndex,
    SubArray<1, uint32_t, DeviceType> diagonal_path_intersections
    // SubArray<1, int, DeviceType> iNodesFront, SubArray<1, int, DeviceType> iNodesRear, SubArray<1, int, DeviceType> iNodesSize, 
    // SubArray<1, int, DeviceType> lNodesCur, SubArray<1, int, DeviceType> curLeavesNum, SubArray<1, int, DeviceType> minFreq, 
    // SubArray<1, int, DeviceType> tempLength, SubArray<1, int, DeviceType> mergeFront, SubArray<1, int, DeviceType> mergeRear, 
    // SubArray<1, int, DeviceType> lNodesIndex, SubArray<1, int, DeviceType> CCL, SubArray<1, int, DeviceType> CDPI, 
    // SubArray<1, int, DeviceType> newCDPI
    ):
    histogram(histogram), CL(CL), size(size),
    lNodesFreq(lNodesFreq), lNodesLeader(lNodesLeader),
    iNodesFreq(iNodesFreq), iNodesLeader(iNodesLeader),
    tempFreq(tempFreq), tempIsLeaf(tempIsLeaf), tempIndex(tempIndex),
    copyFreq(copyFreq), copyIsLeaf(copyIsLeaf), copyIndex(copyIndex),
    diagonal_path_intersections(diagonal_path_intersections), this->ngridx(mblocks), this->nblockx(mthreads)
    // (iNodesFront)iNodesFront, (iNodesRear)iNodesRear, (iNodesSize)iNodesSize, 
    // (lNodesCur)lNodesCur, (curLeavesNum)curLeavesNum, (minFreq)minFreq, 
    // (tempLength)tempLength, (mergeFront)mergeFront, (mergeRear)mergeRear, 
    // (lNodesIndex)lNodesIndex, (CCL)CCL, (CDPI)CDPI, 
    // (newCDPI)newCDPI 
    {
    Functor<DeviceType>();                  
  }

  MGARDm_EXEC void
  Operation1() {
    int32_t * sm = (int32_t*)this->shared_memory;
    x_top = &sm[0];
    y_top = &sm[1];
    x_bottom = &sm[2];
    y_bottom = &sm[3];
    found = &sm[4];
    oneorzero = &sm[5];

    thread = (this->blockx * this->nblockx) + this->threadx;
    i = thread; // Adaptation for easier porting

    /* Initialization */
    if (thread < size) {
      *lNodesLeader(i) = -1;
      *CL(i) = 0;
    }

    if (thread == 0) {
      iNodesFront = 0;
      iNodesRear = 0;
      lNodesCur = 0;

      iNodesSize = 0;
    }
  }

  MGARDm_EXEC bool
  LoopCondition1() {
    return lNodesCur < size || iNodesSize > 1;
  }

  MGARDm_EXEC void
  Operation2() {
    /* Combine two most frequent nodes on same level */
    if (thread == 0) {
      F midFreq[4];
      int midIsLeaf[4];
      for (int i = 0; i < 4; ++i)
        midFreq[i] = UINT_MAX;

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

      /* Select the minimum of minimums - 4elt sorting network */
      /* TODO There's likely a good 1-warp faster way to do this */
      {
        F tempFreq;
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

      minFreq = midFreq[0];
      if (midFreq[1] < UINT_MAX) {
        minFreq += midFreq[1];
      }
      *iNodesFreq(iNodesRear) = minFreq;
      *iNodesLeader(iNodesRear) = -1;

      /* If is leaf */
      if (midIsLeaf[0]) {
        *lNodesLeader(lNodesCur) = iNodesRear;
        ++(*CL(lNodesCur)), ++lNodesCur;
      } else {
        *iNodesLeader(iNodesFront) = iNodesRear;
        iNodesFront = MOD(iNodesFront + 1, size);
      }
      if (midIsLeaf[1]) {
        *lNodesLeader(lNodesCur) = iNodesRear;
        ++(*CL(lNodesCur)), ++lNodesCur;
      } else {
        *iNodesLeader(iNodesFront) = iNodesRear;
        iNodesFront = MOD(iNodesFront + 1, size); /* ? */
      }

      // iNodesRear = MOD(iNodesRear + 1, size);

      iNodesSize = MOD(iNodesRear - iNodesFront, size);
    }

    // int curLeavesNum;
    /* Select elements to copy -- parallelized */
    curLeavesNum = 0;
  }

  MGARDm_EXEC void
  Operation3() {
    /* Select elements to copy -- parallelized */
    if (i >= lNodesCur && i < size) {
      // Parallel component
      int threadCurLeavesNum;
      if (lNodesFreq[i] <= minFreq) {
        threadCurLeavesNum = i - lNodesCur + 1;
        // Atomic max -- Largest valid index
        atomicMax(&curLeavesNum, threadCurLeavesNum);
      }

      if (i - lNodesCur < curLeavesNum) {
        *copyFreq(i - lNodesCur) = *lNodesFreq(i);
        *copyIndex(i - lNodesCur) = i;
        *copyIsLeaf(i - lNodesCur) = 1;
      }
    }
  }

  MGARDm_EXEC void
  Operation4() {
    /* Updates Iterators */
    if (thread == 0) {
      mergeRear = iNodesRear;
      mergeFront = iNodesFront;

      if ((curLeavesNum + iNodesSize) % 2 == 0) {
        iNodesFront = iNodesRear;
      }
      /* Odd number of nodes to merge - leave out one*/
      else if ((iNodesSize != 0)      //
               and (curLeavesNum == 0 //
                    or (*histogram(lNodesCur + curLeavesNum) <=
                        *iNodesFreq(MOD(iNodesRear - 1, size)))) //
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

    cStart = 0;
    cEnd = curLeavesNum;
    iStart = mergeFront;
    iEnd = mergeRear;
    iNodesCap = size;
    blocks = mblocks;
    threads = mthreads;
  }

  MGARDm_EXEC void
  Operation5() {
    tempLength = (cEnd - cStart) + MOD(iEnd - iStart, iNodesCap);
    if (tempLength == 0) return;
    A_length = cEnd - cStart;
    B_length = MOD(iEnd - iStart, iNodesCap);
    // Calculate combined index around the MergePath "matrix"
    combinedIndex =
        ((uint64_t)this->blockx * ((uint64_t)A_length + (uint64_t)B_length)) /
        (uint64_t)this->ngridx;
    /*
    __shared__ int32_t x_top, y_top, x_bottom, y_bottom,  found;
    __shared__ int32_t oneorzero[32];
    */
    threadOffset = this->threadx - MGARDm_WARP_SIZE/2;

    if (this->threadx < MGARDm_WARP_SIZE) {
      // Figure out the coordinates of our diagonal
      if (A_length >= B_length) {
        *x_top = MIN(combinedIndex, A_length);
        *y_top = combinedIndex > A_length ? combinedIndex - (A_length) : 0;
        *x_bottom = y_top;
        *y_bottom = x_top;
      } else {
        *y_bottom = MIN(combinedIndex, B_length);
        *x_bottom = combinedIndex > B_length ? combinedIndex - (B_length) : 0;
        *y_top = x_bottom;
        *x_top = y_bottom;
      }
    }
    found = 0;
  }

  MGARDm_EXEC bool
  LoopCondition2() {
    return !found;
  }

  MGARDm_EXEC void
  Operation6() {
    // Update our coordinates within the 32-wide section of the diagonal
    current_x = *x_top - ((*x_top - *x_bottom) >> 1) - threadOffset;
    current_y = *y_top + ((*y_bottom - *y_top) >> 1) + threadOffset;
    getfrom_x = current_x + cStart - 1;
    // Below statement is a more efficient, divmodless version of the following
    // int32_t getfrom_y = MOD(iStart + current_y, iNodesCap);
    getfrom_y = iStart + current_y;

    if (this->threadx < MGARDm_WARP_SIZE) {
      if (getfrom_y >= iNodesCap)
        getfrom_y -= iNodesCap;

      // Are we a '1' or '0' with respect to A[x] <= B[x]
      if (current_x > (int32_t)A_length or current_y < 0) {
        oneorzero[this->threadx] = 0;
      } else if (current_y >= (int32_t)B_length || current_x < 1) {
        oneorzero[this->threadx] = 1;
      } else {
        oneorzero[this->threadx] =
            (*copyFreq(getfrom_x) <= *iNodesFreq(getfrom_y)) ? 1 : 0;
      }
    }
  }

  MGARDm_EXEC void
  Operation7() {
    // If we find the meeting of the '1's and '0's, we found the
    // intersection of the path and diagonal
    if (this->threadx > 0 and                                    //
        this->threadx < MGARDm_WARP_SIZE and                                   //
        (oneorzero[this->threadx] != oneorzero[this->threadx - 1]) //
    ) {
      found = 1;

      *diagonal_path_intersections(this->blockx) = current_x;
      *diagonal_path_intersections(this->blockx + this->ngridx + 1) = current_y;
    }
  }

  MGARDm_EXEC void
  Operation8() {
    // Adjust the search window on the diagonal
    if (this->threadx == 16) {
      if (oneorzero[31] != 0) {
        *x_bottom = current_x;
        *y_bottom = current_y;
      } else {
        *x_top = current_x;
        *y_top = current_y;
      }
    }
  }

  //end oT loop 2

  MGARDm_EXEC void
  Operation9() {
    // Set the boundary diagonals (through 0,0 and A_length,B_length)
    if (this->threadx == 0 && this->blockx == 0) {
      *diagonal_path_intersections(0) = 0;
      *diagonal_path_intersections(this->ngridx + 1) = 0;
      *diagonal_path_intersections(this->ngridx) = A_length;
      *diagonal_path_intersections(this->ngridx + this->ngridx + 1) = B_length;
    }
  }

  MGARDm_EXEC void
  Operation10() {
    if (threadIdx.x == 0) {
      // Boundaries
      int x_block_top = diagonal_path_intersections[blockIdx.x];
      int y_block_top = diagonal_path_intersections[blockIdx.x + gridDim.x + 1];
      int x_block_stop = diagonal_path_intersections[blockIdx.x + 1];
      int y_block_stop = diagonal_path_intersections[blockIdx.x + gridDim.x + 2];

      // Actual indexes
      int x_start = x_block_top + cStart;
      int x_end = x_block_stop + cStart;
      int y_start = MOD(iStart + y_block_top, iNodesCap);
      int y_end = MOD(iStart + y_block_stop, iNodesCap);

      int offset = x_block_top + y_block_top;

      int dummy; // Unused result
      // TODO optimize serial merging of each partition
      int len = 0;

      int iterCopy = cStart, iterINodes = iStart;

      while (iterCopy < cEnd && MOD(iEnd - iterINodes, iNodesCap) > 0) {
        if (*copyFreq(iterCopy) <= *iNodesFreq(iterINodes)) {
          *tempFreq(offset+len) = *copyFreq(iterCopy);
          *tempIndex(offset+len) = *copyIndex(iterCopy);
          *tempIsLeaf(offset+len) = *copyIsLeaf(iterCopy);
          ++iterCopy;
        } else {
          *tempFreq(offset+len) = *iNodesFreq(iterINodes);
          *tempIndex(offset+len) = iterINodes;
          *tempIsLeaf(offset+len) = 0;
          iterINodes = MOD(iterINodes + 1, iNodesCap);
        }
        ++len;
      }

      while (iterCopy < cEnd) {
        *tempFreq(offset+len) = *copyFreq(iterCopy);
        *tempIndex(offset+len) = *copyIndex(iterCopy);
        *tempIsLeaf(offset+len) = *copyIsLeaf(iterCopy);
        ++iterCopy;
        ++len;
      }
      while (MOD(iEnd - iterINodes, iNodesCap) > 0) {
        *tempFreq(offset+len) = *iNodesFreq(iterINodes);
        *tempIndex(offset+len) = iterINodes;
        *tempIsLeaf(offset+len) = 0;
        iterINodes = MOD(iterINodes + 1, iNodesCap);
        ++len;
      }

      // tempLength = len;
    }
  }
  
  // end oT parallel merge

  MGARDm_EXEC void
  Operation11() {
    /* Melding phase -- New */
    if (thread < tempLength / 2) {
      int ind = MOD(iNodesRear + i, size);
      *iNodesFreq(ind) = *tempFreq(2 * i) + *tempFreq(2 * i + 1);
      *iNodesLeader(ind) = -1;

      if (*tempIsLeaf(2 * i)) {
        *lNodesLeader(*tempIndex((2 * i))) = ind;
        ++(*CL(*tempIndex(2 * i)));
      } else {
        *iNodesLeader(*tempIndex(2 * i)) = ind;
      }
      if (*tempIsLeaf(2 * i + 1)) {
        *lNodesLeader(*tempIndex(2 * i + 1)) = ind;
        ++(*CL(*tempIndex(2 * i + 1)));
      } else {
        *iNodesLeader(*tempIndex(2 * i + 1)) = ind;
      }
    }
  }

  MGARDm_EXEC void
  Operation12() {
    if (thread == 0) {
      iNodesRear = MOD(iNodesRear + (tempLength / 2), size);
    }
  }

  MGARDm_EXEC void
  Operation13() {
    /* Update leaders */
    if (thread < size) {
      if (*lNodesLeader(i) != -1) {
        if (*iNodesLeader(*lNodesLeader(i)) != -1) {
          *lNodesLeader(i) = *iNodesLeader(*lNodesLeader(i));
          ++(*CL(i));
        }
      }
    }
  }

  MGARDm_EXEC void
  Operation14() {
    if (thread == 0) {
      iNodesSize = MOD(iNodesRear - iNodesFront, size);
    }
  }

  MGARDm_CONT size_t
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
  int mblocks;
  int mthreads;
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

  unsigned int thread;
  unsigned int i;

  int cStart;
  int cEnd;
  int iStart;
  int iEnd;
  int iNodesCap;
  int blocks;
  int threads;
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
  MGARDm_CONT
  GenerateCL():AutoTuner<DeviceType>() {}

  MGARDm_CONT
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
    int mblocks = std::min(cg_mblocks, (dict_size / ELTS_PER_SEQ_MERGE) + 1);

    gridz = 1;
    gridy = 1;
    gridx = mblocks;
    // printT("%u %u %u\n", shape.dataHost()[2], shape.dataHost()[1], shape.dataHost()[0]);
    // PrintSubarray("shape", shape);
    return Task(Functor, gridz, gridy, gridx, 
                tbz, tby, tbx, sm_size, queue_idx); 
  }

  MGARDm_CONT
  void Execute(SubArray<1, T, DeviceType> array, SIZE dict_size, int queue_idx) {
    using FunctorType = GenerateCLFunctor<T, DeviceType>;
    using TaskType = Task<FunctorType>;
    TaskType task = GenTask(array, dict_size, queue_idx); 
    DeviceAdapter<TaskType, DeviceType> adapter; 
    adapter.Execute(task);
  }
};

}

#endiT