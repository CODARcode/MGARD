/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#ifndef MGRAD_CUDA_GENERATE_CW_TEMPLATE_HPP
#define MGRAD_CUDA_GENERATE_CW_TEMPLATE_HPP

#include "../CommonInternal.h"

namespace mgard_cuda {

// GenerateCW Locals
__device__ int CCL;
__device__ int CDPI;
__device__ int newCDPI;

template <typename T, typename H, typename DeviceType>
class GenerateCWFunctor: public HuffmanCWCustomizedFunctor<DeviceType> {
  public:
  MGARDm_CONT GenerateCWFunctor(SubArray<1, T, DeviceType> CL,
                                SubArray<1, H, DeviceType> first,
                                SubArray<1, H, DeviceType> entry,
                                SIZE size):
                                CL(CL), first(first), entry(entry), size(size) {
    HuffmanCWCustomizedFunctor<DeviceType>();                            
  }

  MGARDm_EXEC void
  Operation1() {
    thread = (blockIdx.x * blockDim.x) + threadIdx.x;
    i = thread; // Porting convenience
    type_bw = sizeof(H) * 8;

    /* Reverse in place - Probably a more CUDA-appropriate way */
    if (thread < size / 2) {
      T temp = *CL(i);
      *CL(i) = *CL(size - i - 1);
      *CL(size - i - 1) = temp;
    }
  }

  MGARDm_EXEC void
  Operation2() {
    if (thread == 0) {
      CCL = *CL(0);
      CDPI = 0;
      newCDPI = size - 1;
      *entry(CCL) = 0;

      // Edge case -- only one input symbol
      *CW(CDPI) = 0;
      *first(CCL) = *CW(CDPI) ^ (((H)1 << (H)*CL(CDPI)) - 1);
      *entry(CCL + 1) = 1;
    }
  }

  MGARDm_EXEC void
  Operation3() {
    // Initialize first and entry arrays
    if (thread < CCL) {
      // Initialization of first to Max ensures that unused code
      // lengths are skipped over in decoding.
      *first(i) = std::numeric_limits<H>::max();
      *entry(i) = 0;
    }
  }

  MGARDm_EXEC bool
  LoopCondition1() {
    return CDPI < size - 1;
  }

  MGARDm_EXEC void
  Operation4() {
    // CDPI update
    if (i < size - 1 && *CL(i + 1) > CCL) {
      Atomic<DeviceType>::Min(&newCDPI, i);
    }
  }

  MGARDm_EXEC void
  Operation5() {
    // Last element to update
    updateEnd = (newCDPI >= size - 1) ? type_bw : *CL(newCDPI + 1);
    // Fill base
    curEntryVal = *entry(CCL);
    // Number of elements of length CCL
    numCCL = (newCDPI - CDPI + 1);

    // Get first codeword
    if (i == 0) {
      if (CDPI == 0) {
        *CW(newCDPI) = 0;
      } else {
        *CW(newCDPI) = *CW(CDPI); // Pre-stored
      }
    }
  }

  MGARDm_EXEC void
  Operation6() {
    if (i < size) {
      // Parallel canonical codeword generation
      if (i >= CDPI && i < newCDPI) {
        *CW(i) = *CW(newCDPI) + (newCDPI - i);
      }
    }

    // Update entry and first arrays in O(1) time
    // Jieyang: not useful?
    if (thread > CCL && thread < updateEnd) {
      *entry(i) = curEntryVal + numCCL;
    }
    // Add number of entries to next CCL
    if (thread == 0) {
      if (updateEnd < type_bw) {
        *entry(updateEnd) = curEntryVal + numCCL;
      }
    }
  }

  MGARDm_EXEC void
  Operation7() {
    // Update first array in O(1) time
    if (thread == CCL) {
      // Flip least significant CL[CDPI] bits
      *first(CCL) = *CW(CDPI) ^ (((H)1 << (H)*CL(CDPI)) - 1);
      // printf("first[%d]: %llu\n", CCL, first[CCL]);
    }
    if (thread > CCL && thread < updateEnd) {
      *first(i) = std::numeric_limits<H>::max();
    }
  }

  MGARDm_EXEC void
  Operation8() {
    if (thread == 0) {
      if (newCDPI < size - 1) {
        int CLDiff = *CL(newCDPI + 1) - *CL(newCDPI);
        // Add and shift -- Next canonical code
        *CW(newCDPI + 1) = ((*CW(CDPI) + 1) << CLDiff);
        CCL = *CL(newCDPI + 1);

        ++newCDPI;
      }

      // Update CDPI to newCDPI after codeword length increase
      CDPI = newCDPI;
      newCDPI = size - 1;
    }
  }

  MGARDm_EXEC void
  Operation9() {
    // encoding CL into CW (highest 8 bits)
    if (thread < size) {
      *CW(i) = (*CW(i) | (((H)*CL(i) & (H)0xffu) << ((sizeof(H) * 8) - 8))) ^
              (((H)1 << (H)*CL(i)) - 1);
    }
  }

  MGARDm_EXEC void
  Operation10() {
    /* Reverse partial codebook */
    if (thread < size / 2) {
      H temp = *CW(i);
      *CW(i) = *CW(size - i - 1);
      *CW(size - i - 1) = temp;
    }
  }

  MGARDm_CONT size_t
  shared_memory_size() { return 0; }

  private:
  SubArray<1, T, DeviceType> CL;
  SubArray<1, H, DeviceType> first;
  SubArray<1, H, DeviceType> entry;
  SIZE size;

  unsigned int thread;
  unsigned int i;
  size_t type_bw;

  int updateEnd;
  int curEntryVal;
  int numCCL;
};


template <typename T, typename DeviceType>
class GenerateCW: public AutoTuner<DeviceType> {
public:
  MGARDm_CONT
  GenerateCW():AutoTuner<DeviceType>() {}

  MGARDm_CONT
  Task<GenerateCWFunctor<T, DeviceType> > 
  GenTask(SubArray<1, T, DeviceType> CL,
          SubArray<1, H, DeviceType> first,
          SubArray<1, H, DeviceType> entry,
          SIZE dict_size, int queue_idx) {
    using FunctorType = GenerateCWFunctor<T, DeviceType>;
    FunctorType functor(CL, first, entry, dict_size, queue_idx);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = DeviceRuntime<DeviceType>::GetMaxNumThreadsPerSM();

    int cg_cw_mblocks = (cg_mblocks * mthreads) / DeviceRuntime<DeviceType>::GetMaxNumThreadsPerSM();

    gridz = 1;
    gridy = 1;
    gridx = (dict_size / tbx) + 1;
    // printf("%u %u %u\n", shape.dataHost()[2], shape.dataHost()[1], shape.dataHost()[0]);
    // PrintSubarray("shape", shape);
    return Task(functor, gridz, gridy, gridx, 
                tbz, tby, tbx, sm_size, queue_idx); 
  }

  MGARDm_CONT
  void Execute(SubArray<1, T, DeviceType> CL,
                SubArray<1, H, DeviceType> first,
                SubArray<1, H, DeviceType> entry,
                SIZE dict_size, int queue_idx) {
    using FunctorType = GenerateCWFunctor<T, DeviceType>;
    using TaskType = Task<FunctorType>;
    TaskType task = GenTask(CL, first, entry, dict_size, queue_idx); 
    DeviceAdapter<TaskType, DeviceType> adapter; 
    adapter.Execute(task);
  }
};

}

#endif