/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_GENERATE_CW_TEMPLATE_HPP
#define MGARD_X_GENERATE_CW_TEMPLATE_HPP

#include "../../RuntimeX/RuntimeX.h"

namespace mgard_x {

// GenerateCW Locals
MGARDX_EXEC int CCL;
MGARDX_EXEC int CDPI;
MGARDX_EXEC int newCDPI;

template <typename T, typename H, typename DeviceType>
class GenerateCWFunctor: public HuffmanCWCustomizedFunctor<DeviceType> {
  public:
  MGARDX_CONT GenerateCWFunctor(){}
  MGARDX_CONT GenerateCWFunctor(SubArray<1, T, DeviceType> CL,
                                SubArray<1, H, DeviceType> CW,
                                SubArray<1, H, DeviceType> first,
                                SubArray<1, H, DeviceType> entry,
                                SIZE size):
                                CL(CL), CW(CW), first(first), entry(entry), size(size) {
    HuffmanCWCustomizedFunctor<DeviceType>();                            
  }

  MGARDX_EXEC void
  Operation1() {
    thread = (FunctorBase<DeviceType>::GetBlockIdX() * FunctorBase<DeviceType>::GetBlockDimX()) + FunctorBase<DeviceType>::GetThreadIdX();
    i = thread; // Porting convenience
    type_bw = sizeof(H) * 8;

    /* Reverse in place - Probably a more CUDA-appropriate way */
    if (thread < size / 2) {
      T temp = *CL((IDX)i);
      *CL((IDX)i) = *CL((IDX)size - i - 1);
      *CL((IDX)size - i - 1) = temp;
    }
  }

  MGARDX_EXEC void
  Operation2() {
    if (thread == 0) {
      CCL = *CL((IDX)0);
      CDPI = 0;
      newCDPI = size - 1;
      *entry((IDX)CCL) = 0;

      // Edge case -- only one input symbol
      *CW((IDX)CDPI) = 0;
      *first((IDX)CCL) = *CW((IDX)CDPI) ^ (((H)1 << (H)*CL((IDX)CDPI)) - 1);
      *entry((IDX)CCL + 1) = 1;
    }
  }

  MGARDX_EXEC void
  Operation3() {
    // Initialize first and entry arrays
    if (thread < CCL) {
      // Initialization of first to Max ensures that unused code
      // lengths are skipped over in decoding.
      *first((IDX)i) = std::numeric_limits<H>::max();
      *entry((IDX)i) = 0;
    }
  }

  MGARDX_EXEC bool
  LoopCondition1() {
    // if (! thread)
      // printf("thread: %u, CDPI: %d, newCDPI: %d, size: %u\n", thread, CDPI, newCDPI, size);
    return CDPI < size - 1;
  }

  MGARDX_EXEC void
  Operation4() {
    // CDPI update
    if (i < size - 1 && *CL((IDX)i + 1) > CCL) {
      Atomic<DeviceType>::Min(&newCDPI, (int)i);
    }
  }

  MGARDX_EXEC void
  Operation5() {
    // Last element to update
    updateEnd = (newCDPI >= size - 1) ? type_bw : *CL((IDX)newCDPI + 1);
    // Fill base
    curEntryVal = *entry((IDX)CCL);
    // Number of elements of length CCL
    numCCL = (newCDPI - CDPI + 1);

    // Get first codeword
    if (i == 0) {
      if (CDPI == 0) {
        *CW((IDX)newCDPI) = 0;
      } else {
        *CW((IDX)newCDPI) = *CW((IDX)CDPI); // Pre-stored
      }
    }
  }

  MGARDX_EXEC void
  Operation6() {
    if (i < size) {
      // Parallel canonical codeword generation
      if (i >= CDPI && i < newCDPI) {
        *CW((IDX)i) = *CW((IDX)newCDPI) + (newCDPI - i);
      }
    }

    // Update entry and first arrays in O(1) time
    // Jieyang: not useful?
    if (thread > CCL && thread < updateEnd) {
      *entry((IDX)i) = curEntryVal + numCCL;
    }
    // Add number of entries to next CCL
    if (thread == 0) {
      if (updateEnd < type_bw) {
        *entry((IDX)updateEnd) = curEntryVal + numCCL;
      }
    }
  }

  MGARDX_EXEC void
  Operation7() {
    // Update first array in O(1) time
    if (thread == CCL) {
      // Flip least significant CL[CDPI] bits
      *first((IDX)CCL) = *CW((IDX)CDPI) ^ (((H)1 << (H)*CL((IDX)CDPI)) - 1);
      // printf("first[%d]: %llu\n", CCL, first[CCL]);
    }
    if (thread > CCL && thread < updateEnd) {
      *first((IDX)i) = std::numeric_limits<H>::max();
    }
  }

  MGARDX_EXEC void
  Operation8() {
    if (thread == 0) {
      if (newCDPI < size - 1) {
        int CLDiff = *CL((IDX)newCDPI + 1) - *CL((IDX)newCDPI);
        // Add and shift -- Next canonical code
      
        *CW((IDX)newCDPI + 1) = ((*CW((IDX)CDPI) + 1) << CLDiff);
        CCL = *CL((IDX)newCDPI + 1);

        H temp = (*CW((IDX)CDPI) + 1);

        // printf("CLDiff: %d  %llu <- %llu\n", CLDiff, (H)(temp << CLDiff), temp);

        ++newCDPI;
      }

      // Update CDPI to newCDPI after codeword length increase
      CDPI = newCDPI;
      newCDPI = size - 1;
      // printf("thread: %u, CDPI: %d, newCDPI: %d, size: %u\n", thread, CDPI, newCDPI, size);
    }
  }

  MGARDX_EXEC void
  Operation9() {
    // encoding CL into CW (highest 8 bits)
    if (thread < size) {
      *CW((IDX)i) = (*CW((IDX)i) | (((H)*CL((IDX)i) & (H)0xffu) << ((sizeof(H) * 8) - 8))) ^
              (((H)1 << (H)*CL((IDX)i)) - 1);
      // printf("flip: %llu ^ 1 << %u -> %llu\n", *CW((IDX)i), *CL((IDX)i), (*CW((IDX)i)) ^ (((H)1 << (H)*CL((IDX)i)) - 1) );
      // *CW((IDX)i) = (*CW((IDX)i)) ^
      //         (((H)1 << (H)*CL((IDX)i)) - 1);
    }
  }

  MGARDX_EXEC void
  Operation10() {
    /* Reverse partial codebook */

    if (thread < size / 2) {
      // printf("Reverse\n");
      H temp = *CW((IDX)i);
      *CW((IDX)i) = *CW((IDX)size - i - 1);
      *CW((IDX)size - i - 1) = temp;
    }
  }

  MGARDX_CONT size_t
  shared_memory_size() { return 0; }

  private:
  SubArray<1, T, DeviceType> CL;
  SubArray<1, H, DeviceType> CW;
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


template <typename T, typename H, typename DeviceType>
class GenerateCW: public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  GenerateCW():AutoTuner<DeviceType>() {}

  MGARDX_CONT
  Task<GenerateCWFunctor<T, H, DeviceType> > 
  GenTask(SubArray<1, T, DeviceType> CL,
          SubArray<1, H, DeviceType> CW,
          SubArray<1, H, DeviceType> first,
          SubArray<1, H, DeviceType> entry,
          SIZE dict_size, int queue_idx) {
    using FunctorType = GenerateCWFunctor<T, H, DeviceType>;
    FunctorType Functor(CL, CW, first, entry, dict_size);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = Functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = DeviceRuntime<DeviceType>::GetMaxNumThreadsPerTB();

    int nz_nblocks = (dict_size / tbx) + 1;
    int cg_blocks_sm = DeviceRuntime<DeviceType>::GetOccupancyMaxActiveBlocksPerSM(Functor, tbx, sm_size);
    int cg_mblocks = cg_blocks_sm * DeviceRuntime<DeviceType>::GetNumSMs();
    cg_mblocks = std::min(nz_nblocks, cg_mblocks);
    gridz = 1;
    gridy = 1;
    gridx = cg_mblocks;

    int cw_tthreads = gridx * tbx;
    if (cw_tthreads < dict_size) {
      std::cout << log::log_err << "Insufficient on-device parallelism to construct a "
           << dict_size << " non-zero item codebook" << std::endl;
      std::cout << log::log_err << "Provided parallelism: " << gridx << " blocks, "
           << tbx << " threads, " << cw_tthreads << " total" << std::endl
           << std::endl;
      exit(1);
    }

        // printf("gridx: %d, tbx: %d\n", gridx, tbx);


    // printf("%u %u %u\n", shape.dataHost()[2], shape.dataHost()[1], shape.dataHost()[0]);
    // PrintSubarray("shape", shape);
    return Task(Functor, gridz, gridy, gridx, 
                tbz, tby, tbx, sm_size, queue_idx, "GenerateCW"); 
  }

  MGARDX_CONT
  void Execute(SubArray<1, T, DeviceType> CL,
                SubArray<1, H, DeviceType> CW,
                SubArray<1, H, DeviceType> first,
                SubArray<1, H, DeviceType> entry,
                SIZE dict_size, int queue_idx) {
    using FunctorType = GenerateCWFunctor<T, H, DeviceType>;
    using TaskType = Task<FunctorType>;
    TaskType task = GenTask(CL, CW, first, entry, dict_size, queue_idx); 
    DeviceAdapter<TaskType, DeviceType> adapter; 
    adapter.Execute(task);
  }
};

}

#endif