/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_GENERATE_CW_TEMPLATE_HPP
#define MGARD_X_GENERATE_CW_TEMPLATE_HPP

#include "../../RuntimeX/RuntimeX.h"

namespace mgard_x {

// GenerateCW Locals
#define _CCL 0
#define _CDPI 1
#define _newCDPI 2

#define _updateEnd 3
#define _curEntryVal 4
#define _numCCL 5

template <typename T, typename H, typename DeviceType>
class GenerateCWFunctor : public HuffmanCWCustomizedFunctor<DeviceType> {
public:
  MGARDX_CONT GenerateCWFunctor() {}
  MGARDX_CONT GenerateCWFunctor(SubArray<1, T, DeviceType> CL,
                                SubArray<1, H, DeviceType> CW,
                                SubArray<1, H, DeviceType> first,
                                SubArray<1, H, DeviceType> entry, SIZE size,
                                SubArray<1, int, DeviceType> status)
      : CL(CL), CW(CW), first(first), entry(entry), size(size), status(status) {
    HuffmanCWCustomizedFunctor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    i = (FunctorBase<DeviceType>::GetBlockIdX() *
         FunctorBase<DeviceType>::GetBlockDimX()) +
        FunctorBase<DeviceType>::GetThreadIdX();
    // i = thread; // Porting convenience
    type_bw = sizeof(H) * 8;

    /* Reverse in place - Probably a more CUDA-appropriate way */
    if (i < size / 2) {
      T temp = *CL((IDX)i);
      *CL((IDX)i) = *CL((IDX)size - i - 1);
      *CL((IDX)size - i - 1) = temp;
    }
  }

  MGARDX_EXEC void Operation2() {
    i = (FunctorBase<DeviceType>::GetBlockIdX() *
         FunctorBase<DeviceType>::GetBlockDimX()) +
        FunctorBase<DeviceType>::GetThreadIdX();
    if (i == 0) {
      (*status((IDX)_CCL)) = *CL((IDX)0);
      (*status((IDX)_CDPI)) = 0;
      (*status((IDX)_newCDPI)) = size - 1;
      *entry((IDX)(*status((IDX)_CCL))) = 0;

      // Edge case -- only one input symbol
      *CW((IDX)(*status((IDX)_CDPI))) = 0;
      *first((IDX)(*status((IDX)_CCL))) =
          *CW((IDX)(*status((IDX)_CDPI))) ^
          (((H)1 << (H)*CL((IDX)(*status((IDX)_CDPI)))) - 1);
      *entry((IDX)(*status((IDX)_CCL)) + 1) = 1;
    }
  }

  MGARDX_EXEC void Operation3() {
    i = (FunctorBase<DeviceType>::GetBlockIdX() *
         FunctorBase<DeviceType>::GetBlockDimX()) +
        FunctorBase<DeviceType>::GetThreadIdX();
    // Initialize first and entry arrays
    if (i < (*status((IDX)_CCL))) {
      // Initialization of first to Max ensures that unused code
      // lengths are skipped over in decoding.
      *first((IDX)i) = std::numeric_limits<H>::max();
      *entry((IDX)i) = 0;
    }
  }

  MGARDX_CONT_EXEC bool LoopCondition1() {
    // if (! thread)
    // printf("thread: %u, (*status((IDX)_CDPI)): %d, (*status((IDX)_newCDPI)):
    // %d, size: %u\n", thread, (*status((IDX)_CDPI)),
    // (*status((IDX)_newCDPI)), size);
    return (*status((IDX)_CDPI)) < size - 1;
  }

  MGARDX_EXEC void Operation4() {
    i = (FunctorBase<DeviceType>::GetBlockIdX() *
         FunctorBase<DeviceType>::GetBlockDimX()) +
        FunctorBase<DeviceType>::GetThreadIdX();
    // (*status((IDX)_CDPI)) update
    if (i < size - 1 && *CL((IDX)i + 1) > (*status((IDX)_CCL))) {
      Atomic<DeviceType>::Min(&(*status((IDX)_newCDPI)), (int)i);
    }
  }

  MGARDX_EXEC void Operation5() {
    i = (FunctorBase<DeviceType>::GetBlockIdX() *
         FunctorBase<DeviceType>::GetBlockDimX()) +
        FunctorBase<DeviceType>::GetThreadIdX();
    type_bw = sizeof(H) * 8;
    // Last element to update
    (*status((IDX)_updateEnd)) = ((*status((IDX)_newCDPI)) >= size - 1)
                                     ? type_bw
                                     : *CL((IDX)(*status((IDX)_newCDPI)) + 1);
    // Fill base
    (*status((IDX)_curEntryVal)) = *entry((IDX)(*status((IDX)_CCL)));
    // Number of elements of length (*status((IDX)_CCL))
    (*status((IDX)_numCCL)) =
        ((*status((IDX)_newCDPI)) - (*status((IDX)_CDPI)) + 1);

    // Get first codeword
    if (i == 0) {
      if ((*status((IDX)_CDPI)) == 0) {
        *CW((IDX)(*status((IDX)_newCDPI))) = 0;
      } else {
        *CW((IDX)(*status((IDX)_newCDPI))) =
            *CW((IDX)(*status((IDX)_CDPI))); // Pre-stored
      }
    }
  }

  MGARDX_EXEC void Operation6() {
    i = (FunctorBase<DeviceType>::GetBlockIdX() *
         FunctorBase<DeviceType>::GetBlockDimX()) +
        FunctorBase<DeviceType>::GetThreadIdX();
    type_bw = sizeof(H) * 8;
    if (i < size) {
      // Parallel canonical codeword generation
      if (i >= (*status((IDX)_CDPI)) && i < (*status((IDX)_newCDPI))) {
        *CW((IDX)i) =
            *CW((IDX)(*status((IDX)_newCDPI))) + ((*status((IDX)_newCDPI)) - i);
      }
    }

    // Update entry and first arrays in O(1) time
    // Jieyang: not useful?
    if (i > (*status((IDX)_CCL)) && i < (*status((IDX)_updateEnd))) {
      *entry((IDX)i) = (*status((IDX)_curEntryVal)) + (*status((IDX)_numCCL));
    }
    // Add number of entries to next (*status((IDX)_CCL))
    if (i == 0) {
      if ((*status((IDX)_updateEnd)) < type_bw) {
        *entry((IDX)(*status((IDX)_updateEnd))) =
            (*status((IDX)_curEntryVal)) + (*status((IDX)_numCCL));
      }
    }
  }

  MGARDX_EXEC void Operation7() {
    i = (FunctorBase<DeviceType>::GetBlockIdX() *
         FunctorBase<DeviceType>::GetBlockDimX()) +
        FunctorBase<DeviceType>::GetThreadIdX();
    // Update first array in O(1) time
    if (i == (*status((IDX)_CCL))) {
      // Flip least significant CL[(*status((IDX)_CDPI))] bits
      *first((IDX)(*status((IDX)_CCL))) =
          *CW((IDX)(*status((IDX)_CDPI))) ^
          (((H)1 << (H)*CL((IDX)(*status((IDX)_CDPI)))) - 1);
      // printf("first[%d]: %llu\n", (*status((IDX)_CCL)),
      // first[(*status((IDX)_CCL))]);
    }
    if (i > (*status((IDX)_CCL)) && i < (*status((IDX)_updateEnd))) {
      *first((IDX)i) = std::numeric_limits<H>::max();
    }
  }

  MGARDX_EXEC void Operation8() {
    i = (FunctorBase<DeviceType>::GetBlockIdX() *
         FunctorBase<DeviceType>::GetBlockDimX()) +
        FunctorBase<DeviceType>::GetThreadIdX();
    if (i == 0) {
      if ((*status((IDX)_newCDPI)) < size - 1) {
        int CLDiff = *CL((IDX)(*status((IDX)_newCDPI)) + 1) -
                     *CL((IDX)(*status((IDX)_newCDPI)));
        // Add and shift -- Next canonical code

        *CW((IDX)(*status((IDX)_newCDPI)) + 1) =
            ((*CW((IDX)(*status((IDX)_CDPI))) + 1) << CLDiff);
        (*status((IDX)_CCL)) = *CL((IDX)(*status((IDX)_newCDPI)) + 1);

        H temp = (*CW((IDX)(*status((IDX)_CDPI))) + 1);

        // printf("CLDiff: %d  %llu <- %llu\n", CLDiff, (H)(temp << CLDiff),
        // temp);

        ++(*status((IDX)_newCDPI));
      }

      // Update (*status((IDX)_CDPI)) to (*status((IDX)_newCDPI)) after codeword
      // length increase
      (*status((IDX)_CDPI)) = (*status((IDX)_newCDPI));
      (*status((IDX)_newCDPI)) = size - 1;
      // printf("thread: %u, (*status((IDX)_CDPI)): %d,
      // (*status((IDX)_newCDPI)): %d, size: %u\n", thread,
      // (*status((IDX)_CDPI)),
      // (*status((IDX)_newCDPI)), size);
    }
  }

  MGARDX_EXEC void Operation9() {
    i = (FunctorBase<DeviceType>::GetBlockIdX() *
         FunctorBase<DeviceType>::GetBlockDimX()) +
        FunctorBase<DeviceType>::GetThreadIdX();
    // encoding CL into CW (highest 8 bits)
    if (i < size) {
      *CW((IDX)i) = (*CW((IDX)i) |
                     (((H)*CL((IDX)i) & (H)0xffu) << ((sizeof(H) * 8) - 8))) ^
                    (((H)1 << (H)*CL((IDX)i)) - 1);
      // printf("flip: %llu ^ 1 << %u -> %llu\n", *CW((IDX)i), *CL((IDX)i),
      // (*CW((IDX)i)) ^ (((H)1 << (H)*CL((IDX)i)) - 1) ); *CW((IDX)i) =
      // (*CW((IDX)i)) ^
      //         (((H)1 << (H)*CL((IDX)i)) - 1);
    }
  }

  MGARDX_EXEC void Operation10() {
    i = (FunctorBase<DeviceType>::GetBlockIdX() *
         FunctorBase<DeviceType>::GetBlockDimX()) +
        FunctorBase<DeviceType>::GetThreadIdX();
    /* Reverse partial codebook */

    if (i < size / 2) {
      // printf("Reverse\n");
      H temp = *CW((IDX)i);
      *CW((IDX)i) = *CW((IDX)size - i - 1);
      *CW((IDX)size - i - 1) = temp;
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, T, DeviceType> CL;
  SubArray<1, H, DeviceType> CW;
  SubArray<1, H, DeviceType> first;
  SubArray<1, H, DeviceType> entry;
  SubArray<1, int, DeviceType> status;
  SIZE size;

  // unsigned int thread;
  unsigned int i;
  size_t type_bw;

  // int (*status((IDX)_updateEnd));
  // int (*status((IDX)_curEntryVal));
  // int (*status((IDX)_numCCL));
};

template <typename T, typename H, typename DeviceType>
class GenerateCW : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  GenerateCW() : AutoTuner<DeviceType>() {}

  MGARDX_CONT
  Task<GenerateCWFunctor<T, H, DeviceType>>
  GenTask(SubArray<1, T, DeviceType> CL, SubArray<1, H, DeviceType> CW,
          SubArray<1, H, DeviceType> first, SubArray<1, H, DeviceType> entry,
          SIZE dict_size, SubArray<1, int, DeviceType> status, int queue_idx) {
    using FunctorType = GenerateCWFunctor<T, H, DeviceType>;
    FunctorType Functor(CL, CW, first, entry, dict_size, status);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = Functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = DeviceRuntime<DeviceType>::GetMaxNumThreadsPerTB();

    int nz_nblocks = (dict_size / tbx) + 1;
    int cg_blocks_sm =
        DeviceRuntime<DeviceType>::GetOccupancyMaxActiveBlocksPerSM(
            Functor, tbx, sm_size);
    int cg_mblocks = cg_blocks_sm * DeviceRuntime<DeviceType>::GetNumSMs();
    cg_mblocks = cg_mblocks; // std::min(nz_nblocks, cg_mblocks);
    gridz = 1;
    gridy = 1;
    gridx = cg_mblocks;

    int cw_tthreads = gridx * tbx;
    if (cw_tthreads >= dict_size) {
      if (DeviceRuntime<DeviceType>::PrintKernelConfig) {
        std::cout << log::log_info << "GenerateCW: using Cooperative Groups\n";
      }
      Functor.use_CG = true;
    } else {
      if (DeviceRuntime<DeviceType>::PrintKernelConfig) {
        std::cout << log::log_info
                  << "GenerateCW: not using Cooperative Groups\n";
      }
      Functor.use_CG = false;
      gridx = (dict_size - 1) / tbx + 1;
    }

    return Task(Functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "GenerateCW");
  }

  MGARDX_CONT
  void Execute(SubArray<1, T, DeviceType> CL, SubArray<1, H, DeviceType> CW,
               SubArray<1, H, DeviceType> first,
               SubArray<1, H, DeviceType> entry, SIZE dict_size,
               int queue_idx) {
    Array<1, int, DeviceType> status({(SIZE)16}, false, true);
    using FunctorType = GenerateCWFunctor<T, H, DeviceType>;
    using TaskType = Task<FunctorType>;
    TaskType task =
        GenTask(CL, CW, first, entry, dict_size, SubArray(status), queue_idx);
    DeviceAdapter<TaskType, DeviceType> adapter;
    adapter.Execute(task);
  }
};

} // namespace mgard_x

#endif