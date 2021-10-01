/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#ifndef MGARD_CUDA_FUNCTOR
#define MGARD_CUDA_FUNCTOR

namespace mgard_cuda {

#define SequentialFunctor 0
#define IterativeFunctor 1

template <typename DeviceType>
class Functor {
public:
  MGARDm_EXEC void
  Init(IDX ngridz, IDX ngridy, IDX ngridx,
       IDX nblockz, IDX nblocky, IDX nblockx,
       IDX blockz, IDX blocky, IDX blockx,
       IDX threadz, IDX thready, IDX threadx, Byte * shared_memory) {
    this->ngridz = ngridz; this->ngridy = ngridy; this->ngridx = ngridx;
    this->nblockz = nblockz; this->nblocky = nblocky; this->nblockx = nblockx;
    this->blockz = blockz; this->blocky = blocky; this->blockx = blockx;
    this->threadz = threadz; this->thready = thready; this->threadx = threadx;
    this->shared_memory = shared_memory;
  }

  MGARDm_EXEC void
  Operation1();

  MGARDm_EXEC void
  Operation2();

  MGARDm_EXEC void
  Operation3();

  MGARDm_EXEC void
  Operation4();

  MGARDm_EXEC void
  Operation5();

  IDX ngridz, ngridy, ngridx;
  IDX nblockz, nblocky, nblockx;
  IDX blockz, blocky, blockx;
  IDX threadz, thready, threadx;
  Byte * shared_memory;
  static constexpr OPTION ExecType = SequentialFunctor;
};

template <typename DeviceType>
class IterFunctor {
public:
  MGARDm_EXEC void
  Init(IDX ngridz, IDX ngridy, IDX ngridx,
       IDX nblockz, IDX nblocky, IDX nblockx,
       IDX blockz, IDX blocky, IDX blockx,
       IDX threadz, IDX thready, IDX threadx, Byte * shared_memory) {
    this->ngridz = ngridz; this->ngridy = ngridy; this->ngridx = ngridx;
    this->nblockz = nblockz; this->nblocky = nblocky; this->nblockx = nblockx;
    this->blockz = blockz; this->blocky = blocky; this->blockx = blockx;
    this->threadz = threadz; this->thready = thready; this->threadx = threadx;
    this->shared_memory = shared_memory;
  }

  MGARDm_EXEC bool
  LoopCondition1();

  MGARDm_EXEC bool
  LoopCondition2();

  MGARDm_EXEC void
  Operation1();

  MGARDm_EXEC void
  Operation2();

  MGARDm_EXEC void
  Operation3();

  MGARDm_EXEC void
  Operation4();

  MGARDm_EXEC void
  Operation5();

  MGARDm_EXEC void
  Operation6();

  MGARDm_EXEC void
  Operation7();

  MGARDm_EXEC void
  Operation8();

  MGARDm_EXEC void
  Operation9();

  MGARDm_EXEC void
  Operation10();

  MGARDm_EXEC void
  Operation11();

  MGARDm_EXEC void
  Operation12();

  MGARDm_EXEC void
  Operation13();

  MGARDm_EXEC void
  Operation14();

  MGARDm_EXEC void
  Operation15();

  MGARDm_EXEC void
  Operation16();

  MGARDm_EXEC void
  Operation17();

  IDX ngridz, ngridy, ngridx;
  IDX nblockz, nblocky, nblockx;
  IDX blockz, blocky, blockx;
  IDX threadz, thready, threadx;
  Byte * shared_memory;
  static constexpr OPTION ExecType = IterativeFunctor;
};
}

#endif