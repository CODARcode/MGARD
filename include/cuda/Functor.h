/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#ifndef MGARD_CUDA_FUNCTOR
#define MGARD_CUDA_FUNCTOR

namespace mgard_cuda {

template <typename DeviceType>
class MGARDm_ALIGN(16) Functor {
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
  Init_config(IDX ngridz, IDX ngridy, IDX ngridx,
              IDX nblockz, IDX nblocky, IDX nblockx) {
    this->ngridz = ngridz; this->ngridy = ngridy; this->ngridx = ngridx;
    this->nblockz = nblockz; this->nblocky = nblocky; this->nblockx = nblockx;
  }

  MGARDm_EXEC void
  Init_block_id(IDX blockz, IDX blocky, IDX blockx) {
    this->blockz = blockz; this->blocky = blocky; this->blockx = blockx;
  }

  MGARDm_EXEC void
  Init_thread_id(IDX threadz, IDX thready, IDX threadx) {
    this->threadz = threadz; this->thready = thready; this->threadx = threadx;
  }

  MGARDm_EXEC void
  Init_shared_memory(Byte * shared_memory) {
    this->shared_memory = shared_memory;
  }

  MGARDm_EXEC void
  Operation1() {}

  MGARDm_EXEC void
  Operation2() {}

  MGARDm_EXEC void
  Operation3() {}

  MGARDm_EXEC void
  Operation4() {}

  MGARDm_EXEC void
  Operation5() {}

  MGARDm_EXEC void
  Operation6() {}

  MGARDm_EXEC void
  Operation7() {}

  MGARDm_EXEC void
  Operation8() {}

  MGARDm_EXEC void
  Operation9() {}

  MGARDm_EXEC void
  Operation10() {}

  IDX ngridz, ngridy, ngridx;
  IDX nblockz, nblocky, nblockx;
  IDX blockz, blocky, blockx;
  IDX threadz, thready, threadx;
  Byte * shared_memory;
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

  MGARDm_EXEC void
  Init_config(IDX ngridz, IDX ngridy, IDX ngridx,
              IDX nblockz, IDX nblocky, IDX nblockx) {
    this->ngridz = ngridz; this->ngridy = ngridy; this->ngridx = ngridx;
    this->nblockz = nblockz; this->nblocky = nblocky; this->nblockx = nblockx;
  }

  MGARDm_EXEC void
  Init_block_id(IDX blockz, IDX blocky, IDX blockx) {
    this->blockz = blockz; this->blocky = blocky; this->blockx = blockx;
  }

  MGARDm_EXEC void
  Init_thread_id(IDX threadz, IDX thready, IDX threadx) {
    this->threadz = threadz; this->thready = thready; this->threadx = threadx;
  }

  MGARDm_EXEC void
  Init_shared_memory(Byte * shared_memory) {
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
};


template <typename DeviceType>
class HuffmanCLCustomizedFunctor {
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
  Init_config(IDX ngridz, IDX ngridy, IDX ngridx,
              IDX nblockz, IDX nblocky, IDX nblockx) {
    this->ngridz = ngridz; this->ngridy = ngridy; this->ngridx = ngridx;
    this->nblockz = nblockz; this->nblocky = nblocky; this->nblockx = nblockx;
  }

  MGARDm_EXEC void
  Init_block_id(IDX blockz, IDX blocky, IDX blockx) {
    this->blockz = blockz; this->blocky = blocky; this->blockx = blockx;
  }

  MGARDm_EXEC void
  Init_thread_id(IDX threadz, IDX thready, IDX threadx) {
    this->threadz = threadz; this->thready = thready; this->threadx = threadx;
  }

  MGARDm_EXEC void
  Init_shared_memory(Byte * shared_memory) {
    this->shared_memory = shared_memory;
  }

  MGARDm_EXEC void
  Operation1(); //init

  MGARDm_EXEC bool
  LoopCondition1(); // global loop

  MGARDm_EXEC void
  Operation2(); // combine two nodes

  MGARDm_EXEC void
  Operation3(); // copy to temp

  MGARDm_EXEC void
  Operation4(); // updatre iterator

  MGARDm_EXEC void
  Operation5(); // parallel merge: diagonal devide: init

  MGARDm_EXEC bool
  LoopCondition2(); // parallel merge: diagonal devide: found

  MGARDm_EXEC void
  Operation6(); // parallel merge: diagonal devide: generate 0/1

  MGARDm_EXEC void
  Operation7(); // parallel merge: diagonal devide: check 0/1

  MGARDm_EXEC void
  Operation8(); // parallel merge: diagonal devide: adjust window

  //end of loop 2

  MGARDm_EXEC void
  Operation9(); // parallel merge: diagonal devide: boundary cases

  MGARDm_EXEC void
  Operation10(); // parallel merge: merge path
  
  // end of parallel merge

  MGARDm_EXEC void
  Operation11(); // meld

  MGARDm_EXEC void
  Operation12(); // update iNodeRear

  MGARDm_EXEC void
  Operation13(); // update leaders

  MGARDm_EXEC void
  Operation14(); // update iNodesSize

  IDX ngridz, ngridy, ngridx;
  IDX nblockz, nblocky, nblockx;
  IDX blockz, blocky, blockx;
  IDX threadz, thready, threadx;
  Byte * shared_memory;
};

template <typename DeviceType>
class HuffmanCWCustomizedFunctor {
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
  Init_config(IDX ngridz, IDX ngridy, IDX ngridx,
              IDX nblockz, IDX nblocky, IDX nblockx) {
    this->ngridz = ngridz; this->ngridy = ngridy; this->ngridx = ngridx;
    this->nblockz = nblockz; this->nblocky = nblocky; this->nblockx = nblockx;
  }

  MGARDm_EXEC void
  Init_block_id(IDX blockz, IDX blocky, IDX blockx) {
    this->blockz = blockz; this->blocky = blocky; this->blockx = blockx;
  }

  MGARDm_EXEC void
  Init_thread_id(IDX threadz, IDX thready, IDX threadx) {
    this->threadz = threadz; this->thready = thready; this->threadx = threadx;
  }

  MGARDm_EXEC void
  Init_shared_memory(Byte * shared_memory) {
    this->shared_memory = shared_memory;
  }

  MGARDm_EXEC void
  Operation1();

  MGARDm_EXEC void
  Operation2();

  MGARDm_EXEC void
  Operation3();

  MGARDm_EXEC bool
  LoopCondition1(); // global loop

  MGARDm_EXEC void
  Operation4();

  MGARDm_EXEC void
  Operation5();

  MGARDm_EXEC bool
  BranchCondition1();

  MGARDm_EXEC void
  Operation6();

  MGARDm_EXEC void
  Operation7();

  MGARDm_EXEC void
  Operation8();

  //end of loop

  MGARDm_EXEC void
  Operation9();

  MGARDm_EXEC void
  Operation10();

  IDX ngridz, ngridy, ngridx;
  IDX nblockz, nblocky, nblockx;
  IDX blockz, blocky, blockx;
  IDX threadz, thready, threadx;
  Byte * shared_memory;
};


}

#endif