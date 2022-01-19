/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_FUNCTOR
#define MGARD_X_FUNCTOR

namespace mgard_x {

template <typename DeviceType>
class MGARDX_ALIGN(16) FunctorBase {
  public:
  MGARDX_CONT
  FunctorBase() {}
  MGARDX_EXEC void
  Init(THREAD_IDX ngridz, THREAD_IDX ngridy, THREAD_IDX ngridx,
       THREAD_IDX nblockz, THREAD_IDX nblocky, THREAD_IDX nblockx,
       THREAD_IDX blockz, THREAD_IDX blocky, THREAD_IDX blockx,
       THREAD_IDX threadz, THREAD_IDX thready, THREAD_IDX threadx, Byte * shared_memory) {
    this->ngridz = ngridz; this->ngridy = ngridy; this->ngridx = ngridx;
    this->nblockz = nblockz; this->nblocky = nblocky; this->nblockx = nblockx;
    this->blockz = blockz; this->blocky = blocky; this->blockx = blockx;
    this->threadz = threadz; this->thready = thready; this->threadx = threadx;
    this->shared_memory = shared_memory;
  }

  MGARDX_EXEC void
  InitConfig(THREAD_IDX ngridz, THREAD_IDX ngridy, THREAD_IDX ngridx,
              THREAD_IDX nblockz, THREAD_IDX nblocky, THREAD_IDX nblockx) {
    this->ngridz = ngridz; this->ngridy = ngridy; this->ngridx = ngridx;
    this->nblockz = nblockz; this->nblocky = nblocky; this->nblockx = nblockx;
  }

  MGARDX_EXEC void
  InitBlockId(THREAD_IDX blockz, THREAD_IDX blocky, THREAD_IDX blockx) {
    this->blockz = blockz; this->blocky = blocky; this->blockx = blockx;
  }

  MGARDX_EXEC void
  InitThreadId(THREAD_IDX threadz, THREAD_IDX thready, THREAD_IDX threadx) {
    this->threadz = threadz; this->thready = thready; this->threadx = threadx;
  }

  MGARDX_EXEC void
  InitSharedMemory(Byte * shared_memory) {
    this->shared_memory = shared_memory;
  }

  MGARDX_EXEC
  const THREAD_IDX& GetThreadIdX() const { return threadx; }

  MGARDX_EXEC
  const THREAD_IDX& GetThreadIdY() const { return thready; }

  MGARDX_EXEC
  const THREAD_IDX& GetThreadIdZ() const { return threadz; }

  MGARDX_EXEC
  const THREAD_IDX& GetBlockDimX() const { return nblockx; }

  MGARDX_EXEC
  const THREAD_IDX& GetBlockDimY() const { return nblocky; }

  MGARDX_EXEC
  const THREAD_IDX& GetBlockDimZ() const { return nblockz; }

  MGARDX_EXEC
  const THREAD_IDX& GetBlockIdX() const { return blockx; }

  MGARDX_EXEC
  const THREAD_IDX& GetBlockIdY() const { return blocky; }

  MGARDX_EXEC
  const THREAD_IDX& GetBlockIdZ() const { return blockz; }

  MGARDX_EXEC
  const THREAD_IDX& GetGridDimX() const { return ngridx; }

  MGARDX_EXEC
  const THREAD_IDX& GetGridDimY() const { return ngridy; }

  MGARDX_EXEC
  const THREAD_IDX& GetGridDimZ() const { return ngridz; }

  MGARDX_EXEC
  Byte * GetSharedMemory() { return shared_memory; }

  private:
  Byte * shared_memory;
  THREAD_IDX threadz, thready, threadx;
  THREAD_IDX blockz, blocky, blockx;
  THREAD_IDX ngridz, ngridy, ngridx;
  THREAD_IDX nblockz, nblocky, nblockx;

};

template <typename DeviceType>
class MGARDX_ALIGN(16) Functor: public FunctorBase<DeviceType> {
public:

  MGARDX_EXEC void
  Operation1() {}

  MGARDX_EXEC void
  Operation2() {}

  MGARDX_EXEC void
  Operation3() {}

  MGARDX_EXEC void
  Operation4() {}

  MGARDX_EXEC void
  Operation5() {}

  MGARDX_EXEC void
  Operation6() {}

  MGARDX_EXEC void
  Operation7() {}

  MGARDX_EXEC void
  Operation8() {}

  MGARDX_EXEC void
  Operation9() {}

  MGARDX_EXEC void
  Operation10() {}
};

template <typename DeviceType>
class IterFunctor: public FunctorBase<DeviceType> {
public:
  MGARDX_EXEC bool
  LoopCondition1() { return false; }

  MGARDX_EXEC bool
  LoopCondition2() {return false; }

  MGARDX_EXEC void
  Operation1() {}

  MGARDX_EXEC void
  Operation2() {}

  MGARDX_EXEC void
  Operation3() {}

  MGARDX_EXEC void
  Operation4() {}

  MGARDX_EXEC void
  Operation5() {}

  MGARDX_EXEC void
  Operation6() {}

  MGARDX_EXEC void
  Operation7() {}

  MGARDX_EXEC void
  Operation8() {}

  MGARDX_EXEC void
  Operation9() {}

  MGARDX_EXEC void
  Operation10() {}

  MGARDX_EXEC void
  Operation11() {}

  MGARDX_EXEC void
  Operation12() {}

  MGARDX_EXEC void
  Operation13() {}

  MGARDX_EXEC void
  Operation14() {}

  MGARDX_EXEC void
  Operation15() {}

  MGARDX_EXEC void
  Operation16() {}

  MGARDX_EXEC void
  Operation17() {}
};


template <typename DeviceType>
class HuffmanCLCustomizedFunctor: public FunctorBase<DeviceType> {
public:

  MGARDX_EXEC void
  Operation1(); //init

  MGARDX_EXEC bool
  LoopCondition1(); // global loop

  MGARDX_EXEC void
  Operation2(); // combine two nodes

  MGARDX_EXEC void
  Operation3(); // copy to temp

  MGARDX_EXEC void
  Operation4(); // updatre iterator

  MGARDX_EXEC void
  Operation5(); // parallel merge: diagonal devide: init

  MGARDX_EXEC bool
  LoopCondition2(); // parallel merge: diagonal devide: found

  MGARDX_EXEC void
  Operation6(); // parallel merge: diagonal devide: generate 0/1

  MGARDX_EXEC void
  Operation7(); // parallel merge: diagonal devide: check 0/1

  MGARDX_EXEC void
  Operation8(); // parallel merge: diagonal devide: adjust window

  //end of loop 2

  MGARDX_EXEC void
  Operation9(); // parallel merge: diagonal devide: boundary cases

  MGARDX_EXEC void
  Operation10(); // parallel merge: merge path
  
  // end of parallel merge

  MGARDX_EXEC void
  Operation11(); // meld

  MGARDX_EXEC void
  Operation12(); // update iNodeRear

  MGARDX_EXEC void
  Operation13(); // update leaders

  MGARDX_EXEC void
  Operation14(); // update iNodesSize

  bool use_CG;
};

template <typename DeviceType>
class HuffmanCWCustomizedFunctor: public FunctorBase<DeviceType> {
public:

  MGARDX_EXEC void
  Operation1();

  MGARDX_EXEC void
  Operation2();

  MGARDX_EXEC void
  Operation3();

  MGARDX_EXEC bool
  LoopCondition1(); // global loop

  MGARDX_EXEC void
  Operation4();

  MGARDX_EXEC void
  Operation5();

  MGARDX_EXEC bool
  BranchCondition1();

  MGARDX_EXEC void
  Operation6();

  MGARDX_EXEC void
  Operation7();

  MGARDX_EXEC void
  Operation8();

  //end of loop

  MGARDX_EXEC void
  Operation9();

  MGARDX_EXEC void
  Operation10();

  bool use_CG;

};


}

#endif