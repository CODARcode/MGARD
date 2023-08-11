/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */
#ifndef MGARD_X_TASK
#define MGARD_X_TASK

#include "../Functors/Functor.h"

namespace mgard_x {

template <typename FunctorType> class Task {
public:
  MGARDX_CONT
  Task() {}

  MGARDX_CONT
  Task(FunctorType functor, IDX ngridz, IDX ngridy, IDX ngridx, IDX nblockz,
       IDX nblocky, IDX nblockx, SIZE shared_memory_size, int queue_idx)
      : functor(functor), ngridz(ngridz), ngridy(ngridy), ngridx(ngridx),
        nblockz(nblockz), nblocky(nblocky), nblockx(nblockx),
        shared_memory_size(shared_memory_size), queue_idx(queue_idx) {}

  MGARDX_CONT
  Task(FunctorType functor, IDX ngridz, IDX ngridy, IDX ngridx, IDX nblockz,
       IDX nblocky, IDX nblockx, SIZE shared_memory_size, int queue_idx,
       std::string functor_name)
      : functor(functor), ngridz(ngridz), ngridy(ngridy), ngridx(ngridx),
        nblockz(nblockz), nblocky(nblocky), nblockx(nblockx),
        shared_memory_size(shared_memory_size), queue_idx(queue_idx),
        functor_name(functor_name) {}

  // copy contructure does not work in device so return by reference
  MGARDX_CONT_EXEC FunctorType &GetFunctor() { return functor; }
  MGARDX_CONT int GetQueueIdx() { return queue_idx; }
  MGARDX_CONT_EXEC IDX GetGridDimZ() { return ngridz; }
  MGARDX_CONT_EXEC IDX GetGridDimY() { return ngridy; }
  MGARDX_CONT_EXEC IDX GetGridDimX() { return ngridx; }
  MGARDX_CONT_EXEC IDX GetBlockDimZ() { return nblockz; }
  MGARDX_CONT_EXEC IDX GetBlockDimY() { return nblocky; }
  MGARDX_CONT_EXEC IDX GetBlockDimX() { return nblockx; }
  MGARDX_CONT_EXEC SIZE GetSharedMemorySize() { return shared_memory_size; }
  MGARDX_CONT void SetFunctorName(std::string functor_name) {
    this->functor_name = functor_name;
  }
  MGARDX_CONT std::string GetFunctorName() { return this->functor_name; }
  using Functor = FunctorType;

private:
  FunctorType functor;
  IDX ngridz, ngridy, ngridx;
  IDX nblockz, nblocky, nblockx;
  SIZE shared_memory_size;
  int queue_idx;
  std::string functor_name;
};

// template <typename FunctorType, typename T_reduce>
// class ReduceTask {
// public:
//   MGARDX_CONT
//   ReduceTask(int num_items, SubArray<1, T_reduce> d_in, SubArray<1, T_reduce>
//   d_out, T_reduce init, int queue_idx):
//             num_items(num_items), d_in(d_in), d_out(d_out), init(init),
//             queue_idx(queue_idx) {}
//   MGARDX_CONT int get_queue_idx() { return queue_idx; }
//   MGARDX_CONT int get_num_items() { return num_items; }
//   MGARDX_CONT SubArray<1, T_reduce>  get_d_in() { return d_in; }
//   MGARDX_CONT SubArray<1, T_reduce>  get_d_out() { return d_out; }
//   MGARDX_CONT T_reduce get_init() { return init; }
// private:
//   int num_items;
//   SubArray<1, T_reduce> d_in;
//   SubArray<1, T_reduce> d_out;
//   T_reduce init;
//   int queue_idx;
// };

} // namespace mgard_x
#endif