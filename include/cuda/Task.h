/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */
#ifndef MGARD_CUDA_TASK
#define MGARD_CUDA_TASK

#include "Functor.h"

namespace mgard_cuda {

template <typename FunctorType>
class Task {
public:
    MGARDm_CONT
    Task(FunctorType functor, 
         IDX ngridz, IDX ngridy, IDX ngridx,
         IDX nblockz, IDX nblocky, IDX nblockx,
         LENGTH shared_memory_size,
         int queue_idx): 
          functor(functor), 
          ngridz(ngridz), ngridy(ngridy), ngridx(ngridx),
          nblockz(nblockz), nblocky(nblocky), nblockx(nblockx),
          shared_memory_size(shared_memory_size),
          queue_idx(queue_idx) {}

    // copy contructure does not work in device so return by reference
    MGARDm_EXEC FunctorType& get_functor() {return functor;}
    MGARDm_CONT int get_queue_idx() {return queue_idx;}
    MGARDm_CONT IDX get_ngridz() {return ngridz;}
    MGARDm_CONT IDX get_ngridy() {return ngridy;}
    MGARDm_CONT IDX get_ngridx() {return ngridx;}
    MGARDm_CONT IDX get_nblockz() {return nblockz;}
    MGARDm_CONT IDX get_nblocky() {return nblocky;}
    MGARDm_CONT IDX get_nblockx() {return nblockx;}
    MGARDm_CONT LENGTH get_shared_memory_size () {return shared_memory_size; }
    using Functor = FunctorType;
  private:
    FunctorType functor;
    IDX ngridz, ngridy, ngridx;
    IDX nblockz, nblocky, nblockx;
    LENGTH shared_memory_size;
    int queue_idx;
};


template <typename FunctorType, typename T_reduce>
class ReduceTask {
public:
  MGARDm_CONT
  ReduceTask(int num_items, SubArray<1, T_reduce> d_in, SubArray<1, T_reduce> d_out, T_reduce init, int queue_idx): 
            num_items(num_items), d_in(d_in), d_out(d_out), init(init), queue_idx(queue_idx) {}
  MGARDm_CONT int get_queue_idx() { return queue_idx; }
  MGARDm_CONT int get_num_items() { return num_items; }
  MGARDm_CONT SubArray<1, T_reduce>  get_d_in() { return d_in; }
  MGARDm_CONT SubArray<1, T_reduce>  get_d_out() { return d_out; }
  MGARDm_CONT T_reduce get_init() { return init; }
private:
  int num_items;
  SubArray<1, T_reduce> d_in;
  SubArray<1, T_reduce> d_out;
  T_reduce init;
  int queue_idx;
};


}
#endif