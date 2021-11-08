/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#ifndef MGRAD_CUDA_FILL_ARRAY_SEQUENCE_TEMPLATE_HPP
#define MGRAD_CUDA_FILL_ARRAY_SEQUENCE_TEMPLATE_HPP

#include "../CommonInternal.h"

namespace mgard_cuda {

template <typename T, typename DeviceType>
class FillArraySequenceFunctor: public Functor<DeviceType> {
  public:
  MGARDm_CONT FillArraySequenceFunctor(SubArray<1, T, DeviceType> array, 
                                       SIZE size):
                                       array(array), size(size) {
    Functor<DeviceType>();                            
  }

  MGARDm_EXEC void
  Operation1() {
    unsigned int thread = (this->blockx * this->nblockx) + this->threadx;
    if (thread < size) {
      *array(thread) = thread;
    }
  }

  MGARDm_EXEC void
  Operation2() { }

  MGARDm_EXEC void
  Operation3() { }

  MGARDm_EXEC void
  Operation4() { }

  MGARDm_EXEC void
  Operation5() { }

  MGARDm_CONT size_t
  shared_memory_size() { return 0; }

  private:
  SubArray<1, T, DeviceType> array;
  SIZE size; 
};


template <typename T, typename DeviceType>
class FillArraySequence: public AutoTuner<DeviceType> {
public:
  MGARDm_CONT
  FillArraySequence():AutoTuner<DeviceType>() {}

  MGARDm_CONT
  Task<FillArraySequenceFunctor<T, DeviceType> > 
  GenTask(SubArray<1, T, DeviceType> array, SIZE dict_size, int queue_idx) {
    using FunctorType = FillArraySequenceFunctor<T, DeviceType>;
    FunctorType functor(array, dict_size);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = DeviceRuntime<DeviceType>::GetMaxNumThreadsPerSM();
    gridz = 1;
    gridy = 1;
    gridx = (dict_size / tbx) + 1;
    // printf("%u %u %u\n", shape.dataHost()[2], shape.dataHost()[1], shape.dataHost()[0]);
    // PrintSubarray("shape", shape);
    return Task(functor, gridz, gridy, gridx, 
                tbz, tby, tbx, sm_size, queue_idx); 
  }

  MGARDm_CONT
  void Execute(SubArray<1, T, DeviceType> array, SIZE dict_size, int queue_idx) {
    using FunctorType = FillArraySequenceFunctor<T, DeviceType>;
    using TaskType = Task<FunctorType>;
    TaskType task = GenTask(array, dict_size, queue_idx); 
    DeviceAdapter<TaskType, DeviceType> adapter; 
    adapter.Execute(task);
  }
};

}

#endif