/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#ifndef MGRAD_CUDA_REORDER_BY_INDEX_TEMPLATE_HPP
#define MGRAD_CUDA_REORDER_BY_INDEX_TEMPLATE_HPP

#include "../CommonInternal.h"

namespace mgard_cuda {

// Jieyang: this kernel rely on whole grid sychronized execution
// For example, adding
//   if (thread == 0) {
//      __nanosleep(1e9);
//   }
// will cause incorrect results

template <typename T, typename Q, typename DeviceType>
class ReorderByIndexFunctor: public Functor<DeviceType> {
  public:
  MGARDm_CONT ReorderByIndexFunctor(SubArray<1, T, DeviceType> array, 
                                    SubArray<1, Q, DeviceType> index, 
                                  SIZE size):
                                  array(array), index(index), size(size) {
    Functor<DeviceType>();                            
  }

  MGARDm_EXEC void
  Operation1() {
    unsigned int thread = (this->blockx * this->nblockx) + this->threadx;
    T temp;
    Q newIndex;
    if (thread < size) {
      temp = *array(thread);
      newIndex = *index(thread);
      *array(newIndex) = temp;
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
  SubArray<1, Q, DeviceType> index;
  SIZE size; 
};


template <typename T, typename Q, typename DeviceType>
class ReorderByIndex: public AutoTuner<DeviceType> {
public:
  MGARDm_CONT
  ReorderByIndex():AutoTuner<DeviceType>() {}

  MGARDm_CONT
  Task<ReorderByIndexFunctor<T, Q, DeviceType> > 
  GenTask(SubArray<1, T, DeviceType> array, SubArray<1, Q, DeviceType> index, SIZE size, int queue_idx) {
    using FunctorType = ReorderByIndexFunctor<T, Q, DeviceType>;
    FunctorType functor(array, index, size);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = DeviceRuntime<DeviceType>::GetMaxNumThreadsPerSM();
    gridz = 1;
    gridy = 1;
    gridx = (size / tbx) + 1;
    if (gridx > DeviceRuntime<DeviceType>::GetNumSMs()) {
      std::cout << log::log_err << "ReorderByIndex: too much threadblocks for concurrent reordering!\n";
      exit(-1);
    }
    // printf("%u %u %u\n", shape.dataHost()[2], shape.dataHost()[1], shape.dataHost()[0]);
    // PrintSubarray("shape", shape);
    return Task(functor, gridz, gridy, gridx, 
                tbz, tby, tbx, sm_size, queue_idx, "ReorderByIndex"); 
  }

  MGARDm_CONT
  void Execute(SubArray<1, T, DeviceType> array, SubArray<1, Q, DeviceType> index, SIZE size, int queue_idx) {
    using FunctorType = ReorderByIndexFunctor<T, Q, DeviceType>;
    using TaskType = Task<FunctorType>;
    TaskType task = GenTask(array, index, size, queue_idx); 
    DeviceAdapter<TaskType, DeviceType> adapter; 
    adapter.Execute(task);
  }
};

}

#endif