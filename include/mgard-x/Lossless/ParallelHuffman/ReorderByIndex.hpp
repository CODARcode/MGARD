/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_REORDER_BY_INDEX_TEMPLATE_HPP
#define MGARD_X_REORDER_BY_INDEX_TEMPLATE_HPP

#include "../../RuntimeX/RuntimeX.h"

namespace mgard_x {

// Jieyang: this kernel rely on whole grid sychronized execution
// For example, adding
//   if (thread == 0) {
//      __nanosleep(1e9);
//   }
// will cause incorrect results

template <typename T, typename Q, typename DeviceType>
class ReorderByIndexFunctor: public Functor<DeviceType> {
  public:
  MGARDX_CONT ReorderByIndexFunctor(){}
  MGARDX_CONT ReorderByIndexFunctor(SubArray<1, T, DeviceType> old_array, 
                                    SubArray<1, T, DeviceType> new_array, 
                                    SubArray<1, Q, DeviceType> index, 
                                  SIZE size):
                                  old_array(old_array), new_array(new_array), index(index), size(size) {
    Functor<DeviceType>();                            
  }

  MGARDX_EXEC void
  Operation1() {
    unsigned int thread = (FunctorBase<DeviceType>::GetBlockIdX() * FunctorBase<DeviceType>::GetBlockDimX()) + FunctorBase<DeviceType>::GetThreadIdX();
    T temp;
    Q newIndex;
    if (thread < size) {
      temp = *old_array(thread);
      newIndex = *index(thread);
      *new_array(newIndex) = temp;
    }
  }

  MGARDX_EXEC void
  Operation2() { }

  MGARDX_EXEC void
  Operation3() { }

  MGARDX_EXEC void
  Operation4() { }

  MGARDX_EXEC void
  Operation5() { }

  MGARDX_CONT size_t
  shared_memory_size() { return 0; }

  private:
  SubArray<1, T, DeviceType> old_array;
  SubArray<1, T, DeviceType> new_array;
  SubArray<1, Q, DeviceType> index;
  SIZE size; 
};


template <typename T, typename Q, typename DeviceType>
class ReorderByIndex: public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  ReorderByIndex():AutoTuner<DeviceType>() {}

  MGARDX_CONT
  Task<ReorderByIndexFunctor<T, Q, DeviceType> > 
  GenTask(SubArray<1, T, DeviceType> old_array, 
          SubArray<1, T, DeviceType> new_array, 
          SubArray<1, Q, DeviceType> index, SIZE size, int queue_idx) {
    using FunctorType = ReorderByIndexFunctor<T, Q, DeviceType>;
    FunctorType functor(old_array, new_array, index, size);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = DeviceRuntime<DeviceType>::GetMaxNumThreadsPerTB();
    gridz = 1;
    gridy = 1;
    gridx = (size / tbx) + 1;
    // if (gridx > DeviceRuntime<DeviceType>::GetNumSMs()) {
    //   std::cout << log::log_err << "ReorderByIndex: too much threadblocks for concurrent reordering!\n";
    //   exit(-1);
    // }
    // printf("%u %u %u\n", shape.dataHost()[2], shape.dataHost()[1], shape.dataHost()[0]);
    // PrintSubarray("shape", shape);
    return Task(functor, gridz, gridy, gridx, 
                tbz, tby, tbx, sm_size, queue_idx, "ReorderByIndex"); 
  }

  MGARDX_CONT
  void Execute(SubArray<1, T, DeviceType> old_array, 
               SubArray<1, T, DeviceType> new_array, 
               SubArray<1, Q, DeviceType> index, SIZE size, int queue_idx) {
    using FunctorType = ReorderByIndexFunctor<T, Q, DeviceType>;
    using TaskType = Task<FunctorType>;
    TaskType task = GenTask(old_array, new_array, index, size, queue_idx); 
    DeviceAdapter<TaskType, DeviceType> adapter; 
    adapter.Execute(task);
  }
};

}

#endif