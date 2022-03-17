/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_FILL_ARRAY_SEQUENCE_TEMPLATE_HPP
#define MGARD_X_FILL_ARRAY_SEQUENCE_TEMPLATE_HPP

#include "../../RuntimeX/RuntimeX.h"

namespace mgard_x {

template <typename T, typename DeviceType>
class FillArraySequenceFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT FillArraySequenceFunctor() {}
  MGARDX_CONT FillArraySequenceFunctor(SubArray<1, T, DeviceType> array,
                                       SIZE size)
      : array(array), size(size) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    unsigned int thread = (FunctorBase<DeviceType>::GetBlockIdX() *
                           FunctorBase<DeviceType>::GetBlockDimX()) +
                          FunctorBase<DeviceType>::GetThreadIdX();
    if (thread < size) {
      *array(thread) = thread;
    }
  }

  MGARDX_EXEC void Operation2() {}

  MGARDX_EXEC void Operation3() {}

  MGARDX_EXEC void Operation4() {}

  MGARDX_EXEC void Operation5() {}

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, T, DeviceType> array;
  SIZE size;
};

template <typename T, typename DeviceType>
class FillArraySequence : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  FillArraySequence() : AutoTuner<DeviceType>() {}

  MGARDX_CONT
  Task<FillArraySequenceFunctor<T, DeviceType>>
  GenTask(SubArray<1, T, DeviceType> array, SIZE dict_size, int queue_idx) {
    using FunctorType = FillArraySequenceFunctor<T, DeviceType>;
    FunctorType functor(array, dict_size);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = DeviceRuntime<DeviceType>::GetMaxNumThreadsPerTB();
    gridz = 1;
    gridy = 1;
    gridx = (dict_size / tbx) + 1;
    // printf("tbx: %u, gridx: %u\n", tbx, gridx);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "FillArraySequence");
  }

  MGARDX_CONT
  void Execute(SubArray<1, T, DeviceType> array, SIZE dict_size,
               int queue_idx) {
    using FunctorType = FillArraySequenceFunctor<T, DeviceType>;
    using TaskType = Task<FunctorType>;
    TaskType task = GenTask(array, dict_size, queue_idx);
    DeviceAdapter<TaskType, DeviceType> adapter;
    adapter.Execute(task);
  }
};

} // namespace mgard_x

#endif