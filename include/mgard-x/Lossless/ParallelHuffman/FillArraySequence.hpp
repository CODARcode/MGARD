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
  MGARDX_CONT FillArraySequenceFunctor(SubArray<1, T, DeviceType> array)
      : array(array) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    unsigned int thread = (FunctorBase<DeviceType>::GetBlockIdX() *
                           FunctorBase<DeviceType>::GetBlockDimX()) +
                          FunctorBase<DeviceType>::GetThreadIdX();
    if (thread < array.shape(0)) {
      *array(thread) = thread;
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, T, DeviceType> array;
};

template <typename T, typename DeviceType>
class FillArraySequenceKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "filling array sequence";
  MGARDX_CONT
  FillArraySequenceKernel(SubArray<1, T, DeviceType> array) : array(array) {}

  MGARDX_CONT
  Task<FillArraySequenceFunctor<T, DeviceType>> GenTask(int queue_idx) {
    using FunctorType = FillArraySequenceFunctor<T, DeviceType>;
    FunctorType functor(array);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = DeviceRuntime<DeviceType>::GetMaxNumThreadsPerTB();
    gridz = 1;
    gridy = 1;
    gridx = (array.shape(0) / tbx) + 1;
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<1, T, DeviceType> array;
};
} // namespace mgard_x

#endif