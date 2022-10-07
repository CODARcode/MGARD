/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_GET_FIRST_NONZERO_INDEX_TEMPLATE_HPP
#define MGARD_X_GET_FIRST_NONZERO_INDEX_TEMPLATE_HPP

#include "../../RuntimeX/RuntimeX.h"

namespace mgard_x {

template <typename T, typename DeviceType>
class GetFirstNonzeroIndexFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT GetFirstNonzeroIndexFunctor() {}
  MGARDX_CONT GetFirstNonzeroIndexFunctor(SubArray<1, T, DeviceType> array,
                                          SubArray<1, T, DeviceType> result)
      : array(array), result(result) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    unsigned int thread = (FunctorBase<DeviceType>::GetBlockIdX() *
                           FunctorBase<DeviceType>::GetBlockDimX()) +
                          FunctorBase<DeviceType>::GetThreadIdX();
    if (thread < array.shape(0) && *array(thread) != 0) {
      Atomic<unsigned int, AtomicGlobalMemory, AtomicDeviceScope,
             DeviceType>::Min(result((IDX)0), thread);
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, T, DeviceType> array;
  SubArray<1, T, DeviceType> result;
};

template <typename T, typename DeviceType>
class GetFirstNonzeroIndexKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "get first non-zero";
  MGARDX_CONT
  GetFirstNonzeroIndexKernel(SubArray<1, T, DeviceType> array,
                             SubArray<1, T, DeviceType> result)
      : array(array), result(result) {}

  MGARDX_CONT
  Task<GetFirstNonzeroIndexFunctor<T, DeviceType>> GenTask(int queue_idx) {
    using FunctorType = GetFirstNonzeroIndexFunctor<T, DeviceType>;
    FunctorType functor(array, result);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = DeviceRuntime<DeviceType>::GetMaxNumThreadsPerTB();
    gridz = 1;
    gridy = 1;
    gridx = (array.shape(0) / tbx) + 1;
    // printf("%u %u %u\n", shape.dataHost()[2], shape.dataHost()[1],
    // shape.dataHost()[0]); PrintSubarray("shape", shape);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<1, T, DeviceType> array;
  SubArray<1, T, DeviceType> result;
};

} // namespace mgard_x

#endif