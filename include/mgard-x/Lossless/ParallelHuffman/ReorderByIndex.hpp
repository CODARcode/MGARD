/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
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
class ReorderByIndexFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT ReorderByIndexFunctor() {}
  MGARDX_CONT ReorderByIndexFunctor(SubArray<1, T, DeviceType> old_array,
                                    SubArray<1, T, DeviceType> new_array,
                                    SubArray<1, Q, DeviceType> index)
      : old_array(old_array), new_array(new_array), index(index) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    unsigned int thread = (FunctorBase<DeviceType>::GetBlockIdX() *
                           FunctorBase<DeviceType>::GetBlockDimX()) +
                          FunctorBase<DeviceType>::GetThreadIdX();
    T temp;
    Q newIndex;
    if (thread < old_array.shape(0)) {
      temp = *old_array(thread);
      newIndex = *index(thread);
      *new_array(newIndex) = temp;
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, T, DeviceType> old_array;
  SubArray<1, T, DeviceType> new_array;
  SubArray<1, Q, DeviceType> index;
};

template <typename T, typename Q, typename DeviceType>
class ReorderByIndexKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "reorder by index";

  MGARDX_CONT
  ReorderByIndexKernel(SubArray<1, T, DeviceType> old_array,
                       SubArray<1, T, DeviceType> new_array,
                       SubArray<1, Q, DeviceType> index)
      : old_array(old_array), new_array(new_array), index(index) {}

  MGARDX_CONT
  Task<ReorderByIndexFunctor<T, Q, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = ReorderByIndexFunctor<T, Q, DeviceType>;
    FunctorType functor(old_array, new_array, index);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = DeviceRuntime<DeviceType>::GetMaxNumThreadsPerTB();
    gridz = 1;
    gridy = 1;
    gridx = (old_array.shape(0) / tbx) + 1;
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<1, T, DeviceType> old_array;
  SubArray<1, T, DeviceType> new_array;
  SubArray<1, Q, DeviceType> index;
};

} // namespace mgard_x

#endif