/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_160_RLE_START_MARKS_TEMPLATE_HPP
#define MGARD_X_160_RLE_START_MARKS_TEMPLATE_HPP

#include "../../RuntimeX/RuntimeX.h"

namespace mgard_x_160 {
namespace parallel_rle {
template <typename T_symbol, typename C_run, typename C_global,
          typename DeviceType>
class StartMarksFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT StartMarksFunctor() {}
  MGARDX_CONT StartMarksFunctor(SubArray<1, T_symbol, DeviceType> data,
                                SubArray<1, C_global, DeviceType> start_marks)
      : data(data), start_marks(start_marks) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    IDX start = FunctorBase<DeviceType>::GetBlockIdX() *
                    FunctorBase<DeviceType>::GetBlockDimX() +
                FunctorBase<DeviceType>::GetThreadIdX();
    IDX n = data.shape(0);
    IDX grid_size = FunctorBase<DeviceType>::GetGridDimX() *
                    FunctorBase<DeviceType>::GetBlockDimX();
    // HIP will fail if making the following line a constexpr
    IDX MAX_RUN = (IDX)1 << (sizeof(C_run) * 8);
    for (IDX i = start; i < n; i += grid_size) {
      if (i == 0) {
        *start_marks(i) = 1;
      } else {
        if (i % MAX_RUN == 0) {
          *start_marks(i) = 1;
        } else {
          *start_marks(i) = (*data(i) != *data(i - 1) ? 1 : 0);
        }
      }
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, T_symbol, DeviceType> data;
  SubArray<1, C_global, DeviceType> start_marks;
};

template <typename T_symbol, typename C_run, typename C_global,
          typename DeviceType>
class StartMarksKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "start marks";
  MGARDX_CONT
  StartMarksKernel(SubArray<1, T_symbol, DeviceType> data,
                   SubArray<1, C_global, DeviceType> start_marks)
      : data(data), start_marks(start_marks) {}

  MGARDX_CONT Task<StartMarksFunctor<T_symbol, C_run, C_global, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType =
        StartMarksFunctor<T_symbol, C_run, C_global, DeviceType>;
    FunctorType functor(data, start_marks);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    SIZE repeat_factor = 4;
    tbz = 1;
    tby = 1;
    tbx = 256;
    gridz = 1;
    gridy = 1;
    gridx = (data.shape(0) - 1) / tbx + 1;
    gridx = std::max((SIZE)DeviceRuntime<DeviceType>::GetNumSMs(),
                     gridx / repeat_factor);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<1, T_symbol, DeviceType> data;
  SubArray<1, C_global, DeviceType> start_marks;
};
} // namespace parallel_rle
} // namespace mgard_x_160

#endif