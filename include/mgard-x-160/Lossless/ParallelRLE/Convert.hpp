/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_160_RLE_CONVERT_TEMPLATE_HPP
#define MGARD_X_160_RLE_CONVERT_TEMPLATE_HPP

#include "../../RuntimeX/RuntimeX.h"

namespace mgard_x_160 {
namespace parallel_rle {
template <typename T_symbol, typename C_run, typename C_global,
          typename DeviceType>
class ConvertFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT ConvertFunctor() {}
  MGARDX_CONT
  ConvertFunctor(SubArray<1, C_run, DeviceType> counts_in,
                 SubArray<1, C_global, DeviceType> counts_out)
      : counts_in(counts_in), counts_out(counts_out) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    IDX start = FunctorBase<DeviceType>::GetBlockIdX() *
                    FunctorBase<DeviceType>::GetBlockDimX() +
                FunctorBase<DeviceType>::GetThreadIdX();
    IDX grid_size = FunctorBase<DeviceType>::GetGridDimX() *
                    FunctorBase<DeviceType>::GetBlockDimX();
    IDX n = counts_in.shape(0);

    for (IDX i = start; i < n; i += grid_size) {
      *counts_out(i) = (C_global)*counts_in(i);
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, C_run, DeviceType> counts_in;
  SubArray<1, C_global, DeviceType> counts_out;
};

template <typename T_symbol, typename C_run, typename C_global,
          typename DeviceType>
class ConvertKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "convert";
  MGARDX_CONT
  ConvertKernel(SubArray<1, C_run, DeviceType> counts_in,
                SubArray<1, C_global, DeviceType> counts_out)
      : counts_in(counts_in), counts_out(counts_out) {}

  MGARDX_CONT Task<ConvertFunctor<T_symbol, C_run, C_global, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = ConvertFunctor<T_symbol, C_run, C_global, DeviceType>;
    FunctorType functor(counts_in, counts_out);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    SIZE repeat_factor = 4;
    tbz = 1;
    tby = 1;
    tbx = 256;
    gridz = 1;
    gridy = 1;
    gridx = (counts_in.shape(0) - 1) / tbx + 1;
    gridx = std::max((SIZE)DeviceRuntime<DeviceType>::GetNumSMs(),
                     gridx / repeat_factor);

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<1, C_run, DeviceType> counts_in;
  SubArray<1, C_global, DeviceType> counts_out;
};
} // namespace parallel_rle
} // namespace mgard_x_160

#endif