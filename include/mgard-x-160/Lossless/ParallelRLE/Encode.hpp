/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_160_RLE_ENCODE_TEMPLATE_HPP
#define MGARD_X_160_RLE_ENCODE_TEMPLATE_HPP

#include "../../RuntimeX/RuntimeX.h"

namespace mgard_x_160 {
namespace parallel_rle {
template <typename T_symbol, typename C_run, typename C_global,
          typename DeviceType>
class EncodeFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT EncodeFunctor() {}
  MGARDX_CONT
  EncodeFunctor(C_global total_run_length,
                SubArray<1, T_symbol, DeviceType> data,
                SubArray<1, C_global, DeviceType> start_positions,
                SubArray<1, C_run, DeviceType> counts,
                SubArray<1, T_symbol, DeviceType> symbols)
      : total_run_length(total_run_length), data(data),
        start_positions(start_positions), counts(counts), symbols(symbols) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    IDX start = FunctorBase<DeviceType>::GetBlockIdX() *
                    FunctorBase<DeviceType>::GetBlockDimX() +
                FunctorBase<DeviceType>::GetThreadIdX();

    IDX grid_size = FunctorBase<DeviceType>::GetGridDimX() *
                    FunctorBase<DeviceType>::GetBlockDimX();

    for (IDX i = start; i < total_run_length; i += grid_size) {
      C_global curr_start_pos = *start_positions(i);
      C_global next_start_pos = *start_positions(i + 1);

      *symbols(i) = *data(curr_start_pos);
      *counts(i) = (C_run)(next_start_pos - curr_start_pos);
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  C_global total_run_length;
  SubArray<1, T_symbol, DeviceType> data;
  SubArray<1, C_global, DeviceType> start_positions;
  SubArray<1, C_run, DeviceType> counts;
  SubArray<1, T_symbol, DeviceType> symbols;
};

template <typename T_symbol, typename C_run, typename C_global,
          typename DeviceType>
class EncodeKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "encode";
  MGARDX_CONT
  EncodeKernel(C_global total_run_length,
               SubArray<1, T_symbol, DeviceType> data,
               SubArray<1, C_global, DeviceType> start_positions,
               SubArray<1, C_run, DeviceType> counts,
               SubArray<1, T_symbol, DeviceType> symbols)
      : total_run_length(total_run_length), data(data),
        start_positions(start_positions), counts(counts), symbols(symbols) {}

  MGARDX_CONT Task<EncodeFunctor<T_symbol, C_run, C_global, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = EncodeFunctor<T_symbol, C_run, C_global, DeviceType>;
    FunctorType functor(total_run_length, data, start_positions, counts,
                        symbols);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    SIZE repeat_factor = 4;
    tbz = 1;
    tby = 1;
    tbx = 256;
    gridz = 1;
    gridy = 1;
    gridx = (total_run_length - 1) / tbx + 1;
    gridx = std::max((SIZE)DeviceRuntime<DeviceType>::GetNumSMs(),
                     gridx / repeat_factor);

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  C_global total_run_length;
  SubArray<1, T_symbol, DeviceType> data;
  SubArray<1, C_global, DeviceType> start_positions;
  SubArray<1, C_run, DeviceType> counts;
  SubArray<1, T_symbol, DeviceType> symbols;
};
} // namespace parallel_rle
} // namespace mgard_x_160

#endif