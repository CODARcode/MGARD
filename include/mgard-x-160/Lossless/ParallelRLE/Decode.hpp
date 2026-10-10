/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_160_RLE_DECODE_TEMPLATE_HPP
#define MGARD_X_160_RLE_DECODE_TEMPLATE_HPP

#include "../../RuntimeX/RuntimeX.h"

namespace mgard_x_160 {
namespace parallel_rle {
template <typename T_symbol, typename C_run, typename C_global,
          typename DeviceType>
class DecodeFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT DecodeFunctor() {}
  MGARDX_CONT DecodeFunctor(SubArray<1, C_run, DeviceType> counts,
                            SubArray<1, T_symbol, DeviceType> symbols,
                            SubArray<1, C_global, DeviceType> start_positions,
                            SubArray<1, T_symbol, DeviceType> data)
      : counts(counts), symbols(symbols), start_positions(start_positions),
        data(data) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {

    IDX start = FunctorBase<DeviceType>::GetBlockIdX();
    //  *
    //     FunctorBase<DeviceType>::GetBlockDimX() +
    // FunctorBase<DeviceType>::GetThreadIdX();

    IDX n = counts.shape(0);
    IDX grid_size = FunctorBase<DeviceType>::GetGridDimX();
    // *
    // FunctorBase<DeviceType>::GetBlockDimX();

    for (IDX i = start; i < n; i += grid_size) {
      C_global curr_start_pos = i == 0 ? 0 : *start_positions(i - 1);
      C_global next_start_pos = *start_positions(i);
      T_symbol symbol = *symbols(i);
      for (C_global j =
               FunctorBase<DeviceType>::GetThreadIdX() + curr_start_pos;
           j < next_start_pos; j += FunctorBase<DeviceType>::GetBlockDimX()) {
        *data(j) = symbol;
      }

      // for (SIZE j = curr_start_pos; j < curr_start_pos+1; j++) {
      //   *data(j) = symbol;
      // }
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, C_run, DeviceType> counts;
  SubArray<1, T_symbol, DeviceType> symbols;
  SubArray<1, C_global, DeviceType> start_positions;
  SubArray<1, T_symbol, DeviceType> data;
};

template <typename T_symbol, typename C_run, typename C_global,
          typename DeviceType>
class DecodeKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "decode";
  MGARDX_CONT
  DecodeKernel(SubArray<1, C_run, DeviceType> counts,
               SubArray<1, T_symbol, DeviceType> symbols,
               SubArray<1, C_global, DeviceType> start_positions,
               SubArray<1, T_symbol, DeviceType> data)
      : counts(counts), symbols(symbols), start_positions(start_positions),
        data(data) {}

  MGARDX_CONT Task<DecodeFunctor<T_symbol, C_run, C_global, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = DecodeFunctor<T_symbol, C_run, C_global, DeviceType>;
    FunctorType functor(counts, symbols, start_positions, data);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    SIZE repeat_factor = 8;
    tbz = 1;
    tby = 1;
    tbx = 256;
    gridz = 1;
    gridy = 1;
    gridx = counts.shape(0);
    gridx = std::max((SIZE)DeviceRuntime<DeviceType>::GetNumSMs(),
                     gridx / repeat_factor);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<1, C_run, DeviceType> counts;
  SubArray<1, T_symbol, DeviceType> symbols;
  SubArray<1, C_global, DeviceType> start_positions;
  SubArray<1, T_symbol, DeviceType> data;
};
} // namespace parallel_rle
} // namespace mgard_x_160

#endif