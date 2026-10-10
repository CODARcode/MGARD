/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_160_ZERO_RLE_ENCODE_TEMPLATE_HPP
#define MGARD_X_160_ZERO_RLE_ENCODE_TEMPLATE_HPP

#include "../../RuntimeX/RuntimeX.h"

namespace mgard_x_160 {
namespace parallel_rle {

// Zero-RLE encode. start_positions holds the absolute index of each of the
// num_symbols nonzeros (computed by the shared StartPositionsKernel), so for
// symbol s:
//   symbols[s] = data[start_positions[s]]
//   counts[s]  = number of zeros immediately preceding it
//              = start_positions[s] - start_positions[s-1] - 1   (s > 0)
//              = start_positions[0]                              (s == 0)
// Trailing zeros after the last nonzero are not stored; they are recovered on
// decode from the known original length.
template <typename T_symbol, typename C_run, typename C_global,
          typename DeviceType>
class ZeroEncodeFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT ZeroEncodeFunctor() {}
  MGARDX_CONT
  ZeroEncodeFunctor(C_global num_symbols,
                    SubArray<1, T_symbol, DeviceType> data,
                    SubArray<1, C_global, DeviceType> start_positions,
                    SubArray<1, C_run, DeviceType> counts,
                    SubArray<1, T_symbol, DeviceType> symbols)
      : num_symbols(num_symbols), data(data), start_positions(start_positions),
        counts(counts), symbols(symbols) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    IDX start = FunctorBase<DeviceType>::GetBlockIdX() *
                    FunctorBase<DeviceType>::GetBlockDimX() +
                FunctorBase<DeviceType>::GetThreadIdX();

    IDX grid_size = FunctorBase<DeviceType>::GetGridDimX() *
                    FunctorBase<DeviceType>::GetBlockDimX();

    for (IDX i = start; i < num_symbols; i += grid_size) {
      C_global curr_pos = *start_positions(i);
      C_global prev_pos = (i == 0) ? 0 : (*start_positions(i - 1) + 1);

      *symbols(i) = *data(curr_pos);
      *counts(i) = (C_run)(curr_pos - prev_pos);
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  C_global num_symbols;
  SubArray<1, T_symbol, DeviceType> data;
  SubArray<1, C_global, DeviceType> start_positions;
  SubArray<1, C_run, DeviceType> counts;
  SubArray<1, T_symbol, DeviceType> symbols;
};

template <typename T_symbol, typename C_run, typename C_global,
          typename DeviceType>
class ZeroEncodeKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "zero rle encode";
  MGARDX_CONT
  ZeroEncodeKernel(C_global num_symbols, SubArray<1, T_symbol, DeviceType> data,
                   SubArray<1, C_global, DeviceType> start_positions,
                   SubArray<1, C_run, DeviceType> counts,
                   SubArray<1, T_symbol, DeviceType> symbols)
      : num_symbols(num_symbols), data(data), start_positions(start_positions),
        counts(counts), symbols(symbols) {}

  MGARDX_CONT Task<ZeroEncodeFunctor<T_symbol, C_run, C_global, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType =
        ZeroEncodeFunctor<T_symbol, C_run, C_global, DeviceType>;
    FunctorType functor(num_symbols, data, start_positions, counts, symbols);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    SIZE repeat_factor = 4;
    tbz = 1;
    tby = 1;
    tbx = 256;
    gridz = 1;
    gridy = 1;
    gridx = (num_symbols - 1) / tbx + 1;
    gridx = std::max((SIZE)DeviceRuntime<DeviceType>::GetNumSMs(),
                     gridx / repeat_factor);

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  C_global num_symbols;
  SubArray<1, T_symbol, DeviceType> data;
  SubArray<1, C_global, DeviceType> start_positions;
  SubArray<1, C_run, DeviceType> counts;
  SubArray<1, T_symbol, DeviceType> symbols;
};
} // namespace parallel_rle
} // namespace mgard_x_160

#endif
