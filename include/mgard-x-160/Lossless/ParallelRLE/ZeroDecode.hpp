/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_160_ZERO_RLE_DECODE_TEMPLATE_HPP
#define MGARD_X_160_ZERO_RLE_DECODE_TEMPLATE_HPP

#include "../../RuntimeX/RuntimeX.h"

namespace mgard_x_160 {
namespace parallel_rle {

// Step 1 of zero-RLE decode: widen each run count and add one, producing the
// per-symbol stride (preceding zeros + the symbol itself). The inclusive scan
// of these strides yields cum[s] = position[s] + 1, so position[s] = cum[s] - 1
// recovers each nonzero's absolute index.
template <typename T_symbol, typename C_run, typename C_global,
          typename DeviceType>
class ZeroStrideFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT ZeroStrideFunctor() {}
  MGARDX_CONT
  ZeroStrideFunctor(SubArray<1, C_run, DeviceType> counts,
                    SubArray<1, C_global, DeviceType> strides)
      : counts(counts), strides(strides) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    IDX start = FunctorBase<DeviceType>::GetBlockIdX() *
                    FunctorBase<DeviceType>::GetBlockDimX() +
                FunctorBase<DeviceType>::GetThreadIdX();
    IDX grid_size = FunctorBase<DeviceType>::GetGridDimX() *
                    FunctorBase<DeviceType>::GetBlockDimX();
    IDX n = counts.shape(0);

    for (IDX i = start; i < n; i += grid_size) {
      *strides(i) = (C_global)(*counts(i)) + 1;
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, C_run, DeviceType> counts;
  SubArray<1, C_global, DeviceType> strides;
};

template <typename T_symbol, typename C_run, typename C_global,
          typename DeviceType>
class ZeroStrideKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "zero rle stride";
  MGARDX_CONT
  ZeroStrideKernel(SubArray<1, C_run, DeviceType> counts,
                   SubArray<1, C_global, DeviceType> strides)
      : counts(counts), strides(strides) {}

  MGARDX_CONT Task<ZeroStrideFunctor<T_symbol, C_run, C_global, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType =
        ZeroStrideFunctor<T_symbol, C_run, C_global, DeviceType>;
    FunctorType functor(counts, strides);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    SIZE repeat_factor = 4;
    tbz = 1;
    tby = 1;
    tbx = 256;
    gridz = 1;
    gridy = 1;
    gridx = (counts.shape(0) - 1) / tbx + 1;
    gridx = std::max((SIZE)DeviceRuntime<DeviceType>::GetNumSMs(),
                     gridx / repeat_factor);

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<1, C_run, DeviceType> counts;
  SubArray<1, C_global, DeviceType> strides;
};

// Step 2 of zero-RLE decode: scatter each nonzero symbol into its recovered
// absolute position. The output buffer is pre-zeroed by the caller, so only the
// nonzeros need to be written; runs of zeros (including any trailing zeros) are
// left untouched.
template <typename T_symbol, typename C_run, typename C_global,
          typename DeviceType>
class ZeroScatterFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT ZeroScatterFunctor() {}
  MGARDX_CONT
  ZeroScatterFunctor(SubArray<1, T_symbol, DeviceType> symbols,
                     SubArray<1, C_global, DeviceType> strides,
                     SubArray<1, T_symbol, DeviceType> data)
      : symbols(symbols), strides(strides), data(data) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    IDX start = FunctorBase<DeviceType>::GetBlockIdX() *
                    FunctorBase<DeviceType>::GetBlockDimX() +
                FunctorBase<DeviceType>::GetThreadIdX();
    IDX grid_size = FunctorBase<DeviceType>::GetGridDimX() *
                    FunctorBase<DeviceType>::GetBlockDimX();
    IDX n = symbols.shape(0);

    for (IDX i = start; i < n; i += grid_size) {
      // strides has been inclusive-scanned: strides[i] == position[i] + 1.
      C_global pos = *strides(i) - 1;
      *data(pos) = *symbols(i);
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, T_symbol, DeviceType> symbols;
  SubArray<1, C_global, DeviceType> strides;
  SubArray<1, T_symbol, DeviceType> data;
};

template <typename T_symbol, typename C_run, typename C_global,
          typename DeviceType>
class ZeroScatterKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "zero rle scatter";
  MGARDX_CONT
  ZeroScatterKernel(SubArray<1, T_symbol, DeviceType> symbols,
                    SubArray<1, C_global, DeviceType> strides,
                    SubArray<1, T_symbol, DeviceType> data)
      : symbols(symbols), strides(strides), data(data) {}

  MGARDX_CONT Task<ZeroScatterFunctor<T_symbol, C_run, C_global, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType =
        ZeroScatterFunctor<T_symbol, C_run, C_global, DeviceType>;
    FunctorType functor(symbols, strides, data);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    SIZE repeat_factor = 4;
    tbz = 1;
    tby = 1;
    tbx = 256;
    gridz = 1;
    gridy = 1;
    gridx = (symbols.shape(0) - 1) / tbx + 1;
    gridx = std::max((SIZE)DeviceRuntime<DeviceType>::GetNumSMs(),
                     gridx / repeat_factor);

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<1, T_symbol, DeviceType> symbols;
  SubArray<1, C_global, DeviceType> strides;
  SubArray<1, T_symbol, DeviceType> data;
};
} // namespace parallel_rle
} // namespace mgard_x_160

#endif
