/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_160_ZERO_RLE_START_POSITIONS_TEMPLATE_HPP
#define MGARD_X_160_ZERO_RLE_START_POSITIONS_TEMPLATE_HPP

#include "../../RuntimeX/RuntimeX.h"

namespace mgard_x_160 {
namespace parallel_rle {

// Compact the absolute positions of the nonzeros. scanned_start_marks is the
// inclusive scan of the nonzero marks, so scanned[i] is the count of nonzeros
// in [0, i]. Element i is a nonzero exactly when its running count differs from
// the previous one; in that case its 0-based rank is scanned[i] - 1 and its
// position is i. Unlike the full-RLE StartPositions there is no i == 0 override
// (position 0 is only a symbol when data[0] is nonzero) and no terminal entry
// (ZeroEncode derives gaps from consecutive positions, not a sentinel).
template <typename T_symbol, typename C_run, typename C_global,
          typename DeviceType>
class ZeroStartPositionsFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT ZeroStartPositionsFunctor() {}
  MGARDX_CONT ZeroStartPositionsFunctor(
      SubArray<1, C_global, DeviceType> scanned_start_marks,
      SubArray<1, C_global, DeviceType> start_positions)
      : scanned_start_marks(scanned_start_marks),
        start_positions(start_positions) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    IDX start = FunctorBase<DeviceType>::GetBlockIdX() *
                    FunctorBase<DeviceType>::GetBlockDimX() +
                FunctorBase<DeviceType>::GetThreadIdX();
    IDX grid_size = FunctorBase<DeviceType>::GetGridDimX() *
                    FunctorBase<DeviceType>::GetBlockDimX();
    IDX n = scanned_start_marks.shape(0);

    for (IDX i = start; i < n; i += grid_size) {
      C_global curr_pos = *scanned_start_marks(i);
      C_global prev_pos = i > 0 ? *scanned_start_marks(i - 1) : 0;
      if (curr_pos != prev_pos) {
        *start_positions(curr_pos - 1) = i;
      }
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, C_global, DeviceType> scanned_start_marks;
  SubArray<1, C_global, DeviceType> start_positions;
};

template <typename T_symbol, typename C_run, typename C_global,
          typename DeviceType>
class ZeroStartPositionsKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "zero rle start positions";
  MGARDX_CONT
  ZeroStartPositionsKernel(
      SubArray<1, C_global, DeviceType> scanned_start_marks,
      SubArray<1, C_global, DeviceType> start_positions)
      : scanned_start_marks(scanned_start_marks),
        start_positions(start_positions) {}

  MGARDX_CONT
  Task<ZeroStartPositionsFunctor<T_symbol, C_run, C_global, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType =
        ZeroStartPositionsFunctor<T_symbol, C_run, C_global, DeviceType>;
    FunctorType functor(scanned_start_marks, start_positions);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    SIZE repeat_factor = 4;
    tbz = 1;
    tby = 1;
    tbx = 256;
    gridz = 1;
    gridy = 1;
    gridx = (scanned_start_marks.shape(0) - 1) / tbx + 1;
    gridx = std::max((SIZE)1, gridx / repeat_factor);

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<1, C_global, DeviceType> scanned_start_marks;
  SubArray<1, C_global, DeviceType> start_positions;
};
} // namespace parallel_rle
} // namespace mgard_x_160

#endif
