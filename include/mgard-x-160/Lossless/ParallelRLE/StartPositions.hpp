/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_160_RLE_START_POSITIONS_TEMPLATE_HPP
#define MGARD_X_160_RLE_START_POSITIONS_TEMPLATE_HPP

#include "../../RuntimeX/RuntimeX.h"

namespace mgard_x_160 {
namespace parallel_rle {
template <typename T_symbol, typename C_run, typename C_global,
          typename DeviceType>
class StartPositionsFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT StartPositionsFunctor() {}
  MGARDX_CONT
  StartPositionsFunctor(SubArray<1, C_global, DeviceType> scanned_start_marks,
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

      if (i == n - 1) {
        *start_positions(curr_pos) = n;
      }

      if (i == 0) {
        *start_positions((IDX)0) = 0;
      } else if (curr_pos != prev_pos) {
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
class StartPositionsKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "start positions";
  MGARDX_CONT
  StartPositionsKernel(SubArray<1, C_global, DeviceType> scanned_start_marks,
                       SubArray<1, C_global, DeviceType> start_positions)
      : scanned_start_marks(scanned_start_marks),
        start_positions(start_positions) {}

  MGARDX_CONT Task<StartPositionsFunctor<T_symbol, C_run, C_global, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType =
        StartPositionsFunctor<T_symbol, C_run, C_global, DeviceType>;
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