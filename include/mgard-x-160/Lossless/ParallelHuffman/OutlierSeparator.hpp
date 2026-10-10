/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_160_OUTLIER_SEPARATOR_TEMPLATE_HPP
#define MGARD_X_160_OUTLIER_SEPARATOR_TEMPLATE_HPP
#include "../../RuntimeX/RuntimeX.h"

#define MGARDX_SEPARATE_OUTLIER 1
#define MGARDX_RESTORE_OUTLIER 2

namespace mgard_x_160 {

template <typename T, OPTION OP, typename DeviceType>
class OutlierSeparatorFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT OutlierSeparatorFunctor() {}
  MGARDX_CONT
  OutlierSeparatorFunctor(SubArray<1, T, DeviceType> v, SIZE dict_size,
                          SubArray<1, ATOMIC_IDX, DeviceType> outlier_count,
                          SubArray<1, ATOMIC_IDX, DeviceType> outlier_index,
                          SubArray<1, T, DeviceType> outlier_value)
      : v(v), dict_size(dict_size), outlier_count(outlier_count),
        outlier_index(outlier_index), outlier_value(outlier_value) {
    Functor<DeviceType>();
  }

  // SEPARATE uses block-aggregated atomics: every outlier first reserves a slot
  // in a block-local counter (shared memory, block scope), then a single thread
  // reserves the whole block's contiguous output range with one device-scope
  // atomic. This collapses up to blockDim global atomics on the single counter
  // address into one per block, removing the contention that made this the
  // bottleneck. The functor framework inserts a block barrier between
  // consecutive Operation*() phases.
  MGARDX_EXEC void Operation1() {
    id = FunctorBase<DeviceType>::GetBlockIdX() *
             FunctorBase<DeviceType>::GetBlockDimX() +
         FunctorBase<DeviceType>::GetThreadIdX();
    if constexpr (OP == MGARDX_SEPARATE_OUTLIER) {
      // sm[0]: block-local outlier count, sm[1]: block base offset
      sm = (ATOMIC_IDX *)FunctorBase<DeviceType>::GetSharedMemory();
      if (FunctorBase<DeviceType>::GetThreadIdX() == 0) {
        sm[0] = 0;
      }
      is_outlier = false;
      if (id < v.shape(0)) {
        value = *v(id);
        if (value < 0 || value >= dict_size) {
          is_outlier = true;
        }
      }
    } else if constexpr (OP == MGARDX_RESTORE_OUTLIER) {
      // Grid is sized to the outlier count, so this is a sparse scatter.
      if (id < outlier_value.shape(0)) {
        ATOMIC_IDX index = *outlier_index(id);
        T val = *outlier_value(id);
        *v(index) = val;
      }
    }
  }

  // Reserve a block-local slot for each outlier.
  MGARDX_EXEC void Operation2() {
    if constexpr (OP == MGARDX_SEPARATE_OUTLIER) {
      if (is_outlier) {
        local_offset = Atomic<ATOMIC_IDX, AtomicSharedMemory, AtomicDeviceScope,
                              DeviceType>::Add(&sm[0], (ATOMIC_IDX)1);
      }
    }
  }

  // One global atomic per block reserves the block's output range.
  MGARDX_EXEC void Operation3() {
    if constexpr (OP == MGARDX_SEPARATE_OUTLIER) {
      if (FunctorBase<DeviceType>::GetThreadIdX() == 0) {
        sm[1] = Atomic<ATOMIC_IDX, AtomicGlobalMemory, AtomicDeviceScope,
                       DeviceType>::Add(outlier_count((IDX)0), sm[0]);
      }
    }
  }

  // Scatter outliers to their global slots and zero them in the primary stream.
  MGARDX_EXEC void Operation4() {
    if constexpr (OP == MGARDX_SEPARATE_OUTLIER) {
      if (is_outlier) {
        ATOMIC_IDX outlier_write_index = sm[1] + local_offset;
        if (outlier_write_index < outlier_index.shape(0)) {
          *outlier_index(outlier_write_index) = id;
          *outlier_value(outlier_write_index) = value;
          *v(id) = 0;
        }
      }
    }
  }

  MGARDX_EXEC void Operation5() {}

  MGARDX_CONT size_t shared_memory_size() {
    if constexpr (OP == MGARDX_SEPARATE_OUTLIER) {
      return 2 * sizeof(ATOMIC_IDX);
    } else {
      return 0;
    }
  }

private:
  SubArray<1, T, DeviceType> v;
  SIZE dict_size;
  SubArray<1, ATOMIC_IDX, DeviceType> outlier_count;
  SubArray<1, ATOMIC_IDX, DeviceType> outlier_index;
  SubArray<1, T, DeviceType> outlier_value;

  // Per-thread state carried across the Operation*() phases (SEPARATE only).
  SIZE id;
  bool is_outlier;
  T value;
  ATOMIC_IDX local_offset;
  ATOMIC_IDX *sm;
};

template <typename T, OPTION OP, typename DeviceType>
class OutlierSeparatorKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "OutlierSeparator";
  MGARDX_CONT
  OutlierSeparatorKernel(SubArray<1, T, DeviceType> v, SIZE dict_size,
                         SubArray<1, ATOMIC_IDX, DeviceType> outlier_count,
                         SubArray<1, ATOMIC_IDX, DeviceType> outlier_index,
                         SubArray<1, T, DeviceType> outlier_value)
      : v(v), dict_size(dict_size), outlier_count(outlier_count),
        outlier_index(outlier_index), outlier_value(outlier_value) {}

  MGARDX_CONT
  Task<OutlierSeparatorFunctor<T, OP, DeviceType>> GenTask(int queue_idx) {
    using FunctorType = OutlierSeparatorFunctor<T, OP, DeviceType>;
    FunctorType functor(v, dict_size, outlier_count, outlier_index,
                        outlier_value);
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    if constexpr (std::is_same<DeviceType, HIP>::value) {
      // Empirically tuned on MI300: a direct sweep (64/128/256/512/768/896/
      // 960/1024) found throughput climbing steeply with block size --
      // 256 (the original) measures ~224 GB/s, 960 measures ~815 GB/s, a
      // >3.5x improvement with no logic change. 1024 (the hardware max)
      // reliably hung the kernel launch on this system, so 960 is used as
      // the largest safely-tested value rather than chasing the last ~7%
      // theoretical gain from 1024 at the risk of a hang. This tuning is
      // HIP/CDNA3-specific and untested on other backends, so it is guarded
      // here rather than applied unconditionally.
      tbx = 960;
    } else {
      tbx = 256;
    }
    gridz = 1;
    gridy = 1;
    // RESTORE is a sparse scatter over the outliers only (outlier_value is
    // sized to the outlier count after deserialization), so launch just enough
    // threads instead of one per element of the full array.
    SIZE launch_count;
    if constexpr (OP == MGARDX_RESTORE_OUTLIER) {
      launch_count = outlier_value.shape(0);
    } else {
      launch_count = v.shape(0);
    }
    gridx = (launch_count == 0) ? 1 : (launch_count - 1) / tbx + 1;
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<1, T, DeviceType> v;
  SIZE dict_size;
  SubArray<1, ATOMIC_IDX, DeviceType> outlier_count;
  SubArray<1, ATOMIC_IDX, DeviceType> outlier_index;
  SubArray<1, T, DeviceType> outlier_value;
};

} // namespace mgard_x_160

#endif