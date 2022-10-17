/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_CONDENSE_TEMPLATE_HPP
#define MGARD_X_CONDENSE_TEMPLATE_HPP
#include "../../RuntimeX/RuntimeX.h"

namespace mgard_x {

template <typename H, typename DeviceType>
class CondenseFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT CondenseFunctor() {}
  MGARDX_CONT CondenseFunctor(SubArray<1, H, DeviceType> v,
                              SubArray<1, size_t, DeviceType> write_offsets,
                              SubArray<1, size_t, DeviceType> actual_lengths,
                              SubArray<1, H, DeviceType> condensed_v,
                              SIZE chunck_size)
      : v(v), write_offsets(write_offsets), actual_lengths(actual_lengths),
        condensed_v(condensed_v), chunck_size(chunck_size) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE chunck_id = FunctorBase<DeviceType>::GetBlockIdX();
    SIZE thread_id = FunctorBase<DeviceType>::GetThreadIdX();
    size_t actual_length = *actual_lengths(chunck_id);
    size_t write_offset = *write_offsets(chunck_id);
    for (SIZE read_offset = thread_id; read_offset < actual_length;
         read_offset += FunctorBase<DeviceType>::GetBlockDimX()) {
      *condensed_v(write_offset + read_offset) =
          *v(chunck_id * chunck_size + read_offset);
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, H, DeviceType> v;
  SubArray<1, size_t, DeviceType> write_offsets;
  SubArray<1, size_t, DeviceType> actual_lengths;
  SubArray<1, H, DeviceType> condensed_v;
  SIZE chunck_size;
};

template <typename H, typename DeviceType>
class CondenseKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "condense";
  MGARDX_CONT
  CondenseKernel(SubArray<1, H, DeviceType> v,
                 SubArray<1, size_t, DeviceType> write_offsets,
                 SubArray<1, size_t, DeviceType> actual_lengths,
                 SubArray<1, H, DeviceType> condensed_v, SIZE chunck_size)
      : v(v), write_offsets(write_offsets), actual_lengths(actual_lengths),
        condensed_v(condensed_v), chunck_size(chunck_size) {}

  MGARDX_CONT
  Task<CondenseFunctor<H, DeviceType>> GenTask(int queue_idx) {
    using FunctorType = CondenseFunctor<H, DeviceType>;
    FunctorType functor(v, write_offsets, actual_lengths, condensed_v,
                        chunck_size);
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = 256;
    gridz = 1;
    gridy = 1;
    gridx = write_offsets.shape(0);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<1, H, DeviceType> v;
  SubArray<1, size_t, DeviceType> write_offsets;
  SubArray<1, size_t, DeviceType> actual_lengths;
  SubArray<1, H, DeviceType> condensed_v;
  SIZE chunck_size;
};

} // namespace mgard_x

#endif