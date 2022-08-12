#ifndef _MDR_DIRECT_INTERLEAVER_HPP
#define _MDR_DIRECT_INTERLEAVER_HPP

#include "../../RuntimeX/RuntimeX.h"

#include "../../Linearization/LevelLinearizer.hpp"

#include "InterleaverInterface.hpp"

namespace mgard_x {
namespace MDR {

template <DIM D, typename T, OPTION Direction, typename DeviceType>
class DirectInterleaverKernel : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  DirectInterleaverKernel() : AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<LevelLinearizerFunctor<D, T, R, C, F, Direction, DeviceType>>
  GenTask(SubArray<2, SIZE, DeviceType> level_ranges, SIZE l_target,
          SubArray<D, T, DeviceType> v, SubArray<1, T, DeviceType> *level_v,
          int queue_idx) {
    using FunctorType =
        LevelLinearizerFunctor<D, T, R, C, F, Direction, DeviceType>;
    FunctorType functor(level_ranges, l_target, v, level_v);
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    int total_thread_z = v.shape(D - 3);
    int total_thread_y = v.shape(D - 2);
    int total_thread_x = v.shape(D - 1);
    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);
    for (int d = D - 4; d >= 0; d--) {
      gridx *= v.shape(d);
    }
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "LevelLinearizer");
  }

  MGARDX_CONT
  void Execute(SubArray<2, SIZE, DeviceType> level_ranges, SIZE l_target,
               SubArray<D, T, DeviceType> v,
               SubArray<1, T, DeviceType> *level_v, int queue_idx) {
#define KERNEL(R, C, F)                                                        \
  {                                                                            \
    using FunctorType =                                                        \
        LevelLinearizerFunctor<D, T, R, C, F, Direction, DeviceType>;          \
    using TaskType = Task<FunctorType>;                                        \
    TaskType task =                                                            \
        GenTask<R, C, F>(level_ranges, l_target, v, level_v, queue_idx);       \
    DeviceAdapter<TaskType, DeviceType> adapter;                               \
    adapter.Execute(task);                                                     \
  }

    if (D >= 3) {
      KERNEL(4, 4, 16)
    }
    if (D == 2) {
      KERNEL(1, 4, 32)
    }
    if (D == 1) {
      KERNEL(1, 1, 64)
    }
#undef KERNEL
  }
};

// direct interleaver with in-order recording
template <DIM D, typename T, typename DeviceType>
class DirectInterleaver
    : public concepts::InterleaverInterface<D, T, DeviceType> {
public:
  DirectInterleaver(Hierarchy<D, T, DeviceType> &hierarchy)
      : hierarchy(hierarchy) {}
  void interleave(SubArray<D, T, DeviceType> decomposed_data,
                  SubArray<1, T, DeviceType> *levels_decomposed_data,
                  SIZE target_level, int queue_idx) const {
    // PrintSubarray("decomposed_data", decomposed_data);
    SubArray<1, T, DeviceType> *levels_decomposed_data_device;

    MemoryManager<DeviceType>::Malloc1D(levels_decomposed_data_device,
                                        target_level + 1, queue_idx);
    MemoryManager<DeviceType>::Copy1D(levels_decomposed_data_device,
                                      levels_decomposed_data, target_level + 1,
                                      queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    DirectInterleaverKernel<D, T, Interleave, DeviceType>().Execute(
        SubArray<2, SIZE, DeviceType>(hierarchy.level_ranges(), true),
        target_level, decomposed_data, levels_decomposed_data_device,
        queue_idx);

    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    // handle.sync(queue_idx);
    // for (int i = 0; i < this->handle.l_target+1; i++) {
    //   printf("l = %d\n", i);
    //   PrintSubarray("levels_decomposed_data",
    //   levels_decomposed_data[i]);
    // }
  }
  void reposition(SubArray<1, T, DeviceType> *levels_decomposed_data,
                  SubArray<D, T, DeviceType> decomposed_data, SIZE target_level,
                  int queue_idx) const {
    SubArray<1, T, DeviceType> *levels_decomposed_data_device;

    MemoryManager<DeviceType>::Malloc1D(levels_decomposed_data_device,
                                        target_level + 1, queue_idx);
    MemoryManager<DeviceType>::Copy1D(levels_decomposed_data_device,
                                      levels_decomposed_data, target_level + 1,
                                      queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    DirectInterleaverKernel<D, T, Reposition, DeviceType>().Execute(
        SubArray<2, SIZE, DeviceType>(hierarchy.level_ranges(), true),
        target_level, decomposed_data, levels_decomposed_data_device,
        queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
  }
  void print() const { std::cout << "Direct interleaver" << std::endl; }

private:
  Hierarchy<D, T, DeviceType> &hierarchy;
};

} // namespace MDR
} // namespace mgard_x

#endif
