#ifndef _MDR_DIRECT_INTERLEAVER_HPP
#define _MDR_DIRECT_INTERLEAVER_HPP

#include "../../RuntimeX/RuntimeX.h"

#include "../../Linearization/LevelLinearizer.hpp"

#include "InterleaverInterface.hpp"

namespace mgard_x {
namespace MDR {

template <DIM D, typename T, OPTION Direction, typename DeviceType>
class DirectInterleaverKernel : public Kernel {
public:
  constexpr static DIM NumDim = D;
  using DataType = T;
  constexpr static std::string_view Name = "llk";
  MGARDX_CONT
  DirectInterleaverKernel(SubArray<2, SIZE, DeviceType> level_ranges,
                          SubArray<2, int, DeviceType> level_marks,
                          SIZE l_target, SubArray<D, T, DeviceType> v,
                          SubArray<1, T, DeviceType> *level_v)
      : level_ranges(level_ranges), level_marks(level_marks),
        l_target(l_target), v(v), level_v(level_v) {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<LevelLinearizerFunctor<D, T, R, C, F, Direction, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType =
        LevelLinearizerFunctor<D, T, R, C, F, Direction, DeviceType>;
    FunctorType functor(level_ranges, level_marks, l_target, v, level_v);
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
                std::string(Name));
  }

private:
  SubArray<2, SIZE, DeviceType> level_ranges;
  SubArray<2, int, DeviceType> level_marks;
  SIZE l_target;
  SubArray<D, T, DeviceType> v;
  SubArray<1, T, DeviceType> *level_v;
};

// direct interleaver with in-order recording
template <DIM D, typename T, typename DeviceType>
class DirectInterleaver
    : public concepts::InterleaverInterface<D, T, DeviceType> {
public:
  DirectInterleaver(Hierarchy<D, T, DeviceType> hierarchy)
      : hierarchy(hierarchy) {
    MemoryManager<DeviceType>::Malloc1D(levels_decomposed_data_device,
                                        hierarchy.l_target() + 1);
  }
  ~DirectInterleaver() {
    MemoryManager<DeviceType>::Free(levels_decomposed_data_device);
  }

  DirectInterleaver(const DirectInterleaver &direct_interleaver) {
    hierarchy = direct_interleaver.hierarchy;
    MemoryManager<DeviceType>::Malloc1D(levels_decomposed_data_device,
                                        hierarchy.l_target() + 1);
  }

  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape) {
    size_t size = 0;
    Hierarchy<D, T, DeviceType> hierarchy;
    size += hierarchy.EstimateMemoryFootprint(shape);
    size += sizeof(SubArray<1, T, DeviceType>) * (hierarchy.l_target() + 1);
    return size;
  }

  void interleave(SubArray<D, T, DeviceType> decomposed_data,
                  SubArray<1, T, DeviceType> *levels_decomposed_data,
                  SIZE target_level, int queue_idx) {
    MemoryManager<DeviceType>::Copy1D(levels_decomposed_data_device,
                                      levels_decomposed_data, target_level + 1,
                                      queue_idx);
    DeviceLauncher<DeviceType>::Execute(
        DirectInterleaverKernel<D, T, Interleave, DeviceType>(
            SubArray(hierarchy.level_ranges()),
            SubArray(hierarchy.level_marks()), target_level, decomposed_data,
            levels_decomposed_data_device),
        queue_idx);
  }
  void reposition(SubArray<1, T, DeviceType> *levels_decomposed_data,
                  SubArray<D, T, DeviceType> decomposed_data, SIZE target_level,
                  int queue_idx) {
    MemoryManager<DeviceType>::Copy1D(levels_decomposed_data_device,
                                      levels_decomposed_data, target_level + 1,
                                      queue_idx);
    DeviceLauncher<DeviceType>::Execute(
        DirectInterleaverKernel<D, T, Reposition, DeviceType>(
            SubArray(hierarchy.level_ranges()),
            SubArray(hierarchy.level_marks()), target_level, decomposed_data,
            levels_decomposed_data_device),
        queue_idx);
  }
  void print() const { std::cout << "Direct interleaver" << std::endl; }

private:
  Hierarchy<D, T, DeviceType> hierarchy;
  SubArray<1, T, DeviceType> *levels_decomposed_data_device = nullptr;
};

} // namespace MDR
} // namespace mgard_x

#endif
