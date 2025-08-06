#ifndef _MDR_DIRECT_INTERLEAVER_GPU_HPP
#define _MDR_DIRECT_INTERLEAVER_GPU_HPP

#include "../../CommonInternal.h"
#include "../../Functor.h"
#include "../../AutoTuner.h"
#include "../../Task.h"
#include "../../DeviceAdapters/DeviceAdapterCuda.h"

#include "InterleaverInterface.hpp"
namespace mgard_m {
namespace MDR {

#define Interleave 0
#define Reposition 1

template <mgard_cuda::DIM D, typename T, int R, int C, int F, OPTION Direction, typename DeviceType>
class DirectInterleaverGPUFunctor: public Functor<DeviceType> {
public:
  MGARDm_CONT
  DirectInterleaverGPUFunctor(mgard_cuda::SIZE *ranges, mgard_cuda::SIZE l_target, 
                              SubArray<D, T> v,
                              SubArray<1, T> * level_v): 
                              ranges(ranges), l_target(l_target), v(v), level_v(level_v){
    Functor<DeviceType>();
  }

  MGARDm_EXEC void
  Operation1() {
    threadId = (this->threadx * this->nblockx * this->blocky) +
                (this->thready * this->nblockx) + this->threadx;

    int8_t * sm_p = (int8_t *)this->shared_memory;
    mgard_cuda::SIZE *ranges_sm = (mgard_cuda::SIZE*) sm_p; sm_p += D * (l_target + 2) * sizeof(mgard_cuda::SIZE);

    for (mgard_cuda::SIZE i = threadId; i < D * (l_target + 2); i += this->nblockx * this->blocky * this->nblockz) {
      ranges_sm[i] = ranges[i];
    }
  }

  MGARDm_EXEC void
  Operation2() {
    mgard_cuda::SIZE firstD = div_roundup(ranges_sm[l_target + 1], F);
    mgard_cuda::SIZE bidx = blockx;
    idx[0] = (bidx % firstD) * F + this->threadx;
    bidx /= firstD;
    if (D >= 2) {
      idx[1] = this->blocky * this->nblocky + this->thready;
    }
    if (D >= 3) {
      idx[2] = this->blockz * this->nblockz + this->threadz;
    }

    for (mgard_cuda::DIM d = 3; d < D; d++) {
      idx[d] = bidx % ranges_sm[(l_target + 2) * d + l_target + 1];
      bidx /= ranges_sm[(l_target + 2) * d + l_target + 1];
    }

    level = 0;
    for (mgard_cuda::DIM d = 0; d < D; d++) {
      l_bit[d] = 0l;
      for (mgard_cuda::SIZE l = 0; l < l_target + 1; l++) {
        long long unsigned int bit = (idx[d] >= ranges_sm[(l_target + 2) * d + l]) &&
                  (idx[d] < ranges_sm[(l_target + 2) * d + l + 1]);
        l_bit[d] += bit << l;
        // printf("idx: %d %d d: %d l_bit: %llu\n", idx[1], idx[0], d, l_bit);
      }
      level = max(level, __ffsll(l_bit[d]));
    }

    // distinguish different regions
    curr_region = 0;
    for (mgard_cuda::DIM d = 0; d < D; d++) {
      mgard_cuda::SIZE bit = !(level == __ffsll(l_bit[d]));
      curr_region += bit << D - 1 - d;
    }

    // region size
    for (mgard_cuda::DIM d = 0; d < D; d++) {
      coarse_level_size[d] = ranges_sm[(l_target + 2) * d + level];
      diff_level_size[d] = ranges_sm[(l_target + 2) * d + level + 1] - 
                           ranges_sm[(l_target + 2) * d + level];
    }

    for (mgard_cuda::DIM d = 0; d < D; d++) {
      mgard_cuda::SIZE bit = (curr_region >> D - 1 - d) & 1u;
      curr_region_dims[d] = bit ? coarse_level_size[d] : diff_level_size[d];
    }

    mgard_cuda::SIZE curr_region_size = 1;
    for (mgard_cuda::DIM d = 0; d < D; d++) {
      curr_region_size *= curr_region_dims[d];
    }

    // region offset
    mgard_cuda::SIZE curr_region_offset = 0;
    for (mgard_cuda::SIZE prev_region = 0; prev_region < curr_region; prev_region++) {
      mgard_cuda::SIZE prev_region_size = 1;
      for (mgard_cuda::DIM d = 0; d < D; d++) {
        mgard_cuda::SIZE bit = (prev_region >> D - 1 - d) & 1u;
        size *= bit ? coarse_level_size[d] : diff_level_size[d];
        prev_region_size *= bit ? coarse_level_size[d] : diff_level_size[d];
      }
      curr_region_offset += prev_region_size;
    }

    // thread offset
    mgard_cuda::SIZE curr_thread_offset = 0;
    for (mgard_cuda::SIZE d = 0; d < D; d++) {
      mgard_cuda::SIZE bit = (curr_region >> D - 1 - d) & 1u;
      curr_region_thread_idx[d] = bit ? idx[d] : idx[d] - coarse_level_size[d];
    }

    mgard_cuda::SIZE stride = 1;
    for (mgard_cuda::SIZE d = 0; d < D; d++) {
      curr_thread_offset += curr_region_thread_idx[d] * stride;
      stride *= curr_region_dims[d];
    }

    in_range = true;
    for (mgard_cuda::DIM d = 0; d < D; d++) {
      if (idx[d] >= ranges_sm[(l_target + 2) * d + l_target + 1]) {
        in_range = false;
      }
    }

    // convert to 0 based level
    level = level - 1;

    mgard_cuda::SIZE level_offset = curr_region_offset + curr_thread_offset;
    if (Direction == Interleave) {
      *level_v[level](level_offset) = v.dv[get_idx<D>(g.ldvs_d, idx)];
    } else if (Direction == Reposition) {
      v.dv[get_idx<D>(g.ldvs_d, idx)] = *level_v[level](level_offset);
    }
  }

  MGARDm_EXEC void
  Operation3() {}

  MGARDm_EXEC void
  Operation4() {}

  MGARDm_EXEC void
  Operation5() {}

  MGARDm_CONT size_t
  shared_memory_size() {
    size_t size = 0;
    size += D * (l_target + 2) * sizeof(mgard_cuda::SIZE);
    return size;
  }


private:
  mgard_cuda::SIZE *ranges;
  mgard_cuda::SIZE l_target;
  SubArray<D, T> v;
  SubArray<1, T> * level_v;

  // thread private variables
  size_t threadId;
  mgard_cuda::SIZE * ranges_sm;
  mgard_cuda::SIZE idx[D];
  mgard_cuda::SIZE coarse_level_size[D];
  mgard_cuda::SIZE diff_level_size[D];
  long long unsigned int l_bit[D];
  mgard_cuda::SIZE curr_region;
  mgard_cuda::SIZE curr_region_size;
  mgard_cuda::SIZE curr_region_offset;
  mgard_cuda::SIZE curr_region_dims[D];
  mgard_cuda::SIZE curr_region_thread_idx[D];
  bool in_range;
  int level;
};

template <typename HandleType, mgard_cuda::DIM D, typename T, OPTION Direction, typename DeviceType>
class DirectInterleaverGPU: public mgard_cuda::AutoTuner<HandleType, DeviceType> {
  public:
  MGARDm_CONT
  DirectInterleaverGPU(HandleType& handle): mgard_cuda::AutoTuner<HandleType, DeviceType>(handle) {}

  template <mgard_cuda::SIZE R, mgard_cuda::SIZE C, mgard_cuda::SIZE F>
  MGARDm_CONT
  Task<DirectInterleaverGPUFunctor<D, T, R, C, F, DeviceType>> 
  GenTask(mgard_cuda::SIZE *shapes_h, mgard_cuda::SIZE *ranges_d, mgard_cuda::SIZE l_target, 
          SubArray<D, T> v,
          SubArray<1, T> * level_v) {
    using FunctorType = DirectInterleaverGPUFunctor<D, T, R, C, F, Direction, DeviceType>;
    FunctorType functor(ranges_d, l_target, v, level_v);
    mgard_cuda::SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    int total_thread_z = shapes_h[2];
    int total_thread_y = shapes_h[1];
    int total_thread_x = shapes_h[0];
    // linearize other dimensions
    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);
    for (int d = 3; d < D; d++) {
      gridx *= shapes_h[d];
    }
    return mgard_cuda::Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx); 
  }

  MGARDm_CONT
  void Execute(mgard_cuda::SIZE *shapes_h, mgard_cuda::SIZE *ranges_d, mgard_cuda::SIZE l_target, 
               SubArray<D, T> v,
               SubArray<1, T> * level_v
               int queue_idx) {
    #define KERNEL(R, C, F)\
    {\
      using FunctorType = DirectInterleaverGPUFunctor<T, R, C, F, Direction, DeviceType>;\
      using TaskType = Task<FunctorType>;\
      TaskType task = GenTask<R, C, F>(\
                              shapes_h, ranges_d, l_target\
                              v, level_v);\
      mgard_cuda::DeviceAdapter<HandleType, TaskType, DeviceType> adapter(this->handle); \
      adapter.Execute(task);\
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






}
}

#endif