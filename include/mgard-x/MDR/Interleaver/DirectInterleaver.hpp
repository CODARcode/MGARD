#ifndef _MDR_DIRECT_INTERLEAVER_HPP
#define _MDR_DIRECT_INTERLEAVER_HPP

#include "../../RuntimeX/RuntimeX.h"

#include "InterleaverInterface.hpp"
namespace mgard_x {
namespace MDR {
// direct interleaver with in-order recording
template <DIM D, typename T>
class DirectInterleaver : public concepts::InterleaverInterface<D, T> {
public:
  DirectInterleaver(Handle<D, T> &handle) : _handle(handle) {}
  void interleave(T const *data, const std::vector<SIZE> &dims,
                  const std::vector<SIZE> &dims_fine,
                  const std::vector<SIZE> &dims_coasre, T *buffer) const {
    SIZE dim0_offset = dims[1] * dims[2];
    SIZE dim1_offset = dims[2];
    SIZE count = 0;
    for (int i = 0; i < dims_fine[0]; i++) {
      for (int j = 0; j < dims_fine[1]; j++) {
        for (int k = 0; k < dims_fine[2]; k++) {
          if ((i < dims_coasre[0]) && (j < dims_coasre[1]) &&
              (k < dims_coasre[2]))
            continue;
          buffer[count++] = data[i * dim0_offset + j * dim1_offset + k];
        }
      }
    }
  }
  void reposition(T const *buffer, const std::vector<SIZE> &dims,
                  const std::vector<SIZE> &dims_fine,
                  const std::vector<SIZE> &dims_coasre, T *data) const {
    SIZE dim0_offset = dims[1] * dims[2];
    SIZE dim1_offset = dims[2];
    SIZE count = 0;
    for (int i = 0; i < dims_fine[0]; i++) {
      for (int j = 0; j < dims_fine[1]; j++) {
        for (int k = 0; k < dims_fine[2]; k++) {
          if ((i < dims_coasre[0]) && (j < dims_coasre[1]) &&
              (k < dims_coasre[2]))
            continue;
          data[i * dim0_offset + j * dim1_offset + k] = buffer[count++];
        }
      }
    }
  }
  void print() const { std::cout << "Direct interleaver" << std::endl; }

private:
  Handle<D, T> &_handle;
};
} // namespace MDR
} // namespace mgard_x

namespace mgard_m {
namespace MDR {

#define Interleave 0
#define Reposition 1

template <mgard_x::DIM D, typename T, int R, int C, int F,
          mgard_x::OPTION Direction, typename DeviceType>
class DirectInterleaverFunctor : public mgard_x::Functor<DeviceType> {
public:
  MGARDX_CONT
  DirectInterleaverFunctor(
      mgard_x::SubArray<1, mgard_x::SIZE, DeviceType> ranges,
      mgard_x::SIZE l_target, mgard_x::SubArray<D, T, DeviceType> v,
      mgard_x::SubArray<1, T, DeviceType> *level_v)
      : ranges(ranges), l_target(l_target), v(v), level_v(level_v) {
    mgard_x::Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {

    debug = false;
    if (FunctorBase<DeviceType>::GetBlockIdZ() == 0 &&
        FunctorBase<DeviceType>::GetBlockIdY() == 0 &&
        FunctorBase<DeviceType>::GetBlockIdX() == 0 &&
        FunctorBase<DeviceType>::GetThreadIdX() == 0 &&
        FunctorBase<DeviceType>::GetThreadIdY() == 0 &&
        FunctorBase<DeviceType>::GetThreadIdZ() == 0)
      debug = true;

    mgard_x::SIZE threadId = (FunctorBase<DeviceType>::GetThreadIdZ() *
                              FunctorBase<DeviceType>::GetBlockDimX() *
                              FunctorBase<DeviceType>::GetBlockDimY()) +
                             (FunctorBase<DeviceType>::GetThreadIdY() *
                              FunctorBase<DeviceType>::GetBlockDimX()) +
                             FunctorBase<DeviceType>::GetThreadIdX();

    int8_t *sm_p = (int8_t *)FunctorBase<DeviceType>::GetSharedMemory();
    ranges_sm = (mgard_x::SIZE *)sm_p;
    sm_p += D * (l_target + 2) * sizeof(mgard_x::SIZE);

    for (mgard_x::SIZE i = threadId; i < D * (l_target + 2);
         i += FunctorBase<DeviceType>::GetBlockDimX() *
              FunctorBase<DeviceType>::GetBlockDimY() *
              FunctorBase<DeviceType>::GetBlockDimZ()) {
      ranges_sm[i] = *ranges(i);
    }

    __syncthreads();
    if (debug) {
      printf("l_target = %u\n", l_target);
      for (int d = 0; d < D; d++) {
        printf("ranges_sm[d = %d]: ", d);
        for (int l = 0; l < l_target + 2; l++) {
          printf("%u ", ranges_sm[d * (l_target + 2) + l]);
        }
        printf("\n");
      }
    }

    __syncthreads();
  }

  MGARDX_EXEC void Operation2() {

    mgard_x::SIZE idx[D];

    mgard_x::SIZE firstD = mgard_x::div_roundup(ranges_sm[l_target + 1], F);
    if (debug) {
      // printf("ranges_sm[l_target + 1]: %u\n", ranges_sm[l_target + 1]);
    }
    mgard_x::SIZE bidx = FunctorBase<DeviceType>::GetBlockIdX();
    idx[0] = (bidx % firstD) * F + FunctorBase<DeviceType>::GetThreadIdX();
    bidx /= firstD;
    if (D >= 2) {
      idx[1] = FunctorBase<DeviceType>::GetBlockIdY() *
                   FunctorBase<DeviceType>::GetBlockDimY() +
               FunctorBase<DeviceType>::GetThreadIdY();
    }
    if (D >= 3) {
      idx[2] = FunctorBase<DeviceType>::GetBlockIdZ() *
                   FunctorBase<DeviceType>::GetBlockDimZ() +
               FunctorBase<DeviceType>::GetThreadIdZ();
    }

    for (mgard_x::DIM d = 3; d < D; d++) {
      idx[d] = bidx % ranges_sm[(l_target + 2) * d + l_target + 1];
      bidx /= ranges_sm[(l_target + 2) * d + l_target + 1];
    }

    bool in_range = true;
    for (mgard_x::DIM d = 0; d < D; d++) {
      if (idx[d] >= ranges_sm[(l_target + 2) * d + l_target + 1]) {
        in_range = false;
      }
    }

    if (in_range) {
      mgard_x::SIZE level = 0;
      long long unsigned int l_bit[D];
      for (mgard_x::DIM d = 0; d < D; d++) {
        l_bit[d] = 0l;
        for (mgard_x::SIZE l = 0; l < l_target + 1; l++) {
          long long unsigned int bit =
              (idx[d] >= ranges_sm[(l_target + 2) * d + l]) &&
              (idx[d] < ranges_sm[(l_target + 2) * d + l + 1]);
          l_bit[d] += bit << l;
        }
        level = max((int)level, __ffsll(l_bit[d]));
      }

      // distinguish different regions
      mgard_x::SIZE curr_region = 0;
      for (mgard_x::DIM d = 0; d < D; d++) {
        mgard_x::SIZE bit = !(level == __ffsll(l_bit[d]));
        curr_region += bit << (D - 1 - d);
      }

      // convert to 0 based level
      level = level - 1;

      // region size
      mgard_x::SIZE coarse_level_size[D];
      mgard_x::SIZE diff_level_size[D];
      mgard_x::SIZE curr_region_dims[D];
      for (mgard_x::DIM d = 0; d < D; d++) {
        coarse_level_size[d] = ranges_sm[(l_target + 2) * d + level];
        diff_level_size[d] = ranges_sm[(l_target + 2) * d + level + 1] -
                             ranges_sm[(l_target + 2) * d + level];
      }

      for (mgard_x::DIM d = 0; d < D; d++) {
        mgard_x::SIZE bit = (curr_region >> (D - 1 - d)) & 1u;
        curr_region_dims[d] = bit ? coarse_level_size[d] : diff_level_size[d];
      }

      mgard_x::SIZE curr_region_size = 1;
      for (mgard_x::DIM d = 0; d < D; d++) {
        curr_region_size *= curr_region_dims[d];
      }

      // printf("(%u %u %u): level: %u, region: %u, dims: %u %u %u
      // coarse_level_size: %u %u %u, diff_level_size: %u %u %u\n", idx[0],
      // idx[1], idx[2],
      //                                           level, curr_region,
      //                                           curr_region_dims[0],
      //                                           curr_region_dims[1],
      //                                           curr_region_dims[2],
      //                                           coarse_level_size[0],
      //                                           coarse_level_size[1],
      //                                           coarse_level_size[2],
      //                                           diff_level_size[0],
      //                                           diff_level_size[1],
      //                                           diff_level_size[2]);

      // region offset
      mgard_x::SIZE curr_region_offset = 0;
      for (mgard_x::SIZE prev_region = 0; prev_region < curr_region;
           prev_region++) {
        mgard_x::SIZE prev_region_size = 1;
        for (mgard_x::DIM d = 0; d < D; d++) {
          mgard_x::SIZE bit = (prev_region >> (D - 1 - d)) & 1u;
          prev_region_size *= bit ? coarse_level_size[d] : diff_level_size[d];
        }
        curr_region_offset += prev_region_size;
      }

      // printf("(%u %u %u): level: %u, curr_region_offset: %u\n", idx[0],
      // idx[1], idx[2],
      //                                           level, curr_region_offset);

      // thread offset
      mgard_x::SIZE curr_region_thread_idx[D];
      mgard_x::SIZE curr_thread_offset = 0;
      for (mgard_x::SIZE d = 0; d < D; d++) {
        mgard_x::SIZE bit = (curr_region >> D - 1 - d) & 1u;
        curr_region_thread_idx[d] =
            bit ? idx[d] : idx[d] - coarse_level_size[d];
      }

      mgard_x::SIZE stride = 1;
      for (mgard_x::SIZE d = 0; d < D; d++) {
        curr_thread_offset += curr_region_thread_idx[d] * stride;
        stride *= curr_region_dims[d];
      }

      mgard_x::SIZE level_offset = curr_region_offset + curr_thread_offset;

      // printf("(%u %u %u): level: %u, region: %u, size: %u, region_offset: %u,
      // thread_offset: %u, level_offset: %u, l_bit %llu %llu %llu\n",
      //                                           idx[0], idx[1], idx[2],
      //                                           level, curr_region,
      //                                           curr_region_size,
      //                                           curr_region_offset,
      //                                           curr_thread_offset,
      //                                           level_offset, l_bit[0],
      //                                           l_bit[1], l_bit[2]);

      if (Direction == Interleave) {
        // printf("%u %u %u (%f) --> %u\n", idx[0], idx[1], idx[2], *v(idx),
        // level_offset);
        *(level_v[level]((mgard_x::IDX)level_offset)) = *v(idx);
      } else if (Direction == Reposition) {
        *v(idx) = *(level_v[level]((mgard_x::IDX)level_offset));
      }
    }
  }

  MGARDX_EXEC void Operation3() {}

  MGARDX_EXEC void Operation4() {}

  MGARDX_EXEC void Operation5() {}

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    size += D * (l_target + 2) * sizeof(mgard_x::SIZE);
    printf("sm_size: %llu\n", size);
    return size;
  }

private:
  mgard_x::SubArray<1, mgard_x::SIZE, DeviceType> ranges;
  mgard_x::SIZE l_target;
  mgard_x::SubArray<D, T, DeviceType> v;
  mgard_x::SubArray<1, T, DeviceType> *level_v;

  // thread private variables
  bool debug;
  mgard_x::SIZE *ranges_sm;
};

template <mgard_x::DIM D, typename T, mgard_x::OPTION Direction,
          typename DeviceType>
class DirectInterleaverKernel : public mgard_x::AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  DirectInterleaverKernel() : mgard_x::AutoTuner<DeviceType>() {}

  template <mgard_x::SIZE R, mgard_x::SIZE C, mgard_x::SIZE F>
  MGARDX_CONT mgard_x::Task<
      DirectInterleaverFunctor<D, T, R, C, F, Direction, DeviceType>>
  GenTask(mgard_x::SubArray<1, mgard_x::SIZE, DeviceType> shape,
          mgard_x::SIZE l_target,
          mgard_x::SubArray<1, mgard_x::SIZE, DeviceType> ranges,
          mgard_x::SubArray<D, T, DeviceType> v,
          mgard_x::SubArray<1, T, DeviceType> *level_v, int queue_idx) {
    using FunctorType =
        DirectInterleaverFunctor<D, T, R, C, F, Direction, DeviceType>;
    FunctorType functor(ranges, l_target, v, level_v);
    mgard_x::SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    int total_thread_z = shape.dataHost()[2];
    int total_thread_y = shape.dataHost()[1];
    int total_thread_x = shape.dataHost()[0];
    // linearize other dimensions
    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);
    printf("DirectInterleaverKernel config: %u %u %u %u %u %u\n", tbx, tby, tbz,
           gridx, gridy, gridz);
    for (int d = 3; d < D; d++) {
      gridx *= shape.dataHost()[d];
    }
    return mgard_x::Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size,
                         queue_idx);
  }

  MGARDX_CONT
  void Execute(mgard_x::SubArray<1, mgard_x::SIZE, DeviceType> shape,
               mgard_x::SIZE l_target,
               mgard_x::SubArray<1, mgard_x::SIZE, DeviceType> ranges,
               mgard_x::SubArray<D, T, DeviceType> v,
               mgard_x::SubArray<1, T, DeviceType> *level_v, int queue_idx) {
#define KERNEL(R, C, F)                                                        \
  {                                                                            \
    using FunctorType =                                                        \
        DirectInterleaverFunctor<D, T, R, C, F, Direction, DeviceType>;        \
    using TaskType = mgard_x::Task<FunctorType>;                               \
    TaskType task =                                                            \
        GenTask<R, C, F>(shape, l_target, ranges, v, level_v, queue_idx);      \
    mgard_x::DeviceAdapter<TaskType, DeviceType> adapter;                      \
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
template <typename HandleType, mgard_x::DIM D, typename T>
class DirectInterleaver
    : public concepts::InterleaverInterface<HandleType, D, T> {
public:
  DirectInterleaver(HandleType &handle) : handle(handle) {}
  void
  interleave(mgard_x::SubArray<D, T, mgard_x::CUDA> decomposed_data,
             mgard_x::SubArray<1, T, mgard_x::CUDA> *levels_decomposed_data,
             int queue_idx) const {
    // mgard_x::PrintSubarray("decomposed_data", decomposed_data);
    mgard_x::SubArray<1, T, mgard_x::CUDA> *levels_decomposed_data_device;
    // mgard_x::cudaMallocHelper(this->handle,
    // (void**)&levels_decomposed_data_device,
    //                     sizeof(mgard_x::SubArray<1, T,
    //                     mgard_x::CUDA>)*(this->handle.l_target+1));
    // mgard_x::cudaMemcpyAsyncHelper(this->handle,
    // levels_decomposed_data_device, levels_decomposed_data,
    //                                   sizeof(mgard_x::SubArray<1, T,
    //                                   mgard_x::CUDA>)*(this->handle.l_target+1),
    //                                   mgard_x::AUTO, queue_idx);
    // handle.sync(queue_idx);

    MemoryManager<CUDA>::Malloc1D(levels_decomposed_data_device,
                                  this->handle.l_target + 1, 0);
    MemoryManager<CUDA>::Copy1D(levels_decomposed_data_device,
                                levels_decomposed_data,
                                this->handle.l_target + 1, 0);
    DeviceRuntime<CUDA>::SyncQueue(0);

    DirectInterleaverKernel<D, T, Interleave, mgard_x::CUDA>().Execute(
        mgard_x::SubArray<1, mgard_x::SIZE, mgard_x::CUDA>(handle.shapes[0],
                                                           true),
        handle.l_target,
        mgard_x::SubArray<1, mgard_x::SIZE, mgard_x::CUDA>(handle.ranges),
        decomposed_data, levels_decomposed_data_device, queue_idx);

    DeviceRuntime<CUDA>::SyncQueue(queue_idx);
    // handle.sync(queue_idx);
    // for (int i = 0; i < this->handle.l_target+1; i++) {
    //   printf("l = %d\n", i);
    //   mgard_x::PrintSubarray("levels_decomposed_data",
    //   levels_decomposed_data[i]);
    // }
  }
  void
  reposition(mgard_x::SubArray<1, T, mgard_x::CUDA> *levels_decomposed_data,
             mgard_x::SubArray<D, T, mgard_x::CUDA> decomposed_data,
             int queue_idx) const {

    mgard_x::SubArray<1, T, mgard_x::CUDA> *levels_decomposed_data_device;
    // mgard_x::cudaMallocHelper(this->handle,
    // (void**)&levels_decomposed_data_device,
    //                     sizeof(mgard_x::SubArray<1, T,
    //                     mgard_x::CUDA>)*(this->handle.l_target+1));
    // mgard_x::cudaMemcpyAsyncHelper(this->handle,
    // levels_decomposed_data_device, levels_decomposed_data,
    //                                   sizeof(mgard_x::SubArray<1, T,
    //                                   mgard_x::CUDA>)*(this->handle.l_target+1),
    //                                   mgard_x::AUTO, queue_idx);
    // handle.sync(queue_idx);

    MemoryManager<CUDA>::Malloc1D(levels_decomposed_data_device,
                                  this->handle.l_target + 1, 0);
    MemoryManager<CUDA>::Copy1D(levels_decomposed_data_device,
                                levels_decomposed_data,
                                this->handle.l_target + 1, 0);
    DeviceRuntime<CUDA>::SyncQueue(0);

    DirectInterleaverKernel<D, T, Reposition, mgard_x::CUDA>().Execute(
        mgard_x::SubArray<1, mgard_x::SIZE, mgard_x::CUDA>(handle.shapes[0],
                                                           true),
        handle.l_target,
        mgard_x::SubArray<1, mgard_x::SIZE, mgard_x::CUDA>(handle.ranges),
        decomposed_data, levels_decomposed_data_device, queue_idx);
    DeviceRuntime<CUDA>::SyncQueue(queue_idx);
  }
  void print() const { std::cout << "Direct interleaver" << std::endl; }

private:
  HandleType &handle;
};

} // namespace MDR
} // namespace mgard_m
#endif
