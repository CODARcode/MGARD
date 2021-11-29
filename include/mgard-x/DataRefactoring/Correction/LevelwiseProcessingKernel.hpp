/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_LEVELWISE_PROCESSING_KERNEL_TEMPLATE
#define MGARD_X_LEVELWISE_PROCESSING_KERNEL_TEMPLATE

#include "../../RuntimeX/RuntimeX.h"

namespace mgard_x {

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, OPTION OP, typename DeviceType>
class LwpkReoFunctor: public Functor<DeviceType> {
  public:

  MGARDX_CONT LwpkReoFunctor() {}
  MGARDX_CONT LwpkReoFunctor(SubArray<1, SIZE, DeviceType> shape, 
                              SubArray<D, T, DeviceType> v, SubArray<D, T, DeviceType> work):
                              shape(shape), v(v), work(work) {
    Functor<DeviceType>();                            
  }

  MGARDX_EXEC void
  Operation1() {
    threadId = (FunctorBase<DeviceType>::GetThreadIdZ() * (FunctorBase<DeviceType>::GetBlockDimX() * FunctorBase<DeviceType>::GetBlockDimY())) +
                    (FunctorBase<DeviceType>::GetThreadIdY() * FunctorBase<DeviceType>::GetBlockDimX()) + FunctorBase<DeviceType>::GetThreadIdX();

    SIZE * sm = (SIZE*)FunctorBase<DeviceType>::GetSharedMemory();
    shape_sm = sm;

    if (threadId < D) {
      shape_sm[threadId] = *shape(threadId);
    }
  }

  MGARDX_EXEC void
  Operation2() {
    SIZE idx[D];
    SIZE firstD = div_roundup(shape_sm[0], F);

    SIZE bidx = FunctorBase<DeviceType>::GetBlockIdX();
    idx[0] = (bidx % firstD) * F + FunctorBase<DeviceType>::GetThreadIdX();

    // printf("firstD %d idx[0] %d\n", firstD, idx[0]);

    bidx /= firstD;
    if (D >= 2)
      idx[1] = FunctorBase<DeviceType>::GetBlockIdY() * FunctorBase<DeviceType>::GetBlockDimY() + FunctorBase<DeviceType>::GetThreadIdY();
    if (D >= 3)
      idx[2] = FunctorBase<DeviceType>::GetBlockIdZ() * FunctorBase<DeviceType>::GetBlockDimZ() + FunctorBase<DeviceType>::GetThreadIdZ();

    for (DIM d = 3; d < D; d++) {
      idx[d] = bidx % shape_sm[d];
      bidx /= shape_sm[d];
    }
    // int z = blockIdx.z * blockDim.z + threadIdx.z;
    // int y = blockIdx.y * blockDim.y + threadIdx.y;
    // int x = blockIdx.z * blockDim.z + threadIdx.z;
    bool in_range = true;
    for (DIM d = 0; d < D; d++) {
      if (idx[d] >= shape_sm[d])
        in_range = false;
    }
    if (in_range) {
      // printf("%d %d %d %d\n", idx[3], idx[2], idx[1], idx[0]);
      if (OP == COPY)
        *work(idx) = *v(idx);
      if (OP == ADD)
        *work(idx) += *v(idx);
      if (OP == SUBTRACT)
        *work(idx) -= *v(idx);
    }
  }

  MGARDX_EXEC void
  Operation3() {}

  MGARDX_EXEC void
  Operation4() {}

  MGARDX_EXEC void
  Operation5() {}

  MGARDX_CONT size_t
  shared_memory_size() {
    size_t size = 0;
    size = D * sizeof(SIZE);
    return size;
  }

  private:
  SubArray<1, SIZE, DeviceType> shape;
  SubArray<D, T, DeviceType> v, work;

  IDX threadId;
  SIZE *shape_sm;
};

template <DIM D, typename T, OPTION OP, typename DeviceType>
class LwpkReo: public AutoTuner<DeviceType> {
  public:
  MGARDX_CONT
  LwpkReo():AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT
  Task<LwpkReoFunctor<D, T, R, C, F, OP, DeviceType> > 
  GenTask(SubArray<1, SIZE, DeviceType> shape,
          SubArray<D, T, DeviceType> v, SubArray<D, T, DeviceType> work,
          int queue_idx) {
    using FunctorType = LwpkReoFunctor<D, T, R, C, F, OP, DeviceType>;
    FunctorType functor(shape, v, work);

    SIZE total_thread_z = shape.dataHost()[2];
    SIZE total_thread_y = shape.dataHost()[1];
    SIZE total_thread_x = shape.dataHost()[0];

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);
    for (DIM d = 3; d < D; d++) {
      gridx *= shape.dataHost()[d];
    }
    // printf("%u %u %u\n", shape.dataHost()[2], shape.dataHost()[1], shape.dataHost()[0]);
    // PrintSubarray("shape", shape);
    return Task(functor, gridz, gridy, gridx, 
                tbz, tby, tbx, sm_size, queue_idx, "LwpkReo"); 
  }

  MGARDX_CONT
  void Execute(SubArray<1, SIZE, DeviceType> shape,
              SubArray<D, T, DeviceType> v, SubArray<D, T, DeviceType> work,
              int queue_idx) {
    const int R=LWPK_CONFIG[D-1][0];
    const int C=LWPK_CONFIG[D-1][1];
    const int F=LWPK_CONFIG[D-1][2];
    using FunctorType = LwpkReoFunctor<D, T, R, C, F, OP, DeviceType>;
    using TaskType = Task<FunctorType>;
    TaskType task = GenTask<R, C, F>(shape, v, work, queue_idx); 
    DeviceAdapter<TaskType, DeviceType> adapter; 
    adapter.Execute(task);
  }
};


// template <DIM D, typename T, SIZE R, SIZE C, SIZE F, int OP>
// __global__ void _lwpk(SIZE *shape, T *dv, SIZE *ldvs, T *dwork, SIZE *ldws) {

//   size_t threadId = (threadIdx.z * (blockDim.x * blockDim.y)) +
//                     (threadIdx.y * blockDim.x) + threadIdx.x;
//   SIZE *sm = SharedMemory<SIZE>();
//   SIZE *shape_sm = sm;
//   SIZE *ldvs_sm = shape_sm + D;
//   SIZE *ldws_sm = ldvs_sm + D;

//   if (threadId < D) {
//     shape_sm[threadId] = shape[threadId];
//     ldvs_sm[threadId] = ldvs[threadId];
//     ldws_sm[threadId] = ldws[threadId];
//   }
//   __syncthreads();

//   SIZE idx[D];
//   SIZE firstD = div_roundup(shape_sm[0], F);

//   SIZE bidx = FunctorBase<DeviceType>::GetBlockIdX();
//   idx[0] = (bidx % firstD) * F + threadIdx.x;

//   // printf("firstD %d idx[0] %d\n", firstD, idx[0]);

//   bidx /= firstD;
//   if (D >= 2)
//     idx[1] = blockIdx.y * blockDim.y + threadIdx.y;
//   if (D >= 3)
//     idx[2] = blockIdx.z * blockDim.z + threadIdx.z;

//   for (DIM d = 3; d < D; d++) {
//     idx[d] = bidx % shape_sm[d];
//     bidx /= shape_sm[d];
//   }
//   // int z = blockIdx.z * blockDim.z + threadIdx.z;
//   // int y = blockIdx.y * blockDim.y + threadIdx.y;
//   // int x = blockIdx.z * blockDim.z + threadIdx.z;
//   bool in_range = true;
//   for (DIM d = 0; d < D; d++) {
//     if (idx[d] >= shape_sm[d])
//       in_range = false;
//   }
//   if (in_range) {
//     // printf("%d %d %d %d\n", idx[3], idx[2], idx[1], idx[0]);
//     if (OP == COPY)
//       dwork[get_idx<D>(ldws, idx)] = dv[get_idx<D>(ldvs, idx)];
//     if (OP == ADD)
//       dwork[get_idx<D>(ldws, idx)] += dv[get_idx<D>(ldvs, idx)];
//     if (OP == SUBTRACT)
//       dwork[get_idx<D>(ldws, idx)] -= dv[get_idx<D>(ldvs, idx)];
//   }
// }

// template <DIM D, typename T, SIZE R, SIZE C, SIZE F, int OP>
// void lwpk_adaptive_launcher(Handle<D, T> &handle, SIZE *shape_h, SIZE *shape_d,
//                             T *dv, SIZE *ldvs, T *dwork, SIZE *ldws,
//                             int queue_idx) {

//   SIZE total_thread_z = shape_h[2];
//   SIZE total_thread_y = shape_h[1];
//   SIZE total_thread_x = shape_h[0];
//   // linearize other dimensions
//   SIZE tbz = R;
//   SIZE tby = C;
//   SIZE tbx = F;
//   SIZE gridz = ceil((float)total_thread_z / tbz);
//   SIZE gridy = ceil((float)total_thread_y / tby);
//   SIZE gridx = ceil((float)total_thread_x / tbx);
//   for (DIM d = 3; d < D; d++) {
//     gridx *= shape_h[d];
//   }

//   // printf("exec: %d %d %d %d %d %d\n", tbx, tby, tbz, gridx, gridy, gridz);
//   dim3 threadsPerBlock(tbx, tby, tbz);
//   dim3 blockPerGrid(gridx, gridy, gridz);
//   size_t sm_size = (D * 3) * sizeof(SIZE);
//   _lwpk<D, T, R, C, F, OP><<<blockPerGrid, threadsPerBlock, sm_size,
//                              *(cudaStream_t *)handle.get(queue_idx)>>>(
//       shape_d, dv, ldvs, dwork, ldws);

//   gpuErrchk(cudaGetLastError());
//   if (handle.sync_and_check_all_kernels) {
//     gpuErrchk(cudaDeviceSynchronize());
//   }
// }

// template <DIM D, typename T, int OP>
// void lwpk(Handle<D, T> &handle, SIZE *shape_h, SIZE *shape_d, T *dv, SIZE *ldvs,
//           T *dwork, SIZE *ldws, int queue_idx) {
// #define COPYLEVEL(R, C, F)                                                     \
//   {                                                                            \
//     lwpk_adaptive_launcher<D, T, R, C, F, OP>(handle, shape_h, shape_d, dv,    \
//                                               ldvs, dwork, ldws, queue_idx);   \
//   }
//   if (D >= 3) {
//     COPYLEVEL(4, 4, 4)
//   }
//   if (D == 2) {
//     COPYLEVEL(1, 4, 4)
//   }
//   if (D == 1) {
//     COPYLEVEL(1, 1, 8)
//   }

// #undef COPYLEVEL
// }


template <mgard_x::DIM D, typename T, int R, int C, int F, OPTION OP, typename DeviceType>
class LevelwiseCalcNDFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT
  LevelwiseCalcNDFunctor(SIZE *shape, SubArray<D, T, DeviceType> v, SubArray<D, T, DeviceType> w): 
                        shape(shape), v(v), w(w) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void
  Operation1() {
    threadId = (FunctorBase<DeviceType>::GetThreadIdZ() * (FunctorBase<DeviceType>::GetBlockDimX() * FunctorBase<DeviceType>::GetBlockDimY())) +
                    (FunctorBase<DeviceType>::GetThreadIdY() * FunctorBase<DeviceType>::GetBlockDimX()) + FunctorBase<DeviceType>::GetThreadIdX();

    int8_t * sm_p = (int8_t *)FunctorBase<DeviceType>::GetSharedMemory();
    shape_sm = (SIZE *)sm_p; sm_p += D * sizeof(SIZE);

    if (threadId < D) {
      shape_sm[threadId] = shape[threadId];
    }
  }

  MGARDX_EXEC void
  Operation2() {

    SIZE firstD = div_roundup(shape_sm[0], F);

    SIZE bidx = FunctorBase<DeviceType>::GetBlockIdX();
    idx[0] = (bidx % firstD) * F + FunctorBase<DeviceType>::GetThreadIdX();

    // printf("firstD %d idx[0] %d\n", firstD, idx[0]);

    bidx /= firstD;
    if (D >= 2)
      idx[1] = FunctorBase<DeviceType>::GetBlockIdY() * FunctorBase<DeviceType>::GetBlockDimY() + FunctorBase<DeviceType>::GetThreadIdY();
    if (D >= 3)
      idx[2] = FunctorBase<DeviceType>::GetBlockIdZ() * FunctorBase<DeviceType>::GetBlockDimZ() + FunctorBase<DeviceType>::GetThreadIdZ();

    for (DIM d = 3; d < D; d++) {
      idx[d] = bidx % shape_sm[d];
      bidx /= shape_sm[d];
    }

    bool in_range = true;
    for (DIM d = 0; d < D; d++) {
      if (idx[d] >= shape_sm[d])
        in_range = false;
    }
    if (in_range) {
      // printf("%d %d %d %d\n", idx[3], idx[2], idx[1], idx[0]);
      if (OP == COPY)
        *w(idx) = *v(idx);
      if (OP == ADD)
        *w(idx) += *v(idx);
      if (OP == SUBTRACT)
        *w(idx) -= *v(idx);
    }
  }

  MGARDX_EXEC void
  Operation3() {}

  MGARDX_EXEC void
  Operation4() {}

  MGARDX_EXEC void
  Operation5() {}

  MGARDX_CONT size_t
  shared_memory_size() {
    size_t size = 0;
    size += D * sizeof(SIZE);
    return size;
  }

private:
  SIZE *shape;
  SubArray<D, T, DeviceType> v;
  SubArray<D, T, DeviceType> w;

  SIZE *shape_sm;
  size_t threadId;
  SIZE idx[D];

};

template <DIM D, typename T, OPTION Direction, typename DeviceType>
class LevelwiseCalcNDKernel: public AutoTuner<DeviceType> {

public:
  MGARDX_CONT
  LevelwiseCalcNDKernel(): AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT
  Task<LevelwiseCalcNDFunctor<D, T, R, C, F, Direction, DeviceType>> 
  GenTask(SIZE *shape_h, SIZE *shape_d, SubArray<D, T, DeviceType> v, SubArray<D, T, DeviceType> w, int queue_idx) {
    using FunctorType = LevelwiseCalcNDFunctor<D, T, R, C, F, Direction, DeviceType>;
    FunctorType functor(shape_d, v, w);
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    int total_thread_z = shape_h[2];
    int total_thread_y = shape_h[1];
    int total_thread_x = shape_h[0];
    // linearize other dimensions
    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);
    for (int d = 3; d < D; d++) {
      gridx *= shape_h[d];
    }
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx); 
  }

  MGARDX_CONT
  void Execute(SIZE *shape_h, SIZE *shape_d, SubArray<D, T, DeviceType> v, SubArray<D, T, DeviceType> w, int queue_idx) {
    #define KERNEL(R, C, F)\
    {\
      using FunctorType = LevelwiseCalcNDFunctor<D, T, R, C, F, Direction, DeviceType>;\
      using TaskType = Task<FunctorType>;\
      TaskType task = GenTask<R, C, F>(shape_h, shape_d, v, w, queue_idx);\
      DeviceAdapter<TaskType, DeviceType> adapter; \
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
};

} // namespace mgard_x

#endif