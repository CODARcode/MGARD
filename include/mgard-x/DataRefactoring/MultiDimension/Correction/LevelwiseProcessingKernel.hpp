/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_LEVELWISE_PROCESSING_KERNEL_TEMPLATE
#define MGARD_X_LEVELWISE_PROCESSING_KERNEL_TEMPLATE

#include "../../../RuntimeX/RuntimeX.h"

namespace mgard_x {

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, OPTION OP,
          typename DeviceType>
class LwpkReoFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT LwpkReoFunctor() {}
  MGARDX_CONT LwpkReoFunctor(SubArray<D, T, DeviceType> v,
                             SubArray<D, T, DeviceType> work)
      : v(v), work(work) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    threadId = (FunctorBase<DeviceType>::GetThreadIdZ() *
                (FunctorBase<DeviceType>::GetBlockDimX() *
                 FunctorBase<DeviceType>::GetBlockDimY())) +
               (FunctorBase<DeviceType>::GetThreadIdY() *
                FunctorBase<DeviceType>::GetBlockDimX()) +
               FunctorBase<DeviceType>::GetThreadIdX();

    SIZE idx[D];
    SIZE firstD = div_roundup(v.shape(D-1), F);

    SIZE bidx = FunctorBase<DeviceType>::GetBlockIdX();
    // idx[0] = (bidx % firstD) * F + FunctorBase<DeviceType>::GetThreadIdX();
    idx[D-1] = (bidx % firstD) * F + FunctorBase<DeviceType>::GetThreadIdX();

    // printf("firstD %d idx[0] %d\n", firstD, idx[0]);

    bidx /= firstD;
    if (D >= 2)
      idx[D-2] = FunctorBase<DeviceType>::GetBlockIdY() *
                   FunctorBase<DeviceType>::GetBlockDimY() +
               FunctorBase<DeviceType>::GetThreadIdY();
    if (D >= 3)
      idx[D-3] = FunctorBase<DeviceType>::GetBlockIdZ() *
                   FunctorBase<DeviceType>::GetBlockDimZ() +
               FunctorBase<DeviceType>::GetThreadIdZ();

    // for (DIM d = 3; d < D; d++) {
    //   idx[d] = bidx % v.getShape(d);
    //   bidx /= v.getShape(d);
    // }
    for (int d = D-4; d >= 0; d--) {
      idx[d] = bidx % v.shape(d);
      bidx /= v.shape(d);
    }
    // int z = blockIdx.z * blockDim.z + threadIdx.z;
    // int y = blockIdx.y * blockDim.y + threadIdx.y;
    // int x = blockIdx.z * blockDim.z + threadIdx.z;
    bool in_range = true;
    // for (DIM d = 0; d < D; d++) {
    //   if (idx[d] >= v.getShape(d))
    //     in_range = false;
    // }

    for (DIM d = 0; d < D; d++) {
      if (idx[d] >= v.shape(d))
        in_range = false;
    }
    if (in_range) {
      // printf("%d %d %d %d\n", idx[3], idx[2], idx[1], idx[0]);
      if (OP == COPY) {
        // *work(idx) = *v(idx);
        work[idx] = v[idx];
      }
      if (OP == ADD) {
        work[idx] += v[idx];
      }
      if (OP == SUBTRACT) {
        work[idx] -= v[idx];
      }
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    return size;
  }

private:
  SubArray<D, T, DeviceType> v, work;
  IDX threadId;
};

template <DIM D, typename T, OPTION OP, typename DeviceType>
class LwpkReo : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  LwpkReo() : AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<LwpkReoFunctor<D, T, R, C, F, OP, DeviceType>>
  GenTask(SubArray<D, T, DeviceType> v, SubArray<D, T, DeviceType> work,
          int queue_idx) {
    using FunctorType = LwpkReoFunctor<D, T, R, C, F, OP, DeviceType>;
    FunctorType functor(v, work);

    SIZE total_thread_z = 1;
    SIZE total_thread_y = 1;
    SIZE total_thread_x = 1;
    if (D >= 3)
      total_thread_z = v.shape(D-3);
    if (D >= 2)
      total_thread_y = v.shape(D-2);
    total_thread_x = v.shape(D-1);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);
    for (int d = D-4; d >= 0; d--) {
      gridx *= v.shape(d);
    }
    // printf("%u %u %u\n", shape.dataHost()[2], shape.dataHost()[1],
    // shape.dataHost()[0]); PrintSubarray("shape", shape);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "LwpkReo");
  }

  MGARDX_CONT
  void Execute(SubArray<D, T, DeviceType> v, SubArray<D, T, DeviceType> work,
               int queue_idx) {

    int range_l = std::min(6, (int)std::log2(v.shape(D-1)) - 1);
    int prec = TypeToIdx<T>();
    int config = AutoTuner<DeviceType>::autoTuningTable.lwpk[prec][range_l];

    while (LWPK_CONFIG[D - 1][config][0] * LWPK_CONFIG[D - 1][config][1] *
               LWPK_CONFIG[D - 1][config][2] >
           DeviceRuntime<DeviceType>::GetMaxNumThreadsPerTB()) {
      config--;
      if (config < 0) {
        std::cout << log::log_err
                  << "Cannot find suitable config for LwpkReo.\n";
      }
    }

    double min_time = std::numeric_limits<double>::max();
    int min_config = 0;

#define LWPK(CONFIG)                                                           \
  if (config == CONFIG || AutoTuner<DeviceType>::ProfileKernels) {             \
    const int R = LWPK_CONFIG[D - 1][CONFIG][0];                               \
    const int C = LWPK_CONFIG[D - 1][CONFIG][1];                               \
    const int F = LWPK_CONFIG[D - 1][CONFIG][2];                               \
    using FunctorType = LwpkReoFunctor<D, T, R, C, F, OP, DeviceType>;         \
    using TaskType = Task<FunctorType>;                                        \
    TaskType task = GenTask<R, C, F>(v, work, queue_idx);                      \
    DeviceAdapter<TaskType, DeviceType> adapter;                               \
    ExecutionReturn ret = adapter.Execute(task);                               \
    if (AutoTuner<DeviceType>::ProfileKernels) {                               \
      if (min_time > ret.execution_time) {                                     \
        min_time = ret.execution_time;                                         \
        min_config = CONFIG;                                                   \
      }                                                                        \
    }                                                                          \
  }

    LWPK(0)
    LWPK(1)
    LWPK(2)
    LWPK(3)
    LWPK(4)
    LWPK(5)
    LWPK(6)
#undef LWPK

    if (AutoTuner<DeviceType>::ProfileKernels) {
      FillAutoTunerTable<DeviceType>("lwpk", prec, range_l, min_config);
    }
  }
};

template <mgard_x::DIM D, typename T, int R, int C, int F, OPTION OP,
          typename DeviceType>
class LevelwiseCalcNDFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT
  LevelwiseCalcNDFunctor(SIZE *shape, SubArray<D, T, DeviceType> v,
                         SubArray<D, T, DeviceType> w)
      : shape(shape), v(v), w(w) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    threadId = (FunctorBase<DeviceType>::GetThreadIdZ() *
                (FunctorBase<DeviceType>::GetBlockDimX() *
                 FunctorBase<DeviceType>::GetBlockDimY())) +
               (FunctorBase<DeviceType>::GetThreadIdY() *
                FunctorBase<DeviceType>::GetBlockDimX()) +
               FunctorBase<DeviceType>::GetThreadIdX();

    int8_t *sm_p = (int8_t *)FunctorBase<DeviceType>::GetSharedMemory();
    shape_sm = (SIZE *)sm_p;
    sm_p += D * sizeof(SIZE);

    if (threadId < D) {
      shape_sm[threadId] = shape[threadId];
    }
  }

  MGARDX_EXEC void Operation2() {

    SIZE firstD = div_roundup(shape_sm[0], F);

    SIZE bidx = FunctorBase<DeviceType>::GetBlockIdX();
    idx[0] = (bidx % firstD) * F + FunctorBase<DeviceType>::GetThreadIdX();

    // printf("firstD %d idx[0] %d\n", firstD, idx[0]);

    bidx /= firstD;
    if (D >= 2)
      idx[1] = FunctorBase<DeviceType>::GetBlockIdY() *
                   FunctorBase<DeviceType>::GetBlockDimY() +
               FunctorBase<DeviceType>::GetThreadIdY();
    if (D >= 3)
      idx[2] = FunctorBase<DeviceType>::GetBlockIdZ() *
                   FunctorBase<DeviceType>::GetBlockDimZ() +
               FunctorBase<DeviceType>::GetThreadIdZ();

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

  MGARDX_EXEC void Operation3() {}

  MGARDX_EXEC void Operation4() {}

  MGARDX_EXEC void Operation5() {}

  MGARDX_CONT size_t shared_memory_size() {
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
class LevelwiseCalcNDKernel : public AutoTuner<DeviceType> {

public:
  MGARDX_CONT
  LevelwiseCalcNDKernel() : AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<LevelwiseCalcNDFunctor<D, T, R, C, F, Direction, DeviceType>>
  GenTask(SIZE *shape_h, SIZE *shape_d, SubArray<D, T, DeviceType> v,
          SubArray<D, T, DeviceType> w, int queue_idx) {
    using FunctorType =
        LevelwiseCalcNDFunctor<D, T, R, C, F, Direction, DeviceType>;
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
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size,
                queue_idx);
  }

  MGARDX_CONT
  void Execute(SIZE *shape_h, SIZE *shape_d, SubArray<D, T, DeviceType> v,
               SubArray<D, T, DeviceType> w, int queue_idx) {
#define KERNEL(R, C, F)                                                        \
  {                                                                            \
    using FunctorType =                                                        \
        LevelwiseCalcNDFunctor<D, T, R, C, F, Direction, DeviceType>;          \
    using TaskType = Task<FunctorType>;                                        \
    TaskType task = GenTask<R, C, F>(shape_h, shape_d, v, w, queue_idx);       \
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

} // namespace mgard_x

#endif