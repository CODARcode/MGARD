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
    SIZE firstD = div_roundup(v.shape(D - 1), F);

    SIZE bidx = FunctorBase<DeviceType>::GetBlockIdX();
    // idx[0] = (bidx % firstD) * F + FunctorBase<DeviceType>::GetThreadIdX();
    idx[D - 1] = (bidx % firstD) * F + FunctorBase<DeviceType>::GetThreadIdX();

    // printf("firstD %d idx[0] %d\n", firstD, idx[0]);

    bidx /= firstD;
    if (D >= 2)
      idx[D - 2] = FunctorBase<DeviceType>::GetBlockIdY() *
                       FunctorBase<DeviceType>::GetBlockDimY() +
                   FunctorBase<DeviceType>::GetThreadIdY();
    if (D >= 3)
      idx[D - 3] = FunctorBase<DeviceType>::GetBlockIdZ() *
                       FunctorBase<DeviceType>::GetBlockDimZ() +
                   FunctorBase<DeviceType>::GetThreadIdZ();

    // for (DIM d = 3; d < D; d++) {
    //   idx[d] = bidx % v.getShape(d);
    //   bidx /= v.getShape(d);
    // }
    for (int d = D - 4; d >= 0; d--) {
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
      total_thread_z = v.shape(D - 3);
    if (D >= 2)
      total_thread_y = v.shape(D - 2);
    total_thread_x = v.shape(D - 1);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);
    for (int d = D - 4; d >= 0; d--) {
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

    int range_l = std::min(6, (int)std::log2(v.shape(D - 1)) - 1);
    int prec = TypeToIdx<T>();
    int config = AutoTuner<DeviceType>::autoTuningTable.lwpk[prec][range_l];
    double min_time = std::numeric_limits<double>::max();
    int min_config = 0;
    ExecutionReturn ret;

#define LWPK(CONFIG)                                                           \
  if (config == CONFIG || AutoTuner<DeviceType>::ProfileKernels) {             \
    const int R = LWPK_CONFIG[D - 1][CONFIG][0];                               \
    const int C = LWPK_CONFIG[D - 1][CONFIG][1];                               \
    const int F = LWPK_CONFIG[D - 1][CONFIG][2];                               \
    using FunctorType = LwpkReoFunctor<D, T, R, C, F, OP, DeviceType>;         \
    using TaskType = Task<FunctorType>;                                        \
    TaskType task = GenTask<R, C, F>(v, work, queue_idx);                      \
    DeviceAdapter<TaskType, DeviceType> adapter;                               \
    ret = adapter.Execute(task);                                               \
    if (AutoTuner<DeviceType>::ProfileKernels) {                               \
      if (ret.success && min_time > ret.execution_time) {                      \
        min_time = ret.execution_time;                                         \
        min_config = CONFIG;                                                   \
      }                                                                        \
    }                                                                          \
  }

    LWPK(6) if (!ret.success) config--;
    LWPK(5) if (!ret.success) config--;
    LWPK(4) if (!ret.success) config--;
    LWPK(3) if (!ret.success) config--;
    LWPK(2) if (!ret.success) config--;
    LWPK(1) if (!ret.success) config--;
    LWPK(0) if (!ret.success) config--;
    if (config < 0 && !ret.success) {
      std::cout << log::log_err << "no suitable config for LwpkReo.\n";
      exit(-1);
    }
#undef LWPK

    if (AutoTuner<DeviceType>::ProfileKernels) {
      FillAutoTunerTable<DeviceType>("lwpk", prec, range_l, min_config);
    }
  }
};

} // namespace mgard_x

#endif