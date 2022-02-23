/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_SINGLE_DIMENSION_MASSTRANS_KERNEL_TEMPLATE
#define MGARD_X_SINGLE_DIMENSION_MASSTRANS_KERNEL_TEMPLATE

#include "../../../RuntimeX/RuntimeX.h"

#include "../../MultiDimension/Correction/LPKFunctor.h"

namespace mgard_x {

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, typename DeviceType>
class SingleDimensionMassTransFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT SingleDimensionMassTransFunctor() {}
  MGARDX_CONT SingleDimensionMassTransFunctor(DIM current_dim,
      SubArray<1, T, DeviceType> dist, SubArray<1, T, DeviceType> ratio, 
      SubArray<D, T, DeviceType> coeff, SubArray<D, T, DeviceType> v)
      : current_dim(current_dim), dist(dist), ratio(ratio), 
        coeff(coeff), v(v) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE v_idx[D];

    SIZE firstD = div_roundup(v.getShape(0), F);

    SIZE bidx = FunctorBase<DeviceType>::GetBlockIdX();
    v_idx[0] =
        (bidx % firstD) * F + FunctorBase<DeviceType>::GetThreadIdX();

    bidx /= firstD;
    if (D >= 2)
      v_idx[1] = FunctorBase<DeviceType>::GetBlockIdY() *
                         FunctorBase<DeviceType>::GetBlockDimY() +
                     FunctorBase<DeviceType>::GetThreadIdY();
    if (D >= 3)
      v_idx[2] = FunctorBase<DeviceType>::GetBlockIdZ() *
                         FunctorBase<DeviceType>::GetBlockDimZ() +
                     FunctorBase<DeviceType>::GetThreadIdZ();

    for (DIM d = 3; d < D; d++) {
      v_idx[d] = bidx % v.getShape(d);
      bidx /= v.getShape(d);
    }

    bool in_range = true;
    for (DIM d = 0; d < D; d++) {
      if (v_idx[d] >= v.getShape(d))
        in_range = false;
    }

    if (in_range) {
      
      const T a = 0.0;
      const T c = 0.0;
      const T e = 0.0;
      T b = 0.0;
      T d = 0.0;


      if (v_idx[current_dim] > 0 && v_idx[current_dim] < coeff.getShape(current_dim)) {
        v_idx[current_dim] --;
        b = *coeff(v_idx);
        v_idx[current_dim] ++;
      }  

      if (v_idx[current_dim] < coeff.getShape(current_dim)) {
        // v_idx[current_dim] ++;
        d = *coeff(v_idx);
        // v_idx[current_dim] --;
      }  

      T h1 = 0, h2 = 0, h3 = 0, h4 = 0;
      T r1 = 0, r2 = 0, r3 = 0, r4 = 0;

      if (v_idx[current_dim] > 0 && v_idx[current_dim] * 2 < coeff.getShape(current_dim) + v.getShape(current_dim) - 1) {
        h1 = *dist(v_idx[current_dim] * 2 - 2);
        h2 = *dist(v_idx[current_dim] * 2 - 1);
        r1 = *ratio(v_idx[current_dim] * 2 - 2);
        r2 = *ratio(v_idx[current_dim] * 2 - 1);
      }
      if (v_idx[current_dim] * 2 < coeff.getShape(current_dim) + v.getShape(current_dim) - 1) {
        h3 = *dist(v_idx[current_dim] * 2);
        h4 = *dist(v_idx[current_dim] * 2 + 1);
        r3 = *ratio(v_idx[current_dim] * 2);
        r4 = 1 - r3;
      }

      // printf("v_idx = [%d %d] %f %f %f %f %f f_sm_h %f %f %f %f f_sm_r %f %f %f %f, out: %f\n",
      //   v_idx[1], v_idx[0], a,b,c,d,e, h1,h2,h3,h4, r1,r2,r3,r4,
      //   mass_trans(a, b, c, d, e, h1, h2, h3, h4, r1, r2, r3, r4));

      *v(v_idx) =
          mass_trans(a, b, c, d, e, h1, h2, h3, h4, r1, r2, r3, r4);
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  // functor parameters
  DIM current_dim;
  SubArray<1, T, DeviceType> dist;
  SubArray<1, T, DeviceType> ratio;
  SubArray<D, T, DeviceType> coeff;
  SubArray<D, T, DeviceType> v;
};


template <DIM D, typename T, typename DeviceType>
class SingleDimensionMassTrans : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  SingleDimensionMassTrans() : AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<SingleDimensionMassTransFunctor<D, T, R, C, F, DeviceType>>
  GenTask(DIM current_dim,
          SubArray<1, T, DeviceType> dist, SubArray<1, T, DeviceType> ratio, 
          SubArray<D, T, DeviceType> coeff, SubArray<D, T, DeviceType> v,
          int queue_idx) {

    using FunctorType =
        SingleDimensionMassTransFunctor<D, T, R, C, F, DeviceType>;
    FunctorType functor(current_dim, dist, ratio, coeff, v);

    SIZE nr = 1, nc = 1, nf = 1;
    if (D >= 3) nr = v.getShape(2);
    if (D >= 2) nc = v.getShape(1);
    nf = v.getShape(0);

    SIZE total_thread_z = nr;
    SIZE total_thread_y = nc;
    SIZE total_thread_x = nf;

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();

    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);

    for (DIM d = 3; d < D; d++) {
        gridx *= coeff.getShape(d);
    }

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "SingleDimensionMassTrans");
  }

  MGARDX_CONT
  void Execute(DIM current_dim,
               SubArray<1, T, DeviceType> dist, SubArray<1, T, DeviceType> ratio, 
               SubArray<D, T, DeviceType> coeff, SubArray<D, T, DeviceType> v,
               int queue_idx) {
    int range_l = std::min(6, (int)std::log2(v.getShape(0)) - 1);
    int arch = DeviceRuntime<DeviceType>::GetArchitectureGeneration();
    int prec = TypeToIdx<T>();
    int config =
        AutoTuner<DeviceType>::autoTuningTable.gpk_reo_nd[prec][range_l];

    double min_time = std::numeric_limits<double>::max();
    int min_config = 0;

#define GPK(CONFIG)                                                            \
  if (config == CONFIG || AutoTuner<DeviceType>::ProfileKernels) {             \
    const int R = GPK_CONFIG[D - 1][CONFIG][0];                          \
    const int C = GPK_CONFIG[D - 1][CONFIG][1];                          \
    const int F = GPK_CONFIG[D - 1][CONFIG][2];                          \
    using FunctorType =                                                        \
        SingleDimensionMassTransFunctor<D, T, R, C, F, DeviceType>;          \
    using TaskType = Task<FunctorType>;                                        \
    TaskType task = GenTask<R, C, F>(current_dim, dist, ratio, coeff, v, queue_idx);  \
    DeviceAdapter<TaskType, DeviceType> adapter;                               \
    ExecutionReturn ret = adapter.Execute(task);                               \
    if (AutoTuner<DeviceType>::ProfileKernels) {                               \
      if (min_time > ret.execution_time) {                                     \
        min_time = ret.execution_time;                                         \
        min_config = CONFIG;                                                   \
      }                                                                        \
    }                                                                          \
  }

    GPK(0)
    GPK(1)
    GPK(2)
    GPK(3)
    GPK(4)
    GPK(5)
    GPK(6)
#undef GPK

    if (AutoTuner<DeviceType>::ProfileKernels) {
      FillAutoTunerTable<DeviceType>("SingleDimensionMassTrans", prec, range_l, min_config);
    }
  }
};

} // namespace mgard_x

#endif