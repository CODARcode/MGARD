/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_SINGLE_DIMENSION_COEFFICIENT_KERNEL_TEMPLATE
#define MGARD_X_SINGLE_DIMENSION_COEFFICIENT_KERNEL_TEMPLATE

#include "../../../RuntimeX/RuntimeX.h"

#include "../../MultiDimension/Coefficient/GPKFunctor.h"

namespace mgard_x {

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, typename DeviceType>
class SingleDimensionCoefficientFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT SingleDimensionCoefficientFunctor() {}
  MGARDX_CONT SingleDimensionCoefficientFunctor(DIM current_dim,
      SubArray<1, T, DeviceType> ratio, SubArray<D, T, DeviceType> v,
      SubArray<D, T, DeviceType> coarse, SubArray<D, T, DeviceType> coeff)
      : current_dim(current_dim),
        ratio(ratio), v(v), coarse(coarse), coeff(coeff) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE v_left_idx[D];
    SIZE v_middle_idx[D];
    SIZE v_right_idx[D];
    SIZE coeff_idx[D];
    SIZE corase_idx[D];

    SIZE firstD = div_roundup(coeff.getShape(0), F);

    SIZE bidx = FunctorBase<DeviceType>::GetBlockIdX();
    coeff_idx[0] =
        (bidx % firstD) * F + FunctorBase<DeviceType>::GetThreadIdX();

    bidx /= firstD;
    if (D >= 2)
      coeff_idx[1] = FunctorBase<DeviceType>::GetBlockIdY() *
                         FunctorBase<DeviceType>::GetBlockDimY() +
                     FunctorBase<DeviceType>::GetThreadIdY();
    if (D >= 3)
      coeff_idx[2] = FunctorBase<DeviceType>::GetBlockIdZ() *
                         FunctorBase<DeviceType>::GetBlockDimZ() +
                     FunctorBase<DeviceType>::GetThreadIdZ();

    for (DIM d = 3; d < D; d++) {
      coeff_idx[d] = bidx % coeff.getShape(d);
      bidx /= coeff.getShape(d);
    }

    bool in_range = true;
    for (DIM d = 0; d < D; d++) {
      if (coeff_idx[d] >= coeff.getShape(d))
        in_range = false;
    }

    if (in_range) {
      for (DIM d = 0; d < D; d++) {
        if (d != current_dim) {
          v_left_idx[d]   = coeff_idx[d];
          v_middle_idx[d] = coeff_idx[d];
          v_right_idx[d]  = coeff_idx[d];
          corase_idx[d]   = coeff_idx[d];
        } else {
          v_left_idx[d]   = coeff_idx[d] * 2;
          v_middle_idx[d] = coeff_idx[d] * 2 + 1;
          v_right_idx[d]  = coeff_idx[d] * 2 + 2;
          corase_idx[d]   = coeff_idx[d];
        }
      }
      *coeff(coeff_idx) = *v(v_middle_idx) - lerp(*v(v_left_idx), *v(v_right_idx), *ratio(v_left_idx[current_dim]));
      // if (coeff_idx[current_dim] == 1) {
      //   printf("left: %f, right: %f, middle: %f, ratio: %f, coeff: %f\n",
      //         *v(v_left_idx), *v(v_right_idx), *v(v_middle_idx), *ratio(v_left_idx[current_dim]), *coeff(coeff_idx));
      // }
      *coarse(corase_idx) = *v(v_left_idx);
      if (coeff_idx[current_dim] == coeff.getShape(current_dim) - 1) {
        corase_idx[current_dim]++;
        *coarse(corase_idx) = *v(v_right_idx);
        if (v.getShape(current_dim) % 2 == 0) {
          v_right_idx[current_dim]++;
          corase_idx[current_dim]++;
          *coarse(corase_idx) = *v(v_right_idx);
        }
      }
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  // functor parameters
  DIM current_dim;
  SubArray<1, T, DeviceType> ratio;
  SubArray<D, T, DeviceType> v;
  SubArray<D, T, DeviceType> coarse;
  SubArray<D, T, DeviceType> coeff;
};


template <DIM D, typename T, typename DeviceType>
class SingleDimensionCoefficient : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  SingleDimensionCoefficient() : AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<SingleDimensionCoefficientFunctor<D, T, R, C, F, DeviceType>>
  GenTask(DIM current_dim,
          SubArray<1, T, DeviceType> ratio, SubArray<D, T, DeviceType> v,
          SubArray<D, T, DeviceType> coarse, SubArray<D, T, DeviceType> coeff, 
          int queue_idx) {

    using FunctorType =
        SingleDimensionCoefficientFunctor<D, T, R, C, F, DeviceType>;
    FunctorType functor(current_dim, ratio, v, coarse, coeff);

    SIZE nr = coeff.getShape(2);
    SIZE nc = coeff.getShape(1);
    SIZE nf = coeff.getShape(0);
    if (D == 2) {
      nr = 1;
    }
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
                "SingleDimensionCoefficient");
  }

  MGARDX_CONT
  void Execute(DIM current_dim,
               SubArray<1, T, DeviceType> ratio, SubArray<D, T, DeviceType> v,
               SubArray<D, T, DeviceType> coarse, SubArray<D, T, DeviceType> coeff, 
               int queue_idx) {
    int range_l = std::min(6, (int)std::log2(coeff.getShape(0)) - 1);
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
        SingleDimensionCoefficientFunctor<D, T, R, C, F, DeviceType>;          \
    using TaskType = Task<FunctorType>;                                        \
    TaskType task = GenTask<R, C, F>(current_dim, ratio, v, coarse, coeff, queue_idx);  \
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
      FillAutoTunerTable<DeviceType>("SingleDimensionCoefficient", prec, range_l, min_config);
    }
  }
};

} // namespace mgard_x

#endif