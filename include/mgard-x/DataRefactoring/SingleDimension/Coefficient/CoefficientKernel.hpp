/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_SINGLE_DIMENSION_COEFFICIENT_KERNEL_TEMPLATE
#define MGARD_X_SINGLE_DIMENSION_COEFFICIENT_KERNEL_TEMPLATE

#include "../../../RuntimeX/RuntimeX.h"

#include "../../MultiDimension/Coefficient/GPKFunctor.h"

#define DECOMPOSE 0
#define RECOMPOSE 1

namespace mgard_x {

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, OPTION OP,
          typename DeviceType>
class SingleDimensionCoefficientFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT SingleDimensionCoefficientFunctor() {}
  MGARDX_CONT SingleDimensionCoefficientFunctor(
      DIM current_dim, SubArray<1, T, DeviceType> ratio,
      SubArray<D, T, DeviceType> v, SubArray<D, T, DeviceType> coarse,
      SubArray<D, T, DeviceType> coeff)
      : current_dim(current_dim), ratio(ratio), v(v), coarse(coarse),
        coeff(coeff) {
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
          v_left_idx[d] = coeff_idx[d];
          v_middle_idx[d] = coeff_idx[d];
          v_right_idx[d] = coeff_idx[d];
          corase_idx[d] = coeff_idx[d];
        } else {
          v_left_idx[d] = coeff_idx[d] * 2;
          v_middle_idx[d] = coeff_idx[d] * 2 + 1;
          v_right_idx[d] = coeff_idx[d] * 2 + 2;
          corase_idx[d] = coeff_idx[d];
        }
      }

      if (OP == DECOMPOSE) {
        *coeff(coeff_idx) =
            *v(v_middle_idx) - lerp(*v(v_left_idx), *v(v_right_idx),
                                    *ratio(v_left_idx[current_dim]));
        // if (coeff_idx[current_dim] == 1) {
        //   printf("left: %f, right: %f, middle: %f, ratio: %f, coeff: %f\n",
        //         *v(v_left_idx), *v(v_right_idx), *v(v_middle_idx),
        //         *ratio(v_left_idx[current_dim]), *coeff(coeff_idx));
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
      } else if (OP == RECOMPOSE) {
        T left = *coarse(corase_idx);
        corase_idx[current_dim]++;
        T right = *coarse(corase_idx);
        corase_idx[current_dim]--;

        *v(v_left_idx) = left;
        if (coeff_idx[current_dim] == coeff.getShape(current_dim) - 1) {
          corase_idx[current_dim]++;
          *v(v_right_idx) = right;
          if (v.getShape(current_dim) % 2 == 0) {
            v_right_idx[current_dim]++;
            corase_idx[current_dim]++;
            *v(v_right_idx) = *coarse(corase_idx);
            v_right_idx[current_dim]--;
            corase_idx[current_dim]--;
          }
          corase_idx[current_dim]--;
        }

        *v(v_middle_idx) = *coeff(coeff_idx) +
                           lerp(left, right, *ratio(v_left_idx[current_dim]));
        // if (coeff_idx[current_dim] == 1) {
        // printf("left: %f, right: %f, middle: %f (%f), ratio: %f, coeff:
        // %f\n",
        //       *v(v_left_idx), *v(v_right_idx), *v(v_middle_idx),
        //       *coeff(coeff_idx) + lerp(*v(v_left_idx), *v(v_right_idx),
        //       *ratio(v_left_idx[current_dim])),
        //       *ratio(v_left_idx[current_dim]), *coeff(coeff_idx));
        // }
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

template <DIM D, typename T, OPTION OP, typename DeviceType>
class SingleDimensionCoefficient : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  SingleDimensionCoefficient() : AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT
      Task<SingleDimensionCoefficientFunctor<D, T, R, C, F, OP, DeviceType>>
      GenTask(DIM current_dim, SubArray<1, T, DeviceType> ratio,
              SubArray<D, T, DeviceType> v, SubArray<D, T, DeviceType> coarse,
              SubArray<D, T, DeviceType> coeff, int queue_idx) {

    using FunctorType =
        SingleDimensionCoefficientFunctor<D, T, R, C, F, OP, DeviceType>;
    FunctorType functor(current_dim, ratio, v, coarse, coeff);

    SIZE nr = 1, nc = 1, nf = 1;
    if (D >= 3)
      nr = coeff.getShape(2);
    if (D >= 2)
      nc = coeff.getShape(1);
    nf = coeff.getShape(0);

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
  void Execute(DIM current_dim, SubArray<1, T, DeviceType> ratio,
               SubArray<D, T, DeviceType> v, SubArray<D, T, DeviceType> coarse,
               SubArray<D, T, DeviceType> coeff, int queue_idx) {
    int range_l = std::min(6, (int)std::log2(coeff.getShape(0)) - 1);
    int prec = TypeToIdx<T>();
    int config =
        AutoTuner<DeviceType>::autoTuningTable.gpk_reo_nd[prec][range_l];
    double min_time = std::numeric_limits<double>::max();
    int min_config = 0;
    ExecutionReturn ret;

#define GPK(CONFIG)                                                            \
  if (config == CONFIG || AutoTuner<DeviceType>::ProfileKernels) {             \
    const int R = GPK_CONFIG[D - 1][CONFIG][0];                                \
    const int C = GPK_CONFIG[D - 1][CONFIG][1];                                \
    const int F = GPK_CONFIG[D - 1][CONFIG][2];                                \
    using FunctorType =                                                        \
        SingleDimensionCoefficientFunctor<D, T, R, C, F, OP, DeviceType>;      \
    using TaskType = Task<FunctorType>;                                        \
    TaskType task =                                                            \
        GenTask<R, C, F>(current_dim, ratio, v, coarse, coeff, queue_idx);     \
    DeviceAdapter<TaskType, DeviceType> adapter;                               \
    ret = adapter.Execute(task);                                               \
    if (AutoTuner<DeviceType>::ProfileKernels) {                               \
      if (ret.success && min_time > ret.execution_time) {                      \
        min_time = ret.execution_time;                                         \
        min_config = CONFIG;                                                   \
      }                                                                        \
    }                                                                          \
  }

    GPK(6) if (!ret.success) config--;
    GPK(5) if (!ret.success) config--;
    GPK(4) if (!ret.success) config--;
    GPK(3) if (!ret.success) config--;
    GPK(2) if (!ret.success) config--;
    GPK(1) if (!ret.success) config--;
    GPK(0) if (!ret.success) config--;
    if (config < 0 && !ret.success) {
      std::cout << log::log_err
                << "no suitable config for SingleDimensionCoefficient.\n";
      exit(-1);
    }
#undef GPK

    if (AutoTuner<DeviceType>::ProfileKernels) {
      FillAutoTunerTable<DeviceType>("SingleDimensionCoefficient", prec,
                                     range_l, min_config);
    }
  }
};

} // namespace mgard_x

#endif