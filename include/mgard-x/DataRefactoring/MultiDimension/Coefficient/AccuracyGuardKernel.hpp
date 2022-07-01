/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_ACCURACY_GUARD_TEMPLATE
#define MGARD_X_ACCURACY_GUARD_TEMPLATE

#include "../../../RuntimeX/RuntimeX.h"

namespace mgard_x {

#define NO_FEATURE 0
#define HAS_FEATURE 1
#define DISCARD_CHILDREN 0
#define KEEP_CHILDREN 1

template <typename T, typename DeviceType>
MGARDX_EXEC void combine_error_impact(SIZE total_level, SIZE current_level, const T * prev_error_impact, const T * local_error_impact, T * estimated_error_impact) {
  for (int l = 0; l < current_level+1; l++) {
    estimated_error_impact[l] = Math<DeviceType>::Max(local_error_impact[l], prev_error_impact[l]);
  }
  for (int l = current_level+1; l < total_level; l++) {
    estimated_error_impact[l] = prev_error_impact[l];
  }
}

template <typename T>
MGARDX_EXEC bool check_error_impact_budget(SIZE total_level, T * estimated_error_impact, T * error_impact_budget) {
  for (int l = 0; l < total_level; l++) {
    if (estimated_error_impact[l] > error_impact_budget[l]) {
      return false;
    }
  }
  return true;
}

template <typename T, typename DeviceType>
MGARDX_EXEC void update_error_impact(SIZE total_level, T * current_error_impact, T * estimated_error_impact) {
  using AtomicOp = Atomic<T, AtomicGlobalMemory, AtomicDeviceScope, DeviceType>;
  for (int l = 0; l < total_level; l++) {
    AtomicOp::Max(&current_error_impact[l], estimated_error_impact[l]);
  }
}

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, SIZE MAX_LEVEL, typename DeviceType>
class AccuracyGuardFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT AccuracyGuardFunctor() {}
  MGARDX_CONT AccuracyGuardFunctor(
    SIZE total_level, SIZE current_level, 
    SubArray<1, T, DeviceType> error_impact_budget,
    SubArray<1, T, DeviceType> previous_error_impact,
    SubArray<1, T, DeviceType> current_error_impact,
    SubArray<D+1, T, DeviceType> max_abs_coefficient,
    SubArray<D, SIZE, DeviceType> refinement_flag)
  : error_impact_budget(error_impact_budget), 
    previous_error_impact(previous_error_impact), 
    current_error_impact(current_error_impact),
    max_abs_coefficient(max_abs_coefficient), refinement_flag(refinement_flag) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE r = FunctorBase<DeviceType>::GetBlockIdZ() *
              FunctorBase<DeviceType>::GetBlockDimZ() +
              FunctorBase<DeviceType>::GetThreadIdZ();
    SIZE c = FunctorBase<DeviceType>::GetBlockIdY() *
              FunctorBase<DeviceType>::GetBlockDimY() +
              FunctorBase<DeviceType>::GetThreadIdY();
    SIZE f = FunctorBase<DeviceType>::GetBlockIdX() *
              FunctorBase<DeviceType>::GetBlockDimX() +
              FunctorBase<DeviceType>::GetThreadIdX();

    if (r >= max_abs_coefficient.getShape(2) ||
        c >= max_abs_coefficient.getShape(1) ||
        f >= max_abs_coefficient.getShape(0)) {
      return;
    }
    T local_error_impact[MAX_LEVEL];
    T estimated_error_impact[MAX_LEVEL];
    for (int l = 0; l < current_level+1; l++) local_error_impact[l] = *max_abs_coefficient(l, r, c, f);
    for (int l = current_level+1; l < total_level; l++) local_error_impact[l] = 0;

    combine_error_impact<T, DeviceType>(total_level, current_level, previous_error_impact.data(), local_error_impact, estimated_error_impact);
    if (check_error_impact_budget<T>(total_level, estimated_error_impact, error_impact_budget.data()) && *refinement_flag(r, c, f) == NO_FEATURE) { // discarding all children
      *refinement_flag(r, c, f) = DISCARD_CHILDREN;
      update_error_impact<T, DeviceType>(total_level, current_error_impact.data(), estimated_error_impact);
    } else {
      *refinement_flag(r, c, f) = KEEP_CHILDREN;
    }
  }

private:
  SIZE total_level, current_level;
  SubArray<1, T, DeviceType> error_impact_budget;
  SubArray<1, T, DeviceType> previous_error_impact;
  SubArray<1, T, DeviceType> current_error_impact;
  SubArray<D+1, T, DeviceType> max_abs_coefficient;
  SubArray<D, SIZE, DeviceType> refinement_flag;
};

template <DIM D, typename T, typename DeviceType>
class AccuracyGuardKernel : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  AccuracyGuardKernel() : AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F, SIZE MAX_LEVEL>
  MGARDX_CONT
      Task<AccuracyGuardFunctor<D, T, R, C, F, MAX_LEVEL, DeviceType>>
      GenTask(SIZE total_level, SIZE current_level, 
              SubArray<1, T, DeviceType> error_impact_budget,
              SubArray<1, T, DeviceType> previous_error_impact,
              SubArray<1, T, DeviceType> current_error_impact,
              SubArray<D+1, T, DeviceType> max_abs_coefficient,
              SubArray<D, SIZE, DeviceType> refinement_flag,
              int queue_idx) {
    using FunctorType =
        AccuracyGuardFunctor<D, T, R, C, F, MAX_LEVEL, DeviceType>;
    FunctorType functor(total_level, current_level, error_impact_budget, previous_error_impact,
                        current_error_impact, max_abs_coefficient, refinement_flag);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = 0;
    int total_thread_z = refinement_flag.getShape(2);
    int total_thread_y = refinement_flag.getShape(1);
    int total_thread_x = refinement_flag.getShape(0);
    // linearize other dimensions
    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);
    // printf("LevelMax config: %u %u %u %u %u %u\n", tbx, tby, tbz,
    // gridx, gridy, gridz);
    for (int d = 3; d < D; d++) {
      gridx *= refinement_flag.getShape(d);
    }
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size,
                queue_idx, "AccuracyGuardKernel");
  }

  MGARDX_CONT
  void Execute(SIZE total_level, SIZE current_level, 
              SubArray<1, T, DeviceType> error_impact_budget,
              SubArray<1, T, DeviceType> previous_error_impact,
              SubArray<1, T, DeviceType> current_error_impact,
              SubArray<D+1, T, DeviceType> max_abs_coefficient,
              SubArray<D, SIZE, DeviceType> refinement_flag,
              int queue_idx) {

    int range_l = std::min(6, (int)std::log2(refinement_flag.getShape(0)) - 1);
    int prec = TypeToIdx<T>();
    int config = AutoTuner<DeviceType>::autoTuningTable.llk[prec][range_l];
    double min_time = std::numeric_limits<double>::max();
    int min_config = 0;
    ExecutionReturn ret;

#define LLK(CONFIG)                                                            \
    if (config == CONFIG || AutoTuner<DeviceType>::ProfileKernels) {             \
      const int R = LWPK_CONFIG[D - 1][CONFIG][0];                               \
      const int C = LWPK_CONFIG[D - 1][CONFIG][1];                               \
      const int F = LWPK_CONFIG[D - 1][CONFIG][2];                               \
      const int MAX_LEVEL = 10;                                                  \
      using FunctorType =                                                        \
          AccuracyGuardFunctor<D, T, R, C, F, MAX_LEVEL, DeviceType>;                 \
      using TaskType = Task<FunctorType>;                                        \
      TaskType task =                                                            \
          GenTask<R, C, F, MAX_LEVEL>(total_level, current_level, error_impact_budget, previous_error_impact,       \
                        current_error_impact, max_abs_coefficient, refinement_flag, queue_idx);           \
      DeviceAdapter<TaskType, DeviceType> adapter;                               \
      ret = adapter.Execute(task);                                               \
      if (AutoTuner<DeviceType>::ProfileKernels) {                               \
        if (ret.success && min_time > ret.execution_time) {                      \
          min_time = ret.execution_time;                                         \
          min_config = CONFIG;                                                   \
        }                                                                        \
      }                                                                          \
    }
    LLK(6) if (!ret.success) config--;
    LLK(5) if (!ret.success) config--;
    LLK(4) if (!ret.success) config--;
    LLK(3) if (!ret.success) config--;
    LLK(2) if (!ret.success) config--;
    LLK(1) if (!ret.success) config--;
    LLK(0) if (!ret.success) config--;
    if (config < 0 && !ret.success) {
      std::cout << log::log_err << "no suitable config for LevelLinearizer.\n";
      exit(-1);
    }
#undef LLK
    if (AutoTuner<DeviceType>::ProfileKernels) {
      FillAutoTunerTable<DeviceType>("llk", prec, range_l, min_config);
    }
  }
};

}

#endif