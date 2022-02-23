/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_SINGLE_DIMENSION_SOLVE_TRIDIAG_KERNEL_TEMPLATE
#define MGARD_X_SINGLE_DIMENSION_SOLVE_TRIDIAG_KERNEL_TEMPLATE

#include "../../../RuntimeX/RuntimeX.h"

// #include "../../MultiDimension/Correction/LPKFunctor.h"

namespace mgard_x {

template <typename T, SIZE F, typename DeviceType>
class ForwardPassMultCoefficientFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT ForwardPassMultCoefficientFunctor() {}
  MGARDX_CONT ForwardPassMultCoefficientFunctor(
      SubArray<1, T, DeviceType> am, SubArray<1, T, DeviceType> bm,
      SubArray<1, T, DeviceType> amXbm)
      : am(am), bm(bm), (amXbm) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE id = FunctorBase<DeviceType>::GetBlockIdX() * 
                FunctorBase<DeviceType>::GetBlockDimX() + 
                FunctorBase<DeviceType>::GetThreadIdX();

    if (id < am.getShape(0)) {
      *amXbm(id) = (*am(id)) * (*bm(id));
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  // functor parameters
  SubArray<1, T, DeviceType> am;
  SubArray<1, T, DeviceType> bm;
  SubArray<1, T, DeviceType> amXbm;
};


template <DIM D, typename T, typename DeviceType>
class SingleDimensionSolveTridiag : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  SingleDimensionSolveTridiag() : AutoTuner<DeviceType>() {}

  template <SIZE F>
  MGARDX_CONT Task<ForwardPassMultCoefficient<T, F, DeviceType>>
  GenTask(SubArray<1, T, DeviceType> am, SubArray<1, T, DeviceType> bm,
          SubArray<1, T, DeviceType> amXbm, int queue_idx) {

    using FunctorType =
        ForwardPassMultCoefficientFunctor<T, F, DeviceType>;
    FunctorType functor(am, bm, amXbm);

    SIZE nf = v.getShape(0);
    SIZE total_thread_x = nf;

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();

    tbz = 1;
    tby = 1;
    tbx = F;
    gridz = 1;
    gridy = 1;
    gridx = ceil((float)total_thread_x / tbx);

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "ForwardPassMultCoefficient");
  }

  MGARDX_CONT
  void Execute(DIM current_dim,
               SubArray<1, T, DeviceType> dist, SubArray<1, T, DeviceType> ratio, 
               SubArray<D, T, DeviceType> coeff, SubArray<D, T, DeviceType> v,
               SubArray<1, T, DeviceType> am, SubArray<1, T, DeviceType> bm, 
               int queue_idx) {

    Array<1, T, DeviceType> amXbm(am.getShape(0));

    int range_l = std::min(6, (int)std::log2(coeff.getShape(0)) - 1);
    int arch = DeviceRuntime<DeviceType>::GetArchitectureGeneration();
    int prec = TypeToIdx<T>();
    int config =
        AutoTuner<DeviceType>::autoTuningTable.gpk_reo_nd[prec][range_l];

    double min_time = std::numeric_limits<double>::max();
    int min_config = 0;

#define GPK(CONFIG)                                                              \
    if (config == CONFIG || AutoTuner<DeviceType>::ProfileKernels) {             \
      const int F = GPK_CONFIG[D - 1][CONFIG][2];                                \
      using FunctorType =                                                        \
          ForwardPassMultCoefficient<T, F, DeviceType>;                          \
      using TaskType = Task<FunctorType>;                                        \
      TaskType task = GenTask<F>(am, bm, SubArray(amXbm), queue_idx);            \
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
      FillAutoTunerTable<DeviceType>("SingleDimensionSolveTridiag", prec, range_l, min_config);
    }
  }
};

} // namespace mgard_x

#endif