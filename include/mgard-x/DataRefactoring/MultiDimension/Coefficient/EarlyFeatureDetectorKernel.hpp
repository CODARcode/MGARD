/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_EARLY_FEATURE_DETECTOR_TEMPLATE
#define MGARD_X_EARLY_FEATURE_DETECTOR_TEMPLATE

#include "../../../RuntimeX/RuntimeX.h"

#define NO_FEATURE 0
#define HAS_FEATURE 1
#define DISCARD_CHILDREN 0
#define KEEP_CHILDREN 1

namespace mgard_x {

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, typename DeviceType>
class EarlyFeatureDetectorFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT EarlyFeatureDetectorFunctor() {}
  MGARDX_CONT EarlyFeatureDetectorFunctor(
    SubArray<D, T, DeviceType> v, T current_error, T iso_value,
    SubArray<D, SIZE, DeviceType> feature_flag_coarser,
    SubArray<D, SIZE, DeviceType> feature_flag)
  : v(v), current_error(current_error), iso_value(iso_value), 
    feature_flag_coarser(feature_flag_coarser), feature_flag(feature_flag) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    r_gl = FunctorBase<DeviceType>::GetBlockIdZ() *
              FunctorBase<DeviceType>::GetBlockDimZ() +
              FunctorBase<DeviceType>::GetThreadIdZ();
    c_gl = FunctorBase<DeviceType>::GetBlockIdY() *
              FunctorBase<DeviceType>::GetBlockDimY() +
              FunctorBase<DeviceType>::GetThreadIdY();
    f_gl = FunctorBase<DeviceType>::GetBlockIdX() *
              FunctorBase<DeviceType>::GetBlockDimX() +
              FunctorBase<DeviceType>::GetThreadIdX();

    r_sm = FunctorBase<DeviceType>::GetThreadIdZ();
    c_sm = FunctorBase<DeviceType>::GetThreadIdY();
    f_sm = FunctorBase<DeviceType>::GetThreadIdX();

    early_return = false;
    if (r_gl >= v.getShape(2) || c_gl >= v.getShape(1) || f_gl >= v.getShape(0)) {
      early_return = true;
      return;
    }
    // if (!feature_flag_coarser.isNull() && *feature_flag_coarser(r_gl/2, c_gl/2, f_gl/2) == NO_FEATURE) {

    //   early_return = true;
    //   return;
    // }

    sm = (T *)FunctorBase<DeviceType>::GetSharedMemory();
    ldsm1 = F + 1;
    ldsm2 = C + 1;
    sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = *v(r_gl, c_gl, f_gl);
    if (r_sm == 0) {
      // if (FunctorBase<DeviceType>::GetBlockIdZ() == 0 && FunctorBase<DeviceType>::GetBlockIdY() == 0 && FunctorBase<DeviceType>::GetBlockIdX() == 0 ) {
      //   printf("load r+: %f\n", *v(r_gl+1, c_gl, f_gl));
      // }
      sm[get_idx(ldsm1, ldsm2, R, c_sm, f_sm)] = *v(r_gl+R, c_gl, f_gl);
    }
    if (c_sm == 0) {
      sm[get_idx(ldsm1, ldsm2, r_sm, C, f_sm)] = *v(r_gl, c_gl+C, f_gl);
    }
    if (f_sm == 0) {
      sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, F)] = *v(r_gl, c_gl, f_gl+F);
    }
    if (r_sm == 0 && c_sm == 0) {
      sm[get_idx(ldsm1, ldsm2, R, C, f_sm)] = *v(r_gl+R, c_gl+C, f_gl);
    }
    if (r_sm == 0 && f_sm == 0) {
      sm[get_idx(ldsm1, ldsm2, R, c_sm, F)] = *v(r_gl+R, c_gl, f_gl+F);
    }
    if (c_sm == 0 && f_sm == 0) {
      sm[get_idx(ldsm1, ldsm2, r_sm, C, F)] = *v(r_gl, c_gl+C, f_gl+F);
    }
    if (r_sm == 0 && c_sm == 0 && f_sm == 0) {
      sm[get_idx(ldsm1, ldsm2, R, C, F)] = *v(r_gl+R, c_gl+C, f_gl+F);
    }
  }

  MGARDX_EXEC void Operation2() {
    // if (r_gl == 0 && c_gl == 0 && f_gl == 0 && r_sm == 0 && c_sm == 0 && f_sm == 0) {
    //   for (int i = 0; i < R+1; i++) {
    //     printf("i = %d\n", i);
    //     for (int j = 0; j < C+1; j++) {
    //       for (int k = 0; k < F+1; k++) {
    //         printf("%f ", sm[get_idx(ldsm1, ldsm2, i, j, k)]);
    //       }
    //       printf("\n");
    //     }
    //     printf("\n");
    //   }

    //   for (int i = 0; i < R+1; i++) {
    //     printf("i = %d\n", i);
    //     for (int j = 0; j < C+1; j++) {
    //       for (int k = 0; k < F+1; k++) {
    //         printf("%f ", *v(i, j, k));
    //       }
    //       printf("\n");
    //     }
    //     printf("\n");
    //   }
    // }

  }
  MGARDX_EXEC void Operation3() {
    if (early_return) {
      return;
    }
    if (r_gl >= v.getShape(2)-1 || c_gl >= v.getShape(1)-1 || f_gl >= v.getShape(0)-1) {
      early_return = true;
      return;
    }
    T max_data = std::numeric_limits<T>::min();
    T min_data = std::numeric_limits<T>::max();
    max_data = std::max(max_data, sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)]);
    min_data = std::min(min_data, sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)]);

    max_data = std::max(max_data, sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm+1)]);
    min_data = std::min(min_data, sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm+1)]);

    max_data = std::max(max_data, sm[get_idx(ldsm1, ldsm2, r_sm, c_sm+1, f_sm)]);
    min_data = std::min(min_data, sm[get_idx(ldsm1, ldsm2, r_sm, c_sm+1, f_sm)]);

    max_data = std::max(max_data, sm[get_idx(ldsm1, ldsm2, r_sm, c_sm+1, f_sm+1)]);
    min_data = std::min(min_data, sm[get_idx(ldsm1, ldsm2, r_sm, c_sm+1, f_sm+1)]);

    max_data = std::max(max_data, sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm, f_sm)]);
    min_data = std::min(min_data, sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm, f_sm)]);

    max_data = std::max(max_data, sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm, f_sm+1)]);
    min_data = std::min(min_data, sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm, f_sm+1)]);

    max_data = std::max(max_data, sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm+1, f_sm)]);
    min_data = std::min(min_data, sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm+1, f_sm)]);

    max_data = std::max(max_data, sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm+1, f_sm+1)]);
    min_data = std::min(min_data, sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm+1, f_sm+1)]);

    // if (FunctorBase<DeviceType>::GetBlockIdZ() == 0 && FunctorBase<DeviceType>::GetBlockIdY() == 0 && FunctorBase<DeviceType>::GetBlockIdX() == 0 ) {
    //   printf("rcf: %u %u %u, max: %.10f min: %.10f\n", r_gl, c_gl, f_gl, max_data, min_data);
    // }

    if (max_data + current_error >= iso_value &&
        min_data - current_error <= iso_value) {
      *feature_flag(r_gl, c_gl, f_gl) = HAS_FEATURE;
      if (r_gl == 0 && c_gl == 0 && f_gl == 0 ) {
        // printf("HAS_FEATURE[%u %u %u]: v: %f %f %f %f %f %f %f %f\n",
        //                         r_gl, c_gl, f_gl,
        //                         *v(r_gl, c_gl, f_gl),
        //                         *v(r_gl, c_gl, f_gl+1),
        //                         *v(r_gl, c_gl+1, f_gl),
        //                         *v(r_gl, c_gl+1, f_gl+1),
        //                         *v(r_gl+1, c_gl, f_gl),
        //                         *v(r_gl+1, c_gl, f_gl+1),
        //                         *v(r_gl+1, c_gl+1, f_gl),
        //                         *v(r_gl+1, c_gl+1, f_gl+1));
                                // sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)],
                                // sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm+1)],
                                // sm[get_idx(ldsm1, ldsm2, r_sm, c_sm+1, f_sm)],
                                // sm[get_idx(ldsm1, ldsm2, r_sm, c_sm+1, f_sm+1)],
                                // sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm, f_sm)],
                                // sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm, f_sm+1)],
                                // sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm+1, f_sm)],
                                // sm[get_idx(ldsm1, ldsm2, r_sm+1, c_sm+1, f_sm+1)]);
      }
    } else {
      *feature_flag(r_gl, c_gl, f_gl) = NO_FEATURE;
    }

    
  }

private:
  SubArray<D, T, DeviceType> v;
  T current_error, iso_value;
  SubArray<D, SIZE, DeviceType> feature_flag_coarser;
  SubArray<D, SIZE, DeviceType> feature_flag;
  bool early_return;

  SIZE r_gl, c_gl, f_gl;
  SIZE r_sm, c_sm, f_sm;
  T * sm;
  SIZE ldsm1, ldsm2;
};

template <DIM D, typename T, typename DeviceType>
class EarlyFeatureDetectorKernel : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  EarlyFeatureDetectorKernel() : AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<EarlyFeatureDetectorFunctor<D, T, R, C, F, DeviceType>> GenTask(
      SubArray<D, T, DeviceType> v, T current_error, T iso_value,
    SubArray<D, SIZE, DeviceType> feature_flag_coarser,
    SubArray<D, SIZE, DeviceType> feature_flag, int queue_idx) {
    using FunctorType = EarlyFeatureDetectorFunctor<D, T, R, C, F, DeviceType>;
    FunctorType functor(v, current_error, iso_value, feature_flag_coarser, feature_flag);

    SIZE total_thread_z = std::max(v.getShape(2) - 1, (SIZE)1);
    SIZE total_thread_y = std::max(v.getShape(1) - 1, (SIZE)1);
    SIZE total_thread_x = std::max(v.getShape(0) - 1, (SIZE)1);
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size;
    tbz = R;
    tby = C;
    tbx = F;
    sm_size = ((R + 1) * (C + 1) * (F + 1)) * sizeof(T);
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "EarlyFeatureDetectorKernel");
  }

  MGARDX_CONT
  void Execute(SubArray<D, T, DeviceType> v, T current_error, T iso_value,
              SubArray<D, SIZE, DeviceType> feature_flag_coarser,
              SubArray<D, SIZE, DeviceType> feature_flag, int queue_idx) {
    int range_l = std::min(6, (int)std::log2(v.getShape(0)) - 1);
    int prec = TypeToIdx<T>();
    int config =
        AutoTuner<DeviceType>::autoTuningTable.gpk_rev_3d[prec][range_l];
    double min_time = std::numeric_limits<double>::max();
    int min_config = 0;
    ExecutionReturn ret;

#define GPK(CONFIG)                                                            \
    if (config == CONFIG || AutoTuner<DeviceType>::ProfileKernels) {             \
      const int R = GPK_CONFIG[D - 1][CONFIG][0];                                \
      const int C = GPK_CONFIG[D - 1][CONFIG][1];                                \
      const int F = GPK_CONFIG[D - 1][CONFIG][2];                                \
      using FunctorType = EarlyFeatureDetectorFunctor<D, T, R, C, F, DeviceType>;            \
      using TaskType = Task<FunctorType>;                                        \
      TaskType task = GenTask<R, C, F>(                                          \
          v, current_error, iso_value, feature_flag_coarser, feature_flag, queue_idx);     \
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
      std::cout << log::log_err << "no suitable config for EarlyFeatureDetectorKernel.\n";
      exit(-1);
    }
#undef GPK

    if (AutoTuner<DeviceType>::ProfileKernels) {
      FillAutoTunerTable<DeviceType>("gpk_rev_3d", prec, range_l, min_config);
    }
  }
};

}

#endif