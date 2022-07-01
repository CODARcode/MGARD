/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_DENSE_TO_COMPRESSED_SPARSE_CELL_TEMPLATE
#define MGARD_X_DENSE_TO_COMPRESSED_SPARSE_CELL_TEMPLATE

#include "../../../RuntimeX/RuntimeX.h"

#include "../AdaptiveResolution/CompressedSparseCell.hpp"
namespace mgard_x {

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, typename DeviceType>
class DenseToCompressedSparseCellFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT DenseToCompressedSparseCellFunctor() {}
  MGARDX_CONT DenseToCompressedSparseCellFunctor(
    SubArray<D, T, DeviceType> v, SubArray<D, SIZE, DeviceType> cell_flag,
    SubArray<D, SIZE, DeviceType> write_index, 
    SubArray<1, T, DeviceType> value0,
    SubArray<1, T, DeviceType> value1,
    SubArray<1, T, DeviceType> value2,
    SubArray<1, T, DeviceType> value3,
    SubArray<1, T, DeviceType> value4,
    SubArray<1, T, DeviceType> value5,
    SubArray<1, T, DeviceType> value6,
    SubArray<1, T, DeviceType> value7,
    SubArray<1, SIZE, DeviceType> index0,
    SubArray<1, SIZE, DeviceType> index1,
    SubArray<1, SIZE, DeviceType> index2,
    SubArray<1, SIZE, DeviceType> size0,
    SubArray<1, SIZE, DeviceType> size1,
    SubArray<1, SIZE, DeviceType> size2)
  : v(v), cell_flag(cell_flag), write_index(write_index), 
   value0(value0), value1(value1), value2(value2), value3(value3),
   value4(value4), value5(value5), value6(value6), value7(value7),
   index0(index0), index1(index1), index2(index2),
   size0(size0), size1(size1), size2(size2) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE r_gl = FunctorBase<DeviceType>::GetBlockIdZ() *
              FunctorBase<DeviceType>::GetBlockDimZ() +
              FunctorBase<DeviceType>::GetThreadIdZ();
    SIZE c_gl = FunctorBase<DeviceType>::GetBlockIdY() *
              FunctorBase<DeviceType>::GetBlockDimY() +
              FunctorBase<DeviceType>::GetThreadIdY();
    SIZE f_gl = FunctorBase<DeviceType>::GetBlockIdX() *
              FunctorBase<DeviceType>::GetBlockDimX() +
              FunctorBase<DeviceType>::GetThreadIdX();

    if (r_gl >= cell_flag.getShape(2) || c_gl >= cell_flag.getShape(1) || f_gl >= cell_flag.getShape(0)) {
      return;
    }
    if (*cell_flag(r_gl, c_gl, f_gl) == 0) return;
    SIZE local_write_index = *write_index(r_gl, c_gl, f_gl);
    *value0(local_write_index) = *v(r_gl, c_gl, f_gl);
    *value1(local_write_index) = *v(r_gl, c_gl+1, f_gl);
    *value2(local_write_index) = *v(r_gl, c_gl, f_gl+1);
    *value3(local_write_index) = *v(r_gl, c_gl+1, f_gl+1);
    *value4(local_write_index) = *v(r_gl+1, c_gl, f_gl);
    *value5(local_write_index) = *v(r_gl+1, c_gl+1, f_gl);
    *value6(local_write_index) = *v(r_gl+1, c_gl, f_gl+1);
    *value7(local_write_index) = *v(r_gl+1, c_gl+1, f_gl+1);
    *index0(local_write_index) = f_gl;
    *index1(local_write_index) = c_gl;
    *index2(local_write_index) = r_gl;
    *size0(local_write_index) = 1;
    *size1(local_write_index) = 1;
    *size2(local_write_index) = 1;
    // if (r_gl < 5 && c_gl < 5 && f_gl < 5) {
    //   printf("COPY[%u %u %u] rcf: v: %f %f %f %f %f %f %f %f\n",
    //                     r_gl, c_gl, f_gl,
    //                     *v(r_gl, c_gl, f_gl),
    //                     *v(r_gl, c_gl, f_gl+1),
    //                     *v(r_gl, c_gl+1, f_gl),
    //                     *v(r_gl, c_gl+1, f_gl+1),
    //                     *v(r_gl+1, c_gl, f_gl),
    //                     *v(r_gl+1, c_gl, f_gl+1),
    //                     *v(r_gl+1, c_gl+1, f_gl),
    //                     *v(r_gl+1, c_gl+1, f_gl+1));
    // }
  }

private:
  SubArray<D, T, DeviceType> v;
  SubArray<D, SIZE, DeviceType> cell_flag;
  SubArray<D, SIZE, DeviceType> write_index;
  SubArray<1, T, DeviceType> value0;
  SubArray<1, T, DeviceType> value1;
  SubArray<1, T, DeviceType> value2;
  SubArray<1, T, DeviceType> value3;
  SubArray<1, T, DeviceType> value4;
  SubArray<1, T, DeviceType> value5;
  SubArray<1, T, DeviceType> value6;
  SubArray<1, T, DeviceType> value7;
  SubArray<1, SIZE, DeviceType> index0;
  SubArray<1, SIZE, DeviceType> index1;
  SubArray<1, SIZE, DeviceType> index2;
  SubArray<1, SIZE, DeviceType> size0;
  SubArray<1, SIZE, DeviceType> size1;
  SubArray<1, SIZE, DeviceType> size2;

};

template <DIM D, typename T, typename DeviceType>
class DenseToCompressedSparseCellKernel : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  DenseToCompressedSparseCellKernel() : AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<DenseToCompressedSparseCellFunctor<D, T, R, C, F, DeviceType>> GenTask(
      SubArray<D, T, DeviceType> v, SubArray<D, SIZE, DeviceType> cell_flag,
      SubArray<D, SIZE, DeviceType> write_index, 
      SubArray<1, T, DeviceType> value0,
      SubArray<1, T, DeviceType> value1,
      SubArray<1, T, DeviceType> value2,
      SubArray<1, T, DeviceType> value3,
      SubArray<1, T, DeviceType> value4,
      SubArray<1, T, DeviceType> value5,
      SubArray<1, T, DeviceType> value6,
      SubArray<1, T, DeviceType> value7,
      SubArray<1, SIZE, DeviceType> index0,
      SubArray<1, SIZE, DeviceType> index1,
      SubArray<1, SIZE, DeviceType> index2,
      SubArray<1, SIZE, DeviceType> size0,
      SubArray<1, SIZE, DeviceType> size1,
      SubArray<1, SIZE, DeviceType> size2, 
      int queue_idx) {
    using FunctorType = DenseToCompressedSparseCellFunctor<D, T, R, C, F, DeviceType>;
    FunctorType functor(v, cell_flag, write_index,
                        value0, value1, value2, value3,
                        value4, value5, value6, value7,
                        index0, index1, index2,
                        size0, size1, size2);

    SIZE total_thread_z = std::max(v.getShape(2) - 1, (SIZE)1);
    SIZE total_thread_y = std::max(v.getShape(1) - 1, (SIZE)1);
    SIZE total_thread_x = std::max(v.getShape(0) - 1, (SIZE)1);
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size;
    tbz = R;
    tby = C;
    tbx = F;
    sm_size = 0;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "DenseToCompressedSparseCellKernel");
  }

  MGARDX_CONT
  void Execute(SubArray<D, T, DeviceType> v, SubArray<D, SIZE, DeviceType> cell_flag,
      SubArray<D, SIZE, DeviceType> write_index, CompressedSparseCell<T, DeviceType>& csc, int queue_idx) {
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
      using FunctorType = DenseToCompressedSparseCellFunctor<D, T, R, C, F, DeviceType>;\
      using TaskType = Task<FunctorType>;                                        \
      TaskType task = GenTask<R, C, F>(                                          \
          v, cell_flag, write_index,                                             \
          csc.value[0], csc.value[1], csc.value[2], csc.value[3],                \
          csc.value[4], csc.value[5], csc.value[6], csc.value[7],                \
          csc.index[0], csc.index[1], csc.index[2],                              \
          csc.size[0], csc.size[1], csc.size[2], queue_idx);                     \
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
      std::cout << log::log_err << "no suitable config for DenseToCompressedSparseCell.\n";
      exit(-1);
    }
#undef GPK

    if (AutoTuner<DeviceType>::ProfileKernels) {
      FillAutoTunerTable<DeviceType>("gpk_rev_3d", prec, range_l, min_config);
    }
  }
};

template <DIM D, typename T, typename DeviceType>
void DenseToCompressedSparseCell(SubArray<D, T, DeviceType> v, SubArray<D, SIZE, DeviceType> cell_flag,
      SubArray<D, SIZE, DeviceType> write_index, CompressedSparseCell<T, DeviceType>& csc, int queue_idx) {
  DenseToCompressedSparseCellKernel<D, T, DeviceType>().Execute(v, cell_flag, write_index, csc, queue_idx);
}


}

#endif