/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_LINEAR_QUANTIZATION_TEMPLATE
#define MGARD_X_LINEAR_QUANTIZATION_TEMPLATE

#include "../RuntimeX/RuntimeX.h"
// #include "LinearQuantization.h"

namespace mgard_x {

template <DIM D, typename T>
void calc_quantizers(size_t dof, T *quantizers, enum error_bound_type type,
                     T tol, T s, T norm, SIZE l_target,
                     enum decomposition_type decomposition, bool reciprocal) {

  double abs_tol = tol;
  if (type == error_bound_type::REL) {
    abs_tol *= norm;
  }

  // printf("tol %f, l_target %d, D %d\n", tol, l_target, D);

  abs_tol *= 2;

  // original
  // tol /= l_target + 2;
  // for (int l = 0; l < l_target+1; l++) {
  //   quantizers[l] = tol;
  // }
  // printf("l_target %d\n", l_target);

  // levelwise
  // tol *= 2;
  // T C2 = 1 + 3*std::sqrt(3)/4;
  // T c = std::sqrt(std::pow(2, D));
  // T cc = (1 - c) / (1 - std::pow(c, l_target+1));
  // T level_eb = cc * tol / C2;

  // for (int l = 0; l < l_target+1; l++) {
  //   quantizers[l] = level_eb;
  //   level_eb *= c;
  // }

  // s = 0;

  // levelwise with s
  // tol *= 2;
  // T C2 = 1 + 3 * std::sqrt(3) / 4;
  // T c = std::sqrt(std::pow(2, D - 2 * s));
  // T cc = (1 - c) / (1 - std::pow(c, l_target + 1));
  // T level_eb = cc * tol / C2;

  // for (int l = 0; l < l_target + 1; l++) {
  //   quantizers[l] = level_eb;
  //   // T c = std::sqrt(std::pow(2, 2*s*l + D * (l_target - l)));
  //   level_eb *= c;
  //   if (reciprocal)
  //     quantizers[l] = 1.0f / quantizers[l];
  // }

  if (s == std::numeric_limits<T>::infinity()) {

    // printf("quantizers: ");
    for (int l = 0; l < l_target + 1; l++) {
      if (decomposition == decomposition_type::MultiDim) {
        // ben
        quantizers[l] = (abs_tol) / ((l_target + 1) * (1 + std::pow(3, D)));
        // xin
        // quantizers[l] = (tol) / ((l_target + 1) * (1 + 3 * std::sqrt(3) /
        // 4));
      } else if (decomposition == decomposition_type::SingleDim) {
        // ken
        quantizers[l] = (abs_tol) / ((l_target + 1) * D * (1 + std::pow(3, 1)));
      }

      // printf("%f ", quantizers[l]);
      if (reciprocal)
        quantizers[l] = 1.0f / quantizers[l];
    }
    // printf("\n");

  } else { // s != inf
    // xin - uniform
    // T C2 = 1 + 3 * std::sqrt(3) / 4;
    // T c = std::sqrt(std::pow(2, D - 2 * s));
    // T cc = (1 - c) / (1 - std::pow(c, l_target + 1));
    // T level_eb = cc * tol / C2;
    // for (int l = 0; l < l_target + 1; l++) {
    //   quantizers[l] = level_eb;
    //   // T c = std::sqrt(std::pow(2, 2*s*l + D * (l_target - l)));
    //   level_eb *= c;
    //   if (reciprocal)
    //     quantizers[l] = 1.0f / quantizers[l];
    // }

    // ben - uniform
    // printf("quantizers: ");

    // size_t dof = 1;
    // for (int d = 0; d < D; d++) dof *= handle.dofs[d][0];
    // printf("tol: %f, dof: %llu\n", tol, dof);
    // printf ("dof = %llu\n", dof);
    for (int l = 0; l < l_target + 1; l++) {

      quantizers[l] = (abs_tol) / (std::exp2(s * l) * std::sqrt(dof));

      // printf("l %d, vol: %f quantizer: %f \n", l, std::pow(2, (l_target - l)
      // * D), quantizers[l]);

      // printf("tol: %f quant: %e \n", tol, quantizers[l]);
      if (reciprocal)
        quantizers[l] = 1.0f / quantizers[l];
    }
    // printf("\n");
  }

  // print quantizers
  // printf("quantizers: ");
  // for (int l = 0; l < l_target+1; l++) {
  //   printf("%f ", 1.0f/quantizers[l]);
  // }
  // printf("\n");
}

#define MGARDX_QUANTIZE 1
#define MGARDX_DEQUANTIZE 2

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, OPTION OP,
          typename DeviceType>
class LevelwiseLinearQuantizerNDFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT LevelwiseLinearQuantizerNDFunctor() {}
  MGARDX_CONT LevelwiseLinearQuantizerNDFunctor(
      SubArray<2, SIZE, DeviceType> level_ranges, SIZE l_target,
      SubArray<1, T, DeviceType> quantizers,
      SubArray<3, T, DeviceType> level_volumes, SubArray<D, T, DeviceType> v,
      SubArray<D, QUANTIZED_INT, DeviceType> work, bool prep_huffman,
      bool calc_vol, SIZE dict_size,
      SubArray<1, LENGTH, DeviceType> outlier_count,
      SubArray<1, LENGTH, DeviceType> outlier_idx,
      SubArray<1, QUANTIZED_INT, DeviceType> outliers)
      : level_ranges(level_ranges), l_target(l_target), quantizers(quantizers),
        level_volumes(level_volumes), v(v), work(work),
        prep_huffman(prep_huffman), calc_vol(calc_vol), dict_size(dict_size),
        outlier_count(outlier_count), outlier_idx(outlier_idx),
        outliers(outliers) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    threadId = (FunctorBase<DeviceType>::GetThreadIdZ() *
                (FunctorBase<DeviceType>::GetBlockDimX() *
                 FunctorBase<DeviceType>::GetBlockDimY())) +
               (FunctorBase<DeviceType>::GetThreadIdY() *
                FunctorBase<DeviceType>::GetBlockDimX()) +
               FunctorBase<DeviceType>::GetThreadIdX();

    Byte *sm = FunctorBase<DeviceType>::GetSharedMemory();
    volumes_0 = (T *)sm;
    if (calc_vol)
      sm += roundup<SIZE>(FunctorBase<DeviceType>::GetBlockDimX() *
                          (l_target + 1) * sizeof(T));
    volumes_1 = (T *)sm;
    if (calc_vol)
      sm += roundup<SIZE>(FunctorBase<DeviceType>::GetBlockDimY() *
                          (l_target + 1) * sizeof(T));
    volumes_2 = (T *)sm;
    if (calc_vol)
      sm += roundup<SIZE>(FunctorBase<DeviceType>::GetBlockDimZ() *
                          (l_target + 1) * sizeof(T));
    volumes_3_plus = (T *)sm;
    if (calc_vol && D > 3)
      sm += roundup<SIZE>((D - 3) * (l_target + 1) * sizeof(T));

    // determine global idx
    SIZE firstD = div_roundup(v.shape(D - 1), F);

    SIZE bidx = FunctorBase<DeviceType>::GetBlockIdX();
    idx[D - 1] = (bidx % firstD) * F + FunctorBase<DeviceType>::GetThreadIdX();
    idx0[D - 1] = (bidx % firstD) * F;

    bidx /= firstD;
    if (D >= 2) {
      idx[D - 2] = FunctorBase<DeviceType>::GetBlockIdY() *
                       FunctorBase<DeviceType>::GetBlockDimY() +
                   FunctorBase<DeviceType>::GetThreadIdY();
      idx0[D - 2] = FunctorBase<DeviceType>::GetBlockIdY() *
                    FunctorBase<DeviceType>::GetBlockDimY();
    }
    if (D >= 3) {
      idx[D - 3] = FunctorBase<DeviceType>::GetBlockIdZ() *
                       FunctorBase<DeviceType>::GetBlockDimZ() +
                   FunctorBase<DeviceType>::GetThreadIdZ();
      idx0[D - 3] = FunctorBase<DeviceType>::GetBlockIdZ() *
                    FunctorBase<DeviceType>::GetBlockDimZ();
    }

    for (int d = D - 4; d >= 0; d--) {
      idx[d] = bidx % v.shape(d);
      idx0[d] = idx[d];
      bidx /= v.shape(d);
    }

    if (calc_vol) {
      // cache volumes
      for (int l = 0; l < l_target + 1; l++) {
        // volumes 0
        if (threadId < FunctorBase<DeviceType>::GetBlockDimX() &&
            idx0[D - 1] + threadId < v.shape(D - 1)) {
          volumes_0[l * FunctorBase<DeviceType>::GetBlockDimX() + threadId] =
              *level_volumes(l, D - 1, idx0[D - 1] + threadId);
        }
        if (D >= 2) {
          // volumes 1
          if (threadId < FunctorBase<DeviceType>::GetBlockDimY() &&
              idx0[D - 2] + threadId < v.shape(D - 2)) {
            volumes_1[l * FunctorBase<DeviceType>::GetBlockDimY() + threadId] =
                *level_volumes(l, D - 2, idx0[D - 2] + threadId);
          }
        }
        if (D >= 3) {
          // volumes 2
          if (threadId < FunctorBase<DeviceType>::GetBlockDimZ() &&
              idx0[D - 3] + threadId < v.shape(D - 3)) {
            volumes_2[l * FunctorBase<DeviceType>::GetBlockDimZ() + threadId] =
                *level_volumes(l, D - 3, idx0[D - 3] + threadId);
          }
        }
      }

      if (D >= 4) {
        if (threadId < 1) {
          // for (int d = 3; d < D; d++) {
          for (int d = D - 4; d >= 0; d--) {
            for (int l = 0; l < l_target + 1; l++) {
              volumes_3_plus[l * (D - 3) + d] = *level_volumes(l, d, idx0[d]);
            }
          }
        }
      }
    }
  }

  MGARDX_EXEC void Operation2() {
    int level = 0;
    for (int d = D - 1; d >= 0; d--) {
      long long unsigned int l_bit = 0l;
      for (SIZE l = 0; l < l_target + 1; l++) {
        long long unsigned int bit = (idx[d] >= *level_ranges(l, d)) &&
                                     (idx[d] < *level_ranges(l + 1, d));
        l_bit += bit << l;
        // printf("idx: %d %d d: %d l_bit: %llu\n", idx[1], idx[0], d, l_bit);
      }
      level = Math<DeviceType>::Max(level, Math<DeviceType>::ffsll(l_bit));
    }
    level = level - 1;

    bool in_range = true;
    for (int d = D - 1; d >= 0; d--) {
      if (idx[d] >= v.shape(d))
        in_range = false;
    }

    if (level >= 0 && level <= l_target && in_range) {
      T t = v[idx];
      T volume = 1;
      if (calc_vol) {
        volume *= volumes_0[level * FunctorBase<DeviceType>::GetBlockDimX() +
                            FunctorBase<DeviceType>::GetThreadIdX()];
        if (D >= 2) {
          volume *= volumes_1[level * FunctorBase<DeviceType>::GetBlockDimY() +
                              FunctorBase<DeviceType>::GetThreadIdY()];
        }
        if (D >= 3) {
          volume *= volumes_2[level * FunctorBase<DeviceType>::GetBlockDimZ() +
                              FunctorBase<DeviceType>::GetThreadIdZ()];
        }
        if (D >= 4) {
          for (int d = D - 4; d >= 0; d--) {
            volume *= volumes_3_plus[level * (D - 3) + d];
          }
        }
        if (sizeof(T) == sizeof(double))
          volume = sqrt(volume);
        else if (sizeof(T) == sizeof(float))
          volume = sqrtf(volume);
      }

      T quantizer = *quantizers(level);
      QUANTIZED_INT quantized_data;

      if constexpr (OP == MGARDX_QUANTIZE) {
        quantized_data =
            copysign(0.5 + fabs(t * quantizer * (1.0 / volume)), t);
        if (prep_huffman) {
          quantized_data += dict_size / 2;
          if (quantized_data >= 0 && quantized_data < dict_size) {
            // do nothing
          } else {
            LENGTH i =
                Atomic<LENGTH, AtomicGlobalMemory, AtomicDeviceScope,
                       DeviceType>::Add(outlier_count((IDX)0), (LENGTH)1);
            // Get linearized index
            LENGTH curr_stride = 1;
            LENGTH linearized_idx = 0;
            for (int d = D - 1; d >= 0; d--) {
              linearized_idx += idx[d] * curr_stride;
              curr_stride *= v.shape(d);
            }
            // Avoid out of range error
            // If we have too much outlier than our allocation 
            // we return the true outlier_count and do quanziation again
            if (i < outlier_idx.shape(0)) {
              *outlier_idx(i) = linearized_idx;
              *outliers(i) = quantized_data;
            }
            quantized_data = 0;
          }
        }
        work[idx] = quantized_data;
      } else if constexpr (OP == MGARDX_DEQUANTIZE) {
        quantized_data = work[idx];
        if (prep_huffman) {
          quantized_data -= dict_size / 2;
        }
        v[idx] = (quantizer * volume) * (T)quantized_data;
      }
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    size += roundup<SIZE>(F * (l_target + 1) * sizeof(T));
    size += roundup<SIZE>(C * (l_target + 1) * sizeof(T));
    size += roundup<SIZE>(R * (l_target + 1) * sizeof(T));
    if (D > 3)
      size += roundup<SIZE>((D - 3) * (l_target + 1) * sizeof(T));
    return size;
  }

private:
  IDX threadId;
  SubArray<2, SIZE, DeviceType> level_ranges;
  SIZE l_target;
  SubArray<1, T, DeviceType> quantizers;
  SubArray<3, T, DeviceType> level_volumes;
  SubArray<D, T, DeviceType> v;
  SubArray<D, QUANTIZED_INT, DeviceType> work;
  bool prep_huffman;
  bool calc_vol;
  SIZE dict_size;
  SubArray<1, SIZE, DeviceType> shape;
  SubArray<1, LENGTH, DeviceType> outlier_count;
  SubArray<1, LENGTH, DeviceType> outlier_idx;
  SubArray<1, QUANTIZED_INT, DeviceType> outliers;

  T *volumes_0;
  T *volumes_1;
  T *volumes_2;
  T *volumes_3_plus;

  SIZE idx[D];  // thread global idx
  SIZE idx0[D]; // block global idx
};

template <DIM D, typename T, typename DeviceType>
class OutlierRestoreFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT OutlierRestoreFunctor() {}
  MGARDX_CONT
  OutlierRestoreFunctor(SubArray<D, QUANTIZED_INT, DeviceType> work,
                        LENGTH outlier_count,
                        SubArray<1, LENGTH, DeviceType> outlier_idx,
                        SubArray<1, QUANTIZED_INT, DeviceType> outliers)
      : work(work), outlier_count(outlier_count), outlier_idx(outlier_idx),
        outliers(outliers) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    threadId = (FunctorBase<DeviceType>::GetThreadIdZ() *
                (FunctorBase<DeviceType>::GetBlockDimX() *
                 FunctorBase<DeviceType>::GetBlockDimY())) +
               (FunctorBase<DeviceType>::GetThreadIdY() *
                FunctorBase<DeviceType>::GetBlockDimX()) +
               FunctorBase<DeviceType>::GetThreadIdX();
    blockId = (FunctorBase<DeviceType>::GetBlockIdZ() *
               (FunctorBase<DeviceType>::GetGridDimX() *
                FunctorBase<DeviceType>::GetGridDimY())) +
              (FunctorBase<DeviceType>::GetBlockIdY() *
               FunctorBase<DeviceType>::GetGridDimX()) +
              FunctorBase<DeviceType>::GetBlockIdX();
    gloablId = blockId * FunctorBase<DeviceType>::GetBlockDimX() *
                   FunctorBase<DeviceType>::GetBlockDimY() *
                   FunctorBase<DeviceType>::GetBlockDimZ() +
               threadId;

    if (gloablId < outlier_count) {
      LENGTH linerized_idx = *outlier_idx(gloablId);
      QUANTIZED_INT outliter = *outliers(gloablId);
      *work(linerized_idx) = outliter;
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  IDX threadId, blockId, gloablId;
  SubArray<D, QUANTIZED_INT, DeviceType> work;
  LENGTH outlier_count;
  SubArray<1, LENGTH, DeviceType> outlier_idx;
  SubArray<1, QUANTIZED_INT, DeviceType> outliers;
};

template <DIM D, typename T, OPTION OP, typename DeviceType>
class LevelwiseLinearQuantizerND : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  LevelwiseLinearQuantizerND() : AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT
      Task<LevelwiseLinearQuantizerNDFunctor<D, T, R, C, F, OP, DeviceType>>
      GenTaskQuantizer(SubArray<2, SIZE, DeviceType> level_ranges,
                       SIZE l_target, SubArray<1, T, DeviceType> quantizers,
                       SubArray<3, T, DeviceType> level_volumes, T s,
                       SIZE huff_dict_size, SubArray<D, T, DeviceType> v,
                       SubArray<D, QUANTIZED_INT, DeviceType> work,
                       bool prep_huffman,
                       // SubArray<1, SIZE, DeviceType> shape,
                       SubArray<1, LENGTH, DeviceType> outlier_count,
                       SubArray<1, LENGTH, DeviceType> outlier_idx,
                       SubArray<1, QUANTIZED_INT, DeviceType> outliers,
                       int queue_idx) {
    using FunctorType =
        LevelwiseLinearQuantizerNDFunctor<D, T, R, C, F, OP, DeviceType>;

    bool calc_vol =
        s != std::numeric_limits<T>::infinity(); // m.ntype == norm_type::L_2;
    FunctorType functor(level_ranges, l_target, quantizers, level_volumes, v,
                        work, prep_huffman, calc_vol, huff_dict_size,
                        outlier_count, outlier_idx, outliers);

    SIZE total_thread_z = v.shape(D - 3);
    SIZE total_thread_y = v.shape(D - 2);
    SIZE total_thread_x = v.shape(D - 1);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);
    for (DIM d = 3; d < D; d++) {
      gridx *= v.shape(D - (d + 1));
    }

    // printf("%u %u %u %u %u %u %u %u %u\n", total_thread_x, total_thread_y,
    // total_thread_z, tbx, tby, tbz, gridx, gridy, gridz);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "LevelwiseLinearQuantizerND");
  }

  template <SIZE F>
  MGARDX_CONT Task<OutlierRestoreFunctor<D, T, DeviceType>> GenTaskOutlier(
      SubArray<D, QUANTIZED_INT, DeviceType> work, LENGTH outlier_count,
      SubArray<1, LENGTH, DeviceType> outlier_idx,
      SubArray<1, QUANTIZED_INT, DeviceType> outliers, int queue_idx) {
    using FunctorType = OutlierRestoreFunctor<D, T, DeviceType>;
    FunctorType functor(work, outlier_count, outlier_idx, outliers);
    SIZE total_thread_z = 1;
    SIZE total_thread_y = 1;
    SIZE total_thread_x = outlier_count;
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = F;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "OutlierRestore");
  }

  MGARDX_CONT
  void Execute(SubArray<2, SIZE, DeviceType> level_ranges, SIZE l_target,
               SubArray<1, T, DeviceType> quantizers,
               SubArray<3, T, DeviceType> level_volumes, T s,
               SIZE huff_dict_size, SubArray<D, T, DeviceType> v,
               SubArray<D, QUANTIZED_INT, DeviceType> work, bool prep_huffman,
               SubArray<1, LENGTH, DeviceType> outlier_count,
               SubArray<1, LENGTH, DeviceType> outlier_idx,
               SubArray<1, QUANTIZED_INT, DeviceType> outliers, int queue_idx) {

    if constexpr (OP == MGARDX_DEQUANTIZE) {
      LENGTH outlier_count_host;
      // Not providing queue_idx makes this Copy1D a synchronized call
      MemoryManager<DeviceType>::Copy1D(&outlier_count_host,
                                        outlier_count.data(), 1);
      if (prep_huffman && outlier_count_host) {
        using FunctorType = OutlierRestoreFunctor<D, T, DeviceType>;
        using TaskType = Task<FunctorType>;
        TaskType task = GenTaskOutlier<256>(work, outlier_count_host,
                                            outlier_idx, outliers, queue_idx);
        DeviceAdapter<TaskType, DeviceType> adapter;
        adapter.Execute(task);
      }
    }

    int range_l = std::min(6, (int)std::log2(v.shape(D - 1)) - 1);
    int prec = TypeToIdx<T>();
    int config = AutoTuner<DeviceType>::autoTuningTable.lwqzk[prec][range_l];
    double min_time = std::numeric_limits<double>::max();
    int min_config = 0;
    ExecutionReturn ret;

#define LWQZK(CONFIG)                                                          \
  if (config == CONFIG || AutoTuner<DeviceType>::ProfileKernels) {             \
    const int R = LWPK_CONFIG[D - 1][CONFIG][0];                               \
    const int C = LWPK_CONFIG[D - 1][CONFIG][1];                               \
    const int F = LWPK_CONFIG[D - 1][CONFIG][2];                               \
    using FunctorType =                                                        \
        LevelwiseLinearQuantizerNDFunctor<D, T, R, C, F, OP, DeviceType>;      \
    using TaskType = Task<FunctorType>;                                        \
    TaskType task = GenTaskQuantizer<R, C, F>(                                 \
        level_ranges, l_target, quantizers, level_volumes, s, huff_dict_size,  \
        v, work, prep_huffman, outlier_count, outlier_idx, outliers,           \
        queue_idx);                                                            \
    DeviceAdapter<TaskType, DeviceType> adapter;                               \
    ret = adapter.Execute(task);                                               \
    if (AutoTuner<DeviceType>::ProfileKernels) {                               \
      if (ret.success && min_time > ret.execution_time) {                      \
        min_time = ret.execution_time;                                         \
        min_config = CONFIG;                                                   \
      }                                                                        \
    }                                                                          \
  }
    LWQZK(6) if (!ret.success) config--;
    LWQZK(5) if (!ret.success) config--;
    LWQZK(4) if (!ret.success) config--;
    LWQZK(3) if (!ret.success) config--;
    LWQZK(2) if (!ret.success) config--;
    LWQZK(1) if (!ret.success) config--;
    LWQZK(0) if (!ret.success) config--;
    if (config < 0 && !ret.success) {
      std::cout << log::log_err
                << "no suitable config for LevelwiseLinearQuantizerND.\n";
      exit(-1);
    }
#undef LWQZK
    if (AutoTuner<DeviceType>::ProfileKernels) {
      FillAutoTunerTable<DeviceType>("lwqzk", prec, range_l, min_config);
    }
  }
};

} // namespace mgard_x

#endif