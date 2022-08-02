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

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, typename DeviceType>
class LevelwiseLinearQuantizeNDFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT LevelwiseLinearQuantizeNDFunctor() {}
  MGARDX_CONT LevelwiseLinearQuantizeNDFunctor(
      SubArray<2, SIZE, DeviceType> level_ranges,
      SIZE l_target,
      SubArray<1, T, DeviceType> quantizers,
      SubArray<3, T, DeviceType> level_volumes,
      SubArray<D, T, DeviceType> v, SubArray<D, QUANTIZED_INT, DeviceType> work,
      bool prep_huffman, bool calc_vol, SIZE dict_size,
      SubArray<1, LENGTH, DeviceType> outlier_count,
      SubArray<1, LENGTH, DeviceType> outlier_idx,
      SubArray<1, QUANTIZED_INT, DeviceType> outliers)
      : level_ranges(level_ranges), l_target(l_target), quantizers(quantizers),
        level_volumes(level_volumes), v(v), work(work), prep_huffman(prep_huffman),
        calc_vol(calc_vol), dict_size(dict_size),
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
    SIZE firstD = div_roundup(v.shape(D-1), F);

    SIZE bidx = FunctorBase<DeviceType>::GetBlockIdX();
    idx[D-1] = (bidx % firstD) * F + FunctorBase<DeviceType>::GetThreadIdX();
    idx0[D-1] = (bidx % firstD) * F;

    bidx /= firstD;
    if (D >= 2) {
      idx[D-2] = FunctorBase<DeviceType>::GetBlockIdY() *
                   FunctorBase<DeviceType>::GetBlockDimY() +
               FunctorBase<DeviceType>::GetThreadIdY();
      idx0[D-2] = FunctorBase<DeviceType>::GetBlockIdY() *
                FunctorBase<DeviceType>::GetBlockDimY();
    }
    if (D >= 3) {
      idx[D-3] = FunctorBase<DeviceType>::GetBlockIdZ() *
                   FunctorBase<DeviceType>::GetBlockDimZ() +
               FunctorBase<DeviceType>::GetThreadIdZ();
      idx0[D-3] = FunctorBase<DeviceType>::GetBlockIdZ() *
                FunctorBase<DeviceType>::GetBlockDimZ();
    }

    for (int d = D-4; d >= 0; d--) {
      idx[d] = bidx % v.shape(d);
      idx0[d] = idx[d];
      bidx /= v.shape(d);
    }

    if (calc_vol) {
      // cache volumes
      for (int l = 0; l < l_target + 1; l++) {
        // volumes 0
        if (threadId < FunctorBase<DeviceType>::GetBlockDimX() &&
            idx0[D-1] + threadId < v.shape(D-1)) {
          volumes_0[l * FunctorBase<DeviceType>::GetBlockDimX() + threadId] =
              *level_volumes(l, D-1, idx0[D-1] + threadId);
        }
        if (D >= 2) {
          // volumes 1
          if (threadId < FunctorBase<DeviceType>::GetBlockDimY() &&
              idx0[D-2] + threadId < v.shape(D-2)) {
            volumes_1[l * FunctorBase<DeviceType>::GetBlockDimY() + threadId] =
              *level_volumes(l, D-2, idx0[D-2] + threadId);
          }
        }
        if (D >= 3) {
          // volumes 2
          if (threadId < FunctorBase<DeviceType>::GetBlockDimZ() &&
              idx0[D-3] + threadId < v.shape(D-3)) {
            volumes_2[l * FunctorBase<DeviceType>::GetBlockDimZ() + threadId] =
              *level_volumes(l, D-3, idx0[D-3] + threadId);
          }
        }
      }

      if (D >= 4) {
        if (threadId < 1) {
          // for (int d = 3; d < D; d++) {
          for (int d = D-4; d >=0; d--) {
            for (int l = 0; l < l_target + 1; l++) {
              volumes_3_plus[l * (D-3) + d] =
                  *level_volumes(l, d, idx0[d]);
            }
          }
        }
      }
    }
  }

  MGARDX_EXEC void Operation2() {
    int level = 0;
    for (int d = D-1; d >= 0; d--) {
      long long unsigned int l_bit = 0l;
      for (SIZE l = 0; l < l_target + 1; l++) {
        int bit = (idx[d] >= *level_ranges(l,   d)) &&
                  (idx[d] <  *level_ranges(l+1, d));
        l_bit += bit << l;
        // printf("idx: %d %d d: %d l_bit: %llu\n", idx[1], idx[0], d, l_bit);
      }
      level = Math<DeviceType>::Max(level, Math<DeviceType>::ffsll(l_bit));
    }
    level = level - 1;

    bool in_range = true;
    for (int d = D-1; d >= 0; d--) {
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
          for (int d = D-4; d >= 0; d--) {
            volume *= volumes_3_plus[level*(D-3)+d];
          }
        }
        if (sizeof(T) == sizeof(double))
          volume = sqrt(volume);
        else if (sizeof(T) == sizeof(float))
          volume = sqrtf(volume);
      }

      T quantizer = *quantizers(level);
      QUANTIZED_INT quantized_data =
          copysign(0.5 + fabs(t * quantizer * (1.0 / volume)), t);
      // QUANTIZED_INT quantized_data =
      //     copysign(0.5 + fabs(t / (quantizers_sm[level]/volume)), t);

      // printf("%f / %f * %f = %d\n", t, quantizers_sm[level], volume,
      // quantized_data);

      if (prep_huffman) {
        quantized_data += dict_size / 2;
        if (quantized_data >= 0 && quantized_data < dict_size) {
          // do nothing
        } else {
          LENGTH i = Atomic<LENGTH, AtomicGlobalMemory, AtomicDeviceScope,
                            DeviceType>::Add(outlier_count((IDX)0), (LENGTH)1);
          // Get linearized index
          LENGTH curr_stride = 1;
          LENGTH linearized_idx = 0;
          for (int d = D-1; d >= 0; d--) {
            linearized_idx += idx[d] * curr_stride;
            curr_stride *= v.shape(d);
          }
          *outlier_idx(i) = linearized_idx;
          *outliers(i) = quantized_data;
          quantized_data = 0;
        }
      }

      work[idx] = quantized_data;
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
class LevelwiseLinearQuantizeND : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  LevelwiseLinearQuantizeND() : AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<LevelwiseLinearQuantizeNDFunctor<D, T, R, C, F, DeviceType>>
  GenTask(SubArray<2, SIZE, DeviceType> level_ranges,
          SIZE l_target,
          SubArray<1, T, DeviceType> quantizers,
          SubArray<3, T, DeviceType> level_volumes,
          T s, SIZE huff_dict_size,
          SubArray<D, T, DeviceType> v,
          SubArray<D, QUANTIZED_INT, DeviceType> work, bool prep_huffman,
          // SubArray<1, SIZE, DeviceType> shape,
          SubArray<1, LENGTH, DeviceType> outlier_count,
          SubArray<1, LENGTH, DeviceType> outlier_idx,
          SubArray<1, QUANTIZED_INT, DeviceType> outliers, int queue_idx) {
    using FunctorType =
        LevelwiseLinearQuantizeNDFunctor<D, T, R, C, F, DeviceType>;

    bool calc_vol =
        s != std::numeric_limits<T>::infinity(); // m.ntype == norm_type::L_2;
    FunctorType functor(level_ranges, l_target, quantizers, level_volumes, v, work,
                        prep_huffman, calc_vol, huff_dict_size,
                        outlier_count, outlier_idx, outliers);

    SIZE total_thread_z = v.shape(D-3);
    SIZE total_thread_y = v.shape(D-2);
    SIZE total_thread_x = v.shape(D-1);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);
    for (DIM d = 3; d < D; d++) {
      gridx *= v.shape(D-(d+1));
    }

    // printf("%u %u %u %u %u %u %u %u %u\n", total_thread_x, total_thread_y,
    // total_thread_z, tbx, tby, tbz, gridx, gridy, gridz);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "LevelwiseLinearQuantizeND");
  }

  MGARDX_CONT
  void Execute(SubArray<2, SIZE, DeviceType> level_ranges,
               SIZE l_target,
               SubArray<1, T, DeviceType> quantizers,
               SubArray<3, T, DeviceType> level_volumes,
               T s, SIZE huff_dict_size,
               SubArray<D, T, DeviceType> v,
               SubArray<D, QUANTIZED_INT, DeviceType> work, bool prep_huffman,
               SubArray<1, LENGTH, DeviceType> outlier_count,
               SubArray<1, LENGTH, DeviceType> outlier_idx,
               SubArray<1, QUANTIZED_INT, DeviceType> outliers, int queue_idx) {

    int range_l = std::min(6, (int)std::log2(v.shape(D-1)) - 1);
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
        LevelwiseLinearQuantizeNDFunctor<D, T, R, C, F, DeviceType>;           \
    using TaskType = Task<FunctorType>;                                        \
    TaskType task = GenTask<R, C, F>(                                          \
        level_ranges, l_target, quantizers, level_volumes, s, huff_dict_size, v, work,     \
        prep_huffman, outlier_count, outlier_idx, outliers, queue_idx); \
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
                << "no suitable config for LevelwiseLinearQuantizeND.\n";
      exit(-1);
    }
#undef LWQZK
    if (AutoTuner<DeviceType>::ProfileKernels) {
      FillAutoTunerTable<DeviceType>("lwqzk", prec, range_l, min_config);
    }
  }
};

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, typename DeviceType>
class LevelwiseLinearDequantizeNDFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT LevelwiseLinearDequantizeNDFunctor() {}
  MGARDX_CONT LevelwiseLinearDequantizeNDFunctor(
      SubArray<1, SIZE, DeviceType> shapes, SIZE l_target,
      SubArray<1, T, DeviceType> quantizers, SubArray<2, T, DeviceType> volumes,
      SubArray<D, T, DeviceType> v, SubArray<D, QUANTIZED_INT, DeviceType> work,
      bool prep_huffman, bool calc_vol, SIZE dict_size,
      SubArray<1, SIZE, DeviceType> shape, LENGTH outlier_count,
      SubArray<1, LENGTH, DeviceType> outlier_idx,
      SubArray<1, QUANTIZED_INT, DeviceType> outliers)
      : shapes(shapes), l_target(l_target), quantizers(quantizers),
        volumes(volumes), v(v), work(work), prep_huffman(prep_huffman),
        calc_vol(calc_vol), dict_size(dict_size), shape(shape),
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

    // T * smT = (T*)FunctorBase<DeviceType>::GetSharedMemory();
    // quantizers_sm = smT; smT += roundup<T>(l_target + 1);

    // volumes_0 = smT; if (calc_vol) smT +=
    // roundup<T>(FunctorBase<DeviceType>::GetBlockDimX() * (l_target + 1));
    // volumes_1 = smT; if (calc_vol) smT +=
    // roundup<T>(FunctorBase<DeviceType>::GetBlockDimY() * (l_target + 1));
    // volumes_2 = smT; if (calc_vol) smT +=
    // roundup<T>(FunctorBase<DeviceType>::GetBlockDimZ() * (l_target + 1));
    // volumes_3_plus = smT;
    // if (calc_vol && D > 3) smT += roundup<T>((D-3) * (l_target + 1));

    // SIZE * smInt = (SIZE *)smT;
    // shape_sm = smInt; smInt += roundup<SIZE>(D);
    // shapes_sm = smInt; smInt += roundup<SIZE>(D * (l_target + 2));

    Byte *sm = FunctorBase<DeviceType>::GetSharedMemory();
    quantizers_sm = (T *)sm;
    sm += roundup<SIZE>((l_target + 1) * sizeof(T));

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

    shape_sm = (SIZE *)sm;
    sm += roundup<SIZE>(D * sizeof(SIZE));
    shapes_sm = (SIZE *)sm;
    sm += roundup<SIZE>(D * (l_target + 2) * sizeof(SIZE));

    if (threadId < l_target + 1) {
      quantizers_sm[threadId] = *quantizers(threadId);
    }
    if (threadId < D) {
      shape_sm[threadId] = *shape(threadId);
    }
    if (threadId < D * (l_target + 2)) {
      shapes_sm[threadId] = *shapes(threadId);
    }
  }

  MGARDX_EXEC void Operation2() {
    // determine global idx
    SIZE firstD = div_roundup(shapes_sm[l_target + 1], F);

    SIZE bidx = FunctorBase<DeviceType>::GetBlockIdX();
    idx[0] = (bidx % firstD) * F + FunctorBase<DeviceType>::GetThreadIdX();
    idx0[0] = (bidx % firstD) * F;

    // printf("shapes_sm[l_target+1]: %d firstD %d idx[0] %d\n",
    // shapes_sm[l_target+1], firstD, idx[0]);

    bidx /= firstD;
    if (D >= 2) {
      idx[1] = FunctorBase<DeviceType>::GetBlockIdY() *
                   FunctorBase<DeviceType>::GetBlockDimY() +
               FunctorBase<DeviceType>::GetThreadIdY();
      idx0[1] = FunctorBase<DeviceType>::GetBlockIdY() *
                FunctorBase<DeviceType>::GetBlockDimY();
    }
    if (D >= 3) {
      idx[2] = FunctorBase<DeviceType>::GetBlockIdZ() *
                   FunctorBase<DeviceType>::GetBlockDimZ() +
               FunctorBase<DeviceType>::GetThreadIdZ();
      idx0[2] = FunctorBase<DeviceType>::GetBlockIdZ() *
                FunctorBase<DeviceType>::GetBlockDimZ();
    }

    for (int d = 3; d < D; d++) {
      idx[d] = bidx % shapes_sm[(l_target + 2) * d + l_target + 1];
      idx0[d] = idx[d];
      bidx /= shapes_sm[(l_target + 2) * d + l_target + 1];
    }

    if (calc_vol) {
      // cache volumes
      for (int l = 0; l < l_target + 1; l++) {
        // volumes 0
        if (threadId < FunctorBase<DeviceType>::GetBlockDimX() &&
            idx0[0] + threadId < shapes_sm[(l_target + 2) * 0 + l_target + 1]) {
          volumes_0[l * FunctorBase<DeviceType>::GetBlockDimX() + threadId] =
              *volumes((0 * (l_target + 1) + l), +idx0[0] + threadId);
          // printf("load %f\n", volumes[(0 * (l_target + 1) + l) * ldvolumes +
          // idx0[0] + threadId]);
        }
        if (D >= 2) {
          // volumes 1
          if (threadId < FunctorBase<DeviceType>::GetBlockDimY() &&
              idx0[1] + threadId <
                  shapes_sm[(l_target + 2) * 1 + l_target + 1]) {
            volumes_1[l * FunctorBase<DeviceType>::GetBlockDimY() + threadId] =
                *volumes((1 * (l_target + 1) + l), idx0[1] + threadId);
          }
        }
        if (D >= 3) {
          // volumes 2
          if (threadId < FunctorBase<DeviceType>::GetBlockDimZ() &&
              idx0[2] + threadId <
                  shapes_sm[(l_target + 2) * 2 + l_target + 1]) {
            volumes_2[l * FunctorBase<DeviceType>::GetBlockDimZ() + threadId] =
                *volumes((2 * (l_target + 1) + l), idx0[2] + threadId);
          }
        }
      }

      if (D >= 4) {
        if (threadId < 1) {
          for (int d = 3; d < D; d++) {
            for (int l = 0; l < l_target + 1; l++) {
              volumes_3_plus[(d - 3) * (l_target + 1) + l] =
                  *volumes((d * (l_target + 1) + l), idx[d]);
            }
          }
        }
      }
    }

    // if (blockIdx.y == 0 && blockIdx.x == 0 && blockIdx.z == 0 && threadId ==
    // 0) {
    //   printf("volumes_0: ");
    //   for (int l = 0; l < l_target+1; l++) {
    //     printf("l = %d\n", l);
    //     for (int i = 0; i < min(blockDim.x, shapes_sm[(l_target + 2) * 0 +
    //     l_target + 1]) ; i++) {
    //       printf("%f ", volumes_0[l * blockDim.x + i]);
    //     }
    //     printf("\n");
    //   }
    //   printf("\n");
    //   if (D >= 2) {
    //     printf("volumes_1: ");
    //     for (int l = 0; l < l_target+1; l++) {
    //       printf("l = %d\n", l);
    //       for (int i = 0; i < min(blockDim.y, shapes_sm[(l_target + 2) * 1 +
    //       l_target + 1]); i++) {
    //         printf("%f ", volumes_1[l * blockDim.y + i]);
    //       }
    //       printf("\n");
    //     }

    //     printf("\n");
    //   }
    //   if (D >= 3) {
    //     printf("volumes_2: ");
    //     for (int l = 0; l < l_target+1; l++) {
    //       printf("l = %d\n", l);
    //       for (int i = 0; i < min(blockDim.z, shapes_sm[(l_target + 2) * 2 +
    //       l_target + 1]); i++) {
    //         printf("%f ", volumes_2[l * blockDim.y + i]);
    //       }
    //       printf("\n");
    //     }
    //   }
    // }
  }

  MGARDX_EXEC void Operation3() {
    int level = 0;
    for (DIM d = 0; d < D; d++) {
      long long unsigned int l_bit = 0l;
      for (SIZE l = 0; l < l_target + 1; l++) {
        int bit = (idx[d] >= shapes_sm[(l_target + 2) * d + l]) &&
                  (idx[d] < shapes_sm[(l_target + 2) * d + l + 1]);
        l_bit += bit << l;
        // printf("idx: %d %d d: %d l_bit: %llu\n", idx[1], idx[0], d, l_bit);
      }
      level = Math<DeviceType>::Max(level, Math<DeviceType>::ffsll(l_bit));
    }
    level = level - 1;

    bool in_range = true;
    for (DIM d = 0; d < D; d++) {
      if (idx[d] >= shapes_sm[(l_target + 2) * d + l_target + 1])
        in_range = false;
    }

    // printf("idx %llu, level: %d, in_range: %d idx[0]: shape_sm: %d\n",
    // get_idx<D>(shape_sm, idx), level, in_range, shapes_sm[(l_target+2) * 0 +
    // l_target+1]);

    if (level >= 0 && level <= l_target && in_range) {
      // printf("%d %d %d %d\n", idx[3], idx[2], idx[1], idx[0]);
      // printf("idx: %d %d l: %d\n", idx[1], idx[0], level);
      QUANTIZED_INT quantized_data = *work(idx);
      T volume = 1;
      if (calc_vol) {
        volume *= volumes_0[level * FunctorBase<DeviceType>::GetBlockDimX() +
                            FunctorBase<DeviceType>::GetThreadIdX()];
        if (D >= 2)
          volume *= volumes_1[level * FunctorBase<DeviceType>::GetBlockDimY() +
                              FunctorBase<DeviceType>::GetThreadIdY()];
        if (D >= 3)
          volume *= volumes_2[level * FunctorBase<DeviceType>::GetBlockDimZ() +
                              FunctorBase<DeviceType>::GetThreadIdZ()];
        if (D >= 4) {
          for (int d = 3; d < D; d++) {
            volume *= volumes_3_plus[(d - 3) * (l_target + 1) + level];
          }
        }
        if (sizeof(T) == sizeof(double))
          volume = sqrt(volume);
        else if (sizeof(T) == sizeof(float))
          volume = sqrtf(volume);
      }

      if (prep_huffman) {
        quantized_data -= dict_size / 2;
      }

      // printf("%d %d %d %d %d %d vol %f (%f * %f * %f), dequantizers: %f,
      // before: %d, dequantized: %f\n", blockIdx.z, blockIdx.y, blockIdx.x,
      // threadIdx.z, threadIdx.y, threadIdx.x, volume,
      //   volumes_0[level * blockDim.x + threadIdx.x], volumes_1[level *
      //   blockDim.y + threadIdx.y], volumes_2[level * blockDim.z +
      //   threadIdx.z], quantizers_sm[level] / volume, quantized_data,
      //   (quantizers_sm[level] / volume) * (T)quantized_data);
      *v(idx) = (quantizers_sm[level] * volume) * (T)quantized_data;
      // dwork[get_idx<D>(ldws, idx)] = (quantizers_sm[level] / volume) *
      // (T)quantized_data; dwork[get_idx<D>(ldws, idx)] =
      // (T)dv[get_idx<D>(ldvs, idx)];

      // printf("dw[%llu] %d dequantizers[%d]%f -> dw[%llu]%f \n",
      // get_idx<D>(ldvs, idx),
      //       quantized_data, level, quantizers_sm[level], get_idx<D>(ldws,
      //       idx), quantizers_sm[level] * (T)quantized_data);
    }
  }

  MGARDX_EXEC void Operation4() {}

  MGARDX_EXEC void Operation5() {}

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = roundup<SIZE>(D * sizeof(SIZE));
    // quantizer
    size += roundup<SIZE>((l_target + 1) * sizeof(T));
    // ranges
    size += roundup<SIZE>((l_target + 2) * D * sizeof(SIZE));
    // volumes
    size += roundup<SIZE>(F * (l_target + 1) * sizeof(T));
    size += roundup<SIZE>(C * (l_target + 1) * sizeof(T));
    size += roundup<SIZE>(R * (l_target + 1) * sizeof(T));
    if (D > 3)
      size += roundup<SIZE>((D - 3) * (l_target + 1) * sizeof(T));
    return size;
  }

private:
  IDX threadId, blockId, gloablId;
  SubArray<1, SIZE, DeviceType> shapes;
  SIZE l_target;
  SubArray<1, T, DeviceType> quantizers;
  SubArray<2, T, DeviceType> volumes;
  SubArray<D, T, DeviceType> v;
  SubArray<D, QUANTIZED_INT, DeviceType> work;
  bool prep_huffman;
  bool calc_vol;
  SIZE dict_size;
  SubArray<1, SIZE, DeviceType> shape;
  LENGTH outlier_count;
  SubArray<1, LENGTH, DeviceType> outlier_idx;
  SubArray<1, QUANTIZED_INT, DeviceType> outliers;

  T *quantizers_sm;
  T *volumes_0;
  T *volumes_1;
  T *volumes_2;
  T *volumes_3_plus;

  SIZE *shape_sm;
  SIZE *shapes_sm;

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

  MGARDX_EXEC void Operation2() {}

  MGARDX_EXEC void Operation3() {}

  MGARDX_EXEC void Operation4() {}

  MGARDX_EXEC void Operation5() {}

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  IDX threadId, blockId, gloablId;
  SubArray<D, QUANTIZED_INT, DeviceType> work;
  LENGTH outlier_count;
  SubArray<1, LENGTH, DeviceType> outlier_idx;
  SubArray<1, QUANTIZED_INT, DeviceType> outliers;
};

template <DIM D, typename T, typename DeviceType>
class LevelwiseLinearDequantizeND : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  LevelwiseLinearDequantizeND() : AutoTuner<DeviceType>() {}

  template <SIZE F>
  MGARDX_CONT Task<OutlierRestoreFunctor<D, T, DeviceType>>
  GenTask1(SubArray<D, QUANTIZED_INT, DeviceType> work, LENGTH outlier_count,
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

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT
      Task<LevelwiseLinearDequantizeNDFunctor<D, T, R, C, F, DeviceType>>
      GenTask2(SubArray<1, SIZE, DeviceType> ranges, SIZE l_target,
               SubArray<1, T, DeviceType> quantizers,
               SubArray<2, T, DeviceType> volumes, T s, SIZE huff_dict_size,
               SubArray<D, T, DeviceType> v,
               SubArray<D, QUANTIZED_INT, DeviceType> work, bool prep_huffman,
               SubArray<1, SIZE, DeviceType> shape, LENGTH outlier_count,
               SubArray<1, LENGTH, DeviceType> outlier_idx,
               SubArray<1, QUANTIZED_INT, DeviceType> outliers, int queue_idx) {
    using FunctorType =
        LevelwiseLinearDequantizeNDFunctor<D, T, R, C, F, DeviceType>;

    // T *quantizers = new T[l_target + 1];
    // size_t dof = 1;
    // for (int d = 0; d < D; d++) dof *= shape.dataHost()[d];
    // calc_quantizers<D, T>(dof, quantizers, m, false);

    // Array<1, T, DeviceType> quantizers_array({l_target + 1});
    // quantizers_array.loadData(quantizers);

    // SubArray<1, T, DeviceType> quantizers_subarray(quantizers_array);

    // bool calc_vol = m.ntype == norm_type::L_2;
    bool calc_vol = s != std::numeric_limits<T>::infinity();
    FunctorType functor(ranges, l_target, quantizers, volumes, v, work,
                        prep_huffman, calc_vol, huff_dict_size, shape,
                        outlier_count, outlier_idx, outliers);

    SIZE total_thread_z = shape.dataHost()[2];
    SIZE total_thread_y = shape.dataHost()[1];
    SIZE total_thread_x = shape.dataHost()[0];

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);
    for (DIM d = 3; d < D; d++) {
      gridx *= shape.dataHost()[d];
    }

    // printf("%u %u %u %u %u %u %u %u %u\n", total_thread_x, total_thread_y,
    // total_thread_z, tbx, tby, tbz, gridx, gridy, gridz);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "LevelwiseLinearDequantizeND");
  }

  MGARDX_CONT
  void Execute(SubArray<1, SIZE, DeviceType> ranges, SIZE l_target,
               SubArray<1, T, DeviceType> quantizers,
               SubArray<2, T, DeviceType> volumes, T s, SIZE huff_dict_size,
               SubArray<D, T, DeviceType> v,
               SubArray<D, QUANTIZED_INT, DeviceType> work, bool prep_huffman,
               SubArray<1, SIZE, DeviceType> shape, LENGTH outlier_count,
               SubArray<1, LENGTH, DeviceType> outlier_idx,
               SubArray<1, QUANTIZED_INT, DeviceType> outliers, int queue_idx) {

    if (prep_huffman && outlier_count) {
      using FunctorType = OutlierRestoreFunctor<D, T, DeviceType>;
      using TaskType = Task<FunctorType>;
      TaskType task =
          GenTask1<256>(work, outlier_count, outlier_idx, outliers, queue_idx);
      DeviceAdapter<TaskType, DeviceType> adapter;
      adapter.Execute(task);
    }

    int range_l = std::min(6, (int)std::log2(v.getShape(0)) - 1);
    int prec = TypeToIdx<T>();
    int config = AutoTuner<DeviceType>::autoTuningTable.lwdqzk[prec][range_l];
    double min_time = std::numeric_limits<double>::max();
    int min_config = 0;
    ExecutionReturn ret;

#define LWDQZK(CONFIG)                                                         \
  if (config == CONFIG || AutoTuner<DeviceType>::ProfileKernels) {             \
    const int R = LWPK_CONFIG[D - 1][CONFIG][0];                               \
    const int C = LWPK_CONFIG[D - 1][CONFIG][1];                               \
    const int F = LWPK_CONFIG[D - 1][CONFIG][2];                               \
    using FunctorType =                                                        \
        LevelwiseLinearDequantizeNDFunctor<D, T, R, C, F, DeviceType>;         \
    using TaskType = Task<FunctorType>;                                        \
    TaskType task = GenTask2<R, C, F>(                                         \
        ranges, l_target, quantizers, volumes, s, huff_dict_size, v, work,     \
        prep_huffman, shape, outlier_count, outlier_idx, outliers, queue_idx); \
    DeviceAdapter<TaskType, DeviceType> adapter;                               \
    ret = adapter.Execute(task);                                               \
    if (AutoTuner<DeviceType>::ProfileKernels) {                               \
      if (ret.success && min_time > ret.execution_time) {                      \
        min_time = ret.execution_time;                                         \
        min_config = CONFIG;                                                   \
      }                                                                        \
    }                                                                          \
  }

    LWDQZK(6) if (!ret.success) config--;
    LWDQZK(5) if (!ret.success) config--;
    LWDQZK(4) if (!ret.success) config--;
    LWDQZK(3) if (!ret.success) config--;
    LWDQZK(2) if (!ret.success) config--;
    LWDQZK(1) if (!ret.success) config--;
    LWDQZK(0) if (!ret.success) config--;
    if (config < 0 && !ret.success) {
      std::cout << log::log_err
                << "no suitable config for LevelwiseLinearDequantizeND.\n";
      exit(-1);
    }
#undef LWDQZK
    if (AutoTuner<DeviceType>::ProfileKernels) {
      FillAutoTunerTable<DeviceType>("lwdqzk", prec, range_l, min_config);
    }
  }
};

} // namespace mgard_x

#endif