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
      SubArray<D, QUANTIZED_INT, DeviceType> quantized_v,
      SubArray<1, QUANTIZED_INT, DeviceType> *quantized_linearized_v,
      bool prep_huffman, bool calc_vol, bool level_linearize, SIZE dict_size,
      SubArray<1, LENGTH, DeviceType> outlier_count,
      SubArray<1, LENGTH, DeviceType> outlier_indexes,
      SubArray<1, QUANTIZED_INT, DeviceType> outliers)
      : level_ranges(level_ranges), l_target(l_target), quantizers(quantizers),
        level_volumes(level_volumes), v(v), quantized_v(quantized_v),
        quantized_linearized_v(quantized_linearized_v),
        prep_huffman(prep_huffman), calc_vol(calc_vol),
        level_linearize(level_linearize), dict_size(dict_size),
        outlier_count(outlier_count), outlier_indexes(outlier_indexes),
        outliers(outliers) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC SIZE calc_level_offset() {
    // Use curr_region to encode region id to distinguish different regions
    // curr_region of current level is always >=1,
    // since curr_region=0 refers to the next coarser level
    // most significant bit --> fastest dim
    // least signigiciant bit --> slowest dim
    SIZE curr_region = 0;
    for (int d = D - 1; d >= 0; d--) {
      SIZE bit = (level + 1) == Math<DeviceType>::ffsll(l_bit[d]);
      curr_region += bit << d;
    }

    // region size
    SIZE coarse_level_size[D];
    SIZE diff_level_size[D];
    for (int d = D - 1; d >= 0; d--) {
      coarse_level_size[d] = *level_ranges(level, d);
      diff_level_size[d] =
          *level_ranges(level + 1, d) - *level_ranges(level, d);
    }

    SIZE curr_region_dims[D];
    for (int d = D - 1; d >= 0; d--) {
      // Use region id to decode dimension of this region
      SIZE bit = (curr_region >> d) & 1u;
      curr_region_dims[d] = bit ? diff_level_size[d] : coarse_level_size[d];
    }

    SIZE curr_region_size = 1;
    for (int d = D - 1; d >= 0; d--) {
      curr_region_size *= curr_region_dims[d];
    }

    // region offset
    SIZE curr_region_offset = 0;
    // prev_region start with 1 since that is the region id of the first
    // region of current level
    for (SIZE prev_region = 1; prev_region < curr_region; prev_region++) {
      SIZE prev_region_size = 1;
      for (int d = D - 1; d >= 0; d--) {
        // Use region id to decode dimension of a previous region
        SIZE bit = (prev_region >> d) & 1u;
        // Calculate the num of elements of the previous region
        prev_region_size *= bit ? diff_level_size[d] : coarse_level_size[d];
      }
      curr_region_offset += prev_region_size;
    }

    // printf("(%u %u): level: %u, curr_region: %u, curr_region_offset: %u\n",
    // idx[0], idx[1], level, curr_region, curr_region_offset);

    // thread offset
    SIZE curr_region_thread_idx[D];
    SIZE curr_thread_offset = 0;
    SIZE coarse_level_offset = 0;
    for (int d = D - 1; d >= 0; d--) {
      SIZE bit = (curr_region >> d) & 1u;
      curr_region_thread_idx[d] = bit ? idx[d] - coarse_level_size[d] : idx[d];
    }

    SIZE global_data_idx[D];
    for (int d = D - 1; d >= 0; d--) {
      SIZE bit = (curr_region >> d) & 1u;
      if (level == 0) {
        global_data_idx[d] = curr_region_thread_idx[d];
      } else if (*level_ranges(level + 1, d) % 2 == 0 &&
                 curr_region_thread_idx[d] == *level_ranges(level + 1, d) / 2) {
        global_data_idx[d] = *level_ranges(level + 1, d) - 1;
      } else {
        global_data_idx[d] = curr_region_thread_idx[d] * 2 + bit;
      }
    }

    SIZE stride = 1;
    for (int d = D - 1; d >= 0; d--) {
      curr_thread_offset += global_data_idx[d] * stride;
      stride *= *level_ranges(level + 1, d);
    }

    stride = 1;
    for (int d = D - 1; d >= 0; d--) {
      if (global_data_idx[d] % 2 != 0 &&
          global_data_idx[d] != *level_ranges(level + 1, d) - 1) {
        coarse_level_offset = 0;
      }
      if (global_data_idx[d]) {
        coarse_level_offset += ((global_data_idx[d] - 1) / 2 + 1) * stride;
      }
      stride *= (*level_ranges(level + 1, d)) / 2 + 1;
    }

    if (level == 0)
      coarse_level_offset = 0;

    SIZE level_offset = curr_thread_offset - coarse_level_offset;
    return level_offset;
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
    level = 0;
    // Old appraoch
    // for (int d = D - 1; d >= 0; d--) {
    //   l_bit[d] = 0l;
    //   for (SIZE l = 0; l < l_target + 1; l++) {
    //     long long unsigned int bit = (idx[d] >= *level_ranges(l, d)) &&
    //                                  (idx[d] < *level_ranges(l + 1, d));
    //     l_bit[d] += bit << l;
    //   }
    //   level =
    //       Math<DeviceType>::Max((int)level,
    //       Math<DeviceType>::ffsll(l_bit[d]));
    // }
    // level = level - 1;

    for (int d = D - 1; d >= 0; d--) {
      l_bit[d] = 0l;
      for (SIZE l = 0; l < l_target + 1; l++) {
        if (idx[d] >= *level_ranges(l, d) && idx[d] < *level_ranges(l + 1, d)) {
          l_bit[d] = l;
        }
      }
      level = Math<DeviceType>::Max(level, l_bit[d]);
    }

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
        quantized_data = copysign(0.5 + fabs(t * quantizer * volume), t);
        if (prep_huffman) {
          quantized_data += dict_size / 2;
          if (quantized_data >= 0 && quantized_data < dict_size) {
            // do nothing
          } else {
            LENGTH outlier_write_offset =
                Atomic<LENGTH, AtomicGlobalMemory, AtomicDeviceScope,
                       DeviceType>::Add(outlier_count((IDX)0), (LENGTH)1);

            LENGTH outlier_idx = 0;
            if (!level_linearize) {
              // calculate the outlier index in the non-level linearized order
              LENGTH curr_stride = 1;
              for (int d = D - 1; d >= 0; d--) {
                outlier_idx += idx[d] * curr_stride;
                curr_stride *= v.shape(d);
              }
            } else {
              // calculate the outlier index in the level linearized order
              SIZE level_offset = calc_level_offset();
              // Assume we put it in quantized_linearized_v and calculate its
              // offset
              outlier_idx = quantized_linearized_v[level](level_offset) -
                            quantized_v.data();
            }
            // Avoid out of range error
            // If we have too much outlier than our allocation
            // we return the true outlier_count and do quanziation again
            if (outlier_write_offset < outlier_indexes.shape(0)) {
              *outlier_indexes(outlier_write_offset) = outlier_idx;
              *outliers(outlier_write_offset) = quantized_data;
            }
            quantized_data = 0;
          }
        }
        if (!level_linearize) {
          // store quantized value in non-level linearized position
          quantized_v[idx] = quantized_data;
        } else {
          // store quantized value in level linearized position
          SIZE level_offset = calc_level_offset();
          *(quantized_linearized_v[level](level_offset)) = quantized_data;
        }
      } else if constexpr (OP == MGARDX_DEQUANTIZE) {
        if (!level_linearize) {
          // read quantized value in non-level linearized position
          quantized_data = quantized_v[idx];
        } else {
          // read quantized value in level linearized position
          SIZE level_offset = calc_level_offset();
          quantized_data = *(quantized_linearized_v[level](level_offset));
        }
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
  SubArray<D, QUANTIZED_INT, DeviceType> quantized_v;
  SubArray<1, QUANTIZED_INT, DeviceType> *quantized_linearized_v;
  bool prep_huffman;
  bool calc_vol;
  bool level_linearize;
  SIZE dict_size;
  SubArray<1, SIZE, DeviceType> shape;
  SubArray<1, LENGTH, DeviceType> outlier_count;
  SubArray<1, LENGTH, DeviceType> outlier_indexes;
  SubArray<1, QUANTIZED_INT, DeviceType> outliers;

  T *volumes_0;
  T *volumes_1;
  T *volumes_2;
  T *volumes_3_plus;

  SIZE idx[D];  // thread global idx
  SIZE idx0[D]; // block global idx

  int level;
  int l_bit[D];
};

template <DIM D, typename T, typename DeviceType>
class OutlierRestoreFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT OutlierRestoreFunctor() {}
  MGARDX_CONT
  OutlierRestoreFunctor(SubArray<D, QUANTIZED_INT, DeviceType> quantized_v,
                        LENGTH outlier_count,
                        SubArray<1, LENGTH, DeviceType> outlier_indexes,
                        SubArray<1, QUANTIZED_INT, DeviceType> outliers)
      : quantized_v(quantized_v), outlier_count(outlier_count),
        outlier_indexes(outlier_indexes), outliers(outliers) {
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
      LENGTH linerized_idx = *outlier_indexes(gloablId);
      QUANTIZED_INT outliter = *outliers(gloablId);
      *quantized_v(linerized_idx) = outliter;
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  IDX threadId, blockId, gloablId;
  SubArray<D, QUANTIZED_INT, DeviceType> quantized_v;
  LENGTH outlier_count;
  SubArray<1, LENGTH, DeviceType> outlier_indexes;
  SubArray<1, QUANTIZED_INT, DeviceType> outliers;
};

template <DIM D, typename T, OPTION OP, typename DeviceType>
class LevelwiseLinearQuantizerKernel : public Kernel {
public:
  constexpr static DIM NumDim = D;
  using DataType = T;
  constexpr static std::string_view Name = "lwqzk";
  MGARDX_CONT
  LevelwiseLinearQuantizerKernel(
      SubArray<2, SIZE, DeviceType> level_ranges, SIZE l_target,
      SubArray<1, T, DeviceType> quantizers,
      SubArray<3, T, DeviceType> level_volumes, T s, SIZE dict_size,
      SubArray<D, T, DeviceType> v,
      SubArray<D, QUANTIZED_INT, DeviceType> quantized_v, bool prep_huffman,
      bool level_linearize,
      SubArray<1, QUANTIZED_INT, DeviceType> *quantized_linearized_v,
      SubArray<1, LENGTH, DeviceType> outlier_count,
      SubArray<1, LENGTH, DeviceType> outlier_indexes,
      SubArray<1, QUANTIZED_INT, DeviceType> outliers)
      : level_ranges(level_ranges), l_target(l_target), quantizers(quantizers),
        level_volumes(level_volumes), s(s), dict_size(dict_size), v(v),
        quantized_v(quantized_v), prep_huffman(prep_huffman),
        level_linearize(level_linearize),
        quantized_linearized_v(quantized_linearized_v),
        outlier_count(outlier_count), outlier_indexes(outlier_indexes),
        outliers(outliers) {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT
      Task<LevelwiseLinearQuantizerNDFunctor<D, T, R, C, F, OP, DeviceType>>
      GenTask(int queue_idx) {
    using FunctorType =
        LevelwiseLinearQuantizerNDFunctor<D, T, R, C, F, OP, DeviceType>;

    bool calc_vol =
        s != std::numeric_limits<T>::infinity(); // m.ntype == norm_type::L_2;
    FunctorType functor(level_ranges, l_target, quantizers, level_volumes, v,
                        quantized_v, quantized_linearized_v, prep_huffman,
                        calc_vol, level_linearize, dict_size, outlier_count,
                        outlier_indexes, outliers);

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
                std::string(Name));
  }

private:
  SubArray<2, SIZE, DeviceType> level_ranges;
  SIZE l_target;
  SubArray<1, T, DeviceType> quantizers;
  SubArray<3, T, DeviceType> level_volumes;
  T s;
  SubArray<D, T, DeviceType> v;
  SubArray<D, QUANTIZED_INT, DeviceType> quantized_v;
  bool prep_huffman;
  bool level_linearize;
  SubArray<1, QUANTIZED_INT, DeviceType> *quantized_linearized_v;
  SIZE dict_size;
  SubArray<1, SIZE, DeviceType> shape;
  SubArray<1, LENGTH, DeviceType> outlier_count;
  SubArray<1, LENGTH, DeviceType> outlier_indexes;
  SubArray<1, QUANTIZED_INT, DeviceType> outliers;
};

template <DIM D, typename T, typename DeviceType>
class OutlierRestoreKernel : public Kernel {
public:
  // 1D parallelization
  constexpr static DIM NumDim = 1;
  using DataType = T;
  constexpr static std::string_view Name = "ork";
  constexpr static bool EnableAutoTuning() { return false; }
  MGARDX_CONT
  OutlierRestoreKernel(SubArray<D, QUANTIZED_INT, DeviceType> quantized_v,
                       LENGTH outlier_count,
                       SubArray<1, LENGTH, DeviceType> outlier_indexes,
                       SubArray<1, QUANTIZED_INT, DeviceType> outliers)
      : quantized_v(quantized_v), outlier_count(outlier_count),
        outlier_indexes(outlier_indexes), outliers(outliers) {}

  MGARDX_CONT Task<OutlierRestoreFunctor<D, T, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = OutlierRestoreFunctor<D, T, DeviceType>;
    FunctorType functor(quantized_v, outlier_count, outlier_indexes, outliers);
    SIZE total_thread_z = 1;
    SIZE total_thread_y = 1;
    SIZE total_thread_x = outlier_count;
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = 256;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<D, QUANTIZED_INT, DeviceType> quantized_v;
  LENGTH outlier_count;
  SubArray<1, LENGTH, DeviceType> outlier_indexes;
  SubArray<1, QUANTIZED_INT, DeviceType> outliers;
};

template <DIM D, typename T, typename DeviceType>
void LinearQuanziation(
    Hierarchy<D, T, DeviceType> &hierarchy, Array<D, T, DeviceType> &in_array,
    Config config, enum error_bound_type type, T tol, T s, T norm,
    CompressionLowLevelWorkspace<D, T, DeviceType> &workspace, int queue_idx) {
  SubArray in_subarray(in_array);
  bool prep_huffman =
      config.lossless != lossless_type::CPU_Lossless; // always do Huffman
  SIZE total_elems = hierarchy.total_num_elems();
  SubArray<2, SIZE, DeviceType> level_ranges_subarray(hierarchy.level_ranges(),
                                                      true);
  SubArray<3, T, DeviceType> level_volumes_subarray(
      hierarchy.level_volumes(false));

  T *quantizers = new T[hierarchy.l_target() + 1];
  calc_quantizers<D, T>(total_elems, quantizers, type, tol, s, norm,
                        hierarchy.l_target(), config.decomposition, true);
  MemoryManager<DeviceType>::Copy1D(workspace.quantizers_subarray.data(),
                                    quantizers, hierarchy.l_target() + 1,
                                    queue_idx);
  LENGTH zero = 0;
  MemoryManager<DeviceType>::Copy1D(workspace.outlier_count_subarray.data(),
                                    &zero, 1, queue_idx);

  SubArray<1, QUANTIZED_INT, DeviceType> *quantized_linearized_v_host = nullptr;
  SubArray<1, QUANTIZED_INT, DeviceType> *quantized_linearized_v = nullptr;

  if (config.reorder) { // only if we need linerization
    quantized_linearized_v_host =
        new SubArray<1, QUANTIZED_INT, DeviceType>[hierarchy.l_target() + 1];
    SIZE *ranges_h = level_ranges_subarray.dataHost();
    SIZE last_level_size = 0;
    for (SIZE l = 0; l < hierarchy.l_target() + 1; l++) {
      SIZE level_size = 1;
      for (DIM d = 0; d < D; d++) {
        level_size *= ranges_h[(l + 1) * D + d];
      }
      quantized_linearized_v_host[l] = SubArray<1, QUANTIZED_INT, DeviceType>(
          {level_size - last_level_size},
          workspace.quantized_subarray(last_level_size));
      last_level_size = level_size;
    }

    MemoryManager<DeviceType>::Malloc1D(quantized_linearized_v,
                                        hierarchy.l_target() + 1, queue_idx);
    DeviceRuntime<DeviceType>::SyncDevice();
    MemoryManager<DeviceType>::Copy1D(quantized_linearized_v,
                                      quantized_linearized_v_host,
                                      hierarchy.l_target() + 1, queue_idx);
    DeviceRuntime<DeviceType>::SyncDevice();
  }

  Timer timer;
  if (log::level & log::TIME)
    timer.start();

  bool done_quantization = false;
  while (!done_quantization) {
    DeviceLauncher<DeviceType>::Execute(
        LevelwiseLinearQuantizerKernel<D, T, MGARDX_QUANTIZE, DeviceType>(
            level_ranges_subarray, hierarchy.l_target(),
            workspace.quantizers_subarray, level_volumes_subarray, s,
            config.huff_dict_size, in_subarray, workspace.quantized_subarray,
            prep_huffman, config.reorder, quantized_linearized_v,
            workspace.outlier_count_subarray, workspace.outlier_idx_subarray,
            workspace.outliers_subarray),
        queue_idx);

    MemoryManager<DeviceType>::Copy1D(&workspace.outlier_count,
                                      workspace.outlier_count_subarray.data(),
                                      1, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    if (workspace.outlier_count <= workspace.outliers_subarray.shape(0)) {
      // outlier buffer has sufficient size
      done_quantization = true;
      if (log::level & log::TIME) {
        timer.end();
        timer.print("Quantization");
        timer.clear();
      }
      log::info(
          "Outlier ratio: " + std::to_string(workspace.outlier_count) + "/" +
          std::to_string(total_elems) + " (" +
          std::to_string((double)100 * workspace.outlier_count / total_elems) +
          "%)");
    } else {
      log::info("Not enough workspace for outliers. Re-allocating to " +
                std::to_string(workspace.outlier_count));
      workspace.outlier_idx_array =
          Array<1, LENGTH, DeviceType>({(SIZE)workspace.outlier_count});
      workspace.outliers_array =
          Array<1, QUANTIZED_INT, DeviceType>({(SIZE)workspace.outlier_count});
      workspace.outlier_idx_subarray = SubArray(workspace.outlier_idx_array);
      workspace.outliers_subarray = SubArray(workspace.outliers_array);
      workspace.outlier_count_array.memset(0);
    }
  }
  if (config.reorder) {
    delete[] quantized_linearized_v_host;
    MemoryManager<DeviceType>::Free(quantized_linearized_v);
  }

  delete[] quantizers;
}

template <DIM D, typename T, typename DeviceType>
void LinearDequanziation(
    Hierarchy<D, T, DeviceType> &hierarchy, Array<D, T, DeviceType> &out_array,
    Config config, enum error_bound_type type, T tol, T s, T norm,
    CompressionLowLevelWorkspace<D, T, DeviceType> &workspace, int queue_idx) {
  SIZE total_elems = hierarchy.total_num_elems();
  out_array.resize(hierarchy.level_shape(hierarchy.l_target()));
  SubArray<D, T, DeviceType> out_subarray(out_array);
  MemoryManager<DeviceType>::Copy1D(workspace.outlier_count_subarray.data(),
                                    &workspace.outlier_count, 1, queue_idx);
  SubArray<2, SIZE, DeviceType> level_ranges_subarray(hierarchy.level_ranges(),
                                                      true);
  SubArray<3, T, DeviceType> level_volumes_subarray(
      hierarchy.level_volumes(true));

  bool prep_huffman = config.lossless != lossless_type::CPU_Lossless;

  T *quantizers = new T[hierarchy.l_target() + 1];
  calc_quantizers<D, T>(total_elems, quantizers, type, tol, s, norm,
                        hierarchy.l_target(), config.decomposition, false);
  MemoryManager<DeviceType>::Copy1D(workspace.quantizers_subarray.data(),
                                    quantizers, hierarchy.l_target() + 1,
                                    queue_idx);
  DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

  SubArray<1, QUANTIZED_INT, DeviceType> *quantized_linearized_v_host = nullptr;
  SubArray<1, QUANTIZED_INT, DeviceType> *quantized_linearized_v = nullptr;
  if (config.reorder) { // only if we need linerization
    quantized_linearized_v_host =
        new SubArray<1, QUANTIZED_INT, DeviceType>[hierarchy.l_target() + 1];
    SIZE *ranges_h = level_ranges_subarray.dataHost();
    SIZE last_level_size = 0;
    for (SIZE l = 0; l < hierarchy.l_target() + 1; l++) {
      SIZE level_size = 1;
      for (DIM d = 0; d < D; d++) {
        level_size *= ranges_h[(l + 1) * D + d];
      }
      quantized_linearized_v_host[l] = SubArray<1, QUANTIZED_INT, DeviceType>(
          {level_size - last_level_size},
          workspace.quantized_subarray(last_level_size));
      last_level_size = level_size;
    }

    MemoryManager<DeviceType>::Malloc1D(quantized_linearized_v,
                                        hierarchy.l_target() + 1, queue_idx);
    DeviceRuntime<DeviceType>::SyncDevice();
    MemoryManager<DeviceType>::Copy1D(quantized_linearized_v,
                                      quantized_linearized_v_host,
                                      hierarchy.l_target() + 1, queue_idx);
    DeviceRuntime<DeviceType>::SyncDevice();
  }

  Timer timer;
  if (log::level & log::TIME)
    timer.start();

  if (prep_huffman && workspace.outlier_count) {
    DeviceLauncher<DeviceType>::Execute(
        OutlierRestoreKernel<D, T, DeviceType>(
            workspace.quantized_subarray, workspace.outlier_count,
            workspace.outlier_idx_subarray, workspace.outliers_subarray),
        queue_idx);
  }

  DeviceLauncher<DeviceType>::Execute(
      LevelwiseLinearQuantizerKernel<D, T, MGARDX_DEQUANTIZE, DeviceType>(
          level_ranges_subarray, hierarchy.l_target(),
          workspace.quantizers_subarray, level_volumes_subarray, s,
          config.huff_dict_size, out_array, workspace.quantized_subarray,
          prep_huffman, config.reorder, quantized_linearized_v,
          workspace.outlier_count_subarray, workspace.outlier_idx_subarray,
          workspace.outliers_subarray),
      queue_idx);

  DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
  if (log::level & log::TIME) {
    timer.end();
    timer.print("Dequantization");
    timer.clear();
  }

  if (config.reorder) {
    delete[] quantized_linearized_v_host;
    MemoryManager<DeviceType>::Free(quantized_linearized_v);
  }
  delete[] quantizers;
}
} // namespace mgard_x

#endif