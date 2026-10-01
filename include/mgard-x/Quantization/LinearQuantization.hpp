/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_LINEAR_QUANTIZATION_TEMPLATE
#define MGARD_X_LINEAR_QUANTIZATION_TEMPLATE

#include "../RuntimeX/RuntimeX.h"
#include "QuantizationInterface.hpp"

namespace mgard_x {

#define MGARDX_QUANTIZE 1
#define MGARDX_DEQUANTIZE 2

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, OPTION OP,
          typename DeviceType>
class LevelwiseLinearQuantizerNDFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT LevelwiseLinearQuantizerNDFunctor() {}
  MGARDX_CONT LevelwiseLinearQuantizerNDFunctor(
      SubArray<2, SIZE, DeviceType> level_ranges,
      SubArray<2, int, DeviceType> level_marks, SIZE l_target,
      SubArray<1, T, DeviceType> quantizers,
      SubArray<1, T, DeviceType> level_volumes,
      SubArray<2, SIZE, DeviceType> level_volume_offsets, bool calc_vol,
      SubArray<D, T, DeviceType> v,
      SubArray<D, QUANTIZED_INT, DeviceType> quantized_v, bool prep_huffman,
      SIZE dict_size)
      : level_ranges(level_ranges), level_marks(level_marks),
        l_target(l_target), quantizers(quantizers),
        level_volumes(level_volumes),
        level_volume_offsets(level_volume_offsets), calc_vol(calc_vol), v(v),
        quantized_v(quantized_v), prep_huffman(prep_huffman),
        dict_size(dict_size) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    // determine global idx
    SIZE firstD = div_roundup(v.shape(D - 1), F);

    SIZE bidx = FunctorBase<DeviceType>::GetBlockIdX();
    idx[D - 1] = (bidx % firstD) * F + FunctorBase<DeviceType>::GetThreadIdX();

    bidx /= firstD;
    if (D >= 2) {
      idx[D - 2] = FunctorBase<DeviceType>::GetBlockIdY() *
                       FunctorBase<DeviceType>::GetBlockDimY() +
                   FunctorBase<DeviceType>::GetThreadIdY();
    }
    if (D >= 3) {
      idx[D - 3] = FunctorBase<DeviceType>::GetBlockIdZ() *
                       FunctorBase<DeviceType>::GetBlockDimZ() +
                   FunctorBase<DeviceType>::GetThreadIdZ();
    }

    for (int d = D - 4; d >= 0; d--) {
      idx[d] = bidx % v.shape(d);
      // idx0[d] = idx[d];
      bidx /= v.shape(d);
    }

    level = 0;

    bool in_range = true;
    for (int d = D - 1; d >= 0; d--) {
      if (idx[d] >= v.shape(d))
        in_range = false;
    }

    if (level >= 0 && level <= l_target && in_range) {
      T t = v[idx];
      T volume = 1;
      if (calc_vol) {
        // Determine level
        for (int d = D - 1; d >= 0; d--) {
          level = Math<DeviceType>::Max(level, *level_marks(d, idx[d]));
        }

        // Determine volume
        for (int d = D - 1; d >= 0; d--) {
          volume *= *level_volumes(*level_volume_offsets(level, d) + idx[d]);
        }
        if (sizeof(T) == sizeof(double))
          volume = sqrt(volume);
        else if (sizeof(T) == sizeof(float))
          volume = sqrtf(volume);
      }

      T quantizer = *quantizers(level);
      QUANTIZED_INT quantized_data;

      if constexpr (OP == MGARDX_QUANTIZE) {
        if constexpr (sizeof(T) == sizeof(double)) {
          quantized_data = copysign((T)0.5 + fabs(t * quantizer * volume), t);
        } else if constexpr (sizeof(T) == sizeof(float)) {
          quantized_data = copysign((T)0.5 + fabsf(t * quantizer * volume), t);
        }
        // Shift quantized values into the non-negative dictionary range the
        // Huffman coder expects. Folding the shift here makes it free (the
        // element is already being written) and removes a standalone full-array
        // pass. Gated by prep_huffman so an alternative lossless backend that
        // does not need outlier handling keeps the raw signed values. The value
        // stays signed (QUANTIZED_INT); out-of-range entries are detected later
        // by the outlier separation pass.
        if (prep_huffman) {
          quantized_data += dict_size / 2;
        }
        quantized_v[idx] = quantized_data;
      } else if constexpr (OP == MGARDX_DEQUANTIZE) {
        quantized_data = quantized_v[idx];
        if (prep_huffman) {
          quantized_data -= dict_size / 2;
        }
        v[idx] = (quantizer * volume) * (T)quantized_data;
      }
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    return size;
  }

private:
  IDX threadId;
  SubArray<2, SIZE, DeviceType> level_ranges;
  SubArray<2, int, DeviceType> level_marks;
  SIZE l_target;
  SubArray<1, T, DeviceType> quantizers;
  SubArray<1, T, DeviceType> level_volumes;
  SubArray<2, SIZE, DeviceType> level_volume_offsets;
  SubArray<D, T, DeviceType> v;
  SubArray<D, QUANTIZED_INT, DeviceType> quantized_v;
  bool calc_vol;
  bool prep_huffman;
  SIZE dict_size;
  SubArray<1, SIZE, DeviceType> shape;

  SIZE idx[D];  // thread global idx
  SIZE idx0[D]; // block global idx

  int level;
};

template <DIM D, typename T, OPTION OP, typename DeviceType>
class LevelwiseLinearQuantizerKernel : public Kernel {
public:
  constexpr static DIM NumDim = D;
  using DataType = T;
  constexpr static std::string_view Name = "lwqzk";
  MGARDX_CONT
  LevelwiseLinearQuantizerKernel(
      SubArray<2, SIZE, DeviceType> level_ranges,
      SubArray<2, int, DeviceType> level_marks, SIZE l_target,
      SubArray<1, T, DeviceType> quantizers,
      SubArray<1, T, DeviceType> level_volumes,
      SubArray<2, SIZE, DeviceType> level_volume_offsets, bool calc_vol,
      SubArray<D, T, DeviceType> v,
      SubArray<D, QUANTIZED_INT, DeviceType> quantized_v, bool prep_huffman,
      SIZE dict_size)
      : level_ranges(level_ranges), level_marks(level_marks),
        l_target(l_target), quantizers(quantizers),
        level_volumes(level_volumes),
        level_volume_offsets(level_volume_offsets), calc_vol(calc_vol), v(v),
        quantized_v(quantized_v), prep_huffman(prep_huffman),
        dict_size(dict_size) {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT
      Task<LevelwiseLinearQuantizerNDFunctor<D, T, R, C, F, OP, DeviceType>>
      GenTask(int queue_idx) {
    using FunctorType =
        LevelwiseLinearQuantizerNDFunctor<D, T, R, C, F, OP, DeviceType>;

    FunctorType functor(level_ranges, level_marks, l_target, quantizers,
                        level_volumes, level_volume_offsets, calc_vol, v,
                        quantized_v, prep_huffman, dict_size);

    SIZE total_thread_z = v.shape(D - 3);
    SIZE total_thread_y = v.shape(D - 2);
    SIZE total_thread_x = v.shape(D - 1);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((double)total_thread_z / tbz);
    gridy = ceil((double)total_thread_y / tby);
    gridx = ceil((double)total_thread_x / tbx);
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
  SubArray<2, int, DeviceType> level_marks;
  SIZE l_target;
  SubArray<1, T, DeviceType> quantizers;
  SubArray<1, T, DeviceType> level_volumes;
  SubArray<2, SIZE, DeviceType> level_volume_offsets;
  bool calc_vol;
  SubArray<D, T, DeviceType> v;
  SubArray<D, QUANTIZED_INT, DeviceType> quantized_v;
  bool prep_huffman;
  SIZE dict_size;
  bool level_linearize;
  SubArray<1, SIZE, DeviceType> shape;
};

template <DIM D, typename T, typename Q, typename DeviceType>
class LinearQuantizer : public QuantizationInterface<D, T, Q, DeviceType> {
public:
  LinearQuantizer() : initialized(false) {}

  LinearQuantizer(Hierarchy<D, T, DeviceType> &hierarchy, Config config)
      : initialized(true), hierarchy(&hierarchy), config(config) {
    quantizers_array = Array<1, T, DeviceType>({hierarchy.l_target() + 1});
  }

  void Adapt(Hierarchy<D, T, DeviceType> &hierarchy, Config config,
             int queue_idx) {
    this->initialized = true;
    this->hierarchy = &hierarchy;
    this->config = config;
    quantizers_array.resize({hierarchy.l_target() + 1}, queue_idx);
  }

  void CalcQuantizers(size_t dof, T *quantizers, enum error_bound_type type,
                      T tol, T s, T norm, SIZE l_target,
                      enum decomposition_type decomposition,
                      bool orthogonal_projection, bool reciprocal) {

    double abs_tol = tol;
    if (type == error_bound_type::REL) {
      abs_tol *= norm;
    }
    abs_tol *= 2;
    if (s == std::numeric_limits<T>::infinity()) {

      // printf("quantizers: ");
      for (int l = 0; l < l_target + 1; l++) {
        if (decomposition == decomposition_type::MultiDim ||
            decomposition == decomposition_type::Hybrid) {
          if (!orthogonal_projection) {
            // Hierarchical basis (no mass-matrix correction): reconstruction is
            // multilinear prolongation, a partition of unity, so a level-l
            // quantization error propagates to any node with amplification
            // <= 1. The max error is then bounded by sum over (l_target+1)
            // levels of (step/2), i.e. the (1+3^D) orthogonal amplification
            // drops out.
            quantizers[l] = (abs_tol) / (l_target + 1);
          } else {
            // ben
            quantizers[l] = (abs_tol) / ((l_target + 1) * (1 + std::pow(3, D)));
          }
          // xin
          // quantizers[l] = (tol) / ((l_target + 1) * (1 + 3 * std::sqrt(3) /
          // 4));
          // quantizers[l] = (0.5 * abs_tol) / std::pow(2, l_target - 1);

        } else if (decomposition == decomposition_type::SingleDim) {
          // ken
          quantizers[l] =
              (abs_tol) / ((l_target + 1) * D * (1 + std::pow(3, 1)));
        }
        if (reciprocal)
          quantizers[l] = 1.0f / quantizers[l];
      }

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
      for (int l = 0; l < l_target + 1; l++) {
        quantizers[l] = (abs_tol) / (std::exp2(s * l) * std::sqrt(dof));
        if (reciprocal)
          quantizers[l] = 1.0f / quantizers[l];
      }
    }
  }

  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape) {
    Hierarchy<D, T, DeviceType> hierarchy;
    hierarchy.EstimateMemoryFootprint(shape);
    size_t size = 0;
    size += sizeof(T) * (hierarchy.l_target() + 1);
    return size;
  }

  void Quantize(SubArray<D, T, DeviceType> original_data,
                enum error_bound_type ebtype, T tol, T s, T norm,
                SubArray<D, Q, DeviceType> quantized_data, int queue_idx) {}

  void Dequantize(SubArray<D, T, DeviceType> original_data,
                  enum error_bound_type ebtype, T tol, T s, T norm,
                  SubArray<D, Q, DeviceType> quantized_data, int queue_idx) {}

  template <typename LosslessCompressorType>
  void Quantize(SubArray<D, T, DeviceType> original_data,
                enum error_bound_type ebtype, T tol, T s, T norm,
                SubArray<D, Q, DeviceType> quantized_data,
                LosslessCompressorType &lossless, int queue_idx,
                bool orthogonal_projection = true) {

    // Toggle controlled from outside via the configured lossless backend: when
    // a backend needs a non-negative Huffman dictionary we fold the dictionary
    // shift into quantization; alternative backends that handle signed values
    // directly leave the quantized data untouched (no shift / no outliers).
    bool prep_huffman = config.lossless != lossless_type::CPU_Lossless &&
                        config.lossless != lossless_type::BlockDelta &&
                        config.lossless != lossless_type::LZ4;
    SIZE total_elems = hierarchy->total_num_elems();
    SubArray<2, SIZE, DeviceType> level_ranges_subarray(
        hierarchy->level_ranges());
    SubArray<2, int, DeviceType> level_marks_subarray(hierarchy->level_marks());
    SubArray<1, T, DeviceType> level_volumes_subarray(
        hierarchy->level_volumes(false));
    SubArray<2, SIZE, DeviceType> level_volume_offsets_subarray(
        hierarchy->level_volume_offsets());
    SubArray<1, T, DeviceType> quantizers_subarray(quantizers_array);
    T *quantizers = new T[hierarchy->l_target() + 1];
    CalcQuantizers(total_elems, quantizers, ebtype, tol, s, norm,
                   hierarchy->l_target(), config.decomposition,
                   orthogonal_projection, true);
    MemoryManager<DeviceType>::Copy1D(quantizers_subarray.data(), quantizers,
                                      hierarchy->l_target() + 1, queue_idx);

    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    bool calc_vol =
        s != std::numeric_limits<T>::infinity(); // m.ntype == norm_type::L_2;
    DeviceLauncher<DeviceType>::Execute(
        LevelwiseLinearQuantizerKernel<D, T, MGARDX_QUANTIZE, DeviceType>(
            level_ranges_subarray, level_marks_subarray, hierarchy->l_target(),
            quantizers_subarray, level_volumes_subarray,
            level_volume_offsets_subarray, calc_vol, original_data,
            quantized_data, prep_huffman, config.huff_dict_size),
        queue_idx);

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Quantization", hierarchy->total_num_elems() * sizeof(T));
      timer.clear();
    }

    delete[] quantizers;
  }

  template <typename LosslessCompressorType>
  void Dequantize(SubArray<D, T, DeviceType> original_data,
                  enum error_bound_type ebtype, T tol, T s, T norm,
                  SubArray<D, Q, DeviceType> quantized_data,
                  LosslessCompressorType &lossless_compressor, int queue_idx,
                  bool orthogonal_projection = true) {

    SIZE total_elems = hierarchy->total_num_elems();
    SubArray<2, SIZE, DeviceType> level_ranges_subarray(
        hierarchy->level_ranges());
    SubArray<2, int, DeviceType> level_marks_subarray(hierarchy->level_marks());
    SubArray<1, T, DeviceType> level_volumes_subarray(
        hierarchy->level_volumes(true));
    SubArray<2, SIZE, DeviceType> level_volume_offsets_subarray(
        hierarchy->level_volume_offsets());

    // Must mirror the toggle used during Quantize so the dictionary shift is
    // undone exactly when it was applied.
    bool prep_huffman = config.lossless != lossless_type::CPU_Lossless &&
                        config.lossless != lossless_type::BlockDelta &&
                        config.lossless != lossless_type::LZ4;

    SubArray<1, T, DeviceType> quantizers_subarray(quantizers_array);
    T *quantizers = new T[hierarchy->l_target() + 1];
    CalcQuantizers(total_elems, quantizers, ebtype, tol, s, norm,
                   hierarchy->l_target(), config.decomposition,
                   orthogonal_projection, false);
    MemoryManager<DeviceType>::Copy1D(quantizers_subarray.data(), quantizers,
                                      hierarchy->l_target() + 1, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    bool calc_vol =
        s != std::numeric_limits<T>::infinity(); // m.ntype == norm_type::L_2;
    DeviceLauncher<DeviceType>::Execute(
        LevelwiseLinearQuantizerKernel<D, T, MGARDX_DEQUANTIZE, DeviceType>(
            level_ranges_subarray, level_marks_subarray, hierarchy->l_target(),
            quantizers_subarray, level_volumes_subarray,
            level_volume_offsets_subarray, calc_vol, original_data,
            quantized_data, prep_huffman, config.huff_dict_size),
        queue_idx);

    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    if (log::level & log::TIME) {
      timer.end();
      timer.print("Dequantization", hierarchy->total_num_elems() * sizeof(T));
      timer.clear();
    }

    delete[] quantizers;
  }

  bool initialized;
  Hierarchy<D, T, DeviceType> *hierarchy;
  Config config;
  Array<1, T, DeviceType> quantizers_array;
};

} // namespace mgard_x

#endif