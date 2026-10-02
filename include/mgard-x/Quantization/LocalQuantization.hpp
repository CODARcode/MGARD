#ifndef MGARD_X_LOCAL_QUANTIZATION_TEMPLATE
#define MGARD_X_LOCAL_QUANTIZATION_TEMPLATE

#include "../Hierarchy/HybridHierarchy.h"
#include "../RuntimeX/RuntimeX.h"

#include "QuantizationInterface.hpp"
#include <algorithm>
#include <cmath>

namespace mgard_x {

#define MGARDX_QUANTIZE 1
#define MGARDX_DEQUANTIZE 2

// Non-ROI Version
template <typename T, typename Q, OPTION OP, typename DeviceType>
class QuantizeLocalLevelFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT QuantizeLocalLevelFunctor() {}
  MGARDX_CONT QuantizeLocalLevelFunctor(T quantizer,
                                        SubArray<1, T, DeviceType> v,
                                        SubArray<1, Q, DeviceType> quantized_v,
                                        bool prep_huffman, SIZE dict_size)
      : quantizer(quantizer), v(v), quantized_v(quantized_v),
        prep_huffman(prep_huffman), dict_size(dict_size) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    idx = FunctorBase<DeviceType>::GetBlockIdX() *
              FunctorBase<DeviceType>::GetBlockDimX() +
          FunctorBase<DeviceType>::GetThreadIdX();

    if (idx < v.shape(0)) {
      T t = *v(idx);
      Q quantized_data;
      T volume = 1;

      if constexpr (OP == MGARDX_QUANTIZE) {
        if constexpr (sizeof(T) == sizeof(double)) {
          quantized_data = copysign((T)0.5 + fabs(t * quantizer * volume), t);
        } else if constexpr (sizeof(T) == sizeof(float)) {
          quantized_data = copysign((T)0.5 + fabsf(t * quantizer * volume), t);
        }
        // Fold the Huffman dictionary shift into quantization (mirrors
        // LevelwiseLinearQuantizerNDFunctor): the lossless stage expects
        // values in the non-negative dictionary range and separates
        // out-of-range entries as outliers.
        if (prep_huffman) {
          quantized_data += dict_size / 2;
        }
        *quantized_v(idx) = quantized_data;
      } else if constexpr (OP == MGARDX_DEQUANTIZE) {
        quantized_data = *quantized_v(idx);
        if (prep_huffman) {
          quantized_data -= dict_size / 2;
        }
        *v(idx) = (quantizer * volume) * (T)quantized_data;
      }
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SIZE idx;
  T quantizer;
  SubArray<1, T, DeviceType> v;
  SubArray<1, Q, DeviceType> quantized_v;
  bool prep_huffman;
  SIZE dict_size;
};

template <typename T, typename Q, OPTION OP, typename DeviceType>
class QuantizeLocalLevelKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "lvl_qk";

  MGARDX_CONT
  QuantizeLocalLevelKernel(T quantizer, SubArray<1, T, DeviceType> v,
                           SubArray<1, Q, DeviceType> quantized_v,
                           bool prep_huffman, SIZE dict_size)
      : quantizer(quantizer), v(v), quantized_v(quantized_v),
        prep_huffman(prep_huffman), dict_size(dict_size) {}

  MGARDX_CONT Task<QuantizeLocalLevelFunctor<T, Q, OP, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = QuantizeLocalLevelFunctor<T, Q, OP, DeviceType>;
    FunctorType functor(quantizer, v, quantized_v, prep_huffman, dict_size);

    SIZE tbx = 256, tby = 1, tbz = 1;
    SIZE gridx = (v.shape(0) + tbx - 1) / tbx;
    SIZE gridy = 1, gridz = 1;

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, 0, queue_idx,
                std::string(Name));
  }

private:
  T quantizer;
  SubArray<1, T, DeviceType> v;
  SubArray<1, Q, DeviceType> quantized_v;
  bool prep_huffman;
  SIZE dict_size;
};

// ROI Version
//
// The coefficient array is a row-major sequence of block-local blocks, each
// contributing hybrid_local_coeff_per_block(D) coefficients, so a
// coefficient's block index -- and therefore its ROI quantizer -- is
// idx / that count. The count is dimension dependent (3 in 1D, 39 in 2D,
// 387 in 3D), hence the D template parameter.
template <DIM D, typename T, typename Q, OPTION OP, typename DeviceType>
class QuantizeLocalLevelROIFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT QuantizeLocalLevelROIFunctor() {}
  MGARDX_CONT QuantizeLocalLevelROIFunctor(
      SubArray<1, T, DeviceType> quantizers, SubArray<1, T, DeviceType> v,
      SubArray<1, Q, DeviceType> quantized_v, bool prep_huffman, SIZE dict_size)
      : quantizers(quantizers), v(v), quantized_v(quantized_v),
        prep_huffman(prep_huffman), dict_size(dict_size) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    idx = FunctorBase<DeviceType>::GetBlockIdX() *
              FunctorBase<DeviceType>::GetBlockDimX() +
          FunctorBase<DeviceType>::GetThreadIdX();

    if (idx < v.shape(0)) {
      SIZE block_idx = idx / kCoeffPerBlock;

      T quantizer = *quantizers(block_idx);

      T t = *v(idx);
      Q quantized_data;
      T volume = 1;

      if constexpr (OP == MGARDX_QUANTIZE) {
        if constexpr (sizeof(T) == sizeof(double)) {
          quantized_data = copysign((T)0.5 + fabs(t * quantizer * volume), t);
        } else if constexpr (sizeof(T) == sizeof(float)) {
          quantized_data = copysign((T)0.5 + fabsf(t * quantizer * volume), t);
        }
        if (prep_huffman) {
          quantized_data += dict_size / 2;
        }
        *quantized_v(idx) = quantized_data;
      } else if constexpr (OP == MGARDX_DEQUANTIZE) {
        quantized_data = *quantized_v(idx);
        if (prep_huffman) {
          quantized_data -= dict_size / 2;
        }
        *v(idx) = (quantizer * volume) * (T)quantized_data;
      }
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  static constexpr SIZE kCoeffPerBlock = (SIZE)hybrid_local_coeff_per_block(D);
  SIZE idx;
  SubArray<1, T, DeviceType> quantizers;
  SubArray<1, T, DeviceType> v;
  SubArray<1, Q, DeviceType> quantized_v;
  bool prep_huffman;
  SIZE dict_size;
};

template <DIM D, typename T, typename Q, OPTION OP, typename DeviceType>
class QuantizeLocalLevelROIKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "lvl_qk_roi";

  MGARDX_CONT
  QuantizeLocalLevelROIKernel(SubArray<1, T, DeviceType> quantizers,
                              SubArray<1, T, DeviceType> v,
                              SubArray<1, Q, DeviceType> quantized_v,
                              bool prep_huffman, SIZE dict_size)
      : quantizers(quantizers), v(v), quantized_v(quantized_v),
        prep_huffman(prep_huffman), dict_size(dict_size) {}

  MGARDX_CONT Task<QuantizeLocalLevelROIFunctor<D, T, Q, OP, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = QuantizeLocalLevelROIFunctor<D, T, Q, OP, DeviceType>;
    FunctorType functor(quantizers, v, quantized_v, prep_huffman, dict_size);

    SIZE tbx = 256, tby = 1, tbz = 1;
    SIZE gridx = (v.shape(0) + tbx - 1) / tbx;
    SIZE gridy = 1, gridz = 1;

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, 0, queue_idx,
                std::string(Name));
  }

private:
  SubArray<1, T, DeviceType> quantizers;
  SubArray<1, T, DeviceType> v;
  SubArray<1, Q, DeviceType> quantized_v;
  bool prep_huffman;
  SIZE dict_size;
};

// Computes per-block ROI quantizers directly on device from a device-resident
// tolerance map, avoiding a host-side loop plus a per-call H2D transfer of the
// result (the tolerance map itself is uploaded once, not on every call).
template <typename T, typename DeviceType>
class ComputeROIQuantizersFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT ComputeROIQuantizersFunctor() {}
  MGARDX_CONT
  ComputeROIQuantizersFunctor(SubArray<1, double, DeviceType> tolerance_map,
                              SIZE level_offset, SIZE num_blocks,
                              double norm_factor, double denom, bool reciprocal,
                              SubArray<1, T, DeviceType> quantizers)
      : tolerance_map(tolerance_map), level_offset(level_offset),
        num_blocks(num_blocks), norm_factor(norm_factor), denom(denom),
        reciprocal(reciprocal), quantizers(quantizers) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    idx = FunctorBase<DeviceType>::GetBlockIdX() *
              FunctorBase<DeviceType>::GetBlockDimX() +
          FunctorBase<DeviceType>::GetThreadIdX();

    if (idx < num_blocks) {
      double block_tol = *tolerance_map(level_offset + idx) * norm_factor * 2;
      double block_quantizer = block_tol / denom;
      *quantizers(idx) =
          reciprocal ? (T)(1.0 / block_quantizer) : (T)block_quantizer;
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SIZE idx;
  SubArray<1, double, DeviceType> tolerance_map;
  SIZE level_offset;
  SIZE num_blocks;
  double norm_factor;
  double denom;
  bool reciprocal;
  SubArray<1, T, DeviceType> quantizers;
};

template <typename T, typename DeviceType>
class ComputeROIQuantizersKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "roi_qcalc";

  MGARDX_CONT
  ComputeROIQuantizersKernel(SubArray<1, double, DeviceType> tolerance_map,
                             SIZE level_offset, SIZE num_blocks,
                             double norm_factor, double denom, bool reciprocal,
                             SubArray<1, T, DeviceType> quantizers)
      : tolerance_map(tolerance_map), level_offset(level_offset),
        num_blocks(num_blocks), norm_factor(norm_factor), denom(denom),
        reciprocal(reciprocal), quantizers(quantizers) {}

  MGARDX_CONT Task<ComputeROIQuantizersFunctor<T, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = ComputeROIQuantizersFunctor<T, DeviceType>;
    FunctorType functor(tolerance_map, level_offset, num_blocks, norm_factor,
                        denom, reciprocal, quantizers);

    SIZE tbx = 256, tby = 1, tbz = 1;
    SIZE gridx = (num_blocks + tbx - 1) / tbx;
    SIZE gridy = 1, gridz = 1;

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, 0, queue_idx,
                std::string(Name));
  }

private:
  SubArray<1, double, DeviceType> tolerance_map;
  SIZE level_offset;
  SIZE num_blocks;
  double norm_factor;
  double denom;
  bool reciprocal;
  SubArray<1, T, DeviceType> quantizers;
};

template <DIM D, typename T, typename Q, typename DeviceType>
class LocalQuantizer : public QuantizationInterface<D, T, Q, DeviceType> {
public:
  LocalQuantizer() : initialized(false) {}
  LocalQuantizer(HybridHierarchy<D, T, DeviceType> &hierarchy, Config config)
      : initialized(true), hierarchy(&hierarchy), config(config) {
    hierarchy.check_config(config);
    this->L = config.num_local_refactoring_level;
    this->M = config.num_global_refactoring_level;
    compute_local_ranges();
    prepare_layers();
  }

  // Add logic to determine if roi or not
  void Adapt(HybridHierarchy<D, T, DeviceType> &hierarchy, Config config,
             int queue_idx) {
    hierarchy.check_config(config);
    this->initialized = true;
    this->hierarchy = &hierarchy;
    this->config = config;
    this->L = config.num_local_refactoring_level;
    this->M = config.num_global_refactoring_level;
    compute_local_ranges();
    prepare_layers();
  }

  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape) {
    size_t size = 0;
    return size;
  }

  void SetOrthogonalProjection(bool enabled) {
    orthogonal_projection = enabled;
  }

  void compute_local_ranges() {
    coarse_shape = hierarchy->input_shape();

    fine_num_elems.clear();
    coarse_num_elems.clear();
    local_coeff_size.clear();
    fine_shapes.clear();
    coarse_shapes.clear();

    for (int l = 0; l < this->L; ++l) {
      SIZE last_level_size = 1, curr_level_size = 1;
      std::vector<SIZE> fine_shape = hierarchy->local_fine_shape(l);
      coarse_shape = hierarchy->local_coarse_shape(l);
      for (DIM d = 0; d < D; ++d) {
        last_level_size *= fine_shape[d];
        curr_level_size *= coarse_shape[d];
      }
      fine_num_elems.push_back(last_level_size);
      coarse_num_elems.push_back(curr_level_size);
      local_coeff_size.push_back(last_level_size - curr_level_size);
      fine_shapes.push_back(fine_shape);
      coarse_shapes.push_back(coarse_shape);
    }
  }

  void prepare_layers() {
    if (this->L == 0) {
      layer_len.clear();
      layer_off.clear();
      return;
    }

    layer_len.assign(this->L + 1, 0);
    layer_off.assign(this->L + 1, 0);

    // The length of coarsest layer (level 0)
    layer_len[0] = coarse_num_elems[this->L - 1];
    layer_off[0] = 0;

    SIZE accum = layer_len[0];

    for (SIZE l = 1; l <= this->L; ++l) {
      layer_len[l] = local_coeff_size[this->L - l];
      layer_off[l] = accum;
      accum += layer_len[l];
    }
  }

  // Calculate quantizers between levels(Used in Non-ROI)
  void CalcQuantizers(size_t dof, T *quantizers, enum error_bound_type type,
                      T tol, T s, T norm, SIZE l_target,
                      enum decomposition_type decomposition, bool reciprocal) {
    double abs_tol = tol;
    if (type == error_bound_type::REL) {
      abs_tol *= norm;
    }
    abs_tol *= 2;

    if (s == std::numeric_limits<T>::infinity()) {
      double C = orthogonal_projection ? (1 + std::pow(3, D)) : 1.0;

      for (int l = 0; l <= l_target; l++) {
        // Modified here
        quantizers[l] = (abs_tol) / (std::pow(2, l + 1) * C);

        if (reciprocal) {
          quantizers[l] = 1.0f / quantizers[l];
        }
      }
    } else {
      // s-norm (L2 family), non-ROI only.
      //
      // Same per-level law as LinearQuantizer's s != inf branch,
      //   step_l = abs_tol / (2^(s*l) * sqrt(dof)),
      // with l counted from the coarsest layer, matching layer_len/layer_off.
      //
      // LinearQuantizer additionally scales each value by its node volume,
      // reading a per-node level_volumes array. We cannot: the block-local
      // quantize kernels take one scalar per layer and have no volume input.
      // We do not need to. Hierarchy::calc_volume fills that array with
      // total_dist / (dof - 1) -- the *average* spacing, identical for every
      // node of a level -- so the per-node volume is a per-level constant and
      // folds into the scalar exactly.
      //
      // Sign convention follows the kernels: quantize multiplies by the
      // reciprocal step and by the volume, dequantize multiplies by the step
      // and by the reciprocal volume, so the pair round-trips.
      for (int l = 0; l <= l_target; l++) {
        double step =
            abs_tol / (std::exp2((double)s * l) * std::sqrt((double)dof));
        double volume = LayerVolume(l);
        quantizers[l] = reciprocal ? (volume / step) : (step / volume);
      }
    }
  }

  // sqrt of the cell volume of the grid that layer `l` lives on, matching
  // LevelwiseLinearQuantizerFunctor's `volume = sqrt(prod_d spacing_d)`.
  //
  // Layer 0 is the coarsest data (grid coarse_shapes[L-1]); layer l >= 1 holds
  // the coefficients dropped by block-local level L - l, which live on that
  // level's fine grid. Coordinates are treated as normalized to a unit extent
  // per dimension, which is what the block-local transform already assumes --
  // its mass/tridiagonal weights are built from a fixed uniform spacing.
  double LayerVolume(int l) {
    if (this->L == 0 || fine_shapes.empty()) {
      return 1.0;
    }
    const std::vector<SIZE> &shape =
        (l == 0) ? coarse_shapes[this->L - 1]
                 : fine_shapes[this->L - std::min<int>(l, this->L)];
    double volume = 1.0;
    for (DIM d = 0; d < D; d++) {
      SIZE n = shape[d];
      volume *= (n > 1) ? (1.0 / (double)(n - 1)) : 1.0;
    }
    return std::sqrt(volume);
  }

  // Reciprocal quantizers indexed by decompose level (level 0 = finest
  // coefficients) for the fused decompose+quantize path. Decompose level l
  // corresponds to non-ROI layer L - l, i.e. quantizer index L - l.
  std::vector<T> DecomposeLevelQuantizers(enum error_bound_type ebtype, T tol,
                                          T s, T norm) {
    std::vector<T> quantizers(this->L + 1);
    CalcQuantizers(hierarchy->total_num_elems(), quantizers.data(), ebtype, tol,
                   s, norm, this->L, config.decomposition, true);
    std::vector<T> level_quantizers(this->L);
    for (SIZE l = 0; l < this->L; l++) {
      level_quantizers[l] = quantizers[this->L - l];
    }
    return level_quantizers;
  }

  // Non-reciprocal dequantizers indexed by decompose level (level 0 = finest
  // coefficients) for the fused dequantize+recompose path. Decompose level l
  // corresponds to non-ROI layer L - l, i.e. quantizer index L - l.
  std::vector<T> RecomposeLevelDequantizers(enum error_bound_type ebtype, T tol,
                                            T s, T norm) {
    std::vector<T> quantizers(this->L + 1);
    CalcQuantizers(hierarchy->total_num_elems(), quantizers.data(), ebtype, tol,
                   s, norm, this->L, config.decomposition, false);
    std::vector<T> level_dequantizers(this->L);
    for (SIZE l = 0; l < this->L; l++) {
      level_dequantizers[l] = quantizers[this->L - l];
    }
    return level_dequantizers;
  }

  // Dequantize only the coarsest layer (layer 0). Used by the fused
  // dequantize+recompose path when there is no global stage; the coefficient
  // layers are dequantized inside the recompose kernels.
  void DequantizeCoarsest(SubArray<1, T, DeviceType> v,
                          SubArray<1, Q, DeviceType> quantized_v,
                          enum error_bound_type ebtype, T tol, T s, T norm,
                          int queue_idx) {
    std::vector<T> quantizers(this->L + 1);
    CalcQuantizers(hierarchy->total_num_elems(), quantizers.data(), ebtype, tol,
                   s, norm, this->L, config.decomposition, false);
    bool prep_huffman = config.lossless != lossless_type::CPU_Lossless &&
                        config.lossless != lossless_type::BlockDelta &&
                        config.lossless != lossless_type::LZ4;
    DeviceLauncher<DeviceType>::Execute(
        QuantizeLocalLevelKernel<T, Q, MGARDX_DEQUANTIZE, DeviceType>(
            quantizers[0], v, quantized_v, prep_huffman, config.huff_dict_size),
        queue_idx);
  }

  // Quantize only the coarsest layer (layer 0). Used by the fused
  // decompose+quantize path when there is no global stage; the coefficient
  // layers have already been quantized inside the decompose kernels.
  void QuantizeCoarsest(SubArray<1, T, DeviceType> v,
                        SubArray<1, Q, DeviceType> quantized_v,
                        enum error_bound_type ebtype, T tol, T s, T norm,
                        int queue_idx) {
    std::vector<T> quantizers(this->L + 1);
    CalcQuantizers(hierarchy->total_num_elems(), quantizers.data(), ebtype, tol,
                   s, norm, this->L, config.decomposition, true);
    bool prep_huffman = config.lossless != lossless_type::CPU_Lossless &&
                        config.lossless != lossless_type::BlockDelta &&
                        config.lossless != lossless_type::LZ4;
    DeviceLauncher<DeviceType>::Execute(
        QuantizeLocalLevelKernel<T, Q, MGARDX_QUANTIZE, DeviceType>(
            quantizers[0], v, quantized_v, prep_huffman, config.huff_dict_size),
        queue_idx);
  }

  void Quantize(SubArray<D, T, DeviceType> original_data,
                enum error_bound_type ebtype, T tol, T s, T norm,
                SubArray<D, Q, DeviceType> quantized_data, int queue_idx) {}

  void Dequantize(SubArray<D, T, DeviceType> original_data,
                  enum error_bound_type ebtype, T tol, T s, T norm,
                  SubArray<D, Q, DeviceType> quantized_data, int queue_idx) {}

  // Non-ROI
  template <typename LosslessCompressorType>
  void Quantize(SubArray<1, T, DeviceType> original_data,
                enum error_bound_type ebtype, T tol, T s, T norm,
                SubArray<1, Q, DeviceType> quantized_data,
                LosslessCompressorType &lossless, int queue_idx) {
    T *host_quantizers = new T[this->L + 1];
    CalcQuantizers(hierarchy->total_num_elems(), host_quantizers, ebtype, tol,
                   s, norm, this->L, config.decomposition, true);
    bool prep_huffman = config.lossless != lossless_type::CPU_Lossless &&
                        config.lossless != lossless_type::BlockDelta &&
                        config.lossless != lossless_type::LZ4;
    SIZE huff_dict_size = config.huff_dict_size;

    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    SIZE start_level = (this->M > 0) ? 1 : 0;
    SIZE offset_adjustment = (this->M > 0) ? layer_off[1] : 0;

    for (SIZE l = start_level; l <= this->L; ++l) {
      SIZE adjusted_off = layer_off[l] - offset_adjustment;
      SubArray<1, T, DeviceType> v_in({layer_len[l]},
                                      original_data((IDX)adjusted_off));
      SubArray<1, Q, DeviceType> qv({layer_len[l]},
                                    quantized_data((IDX)adjusted_off));
      // Launch
      T quantizer = host_quantizers[l];
      DeviceLauncher<DeviceType>::Execute(
          QuantizeLocalLevelKernel<T, Q, MGARDX_QUANTIZE, DeviceType>(
              quantizer, v_in, qv, prep_huffman, huff_dict_size),
          queue_idx);
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Local Quantization",
                  hierarchy->total_num_elems() * sizeof(T));
      timer.clear();
    }

    delete[] host_quantizers;
  }

  // Non-ROI
  template <typename LosslessCompressorType>
  void Dequantize(SubArray<1, T, DeviceType> original_data,
                  enum error_bound_type ebtype, T tol, T s, T norm,
                  SubArray<1, Q, DeviceType> quantized_data,
                  LosslessCompressorType &lossless, int queue_idx) {
    T *host_quantizers = new T[this->L + 1];
    CalcQuantizers(hierarchy->total_num_elems(), host_quantizers, ebtype, tol,
                   s, norm, this->L, config.decomposition, false);
    bool prep_huffman = config.lossless != lossless_type::CPU_Lossless &&
                        config.lossless != lossless_type::BlockDelta &&
                        config.lossless != lossless_type::LZ4;
    SIZE huff_dict_size = config.huff_dict_size;

    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    SIZE start_level = (this->M > 0) ? 1 : 0;
    SIZE offset_adjustment = (this->M > 0) ? layer_off[1] : 0;

    for (SIZE l = start_level; l <= this->L; ++l) {
      SIZE adjusted_off = layer_off[l] - offset_adjustment;
      SubArray<1, T, DeviceType> v_in({layer_len[l]},
                                      original_data((IDX)adjusted_off));
      SubArray<1, Q, DeviceType> qv({layer_len[l]},
                                    quantized_data((IDX)adjusted_off));
      // Launch
      T quantizer = host_quantizers[l];
      DeviceLauncher<DeviceType>::Execute(
          QuantizeLocalLevelKernel<T, Q, MGARDX_DEQUANTIZE, DeviceType>(
              quantizer, v_in, qv, prep_huffman, huff_dict_size),
          queue_idx);
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Local Dequantization",
                  hierarchy->total_num_elems() * sizeof(T));
      timer.clear();
    }

    delete[] host_quantizers;
  }

  // With ROI
  template <typename LosslessCompressorType>
  void Quantize(SubArray<1, T, DeviceType> original_data,
                enum error_bound_type ebtype, double tol, T s, T norm,
                SubArray<1, Q, DeviceType> quantized_data,
                SubArray<1, double, DeviceType> device_roi_tolerance_map,
                const std::vector<SIZE> &level_offsets,
                const std::vector<SIZE> &level_block_counts,
                LosslessCompressorType &lossless, int queue_idx) {
    if (s != std::numeric_limits<T>::infinity()) {
      throw ProcessingException("Only L-inf supported");
    }

    double C = orthogonal_projection ? (1 + std::pow(3, D)) : 1.0;
    double norm_factor = (ebtype == error_bound_type::REL) ? (double)norm : 1.0;
    bool prep_huffman = config.lossless != lossless_type::CPU_Lossless &&
                        config.lossless != lossless_type::BlockDelta &&
                        config.lossless != lossless_type::LZ4;
    SIZE huff_dict_size = config.huff_dict_size;

    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    SIZE accumulated_coeff_size = 0;
    // Process Layer 1 to Layer L with ROI tolerances
    for (SIZE l = 0; l < this->L; ++l) {
      SIZE roi_level = l;
      SIZE level_offset = level_offsets[roi_level];
      SIZE num_blocks = level_block_counts[roi_level];

      // l=0 is finest coefficients (laid out at the end of the data array),
      // which maps to non-ROI layer L. The correct exponent is (L - l + 1).
      double denom = std::pow(2, this->L - l + 1) * C;

      // Compute per-block quantizers directly on device from the
      // already-uploaded tolerance map (reciprocal for quantization), instead
      // of recomputing on host and re-uploading every call.
      Array<1, T, DeviceType> device_quantizers({num_blocks}, queue_idx);
      DeviceLauncher<DeviceType>::Execute(
          ComputeROIQuantizersKernel<T, DeviceType>(
              device_roi_tolerance_map, level_offset, num_blocks, norm_factor,
              denom, /*reciprocal=*/true,
              SubArray<1, T, DeviceType>(device_quantizers)),
          queue_idx);

      accumulated_coeff_size += local_coeff_size[l];
      SubArray<1, T, DeviceType> v_in(
          {local_coeff_size[l]},
          original_data(original_data.shape(0) - accumulated_coeff_size));
      SubArray<1, Q, DeviceType> qv(
          {local_coeff_size[l]},
          quantized_data(quantized_data.shape(0) - accumulated_coeff_size));

      DeviceLauncher<DeviceType>::Execute(
          QuantizeLocalLevelROIKernel<D, T, Q, MGARDX_QUANTIZE, DeviceType>(
              SubArray<1, T, DeviceType>(device_quantizers), v_in, qv,
              prep_huffman, huff_dict_size),
          queue_idx);
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Local Quantization with ROI",
                  hierarchy->total_num_elems() * sizeof(T));
      timer.clear();
    }
  }

  // With ROI
  template <typename LosslessCompressorType>
  void Dequantize(SubArray<1, T, DeviceType> original_data,
                  enum error_bound_type ebtype, double tol, T s, T norm,
                  SubArray<1, Q, DeviceType> quantized_data,
                  SubArray<1, double, DeviceType> device_roi_tolerance_map,
                  const std::vector<SIZE> &level_offsets,
                  const std::vector<SIZE> &level_block_counts,
                  LosslessCompressorType &lossless, int queue_idx) {
    if (s != std::numeric_limits<T>::infinity()) {
      throw ProcessingException("Only L-inf supported");
    }

    double C = orthogonal_projection ? (1 + std::pow(3, D)) : 1.0;
    double norm_factor = (ebtype == error_bound_type::REL) ? (double)norm : 1.0;
    bool prep_huffman = config.lossless != lossless_type::CPU_Lossless &&
                        config.lossless != lossless_type::BlockDelta &&
                        config.lossless != lossless_type::LZ4;
    SIZE huff_dict_size = config.huff_dict_size;

    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    SIZE accumulated_coeff_size = 0;
    // Process Layer 1 to Layer L with ROI tolerances
    for (SIZE l = 0; l < this->L; ++l) {
      SIZE roi_level = l;
      SIZE level_offset = level_offsets[roi_level];
      SIZE num_blocks = level_block_counts[roi_level];

      // l=0 is finest coefficients (laid out at the end of the data array),
      // which maps to non-ROI layer L. The correct exponent is (L - l + 1).
      double denom = std::pow(2, this->L - l + 1) * C;

      // Compute per-block quantizers directly on device from the
      // already-uploaded tolerance map (no reciprocal for dequantization).
      Array<1, T, DeviceType> device_quantizers({num_blocks}, queue_idx);
      DeviceLauncher<DeviceType>::Execute(
          ComputeROIQuantizersKernel<T, DeviceType>(
              device_roi_tolerance_map, level_offset, num_blocks, norm_factor,
              denom, /*reciprocal=*/false,
              SubArray<1, T, DeviceType>(device_quantizers)),
          queue_idx);

      accumulated_coeff_size += local_coeff_size[l];
      SubArray<1, T, DeviceType> v_in(
          {local_coeff_size[l]},
          original_data(original_data.shape(0) - accumulated_coeff_size));
      SubArray<1, Q, DeviceType> qv(
          {local_coeff_size[l]},
          quantized_data(quantized_data.shape(0) - accumulated_coeff_size));

      DeviceLauncher<DeviceType>::Execute(
          QuantizeLocalLevelROIKernel<D, T, Q, MGARDX_DEQUANTIZE, DeviceType>(
              SubArray<1, T, DeviceType>(device_quantizers), v_in, qv,
              prep_huffman, huff_dict_size),
          queue_idx);
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Local Dequantization with ROI",
                  hierarchy->total_num_elems() * sizeof(T));
      timer.clear();
    }
  }

  bool initialized;
  SIZE L;
  SIZE M;
  HybridHierarchy<D, T, DeviceType> *hierarchy;
  Config config;
  bool orthogonal_projection = true;

  // For Non-ROI
  std::vector<SIZE> layer_len;
  std::vector<SIZE> layer_off;

  // For ROI
  std::vector<double> tol_table;

  std::vector<SIZE> fine_num_elems;
  std::vector<SIZE> coarse_num_elems;
  std::vector<SIZE> local_coeff_size;
  std::vector<SIZE> coarse_shape;
  // Grid each block-local level lives on, needed by the s != inf quantizer to
  // weight a layer by its node spacing.
  std::vector<std::vector<SIZE>> fine_shapes;
  std::vector<std::vector<SIZE>> coarse_shapes;
};

} // namespace mgard_x

#endif
