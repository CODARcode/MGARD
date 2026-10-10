/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_160_HYBRID_HIERARCHY_LINEAR_QUANTIZATION_TEMPLATE
#define MGARD_X_160_HYBRID_HIERARCHY_LINEAR_QUANTIZATION_TEMPLATE

#include <string>

#include "../RuntimeX/RuntimeX.h"
#include "LinearQuantization.hpp"
#include "LocalQuantization.hpp"
#include "QuantizationInterface.hpp"

namespace mgard_x_160 {

#define MGARDX_QUANTIZE 1
#define MGARDX_DEQUANTIZE 2

template <DIM D, typename T, typename Q, typename DeviceType>
class HybridHierarchyQuantizer
    : public QuantizationInterface<D, T, Q, DeviceType> {
public:
  HybridHierarchyQuantizer() : initialized(false) {}

  HybridHierarchyQuantizer(HybridHierarchy<D, T, DeviceType> &hierarchy,
                           Config config)
      : initialized(true), hierarchy(&hierarchy), config(config) {
    hierarchy.check_config(config);
    this->L = config.num_local_refactoring_level;
    this->M = config.num_global_refactoring_level;

    if (this->L == 0 && this->M == 0) {
      throw ProcessingException("Both L and M cannot be zero");
    }
  }

  void Adapt(HybridHierarchy<D, T, DeviceType> &hierarchy, Config config,
             int queue_idx) {
    hierarchy.check_config(config);
    this->initialized = true;
    this->hierarchy = &hierarchy;
    this->config = config;
    this->L = config.num_local_refactoring_level;
    this->M = config.num_global_refactoring_level;

    if (this->L == 0 && this->M == 0) {
      throw ProcessingException("Both L and M cannot be zero");
    }

    if (this->L > 0) {
      local_quantizer.Adapt(hierarchy, config, queue_idx);
      if (config.enable_roi) {
        this->initial_block_tolerances = config.roi_tolerance_map;
        ComputeLocalShapes();
        SetBlockTolerances(this->initial_block_tolerances, queue_idx);
      }
    }

    if (this->M > 0) {
      global_hierarchy = &hierarchy.global_hierarchy();
      global_quantizer.Adapt(*global_hierarchy, config, queue_idx);
    } else {
      global_hierarchy = nullptr;
    }
  }

  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape) {
    size_t size = 0;
    return size;
  }

  void SetOrthogonalProjection(bool enabled) {
    orthogonal_projection = enabled;
    local_quantizer.SetOrthogonalProjection(enabled);
  }

  // Set block-level tolerances according to ROI table
  void SetBlockTolerances(const std::vector<double> &initial_block_tolerances,
                          int queue_idx) {
    BuildROIToleranceMap(initial_block_tolerances);
    // Upload once here so Quantize/Dequantize don't have to re-upload the
    // tolerance map (which barely changes) on every call.
    device_roi_tolerance_map.resize({(SIZE)roi_tolerance_map.size()},
                                    queue_idx);
    device_roi_tolerance_map.load(roi_tolerance_map.data(), 0, queue_idx);
  }

  // An s-norm bound is only defensible when the block-local stage is a single
  // level with no global stage: only then is every block transformed
  // independently, so the total squared error is a sum over blocks and the
  // budget divides evenly in quadrature (see restrict_hybrid_config_for_s_norm,
  // which makes the public compress path fall back to exactly this).
  //
  // Here we can only refuse: L and M fixed the buffer layout back at Adapt(),
  // so a low-level caller cannot be silently re-configured mid-flight.
  void CheckSNormConfiguration(T s) {
    if (s == std::numeric_limits<T>::infinity()) {
      return;
    }
    if (this->L != 1 || this->M != 0) {
      throw ProcessingException(
          "hybrid with an s-norm bound requires exactly one block-local level "
          "and no global stage (got L=" +
          std::to_string(this->L) + ", M=" + std::to_string(this->M) +
          "); the public compress path falls back to this automatically");
    }
    if (config.enable_roi) {
      throw ProcessingException("ROI mode is L-inf only");
    }
  }

  // Tolerance for the coarsest layer when it has to be quantized on its own
  // (M == 0, so there is no global stage to cover it). In ROI mode the
  // coarsest layer has no tolerance of its own -- the map only describes the
  // coefficient layers' block grids -- so use the tightest tolerance among
  // the coarsest blocks, which is exactly what the global stage is given when
  // M > 0 (see ErrorBudgetAllocation).
  T CoarsestTolerance(T tol) {
    return config.enable_roi ? (T)GetMinToleranceForGlobal() : tol;
  }

  // Called only when this->M > 0
  T ErrorBudgetAllocation(T tol) {
    T global_tol = tol;

    if (this->L > 0) {
      if (this->config.enable_roi) {
        global_tol = GetMinToleranceForGlobal();
      }
      global_tol = global_tol / (1 << this->L);
    }
    return global_tol;
  }

  void Quantize(SubArray<D, T, DeviceType> original_data,
                enum error_bound_type ebtype, T tol, T s, T norm,
                SubArray<D, Q, DeviceType> quantized_data, int queue_idx) {}

  void Dequantize(SubArray<D, T, DeviceType> original_data,
                  enum error_bound_type ebtype, T tol, T s, T norm,
                  SubArray<D, Q, DeviceType> quantized_data, int queue_idx) {}

  // Quantize the global (coarsest) region at the front of the decomposed
  // array with the global quantizer. Shared between the unfused Quantize()
  // path and the fused decompose+quantize path. Only valid when M > 0.
  template <typename LosslessCompressorType>
  void QuantizeGlobalPart(SubArray<1, T, DeviceType> original_data,
                          enum error_bound_type ebtype, T tol, T s, T norm,
                          SubArray<1, Q, DeviceType> quantized_data,
                          LosslessCompressorType &lossless, int queue_idx) {
    T global_tol = ErrorBudgetAllocation(tol);

    std::vector<SIZE> global_shape =
        global_hierarchy->level_shape(global_hierarchy->l_target());
    SubArray<D, T, DeviceType> global_data_v(global_shape,
                                             original_data.data());
    SubArray<D, Q, DeviceType> global_data_q(global_shape,
                                             quantized_data.data());
    for (DIM d = 0; d < D; d++) {
      global_data_v.setLd(d, global_shape[d]);
      global_data_q.setLd(d, global_shape[d]);
    }
    global_data_v.project(D - 3, D - 2, D - 1);
    global_data_q.project(D - 3, D - 2, D - 1);
    global_quantizer.Quantize(global_data_v, ebtype, global_tol, s, norm,
                              global_data_q, lossless, queue_idx,
                              orthogonal_projection);
  }

  // Dequantize the global (coarsest) region at the front of the decomposed
  // array with the global quantizer. Shared between the unfused Dequantize()
  // path and the fused dequantize+recompose path. Only valid when M > 0.
  template <typename LosslessCompressorType>
  void DequantizeGlobalPart(SubArray<1, T, DeviceType> original_data,
                            enum error_bound_type ebtype, T tol, T s, T norm,
                            SubArray<1, Q, DeviceType> quantized_data,
                            LosslessCompressorType &lossless, int queue_idx) {
    T global_tol = ErrorBudgetAllocation(tol);

    std::vector<SIZE> global_shape =
        global_hierarchy->level_shape(global_hierarchy->l_target());
    SubArray<D, T, DeviceType> global_data_v(global_shape,
                                             original_data.data());
    SubArray<D, Q, DeviceType> global_data_q(global_shape,
                                             quantized_data.data());
    for (DIM d = 0; d < D; d++) {
      global_data_v.setLd(d, global_shape[d]);
      global_data_q.setLd(d, global_shape[d]);
    }
    global_data_v.project(D - 3, D - 2, D - 1);
    global_data_q.project(D - 3, D - 2, D - 1);
    global_quantizer.Dequantize(global_data_v, ebtype, global_tol, s, norm,
                                global_data_q, lossless, queue_idx,
                                orthogonal_projection);
  }

  template <typename LosslessCompressorType>
  void Quantize(SubArray<1, T, DeviceType> original_data,
                enum error_bound_type ebtype, T tol, T s, T norm,
                SubArray<1, Q, DeviceType> quantized_data,
                LosslessCompressorType &lossless, int queue_idx) {
    if (this->L == 0 && this->M == 0) {
      throw ProcessingException("Both L and M cannot be zero");
    }
    CheckSNormConfiguration(s);
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    SIZE global_q_size = 0;

    // Global quantization
    if (this->M > 0) {
      global_q_size = global_hierarchy->total_num_elems();
      QuantizeGlobalPart(original_data, ebtype, tol, s, norm, quantized_data,
                         lossless, queue_idx);
    }

    // Local quantization
    if (this->L > 0) {
      SIZE local_length = original_data.shape(0) - global_q_size;

      SubArray<1, T, DeviceType> local_data_v({local_length},
                                              original_data(global_q_size));
      SubArray<1, Q, DeviceType> local_data_q({local_length},
                                              quantized_data(global_q_size));

      // Switch between ROI and Non-ROI
      if (config.enable_roi) {
        local_quantizer.Quantize(
            local_data_v, ebtype, 0.0, s, norm, local_data_q,
            SubArray<1, double, DeviceType>(device_roi_tolerance_map),
            level_offsets, level_block_counts, lossless, queue_idx);
      } else {
        local_quantizer.Quantize(local_data_v, ebtype, tol, s, norm,
                                 local_data_q, lossless, queue_idx);
      }
    }

    // Coarsest layer, at the front of the decomposed array. When M > 0 the
    // global stage above quantized it; when M == 0 the non-ROI local
    // quantizer covers it as its layer 0, but the ROI local quantizer's
    // per-block quantizers only span the L coefficient layers. Without this
    // the coarsest layer -- (5/8)^D of the array -- would never be written,
    // and only a same-process decompress would appear to work, by reading the
    // values left in the cached decomposed buffer.
    if (this->L > 0 && this->M == 0 && config.enable_roi) {
      SIZE coarsest_size = local_quantizer.layer_len[0];
      SubArray<1, T, DeviceType> coarsest_v({coarsest_size},
                                            original_data.data());
      SubArray<1, Q, DeviceType> coarsest_q({coarsest_size},
                                            quantized_data.data());
      local_quantizer.QuantizeCoarsest(coarsest_v, coarsest_q, ebtype,
                                       CoarsestTolerance(tol), s, norm,
                                       queue_idx);
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Hybrid Quantization",
                  hierarchy->total_num_elems() * sizeof(T));
      timer.clear();
    }
  }

  // Whether the fused decompose+quantize path can be used: it covers the
  // local stage only (L > 0), relies on the 3D in-cache block kernel, and has
  // the same L-inf-only constraint as the local quantizer.
  bool CanFuseQuantize(T s) {
    // s != inf is fine here: its weighting is a per-level scalar (the node
    // volume is constant within a level), which is exactly what the fused
    // kernels already take. ROI is not -- its per-block quantizers have no
    // s-norm formulation, so it stays L-inf only.
    if (config.enable_roi && s != std::numeric_limits<T>::infinity()) {
      return false;
    }
    return this->L > 0 && D >= 1 && D <= 3;
  }

  // Which of the conditions above ruled the fused path out, for logging. Kept
  // next to CanFuseQuantize so the two cannot drift apart. Returns an empty
  // string when fusing is possible.
  std::string WhyCannotFuseQuantize(T s) {
    if (this->L == 0) {
      return "no block-local levels";
    }
    if (D > 3) {
      return "fused kernel supports 1D, 2D and 3D only";
    }
    if (config.enable_roi && s != std::numeric_limits<T>::infinity()) {
      return "ROI mode is L-inf only";
    }
    return "";
  }

  // Fused decompose+quantization driver: the local levels are decomposed and
  // quantized in one kernel per level (coefficients never round-trip through
  // global memory as T), writing symbols directly to their final location in
  // quantized_data. The coarsest region is then handled as in the unfused
  // path: global decompose + global quantize when M > 0, otherwise a single
  // coarsest-layer quantization -- in ROI mode too, since the per-block
  // quantizers only span the coefficient layers.
  template <typename RefactorType, typename LosslessCompressorType>
  void DecomposeQuantize(RefactorType &refactor,
                         SubArray<D, T, DeviceType> data,
                         SubArray<1, T, DeviceType> decomposed_data,
                         SubArray<1, Q, DeviceType> quantized_data,
                         enum error_bound_type ebtype, T tol, T s, T norm,
                         LosslessCompressorType &lossless, int queue_idx) {
    if (!CanFuseQuantize(s)) {
      throw ProcessingException("DecomposeQuantize requires L > 0, D <= 3, and "
                                "(non-ROI or s == inf)");
    }
    CheckSNormConfiguration(s);
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    bool prep_huffman = config.lossless != lossless_type::CPU_Lossless &&
                        config.lossless != lossless_type::BlockDelta &&
                        config.lossless != lossless_type::LZ4;
    SIZE huff_dict_size = config.huff_dict_size;

    if (config.enable_roi) {
      // Per-level per-block reciprocal quantizers from the device-resident
      // tolerance map, same math and block ordering as the ROI Quantize path
      // (the fused kernel indexes them by block id, which matches the
      // idx / hybrid_local_coeff_per_block(D) mapping of the unfused ROI
      // kernel).
      double C = orthogonal_projection ? (1 + std::pow(3, D)) : 1.0;
      double norm_factor =
          (ebtype == error_bound_type::REL) ? (double)norm : 1.0;
      std::vector<Array<1, T, DeviceType>> device_quantizers(this->L);
      std::vector<SubArray<1, T, DeviceType>> block_quantizers(this->L);
      for (SIZE l = 0; l < this->L; ++l) {
        SIZE level_offset = level_offsets[l];
        SIZE num_blocks = level_block_counts[l];
        double denom = std::pow(2, this->L - l + 1) * C;
        device_quantizers[l] = Array<1, T, DeviceType>({num_blocks}, queue_idx);
        DeviceLauncher<DeviceType>::Execute(
            ComputeROIQuantizersKernel<T, DeviceType>(
                SubArray<1, double, DeviceType>(device_roi_tolerance_map),
                level_offset, num_blocks, norm_factor, denom,
                /*reciprocal=*/true,
                SubArray<1, T, DeviceType>(device_quantizers[l])),
            queue_idx);
        block_quantizers[l] = SubArray<1, T, DeviceType>(device_quantizers[l]);
      }
      refactor.local_refactor.DecomposeQuantize(
          data, decomposed_data, quantized_data, std::vector<T>(),
          block_quantizers, prep_huffman, huff_dict_size, queue_idx,
          orthogonal_projection);
    } else {
      std::vector<T> level_quantizers =
          local_quantizer.DecomposeLevelQuantizers(ebtype, tol, s, norm);
      refactor.local_refactor.DecomposeQuantize(
          data, decomposed_data, quantized_data, level_quantizers,
          std::vector<SubArray<1, T, DeviceType>>(), prep_huffman,
          huff_dict_size, queue_idx, orthogonal_projection);
    }

    // Coarsest region (compacted at the front of decomposed_data by the
    // fused local stage).
    if (this->M > 0) {
      refactor.DecomposeGlobal(decomposed_data, queue_idx,
                               orthogonal_projection);
      QuantizeGlobalPart(decomposed_data, ebtype, tol, s, norm, quantized_data,
                         lossless, queue_idx);
    } else {
      // Also in ROI mode: the per-block quantizers only span the coefficient
      // layers, so the coarsest layer needs one of its own here.
      SIZE coarsest_size = local_quantizer.layer_len[0];
      SubArray<1, T, DeviceType> coarsest_v({coarsest_size},
                                            decomposed_data.data());
      SubArray<1, Q, DeviceType> coarsest_q({coarsest_size},
                                            quantized_data.data());
      local_quantizer.QuantizeCoarsest(coarsest_v, coarsest_q, ebtype,
                                       CoarsestTolerance(tol), s, norm,
                                       queue_idx);
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Hybrid Decomposition+Quantization (fused)",
                  hierarchy->total_num_elems() * sizeof(T));
      timer.clear();
    }
  }

  // Fused dequantize+recomposition driver (inverse of DecomposeQuantize):
  // the coarsest region is first reconstructed as in the unfused path
  // (global dequantize + global recompose when M > 0, otherwise a single
  // coarsest-layer dequantization, in ROI mode too),
  // then the local levels are dequantized and recomposed in one kernel per
  // level (coefficients never round-trip through global memory as T),
  // writing the final level directly into the unpadded output.
  template <typename RefactorType, typename LosslessCompressorType>
  void DequantizeRecompose(RefactorType &refactor,
                           SubArray<D, T, DeviceType> data,
                           SubArray<1, T, DeviceType> decomposed_data,
                           SubArray<1, Q, DeviceType> quantized_data,
                           enum error_bound_type ebtype, T tol, T s, T norm,
                           LosslessCompressorType &lossless, int queue_idx) {
    if (!CanFuseQuantize(s)) {
      throw ProcessingException("DequantizeRecompose requires L > 0, D <= 3, "
                                "and (non-ROI or s == inf)");
    }
    CheckSNormConfiguration(s);
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    bool prep_huffman = config.lossless != lossless_type::CPU_Lossless &&
                        config.lossless != lossless_type::BlockDelta &&
                        config.lossless != lossless_type::LZ4;
    SIZE huff_dict_size = config.huff_dict_size;

    // Coarsest region first: it is the input of the local recomposition.
    if (this->M > 0) {
      DequantizeGlobalPart(decomposed_data, ebtype, tol, s, norm,
                           quantized_data, lossless, queue_idx);
      refactor.RecomposeGlobal(decomposed_data, queue_idx,
                               orthogonal_projection);
    } else {
      // Also in ROI mode -- see the matching branch in DecomposeQuantize.
      SIZE coarsest_size = local_quantizer.layer_len[0];
      SubArray<1, T, DeviceType> coarsest_v({coarsest_size},
                                            decomposed_data.data());
      SubArray<1, Q, DeviceType> coarsest_q({coarsest_size},
                                            quantized_data.data());
      local_quantizer.DequantizeCoarsest(coarsest_v, coarsest_q, ebtype,
                                         CoarsestTolerance(tol), s, norm,
                                         queue_idx);
    }

    if (config.enable_roi) {
      // Per-level per-block dequantizers from the device-resident tolerance
      // map, same math and block ordering as the ROI Dequantize path (the
      // fused kernel indexes them by block id, which matches the
      // idx / hybrid_local_coeff_per_block(D) mapping of the unfused ROI
      // kernel).
      double C = orthogonal_projection ? (1 + std::pow(3, D)) : 1.0;
      double norm_factor =
          (ebtype == error_bound_type::REL) ? (double)norm : 1.0;
      std::vector<Array<1, T, DeviceType>> device_dequantizers(this->L);
      std::vector<SubArray<1, T, DeviceType>> block_dequantizers(this->L);
      for (SIZE l = 0; l < this->L; ++l) {
        SIZE level_offset = level_offsets[l];
        SIZE num_blocks = level_block_counts[l];
        double denom = std::pow(2, this->L - l + 1) * C;
        device_dequantizers[l] =
            Array<1, T, DeviceType>({num_blocks}, queue_idx);
        DeviceLauncher<DeviceType>::Execute(
            ComputeROIQuantizersKernel<T, DeviceType>(
                SubArray<1, double, DeviceType>(device_roi_tolerance_map),
                level_offset, num_blocks, norm_factor, denom,
                /*reciprocal=*/false,
                SubArray<1, T, DeviceType>(device_dequantizers[l])),
            queue_idx);
        block_dequantizers[l] =
            SubArray<1, T, DeviceType>(device_dequantizers[l]);
      }
      refactor.local_refactor.RecomposeDequantize(
          data, decomposed_data, quantized_data, std::vector<T>(),
          block_dequantizers, prep_huffman, huff_dict_size, queue_idx,
          orthogonal_projection);
    } else {
      std::vector<T> level_dequantizers =
          local_quantizer.RecomposeLevelDequantizers(ebtype, tol, s, norm);
      refactor.local_refactor.RecomposeDequantize(
          data, decomposed_data, quantized_data, level_dequantizers,
          std::vector<SubArray<1, T, DeviceType>>(), prep_huffman,
          huff_dict_size, queue_idx, orthogonal_projection);
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Hybrid Dequantization+Recomposition (fused)",
                  hierarchy->total_num_elems() * sizeof(T));
      timer.clear();
    }
  }

  template <typename LosslessCompressorType>
  void Dequantize(SubArray<1, T, DeviceType> original_data,
                  enum error_bound_type ebtype, T tol, T s, T norm,
                  SubArray<1, Q, DeviceType> quantized_data,
                  LosslessCompressorType &lossless, int queue_idx) {
    if (this->L == 0 && this->M == 0) {
      throw ProcessingException("Both L and M cannot be zero");
    }
    CheckSNormConfiguration(s);
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    SIZE global_q_size = 0;
    if (this->M > 0) {
      global_q_size = global_hierarchy->total_num_elems();
      // log::info("Total Elems: " + std::to_string(global_q_size));
    }

    // Global dequantization
    if (this->M > 0) {
      DequantizeGlobalPart(original_data, ebtype, tol, s, norm, quantized_data,
                           lossless, queue_idx);
    }

    // Local dequantization
    if (this->L > 0) {
      SIZE local_length = original_data.shape(0) - global_q_size;
      SubArray<1, T, DeviceType> local_data_v({local_length},
                                              original_data(global_q_size));
      SubArray<1, Q, DeviceType> local_data_q({local_length},
                                              quantized_data(global_q_size));
      // Switch between ROI and Non-ROI
      if (config.enable_roi) {
        local_quantizer.Dequantize(
            local_data_v, ebtype, 0.0, s, norm, local_data_q,
            SubArray<1, double, DeviceType>(device_roi_tolerance_map),
            level_offsets, level_block_counts, lossless, queue_idx);
      } else {
        local_quantizer.Dequantize(local_data_v, ebtype, tol, s, norm,
                                   local_data_q, lossless, queue_idx);
      }
    }

    // Coarsest layer -- inverse of the corresponding block in Quantize().
    if (this->L > 0 && this->M == 0 && config.enable_roi) {
      SIZE coarsest_size = local_quantizer.layer_len[0];
      SubArray<1, T, DeviceType> coarsest_v({coarsest_size},
                                            original_data.data());
      SubArray<1, Q, DeviceType> coarsest_q({coarsest_size},
                                            quantized_data.data());
      local_quantizer.DequantizeCoarsest(coarsest_v, coarsest_q, ebtype,
                                         CoarsestTolerance(tol), s, norm,
                                         queue_idx);
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Hybrid Dequantization",
                  hierarchy->total_num_elems() * sizeof(T));
      timer.clear();
    }
  }

  // Compute fine and coarse shapes for all local levels
  void ComputeLocalShapes() {
    fine_shapes.clear();
    coarse_shapes.clear();

    // Only compute if L > 0
    if (this->L == 0) {
      return;
    }

    for (SIZE l = 0; l < this->L; ++l) {
      fine_shapes.push_back(hierarchy->local_fine_shape(l));
      coarse_shapes.push_back(hierarchy->local_coarse_shape(l));
    }
  }

  // Build ROI tolerance map for all local levels (stored as 1D array)
  void BuildROIToleranceMap(const std::vector<double> &initial_tolerances) {
    roi_tolerance_map.clear();
    level_offsets.clear();
    level_block_counts.clear();

    // Only build ROI map if L > 0
    if (this->L == 0) {
      return;
    }

    // Level 0: use initial tolerances directly
    level_offsets.push_back(0);
    level_block_counts.push_back(initial_tolerances.size());
    roi_tolerance_map.insert(roi_tolerance_map.end(),
                             initial_tolerances.begin(),
                             initial_tolerances.end());

    // Propagate tolerances through local levels
    for (SIZE level = 1; level < this->L; ++level) {
      SIZE prev_offset = level_offsets[level - 1];
      SIZE prev_count = level_block_counts[level - 1];

      std::vector<double> prev_level_tolerances(
          roi_tolerance_map.begin() + prev_offset,
          roi_tolerance_map.begin() + prev_offset + prev_count);

      std::vector<double> next_level_tolerances =
          PropagateTolerances(prev_level_tolerances, level - 1, level);

      level_offsets.push_back(roi_tolerance_map.size());
      level_block_counts.push_back(next_level_tolerances.size());
      roi_tolerance_map.insert(roi_tolerance_map.end(),
                               next_level_tolerances.begin(),
                               next_level_tolerances.end());
    }
  }

  // Propagate error to next level
  std::vector<double>
  PropagateTolerances(const std::vector<double> &current_tolerances,
                      SIZE curr_level, SIZE next_level) {
    // Get current and next level block dimensions from computed shapes
    std::vector<SIZE> curr_blocks = GetBlockDimensions(curr_level);
    std::vector<SIZE> next_blocks = GetBlockDimensions(next_level);

    SIZE next_size = 1;
    for (DIM d = 0; d < D; ++d) {
      next_size *= next_blocks[d];
    }

    // Initialize tolerance list for next block
    std::vector<double> next_tolerances(next_size,
                                        std::numeric_limits<double>::max());

    // For each block in next level, find minimum tolerance from contributing
    // blocks
    for (SIZE idx = 0; idx < next_size; ++idx) {
      std::vector<SIZE> next_coord = LinearToCoord(idx, next_blocks);
      double min_tol = std::numeric_limits<double>::max();

      // Find all contributing blocks from current level
      std::vector<SIZE> contributing_blocks =
          GetContributingBlocks(next_coord, curr_blocks);

      for (SIZE contrib_idx : contributing_blocks) {
        if (contrib_idx < current_tolerances.size()) {
          min_tol = std::min(min_tol, current_tolerances[contrib_idx]);
        }
      }

      next_tolerances[idx] = min_tol;
    }

    return next_tolerances;
  }

  // Contributing tables:
  // NEXT BLOCK       CUR BLOCK
  //     0               0,1
  //     1              1,2,3
  //     2               3,4
  //     3              4,5,6
  //     4               6,7
  // Get contributing block indices from previous level for a given next-level
  // block
  std::vector<SIZE>
  GetContributingBlocks(const std::vector<SIZE> &next_coord,
                        const std::vector<SIZE> &curr_blocks) {
    std::vector<std::vector<SIZE>> contrib_per_dim(D);

    // For each dimension, apply the 5->8 inverse mapping
    for (DIM d = 0; d < D; ++d) {
      SIZE next_idx = next_coord[d];
      SIZE group = next_idx / 5;
      SIZE offset = next_idx % 5;
      SIZE base = group * 8;

      // Apply the propagation pattern
      switch (offset) {
      case 0: // min(0, 1)
        contrib_per_dim[d] = {base + 0, base + 1};
        break;
      case 1: // min(1, 2, 3)
        contrib_per_dim[d] = {base + 1, base + 2, base + 3};
        break;
      case 2: // min(3, 4)
        contrib_per_dim[d] = {base + 3, base + 4};
        break;
      case 3: // min(4, 5, 6)
        contrib_per_dim[d] = {base + 4, base + 5, base + 6};
        break;
      case 4: // min(6, 7)
        contrib_per_dim[d] = {base + 6, base + 7};
        break;
      }

      // Filter out-of-bounds indices
      std::vector<SIZE> valid;
      for (SIZE idx : contrib_per_dim[d]) {
        if (idx < curr_blocks[d]) {
          valid.push_back(idx);
        }
      }
      contrib_per_dim[d] = valid;
    }

    // Generate all combinations (Cartesian product)
    return CartesianProduct(contrib_per_dim, curr_blocks);
  }

  // Cartesian product of contributing indices across dimensions
  std::vector<SIZE>
  CartesianProduct(const std::vector<std::vector<SIZE>> &indices_per_dim,
                   const std::vector<SIZE> &blocks) {
    std::vector<SIZE> result;
    std::vector<SIZE> coord(D);
    CartesianProductHelper(indices_per_dim, blocks, 0, coord, result);
    return result;
  }

  void
  CartesianProductHelper(const std::vector<std::vector<SIZE>> &indices_per_dim,
                         const std::vector<SIZE> &blocks, DIM dim,
                         std::vector<SIZE> &coord, std::vector<SIZE> &result) {
    if (dim == D) {
      result.push_back(CoordToLinear(coord, blocks));
      return;
    }

    for (SIZE idx : indices_per_dim[dim]) {
      coord[dim] = idx;
      CartesianProductHelper(indices_per_dim, blocks, dim + 1, coord, result);
    }
  }

  // Convert linear index to coordinate
  std::vector<SIZE> LinearToCoord(SIZE linear_idx,
                                  const std::vector<SIZE> &dims) {
    std::vector<SIZE> coord(D);
    for (int d = D - 1; d >= 0; --d) {
      coord[d] = linear_idx % dims[d];
      linear_idx /= dims[d];
    }
    return coord;
  }

  // Convert coordinate to linear index
  SIZE CoordToLinear(const std::vector<SIZE> &coord,
                     const std::vector<SIZE> &dims) {
    SIZE linear = 0;
    SIZE stride = 1;
    for (int d = D - 1; d >= 0; --d) {
      linear += coord[d] * stride;
      stride *= dims[d];
    }
    return linear;
  }

  // Get block dimensions at a specific level
  std::vector<SIZE> GetBlockDimensions(SIZE level) {
    // Use the fine shape for this level (before decomposition)
    const std::vector<SIZE> &fine_shape = fine_shapes[level];

    // Calculate block size (8x8x8 for local decomposition)
    const SIZE BLOCK_SIZE = 8;

    // Calculate number of blocks in each dimension
    std::vector<SIZE> block_dims(D);
    for (DIM d = 0; d < D; ++d) {
      block_dims[d] = (fine_shape[d] + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }

    return block_dims;
  }

  // Get minimum tolerance for global quantization
  T GetMinToleranceForGlobal() {
    if (!this->config.enable_roi) {
      return std::numeric_limits<double>::max();
    }

    // Get the last level
    SIZE last_level_idx = level_offsets.size() - 1;
    SIZE last_offset = level_offsets[last_level_idx];
    SIZE last_count = level_block_counts[last_level_idx];

    double min_tol = std::numeric_limits<double>::max();
    for (SIZE i = 0; i < last_count; ++i) {
      min_tol = std::min(min_tol, roi_tolerance_map[last_offset + i]);
    }

    return min_tol;
  }

  bool initialized;
  SIZE L; // Number of local levels
  SIZE M; // Number of global levels

  HybridHierarchy<D, T, DeviceType> *hierarchy;
  // hierarchy->global_hierarchy() when M > 0.
  Hierarchy<D, T, DeviceType> *global_hierarchy = nullptr;
  Config config;
  bool orthogonal_projection = true;

  LocalQuantizer<D, T, Q, DeviceType> local_quantizer;
  LinearQuantizer<D, T, Q, DeviceType> global_quantizer;

  // 1D ROI tolerance map: all levels stored consecutively
  std::vector<double> roi_tolerance_map;

  // Device-resident copy of roi_tolerance_map, uploaded once in
  // SetBlockTolerances and reused by every Quantize/Dequantize call.
  Array<1, double, DeviceType> device_roi_tolerance_map;

  // Offset for each level in the 1D tolerance map
  std::vector<SIZE> level_offsets;

  // Number of blocks at each level
  std::vector<SIZE> level_block_counts;

  // Fine and coarse shapes for each local level
  std::vector<std::vector<SIZE>> fine_shapes;
  std::vector<std::vector<SIZE>> coarse_shapes;

  std::vector<double> initial_block_tolerances;
};

} // namespace mgard_x_160

#endif
