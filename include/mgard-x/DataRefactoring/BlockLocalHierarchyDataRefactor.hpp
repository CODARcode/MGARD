#include "../Hierarchy/HybridHierarchy.h"
#include "InCacheBlock/DataRefactoring.h"
#include "MultiDimension/DataRefactoring.h"

#ifndef MGARD_X_BLOCK_LOCAL_HIERARCHY_DATA_REFACTOR_HPP
#define MGARD_X_BLOCK_LOCAL_HIERARCHY_DATA_REFACTOR_HPP

namespace mgard_x {

namespace data_refactoring {

template <DIM D, typename T, typename DeviceType>
class BlockLocalHierarchyDataRefactor {
public:
  BlockLocalHierarchyDataRefactor() : initialized(false) {}

  // Removing all checks for L since processed in HybridHierarchyDataRefactor
  BlockLocalHierarchyDataRefactor(HybridHierarchy<D, T, DeviceType> &hierarchy,
                                  Config config)
      : initialized(true), hierarchy(&hierarchy), config(config) {
    hierarchy.check_config(config);
    this->L = config.num_local_refactoring_level;
    compute_local_ranges();
    w_array = Array<1, T, DeviceType>({DecomposedCoeffSize()});
    temp_coarest = Array<1, T, DeviceType>({coarse_num_elems[this->L - 1]});
    coarse_buffers.resize(2);
    coarse_buffers[0] = Array<D, T, DeviceType>(fine_shapes[0]);
    coarse_buffers[1] = Array<D, T, DeviceType>(fine_shapes[0]);
  }

  void Adapt(HybridHierarchy<D, T, DeviceType> &hierarchy, Config config,
             int queue_idx) {
    hierarchy.check_config(config);
    this->initialized = true;
    this->hierarchy = &hierarchy;
    this->config = config;
    this->L = config.num_local_refactoring_level;
    compute_local_ranges();

    w_array.resize({DecomposedCoeffSize()}, queue_idx);
    temp_coarest.resize({coarse_num_elems[this->L - 1]}, queue_idx);
    coarse_buffers.resize(2);
    coarse_buffers[0].resize(fine_shapes[0], queue_idx);
    coarse_buffers[1].resize(fine_shapes[0], queue_idx);
  }

  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape) {
    // We have 2 arrays for shape switch and another one for output coeff and
    // coarest
    size_t size = 3;
    for (DIM d = 0; d < shape.size(); d++) {
      int dim8 = ((shape[d] - 1) / 8 + 1) * 8;
      size *= dim8;
    }
    return size * sizeof(T);
  }

  size_t DecomposedDataSize() {
    SIZE decomposed_size = coarse_num_elems[this->L - 1];
    for (SIZE l = 0; l < this->L; l++) {
      decomposed_size += local_coeff_size[l];
    }
    return decomposed_size;
  }

  size_t DecomposedCoeffSize() {
    SIZE decomposed_coeff_size = 0;
    for (SIZE l = 0; l < this->L; l++) {
      decomposed_coeff_size += local_coeff_size[l];
    }
    return decomposed_coeff_size;
  }

  void compute_local_ranges() {
    coarse_shape = hierarchy->input_shape();
    fine_num_elems.clear();
    coarse_num_elems.clear();
    local_coeff_size.clear();
    coarse_shapes.clear();
    fine_shapes.clear();

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
      coarse_shapes.push_back(coarse_shape);
      fine_shapes.push_back(fine_shape);
    }
  }

  void Decompose(SubArray<D, T, DeviceType> data,
                 SubArray<1, T, DeviceType> output_decomposed, int queue_idx,
                 bool orthogonal_projection = true) {
    SubArray<D, T, DeviceType> fine(coarse_buffers[1]);
    SubArray<D, T, DeviceType> coarse;
    // Zero the level-0 fine buffer when the input needs padding up to the
    // next multiple of 8: CopyND below only writes the original extent, and
    // 8x8x8 blocks straddling the boundary would otherwise mix uninitialized
    // values into their coefficients. Deeper levels are unaffected (their
    // fine buffer is a coarse buffer that is fully memset before use).
    bool needs_padding = false;
    for (DIM d = 0; d < D; d++) {
      if (data.shape(d) != fine_shapes[0][d]) {
        needs_padding = true;
        break;
      }
    }
    if (needs_padding) {
      coarse_buffers[1].memset(0, queue_idx);
    }
    // CopyND follows the shape of 1st param
    multi_dimension::CopyND(data, fine, queue_idx);
    SubArray<1, T, DeviceType> decomposed_coeff(w_array);

    // Times the transform kernels only; the surrounding copies are covered by
    // the Hybrid Decomposition timer in HybridHierarchyDataRefactor.
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    // Will be reused between decompose and recompose
    accumulated_local_coeff_size = 0;
    for (SIZE l = 0; l < this->L; l++) {
      accumulated_local_coeff_size += local_coeff_size[l];
      // Think about a way to change this local_coeff
      SubArray<1, T, DeviceType> local_coeff(
          {local_coeff_size[l]},
          decomposed_coeff(decomposed_coeff.shape(0) -
                           accumulated_local_coeff_size));

      int buffer_idx = l % 2;
      coarse_buffers[buffer_idx].memset(0, queue_idx);
      coarse = SubArray<D, T, DeviceType>(coarse_shapes[l],
                                          coarse_buffers[buffer_idx].data());
      for (DIM d = 0; d < D; d++) {
        coarse.setLd(d, fine_shapes[0][d]);
      }
      coarse.project(D - 3, D - 2, D - 1);

      in_cache_block::decompose<D, T, DeviceType>(
          fine, coarse, local_coeff, orthogonal_projection, queue_idx);
      if (l < this->L - 1) {
        fine = SubArray<D, T, DeviceType>(fine_shapes[l + 1],
                                          coarse_buffers[buffer_idx].data());
        for (DIM d = 0; d < D; d++) {
          fine.setLd(d, fine_shapes[0][d]);
        }
        fine.project(D - 3, D - 2, D - 1);
      }
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Local Decomposition",
                  hierarchy->total_num_elems() * sizeof(T));
      timer.clear();
    }

    int final_buffer_id = (this->L - 1) % 2;
    SubArray<D, T, DeviceType> coarsest(coarse_shapes[this->L - 1],
                                        coarse_buffers[final_buffer_id].data());
    for (DIM d = 0; d < D; d++) {
      coarsest.setLd(d, fine_shapes[0][d]);
    }
    coarsest.project(D - 3, D - 2, D - 1);
    // log::info("Find read buffer idx: " + std::to_string(final_buffer_id));

    // Write the coarsest level directly into output_decomposed instead of
    // staging through temp_coarest: temp_coarest is unused between here and
    // the next Recompose() call, which repopulates it independently from
    // input_decomposed (see below), so the staging copy was pure overhead.
    SubArray<D, T, DeviceType> coarsest_out(coarse_shapes[this->L - 1],
                                            output_decomposed.data());
    for (DIM d = 0; d < D; d++) {
      coarsest_out.setLd(d, coarse_shapes[this->L - 1][d]);
    }
    coarsest_out.project(D - 3, D - 2, D - 1);
    multi_dimension::CopyND(coarsest, coarsest_out, queue_idx);

    SubArray<1, T, DeviceType> data_coeff({DecomposedCoeffSize()},
                                          output_decomposed.data() +
                                              coarse_num_elems[this->L - 1]);
    multi_dimension::CopyND(decomposed_coeff, data_coeff, queue_idx);

    // PrintSubarray("Temp in decompose:",SubArray(temp_coarest));
  }

  // Fused decompose+quantize. Runs the same per-level 8x8x8 decomposition as
  // Decompose(), but each level's coefficients are quantized in-kernel and
  // written directly to their final location in output_quantized, so the
  // T-typed coefficient staging (w_array) and the copies into
  // output_decomposed disappear. Level 0 reads straight from the (possibly
  // unpadded) input, and deeper levels read the previous coarse buffer at its
  // true extent — the fused kernel zero-fills out-of-range reads, replacing
  // the padding memsets. Only the coarsest level is emitted in T, compacted
  // at the front of output_decomposed for the global stage / coarsest
  // quantization.
  //
  // Level l's quantizer: level_quantizers[l] (reciprocal), or per-block
  // level_block_quantizers[l] in ROI mode (level_quantizers empty).
  template <typename Q>
  void DecomposeQuantize(
      SubArray<D, T, DeviceType> data,
      SubArray<1, T, DeviceType> output_decomposed,
      SubArray<1, Q, DeviceType> output_quantized,
      const std::vector<T> &level_quantizers,
      const std::vector<SubArray<1, T, DeviceType>> &level_block_quantizers,
      bool prep_huffman, SIZE dict_size, int queue_idx,
      bool orthogonal_projection = true) {
    bool use_block_quantizers = level_quantizers.empty();

    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    SubArray<D, T, DeviceType> fine = data;
    SIZE accumulated = 0;
    for (SIZE l = 0; l < this->L; l++) {
      accumulated += local_coeff_size[l];
      SubArray<1, Q, DeviceType> level_quantized(
          {local_coeff_size[l]},
          output_quantized(output_quantized.shape(0) - accumulated));

      int buffer_idx = l % 2;
      SubArray<D, T, DeviceType> coarse;
      if (l == this->L - 1) {
        // Last level: write the coarsest data compactly to its final
        // location instead of staging it in a padded buffer and copying.
        coarse = SubArray<D, T, DeviceType>(coarse_shapes[l],
                                            output_decomposed.data());
        for (DIM d = 0; d < D; d++) {
          coarse.setLd(d, coarse_shapes[l][d]);
        }
      } else {
        coarse = SubArray<D, T, DeviceType>(coarse_shapes[l],
                                            coarse_buffers[buffer_idx].data());
        for (DIM d = 0; d < D; d++) {
          coarse.setLd(d, fine_shapes[0][d]);
        }
      }
      coarse.project(D - 3, D - 2, D - 1);

      in_cache_block::decompose_quantize<D, T, Q, DeviceType>(
          fine, coarse, level_quantized,
          use_block_quantizers ? (T)0 : level_quantizers[l],
          use_block_quantizers ? level_block_quantizers[l]
                               : SubArray<1, T, DeviceType>(),
          use_block_quantizers, prep_huffman, dict_size, orthogonal_projection,
          queue_idx);

      if (l < this->L - 1) {
        // Next level reads the coarse output at its true extent; the fused
        // kernel's boundary handling supplies the zero padding.
        fine = coarse;
      }
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Local Decomposition+Quantization (fused)",
                  hierarchy->total_num_elems() * sizeof(T));
      timer.clear();
    }
  }

  // Fused dequantize+recompose. Runs the same per-level 8x8x8 recomposition
  // as Recompose(), but each level's coefficients are read from their final
  // location in input_quantized and dequantized in-kernel, so the T-typed
  // coefficient region of the decomposed array is never materialized. Only
  // the coarsest level is consumed in T from the front of input_decomposed
  // (produced by the global stage / coarsest dequantization). The staging
  // copies of the unfused path also disappear: the coarsest level is read
  // compactly in place (no temp_coarest restore), the ping-pong buffers are
  // not memset (every coarse value read at level l was written by level l+1,
  // or comes from input_decomposed), and the final level writes directly to
  // the unpadded output (the fused kernel bounds-checks its stores).
  //
  // Level l's dequantizer: level_dequantizers[l] (non-reciprocal), or
  // per-block level_block_dequantizers[l] in ROI mode (level_dequantizers
  // empty). Indexing matches DecomposeQuantize (level 0 = finest).
  template <typename Q>
  void RecomposeDequantize(
      SubArray<D, T, DeviceType> data,
      SubArray<1, T, DeviceType> input_decomposed,
      SubArray<1, Q, DeviceType> input_quantized,
      const std::vector<T> &level_dequantizers,
      const std::vector<SubArray<1, T, DeviceType>> &level_block_dequantizers,
      bool prep_huffman, SIZE dict_size, int queue_idx,
      bool orthogonal_projection = true) {
    bool use_block_quantizers = level_dequantizers.empty();

    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    // Coarsest level, read compactly in place from the front of
    // input_decomposed instead of staging through temp_coarest.
    SubArray<D, T, DeviceType> coarse(coarse_shapes[this->L - 1],
                                      input_decomposed.data());
    for (DIM d = 0; d < D; d++) {
      coarse.setLd(d, coarse_shapes[this->L - 1][d]);
    }
    coarse.project(D - 3, D - 2, D - 1);

    SIZE accumulated = DecomposedCoeffSize();
    for (SIZE l = 0; l < this->L; l++) {
      SIZE level_idx = this->L - l - 1;

      SubArray<1, Q, DeviceType> level_quantized(
          {local_coeff_size[level_idx]},
          input_quantized(input_quantized.shape(0) - accumulated));

      int buffer_idx = l % 2;
      SubArray<D, T, DeviceType> fine;
      if (level_idx == 0) {
        // Last level: write the reconstructed data directly to the unpadded
        // output instead of staging it in a padded buffer and copying.
        fine = data;
      } else {
        fine = SubArray<D, T, DeviceType>(fine_shapes[level_idx],
                                          coarse_buffers[buffer_idx].data());
        for (DIM d = 0; d < D; d++) {
          fine.setLd(d, fine_shapes[0][d]);
        }
        fine.project(D - 3, D - 2, D - 1);
      }

      in_cache_block::recompose_dequantize<D, T, Q, DeviceType>(
          fine, coarse, level_quantized,
          use_block_quantizers ? (T)0 : level_dequantizers[level_idx],
          use_block_quantizers ? level_block_dequantizers[level_idx]
                               : SubArray<1, T, DeviceType>(),
          use_block_quantizers, prep_huffman, dict_size, orthogonal_projection,
          queue_idx);

      if (l < this->L - 1) {
        coarse = SubArray<D, T, DeviceType>(coarse_shapes[level_idx - 1],
                                            coarse_buffers[buffer_idx].data());
        for (DIM d = 0; d < D; d++) {
          coarse.setLd(d, fine_shapes[0][d]);
        }
        coarse.project(D - 3, D - 2, D - 1);
      }
      accumulated -= local_coeff_size[level_idx];
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Local Recomposition+Dequantization (fused)",
                  hierarchy->total_num_elems() * sizeof(T));
      timer.clear();
    }
  }

  void Recompose(SubArray<D, T, DeviceType> data,
                 SubArray<1, T, DeviceType> input_decomposed, int queue_idx,
                 bool orthogonal_projection = true) {
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    // Initialize accumulated_local_coeff_size so that Recompose works correctly
    // regardless of whether Decompose was called first (e.g., standalone
    // decompress).
    accumulated_local_coeff_size = DecomposedCoeffSize();

    // Restore temp_coarest from input_decomposed (the first
    // coarse_num_elems[L-1] elements). This is critical for standalone
    // decompression where Decompose was never called and temp_coarest was never
    // populated. After global Recompose,
    // input_decomposed[0..coarse_num_elems[L-1]-1] holds the correctly
    // reconstructed coarsest values, which we must use here instead of
    // stale/zero temp_coarest.
    multi_dimension::CopyND(
        SubArray<1, T, DeviceType>({coarse_num_elems[this->L - 1]},
                                   input_decomposed.data()),
        SubArray(temp_coarest), queue_idx);

    coarse_buffers[0].memset(0, queue_idx);
    coarse_buffers[1].memset(0, queue_idx);

    SubArray<D, T, DeviceType> coarse(coarse_shapes[this->L - 1],
                                      temp_coarest.data());

    for (SIZE l = 0; l < this->L; l++) {
      SIZE level_idx = this->L - l - 1;

      SubArray<1, T, DeviceType> local_coeff(
          {local_coeff_size[level_idx]},
          input_decomposed(input_decomposed.shape(0) -
                           accumulated_local_coeff_size));

      int buffer_idx = l % 2;

      SubArray<D, T, DeviceType> fine(fine_shapes[level_idx],
                                      coarse_buffers[buffer_idx].data());
      for (DIM d = 0; d < D; d++) {
        fine.setLd(d, fine_shapes[0][d]);
      }
      fine.project(D - 3, D - 2, D - 1);
      // log::info("Buffer idx for fine buffer: " +
      // std::to_string(buffer_idx));

      in_cache_block::recompose<D, T, DeviceType>(
          fine, coarse, local_coeff, orthogonal_projection, queue_idx);

      if (l < this->L - 1) {
        coarse = SubArray<D, T, DeviceType>(coarse_shapes[level_idx - 1],
                                            coarse_buffers[buffer_idx].data());
        for (DIM d = 0; d < D; d++) {
          coarse.setLd(d, fine_shapes[0][d]);
        }
        coarse.project(D - 3, D - 2, D - 1);
      }
      accumulated_local_coeff_size -= local_coeff_size[level_idx];
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Local Recomposition",
                  hierarchy->total_num_elems() * sizeof(T));
      timer.clear();
    }

    // copy back, using ND
    SubArray<D, T, DeviceType> src(
        hierarchy->level_shape(hierarchy->l_target()),
        coarse_buffers[(this->L - 1) % 2].data());

    for (DIM d = 0; d < D; d++) {
      src.setLd(d, fine_shapes[0][d]);
    }
    src.project(D - 3, D - 2, D - 1);

    SubArray<D, T, DeviceType> dst(
        hierarchy->level_shape(hierarchy->l_target()), data.data());

    multi_dimension::CopyND(src, dst, queue_idx);
  }

  std::vector<SIZE> coarse_shape;
  SIZE accumulated_local_coeff_size = 0;
  bool initialized;
  SIZE L;
  HybridHierarchy<D, T, DeviceType> *hierarchy;
  Config config;

  std::vector<SIZE> fine_num_elems;
  std::vector<SIZE> coarse_num_elems;
  std::vector<SIZE> local_coeff_size;
  std::vector<std::vector<SIZE>> coarse_shapes;
  std::vector<std::vector<SIZE>> fine_shapes;
  std::vector<SIZE> original_input_shape;
  std::vector<SIZE> padded_input_shape;

  std::vector<Array<D, T, DeviceType>> coarse_buffers;

  Array<1, T, DeviceType> w_array;
  Array<1, T, DeviceType> temp_coarest;
};

} // namespace data_refactoring
} // namespace mgard_x

#endif
