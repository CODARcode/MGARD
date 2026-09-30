#ifndef _MDR_COMPOSED_RECONSTRUCTOR_HPP
#define _MDR_COMPOSED_RECONSTRUCTOR_HPP

#include "../../RuntimeX/RuntimeX.h"

#include "../../DataRefactoring/MultiDimension/CopyND/AddND.hpp"
#include "../BitplaneEncoder/BitplaneEncoder.hpp"
#include "../Decomposer/Decomposer.hpp"
#include "../ErrorCollector/ErrorCollector.hpp"
#include "../ErrorEstimator/ErrorEstimator.hpp"
#include "../Interleaver/Interleaver.hpp"
#include "../LosslessCompressor/LevelCompressor.hpp"
// #include "../RefactorUtils.hpp"
#include "../Retriever/Retriever.hpp"
#include "../SizeInterpreter/SizeInterpreter.hpp"
#include "ReconstructorInterface.hpp"
#include <stdio.h>
// #include "../DataStructures/MDRData.hpp"

namespace mgard_x {
namespace MDR {
// a decomposition-based scientific data reconstructor: inverse operator of
// composed refactor
template <DIM D, typename T_data, typename DeviceType, bool ControlL2 = false,
          typename Basis = Hierarchical, bool NegaBinary = false>
class ComposedReconstructor
    : public concepts::ReconstructorInterface<D, T_data, DeviceType> {
public:
  using HierarchyType = Hierarchy<D, T_data, DeviceType>;
  using T_bitplane = uint32_t;
  using T_error = double;
  using Decomposer = MGARDDecomposer<D, T_data, Basis, DeviceType>;
  using Interleaver = DirectInterleaver<D, T_data, DeviceType>;

  constexpr static bool ProfileBPEncoder = false;
  // using Encoder = GroupedBPEncoder<D, T_data, T_bitplane, T_error,
  //                                ControlL2, DeviceType>;
  // using Encoder = BPEncoderLocalityBlock<D, T_data, T_bitplane, T_error,
  // NegaBinary,
  //                                ControlL2, DeviceType>;
  using Encoder = BPEncoderRegisterBlock<D, T_data, T_bitplane, T_error,
                                         NegaBinary, ControlL2, DeviceType>;
  // using Encoder = BPEncoderRegisterShift<D, T_data, T_bitplane, T_error,
  // NegaBinary,
  //                              ControlL2, DeviceType>;
  // using Encoder = BPEncoderRegisterBallot<D, T_data, T_bitplane, T_error,
  // NegaBinary,
  //                              ControlL2, DeviceType>;
  // using Encoder = BPEncoderRegisterReduceAll<D, T_data, T_bitplane, T_error,
  // NegaBinary,
  //                              ControlL2, DeviceType>;
  // using Encoder = BPEncoderRegisterMatchAny<D, T_data, T_bitplane, T_error,
  // NegaBinary,
  //                              ControlL2, DeviceType>;

  // using Compressor = DefaultLevelCompressor<T_bitplane, HUFFMAN, DeviceType>;
  // using Compressor = DefaultLevelCompressor<T_bitplane, RLE, DeviceType>;
  using Compressor = HybridLevelCompressor<T_bitplane, DeviceType>;
  // using Compressor = NullLevelCompressor<T_bitplane, DeviceType>;

  static constexpr SIZE BATCH_SIZE = sizeof(T_bitplane) * 8;
  static constexpr SIZE MAX_BITPLANES = sizeof(T_data) * 8;

  ComposedReconstructor() : initialized(false) {}
  ComposedReconstructor(Hierarchy<D, T_data, DeviceType> &hierarchy,
                        Config config) {
    Adapt(hierarchy, config, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
  }

  ~ComposedReconstructor() {}

  void Adapt(Hierarchy<D, T_data, DeviceType> &hierarchy, Config config,
             int queue_idx) {
    this->initialized = true;
    this->hierarchy = &hierarchy;
    decomposer.Adapt(hierarchy, config, queue_idx);
    interleaver.Adapt(hierarchy, queue_idx);
    encoder.Adapt(hierarchy, queue_idx);
    // batched_encoder.Adapt(hierarchy, queue_idx);
    compressor.Adapt(Encoder::bitplane_length(
                         hierarchy.level_num_elems(hierarchy.l_target())),
                     Encoder::MAX_BITPLANES, config, queue_idx);

    prev_reconstructed = false;
    partial_reconsctructed_data.resize(
        hierarchy.level_shape(hierarchy.l_target()), queue_idx);
    interpolation_workspace.resize(hierarchy.level_shape(hierarchy.l_target()),
                                   queue_idx);

    level_data_array.resize(hierarchy.l_target() + 1);
    level_data_subarray.resize(hierarchy.l_target() + 1);
    level_num_elems.resize(hierarchy.l_target() + 1);
    exp.resize(hierarchy.l_target() + 1);
    for (int level_idx = 0; level_idx < hierarchy.l_target() + 1; level_idx++) {
      level_data_array[level_idx].resize(
          {round_up(hierarchy.level_num_elems(level_idx), BATCH_SIZE)},
          queue_idx);
      level_data_subarray[level_idx] =
          SubArray<1, T_data, DeviceType>(level_data_array[level_idx]);
      level_num_elems[level_idx] = hierarchy.level_num_elems(level_idx);
    }
    encoded_bitplanes_array.resize(hierarchy.l_target() + 1);
    encoded_bitplanes_subarray.resize(hierarchy.l_target() + 1);
    level_num_bitplanes.resize(hierarchy.l_target() + 1);
    level_signs_subarray.resize(hierarchy.l_target() + 1);
    abs_max_array.resize(hierarchy.l_target() + 1);
    for (int level_idx = 0; level_idx < hierarchy.l_target() + 1; level_idx++) {
      encoded_bitplanes_array[level_idx].resize(
          {(SIZE)Encoder::MAX_BITPLANES,
           encoder.bitplane_length(hierarchy.level_num_elems(level_idx))},
          queue_idx);
      encoded_bitplanes_subarray[level_idx] =
          SubArray<2, T_bitplane, DeviceType>(
              encoded_bitplanes_array[level_idx]);
      abs_max_array[level_idx].resize({1}, queue_idx);
      abs_max_array[level_idx].hostAllocate(false, queue_idx);
    }
  }

  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape,
                                        Config config) {
    Hierarchy<D, T_data, DeviceType> hierarchy;
    Array<1, T_data, DeviceType> array_with_pitch({1});
    size_t pitch_size = array_with_pitch.ld(0) * sizeof(T_data);
    size_t size = 0;
    size += hierarchy.EstimateMemoryFootprint(shape);
    size_t partial_data_size = 1;
    for (DIM d = 0; d < D; d++) {
      if (d == D - 1) {
        partial_data_size *=
            roundup((size_t)(shape[d]) * sizeof(T_data), pitch_size);
      } else {
        partial_data_size *= shape[d];
      }
    }
    size += partial_data_size * 2; // including interpolation workspace
    for (int level_idx = 0; level_idx < hierarchy.l_target() + 1; level_idx++) {
      size += round_up(hierarchy.level_num_elems(level_idx), BATCH_SIZE) *
              sizeof(T_data);
    }

    for (int level_idx = 0; level_idx < hierarchy.l_target() + 1; level_idx++) {
      size += Encoder::MAX_BITPLANES *
              Encoder::bitplane_length(hierarchy.level_num_elems(level_idx)) *
              sizeof(T_bitplane);
    }

    SIZE max_n = Encoder::bitplane_length(
        hierarchy.level_num_elems(hierarchy.l_target()));

    size += (Encoder::MAX_BITPLANES + 1) * sizeof(T_error);
    size += Decomposer::EstimateMemoryFootprint(shape);
    size += Interleaver::EstimateMemoryFootprint(shape);
    size += Encoder::EstimateMemoryFootprint(shape);
    size += Compressor::EstimateMemoryFootprint(max_n, config);
    return size;
  }

  static std::vector<std::vector<SIZE>>
  EstimateMaxBitplaneSizes(Hierarchy<D, T_data, DeviceType> &hierarchy) {
    std::vector<std::vector<SIZE>> estimation;
    estimation.resize(hierarchy.l_target() + 1);
    for (int level_idx = 0; level_idx < hierarchy.l_target() + 1; level_idx++) {
      estimation[level_idx].resize(Encoder::MAX_BITPLANES);
      for (int bitplane_idx = 0; bitplane_idx < Encoder::MAX_BITPLANES;
           bitplane_idx++) {
        if (bitplane_idx % Compressor::num_merged_bitplanes == 0) {
          estimation[level_idx][bitplane_idx] =
              Encoder::bitplane_length(hierarchy.level_num_elems(level_idx)) *
              sizeof(T_bitplane) * Compressor::num_merged_bitplanes;
          // For Huffman-only model (metadata storage)
          estimation[level_idx][bitplane_idx] += 1e6;
        } else {
          estimation[level_idx][bitplane_idx] = 1;
        }
      }
    }
    return estimation;
  }

  void GenerateRequest(MDRMetadata &mdr_metadata) {
    mgard_x::Timer timer;
    timer.start();
    std::vector<std::vector<double>> level_abs_errors;
    uint8_t target_level = mdr_metadata.level_error_bounds.size() - 1;
    std::vector<std::vector<double>> level_errors =
        mdr_metadata.level_squared_errors;
    std::vector<SIZE> retrieve_sizes;
    if (mdr_metadata.requested_s == std::numeric_limits<double>::infinity()) {
      log::info("ErrorEstimator is base of MaxErrorEstimator, computing "
                "absolute error");
      MDR::MaxErrorCollector<T_data> collector =
          MDR::MaxErrorCollector<T_data>();
      for (int i = 0; i <= target_level; i++) {
        auto collected_error = collector.collect_level_error(
            NULL, 0, mdr_metadata.level_squared_errors[i].size(),
            mdr_metadata.level_error_bounds[i]);
        level_abs_errors.push_back(collected_error);
      }
      level_errors = level_abs_errors;

      if constexpr (std::is_same<Basis, Orthogonal>::value) {
        MaxErrorEstimatorOB<T_data> estimator(D);
        GreedyBasedSizeInterpreter interpreter(estimator);
        if (mdr_metadata.segmented) {
          retrieve_sizes = interpreter.interpret_retrieve_size(
              mdr_metadata.level_sizes, level_errors,
              mdr_metadata.requested_size, mdr_metadata.corresponding_error,
              mdr_metadata.requested_level_num_bitplanes);
        } else if (mdr_metadata.corresponding_error_return) {
          retrieve_sizes = interpreter.interpret_retrieve_size(
              mdr_metadata.level_sizes, level_errors,
              mdr_metadata.requested_tol, mdr_metadata.corresponding_error,
              mdr_metadata.requested_level_num_bitplanes);
        } else {
          retrieve_sizes = interpreter.interpret_retrieve_size(
              mdr_metadata.level_sizes, level_errors,
              mdr_metadata.requested_tol,
              mdr_metadata.requested_level_num_bitplanes);
        }
      } else if constexpr (std::is_same<Basis, Hierarchical>::value) {
        MaxErrorEstimatorHB<T_data> estimator;
        GreedyBasedSizeInterpreter interpreter(estimator);
        if (mdr_metadata.segmented) {
          retrieve_sizes = interpreter.interpret_retrieve_size(
              mdr_metadata.level_sizes, level_errors,
              mdr_metadata.requested_size, mdr_metadata.corresponding_error,
              mdr_metadata.requested_level_num_bitplanes);
        } else if (mdr_metadata.corresponding_error_return) {
          retrieve_sizes = interpreter.interpret_retrieve_size(
              mdr_metadata.level_sizes, level_errors,
              mdr_metadata.requested_tol, mdr_metadata.corresponding_error,
              mdr_metadata.requested_level_num_bitplanes);
        } else {
          retrieve_sizes = interpreter.interpret_retrieve_size(
              mdr_metadata.level_sizes, level_errors,
              mdr_metadata.requested_tol,
              mdr_metadata.requested_level_num_bitplanes);
        }
      }
      // SignExcludeGreedyBasedSizeInterpreter interpreter(estimator);
      // RoundRobinSizeInterpreter interpreter(estimator);
      // InorderSizeInterpreter interpreter(estimator);

    } else {
      log::info("ErrorEstimator is base of SquaredErrorEstimator, using level "
                "squared error directly");

      if constexpr (std::is_same<Basis, Orthogonal>::value) {
        using Estimator = SNormErrorEstimator<T_data>;
        Estimator estimator(D, hierarchy->l_target(), mdr_metadata.requested_s);
        using BinaryInterp = GreedyBasedSizeInterpreter<Estimator>;
        using NegaBinaryInterp =
            NegaBinaryGreedyBasedSizeInterpreter<Estimator>;
        using Interpreter =
            typename std::conditional<NegaBinary, NegaBinaryInterp,
                                      BinaryInterp>::type;
        Interpreter interpreter(estimator);
        retrieve_sizes = interpreter.interpret_retrieve_size(
            mdr_metadata.level_sizes, level_errors,
            std::pow(mdr_metadata.requested_tol, 2),
            mdr_metadata.requested_level_num_bitplanes);
      } else if constexpr (std::is_same<Basis, Hierarchical>::value) {
        using Estimator = L2ErrorEstimator_HB<T_data>;
        Estimator estimator(D, hierarchy->l_target());
        using BinaryInterp = GreedyBasedSizeInterpreter<Estimator>;
        using NegaBinaryInterp =
            NegaBinaryGreedyBasedSizeInterpreter<Estimator>;
        using Interpreter =
            typename std::conditional<NegaBinary, NegaBinaryInterp,
                                      BinaryInterp>::type;
        Interpreter interpreter(estimator);
        retrieve_sizes = interpreter.interpret_retrieve_size(
            mdr_metadata.level_sizes, level_errors,
            std::pow(mdr_metadata.requested_tol, 2),
            mdr_metadata.requested_level_num_bitplanes);
      }
      // using BinaryInterpreter = InorderSizeInterpreter<Estimator>;
      // SignExcludeGreedyBasedSizeInterpreter interpreter(estimator);
      // NegaBinaryGreedyBasedSizeInterpreter interpreter(estimator);
    }

    for (uint8_t &n : mdr_metadata.requested_level_num_bitplanes) {
      // Ensure requested bitplanes is a multiple of num_merged_bitplanes
      // This ensure all each batch of merged bitplanes are used for
      // Reconstruction. Otherwise, unsed bitplanes will not be guaranteed
      // to be in memory in future reconstructions.
      // (a level that needs no bitplanes stays at 0).
      int m = Compressor::num_merged_bitplanes;
      n = (n + m - 1) / m * m;
    }
    timer.end();
    // timer.print("Preprocessing");
  }

  void InterpolateToLevel(Array<D, T_data, DeviceType> &reconstructed_data,
                          int prev_level, int curr_level, int queue_idx) {
    log::info("Interpoate from level " + std::to_string(prev_level) + " to" +
              " level " + std::to_string(curr_level));
    Timer timer;
    timer.start();
    interpolation_workspace.resize(reconstructed_data.shape());
    data_refactoring::multi_dimension::CopyND(SubArray(reconstructed_data),
                                              SubArray(interpolation_workspace),
                                              queue_idx);
    reconstructed_data.resize(hierarchy->level_shape(curr_level));
    reconstructed_data.memset(0, queue_idx);
    data_refactoring::multi_dimension::CopyND(SubArray(interpolation_workspace),
                                              SubArray(reconstructed_data),
                                              queue_idx);
    decomposer.recompose(reconstructed_data, prev_level, curr_level, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    timer.end();
    timer.print("Interpolation");
  }

  void LoadMetadata(MDRMetadata &mdr_metadata, MDRData<DeviceType> &mdr_data,
                    int queue_idx) {
    // All levels, not just up to CurrFinalLevel(): levels with no bitplanes
    // must get level_num_bitplanes = 0 rather than keep a value from a
    // previous use of this reconstructor (ProgressiveReconstruct visits all).
    for (int level_idx = 0; level_idx <= hierarchy->l_target(); level_idx++) {
      level_num_bitplanes[level_idx] =
          mdr_metadata.loaded_level_num_bitplanes[level_idx] -
          mdr_metadata.prev_used_level_num_bitplanes[level_idx];
      level_signs_subarray[level_idx] =
          SubArray<1, bool, DeviceType>(mdr_data.level_signs[level_idx]);

      T_data abs_max = (T_data)mdr_metadata.level_error_bounds[level_idx];
      MemoryManager<DeviceType>::Copy1D(abs_max_array[level_idx].data(),
                                        &abs_max, 1, queue_idx);
    }
  }

  void Decompress(MDRMetadata &mdr_metadata, MDRData<DeviceType> &mdr_data,
                  int queue_idx) {

    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }
    SIZE decompressed_size = 0;
    for (int level_idx = 0; level_idx <= mdr_metadata.CurrFinalLevel();
         level_idx++) {
      // Number of bitplanes need to be retrieved in addition to previously
      // already retrieved bitplanes
      SIZE num_bitplanes =
          mdr_metadata.loaded_level_num_bitplanes[level_idx] -
          mdr_metadata.prev_used_level_num_bitplanes[level_idx];
      // Decompress bitplanes: compressed_bitplanes[level_idx] -->
      // encoded_bitplanes
      compressor.decompress_level(
          mdr_data.compressed_bitplanes[level_idx],
          encoded_bitplanes_subarray[level_idx],
          mdr_metadata.prev_used_level_num_bitplanes[level_idx],
          level_num_bitplanes[level_idx], level_idx, queue_idx);
      decompressed_size += encoded_bitplanes_subarray[level_idx].shape(1) *
                           num_bitplanes * sizeof(T_bitplane);
    }
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      // timer.print("Lossless", hierarchy->total_num_elems() * sizeof(T_data));
      timer.print("Lossless", decompressed_size);
      timer.clear();
      timer.start();
    }
  }
  void ProgressiveReconstruct(MDRMetadata &mdr_metadata,
                              MDRData<DeviceType> &mdr_data,
                              bool adaptive_resolution,
                              Array<D, T_data, DeviceType> &reconstructed_data,
                              int queue_idx) {

    mdr_data.VerifyLoadedBitplans(mdr_metadata);

    Timer timer, timer_all;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer_all.start();
    }
    // Decompress and decode bitplanes of each level
    int prev_final_level = mdr_metadata.PrevFinalLevel();
    int curr_final_level = mdr_metadata.CurrFinalLevel();
    // log::info("Prev Final level: " + std::to_string(prev_final_level));
    // log::info("Curr Final level: " + std::to_string(curr_final_level));

    if (!adaptive_resolution) {
      curr_final_level = hierarchy->l_target();
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    // for (int i = 1; i <= 32; i++) {
    // std::cout << "[";

    for (int level_idx = 0; level_idx <= curr_final_level; level_idx++) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      Timer timer_iter;
      if constexpr (ProfileBPEncoder) {
        DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
        timer_iter.start();
      }
      encoder.progressive_decode(
          level_data_subarray[level_idx].shape(0),
          mdr_metadata.prev_used_level_num_bitplanes[level_idx],
          level_num_bitplanes[level_idx], SubArray(abs_max_array[level_idx]),
          encoded_bitplanes_subarray[level_idx],
          level_signs_subarray[level_idx], level_idx,
          level_data_subarray[level_idx], queue_idx);
      if constexpr (ProfileBPEncoder) {
        DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
        timer_iter.end();
        timer_iter.print(
            "Decoding level (# of coefficients: " +
                std::to_string(level_data_subarray[level_idx].shape(0)) + ")",
            level_data_subarray[level_idx].shape(0) * sizeof(T_data), true);
      }
      // if (level_idx < curr_final_level) {
      //   printf("%.6f, ", timer_iter.get());
      // } else {
      //   printf("%.6f", timer_iter.get());
      // }
    }
    // std::cout << "],\n";
    // }

    for (int level_idx = 0; level_idx <= curr_final_level; level_idx++) {
      if (level_num_bitplanes[level_idx] == 0) {
        level_data_array[level_idx].memset(0, queue_idx);
      }
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Decoding", hierarchy->total_num_elems() * sizeof(T_data));
      timer.clear();
      timer.start();
    }

    partial_reconsctructed_data.resize(
        hierarchy->level_shape(curr_final_level));

    // Put decoded coefficients back to reordered layout
    interleaver.reposition(
        level_data_subarray,
        SubArray<D, T_data, DeviceType>(partial_reconsctructed_data),
        curr_final_level, queue_idx);

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Reposition", hierarchy->total_num_elems() * sizeof(T_data));
      timer.clear();
    }

    decomposer.recompose(partial_reconsctructed_data, 0, curr_final_level,
                         queue_idx);

    if (adaptive_resolution) {
      // Interpolate previous reconstructed data to the same resolution
      InterpolateToLevel(reconstructed_data, prev_final_level, curr_final_level,
                         queue_idx);
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }
    SubArray partial_reconstructed_subarray(partial_reconsctructed_data);
    SubArray reconstructed_subarray(reconstructed_data);
    data_refactoring::multi_dimension::AddND(partial_reconstructed_subarray,
                                             reconstructed_subarray, queue_idx);

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("AddND", hierarchy->total_num_elems() * sizeof(T_data));
      timer.clear();
    }
    mdr_metadata.DoneReconstruct();
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer_all.end();
      timer_all.print("Decoding + Reposition + Recompose",
                      hierarchy->total_num_elems() * sizeof(T_data));
      timer_all.clear();
    }
  }

  const std::vector<SIZE> &get_dimensions() { return dimensions; }

  void print() const {
    std::cout << "Composed reconstructor with the following components."
              << std::endl;
    std::cout << "Decomposer: ";
    decomposer.print();
    std::cout << "Interleaver: ";
    interleaver.print();
    std::cout << "Encoder: ";
    encoder.print();
  }

  bool initialized;

private:
  Hierarchy<D, T_data, DeviceType> *hierarchy;
  Decomposer decomposer;
  Interleaver interleaver;
  Encoder encoder;
  Compressor compressor;

  Array<D, T_data, DeviceType> partial_reconsctructed_data;
  Array<D, T_data, DeviceType> interpolation_workspace;
  std::vector<Array<1, T_data, DeviceType>> level_data_array;
  std::vector<SubArray<1, T_data, DeviceType>> level_data_subarray;
  std::vector<Array<2, T_bitplane, DeviceType>> encoded_bitplanes_array;
  std::vector<SubArray<2, T_bitplane, DeviceType>> encoded_bitplanes_subarray;
  std::vector<SubArray<1, bool, DeviceType>> level_signs_subarray;
  std::vector<Array<1, T_data, DeviceType>> abs_max_array;

  bool prev_reconstructed;

  std::vector<SIZE> level_num_elems;
  std::vector<int32_t> exp;

  std::vector<T_data> data;
  std::vector<SIZE> dimensions;
  std::vector<T_data> level_error_bounds;
  std::vector<uint8_t> level_num_bitplanes;
  std::vector<SIZE> level_num;
  std::vector<std::vector<double>> level_squared_errors;
};
} // namespace MDR
} // namespace mgard_x
#endif
