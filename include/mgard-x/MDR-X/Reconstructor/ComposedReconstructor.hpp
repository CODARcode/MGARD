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
// #include "../DataStructures/MDRData.hpp"

namespace mgard_x {
namespace MDR {
// a decomposition-based scientific data reconstructor: inverse operator of
// composed refactor
template <DIM D, typename T_data, typename DeviceType>
class ComposedReconstructor
    : public concepts::ReconstructorInterface<D, T_data, DeviceType> {
public:
  using HierarchyType = Hierarchy<D, T_data, DeviceType>;
  using T_bitplane = uint32_t;
  using T_error = double;
  using Decomposer = MGARDOrthoganalDecomposer<D, T_data, DeviceType>;
  using Interleaver = DirectInterleaver<D, T_data, DeviceType>;
  using Encoder = GroupedBPEncoder<D, T_data, T_bitplane, T_error, DeviceType>;
  using Compressor = DefaultLevelCompressor<T_bitplane, DeviceType>;
  // using Compressor = NullLevelCompressor<T_bitplane, DeviceType>;
  using Retriever = ConcatLevelFileRetriever;

  ComposedReconstructor(Hierarchy<D, T_data, DeviceType> hierarchy,
                        Config config)
      : hierarchy(hierarchy), decomposer(this->hierarchy),
        interleaver(this->hierarchy), encoder(this->hierarchy),
        compressor(Encoder::buffer_size(
                       hierarchy.level_num_elems(hierarchy.l_target())),
                   config),
        total_num_bitplanes(config.total_num_bitplanes),
        retriever(std::string(""), std::vector<std::string>()) {
    prev_reconstructed = false;
    partial_reconsctructed_data = Array<D, T_data, DeviceType>(
        hierarchy.level_shape(hierarchy.l_target()));
    interpolation_workspace = Array<D, T_data, DeviceType>(
        hierarchy.level_shape(hierarchy.l_target()));
    levels_array = new Array<1, T_data, DeviceType>[hierarchy.l_target() + 1];
    levels_data = new SubArray<1, T_data, DeviceType>[hierarchy.l_target() + 1];
    for (int level_idx = 0; level_idx < hierarchy.l_target() + 1; level_idx++) {
      levels_array[level_idx] =
          Array<1, T_data, DeviceType>({hierarchy.level_num_elems(level_idx)});
      levels_data[level_idx] =
          SubArray<1, T_data, DeviceType>(levels_array[level_idx]);
    }
    encoded_bitplanes_array.resize(hierarchy.l_target() + 1);
    for (int level_idx = 0; level_idx < hierarchy.l_target() + 1; level_idx++) {
      encoded_bitplanes_array[level_idx] = Array<2, T_bitplane, DeviceType>(
          {(SIZE)total_num_bitplanes,
           encoder.buffer_size(hierarchy.level_num_elems(level_idx))});
    }
  }

  ~ComposedReconstructor() {
    delete[] levels_array;
    delete[] levels_data;
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
      size += hierarchy.level_num_elems(level_idx) * sizeof(T_data);
    }

    for (int level_idx = 0; level_idx < hierarchy.l_target() + 1; level_idx++) {
      size += config.total_num_bitplanes *
              Encoder::buffer_size(hierarchy.level_num_elems(level_idx)) *
              sizeof(T_bitplane);
    }

    SIZE max_n =
        Encoder::buffer_size(hierarchy.level_num_elems(hierarchy.l_target()));

    size += (config.total_num_bitplanes + 1) * sizeof(T_error);
    size += Decomposer::EstimateMemoryFootprint(shape);
    size += Interleaver::EstimateMemoryFootprint(shape);
    size += Encoder::EstimateMemoryFootprint(shape);
    size += Compressor::EstimateMemoryFootprint(max_n, config);
    return size;
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

      MaxErrorEstimatorOB<T_data> estimator(D);
      SignExcludeGreedyBasedSizeInterpreter interpreter(estimator);
      // RoundRobinSizeInterpreter interpreter(estimator);
      // InorderSizeInterpreter interpreter(estimator);
      retrieve_sizes = interpreter.interpret_retrieve_size(
          mdr_metadata.level_sizes, level_errors, mdr_metadata.requested_tol,
          mdr_metadata.requested_level_num_bitplanes);
    } else {
      log::info("ErrorEstimator is base of SquaredErrorEstimator, using level "
                "squared error directly");
      SNormErrorEstimator<T_data> estimator(D, hierarchy.l_target(),
                                            mdr_metadata.requested_s);
      // InorderSizeInterpreter interpreter(estimator);
      SignExcludeGreedyBasedSizeInterpreter interpreter(estimator);
      // NegaBinaryGreedyBasedSizeInterpreter interpreter(estimator);
      retrieve_sizes = interpreter.interpret_retrieve_size(
          mdr_metadata.level_sizes, level_errors, mdr_metadata.requested_tol,
          mdr_metadata.requested_level_num_bitplanes);
    }
    timer.end();
    timer.print("Preprocessing");
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
    reconstructed_data.resize(hierarchy.level_shape(curr_level));
    reconstructed_data.memset(0, queue_idx);
    data_refactoring::multi_dimension::CopyND(SubArray(interpolation_workspace),
                                              SubArray(reconstructed_data),
                                              queue_idx);
    decomposer.recompose(reconstructed_data, prev_level, curr_level, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    timer.end();
    timer.print("Interpolation");
  }

  void ProgressiveReconstruct(MDRMetadata &mdr_metadata,
                              MDRData<DeviceType> &mdr_data,
                              bool adaptive_resolution,
                              Array<D, T_data, DeviceType> &reconstructed_data,
                              int queue_idx) {

    mdr_data.VerifyLoadedBitplans(mdr_metadata);

    Timer timer;
    // Decompress and decode bitplanes of each level
    int prev_final_level = mdr_metadata.PrevFinalLevel();
    int curr_final_level = mdr_metadata.CurrFinalLevel();
    log::info("Prev Final level: " + std::to_string(prev_final_level));
    log::info("Curr Final level: " + std::to_string(curr_final_level));

    if (!adaptive_resolution) {
      curr_final_level = hierarchy.l_target();
    }

    for (int level_idx = 0; level_idx <= curr_final_level; level_idx++) {
      timer.start();
      // Number of bitplanes need to be retrieved in addition to previously
      // already retrieved bitplanes
      SIZE num_bitplanes =
          mdr_metadata.loaded_level_num_bitplanes[level_idx] -
          mdr_metadata.prev_used_level_num_bitplanes[level_idx];
      // Decompress bitplanes: compressed_bitplanes[level_idx] -->
      // encoded_bitplanes
      compressor.decompress_level(
          mdr_metadata.level_sizes[level_idx],
          mdr_data.compressed_bitplanes[level_idx],
          encoded_bitplanes_array[level_idx],
          mdr_metadata.prev_used_level_num_bitplanes[level_idx], num_bitplanes,
          queue_idx);
      timer.end();
      timer.print("Lossless");
      timer.start();

      int level_exp = 0;
      frexp(mdr_metadata.level_error_bounds[level_idx], &level_exp);
      encoder.progressive_decode(
          hierarchy.level_num_elems(level_idx),
          mdr_metadata.prev_used_level_num_bitplanes[level_idx], num_bitplanes,
          level_exp,
          SubArray<2, T_bitplane, DeviceType>(
              encoded_bitplanes_array[level_idx]),
          SubArray(mdr_data.level_signs[level_idx]), level_idx,
          levels_data[level_idx], queue_idx);
      if (num_bitplanes == 0) {
        levels_array[level_idx].memset(0);
      }
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      compressor.decompress_release();
      timer.end();
      timer.print("Decoding");
    }

    partial_reconsctructed_data.resize(hierarchy.level_shape(curr_final_level));

    timer.start();
    // Put decoded coefficients back to reordered layout
    interleaver.reposition(
        levels_data,
        SubArray<D, T_data, DeviceType>(partial_reconsctructed_data),
        curr_final_level, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    timer.end();
    timer.print("Reposition");

    timer.start();
    decomposer.recompose(partial_reconsctructed_data, 0, curr_final_level,
                         queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    timer.end();
    timer.print("Recomposing");
    if (adaptive_resolution) {
      // Interpolate previous reconstructed data to the same resolution
      InterpolateToLevel(reconstructed_data, prev_final_level, curr_final_level,
                         queue_idx);
    }

    SubArray partial_reconstructed_subarray(partial_reconsctructed_data);
    SubArray reconstructed_subarray(reconstructed_data);
    data_refactoring::multi_dimension::AddND(partial_reconstructed_subarray,
                                             reconstructed_subarray, queue_idx);
    mdr_metadata.DoneReconstruct();
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
    std::cout << "Retriever: ";
    retriever.print();
  }

private:
  Hierarchy<D, T_data, DeviceType> hierarchy;
  Decomposer decomposer;
  Interleaver interleaver;
  Encoder encoder;
  Compressor compressor;
  Retriever retriever;

  Array<D, T_data, DeviceType> partial_reconsctructed_data;
  Array<D, T_data, DeviceType> interpolation_workspace;
  Array<1, T_data, DeviceType> *levels_array = nullptr;
  SubArray<1, T_data, DeviceType> *levels_data = nullptr;
  std::vector<Array<2, T_bitplane, DeviceType>> encoded_bitplanes_array;
  SIZE total_num_bitplanes;

  bool prev_reconstructed;

  std::vector<T_data> data;
  std::vector<SIZE> dimensions;
  std::vector<T_data> level_error_bounds;
  std::vector<uint8_t> level_num_bitplanes;
  std::vector<std::vector<const uint8_t *>> level_components;
  std::vector<std::vector<SIZE>> level_sizes;
  std::vector<SIZE> level_num;
  std::vector<std::vector<double>> level_squared_errors;
};
} // namespace MDR
} // namespace mgard_x
#endif
