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
#include "../RefactorUtils.hpp"
#include "../Retriever/Retriever.hpp"
#include "../SizeInterpreter/SizeInterpreter.hpp"
#include "ReconstructorInterface.hpp"
#include "../DataStructures/MDRData.hpp"

namespace mgard_x {
namespace MDR {
// a decomposition-based scientific data reconstructor: inverse operator of
// composed refactor
template <DIM D, typename T_data, typename DeviceType>
class ComposedReconstructor
    : public concepts::ReconstructorInterface<D, T_data, DeviceType> {

using T_bitplane = uint32_t;
using T_error = double;
using Decomposer = MGARDOrthoganalDecomposer<D, T_data, DeviceType>;
using Interleaver = DirectInterleaver<D, T_data, DeviceType>;
using Encoder = GroupedBPEncoder<D, T_data, T_bitplane, T_error, DeviceType>;
using Compressor = DefaultLevelCompressor<T_bitplane, DeviceType>;
using Retriever = ConcatLevelFileRetriever;

public:
  ComposedReconstructor(Hierarchy<D, T_data, DeviceType> hierarchy,
                        Config config,
                        std::string metadata_file, std::vector<std::string> files
                        )
      : hierarchy(hierarchy), decomposer(hierarchy), interleaver(hierarchy),
        encoder(hierarchy), compressor(hierarchy.total_num_elems()/8, config.huff_dict_size, config.huff_block_size, 1.0),
        total_num_bitplanes(config.total_num_bitplanes), retriever(metadata_file, files) {
    prev_reconstructed = false;
    partial_reconsctructed_data = Array<D, T_data, DeviceType>(hierarchy.level_shape(hierarchy.l_target()));
  }

  void GenerateRequest(MDRMetaData<D, T_data, DeviceType> &mdr_metadata, double tolerance, double s) {
    mgard_x::Timer timer;
    timer.start();
    std::vector<std::vector<double>> level_abs_errors;
    uint8_t target_level = mdr_metadata.level_error_bounds.size() - 1;
    std::vector<std::vector<double>> level_errors = mdr_metadata.level_squared_errors;
    std::vector<SIZE> retrieve_sizes;
    if (s == std::numeric_limits<double>::infinity()) {
      std::cout << "ErrorEstimator is base of MaxErrorEstimator, computing "
                   "absolute error"
                << std::endl;
      MDR::MaxErrorCollector<T_data> collector = MDR::MaxErrorCollector<T_data>();
      for (int i = 0; i <= target_level; i++) {
        auto collected_error = collector.collect_level_error(
            NULL, 0, mdr_metadata.level_squared_errors[i].size(), mdr_metadata.level_error_bounds[i]);
        level_abs_errors.push_back(collected_error);
      }
      level_errors = level_abs_errors;

      MaxErrorEstimatorOB<T_data> estimator(D);
      SignExcludeGreedyBasedSizeInterpreter interpreter(estimator);
      // RoundRobinSizeInterpreter interpreter(estimator);
      // InorderSizeInterpreter interpreter(estimator);
      retrieve_sizes = interpreter.interpret_retrieve_size(
        mdr_metadata.level_sizes, level_errors, tolerance, mdr_metadata.requested_level_num_bitplanes);
    } else {
      std::cout << "ErrorEstimator is base of SquaredErrorEstimator, using "
                   "level squared error directly"
                << std::endl;
      SNormErrorEstimator<T_data> estimator(D, hierarchy.l_target(), s);
      SignExcludeGreedyBasedSizeInterpreter interpreter(estimator);
      // NegaBinaryGreedyBasedSizeInterpreter interpreter(estimator);      
      retrieve_sizes = interpreter.interpret_retrieve_size(
        mdr_metadata.level_sizes, level_errors, tolerance, mdr_metadata.requested_level_num_bitplanes);
    }
    timer.end();
    timer.print("Preprocessing");
  }

  void ProgressiveReconstruct(MDRMetaData<D, T_data, DeviceType> &mdr_metadata, MDRData<D, T_data, DeviceType> &mdr_data, 
                              Array<D, T_data, DeviceType> &reconstructed_data, int queue_idx) {

    mdr_data.VerifyLoadedBitplans(mdr_metadata);
    // Calculating number of elements/coefficient in each level
    SIZE target_level = hierarchy.l_target();
    std::vector<SIZE> level_num_elems(target_level + 1);
    SIZE prev_num_elems = 0;
    for (int level_idx = 0; level_idx < target_level + 1; level_idx++) {
      SIZE curr_num_elems = 1;
      for (DIM d = 0; d < D; d++) {
        curr_num_elems *= hierarchy.level_shape(level_idx, d);
      }
      level_num_elems[level_idx] = curr_num_elems - prev_num_elems;
      prev_num_elems = curr_num_elems;
      // printf("%u ", level_num_elems[level_idx]);
    }
    // printf("\n");

    // Prepare output data buffer
    // Array<D, T_data, DeviceType> output_data(
    //     hierarchy.level_shape(target_level));
    reconstructed_data.resize(hierarchy.level_shape(target_level));

    MDR::Timer timer;
    timer.start();
    // Retrieve bitplanes of each level
    // std::vector<std::vector<Array<1, Byte, DeviceType>>> compressed_bitplanes;
    // for (int level_idx = 0; level_idx < target_level + 1; level_idx++) {
    //   compressed_bitplanes.push_back(std::vector<Array<1, Byte, DeviceType>>());
    //   int num_bitplanes =
    //       curr_level_num_bitplanes[level_idx] - prev_level_num_bitplanes[level_idx];
    //   for (int bitplane_idx = 0; bitplane_idx < num_bitplanes; bitplane_idx++) {
    //     SIZE size = level_sizes[level_idx][prev_level_num_bitplanes[level_idx] +
    //                                        bitplane_idx];
    //     // printf("level: %d, bitplane_idx: %d, size: %u\n", level_idx,
    //     // bitplane_idx, size);
    //     // Allocate space to hold compressed bitplanes
    //     compressed_bitplanes[level_idx].push_back(
    //         Array<1, Byte, DeviceType>({size}));
    //     // Copy compressed bitplanes to Array
    //     compressed_bitplanes[level_idx][bitplane_idx].load(
    //         level_components[level_idx][bitplane_idx]);
    //   }
    // }

    timer.end();
    timer.print("Reconstruct Preprocessing");

    Array<1, T_data, DeviceType> *levels_array =
        new Array<1, T_data, DeviceType>[target_level + 1];
    SubArray<1, T_data, DeviceType> *levels_data =
        new SubArray<1, T_data, DeviceType>[target_level + 1];

    // Decompress and decode bitplanes of each level
    for (int level_idx = 0; level_idx <= target_level; level_idx++) {
      timer.start();
      // Number of bitplanes need to be retrieved in addition to previously
      // already retrieved bitplanes
      SIZE num_bitplanes =
          mdr_metadata.loaded_level_num_bitplanes[level_idx] - mdr_metadata.prev_used_level_num_bitplanes[level_idx];
      // Allocate space for the bitplanes to be retrieved
      Array<2, T_bitplane, DeviceType> encoded_bitplanes(
          {num_bitplanes, encoder.buffer_size(level_num_elems[level_idx])});

      // Decompress bitplanes: compressed_bitplanes[level_idx] -->
      // encoded_bitplanes
      compressor.decompress_level(
          mdr_metadata.level_sizes[level_idx], mdr_data.compressed_bitplanes[level_idx],
          encoded_bitplanes, mdr_metadata.prev_used_level_num_bitplanes[level_idx],
          mdr_metadata.loaded_level_num_bitplanes[level_idx] - mdr_metadata.prev_used_level_num_bitplanes[level_idx], queue_idx);
      timer.end();
      timer.print("Lossless");
      timer.start();

      // Compute the exponent of max abs value of the current level needed for
      // decoding
      int level_exp = 0;
      frexp(mdr_metadata.level_error_bounds[level_idx], &level_exp);
      // Decode bitplanes: encoded_bitplanes --> levels_array[level_idx]
      levels_array[level_idx] =
          Array<1, T_data, DeviceType>({level_num_elems[level_idx]});
      levels_data[level_idx] = SubArray(levels_array[level_idx]);
      encoder.progressive_decode(
          level_num_elems[level_idx], mdr_metadata.prev_used_level_num_bitplanes[level_idx],
          num_bitplanes, level_exp,
          SubArray<2, T_bitplane, DeviceType>(encoded_bitplanes), level_idx,
          levels_data[level_idx], queue_idx);
      if (num_bitplanes == 0) {
        levels_array[level_idx].memset(0);
      }
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      compressor.decompress_release();
      timer.end();
      timer.print("Decoding");
    }

    timer.start();
    // Put decoded coefficients back to reordered layout
    interleaver.reposition(levels_data,
                           SubArray<D, T_data, DeviceType>(partial_reconsctructed_data),
                           target_level, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    timer.end();
    timer.print("Reposition");

    timer.start();
    // PrintSubarray("before recompose", SubArray(data_array));
    // Recompose data
    decomposer.recompose(partial_reconsctructed_data, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    timer.end();
    timer.print("Recomposing");

    SubArray partial_reconstructed_subarray(partial_reconsctructed_data);
    SubArray reconstructed_subarray(reconstructed_data);
    AddND(partial_reconstructed_subarray, reconstructed_subarray, queue_idx);

    mdr_metadata.DoneReconstruct();
  }


  // reconstruct data from encoded streams
  void reconstruct(double tolerance, double s,
                   Array<D, T_data, DeviceType> &reconstructed_data) {
    mgard_x::Timer timer;
    timer.start();

    std::vector<std::vector<double>> level_abs_errors;
    uint8_t target_level = level_error_bounds.size() - 1;
    std::vector<std::vector<double>> level_errors = level_squared_errors;
    std::vector<uint8_t> prev_level_num_bitplanes(level_num_bitplanes);
    std::vector<SIZE> retrieve_sizes;
    if (s == std::numeric_limits<double>::infinity()) {
      std::cout << "ErrorEstimator is base of MaxErrorEstimator, computing "
                   "absolute error"
                << std::endl;
      MDR::MaxErrorCollector<T_data> collector = MDR::MaxErrorCollector<T_data>();
      for (int i = 0; i <= target_level; i++) {
        auto collected_error = collector.collect_level_error(
            NULL, 0, level_squared_errors[i].size(), level_error_bounds[i]);
        level_abs_errors.push_back(collected_error);
      }
      level_errors = level_abs_errors;

      MaxErrorEstimatorOB<T_data> estimator(D);
      SignExcludeGreedyBasedSizeInterpreter interpreter(estimator);
      // RoundRobinSizeInterpreter interpreter(estimator);
      // InorderSizeInterpreter interpreter(estimator);
      retrieve_sizes = interpreter.interpret_retrieve_size(
        level_sizes, level_errors, tolerance, level_num_bitplanes);
    } else {
      std::cout << "ErrorEstimator is base of SquaredErrorEstimator, using "
                   "level squared error directly"
                << std::endl;
      SNormErrorEstimator<T_data> estimator(D, hierarchy.l_target(), s);
      SignExcludeGreedyBasedSizeInterpreter interpreter(estimator);
      // NegaBinaryGreedyBasedSizeInterpreter interpreter(estimator);      
      retrieve_sizes = interpreter.interpret_retrieve_size(
        level_sizes, level_errors, tolerance, level_num_bitplanes);
    }
    timer.end();
    timer.print("Preprocessing");

    timer.start();
    // auto prev_level_num_bitplanes(level_num_bitplanes);
    // auto retrieve_sizes = interpreter.interpret_retrieve_size(
    //     level_sizes, level_errors, tolerance, level_num_bitplanes);
    // retrieve data
    level_components = retriever.retrieve_level_components(
        level_sizes, retrieve_sizes, prev_level_num_bitplanes,
        level_num_bitplanes);
    // check whether to reconstruct to full resolution
    int skipped_level = 0;
    for (int i = 0; i <= target_level; i++) {
      if (level_num_bitplanes[target_level - i] != 0) {
        skipped_level = i;
        break;
      }
    }
    // TODO: uncomment skip level to reconstruct low resolution data
    // target_level -= skipped_level;
    // printf("target_level: %u\n", target_level);

    int total_num_bitplanes = level_squared_errors[0].size() -1;
    std::vector<std::vector<Array<1, Byte, DeviceType>>> compressed_bitplanes;
    for (int level_idx = 0; level_idx < target_level + 1; level_idx++) {
      // compressed_bitplanes.push_back(std::vector<Array<1, Byte, DeviceType>>());
      compressed_bitplanes.push_back(std::vector<Array<1, Byte, DeviceType>>(total_num_bitplanes));
      int num_bitplanes =
          level_num_bitplanes[level_idx] - prev_level_num_bitplanes[level_idx];
      for (int bitplane_idx = 0; bitplane_idx < num_bitplanes; bitplane_idx++) {
        SIZE size = level_sizes[level_idx][prev_level_num_bitplanes[level_idx] +
                                           bitplane_idx];
        // printf("level: %d, bitplane_idx: %d, size: %u\n", level_idx,
        // bitplane_idx, size);
        // Allocate space to hold compressed bitplanes
        // compressed_bitplanes[level_idx].push_back(
        //     Array<1, Byte, DeviceType>({size}));
        compressed_bitplanes[level_idx][prev_level_num_bitplanes[level_idx] + bitplane_idx] =
                                  Array<1, Byte, DeviceType>({size});
        // Copy compressed bitplanes to Array
        compressed_bitplanes[level_idx][prev_level_num_bitplanes[level_idx] + bitplane_idx].load(
            level_components[level_idx][bitplane_idx]);
      }
    }

    timer.end();
    timer.print("Interpret and retrieval");

    timer.start();
    reconstruct(prev_level_num_bitplanes, level_num_bitplanes, level_error_bounds, level_sizes,
                compressed_bitplanes, reconstructed_data, 0);
    timer.end();
    timer.print("Reconstruct");
    // return reconstructed_data;
    // retriever.release();
    // if (success)
    //   return;
    // else {
    //   std::cerr << "Reconstruct unsuccessful, return NULL pointer" <<
    //   std::endl; return;
    // }
  }

  // reconstruct progressively based on available data
  void progressive_reconstruct(double tolerance, double s, 
                               Array<D, T_data, DeviceType> &reconstructed_data) {

    // printf("start progressive_reconstruct\n");
    if (!prev_reconstructed) { // First time reconstruction
      reconstruct(tolerance, s, reconstructed_data);
      prev_reconstructed = true;
      // return data_array;
    } else {
      Array<D, T_data, DeviceType> curr_data_array(reconstructed_data);
      // Reconstruct based on newly retrieved bitplanes
      reconstruct(tolerance, s, reconstructed_data);
      // TODO: if we change resolusion here, we need to do something
      // Combine previously recomposed data with newly recomposed data
      SubArray reconstructed_subarray(reconstructed_data);
      AddND(SubArray(curr_data_array), reconstructed_subarray, 0);
      DeviceRuntime<DeviceType>::SyncQueue(0);
      // return data_array;
    }
    // return data_array.hostCopy(true);

    // TODO: add resolution changes
    // if(cur_data.size() == data.size()){
    //     for(int i=0; i<data.size(); i++){
    //         data[i] += cur_data[i];
    //     }
    // }
    // else if(cur_data.size()){
    //     std::cerr << "Reconstruct size changes, not supported yet." <<
    //     std::endl; std::cerr << "Sizes before reconstruction: " <<
    //     cur_data.size() << std::endl; std::cerr << "Sizes after
    //     reconstruction: " << data.size() << std::endl; exit(0);
    // }
    // return data.data();
  }

  void load_metadata() {
    uint8_t *metadata = retriever.load_metadata();
    uint8_t const *metadata_pos = metadata;
    uint8_t num_dims = *(metadata_pos++);
    MDR::deserialize(metadata_pos, num_dims, dimensions);
    uint8_t num_levels = *(metadata_pos++);
    MDR::deserialize(metadata_pos, num_levels, level_error_bounds);
    MDR::deserialize(metadata_pos, num_levels, level_squared_errors);
    MDR::deserialize(metadata_pos, num_levels, level_sizes);
    // MDR::deserialize(metadata_pos, num_levels, stopping_indices);
    // MDR::deserialize(metadata_pos, num_levels, level_num);
    level_num_bitplanes = std::vector<uint8_t>(num_levels, 0);
    free(metadata);
  }

  const std::vector<SIZE> &get_dimensions() { return dimensions; }

  ~ComposedReconstructor() {}

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
  void
  reconstruct(std::vector<uint8_t> &prev_level_num_bitplanes,
              std::vector<uint8_t> &curr_level_num_bitplanes,
              std::vector<T_data> &level_error_bounds,
              std::vector<std::vector<SIZE>> &level_sizes,
              std::vector<std::vector<Array<1, Byte, DeviceType>>> &compressed_bitplanes,
              Array<D, T_data, DeviceType> &reconstructed_data, int queue_idx) {

    // printf("level_num_elems: ");
    // Calculating number of elements/coefficient in each level
    SIZE target_level = hierarchy.l_target();
    std::vector<SIZE> level_num_elems(target_level + 1);
    SIZE prev_num_elems = 0;
    for (int level_idx = 0; level_idx < target_level + 1; level_idx++) {
      SIZE curr_num_elems = 1;
      for (DIM d = 0; d < D; d++) {
        curr_num_elems *= hierarchy.level_shape(level_idx, d);
      }
      level_num_elems[level_idx] = curr_num_elems - prev_num_elems;
      prev_num_elems = curr_num_elems;
      // printf("%u ", level_num_elems[level_idx]);
    }
    // printf("\n");

    // Prepare output data buffer
    // Array<D, T_data, DeviceType> output_data(
    //     hierarchy.level_shape(target_level));
    reconstructed_data.resize(hierarchy.level_shape(target_level));

    MDR::Timer timer;
    timer.start();
    // Retrieve bitplanes of each level
    // std::vector<std::vector<Array<1, Byte, DeviceType>>> compressed_bitplanes;
    // for (int level_idx = 0; level_idx < target_level + 1; level_idx++) {
    //   compressed_bitplanes.push_back(std::vector<Array<1, Byte, DeviceType>>());
    //   int num_bitplanes =
    //       curr_level_num_bitplanes[level_idx] - prev_level_num_bitplanes[level_idx];
    //   for (int bitplane_idx = 0; bitplane_idx < num_bitplanes; bitplane_idx++) {
    //     SIZE size = level_sizes[level_idx][prev_level_num_bitplanes[level_idx] +
    //                                        bitplane_idx];
    //     // printf("level: %d, bitplane_idx: %d, size: %u\n", level_idx,
    //     // bitplane_idx, size);
    //     // Allocate space to hold compressed bitplanes
    //     compressed_bitplanes[level_idx].push_back(
    //         Array<1, Byte, DeviceType>({size}));
    //     // Copy compressed bitplanes to Array
    //     compressed_bitplanes[level_idx][bitplane_idx].load(
    //         level_components[level_idx][bitplane_idx]);
    //   }
    // }

    timer.end();
    timer.print("Reconstruct Preprocessing");

    Array<1, T_data, DeviceType> *levels_array =
        new Array<1, T_data, DeviceType>[target_level + 1];
    SubArray<1, T_data, DeviceType> *levels_data =
        new SubArray<1, T_data, DeviceType>[target_level + 1];

    // Decompress and decode bitplanes of each level
    for (int level_idx = 0; level_idx <= target_level; level_idx++) {
      timer.start();
      // Number of bitplanes need to be retrieved in addition to previously
      // already retrieved bitplanes
      SIZE num_bitplanes =
          curr_level_num_bitplanes[level_idx] - prev_level_num_bitplanes[level_idx];
      // Allocate space for the bitplanes to be retrieved
      Array<2, T_bitplane, DeviceType> encoded_bitplanes(
          {num_bitplanes, encoder.buffer_size(level_num_elems[level_idx])});

      // Decompress bitplanes: compressed_bitplanes[level_idx] -->
      // encoded_bitplanes
      compressor.decompress_level(
          level_sizes[level_idx], compressed_bitplanes[level_idx],
          encoded_bitplanes, prev_level_num_bitplanes[level_idx],
          curr_level_num_bitplanes[level_idx] - prev_level_num_bitplanes[level_idx], queue_idx);
      timer.end();
      timer.print("Lossless");
      timer.start();

      // Compute the exponent of max abs value of the current level needed for
      // decoding
      int level_exp = 0;
      frexp(level_error_bounds[level_idx], &level_exp);
      // Decode bitplanes: encoded_bitplanes --> levels_array[level_idx]
      levels_array[level_idx] =
          Array<1, T_data, DeviceType>({level_num_elems[level_idx]});
      levels_data[level_idx] = SubArray(levels_array[level_idx]);
      encoder.progressive_decode(
          level_num_elems[level_idx], prev_level_num_bitplanes[level_idx],
          num_bitplanes, level_exp,
          SubArray<2, T_bitplane, DeviceType>(encoded_bitplanes), level_idx,
          levels_data[level_idx], queue_idx);
      if (num_bitplanes == 0) {
        levels_array[level_idx].memset(0);
      }
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      // levels_data[level_idx] =
      //     SubArray<1, T_data, DeviceType>(levels_array[level_idx]);
      compressor.decompress_release();
      timer.end();
      timer.print("Decoding");
      // std::cout << "recompose: level " << level_idx << " num_bitplanes: " <<
      // num_bitplanes << "\n";

      // if (level_idx == 0) {
      //   PrintSubarray("encoded_bitplanes", SubArray(encoded_bitplanes));
      // }

      // PrintSubarray("levels_data[level_idx]", levels_data[level_idx]);
    }

    timer.start();
    // Put decoded coefficients back to reordered layout
    interleaver.reposition(levels_data,
                           SubArray<D, T_data, DeviceType>(reconstructed_data),
                           target_level, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    timer.end();
    timer.print("Reposition");

    timer.start();
    // PrintSubarray("before recompose", SubArray(data_array));
    // Recompose data
    decomposer.recompose(reconstructed_data, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    // PrintSubarray("after recompose", SubArray(data_array));
    timer.end();
    timer.print("Recomposing");

    // return output_data;
  }

  Hierarchy<D, T_data, DeviceType> hierarchy;
  Decomposer decomposer;
  Interleaver interleaver;
  Encoder encoder;
  Compressor compressor;
  Retriever retriever;
  

  // std::vector<std::vector<Array<1, Byte>>>
  // compressed_bitplanes;

  Array<D, T_data, DeviceType> partial_reconsctructed_data;
  int total_num_bitplanes;

  std::vector<Array<1, T_data, DeviceType>> levels_array;
  std::vector<SubArray<1, T_data, DeviceType>> levels_data;
  Array<D, T_data, DeviceType> data_array;
  bool prev_reconstructed;

  std::vector<T_data> data;
  std::vector<SIZE> dimensions;
  std::vector<T_data> level_error_bounds;
  std::vector<uint8_t> level_num_bitplanes;
  // std::vector<uint8_t> stopping_indices;
  std::vector<std::vector<const uint8_t *>> level_components;
  std::vector<std::vector<SIZE>> level_sizes;
  std::vector<SIZE> level_num;
  std::vector<std::vector<double>> level_squared_errors;
};
} // namespace MDR
} // namespace mgard_x
#endif
