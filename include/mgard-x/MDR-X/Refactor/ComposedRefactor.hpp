#ifndef _MDR_COMPOSED_REFACTOR_HPP
#define _MDR_COMPOSED_REFACTOR_HPP

#include "../BitplaneEncoder/BitplaneEncoder.hpp"
#include "../Decomposer/Decomposer.hpp"
#include "../ErrorCollector/ErrorCollector.hpp"
#include "../Interleaver/Interleaver.hpp"
#include "../LosslessCompressor/LevelCompressor.hpp"
#include "../RefactorUtils.hpp"
#include "../Writer/Writer.hpp"
#include "RefactorInterface.hpp"
// #include "../DataStructures/MDRData.hpp"
#include <algorithm>
#include <iostream>
namespace mgard_x {
namespace MDR {
// a decomposition-based scientific data refactor: compose a refactor using
// decomposer, interleaver, encoder, and error collector
template <DIM D, typename T_data, typename DeviceType>
class ComposedRefactor
    : public concepts::RefactorInterface<D, T_data, DeviceType> {

using T_bitplane = uint32_t;
using T_error = double;
using Decomposer = MGARDOrthoganalDecomposer<D, T_data, DeviceType>;
using Interleaver = DirectInterleaver<D, T_data, DeviceType>;
using Encoder = GroupedBPEncoder<D, T_data, T_bitplane, T_error, DeviceType>;
using Compressor = DefaultLevelCompressor<T_bitplane, DeviceType>;
using ErrorCollector = MaxErrorCollector<T_data>;
using Writer = ConcatLevelFileWriter;

public:
  ComposedRefactor(Hierarchy<D, T_data, DeviceType> hierarchy, Config config,
                   std::string metadata_file, std::vector<std::string> files
                   )
      : hierarchy(hierarchy), decomposer(hierarchy), interleaver(hierarchy),
        encoder(hierarchy), compressor(Encoder::buffer_size(hierarchy.level_num_elems(hierarchy.l_target())), config),
        collector(), total_num_bitplanes(config.total_num_bitplanes), writer(metadata_file, files) {}

  ComposedRefactor(Hierarchy<D, T_data, DeviceType> hierarchy, Config config)
      : hierarchy(hierarchy), decomposer(hierarchy), interleaver(hierarchy),
        encoder(hierarchy), compressor(Encoder::buffer_size(hierarchy.level_num_elems(hierarchy.l_target())), config),
        collector(), total_num_bitplanes(config.total_num_bitplanes), writer(std::string(""), std::vector<std::string>()) {
    levels_array = new Array<1, T_data, DeviceType>[hierarchy.l_target() + 1];
    levels_data = new SubArray<1, T_data, DeviceType>[hierarchy.l_target() + 1];
    for (int level_idx = 0; level_idx < hierarchy.l_target() + 1; level_idx++) {
      levels_array[level_idx] = Array<1, T_data, DeviceType>({hierarchy.level_num_elems(level_idx)});
      levels_data[level_idx] = SubArray<1, T_data, DeviceType>(levels_array[level_idx]);
    }
    abs_max_result_array = Array<1, T_data, DeviceType>({1});
    DeviceCollective<DeviceType>::AbsMax(hierarchy.level_num_elems(hierarchy.l_target()),
                                         SubArray<1, T_data, DeviceType>(), SubArray<1, T_data, DeviceType>(),
                                           abs_max_workspace, 0);
    encoded_bitplanes_array.resize(hierarchy.l_target()+1);
    for (int level_idx = 0; level_idx < hierarchy.l_target() + 1; level_idx++) { 
      encoded_bitplanes_array[level_idx] = Array<2, T_bitplane, DeviceType>({(SIZE)total_num_bitplanes,
           encoder.buffer_size(hierarchy.level_num_elems(level_idx))});
    }
    level_errors_array = Array<1, T_error, DeviceType>({(SIZE)total_num_bitplanes + 1});
  }

  ~ComposedRefactor() {
    delete [] levels_array;
    delete [] levels_data;
  }

  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape, Config config) {
    Hierarchy<D, T_data, DeviceType> hierarchy;
    size_t size = 0;
    size += hierarchy.estimate_memory_usgae(shape);
    for (int level_idx = 0; level_idx < hierarchy.l_target() + 1; level_idx++) {
      size += hierarchy.level_num_elems(level_idx) * sizeof(T_data);
    }
    size += sizeof(T_data);
    Array<1, Byte, DeviceType> tmp;
    DeviceCollective<DeviceType>::AbsMax(hierarchy.level_num_elems(hierarchy.l_target()),
                                         SubArray<1, T_data, DeviceType>(), SubArray<1, T_data, DeviceType>(),
                                           tmp, 0);
    size += tmp.shape(0);
    for (int level_idx = 0; level_idx < hierarchy.l_target() + 1; level_idx++) { 
      size += config.total_num_bitplanes * Encoder::buffer_size(hierarchy.level_num_elems(level_idx)) * sizeof (T_bitplane);
    }

    SIZE max_n = Encoder::buffer_size(hierarchy.level_num_elems(hierarchy.l_target()));

    size += (config.total_num_bitplanes + 1) * sizeof(T_error);
    size += Decomposer::EstimateMemoryFootprint(shape);
    size += Interleaver::EstimateMemoryFootprint(shape);
    size += Encoder::EstimateMemoryFootprint(shape);
    size += Compressor::EstimateMemoryFootprint(max_n, config);
    return size;
  }

  void Refactor(Array<D, T_data, DeviceType> &data_array,
                MDRMetadata &mdr_metadata,
                MDRData<DeviceType> &mdr_data, int queue_idx) {
    SIZE target_level = hierarchy.l_target();
    mdr_metadata.num_levels = hierarchy.l_target()+1;
    mdr_metadata.num_bitplanes = total_num_bitplanes;
    mdr_metadata.InitializeForReconstruction();
    mdr_metadata.level_error_bounds.resize(hierarchy.l_target() + 1);
    mdr_metadata.level_squared_errors.resize(hierarchy.l_target() + 1);
    mdr_data.Resize(hierarchy.l_target() + 1, total_num_bitplanes);

    SubArray<D, T_data, DeviceType> data(data_array);

    Timer timer;
    timer.start();
    decomposer.decompose(data_array, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    timer.end();
    timer.print("Decompose");

    timer.start();
    interleaver.interleave(data, levels_data, hierarchy.l_target(), queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    timer.end();
    timer.print("Interleave");

    for (int level_idx = 0; level_idx < hierarchy.l_target() + 1; level_idx++) {
      timer.start();
      SubArray<1, T_data, DeviceType> result(abs_max_result_array);
      DeviceCollective<DeviceType>::AbsMax(levels_data[level_idx].shape(0),
                                           levels_data[level_idx], result,
                                           abs_max_workspace, queue_idx);
      T_data level_max_error;
      MemoryManager<DeviceType>::Copy1D(&level_max_error, result.data(), 1, queue_idx);
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      int level_exp = 0;
      frexp(level_max_error, &level_exp);
      // printf("level: %d, level_max_error: %.10f, level_exp: %d\n", level_idx,
      // level_max_error, level_exp);
      mdr_metadata.level_error_bounds[level_idx] = level_max_error;
      timer.end();
      timer.print("level_max_error");

      timer.start();
      SubArray<2, T_bitplane, DeviceType> encoded_bitplanes(
          encoded_bitplanes_array[level_idx]);
      SubArray<1, T_error, DeviceType> level_errors(level_errors_array);
      std::vector<SIZE> bitplane_sizes(total_num_bitplanes);
      encoder.encode(hierarchy.level_num_elems(level_idx), total_num_bitplanes, level_exp,
                     levels_data[level_idx], encoded_bitplanes, level_errors,
                     bitplane_sizes, queue_idx);
      std::vector<T_error> squared_error(total_num_bitplanes + 1);
      MemoryManager<DeviceType>::Copy1D(squared_error.data(), level_errors_array.data(), 
                                        total_num_bitplanes + 1, queue_idx);
      mdr_metadata.level_squared_errors[level_idx] = squared_error;
      timer.end();
      timer.print("Encoding");

      timer.start();
      compressor.compress_level(bitplane_sizes, encoded_bitplanes_array[level_idx],
                                mdr_data.compressed_bitplanes[level_idx], queue_idx);
      mdr_metadata.level_sizes.push_back(bitplane_sizes);
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Lossless");
    }
  }

  void refactor(Array<D, T_data, DeviceType> &data_array, uint8_t total_num_bitplanes, 
                MDRMetadata &mdr_metadata,
                MDRData<DeviceType> &mdr_data, int queue_idx) {

    mgard_x::Timer timer;

    timer.start();
    do_refactor(data_array, total_num_bitplanes, mdr_metadata, mdr_data, queue_idx);
    timer.end();
    timer.print("Refactor");
    // timer.start();
    // writer.write_level_components(level_components, mdr_data.level_sizes);
    // timer.end();
    // timer.print("Write");
    // write_metadata(mdr_data);
    // for (int i = 0; i < level_components.size(); i++) {
    //   for (int j = 0; j < level_components[i].size(); j++) {
    //     MemoryManager<DeviceType>::FreeHost(level_components[i][j]);
    //   }
    // }
  }

  void write_metadata(MDRMetadata &mdr_metadata) {
    SIZE metadata_size =
        sizeof(uint8_t) + MDR::get_size(hierarchy.level_shape(hierarchy.l_target())) // dimensions
        + sizeof(uint8_t) + MDR::get_size(mdr_metadata.level_error_bounds) +
        MDR::get_size(mdr_metadata.level_squared_errors) +
        MDR::get_size(mdr_metadata.level_sizes);
        // + MDR::get_size(stopping_indices) + MDR::get_size(level_num);
    uint8_t *metadata = (uint8_t *)malloc(metadata_size);
    uint8_t *metadata_pos = metadata;
    *(metadata_pos++) = (uint8_t)D;
    MDR::serialize(hierarchy.level_shape(hierarchy.l_target()), metadata_pos);
    *(metadata_pos++) = (uint8_t)mdr_metadata.level_error_bounds.size();
    MDR::serialize(mdr_metadata.level_error_bounds, metadata_pos);
    MDR::serialize(mdr_metadata.level_squared_errors, metadata_pos);
    MDR::serialize(mdr_metadata.level_sizes, metadata_pos);
    // MDR::serialize(stopping_indices, metadata_pos);
    // MDR::serialize(level_num, metadata_pos);
    writer.write_metadata(metadata, metadata_size);
    free(metadata);
  }

  void print() const {
    std::cout << "Composed refactor with the following components."
              << std::endl;
    std::cout << "Decomposer: ";
    decomposer.print();
    std::cout << "Interleaver: ";
    interleaver.print();
    std::cout << "Encoder: ";
    encoder.print();
  }

private:
  bool do_refactor(Array<D, T_data, DeviceType> &data_array, SIZE total_num_bitplanes, 
                MDRMetadata &mdr_metadata,
                MDRData<DeviceType> &mdr_data, int queue_idx) {
    SIZE target_level = hierarchy.l_target();
    // printf("target_level = %u\n", target_level);
    // std::cout << "min: " << log2(*min_element(dimensions.begin(),
    // dimensions.end())) << std::endl; uint8_t max_level =
    // log2(*min_element(dimensions.begin(), dimensions.end())) - 1;
    // if(target_level > max_level){
    //     std::cerr << "Target level is higher than " << max_level <<
    //     std::endl; return false;
    // }
    // for (DIM d = 0; d < D; d++) {
    //   dimensions.push_back(hierarchy.level_shape(target_level, d));
    // }

    SubArray<D, T_data, DeviceType> data(data_array);

    Timer timer;
    // decompose data hierarchically
    timer.start();
    decomposer.decompose(data_array, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    timer.end();
    timer.print("Decompose");
    timer.start();

    // printf("level_num_elems: ");
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

    Array<1, T_data, DeviceType> *levels_array =
        new Array<1, T_data, DeviceType>[target_level + 1];
    SubArray<1, T_data, DeviceType> *levels_data =
        new SubArray<1, T_data, DeviceType>[target_level + 1];
    for (int level_idx = 0; level_idx < target_level + 1; level_idx++) {
      levels_array[level_idx] =
          Array<1, T_data, DeviceType>({level_num_elems[level_idx]});
      levels_data[level_idx] =
          SubArray<1, T_data, DeviceType>(levels_array[level_idx]);
    }

    interleaver.interleave(data, levels_data, target_level, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    timer.end();
    timer.print("Interleave");

    std::vector<std::vector<Array<1, Byte, DeviceType>>> compressed_bitplanes;
    for (int level_idx = 0; level_idx < target_level + 1; level_idx++) {

      timer.start();
      Array<1, T_data, DeviceType> result_array({1});
      SubArray<1, T_data, DeviceType> result(result_array);
      Array<1, Byte, DeviceType> workspace;
      DeviceCollective<DeviceType>::AbsMax(levels_data[level_idx].shape(0),
                                           levels_data[level_idx], result,
                                           workspace, queue_idx);
      DeviceCollective<DeviceType>::AbsMax(levels_data[level_idx].shape(0),
                                           levels_data[level_idx], result,
                                           workspace, queue_idx);
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      T_data level_max_error = *(result_array.hostCopy());
      int level_exp = 0;
      frexp(level_max_error, &level_exp);
      // printf("level: %d, level_max_error: %.10f, level_exp: %d\n", level_idx,
      // level_max_error, level_exp);
      mdr_metadata.level_error_bounds.push_back(level_max_error);
      timer.end();
      timer.print("level_max_error");

      timer.start();
      Array<2, T_bitplane, DeviceType> encoded_bitplanes_array(
          {(SIZE)total_num_bitplanes,
           encoder.buffer_size(level_num_elems[level_idx])});
      SubArray<2, T_bitplane, DeviceType> encoded_bitplanes(
          encoded_bitplanes_array);
      Array<1, T_error, DeviceType> level_errors_array({total_num_bitplanes + 1});
      SubArray<1, T_error, DeviceType> level_errors(level_errors_array);
      std::vector<SIZE> bitplane_sizes(total_num_bitplanes);
      encoder.encode(level_num_elems[level_idx], total_num_bitplanes, level_exp,
                     levels_data[level_idx], encoded_bitplanes, level_errors,
                     bitplane_sizes, queue_idx);
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      // PrintSubarray("level_errors", level_errors);
      if (level_idx == 0) {
        // PrintSubarray("levels_data[level_idx]", levels_data[level_idx]);
        // PrintSubarray("encoded_bitplanes", SubArray(encoded_bitplanes));
      }

      // level_sqr_errors.push_back(level_errors_array);

      std::vector<T_error> squared_error;
      T_error *level_errors_host = level_errors_array.hostCopy(true);
      for (int i = 0; i < total_num_bitplanes + 1; i++) {
        squared_error.push_back(level_errors_host[i]);
      }
      mdr_metadata.level_squared_errors.push_back(squared_error);
      timer.end();
      timer.print("Encoding");

      timer.start();
      std::vector<Array<1, Byte, DeviceType>> compressed_encoded_bitplanes(total_num_bitplanes);
      mdr_data.compressed_bitplanes.push_back(compressed_encoded_bitplanes);
      // uint8_t stopping_index =
          compressor.compress_level(bitplane_sizes, encoded_bitplanes_array,
                                    mdr_data.compressed_bitplanes[level_idx], queue_idx);
      // stopping_indices.push_back(stopping_index);
      mdr_metadata.level_sizes.push_back(bitplane_sizes);
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Lossless");
    }

    timer.start();
    // write level components
    for (int level_idx = 0; level_idx < target_level + 1; level_idx++) {
      level_components.push_back(std::vector<uint8_t *>());
      for (int bitplane_idx = 0; bitplane_idx < total_num_bitplanes; bitplane_idx++) {

        Byte *bitplane;
        MemoryManager<DeviceType>::MallocHost(
            bitplane, mdr_data.compressed_bitplanes[level_idx][bitplane_idx].shape(0),
            queue_idx);
        MemoryManager<DeviceType>::Copy1D(
            bitplane, mdr_data.compressed_bitplanes[level_idx][bitplane_idx].data(),
            mdr_data.compressed_bitplanes[level_idx][bitplane_idx].shape(0), queue_idx);
        level_components[level_idx].push_back(bitplane);
      }
    }
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    timer.end();
    timer.print("Copy to CPU");

    return true;
  }

  Hierarchy<D, T_data, DeviceType> hierarchy;
  Decomposer decomposer;
  Interleaver interleaver;
  Encoder encoder;
  Compressor compressor;
  ErrorCollector collector;
  Writer writer;

  Array<1, T_data, DeviceType> *levels_array = nullptr;
  SubArray<1, T_data, DeviceType> *levels_data = nullptr;
  Array<1, T_data, DeviceType> abs_max_result_array;
  Array<1, Byte, DeviceType> abs_max_workspace;
  std::vector<Array<2, T_bitplane, DeviceType>> encoded_bitplanes_array;
  Array<1, T_error, DeviceType> level_errors_array;

  SIZE total_num_bitplanes;
  // Array<D, T_data, DeviceType> data_array;
  // std::vector<T> data;

  // std::vector<SIZE> dimensions;
  // std::vector<T_data> level_error_bounds;
  // std::vector<uint8_t> stopping_indices;
  std::vector<std::vector<Byte *>> level_components;
  // std::vector<std::vector<SIZE>> level_sizes;
  // std::vector<SIZE> level_num;
  // std::vector<std::vector<T_error>> level_squared_errors;
};
} // namespace MDR
} // namespace mgard_x
#endif
