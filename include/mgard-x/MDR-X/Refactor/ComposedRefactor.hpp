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
#include <algorithm>
#include <iostream>
namespace mgard_x {
namespace MDR {
// a decomposition-based scientific data refactor: compose a refactor using
// decomposer, interleaver, encoder, and error collector
template <DIM D, typename T_data, typename T_bitplane, typename T_error, class Decomposer,
          class Interleaver, class Encoder, class Compressor,
          class ErrorCollector, class Writer, typename DeviceType>
class ComposedRefactor
    : public concepts::RefactorInterface<D, T_data, T_bitplane, DeviceType> {

public:
  ComposedRefactor(Hierarchy<D, T_data, DeviceType> &hierarchy,
                   Decomposer decomposer, Interleaver interleaver,
                   Encoder encoder, Compressor compressor,
                   ErrorCollector collector, Writer writer)
      : hierarchy(hierarchy), decomposer(decomposer), interleaver(interleaver),
        encoder(encoder), compressor(compressor), collector(collector),
        writer(writer) {}

  void refactor(Array<D, T_data, DeviceType>& data_array, const std::vector<SIZE> &dims,
                uint8_t target_level, uint8_t num_bitplanes) {

    MDR::Timer timer;
    this->data_array = data_array;
    // PrintSubarray("data_array", SubArray(data_array));

    // { // debug
    //   SubArray org(data_array);
    //   org.resize({5,5,5});
    //   PrintSubarray("refactor::input", org);
    // }


    timer.start();
    if (refactor(target_level, num_bitplanes, 0)) {
      timer.end();
      // timer.print("Refactor");
      timer.start();
      level_num = writer.write_level_components(level_components, level_sizes);
      timer.end();
      // timer.print("Write");
    }

    write_metadata();
    for (int i = 0; i < level_components.size(); i++) {
      for (int j = 0; j < level_components[i].size(); j++) {
        MemoryManager<DeviceType>::FreeHost(level_components[i][j]);
      }
    }
  }

  void write_metadata() const {
    SIZE metadata_size =
        sizeof(uint8_t) + MDR::get_size(dimensions) // dimensions
        + sizeof(uint8_t) + MDR::get_size(level_error_bounds) +
        MDR::get_size(level_squared_errors) +
        MDR::get_size(level_sizes) // level information
        + MDR::get_size(stopping_indices) + MDR::get_size(level_num);
    uint8_t *metadata = (uint8_t *)malloc(metadata_size);
    uint8_t *metadata_pos = metadata;
    *(metadata_pos++) = (uint8_t)dimensions.size();
    MDR::serialize(dimensions, metadata_pos);
    *(metadata_pos++) = (uint8_t)level_error_bounds.size();
    MDR::serialize(level_error_bounds, metadata_pos);
    MDR::serialize(level_squared_errors, metadata_pos);
    MDR::serialize(level_sizes, metadata_pos);
    MDR::serialize(stopping_indices, metadata_pos);
    MDR::serialize(level_num, metadata_pos);
    writer.write_metadata(metadata, metadata_size);
    free(metadata);
  }

  ~ComposedRefactor() {}

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
  bool refactor(SIZE target_level, SIZE num_bitplanes, int queue_idx) {
    // printf("target_level = %u\n", target_level);
    // std::cout << "min: " << log2(*min_element(dimensions.begin(),
    // dimensions.end())) << std::endl; uint8_t max_level =
    // log2(*min_element(dimensions.begin(), dimensions.end())) - 1;
    // if(target_level > max_level){
    //     std::cerr << "Target level is higher than " << max_level <<
    //     std::endl; return false;
    // }
    for (DIM d = 0; d < D; d++) {
      dimensions.push_back(hierarchy.dofs[d][0]);
    }

    SubArray<D, T_data, DeviceType> data(data_array);

    MDR::Timer timer;
    // // decompose data hierarchically
    timer.start();
    // PrintSubarray("before decomposition", data);
    decomposer.decompose(data, target_level, queue_idx);
    // PrintSubarray("after decomposition", data);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    timer.end();
    // timer.print("Decompose");
    timer.start();

    // printf("level_num_elems: ");
    std::vector<SIZE> level_num_elems(target_level + 1);
    SIZE prev_num_elems = 0;
    for (int level_idx = 0; level_idx < target_level + 1; level_idx++) {
      SIZE curr_num_elems = 1;
      for (DIM d = 0; d < D; d++) {
        curr_num_elems *= hierarchy.dofs[d][target_level - level_idx];
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

    // printf("done create levels_data\n");

    interleaver.interleave(data, levels_data, target_level + 1, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    timer.end();
    // timer.print("Interleave");

    DeviceCollective<DeviceType> deviceReduce;

    std::vector<std::vector<Array<1, Byte, DeviceType>>> compressed_bitplanes;
    for (int level_idx = 0; level_idx < target_level + 1; level_idx++) {

      timer.start();
      Array<1, T_data, DeviceType> result_array({1});
      SubArray<1, T_data, DeviceType> result(result_array);

      deviceReduce.AbsMax(levels_data[level_idx].getShape(0),
                          levels_data[level_idx], result, queue_idx);
      T_data level_max_error = *(result_array.hostCopy());
      int level_exp = 0;
      frexp(level_max_error, &level_exp);
      // printf("level: %d, level_max_error: %.10f, level_exp: %d\n", level_idx, level_max_error, level_exp);
      level_error_bounds.push_back(level_max_error);
      timer.end();
      // timer.print("level_max_error");

      timer.start();
      Array<1, T_error, DeviceType> level_errors_array({num_bitplanes + 1});
      SubArray<1, T_error, DeviceType> level_errors(level_errors_array);
      std::vector<SIZE> bitplane_sizes(num_bitplanes);
      Array<2, T_bitplane, DeviceType> encoded_bitplanes = encoder.encode(
          level_num_elems[level_idx], num_bitplanes, level_exp,
          levels_data[level_idx], level_errors, bitplane_sizes, queue_idx);
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

      // PrintSubarray("level_errors", level_errors);
      if (level_idx == 0) {
        // PrintSubarray("levels_data[level_idx]", levels_data[level_idx]);
        // PrintSubarray("encoded_bitplanes", SubArray(encoded_bitplanes));
      }

      // level_sqr_errors.push_back(level_errors_array);

      std::vector<T_error> squared_error;
      T_error *level_errors_host = level_errors_array.hostCopy(true);
      for (int i = 0; i < num_bitplanes + 1; i++) {
        squared_error.push_back(level_errors_host[i]);
      }
      level_squared_errors.push_back(squared_error);
      timer.end();
      // timer.print("Encoding");

      timer.start();
      std::vector<Array<1, Byte, DeviceType>> compressed_encoded_bitplanes;
      compressed_bitplanes.push_back(compressed_encoded_bitplanes);
      uint8_t stopping_index = compressor.compress_level(
          bitplane_sizes, encoded_bitplanes, compressed_bitplanes[level_idx]);
      stopping_indices.push_back(stopping_index);
      level_sizes.push_back(bitplane_sizes);

      timer.end();
      // timer.print("Lossless");
    }

    timer.start();
    // write level components
    for (int level_idx = 0; level_idx < target_level + 1; level_idx++) {
      level_components.push_back(std::vector<uint8_t *>());
      for (int bitplane_idx = 0; bitplane_idx < num_bitplanes; bitplane_idx++) {

        Byte *bitplane;
        MemoryManager<DeviceType>::MallocHost(
            bitplane, compressed_bitplanes[level_idx][bitplane_idx].shape()[0],
            queue_idx);
        MemoryManager<DeviceType>::Copy1D(
            bitplane, compressed_bitplanes[level_idx][bitplane_idx].data(),
            compressed_bitplanes[level_idx][bitplane_idx].shape()[0], queue_idx);
        level_components[level_idx].push_back(bitplane);
      }
    }
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    timer.end();
    // timer.print("Copy to CPU");

    return true;
  }

  Hierarchy<D, T_data, DeviceType> &hierarchy;
  Decomposer decomposer;
  Interleaver interleaver;
  Encoder encoder;
  Compressor compressor;
  ErrorCollector collector;
  Writer writer;

  Array<D, T_data, DeviceType> data_array;
  // std::vector<T> data;

  std::vector<SIZE> dimensions;
  std::vector<T_data> level_error_bounds;
  std::vector<uint8_t> stopping_indices;
  std::vector<std::vector<Byte *>> level_components;
  std::vector<std::vector<SIZE>> level_sizes;
  std::vector<SIZE> level_num;
  std::vector<std::vector<T_error>> level_squared_errors;
};
} // namespace MDR
} // namespace mgard_x
#endif
