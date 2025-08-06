#ifndef _MDR_COMPOSED_REFACTOR_HPP
#define _MDR_COMPOSED_REFACTOR_HPP

#include "../BitplaneEncoder/BitplaneEncoder.hpp"
#include "../Decomposer/Decomposer.hpp"
#include "../ErrorCollector/ErrorCollector.hpp"
#include "../Interleaver/Interleaver.hpp"
#include "../LosslessCompressor/LevelCompressor.hpp"
// #include "../RefactorUtils.hpp"
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
public:
  using HierarchyType = Hierarchy<D, T_data, DeviceType>;
  using T_bitplane = uint32_t;
  using T_error = double;
  using Decomposer = MGARDOrthoganalDecomposer<D, T_data, DeviceType>;
  using Interleaver = DirectInterleaver<D, T_data, DeviceType>;
  using Encoder = GroupedBPEncoder<D, T_data, T_bitplane, T_error, DeviceType>;
  // using Compressor = DefaultLevelCompressor<T_bitplane, DeviceType>;
  using Compressor = NullLevelCompressor<T_bitplane, DeviceType>;

  ComposedRefactor() : initialized(false) {}

  ComposedRefactor(Hierarchy<D, T_data, DeviceType> &hierarchy, Config config) {
    Adapt(hierarchy, config, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
  }

  static SIZE MaxOutputDataSize(std::vector<SIZE> shape, Config config) {
    Hierarchy<D, T_data, DeviceType> hierarchy;
    hierarchy.EstimateMemoryFootprint(shape);
    SIZE size = 0;
    for (int level_idx = 0; level_idx < hierarchy.l_target() + 1; level_idx++) {
      size += config.total_num_bitplanes *
              Encoder::buffer_size(hierarchy.level_num_elems(level_idx)) *
              sizeof(T_bitplane);
    }
    return size;
  }

  ~ComposedRefactor() {
    delete[] levels_array;
    delete[] levels_data;
  }

  void Adapt(Hierarchy<D, T_data, DeviceType> &hierarchy, Config config,
             int queue_idx) {
    this->initialized = true;
    this->hierarchy = &hierarchy;
    decomposer.Adapt(hierarchy, config, queue_idx);
    interleaver.Adapt(hierarchy, queue_idx);
    encoder.Adapt(hierarchy, queue_idx);
    compressor.Adapt(
        Encoder::buffer_size(hierarchy.level_num_elems(hierarchy.l_target())),
        config, queue_idx);
    total_num_bitplanes = config.total_num_bitplanes;

    delete[] levels_array;
    delete[] levels_data;
    levels_array = new Array<1, T_data, DeviceType>[hierarchy.l_target() + 1];
    levels_data = new SubArray<1, T_data, DeviceType>[hierarchy.l_target() + 1];
    for (int level_idx = 0; level_idx < hierarchy.l_target() + 1; level_idx++) {
      levels_array[level_idx].resize({hierarchy.level_num_elems(level_idx)},
                                     queue_idx);
      levels_data[level_idx] =
          SubArray<1, T_data, DeviceType>(levels_array[level_idx]);
    }
    abs_max_result_array.resize({1}, queue_idx);
    DeviceCollective<DeviceType>::AbsMax(
        hierarchy.level_num_elems(hierarchy.l_target()),
        SubArray<1, T_data, DeviceType>(), SubArray<1, T_data, DeviceType>(),
        abs_max_workspace, false, 0);
    encoded_bitplanes_array.resize(hierarchy.l_target() + 1);
    for (int level_idx = 0; level_idx < hierarchy.l_target() + 1; level_idx++) {
      encoded_bitplanes_array[level_idx].resize(
          {(SIZE)total_num_bitplanes,
           encoder.buffer_size(hierarchy.level_num_elems(level_idx))},
          queue_idx);
    }
    level_errors_array.resize({(SIZE)total_num_bitplanes + 1}, queue_idx);
  }

  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape,
                                        Config config) {
    Hierarchy<D, T_data, DeviceType> hierarchy;
    size_t size = 0;
    size += hierarchy.EstimateMemoryFootprint(shape);
    for (int level_idx = 0; level_idx < hierarchy.l_target() + 1; level_idx++) {
      size += hierarchy.level_num_elems(level_idx) * sizeof(T_data);
    }
    size += sizeof(T_data);
    Array<1, Byte, DeviceType> tmp;
    DeviceCollective<DeviceType>::AbsMax(
        hierarchy.level_num_elems(hierarchy.l_target()),
        SubArray<1, T_data, DeviceType>(), SubArray<1, T_data, DeviceType>(),
        tmp, false, 0);
    size += tmp.shape(0);
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

  void Refactor(Array<D, T_data, DeviceType> &data_array,
                MDRMetadata &mdr_metadata, MDRData<DeviceType> &mdr_data,
                int queue_idx) {
    SIZE target_level = hierarchy->l_target();
    mdr_metadata.Initialize(hierarchy->l_target() + 1, total_num_bitplanes);
    mdr_data.Resize(hierarchy->l_target() + 1, total_num_bitplanes);

    SubArray<D, T_data, DeviceType> data(data_array);

    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }
    decomposer.decompose(data_array, hierarchy->l_target(), 0, queue_idx);
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Decompose");
      timer.clear();
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }
    interleaver.interleave(data, levels_data, hierarchy->l_target(), queue_idx);
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Interleave");
      timer.clear();
    }

    for (int level_idx = 0; level_idx < hierarchy->l_target() + 1;
         level_idx++) {

      if (log::level & log::TIME) {
        DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
        timer.start();
      }
      SubArray<1, T_data, DeviceType> result(abs_max_result_array);
      DeviceCollective<DeviceType>::AbsMax(levels_data[level_idx].shape(0),
                                           levels_data[level_idx], result,
                                           abs_max_workspace, true, queue_idx);
      T_data level_max_error;
      MemoryManager<DeviceType>::Copy1D(&level_max_error, result.data(), 1,
                                        queue_idx);
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      int level_exp = 0;
      frexp(level_max_error, &level_exp);
      // printf("level: %d, level_max_error: %.10f, level_exp: %d\n", level_idx,
      // level_max_error, level_exp);
      mdr_metadata.level_error_bounds[level_idx] = level_max_error;
      mdr_metadata.level_num_elems[level_idx] =
          hierarchy->level_num_elems(level_idx);
      if (log::level & log::TIME) {
        DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
        timer.end();
        timer.print("Max Error");
        timer.clear();
        timer.start();
      }

      SubArray<2, T_bitplane, DeviceType> encoded_bitplanes(
          encoded_bitplanes_array[level_idx]);
      SubArray<1, T_error, DeviceType> level_errors(level_errors_array);
      std::vector<SIZE> bitplane_sizes(total_num_bitplanes);
      encoder.encode(hierarchy->level_num_elems(level_idx), total_num_bitplanes,
                     level_exp, levels_data[level_idx], encoded_bitplanes,
                     level_errors, bitplane_sizes, queue_idx);
      std::vector<T_error> squared_error(total_num_bitplanes + 1);
      MemoryManager<DeviceType>::Copy1D(squared_error.data(),
                                        level_errors_array.data(),
                                        total_num_bitplanes + 1, queue_idx);
      mdr_metadata.level_squared_errors[level_idx] = squared_error;
      // PrintSubarray("level_errors", level_errors);
      if (log::level & log::TIME) {
        DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
        timer.end();
        timer.print("Encoding");
        timer.clear();
        timer.start();
      }

      compressor.compress_level(
          bitplane_sizes, encoded_bitplanes_array[level_idx],
          mdr_data.compressed_bitplanes[level_idx], queue_idx);
      mdr_metadata.level_sizes[level_idx] = bitplane_sizes;
      if (log::level & log::TIME) {
        DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
        timer.end();
        timer.print("Compress");
        timer.clear();
      }
    }
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

  bool initialized = false;

private:
  Hierarchy<D, T_data, DeviceType> *hierarchy;
  Decomposer decomposer;
  Interleaver interleaver;
  Encoder encoder;
  Compressor compressor;

  Array<1, T_data, DeviceType> *levels_array = nullptr;
  SubArray<1, T_data, DeviceType> *levels_data = nullptr;
  Array<1, T_data, DeviceType> abs_max_result_array;
  Array<1, Byte, DeviceType> abs_max_workspace;
  std::vector<Array<2, T_bitplane, DeviceType>> encoded_bitplanes_array;
  Array<1, T_error, DeviceType> level_errors_array;

  SIZE total_num_bitplanes;
  std::vector<std::vector<Byte *>> level_components;
};
} // namespace MDR
} // namespace mgard_x
#endif
