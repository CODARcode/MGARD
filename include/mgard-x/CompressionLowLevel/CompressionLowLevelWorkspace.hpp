/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_COMPRESSION_LOW_LEVEL_WORKSPACE_HPP
#define MGARD_X_COMPRESSION_LOW_LEVEL_WORKSPACE_HPP

#include "../DataRefactoring/DataRefactoringWorkspace.hpp"
#include "../Hierarchy/Hierarchy.h"
#include "../Lossless/ParallelHuffman/HuffmanWorkspace.hpp"
#include "../RuntimeX/RuntimeXPublic.h"

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
class CompressionLowLevelWorkspace {
public:
  CompressionLowLevelWorkspace() {
    // By defualt it is not pre-allocated
    pre_allocated = false;
  }

  void initialize_subarray() {
    // Reuse refactoring_w_array
    norm_tmp_subarray = SubArray<1, T, DeviceType>(
        {total_elems},
        (T *)data_refactoring_workspace.refactoring_w_array.data());
    norm_subarray = SubArray(norm_array);
    quantizers_subarray = SubArray(quantizers_array);
    // Reuse refactoring_w_array
    quantized_subarray = SubArray<D, QUANTIZED_INT, DeviceType>(
        shape,
        (QUANTIZED_INT *)data_refactoring_workspace.refactoring_w_array.data());
    outlier_count_subarray = SubArray(outlier_count_array);
    outlier_idx_subarray = SubArray(outlier_idx_array);
    outliers_subarray = SubArray(outliers_array);
  }

  size_t estimate_size(std::vector<SIZE> shape, SIZE l_target, SIZE dict_size,
                       SIZE chunk_size, double estimated_outlier_ratio = 0.5) {
    SIZE total_num_elems = 1;
    for (DIM d = 0; d < D; d++)
      total_num_elems *= shape[d];
    Array<1, T, DeviceType> array_with_pitch({1});
    size_t pitch_size = array_with_pitch.ld(0) * sizeof(T);

    size_t size = 0;
    // size += total_num_elems() * sizeof(T);
    size += sizeof(T);
    size += data_refactoring_workspace.estimate_size(shape);
    size +=
        huffman_workspace.estimate_size(total_num_elems, dict_size, chunk_size);
    size += roundup((l_target + 1) * sizeof(T), pitch_size);
    size += roundup(sizeof(LENGTH), pitch_size);
    size += roundup(
        (size_t)(total_num_elems * estimated_outlier_ratio * sizeof(LENGTH)),
        pitch_size);
    size += roundup((size_t)(total_num_elems * estimated_outlier_ratio *
                             sizeof(QUANTIZED_INT)),
                    pitch_size);
    return size;
  }

  void allocate(Hierarchy<D, T, DeviceType> &hierarchy, Config config,
                double estimated_outlier_ratio) {
    shape = hierarchy.level_shape(hierarchy.l_target());
    total_elems = hierarchy.total_num_elems();
    // Do pre-allocation
    // Reuse refactoring_w_array
    // norm_tmp_array = Array<1, T, DeviceType>({hierarchy.total_num_elems()});
    norm_array = Array<1, T, DeviceType>({1});
    data_refactoring_workspace.allocate(hierarchy);
    huffman_workspace.allocate(total_elems, config.huff_dict_size,
                               config.huff_block_size);
    quantizers_array = Array<1, T, DeviceType>({hierarchy.l_target() + 1});
    // quantized_array = Array<D, QUANTIZED_INT, DeviceType>(
    // hierarchy.level_shape(hierarchy.l_target()), false, false);
    outlier_count_array = Array<1, LENGTH, DeviceType>({1}, false, false);
    outlier_idx_array = Array<1, LENGTH, DeviceType>(
        {(SIZE)(total_elems * estimated_outlier_ratio)});
    outliers_array = Array<1, QUANTIZED_INT, DeviceType>(
        {(SIZE)(total_elems * estimated_outlier_ratio)});
    // Reset the outlier count to zero
    outlier_count_array.memset(0);
    outlier_idx_array.memset(0);
    outliers_array.memset(0);
    initialize_subarray();

    pre_allocated = true;
  }

  void move(const CompressionLowLevelWorkspace<D, T, DeviceType> &workspace) {
    // Move instead of copy
    shape = workspace.shape;
    total_elems = workspace.total_elems;
    outlier_count = outlier_count;
    norm_tmp_array = std::move(workspace.norm_tmp_array);
    norm_array = std::move(workspace.norm_array);
    data_refactoring_workspace =
        std::move(workspace.data_refactoring_workspace);
    huffman_workspace = std::move(workspace.huffman_workspace);
    quantizers_array = std::move(workspace.quantizers_array);
    quantized_array = std::move(workspace.quantized_array);
    outlier_count_array = std::move(workspace.outlier_count_array);
    outlier_idx_array = std::move(workspace.outlier_idx_array);
    outliers_array = std::move(workspace.outliers_array);
    initialize_subarray();
  }

  void move(CompressionLowLevelWorkspace<D, T, DeviceType> &&workspace) {
    // Move instead of copy
    shape = workspace.shape;
    total_elems = workspace.total_elems;
    outlier_count = outlier_count;
    norm_tmp_array = std::move(workspace.norm_tmp_array);
    norm_array = std::move(workspace.norm_array);
    data_refactoring_workspace =
        std::move(workspace.data_refactoring_workspace);
    huffman_workspace = std::move(workspace.huffman_workspace);
    quantizers_array = std::move(workspace.quantizers_array);
    quantized_array = std::move(workspace.quantized_array);
    outlier_count_array = std::move(workspace.outlier_count_array);
    outlier_idx_array = std::move(workspace.outlier_idx_array);
    outliers_array = std::move(workspace.outliers_array);
    initialize_subarray();
  }

  CompressionLowLevelWorkspace(Hierarchy<D, T, DeviceType> &hierarchy,
                               Config config,
                               double estimated_outlier_ratio = 0.3) {
    allocate(hierarchy, config, estimated_outlier_ratio);
  }

  CompressionLowLevelWorkspace(
      const CompressionLowLevelWorkspace<D, T, DeviceType> &workspace) {
    move(std::move(workspace));
  }

  CompressionLowLevelWorkspace &
  operator=(const CompressionLowLevelWorkspace<D, T, DeviceType> &workspace) {
    move(std::move(workspace));
  }

  CompressionLowLevelWorkspace(
      CompressionLowLevelWorkspace<D, T, DeviceType> &&workspace) {
    move(std::move(workspace));
  }

  CompressionLowLevelWorkspace &
  operator=(CompressionLowLevelWorkspace<D, T, DeviceType> &&workspace) {
    move(std::move(workspace));
  }

  bool pre_allocated;

  std::vector<SIZE> shape;
  SIZE total_elems;
  LENGTH outlier_count;

  DataRefactoringWorkspace<D, T, DeviceType> data_refactoring_workspace;
  HuffmanWorkspace<QUANTIZED_UNSIGNED_INT, HUFFMAN_CODE, DeviceType>
      huffman_workspace;

  Array<1, T, DeviceType> norm_tmp_array;
  Array<1, T, DeviceType> norm_array;
  Array<1, T, DeviceType> quantizers_array;
  Array<D, QUANTIZED_INT, DeviceType> quantized_array;
  Array<1, LENGTH, DeviceType> outlier_count_array;
  Array<1, LENGTH, DeviceType> outlier_idx_array;
  Array<1, QUANTIZED_INT, DeviceType> outliers_array;

  SubArray<1, T, DeviceType> norm_tmp_subarray;
  SubArray<1, T, DeviceType> norm_subarray;
  SubArray<1, T, DeviceType> quantizers_subarray;
  SubArray<D, QUANTIZED_INT, DeviceType> quantized_subarray;
  SubArray<1, LENGTH, DeviceType> outlier_count_subarray;
  SubArray<1, LENGTH, DeviceType> outlier_idx_subarray;
  SubArray<1, QUANTIZED_INT, DeviceType> outliers_subarray;
};

} // namespace mgard_x

#endif