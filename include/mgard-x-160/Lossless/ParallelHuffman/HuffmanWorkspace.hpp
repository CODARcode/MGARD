/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_160_HUFFMAN_WORKSPACE_HPP
#define MGARD_X_160_HUFFMAN_WORKSPACE_HPP

#include "../../Hierarchy/Hierarchy.h"
#include "../../RuntimeX/RuntimeXPublic.h"
#include "GenerateCL.hpp"
#include "ParallelDeflate.hpp"

namespace mgard_x_160 {

template <typename Q, typename S, typename H, typename DeviceType>
class HuffmanWorkspace {
public:
  HuffmanWorkspace() {
    // By defualt it is not pre-allocated
    pre_allocated = false;
  }

  void initialize_subarray() {
    outlier_count_subarray = SubArray(outlier_count_array);
    outlier_idx_subarray = SubArray(outlier_idx_array);
    outlier_subarray = SubArray(outlier_array);

    freq_subarray = SubArray(freq_array);
    codebook_subarray = SubArray(codebook_array);
    decodebook_subarray = SubArray(decodebook_array);
    huff_bitwidths_subarray = SubArray(huff_bitwidths_array);
    condense_write_offsets_subarray = SubArray(condense_write_offsets_array);
    condense_actual_lengths_subarray = SubArray(condense_actual_lengths_array);

    deflate_group_bits_subarray = SubArray(deflate_group_bits_array);
    deflate_group_offsets_subarray = SubArray(deflate_group_offsets_array);
    deflate_chunk_words_subarray = SubArray(deflate_chunk_words_array);
    deflate_chunk_word_offsets_subarray =
        SubArray(deflate_chunk_word_offsets_array);

    // Codebook
    first_nonzero_index_subarray = SubArray(first_nonzero_index_array);
    sort_by_key_workspace_subarray = SubArray(sort_by_key_workspace);
    _d_freq_copy_subarray = SubArray(_d_freq_copy_array);
    _d_qcode_copy_subarray = SubArray(_d_qcode_copy_array);
    CL_subarray = SubArray(CL_array);
    lNodesLeader_subarray = SubArray(lNodesLeader_array);
    iNodesFreq_subarray = SubArray(iNodesFreq_array);
    iNodesLeader_subarray = SubArray(iNodesLeader_array);
    tempFreq_subarray = SubArray(tempFreq_array);
    tempIsLeaf_subarray = SubArray(tempIsLeaf_array);
    tempIndex_subarray = SubArray(tempIndex_array);
    copyFreq_subarray = SubArray(copyFreq_array);
    copyIsLeaf_subarray = SubArray(copyIsLeaf_array);
    copyIndex_subarray = SubArray(copyIndex_array);
    _d_codebook_subarray_org = SubArray(_d_codebook_array_org);
    status_subarray = SubArray(status_array);
    diagonal_path_intersections_subarray =
        SubArray(diagonal_path_intersections_array);
  }

  static size_t EstimateMemoryFootprint(SIZE primary_count, SIZE dict_size,
                                        SIZE chunk_size,
                                        double estimated_outlier_ratio = 1) {
    size_t size = 0;
    size += sizeof(ATOMIC_IDX);
    size += primary_count * estimated_outlier_ratio * sizeof(ATOMIC_IDX);
    size += primary_count * estimated_outlier_ratio * sizeof(S);

    size += dict_size * sizeof(unsigned int);
    size += dict_size * sizeof(H);
    size_t type_bw = sizeof(H) * 8;
    size_t decodebook_size = sizeof(H) * (2 * type_bw) + sizeof(Q) * dict_size;
    size += decodebook_size * sizeof(uint8_t);
    size_t nchunk = (primary_count - 1) / chunk_size + 1;
    size += nchunk * sizeof(size_t);
    size += nchunk * sizeof(size_t);
    size += nchunk * sizeof(size_t);

    // Parallel deflate
    SIZE groups_per_chunk =
        (chunk_size - 1) / GetDeflateGroupSize<DeviceType>(chunk_size) + 1;
    size_t ngroups = nchunk * groups_per_chunk;
    size += ngroups * sizeof(size_t);       // group_bits
    size += (ngroups + 1) * sizeof(size_t); // group_offsets
    size += nchunk * sizeof(size_t);        // chunk_words
    size += (nchunk + 1) * sizeof(size_t);  // chunk_word_offsets
    {
      Array<1, Byte, DeviceType> tmp_group_scan, tmp_chunk_scan;
      DeviceCollective<DeviceType>::ScanSumExtended(
          (SIZE)ngroups, SubArray<1, size_t, DeviceType>(),
          SubArray<1, size_t, DeviceType>(), tmp_group_scan, false, 0);
      DeviceCollective<DeviceType>::ScanSumExtended(
          (SIZE)nchunk, SubArray<1, size_t, DeviceType>(),
          SubArray<1, size_t, DeviceType>(), tmp_chunk_scan, false, 0);
      size += tmp_group_scan.shape(0);
      size += tmp_chunk_scan.shape(0);
    }

    size += sizeof(unsigned int);
    Array<1, Byte, DeviceType> tmp;
    DeviceCollective<DeviceType>::SortByKey(
        dict_size, SubArray<1, unsigned int, DeviceType>(),
        SubArray<1, Q, DeviceType>(), SubArray<1, unsigned int, DeviceType>(),
        SubArray<1, Q, DeviceType>(), tmp, false, 0);
    size += tmp.shape(0);
    size += dict_size * sizeof(unsigned int);
    size += dict_size * sizeof(Q);
    size += sizeof(unsigned int) * dict_size * 4;
    size += sizeof(int) * dict_size * 6;
    size += sizeof(H) * dict_size;
    size += sizeof(int) * 16;
    size +=
        GenerateCLKernel<unsigned int,
                         DeviceType>::DiagonalPathIntersectionsSize(dict_size) *
        sizeof(uint32_t);
    size += 7 * sizeof(Byte); // signature_verify_array
    return size;
  }

  void allocate(SIZE primary_count, SIZE dict_size, SIZE chunk_size,
                double estimated_outlier_ratio) {

    outlier_count_array = Array<1, ATOMIC_IDX, DeviceType>({1});
    outlier_idx_array = Array<1, ATOMIC_IDX, DeviceType>(
        {(SIZE)(primary_count * estimated_outlier_ratio)});
    outlier_array = Array<1, S, DeviceType>(
        {(SIZE)(primary_count * estimated_outlier_ratio)});

    freq_array = Array<1, unsigned int, DeviceType>({dict_size});
    freq_array.hostAllocate(false);
    codebook_array = Array<1, H, DeviceType>({dict_size});
    size_t type_bw = sizeof(H) * 8;
    size_t decodebook_size = sizeof(H) * (2 * type_bw) + sizeof(Q) * dict_size;
    decodebook_array = Array<1, uint8_t, DeviceType>({(SIZE)decodebook_size});
    size_t nchunk = (primary_count - 1) / chunk_size + 1;
    huff_bitwidths_array = Array<1, size_t, DeviceType>({(SIZE)nchunk});
    condense_write_offsets_array = Array<1, size_t, DeviceType>({(SIZE)nchunk});
    condense_actual_lengths_array =
        Array<1, size_t, DeviceType>({(SIZE)nchunk});

    // Parallel deflate
    SIZE groups_per_chunk =
        (chunk_size - 1) / GetDeflateGroupSize<DeviceType>(chunk_size) + 1;
    SIZE ngroups = (SIZE)(nchunk * groups_per_chunk);
    deflate_group_bits_array = Array<1, size_t, DeviceType>({ngroups});
    deflate_group_offsets_array = Array<1, size_t, DeviceType>({ngroups + 1});
    deflate_chunk_words_array = Array<1, size_t, DeviceType>({(SIZE)nchunk});
    deflate_chunk_word_offsets_array =
        Array<1, size_t, DeviceType>({(SIZE)nchunk + 1});
    DeviceCollective<DeviceType>::ScanSumExtended(
        ngroups, SubArray<1, size_t, DeviceType>(),
        SubArray<1, size_t, DeviceType>(), deflate_group_scan_workspace, false,
        0);
    DeviceCollective<DeviceType>::ScanSumExtended(
        (SIZE)nchunk, SubArray<1, size_t, DeviceType>(),
        SubArray<1, size_t, DeviceType>(), deflate_chunk_scan_workspace, false,
        0);
    // Codebook
    first_nonzero_index_array = Array<1, unsigned int, DeviceType>({1});
    // first_nonzero_index_array.hostCopy(); // Create host allocation
    // Allocate workspace
    DeviceCollective<DeviceType>::SortByKey(
        dict_size, SubArray<1, unsigned int, DeviceType>(),
        SubArray<1, Q, DeviceType>(), SubArray<1, unsigned int, DeviceType>(),
        SubArray<1, Q, DeviceType>(), sort_by_key_workspace, false, 0);
    _d_freq_copy_array = Array<1, unsigned int, DeviceType>({(SIZE)dict_size});
    _d_qcode_copy_array = Array<1, Q, DeviceType>({(SIZE)dict_size});
    CL_array = Array<1, unsigned int, DeviceType>({dict_size});
    CL_array.hostAllocate(false);
    lNodesLeader_array = Array<1, int, DeviceType>({dict_size});
    iNodesFreq_array = Array<1, unsigned int, DeviceType>({dict_size});
    iNodesLeader_array = Array<1, int, DeviceType>({dict_size});
    tempFreq_array = Array<1, unsigned int, DeviceType>({dict_size});
    tempIsLeaf_array = Array<1, int, DeviceType>({dict_size});
    tempIndex_array = Array<1, int, DeviceType>({dict_size});
    copyFreq_array = Array<1, unsigned int, DeviceType>({dict_size});
    copyIsLeaf_array = Array<1, int, DeviceType>({dict_size});
    copyIndex_array = Array<1, int, DeviceType>({dict_size});
    _d_codebook_array_org = Array<1, H, DeviceType>({dict_size});
    status_array = Array<1, int, DeviceType, false, false>(
        {(SIZE)16}); // non-managed: atomicMin on managed mem unreliable on ROCm
    diagonal_path_intersections_array = Array<1, uint32_t, DeviceType>(
        {GenerateCLKernel<unsigned int, DeviceType>::
             DiagonalPathIntersectionsSize(dict_size)});

    signature_verify_array = Array<1, Byte, DeviceType>({(SIZE)7});
    signature_verify_array.hostAllocate(false);

    // outlier_count_array.memset(0);
    // outlier_idx_array.memset(0);
    // outlier_array.memset(0);

    initialize_subarray();

    pre_allocated = true;
  }

  void resize(SIZE primary_count, SIZE dict_size, SIZE chunk_size,
              double estimated_outlier_ratio, int queue_idx) {

    outlier_count_array.resize({1}, queue_idx);
    outlier_idx_array.resize({(SIZE)(primary_count * estimated_outlier_ratio)},
                             queue_idx);
    outlier_array.resize({(SIZE)(primary_count * estimated_outlier_ratio)},
                         queue_idx);

    freq_array.resize({dict_size}, queue_idx);
    freq_array.hostAllocate(false);
    codebook_array.resize({dict_size}, queue_idx);
    size_t type_bw = sizeof(H) * 8;
    size_t decodebook_size = sizeof(H) * (2 * type_bw) + sizeof(Q) * dict_size;
    decodebook_array.resize({(SIZE)decodebook_size}, queue_idx);
    size_t nchunk = (primary_count - 1) / chunk_size + 1;
    huff_bitwidths_array.resize({(SIZE)nchunk}, queue_idx);
    condense_write_offsets_array.resize({(SIZE)nchunk}, queue_idx);
    condense_actual_lengths_array.resize({(SIZE)nchunk}, queue_idx);

    // Parallel deflate
    SIZE groups_per_chunk =
        (chunk_size - 1) / GetDeflateGroupSize<DeviceType>(chunk_size) + 1;
    SIZE ngroups = (SIZE)(nchunk * groups_per_chunk);
    deflate_group_bits_array.resize({ngroups}, queue_idx);
    deflate_group_offsets_array.resize({ngroups + 1}, queue_idx);
    deflate_chunk_words_array.resize({(SIZE)nchunk}, queue_idx);
    deflate_chunk_word_offsets_array.resize({(SIZE)nchunk + 1}, queue_idx);
    DeviceCollective<DeviceType>::ScanSumExtended(
        ngroups, SubArray<1, size_t, DeviceType>(),
        SubArray<1, size_t, DeviceType>(), deflate_group_scan_workspace, false,
        queue_idx);
    DeviceCollective<DeviceType>::ScanSumExtended(
        (SIZE)nchunk, SubArray<1, size_t, DeviceType>(),
        SubArray<1, size_t, DeviceType>(), deflate_chunk_scan_workspace, false,
        queue_idx);
    // Codebook
    first_nonzero_index_array.resize({1}, queue_idx);
    // first_nonzero_index_array.hostCopy(); // Create host allocation
    // // Allocate workspace
    DeviceCollective<DeviceType>::SortByKey(
        dict_size, SubArray<1, unsigned int, DeviceType>(),
        SubArray<1, Q, DeviceType>(), SubArray<1, unsigned int, DeviceType>(),
        SubArray<1, Q, DeviceType>(), sort_by_key_workspace, false, queue_idx);

    _d_freq_copy_array.resize({(SIZE)dict_size}, queue_idx);
    _d_qcode_copy_array.resize({(SIZE)dict_size}, queue_idx);
    CL_array.resize({dict_size}, queue_idx);
    CL_array.hostAllocate(false);
    lNodesLeader_array.resize({dict_size}, queue_idx);
    iNodesFreq_array.resize({dict_size}, queue_idx);
    iNodesLeader_array.resize({dict_size}, queue_idx);
    tempFreq_array.resize({dict_size}, queue_idx);
    tempIsLeaf_array.resize({dict_size}, queue_idx);
    tempIndex_array.resize({dict_size}, queue_idx);
    copyFreq_array.resize({dict_size}, queue_idx);
    copyIsLeaf_array.resize({dict_size}, queue_idx);
    copyIndex_array.resize({dict_size}, queue_idx);
    _d_codebook_array_org.resize({dict_size}, queue_idx);
    status_array.resize({(SIZE)16}, queue_idx);
    diagonal_path_intersections_array.resize(
        {GenerateCLKernel<unsigned int, DeviceType>::
             DiagonalPathIntersectionsSize(dict_size)},
        queue_idx);

    signature_verify_array.resize({(SIZE)7}, queue_idx);
    signature_verify_array.hostAllocate(false, queue_idx);

    // outlier_count_array.memset(0, queue_idx);
    // outlier_idx_array.memset(0, queue_idx);
    // outlier_array.memset(0, queue_idx);
    initialize_subarray();
    pre_allocated = true;
  }

  void reset(int queue_idx) {
    outlier_count_array.memset(0, queue_idx);
    freq_array.memset(0, queue_idx);
    codebook_array.memset(0, queue_idx);
    decodebook_array.memset(0xff, queue_idx);
    // huff_array.memset(0, queue_idx);
    huff_bitwidths_array.memset(0, queue_idx);
    first_nonzero_index_array.memset(0xff, queue_idx);
    CL_array.memset(0, queue_idx);
  }

  HuffmanWorkspace(SIZE primary_count, SIZE dict_size, SIZE chunk_size,
                   double estimated_outlier_ratio) {
    allocate(primary_count, dict_size, chunk_size, estimated_outlier_ratio);
  }

  bool pre_allocated;

  Array<1, ATOMIC_IDX, DeviceType> outlier_count_array;
  Array<1, ATOMIC_IDX, DeviceType> outlier_idx_array;
  Array<1, S, DeviceType> outlier_array;

  Array<1, unsigned int, DeviceType> freq_array;
  Array<1, H, DeviceType> codebook_array;
  Array<1, uint8_t, DeviceType> decodebook_array;
  Array<1, size_t, DeviceType> huff_bitwidths_array;
  Array<1, size_t, DeviceType> condense_write_offsets_array;
  Array<1, size_t, DeviceType> condense_actual_lengths_array;

  // Parallel deflate (cooperative bit-packing). group_bits holds per-group bit
  // sums; group_offsets is their extended exclusive scan. chunk_words holds
  // per-chunk H-word counts; chunk_word_offsets is their extended exclusive
  // scan (the last element is the total ddata word count).
  Array<1, size_t, DeviceType> deflate_group_bits_array;
  Array<1, size_t, DeviceType> deflate_group_offsets_array;
  Array<1, size_t, DeviceType> deflate_chunk_words_array;
  Array<1, size_t, DeviceType> deflate_chunk_word_offsets_array;
  Array<1, Byte, DeviceType> deflate_group_scan_workspace;
  Array<1, Byte, DeviceType> deflate_chunk_scan_workspace;

  // Codebook
  Array<1, unsigned int, DeviceType> first_nonzero_index_array;
  Array<1, Byte, DeviceType> sort_by_key_workspace;
  Array<1, unsigned int, DeviceType> _d_freq_copy_array;
  Array<1, Q, DeviceType> _d_qcode_copy_array;
  Array<1, unsigned int, DeviceType> CL_array;
  Array<1, int, DeviceType> lNodesLeader_array;
  Array<1, unsigned int, DeviceType> iNodesFreq_array;
  Array<1, int, DeviceType> iNodesLeader_array;
  Array<1, unsigned int, DeviceType> tempFreq_array;
  Array<1, int, DeviceType> tempIsLeaf_array;
  Array<1, int, DeviceType> tempIndex_array;
  Array<1, unsigned int, DeviceType> copyFreq_array;
  Array<1, int, DeviceType> copyIsLeaf_array;
  Array<1, int, DeviceType> copyIndex_array;
  Array<1, H, DeviceType> _d_codebook_array_org;
  Array<1, int, DeviceType, false, true> status_array;
  Array<1, uint32_t, DeviceType> diagonal_path_intersections_array;

  // Holds the deserialized signature during Verify. Only its pinned host copy
  // is used (read back via dataHost()); the device buffer is unused but comes
  // along with the Array and is freed automatically with the workspace.
  Array<1, Byte, DeviceType> signature_verify_array;

  SubArray<1, ATOMIC_IDX, DeviceType> outlier_count_subarray;
  SubArray<1, ATOMIC_IDX, DeviceType> outlier_idx_subarray;
  SubArray<1, S, DeviceType> outlier_subarray;

  SubArray<1, unsigned int, DeviceType> freq_subarray;
  SubArray<1, H, DeviceType> codebook_subarray;
  SubArray<1, uint8_t, DeviceType> decodebook_subarray;
  SubArray<1, size_t, DeviceType> huff_bitwidths_subarray;
  SubArray<1, size_t, DeviceType> condense_write_offsets_subarray;
  SubArray<1, size_t, DeviceType> condense_actual_lengths_subarray;

  SubArray<1, size_t, DeviceType> deflate_group_bits_subarray;
  SubArray<1, size_t, DeviceType> deflate_group_offsets_subarray;
  SubArray<1, size_t, DeviceType> deflate_chunk_words_subarray;
  SubArray<1, size_t, DeviceType> deflate_chunk_word_offsets_subarray;

  // Codebook
  SubArray<1, unsigned int, DeviceType> first_nonzero_index_subarray;
  SubArray<1, Byte, DeviceType> sort_by_key_workspace_subarray;
  SubArray<1, unsigned int, DeviceType> _d_freq_copy_subarray;
  SubArray<1, Q, DeviceType> _d_qcode_copy_subarray;
  SubArray<1, unsigned int, DeviceType> CL_subarray;
  SubArray<1, int, DeviceType> lNodesLeader_subarray;
  SubArray<1, unsigned int, DeviceType> iNodesFreq_subarray;
  SubArray<1, int, DeviceType> iNodesLeader_subarray;
  SubArray<1, unsigned int, DeviceType> tempFreq_subarray;
  SubArray<1, int, DeviceType> tempIsLeaf_subarray;
  SubArray<1, int, DeviceType> tempIndex_subarray;
  SubArray<1, unsigned int, DeviceType> copyFreq_subarray;
  SubArray<1, int, DeviceType> copyIsLeaf_subarray;
  SubArray<1, int, DeviceType> copyIndex_subarray;
  SubArray<1, H, DeviceType> _d_codebook_subarray_org;
  SubArray<1, int, DeviceType, false, true> status_subarray;
  SubArray<1, uint32_t, DeviceType> diagonal_path_intersections_subarray;
};

} // namespace mgard_x_160

#endif