/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#include "../../RuntimeX/Utilities/Exceptions.h"
#include "EntropyCalculator.hpp"
#include "FillArraySequence.hpp"
#include "GenerateCL.hpp"
#include "GenerateCW.hpp"
#include "GetFirstNonzeroIndex.hpp"
#include "HuffmanWorkspace.hpp"
#include "ReorderByIndex.hpp"
#include "ReverseArray.hpp"

#include <stdexcept>
#include <string>

#ifndef MGARD_X_160_GET_CODEBOOK_TEMPLATE_HPP
#define MGARD_X_160_GET_CODEBOOK_TEMPLATE_HPP

namespace mgard_x_160 {
// Parallel codebook generation wrapper
template <typename Q, typename S, typename H, typename DeviceType>
void GetCodebook(int dict_size, size_t primary_count,
                 SubArray<1, unsigned int, DeviceType> _d_freq_subarray,
                 SubArray<1, H, DeviceType> _d_codebook_subarray,
                 SubArray<1, uint8_t, DeviceType> _d_decode_meta_subarray,
                 HuffmanWorkspace<Q, S, H, DeviceType> &workspace,
                 int queue_idx) {
  // Metadata
  auto type_bw = sizeof(H) * 8;

  SubArray<1, H, DeviceType> _d_first_subarray(
      {(SIZE)type_bw}, (H *)_d_decode_meta_subarray((IDX)0));
  SubArray<1, H, DeviceType> _d_entry_subarray(
      {(SIZE)type_bw}, (H *)_d_decode_meta_subarray(sizeof(H) * type_bw));
  SubArray<1, Q, DeviceType> _d_qcode_subarray(
      {(SIZE)dict_size}, (Q *)_d_decode_meta_subarray(sizeof(H) * 2 * type_bw));

  // Sort Qcodes by frequency
  DeviceLauncher<DeviceType>::Execute(
      FillArraySequenceKernel(_d_qcode_subarray), queue_idx);

  MemoryManager<DeviceType>::Copy1D(workspace._d_freq_copy_subarray.data(),
                                    _d_freq_subarray.data(), dict_size,
                                    queue_idx);
  MemoryManager<DeviceType>::Copy1D(workspace._d_qcode_copy_subarray.data(),
                                    _d_qcode_subarray.data(), dict_size,
                                    queue_idx);
  DeviceCollective<DeviceType>::SortByKey(
      (SIZE)dict_size, workspace._d_freq_copy_subarray,
      workspace._d_qcode_copy_subarray, _d_freq_subarray, _d_qcode_subarray,
      workspace.sort_by_key_workspace, true, queue_idx);

  DeviceLauncher<DeviceType>::Execute(
      GetFirstNonzeroIndexKernel<unsigned int, DeviceType>(
          _d_freq_subarray, workspace.first_nonzero_index_subarray),
      queue_idx);

  unsigned int first_nonzero_index;
  MemoryManager<DeviceType>().Copy1D(
      &first_nonzero_index, workspace.first_nonzero_index_subarray(IDX(0)), 1,
      queue_idx);
  DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

  if (debug_print_huffman) {
    PrintSubarray("SortByKey::_d_freq_subarray", _d_freq_subarray);
    PrintSubarray("SortByKey::_d_qcode_subarray", _d_qcode_subarray);
    // std::cout << "first_nonzero_index: " << first_nonzero_index << std::endl;
  }

  // DumpSubArray("freq_"+std::to_string(workspace.huff_array.shape(0))+".dat",
  // _d_freq_subarray);

  int nz_dict_size = dict_size - first_nonzero_index;

  SubArray<1, unsigned int, DeviceType> _nz_d_freq_subarray(
      {(SIZE)nz_dict_size}, _d_freq_subarray(first_nonzero_index));
  SubArray<1, H, DeviceType> _nz_d_codebook_subarray(
      {(SIZE)nz_dict_size}, _d_codebook_subarray(first_nonzero_index));

  DeviceLauncher<DeviceType>::Execute(
      GenerateCLKernel<unsigned int, DeviceType>(
          _nz_d_freq_subarray, workspace.CL_subarray, nz_dict_size,
          _nz_d_freq_subarray, workspace.lNodesLeader_subarray,
          workspace.iNodesFreq_subarray, workspace.iNodesLeader_subarray,
          workspace.tempFreq_subarray, workspace.tempIsLeaf_subarray,
          workspace.tempIndex_subarray, workspace.copyFreq_subarray,
          workspace.copyIsLeaf_subarray, workspace.copyIndex_subarray,
          workspace.diagonal_path_intersections_subarray,
          workspace.status_subarray),
      queue_idx);

  unsigned int max_CL;
  MemoryManager<DeviceType>().Copy1D(&max_CL, workspace.CL_subarray(IDX(0)), 1,
                                     queue_idx);
  DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

  if (log::level & log::INFO) {
    // PrintSubarray("GenerateCL::CL_subarray", workspace.CL_subarray);
    // std::cout << "GenerateCL: max_CL: " << max_CL << std::endl;
    double LC = CalculateLC(primary_count, nz_dict_size, _nz_d_freq_subarray,
                            workspace.CL_subarray, queue_idx);
    double entropy = CalculateEntropy(primary_count, nz_dict_size,
                                      _nz_d_freq_subarray, queue_idx);
    log::dbg("LC: " + std::to_string(LC));
    log::dbg("Entropy: " + std::to_string(entropy));
  }

  // DumpSubArray("cl_"+std::to_string(workspace.huff_array.shape(0))+".dat",
  // _d_freq_subarray);

  int max_CW_bits = (sizeof(H) * 8) - 8;
  if (max_CL > max_CW_bits) {
    // Throw (instead of exit) so callers can catch and fall back to another
    // lossless backend (e.g. raw Zstd) or retry with a smaller huff_dict_size.
    // A longer dictionary makes the tree deeper, so a degenerate/low-entropy
    // input can produce codewords exceeding the H-type budget (sizeof(H)*8 -
    // 8).
    throw ProcessingException(
        "Cannot store all Huffman codewords in " +
        std::to_string(max_CW_bits + 8) +
        "-bit representation; representation requires at least " +
        std::to_string(max_CL + 8) +
        " bits (longest codeword: " + std::to_string(max_CL) + " bits)");
  }

  DeviceLauncher<DeviceType>::Execute(
      GenerateCWKernel<unsigned int, H, DeviceType>(
          workspace.CL_subarray, _nz_d_codebook_subarray, _d_first_subarray,
          _d_entry_subarray, nz_dict_size, workspace.status_subarray),
      queue_idx);

  DeviceLauncher<DeviceType>::Execute(
      ReverseArrayKernel<H, DeviceType>(_d_codebook_subarray), queue_idx);
  DeviceLauncher<DeviceType>::Execute(
      ReverseArrayKernel<Q, DeviceType>(_d_qcode_subarray), queue_idx);

  MemoryManager<DeviceType>().Copy1D(workspace._d_codebook_subarray_org.data(),
                                     _d_codebook_subarray.data(),
                                     _d_codebook_subarray.shape(0), queue_idx);

  DeviceLauncher<DeviceType>::Execute(
      ReorderByIndexKernel<H, Q, DeviceType>(workspace._d_codebook_subarray_org,
                                             _d_codebook_subarray,
                                             _d_qcode_subarray),
      queue_idx);
  DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
}

} // namespace mgard_x_160

#endif