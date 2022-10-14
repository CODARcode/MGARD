/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_HUFFMAN_WORKSPACE_HPP
#define MGARD_X_HUFFMAN_WORKSPACE_HPP

#include "../../Hierarchy/Hierarchy.h"
#include "../../RuntimeX/RuntimeXPublic.h"

namespace mgard_x {

template <typename Q, typename H, typename DeviceType> class HuffmanWorkspace {
public:
  HuffmanWorkspace() {
    // By defualt it is not pre-allocated
    pre_allocated = false;
  }

  void initialize_subarray() {
    freq_subarray = SubArray(freq_array);
    codebook_subarray = SubArray(codebook_array);
    decodebook_subarray = SubArray(decodebook_array);
    huff_subarray = SubArray(huff_array);
    huff_bitwidths_subarray = SubArray(huff_bitwidths_array);
    // Codebook
    first_nonzero_index_subarray = SubArray(first_nonzero_index_array);
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

  size_t estimate_size(SIZE primary_count, SIZE dict_size, SIZE chunk_size) {
    size_t size = 0;
    size += dict_size * sizeof(unsigned int);
    size += dict_size * sizeof(H);
    size_t type_bw = sizeof(H) * 8;
    size_t decodebook_size = sizeof(H) * (2 * type_bw) + sizeof(Q) * dict_size;
    size += decodebook_size * sizeof(uint8_t);
    size += primary_count * sizeof(H);
    size_t nchunk = (primary_count - 1) / chunk_size + 1;
    size += nchunk * sizeof(size_t);

    size += sizeof(unsigned int);
    size += sizeof(unsigned int) * dict_size * 4;
    size += sizeof(int) * dict_size * 6;
    size += sizeof(H) * dict_size;
    size += sizeof(int) * 16;
    SIZE mblocks = (DeviceRuntime<DeviceType>::GetMaxNumThreadsPerTB() /
                    DeviceRuntime<DeviceType>::GetWarpSize()) *
                   DeviceRuntime<DeviceType>::GetNumSMs();
    size += 2 * (mblocks + 1) * sizeof(uint32_t);
    return size;
  }

  void allocate(SIZE primary_count, SIZE dict_size, SIZE chunk_size) {
    freq_array = Array<1, unsigned int, DeviceType>({dict_size});
    codebook_array = Array<1, H, DeviceType>({dict_size});
    size_t type_bw = sizeof(H) * 8;
    size_t decodebook_size = sizeof(H) * (2 * type_bw) + sizeof(Q) * dict_size;
    decodebook_array = Array<1, uint8_t, DeviceType>({(SIZE)decodebook_size});
    huff_array = Array<1, H, DeviceType>({primary_count});
    size_t nchunk = (primary_count - 1) / chunk_size + 1;
    huff_bitwidths_array = Array<1, size_t, DeviceType>({(SIZE)nchunk});
    // Codebook
    first_nonzero_index_array = Array<1, unsigned int, DeviceType>({1});
    first_nonzero_index_array.hostCopy(); // Create host allocation
    CL_array = Array<1, unsigned int, DeviceType>({dict_size});
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
    status_array = Array<1, int, DeviceType>({(SIZE)16}, false, true);
    SIZE mblocks = (DeviceRuntime<DeviceType>::GetMaxNumThreadsPerTB() /
                    DeviceRuntime<DeviceType>::GetWarpSize()) *
                   DeviceRuntime<DeviceType>::GetNumSMs();
    diagonal_path_intersections_array =
        Array<1, uint32_t, DeviceType>({2 * (mblocks + 1)});
    initialize_subarray();

    pre_allocated = true;
  }

  void reset() {
    freq_array.memset(0);
    codebook_array.memset(0);
    decodebook_array.memset(0xff);
    huff_array.memset(0);
    huff_bitwidths_array.memset(0);
    first_nonzero_index_array.memset(0xff);
    CL_array.memset(0);
  }

  void move(const HuffmanWorkspace<Q, H, DeviceType> &workspace) {
    // Move instead of copy
    freq_array = std::move(workspace.freq_array);
    codebook_array = std::move(workspace.codebook_array);
    decodebook_array = std::move(workspace.decodebook_array);
    huff_array = std::move(workspace.huff_array);
    huff_bitwidths_array = std::move(workspace.huff_bitwidths_array);

    first_nonzero_index_array = std::move(workspace.first_nonzero_index_array);
    CL_array = std::move(workspace.CL_array);
    lNodesLeader_array = std::move(workspace.lNodesLeader_array);
    iNodesFreq_array = std::move(workspace.iNodesFreq_array);
    iNodesLeader_array = std::move(workspace.iNodesLeader_array);
    tempFreq_array = std::move(workspace.tempFreq_array);
    tempIsLeaf_array = std::move(workspace.tempIsLeaf_array);
    tempIndex_array = std::move(workspace.tempIndex_array);
    copyFreq_array = std::move(workspace.copyFreq_array);
    copyIsLeaf_array = std::move(workspace.copyIsLeaf_array);
    copyIndex_array = std::move(workspace.copyIndex_array);
    _d_codebook_array_org = std::move(workspace._d_codebook_array_org);
    status_array = std::move(workspace.status_array);
    diagonal_path_intersections_array =
        std::move(workspace.diagonal_path_intersections_array);
    initialize_subarray();
  }

  void move(HuffmanWorkspace<Q, H, DeviceType> &&workspace) {
    // Move instead of copy
    freq_array = std::move(workspace.freq_array);
    codebook_array = std::move(workspace.codebook_array);
    decodebook_array = std::move(workspace.decodebook_array);
    huff_array = std::move(workspace.huff_array);
    huff_bitwidths_array = std::move(workspace.huff_bitwidths_array);

    first_nonzero_index_array = std::move(workspace.first_nonzero_index_array);
    CL_array = std::move(workspace.CL_array);
    lNodesLeader_array = std::move(workspace.lNodesLeader_array);
    iNodesFreq_array = std::move(workspace.iNodesFreq_array);
    iNodesLeader_array = std::move(workspace.iNodesLeader_array);
    tempFreq_array = std::move(workspace.tempFreq_array);
    tempIsLeaf_array = std::move(workspace.tempIsLeaf_array);
    tempIndex_array = std::move(workspace.tempIndex_array);
    copyFreq_array = std::move(workspace.copyFreq_array);
    copyIsLeaf_array = std::move(workspace.copyIsLeaf_array);
    copyIndex_array = std::move(workspace.copyIndex_array);
    _d_codebook_array_org = std::move(workspace._d_codebook_array_org);
    status_array = std::move(workspace.status_array);
    diagonal_path_intersections_array =
        std::move(workspace.diagonal_path_intersections_array);
    initialize_subarray();
  }

  HuffmanWorkspace(SIZE primary_count, SIZE dict_size, SIZE chunk_size) {
    allocate(primary_count, dict_size, chunk_size);
  }

  HuffmanWorkspace(const HuffmanWorkspace<Q, H, DeviceType> &workspace) {
    move(std::move(workspace));
  }

  HuffmanWorkspace &
  operator=(const HuffmanWorkspace<Q, H, DeviceType> &workspace) {
    move(std::move(workspace));
  }

  HuffmanWorkspace(HuffmanWorkspace<Q, H, DeviceType> &&workspace) {
    move(std::move(workspace));
  }

  HuffmanWorkspace &operator=(HuffmanWorkspace<Q, H, DeviceType> &&workspace) {
    move(std::move(workspace));
  }

  bool pre_allocated;

  Array<1, unsigned int, DeviceType> freq_array;
  Array<1, H, DeviceType> codebook_array;
  Array<1, uint8_t, DeviceType> decodebook_array;
  Array<1, H, DeviceType> huff_array;
  Array<1, size_t, DeviceType> huff_bitwidths_array;

  // Codebook
  Array<1, unsigned int, DeviceType> first_nonzero_index_array;
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
  Array<1, int, DeviceType> status_array;
  Array<1, uint32_t, DeviceType> diagonal_path_intersections_array;

  SubArray<1, unsigned int, DeviceType> freq_subarray;
  SubArray<1, H, DeviceType> codebook_subarray;
  SubArray<1, uint8_t, DeviceType> decodebook_subarray;
  SubArray<1, H, DeviceType> huff_subarray;
  SubArray<1, size_t, DeviceType> huff_bitwidths_subarray;

  // Codebook
  SubArray<1, unsigned int, DeviceType> first_nonzero_index_subarray;
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
  SubArray<1, int, DeviceType> status_subarray;
  SubArray<1, uint32_t, DeviceType> diagonal_path_intersections_subarray;
};

} // namespace mgard_x

#endif