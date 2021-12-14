/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#include "FillArraySequence.hpp"
#include "GetFirstNonzeroIndex.hpp"
#include "GenerateCL.hpp"
#include "GenerateCW.hpp"
#include "ReverseArray.hpp"
#include "ReorderByIndex.hpp"

#ifndef MGARD_X_GET_CODEBOOK_TEMPLATE_HPP
#define MGARD_X_GET_CODEBOOK_TEMPLATE_HPP

namespace mgard_x {
  // Parallel codebook generation wrapper
template <typename Q, typename H, typename DeviceType>
void GetCodebook(int dict_size, 
                    SubArray<1, unsigned int, DeviceType> _d_freq_subarray,
                    SubArray<1, H, DeviceType> _d_codebook_subarray,
                    SubArray<1, uint8_t, DeviceType> _d_decode_meta_subarray) {
  // Metadata
  auto type_bw = sizeof(H) * 8;
  // auto _d_first = reinterpret_cast<H *>(_d_decode_meta);
  // auto _d_entry = reinterpret_cast<H *>(_d_decode_meta + (sizeof(H) * type_bw));
  // auto _d_qcode =
  //     reinterpret_cast<Q *>(_d_decode_meta + (sizeof(H) * 2 * type_bw));

  // SubArray<1, uint8_t, DeviceType> _d_decode_meta_subarray(
  //         {(SIZE)(sizeof(H) * (2 * type_bw) + sizeof(Q) * dict_size)}, 
  //         _d_decode_meta);

  // SubArray<1, unsigned int, DeviceType> _d_freq_subarray({(SIZE)dict_size}, _d_freq);
  // SubArray<1, H, DeviceType> _d_codebook_subarray({(SIZE)dict_size}, _d_codebook);

  SubArray<1, H, DeviceType> _d_first_subarray({(SIZE)type_bw}, (H*)_d_decode_meta_subarray((IDX)0));
  SubArray<1, H, DeviceType> _d_entry_subarray({(SIZE)type_bw}, (H*)_d_decode_meta_subarray(sizeof(H) * type_bw));
  SubArray<1, Q, DeviceType> _d_qcode_subarray({(SIZE)dict_size}, (Q*)_d_decode_meta_subarray(sizeof(H) * 2 * type_bw));

  // Sort Qcodes by frequency
  int nblocks = (dict_size / 1024) + 1;
  FillArraySequence<Q, DeviceType>().Execute(_d_qcode_subarray, dict_size, 0);
  DeviceCollective<DeviceType>().SortByKey(dict_size, _d_freq_subarray, _d_qcode_subarray, 0);

  unsigned int *d_first_nonzero_index;
  unsigned int first_nonzero_index;
  Array<1, unsigned int, DeviceType> first_nonzero_index_array({1});
  first_nonzero_index_array.loadData((unsigned int*)&dict_size);

  SubArray<1, unsigned int, DeviceType> d_first_nonzero_index_subarray({1}, d_first_nonzero_index);
  GetFirstNonzeroIndex<unsigned int, DeviceType>().Execute(_d_freq_subarray, first_nonzero_index_array, dict_size, 0);

  DeviceRuntime<DeviceType>::SyncQueue(0);
  first_nonzero_index = first_nonzero_index_array.getDataHost()[0];

  if (debug_print_huffman) {
    PrintSubarray("SortByKey::_d_freq_subarray", _d_freq_subarray);
    PrintSubarray("SortByKey::_d_qcode_subarray", _d_qcode_subarray);
    // std::cout << "first_nonzero_index: " << first_nonzero_index << std::endl;
  }

  int nz_dict_size = dict_size - first_nonzero_index;
  // unsigned int *_nz_d_freq = _d_freq + first_nonzero_index;
  // H *_nz_d_codebook = _d_codebook + first_nonzero_index;
  int nz_nblocks = (nz_dict_size / 1024) + 1;

  SubArray<1, unsigned int, DeviceType> _nz_d_freq_subarray(
        {(SIZE)nz_dict_size}, _d_freq_subarray(first_nonzero_index));
  SubArray<1, H, DeviceType> _nz_d_codebook_subarray(
        {(SIZE)nz_dict_size}, _d_codebook_subarray(first_nonzero_index));

  Array<1, unsigned int, DeviceType> CL_array({(SIZE)nz_dict_size});
  Array<1, int, DeviceType> lNodesLeader_array({(SIZE)nz_dict_size});
  Array<1, unsigned int, DeviceType> iNodesFreq_array({(SIZE)nz_dict_size});
  Array<1, int, DeviceType> iNodesLeader_array({(SIZE)nz_dict_size});
  Array<1, unsigned int, DeviceType> tempFreq_array({(SIZE)nz_dict_size});
  Array<1, int, DeviceType> tempIsLeaf_array({(SIZE)nz_dict_size});
  Array<1, int, DeviceType> tempIndex_array({(SIZE)nz_dict_size});
  Array<1, unsigned int, DeviceType> copyFreq_array({(SIZE)nz_dict_size});
  Array<1, int, DeviceType> copyIsLeaf_array({(SIZE)nz_dict_size});
  Array<1, int, DeviceType> copyIndex_array({(SIZE)nz_dict_size});

  CL_array.memset(0);
  // unsigned int *CL         = CL_array.get_dv();//nullptr;
  // /*unsigned int* lNodesFreq*/         
  // int *lNodesLeader = lNodesLeader_array.get_dv(); //nullptr;
  // unsigned int *iNodesFreq = iNodesFreq_array.get_dv(); //nullptr;  
  // int *iNodesLeader = iNodesLeader_array.get_dv(); //nullptr;
  // unsigned int *tempFreq   = tempFreq_array.get_dv(); //nullptr;  
  // int *tempIsLeaf   = tempIsLeaf_array.get_dv(); //nullptr;  
  // int *tempIndex = tempIndex_array.get_dv(); //nullptr;
  // unsigned int *copyFreq   = copyFreq_array.get_dv(); //nullptr;  
  // int *copyIsLeaf   = copyIsLeaf_array.get_dv(); //nullptr;  
  // int *copyIndex = copyIndex_array.get_dv(); //nullptr;
  // cudaMemset(CL, 0,         nz_dict_size * sizeof(int)          );

  SIZE mblocks = (DeviceRuntime<DeviceType>::GetMaxNumThreadsPerSM() / 
            DeviceRuntime<DeviceType>::GetWarpSize()) * 
            DeviceRuntime<DeviceType>::GetNumSMs();
  uint32_t *diagonal_path_intersections;
  Array<1, uint32_t, DeviceType> diagonal_path_intersections_array({2 * (mblocks + 1)});

  // cudaDeviceSynchronize();
  DeviceRuntime<DeviceType>::SyncDevice();

  SubArray<1, unsigned int, DeviceType> CL_subarray(CL_array);
  SubArray<1, int, DeviceType> lNodesLeader_subarray(lNodesLeader_array);
  SubArray<1, unsigned int, DeviceType> iNodesFreq_subarray(iNodesFreq_array);
  SubArray<1, int, DeviceType> iNodesLeader_subarray(iNodesLeader_array);
  SubArray<1, unsigned int, DeviceType> tempFreq_subarray(tempFreq_array);
  SubArray<1, int, DeviceType> tempIsLeaf_subarray(tempIsLeaf_array);
  SubArray<1, int, DeviceType> tempIndex_subarray(tempIndex_array);
  SubArray<1, unsigned int, DeviceType> copyFreq_subarray(copyFreq_array);
  SubArray<1, int, DeviceType> copyIsLeaf_subarray(copyIsLeaf_array);
  SubArray<1, int, DeviceType> copyIndex_subarray(copyIndex_array);
  SubArray<1, uint32_t, DeviceType> diagonal_path_intersections_subarray(diagonal_path_intersections_array);

  GenerateCL<unsigned int, DeviceType> generateCL;
  generateCL.Execute(_nz_d_freq_subarray, CL_subarray, nz_dict_size,
                      _nz_d_freq_subarray, lNodesLeader_subarray, 
                      iNodesFreq_subarray, iNodesLeader_subarray,
                      tempFreq_subarray, tempIsLeaf_subarray, tempIndex_subarray, 
                      copyFreq_subarray, copyIsLeaf_subarray, copyIndex_subarray,
                      diagonal_path_intersections_subarray, 0);

  unsigned int max_CL;
  MemoryManager<DeviceType>().Copy1D(&max_CL, CL_subarray(IDX(0)), 1, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);


  // if (std::is_same<DeviceType, Serial>::value) {
  //   DumpSubArray("CL_subarray", CL_subarray);
  // }

  // if (std::is_same<DeviceType, HIP>::value) {
  //   LoadSubArray("CL_subarray", CL_subarray);
  // }


  if (debug_print_huffman) {
    PrintSubarray("GenerateCL::CL_subarray", CL_subarray);
    std::cout << "GenerateCL: max_CL" << max_CL << std::endl;
  }

  int max_CW_bits = (sizeof(H) * 8) - 8;
  if (max_CL > max_CW_bits) {
    std::cout << log::log_err << "Cannot store all Huffman codewords in "
         << max_CW_bits + 8 << "-bit representation" << std::endl;
    std::cout << log::log_err << "Huffman codeword representation requires at least "
         << max_CL + 8 << " bits (longest codeword: " << max_CL << " bits)"
         << std::endl;
    exit(1);
  }

  GenerateCW<unsigned int, H, DeviceType> generateCW;
  generateCW.Execute(CL_subarray, _nz_d_codebook_subarray,
                     _d_first_subarray, _d_entry_subarray, 
                     nz_dict_size, 0);

  // PrintSubarray("_d_entry_subarray", _d_entry_subarray);

  if (std::is_same<DeviceType, Serial>::value) {
    DumpSubArray("_nz_d_codebook_subarray", _nz_d_codebook_subarray);
  }

  if (std::is_same<DeviceType, HIP>::value) {
    LoadSubArray("_nz_d_codebook_subarray", _nz_d_codebook_subarray);
  }

  if (std::is_same<DeviceType, Serial>::value) {
    DumpSubArray("_d_first_subarray", _d_first_subarray);
  }

  if (std::is_same<DeviceType, HIP>::value) {
    LoadSubArray("_d_first_subarray", _d_first_subarray);
  }

  // if (std::is_same<DeviceType, Serial>::value) {
  //   DumpSubArray("_d_entry_subarray", _d_entry_subarray);
  // }

  // if (std::is_same<DeviceType, HIP>::value) {
  //   LoadSubArray("_d_entry_subarray", _d_entry_subarray);
  // }




  ReverseArray<H, DeviceType>().Execute(_d_codebook_subarray, dict_size, 0);
  ReverseArray<Q, DeviceType>().Execute(_d_qcode_subarray, dict_size, 0);

  Array<1, H, DeviceType> _d_codebook_array_org({_d_codebook_subarray.getShape(0)});
  _d_codebook_array_org.loadData(_d_codebook_subarray.data());
  SubArray _d_codebook_subarray_org(_d_codebook_array_org);
  ReorderByIndex<H, Q, DeviceType>().Execute(_d_codebook_subarray_org, _d_codebook_subarray, _d_qcode_subarray, dict_size, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);

  



}

}

#endif