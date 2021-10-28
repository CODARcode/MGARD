/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */
#ifndef MGARD_CUDA_DEVICE_ADAPTER_H
#define MGARD_CUDA_DEVICE_ADAPTER_H

#include "../SubArray.hpp"
#include "../Functor.h"
#include "../Task.h"

namespace mgard_cuda {


template <typename DeviceType> 
struct SyncThreads {
  MGARDm_EXEC 
  void operator()();
};

template <typename T, SIZE nblockx, SIZE nblocky, SIZE nblockz, typename DeviceType> 
struct BlockReduce {
  MGARDm_EXEC 
  T Sum(T intput);
  MGARDm_EXEC 
  T Max(T intput);
};


template <typename T, typename DeviceType> 
struct BlockBroadcast {
  MGARDm_EXEC 
  T Broadcast(T input, SIZE src_threadx, SIZE src_thready, SIZE src_threadz);
};

template <typename T, OPTION METHOD, typename DeviceType>
struct EncodeSignBits{
  MGARDm_EXEC 
  T Encode(T sign, SIZE b_idx);
};

template <typename T, OPTION METHOD, typename DeviceType>
struct DecodeSignBits{
  MGARDm_EXEC 
  T Decode(T sign_bitplane, SIZE b_idx);
};

template <typename T_org, typename T_trans, OPTION ALIGN, OPTION METHOD, typename DeviceType> 
struct WarpBitTranspose{
  MGARDm_EXEC 
  void Transpose(T_org * v, SIZE inc_v, T_trans * tv, SIZE inc_tv, SIZE b, SIZE B);

  MGARDm_EXEC 
  T_trans Transpose(T_org v, SIZE b, SIZE B);
};

template <typename T_org, typename T_trans, SIZE nblockx, SIZE nblocky, SIZE nblockz, OPTION ALIGN, OPTION METHOD, typename DeviceType> 
struct BlockBitTranspose{
  MGARDm_EXEC 
  void Transpose(T_org * v, T_trans * tv, SIZE b, SIZE B);
};

template <typename T, typename T_fp, typename T_sfp, typename T_error, OPTION METHOD, OPTION BinaryType, typename DeviceType> 
struct WarpErrorCollect{
  MGARDm_EXEC 
  void Collect(T * v, T_error * errors, SIZE num_elems, SIZE num_bitplanes);

};


template <typename T, typename T_fp, typename T_sfp, typename T_error, SIZE nblockx, SIZE nblocky, SIZE nblockz, OPTION METHOD, OPTION BinaryType, typename DeviceType> 
struct ErrorCollect{
  MGARDm_EXEC 
  void Collect(T * v, T_error * temp, T_error * errors, SIZE num_elems, SIZE num_bitplanes);
};



template <typename DeviceType>
class DeviceSpecification {
  public:
  MGARDm_CONT
  DeviceSpecification(){}
  int MaxSharedMemorySize;
  int WarpSize;
  int NumSMs;
  int ArchitectureGeneration;
};


template <typename DeviceType>
class DeviceQueues {
  public:
  MGARDm_CONT
  DeviceQueues(){}

  template <typename QueueType>
  MGARDm_CONT
  QueueType GetQueue(int dev_id, SIZE queue_id){}

  MGARDm_CONT
  void SyncQueue(int dev_id, SIZE queue_id){}

  MGARDm_CONT
  void SyncAllQueues(int dev_id){}

  MGARDm_CONT
  ~DeviceQueues(){}
};



template <typename TaskType, typename DeviceType>
class DeviceAdapter {
public:
  MGARDm_CONT
  DeviceAdapter(){};
  MGARDm_CONT
  void Execute() {};
};




template <typename T_reduce, typename DeviceType>
class DeviceCollective {
  public:
  MGARDm_CONT
  DeviceCollective(){};
  MGARDm_CONT
  void Sum(SIZE n, SubArray<1, T_reduce, DeviceType>& v, SubArray<1, T_reduce, DeviceType>& result, int queue_idx);
  MGARDm_CONT
  void AbsMax(SIZE n, SubArray<1, T_reduce, DeviceType>& v, SubArray<1, T_reduce, DeviceType>& result, int queue_idx);
  MGARDm_CONT
  void ScanSumInclusive(SIZE n, SubArray<1, T_reduce, DeviceType>& v, SubArray<1, T_reduce, DeviceType>& result, int queue_idx);
  MGARDm_CONT
  void ScanSumExclusive(SIZE n, SubArray<1, T_reduce, DeviceType>& v, SubArray<1, T_reduce, DeviceType>& result, int queue_idx);
  MGARDm_CONT
  void ScanSumExtended(SIZE n, SubArray<1, T_reduce, DeviceType>& v, SubArray<1, T_reduce, DeviceType>& result, int queue_idx);
};





template <typename DeviceType>
class MemoryManager {
  public:
  MGARDm_CONT
  MemoryManager(){};

  template <typename T>
  MGARDm_CONT
  void Malloc1D(T *& ptr, SIZE n, int queue_idx);

  template <typename T>
  MGARDm_CONT
  void MallocND(T *& ptr, SIZE n1, SIZE n2, SIZE &ld, int queue_idx);

  template <typename T>
  MGARDm_CONT
  void Free(T * ptr);

  template <typename T>
  MGARDm_CONT
  void Copy1D(T * dst_ptr, const T * src_ptr, SIZE n, int queue_idx);

  template <typename T>
  MGARDm_CONT
  void CopyND(T * dst_ptr, SIZE dst_ld, const T * src_ptr, SIZE src_ld, SIZE n1, SIZE n2, int queue_idx);

  template <typename T>
  MGARDm_CONT
  void MallocHost(T *& ptr, SIZE n, int queue_idx);

  template <typename T>
  MGARDm_CONT
  void FreeHost(T * ptr);

  static bool ReduceMemoryFootprint;
};


template <typename DeviceType>
class DeviceRuntime {
  public:
  MGARDm_CONT
  DeviceRuntime(){}

  MGARDm_CONT static
  void SelectDevice(SIZE dev_id){}

  template <typename QueueType>
  MGARDm_CONT static
  QueueType GetQueue(SIZE queue_id){}

  MGARDm_CONT static
  void SyncQueue(SIZE queue_id){}

  MGARDm_CONT static
  void SyncAllQueues(){}

  MGARDm_CONT static
  void SyncDevice(){}

  MGARDm_CONT
  ~DeviceRuntime(){}

  static int curr_dev_id;
  static DeviceQueues<DeviceType> queues;
  static bool SyncAllKernelsAndCheckErrors;
  static DeviceSpecification<DeviceType> DeviceSpecs;
};

}

#include "DeviceAdapterCuda.h"

#endif