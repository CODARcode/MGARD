/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */
#ifndef MGARD_CUDA_DEVICE_ADAPTER
#define MGARD_CUDA_DEVICE_ADAPTER

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



template <typename HandleType, typename TaskType, typename DeviceType>
class DeviceAdapter {
public:
  MGARDm_CONT
  DeviceAdapter(HandleType& handle):handle(handle){};
  MGARDm_CONT
  void Execute() {};
  HandleType& handle;
};




template <typename HandleType, typename T_reduce, typename DeviceType>
class DeviceReduce {
public:
  MGARDm_CONT
  DeviceReduce(HandleType& handle):handle(handle){};
  MGARDm_CONT
  void Sum(SIZE n, SubArray<1, T_reduce>& v, SubArray<1, T_reduce>& result, int queue_idx);
  MGARDm_CONT
  void AbsMax(SIZE n, SubArray<1, T_reduce>& v, SubArray<1, T_reduce>& result, int queue_idx);
private:
  HandleType& handle;
};


}

#endif