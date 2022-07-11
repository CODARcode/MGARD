/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */
#ifndef MGARD_X_DEVICE_ADAPTER_H
#define MGARD_X_DEVICE_ADAPTER_H

#include "../DataStructures/SubArray.hpp"
#include "../DataTypes.h"
#include "../Functors/Functor.h"
#include "../Tasks/Task.h"
#include "../Utilities/Message.h"

namespace mgard_x {

struct ExecutionReturn {
  bool success = true;
  double execution_time = std::numeric_limits<double>::max();
};

template <typename DeviceType> struct SyncBlock {
  MGARDX_EXEC static void Sync();
};

template <typename DeviceType> struct SyncGrid {
  MGARDX_EXEC static void Sync();
};

#define AtomicSystemScope 0
#define AtomicDeviceScope 1
#define AtomicBlockScope 2

#define AtomicGlobalMemory 0
#define AtomicSharedMemory 1

#define RESOURCE_ENOUGH 0
#define THREADBLOCK_TOO_LARGE 1
#define SHARED_MEMORY_TOO_LARGE 2

template <typename T, OPTION MemoryType, OPTION Scope, typename DeviceType>
struct Atomic {
  MGARDX_EXEC static T Min(T *result, T value);
  MGARDX_EXEC static T Max(T *result, T value);
  MGARDX_EXEC static T Add(T *result, T value);
};

template <typename DeviceType> struct Math {
  template <typename T> MGARDX_EXEC static T Min(T a, T b);
  template <typename T> MGARDX_EXEC static T Max(T a, T b);

  MGARDX_EXEC static int ffs(unsigned int a);
  MGARDX_EXEC static int ffsll(long long unsigned int a);
  MGARDX_EXEC static uint64_t binary2negabinary(const int64_t x);
  MGARDX_EXEC static uint32_t binary2negabinary(const int32_t x);
  MGARDX_EXEC static int64_t negabinary2binary(const uint64_t x);
  MGARDX_EXEC static int32_t negabinary2binary(const uint32_t x);
};

template <typename T, SIZE nblockx, SIZE nblocky, SIZE nblockz,
          typename DeviceType>
struct BlockReduce {
  MGARDX_EXEC
  static void Sum(T intput, T &output);
  MGARDX_EXEC
  static void Max(T intput, T &output);
};

template <typename T_org, typename T_trans, SIZE nblockx, SIZE nblocky,
          SIZE nblockz, OPTION ALIGN, OPTION METHOD, typename DeviceType>
struct BlockBitTranspose {
  MGARDX_EXEC
  static void Transpose(T_org *v, T_trans *tv, SIZE b, SIZE B, SIZE IdX,
                        SIZE IdY);
};

template <typename T_org, typename T_trans, OPTION ALIGN, OPTION METHOD, SIZE b,
          SIZE B, typename DeviceType>
struct WarpBitTranspose {
  MGARDX_EXEC
  static void Transpose(T_org *v, SIZE inc_v, T_trans *tv, SIZE inc_tv,
                        SIZE LaneId);
};

template <typename T, typename T_fp, typename T_sfp, typename T_error,
          SIZE nblockx, SIZE nblocky, SIZE nblockz, OPTION METHOD,
          OPTION BinaryType, typename DeviceType>
struct BlockErrorCollect {
  MGARDX_EXEC
  static void Collect(T *v, T_error *temp, T_error *errors, SIZE num_elems,
                      SIZE num_bitplanes, SIZE IdX, SIZE IdY);
};

template <typename T, typename T_fp, typename T_sfp, typename T_error,
          OPTION METHOD, OPTION BinaryType, SIZE num_elems, SIZE num_bitplanes,
          typename DeviceType>
struct WarpErrorCollect {
  MGARDX_EXEC
  static void Collect(T *v, T_error *errors, SIZE LaneId);
};

template <typename DeviceType> class DeviceSpecification {
public:
  MGARDX_CONT
  DeviceSpecification() {}
  int NumDevices;
  int *MaxSharedMemorySize;
  int *WarpSize;
  int *NumSMs;
  int *ArchitectureGeneration;
  int *MaxNumThreadsPerSM;
  int *MaxNumThreadsPerTB;
  size_t *AvailableMemory;
  std::string *DeviceNames;
};

template <typename DeviceType> class DeviceQueues {
public:
  MGARDX_CONT
  DeviceQueues() {}

  template <typename QueueType>
  MGARDX_CONT QueueType GetQueue(int dev_id, SIZE queue_id) {}

  MGARDX_CONT
  void SyncQueue(int dev_id, SIZE queue_id) {}

  MGARDX_CONT
  void SyncAllQueues(int dev_id) {}

  MGARDX_CONT
  ~DeviceQueues() {}
};

template <typename TaskType, typename DeviceType> class DeviceAdapter {
public:
  MGARDX_CONT
  DeviceAdapter(){};
  MGARDX_CONT
  int IsResourceEnough() { return false; }
  MGARDX_CONT
  ExecutionReturn Execute(){};
};

template <typename KeyT, typename ValueT> struct KeyValueComparator {
  bool operator()(std::pair<KeyT, ValueT> a, std::pair<KeyT, ValueT> b) const {
    return a.first < b.first;
  }
};

template <typename DeviceType> class DeviceCollective {
public:
  template <typename T> MGARDX_CONT DeviceCollective(){};
  template <typename T>
  MGARDX_CONT static void Sum(SIZE n, SubArray<1, T, DeviceType> &v,
                              SubArray<1, T, DeviceType> &result,
                              int queue_idx);
  template <typename T>
  MGARDX_CONT static void AbsMax(SIZE n, SubArray<1, T, DeviceType> &v,
                                 SubArray<1, T, DeviceType> &result,
                                 int queue_idx);
  template <typename T>
  MGARDX_CONT static void SquareSum(SIZE n, SubArray<1, T, DeviceType> &v,
                                    SubArray<1, T, DeviceType> &result,
                                    int queue_idx);
  template <typename T>
  MGARDX_CONT static void
  ScanSumInclusive(SIZE n, SubArray<1, T, DeviceType> &v,
                   SubArray<1, T, DeviceType> &result, int queue_idx);
  template <typename T>
  MGARDX_CONT static void
  ScanSumExclusive(SIZE n, SubArray<1, T, DeviceType> &v,
                   SubArray<1, T, DeviceType> &result, int queue_idx);
  template <typename T>
  MGARDX_CONT static void ScanSumExtended(SIZE n, SubArray<1, T, DeviceType> &v,
                                          SubArray<1, T, DeviceType> &result,
                                          int queue_idx);
  template <typename KeyT, typename ValueT>
  MGARDX_CONT static void SortByKey(SIZE n, SubArray<1, KeyT, DeviceType> &keys,
                                    SubArray<1, ValueT, DeviceType> &values,
                                    int queue_idx);
};

template <typename DeviceType> class MemoryManager {
public:
  MGARDX_CONT
  MemoryManager(){};

  template <typename T>
  MGARDX_CONT static void Malloc1D(T *&ptr, SIZE n, int queue_idx);

  template <typename T>
  MGARDX_CONT static void MallocND(T *&ptr, SIZE n1, SIZE n2, SIZE &ld,
                                   int queue_idx);

  template <typename T> MGARDX_CONT static void Free(T *ptr);

  template <typename T>
  MGARDX_CONT static void Copy1D(T *dst_ptr, const T *src_ptr, SIZE n,
                                 int queue_idx);

  template <typename T>
  MGARDX_CONT static void CopyND(T *dst_ptr, SIZE dst_ld, const T *src_ptr,
                                 SIZE src_ld, SIZE n1, SIZE n2, int queue_idx);

  template <typename T>
  MGARDX_CONT static void MallocHost(T *&ptr, SIZE n, int queue_idx);

  template <typename T> MGARDX_CONT static void FreeHost(T *ptr);

  template <typename T>
  MGARDX_CONT static void Memset1D(T *ptr, SIZE n, int value, int queue_idx);

  template <typename T>
  MGARDX_CONT static void MemsetND(T *ptr, SIZE ld, SIZE n1, SIZE n2, int value,
                                   int queue_idx);

  template <typename T> MGARDX_CONT static bool IsDevicePointer(T *ptr);

  static bool ReduceMemoryFootprint;
};

template <typename DeviceType> class DeviceRuntime {
public:
  MGARDX_CONT
  DeviceRuntime() {}

  MGARDX_CONT static void SelectDevice(SIZE dev_id) {}

  template <typename QueueType>
  MGARDX_CONT static QueueType GetQueue(SIZE queue_id) {}

  MGARDX_CONT static void SyncQueue(SIZE queue_id) {}

  MGARDX_CONT static void SyncAllQueues() {}

  MGARDX_CONT static void SyncDevice() {}

  MGARDX_CONT static std::string GetDeviceName() { return ""; }

  MGARDX_CONT
  ~DeviceRuntime() {}

  static int curr_dev_id;
  static DeviceQueues<DeviceType> queues;
  static bool SyncAllKernelsAndCheckErrors;
  static DeviceSpecification<DeviceType> DeviceSpecs;
};

} // namespace mgard_x

#endif