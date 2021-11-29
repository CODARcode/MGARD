#ifndef MGARD_X_DEVICE_ADAPTER_SERIAL_H
#define MGARD_X_DEVICE_ADAPTER_SERIAL_H

#include "DeviceAdapter.h"
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <utility>
#include <algorithm>
#include <climits>
#include <cmath>

namespace mgard_x {

template<>
struct SyncBlock<Serial> {
  MGARDX_EXEC static void Sync() {
    // do nothing
  }
};

template<>
struct SyncGrid<Serial> {
  MGARDX_EXEC static void Sync() {
    // do nothing
  }
};

template <> 
struct Atomic<Serial> {
  template <typename T>
  MGARDX_EXEC static
  T Min(T * result, T value) {
    T old = *result;
    *result = std::min(*result, value);
    return old;
  }
  template <typename T>
  MGARDX_EXEC static
  T Max(T * result, T value) {
    T old = *result;
    *result = std::max(*result, value);
    return old;
  }
  template <typename T>
  MGARDX_EXEC static
  T Add(T * result, T value) {
    T old = *result;
    * result += value;
    return old;
  }
};

template <> 
struct Math<Serial> {
  template <typename T>
  MGARDX_EXEC static
  T Min(T a, T b) {
    return std::min(a, b);
  }
  template <typename T>
  MGARDX_EXEC static
  T Max(T a, T b) {
    return std::max(a, b);
  }

  MGARDX_EXEC static
  int ffsl(unsigned int a) {
    return ffs(a);
  }
  MGARDX_EXEC static
  int ffsll(long long unsigned int a) {
    return ffsll(a);
  }
};


#define INIT_BLOCK \
  for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {\
    for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {\
      for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {\
        Task &thread = threads[threadz * task.GetBlockDimY() * task.GetBlockDimX() + \
                thready * task.GetBlockDimX() + threadx];\
        thread = task;\
        thread.GetFunctor().Init(task.GetGridDimZ(), task.GetGridDimY(), task.GetGridDimX(),\
                                 task.GetBlockDimZ(), task.GetBlockDimY(), task.GetBlockDimX(),\
                                 blockz, blocky, blockx, threadz, thready, threadx,\
                                 shared_memory);\
      }\
    }\
  }

#define COMPUTE_BLOCK(OPERATION) \
  for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {\
    for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {\
      for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {\
        Task &thread = threads[threadz * task.GetBlockDimY() * task.GetBlockDimX() + \
                thready * task.GetBlockDimX() + threadx];\
        thread.GetFunctor().OPERATION();\
      }\
    }\
  }

#define COMPUTE_CONDITION_BLOCK(CONDITION_VAR, CONDITION_OP) \
  CONDITION_VAR = threads[0].GetFunctor().CONDITION_OP(); \
  for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {\
    for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {\
      for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {\
        Task &thread = threads[threadz * task.GetBlockDimY() * task.GetBlockDimX() + \
                thready * task.GetBlockDimX() + threadx];\
        if(CONDITION_VAR != thread.GetFunctor().CONDITION_OP()) { \
          std::cout << log::log_err << "IterKernel<Serial> inconsistant condition."; \
          exit(-1); \
        } \
      }\
    }\
  }

#define INIT_GRID \
  Task ****** threads = new Task*****[task.GetBlockDimZ()];\
  Byte ******* shared_memory = new Byte******[task.GetBlockDimZ()];\
  for (SIZE blockz = 0; blockz < task.GetGridDimZ(); blockz++) {\
    threads[blockz] = new Task****[task.GetBlockDimY()];\
    shared_memory[blockz] = new Byte*****[task.GetBlockDimY()];\
    for (SIZE blocky = 0; blocky < task.GetGridDimY(); blocky++) {\
      threads[blockz][blocky] = new Task***[task.GetBlockDimX()];\
      shared_memory[blockz][blocky] = new Byte****[task.GetBlockDimX()];\
      for (SIZE blockx = 0; blockx < task.GetGridDimX(); blockx++) {\
        threads[blockz][blocky][blockx] = new Task**[task.GetBlockDimZ()];\
        shared_memory[blockz][blocky][blockx] = new Byte***[task.GetBlockDimZ()];\
        for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {\
          threads[blockz][blocky][blockx][threadz] = new Task*[task.GetBlockDimY()];\
          shared_memory[blockz][blocky][blockx][threadz] = new Byte**[task.GetBlockDimY()];\
          for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {\
            threads[blockz][blocky][blockx][threadz][thready] = new Task[task.GetBlockDimX()];\
            shared_memory[blockz][blocky][blockx][threadz][thready] = new Byte*[task.GetBlockDimX()];\
            for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {\
              threads[blockz][blocky][blockx][threadz][thready][threadx] = task;\
              shared_memory[blockz][blocky][blockx][threadz][thready][threadx] = new Byte[task.GetSharedMemorySize()];\
              threads[blockz][blocky][blockx][threadz][thready][threadx].GetFunctor().Init(\
                                 task.GetGridDimZ(), task.GetGridDimY(), task.GetGridDimX(),\
                                 task.GetBlockDimZ(), task.GetBlockDimY(), task.GetBlockDimX(),\
                                 blockz, blocky, blockx, threadz, thready, threadx,\
                                 shared_memory[blockz][blocky][blockx][threadz][thready][threadx]);\
            }\
          }\
        }\
      }\
    }\
  }

#define COMPUTE_GRID(OPERATION) \
  for (SIZE blockz = 0; blockz < task.GetGridDimZ(); blockz++) {\
    for (SIZE blocky = 0; blocky < task.GetGridDimY(); blocky++) {\
      for (SIZE blockx = 0; blockx < task.GetGridDimX(); blockx++) {\
        for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {\
          for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {\
            for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {\
              Task &thread = threads[blockz][blocky][blockx][threadz][thready][threadx];\
              thread.GetFunctor().OPERATION();\
            }\
          }\
        }\
      }\
    }\
  }

#define COMPUTE_CONDITION_GRID(CONDITION_VAR, CONDITION_OP) \
  CONDITION_VAR = threads[0][0][0][0][0].GetFunctor().CONDITION_OP(); \
  for (SIZE blockz = 0; blockz < task.GetGridDimZ(); blockz++) {\
    for (SIZE blocky = 0; blocky < task.GetGridDimY(); blocky++) {\
      for (SIZE blockx = 0; blockx < task.GetGridDimX(); blockx++) {\
        for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {\
          for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {\
            for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {\
              Task &thread = threads[blockz][blocky][blockx][threadz][thready][threadx];\
              if(CONDITION_VAR != thread.GetFunctor().CONDITION_OP()) { \
                std::cout << log::log_err << "IterKernel<Serial> inconsistant condition."; \
                exit(-1); \
              } \
            }\
          }\
        }\
      }\
    }\
  }

#define FINALIZE_GRID \
  for (SIZE blockz = 0; blockz < task.GetGridDimZ(); blockz++) {\
    for (SIZE blocky = 0; blocky < task.GetGridDimY(); blocky++) {\
      for (SIZE blockx = 0; blockx < task.GetGridDimX(); blockx++) {\
        for (SIZE threadz = 0; threadz < task.GetBlockDimZ(); threadz++) {\
          for (SIZE thready = 0; thready < task.GetBlockDimY(); thready++) {\
            for (SIZE threadx = 0; threadx < task.GetBlockDimX(); threadx++) {\
              delete [] shared_memory[blockz][blocky][blockx][threadz][thready][threadx];\
            }\
            delete [] threads[blockz][blocky][blockx][threadz][thready];\
            delete [] shared_memory[blockz][blocky][blockx][threadz][thready];\
          }\
          delete [] threads[blockz][blocky][blockx][threadz];\
          delete [] shared_memory[blockz][blocky][blockx][threadz];\
        }\
        delete [] threads[blockz][blocky][blockx];\
        delete [] shared_memory[blockz][blocky][blockx];\
      }\
      delete [] threads[blockz][blocky];\
      delete [] shared_memory[blockz][blocky];\
    }\
    delete [] threads[blockz];\
    delete [] shared_memory[blockz];\
  }\
  delete [] threads;\
  delete [] shared_memory;

template <typename Task>
MGARDX_KERL void SerialKernel(Task task) {
  Byte *shared_memory = new Byte[task.GetSharedMemorySize()];

  Task * threads = new Task[task.GetBlockDimX() *
                            task.GetBlockDimY() *
                            task.GetBlockDimZ()];

  for (SIZE blockz = 0; blockz < task.GetGridDimZ(); blockz++) {
    for (SIZE blocky = 0; blocky < task.GetGridDimY(); blocky++) {
      for (SIZE blockx = 0; blockx < task.GetGridDimX(); blockx++) {
        INIT_BLOCK;
        COMPUTE_BLOCK(Operation1);
        COMPUTE_BLOCK(Operation2);
        COMPUTE_BLOCK(Operation3);
        COMPUTE_BLOCK(Operation4);
        COMPUTE_BLOCK(Operation5);
        COMPUTE_BLOCK(Operation6);
        COMPUTE_BLOCK(Operation7);
        COMPUTE_BLOCK(Operation8);
        COMPUTE_BLOCK(Operation9);
        COMPUTE_BLOCK(Operation10);
      }
    }
  }
  delete [] shared_memory;
  delete [] threads;
}


template <typename Task>
MGARDX_KERL void SerialIterKernel(Task task) {
  Byte *shared_memory = new Byte[task.GetSharedMemorySize()];

  Task * threads = new Task[task.GetBlockDimX() *
                            task.GetBlockDimY() *
                            task.GetBlockDimZ()];
  bool condition1 = false;
  bool condition2 = false;

  for (SIZE blockz = 0; blockz < task.GetGridDimZ(); blockz++) {
    for (SIZE blocky = 0; blocky < task.GetGridDimY(); blocky++) {
      for (SIZE blockx = 0; blockx < task.GetGridDimX(); blockx++) {
        INIT_BLOCK;
        COMPUTE_BLOCK(Operation1);
        COMPUTE_BLOCK(Operation2);
        COMPUTE_CONDITION_BLOCK(condition1, LoopCondition1);
        while (condition1) {
          COMPUTE_BLOCK(Operation3);
          COMPUTE_BLOCK(Operation4);
          COMPUTE_BLOCK(Operation5);
          COMPUTE_BLOCK(Operation6);
          COMPUTE_CONDITION_BLOCK(condition1, LoopCondition1);
        }
        COMPUTE_BLOCK(Operation7);
        COMPUTE_BLOCK(Operation8);
        COMPUTE_BLOCK(Operation9);
        COMPUTE_BLOCK(Operation10);
        COMPUTE_CONDITION_BLOCK(condition2, LoopCondition2);
        while (condition2) {
          COMPUTE_BLOCK(Operation11);
          COMPUTE_BLOCK(Operation12);
          COMPUTE_BLOCK(Operation13);
          COMPUTE_BLOCK(Operation14);
          COMPUTE_CONDITION_BLOCK(condition2, LoopCondition2);
        }
        COMPUTE_BLOCK(Operation15);
        COMPUTE_BLOCK(Operation16);
        COMPUTE_BLOCK(Operation17);
      }
    }
  }
  delete [] shared_memory;
  delete [] threads;
}

template <typename Task>
MGARDX_KERL void SerialHuffmanCLCustomizedKernel(Task task) {

  bool loop_condition1 = false;
  bool loop_condition2 = false;
  bool branch_condition1 = false;

  INIT_GRID;
  COMPUTE_GRID(Operation1);
  COMPUTE_CONDITION_GRID(loop_condition1, LoopCondition1);
  while (loop_condition1) {
    COMPUTE_GRID(Operation2);
    COMPUTE_GRID(Operation3);
    COMPUTE_GRID(Operation4);
    COMPUTE_GRID(Operation5);
    COMPUTE_CONDITION_GRID(branch_condition1, BranchCondition1);
    if (branch_condition1) {
      COMPUTE_CONDITION_GRID(loop_condition2, LoopCondition2);
      while (loop_condition2) {
        COMPUTE_GRID(Operation6);
        COMPUTE_GRID(Operation7);
        COMPUTE_GRID(Operation8);
        COMPUTE_CONDITION_GRID(loop_condition2, LoopCondition2);
      }
      COMPUTE_GRID(Operation9)
      COMPUTE_GRID(Operation10);;
    }

    COMPUTE_GRID(Operation11);
    COMPUTE_GRID(Operation12);
    COMPUTE_GRID(Operation13);
    COMPUTE_GRID(Operation14);

    COMPUTE_CONDITION_GRID(loop_condition1, LoopCondition1);
  }

  FINALIZE_GRID;
}


template <typename Task>
MGARDX_KERL void SerialHuffmanCWCustomizedKernel(Task task) {

  bool loop_condition1 = false;

  INIT_GRID;
  COMPUTE_GRID(Operation1);
  COMPUTE_GRID(Operation2);
  COMPUTE_GRID(Operation3);
  COMPUTE_CONDITION_GRID(loop_condition1, LoopCondition1);
  while (loop_condition1) {
    COMPUTE_GRID(Operation4);
    COMPUTE_GRID(Operation5);
    COMPUTE_GRID(Operation6);
    COMPUTE_GRID(Operation7);
    COMPUTE_GRID(Operation8);
    COMPUTE_CONDITION_GRID(loop_condition1, LoopCondition1);
  }
  COMPUTE_GRID(Operation9)
  COMPUTE_GRID(Operation10);;

  FINALIZE_GRID;
}




template <>
class DeviceSpecification<Serial> {
  public:
  MGARDX_CONT
  DeviceSpecification(){
    NumDevices = 1;
    MaxSharedMemorySize = new int[NumDevices];
    WarpSize = new int[NumDevices];
    NumSMs = new int[NumDevices];
    ArchitectureGeneration = new int[NumDevices];
    MaxNumThreadsPerSM = new int[NumDevices];

    for (int d = 0; d < NumDevices; d++) {
      MaxSharedMemorySize[d] = 1e6;
      WarpSize[d] = 32;
      NumSMs[d] = 1;
      MaxNumThreadsPerSM[d] = 1024;
      ArchitectureGeneration[d] = 1;
    }
  }

  MGARDX_CONT int
  GetNumDevices() {
    return NumDevices;
  }

  MGARDX_CONT int
  GetMaxSharedMemorySize(int dev_id) {
    return MaxSharedMemorySize[dev_id];
  }

  MGARDX_CONT int
  GetWarpSize(int dev_id) {
    return WarpSize[dev_id];
  }

  MGARDX_CONT int
  GetNumSMs(int dev_id) {
    return NumSMs[dev_id];
  }

  MGARDX_CONT int
  GetArchitectureGeneration(int dev_id) {
    return ArchitectureGeneration[dev_id];
  }

  MGARDX_CONT int
  GetMaxNumThreadsPerSM(int dev_id) {
    return MaxNumThreadsPerSM[dev_id];
  }

  MGARDX_CONT
  ~DeviceSpecification() {
    delete [] MaxSharedMemorySize;
    delete [] WarpSize;
    delete [] NumSMs;
    delete [] ArchitectureGeneration;
  }

  int NumDevices;
  int* MaxSharedMemorySize;
  int* WarpSize;
  int* NumSMs;
  int* ArchitectureGeneration;
  int* MaxNumThreadsPerSM;
};

template <>
class DeviceQueues<Serial> {
  public:
  MGARDX_CONT
  DeviceQueues(){
    // do nothing
  }

  MGARDX_CONT int 
  GetQueue(int dev_id, SIZE queue_id){
    return 0;
  }

  MGARDX_CONT void 
  SyncQueue(int dev_id, SIZE queue_id){
    // do nothing
  }

  MGARDX_CONT void 
  SyncAllQueues(int dev_id){
    // do nothing
  }

  MGARDX_CONT
  ~DeviceQueues(){
    // do nothing
  }
};

template <>
class DeviceRuntime<Serial> {
  public:
  MGARDX_CONT
  DeviceRuntime(){}

  MGARDX_CONT static void 
  SelectDevice(SIZE dev_id){
    // do nothing
  }

  MGARDX_CONT static int 
  GetQueue(SIZE queue_id){
    return 0;
  }

  MGARDX_CONT static void 
  SyncQueue(SIZE queue_id){
    // do nothing
  }

  MGARDX_CONT static void 
  SyncAllQueues(){
    // do nothing
  }

  MGARDX_CONT static void
  SyncDevice(){
    // do nothing
  }

  MGARDX_CONT static int
  GetMaxSharedMemorySize() {
    return DeviceSpecs.GetMaxSharedMemorySize(curr_dev_id);
  }

  MGARDX_CONT static int
  GetWarpSize() {
    return DeviceSpecs.GetWarpSize(curr_dev_id);
  }

  MGARDX_CONT static int
  GetNumSMs() {
    return DeviceSpecs.GetNumSMs(curr_dev_id);
  }

  MGARDX_CONT static int
  GetArchitectureGeneration() {
    return DeviceSpecs.GetArchitectureGeneration(curr_dev_id);
  }

  MGARDX_CONT static int
  GetMaxNumThreadsPerSM() {
    return DeviceSpecs.GetMaxNumThreadsPerSM(curr_dev_id);
  }

  template <typename FunctorType>
  MGARDX_CONT static int
  GetOccupancyMaxActiveBlocksPerSM(FunctorType functor, int blockSize, size_t dynamicSMemSize) {
    return 1;
  }

  template <typename FunctorType>
  MGARDX_CONT static void
  SetMaxDynamicSharedMemorySize(FunctorType functor, int maxbytes) {
    // do nothing
  }


  MGARDX_CONT
  ~DeviceRuntime(){
  }

  static int curr_dev_id;
  static DeviceQueues<Serial> queues;
  static bool SyncAllKernelsAndCheckErrors;
  static DeviceSpecification<Serial> DeviceSpecs;
};

template <>
class MemoryManager<Serial> {
  public:
  MGARDX_CONT
  MemoryManager(){};

  template <typename T>
  MGARDX_CONT static
  void Malloc1D(T *& ptr, SIZE n, int queue_idx) {
    ptr = (T*)std::malloc(n * sizeof(T));
    if (ptr == NULL) {
      std::cout << log::log_err << "MemoryManager<Serial>::Malloc1D error.\n";
    }
  }

  template <typename T>
  MGARDX_CONT static
  void MallocND(T *& ptr, SIZE n1, SIZE n2, SIZE &ld, int queue_idx) {
    ptr = (T*)std::malloc(n1 * n2 * sizeof(T));
    ld = n1;
    if (ptr == NULL) {
      std::cout << log::log_err << "MemoryManager<Serial>::Malloc1D error.\n";
    }
  }

  template <typename T>
  MGARDX_CONT static
  void Free(T * ptr) {
    if (ptr == NULL) return;
    std::free(ptr);
  }

  template <typename T>
  MGARDX_CONT static
  void Copy1D(T * dst_ptr, const T * src_ptr, SIZE n, int queue_idx) {
    std::memcpy(dst_ptr, src_ptr, sizeof(T) * n);
  }

  template <typename T>
  MGARDX_CONT static
  void CopyND(T * dst_ptr, SIZE dst_ld, const T * src_ptr, SIZE src_ld, SIZE n1, SIZE n2, int queue_idx) {
    std::memcpy(dst_ptr, src_ptr, sizeof(T) * n1 * n2);
  }

  template <typename T>
  MGARDX_CONT static
  void MallocHost(T *& ptr, SIZE n, int queue_idx) {
    ptr = (T*)std::malloc(n * sizeof(T));
    if (ptr == NULL) {
      std::cout << log::log_err << "MemoryManager<Serial>::Malloc1D error.\n";
    }
  }

  template <typename T>
  MGARDX_CONT static
  void FreeHost(T * ptr) {
    if (ptr == NULL) return;
    std::free(ptr);
  }

  template <typename T>
  MGARDX_CONT static
  void Memset1D(T * ptr, SIZE n, int value) {
    memset(ptr, value, n * sizeof(T));
  }

  template <typename T>
  MGARDX_CONT static
  void MemsetND(T * ptr, SIZE ld, SIZE n1, SIZE n2, int value) {
    memset(ptr, value, n1 * n2 * sizeof(T));
  }

  template <typename T>
  MGARDX_CONT static
  bool IsDevicePointer(T * ptr) {
    return true;
  }
  static bool ReduceMemoryFootprint;
};


template <typename TaskType>
class DeviceAdapter<TaskType, Serial> {
public:
  MGARDX_CONT
  DeviceAdapter(){};

  MGARDX_CONT
  void Execute(TaskType& task) {
    // if constexpr evalute at compile time otherwise this does not compile
    if constexpr (std::is_base_of<Functor<Serial>, typename TaskType::Functor>::value) {
      SerialKernel(task);
    } else if constexpr (std::is_base_of<IterFunctor<Serial>, typename TaskType::Functor>::value) {
      SerialIterKernel(task);
    } else if constexpr (std::is_base_of<HuffmanCLCustomizedFunctor<Serial>, typename TaskType::Functor>::value) {
      SerialHuffmanCLCustomizedKernel(task);
    } else if constexpr (std::is_base_of<HuffmanCWCustomizedFunctor<Serial>, typename TaskType::Functor>::value) {
      SerialHuffmanCWCustomizedKernel(task);
    }
  }
};

template <typename KeyT, typename ValueT>
struct KeyValueComparator{
    bool operator()(std::pair<KeyT, ValueT> a, std::pair<KeyT, ValueT> b) const { return a.first < b.first; }
};

template <>
class DeviceCollective<Serial>{
public:
  MGARDX_CONT
  DeviceCollective(){};

  template <typename T> MGARDX_CONT static
  void Sum(SIZE n, SubArray<1, T, Serial>& v, SubArray<1, T, Serial>& result, int queue_idx) {
    *result((IDX)0) = std::accumulate(v(0), v(n), 0);
  }

  template <typename T> MGARDX_CONT static
  void AbsMax(SIZE n, SubArray<1, T, Serial>& v, SubArray<1, T, Serial>& result, int queue_idx) {
    T max_result = 0;
    for (SIZE i = 0; i < n; ++i) {
      max_result = std::max((T)fabs(*v(i)), max_result);
    }
    *result((IDX)0) = max_result;
  }

  template <typename T> MGARDX_CONT static
  void SquareSum(SIZE n, SubArray<1, T, Serial>& v, SubArray<1, T, Serial>& result, int queue_idx) {
    T sum_result = 0;
    for (SIZE i = 0; i < n; ++i) {
      T tmp = *v(i);
      sum_result += tmp * tmp; 
    }
    *result((IDX)0) = sum_result;
  }
 
 
 template <typename T> MGARDX_CONT static
  void ScanSumInclusive(SIZE n, SubArray<1, T, Serial>& v, SubArray<1, T, Serial>& result, int queue_idx) {
    // std::inclusive_scan(v(0), v(n), result(0));
    std::cout << log::log_err << "ScanSumInclusive<Serial> not implemented.\n";
  }

  template <typename T> MGARDX_CONT static
  void ScanSumExclusive(SIZE n, SubArray<1, T, Serial>& v, SubArray<1, T, Serial>& result, int queue_idx) {
    // std::exclusive_scan(v(0), v(n), result(0));
    std::cout << log::log_err << "ScanSumExclusive<Serial> not implemented.\n";
  }

  template <typename T> MGARDX_CONT static
  void ScanSumExtended(SIZE n, SubArray<1, T, Serial>& v, SubArray<1, T, Serial>& result, int queue_idx) {
    // std::inclusive_scan(v(0), v(n), result(1));
    // result(0) = 0;
    std::cout << log::log_err << "ScanSumExtended<Serial> not implemented.\n";
  }

  template <typename KeyT, typename ValueT> MGARDX_CONT static
  void SortByKey(SIZE n, SubArray<1, KeyT, Serial>& keys, SubArray<1, ValueT, Serial>& values, 
                 int queue_idx) {
    std::vector<std::pair<KeyT, ValueT>> data(n);
    for (SIZE i = 0; i < n; ++i) {
      data[i] = std::pair<KeyT, ValueT>(*keys(i), *values(i));
    }
    std::sort(data.begin(), data.end(), KeyValueComparator<KeyT, ValueT>{});
    for (SIZE i = 0; i < n; ++i) {
      *keys(i) = data[i].first;
      *values(i) = data[i].second;
    }
  }
};


}

#endif