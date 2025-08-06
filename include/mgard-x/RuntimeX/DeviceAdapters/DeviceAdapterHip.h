/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "DeviceAdapter.h"

#include <hip/hip_runtime_api.h>
#include <hipcub/hipcub.hpp>
#include <iostream>
// #include <mma.h>
#include <hip/hip_cooperative_groups.h>

// using namespace nvhip;
namespace cg = cooperative_groups;

#ifndef MGARD_X_DEVICE_ADAPTER_HIP_H
#define MGARD_X_DEVICE_ADAPTER_HIP_H

template <class T> struct SharedMemory {
  MGARDX_EXEC operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  MGARDX_EXEC operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

static __device__ __inline__ uint32_t __mywarpid() {
  uint32_t warpid = threadIdx.y;
  // asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
  return warpid;
}

static __device__ __inline__ uint32_t __mylaneid() {
  uint32_t laneid = threadIdx.x;
  // asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
  return laneid;
}

namespace mgard_x {

template <typename TaskType>
inline void ErrorAsyncCheck(hipError_t code, TaskType &task,
                            bool abort = true) {
  if (code != hipSuccess) {
    std::cout << log::log_err << hipGetErrorString(code) << " while executing "
              << task.GetFunctorName().c_str() << " with HIP (Async-check)\n";
    if (abort)
      exit(code);
  }
}

template <typename TaskType>
inline void ErrorSyncCheck(hipError_t code, TaskType &task, bool abort = true) {
  if (code != hipSuccess) {
    std::cout << log::log_err << hipGetErrorString(code) << " while executing "
              << task.GetFunctorName().c_str() << " with HIP (Sync-check)\n";
    if (abort)
      exit(code);
  }
}

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(hipError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != hipSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", hipGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

template <> struct SyncBlock<HIP> {
  MGARDX_EXEC static void Sync() { __syncthreads(); }
};

template <> struct SyncGrid<HIP> {
  MGARDX_EXEC static void Sync() { cg::this_grid().sync(); }
};

template <typename T, OPTION MemoryType, OPTION Scope>
struct Atomic<T, MemoryType, Scope, HIP> {
  MGARDX_EXEC static T Min(T *result, T value) {
    if constexpr (Scope == AtomicSystemScope) {
      return atomicMin_system(result, value);
    } else if constexpr (Scope == AtomicDeviceScope) {
      return atomicMin(result, value);
    } else {
      return atomicMin_block(result, value);
    }
  }
  MGARDX_EXEC static T Max(T *result, T value) {
    if constexpr (Scope == AtomicSystemScope) {
      return atomicMax_system(result, value);
    } else if constexpr (Scope == AtomicDeviceScope) {
      return atomicMax(result, value);
    } else {
      return atomicMax_block(result, value);
    }
  }
  MGARDX_EXEC static T Add(T *result, T value) {
    if constexpr (Scope == AtomicSystemScope) {
      return atomicAdd_system(result, value);
    } else if constexpr (Scope == AtomicDeviceScope) {
      return atomicAdd(result, value);
    } else {
      return atomicAdd_block(result, value);
    }
  }
};

template <> struct Math<HIP> {
  template <typename T> MGARDX_EXEC static T Min(T a, T b) { return min(a, b); }
  template <typename T> MGARDX_EXEC static T Max(T a, T b) { return max(a, b); }
  MGARDX_EXEC static int ffs(unsigned int a) { return __ffs(a); }
  MGARDX_EXEC static int ffsll(long long unsigned int a) { return __ffsll(a); }
  MGARDX_EXEC
  static uint64_t binary2negabinary(const int64_t x) {
    return (x + (uint64_t)0xaaaaaaaaaaaaaaaaull) ^
           (uint64_t)0xaaaaaaaaaaaaaaaaull;
  }

  MGARDX_EXEC
  static uint32_t binary2negabinary(const int32_t x) {
    return (x + (uint32_t)0xaaaaaaaau) ^ (uint32_t)0xaaaaaaaau;
  }

  MGARDX_EXEC
  static int64_t negabinary2binary(const uint64_t x) {
    return (x ^ 0xaaaaaaaaaaaaaaaaull) - 0xaaaaaaaaaaaaaaaaull;
  }

  MGARDX_EXEC
  static int32_t negabinary2binary(const uint32_t x) {
    return (x ^ 0xaaaaaaaau) - 0xaaaaaaaau;
  }
};

template <typename Task> MGARDX_KERL void kernel() {}

template <typename Task> MGARDX_KERL void Kernel(Task task) {
  Byte *shared_memory = SharedMemory<Byte>();
  task.GetFunctor().Init(gridDim.z, gridDim.y, gridDim.x, blockDim.z,
                         blockDim.y, blockDim.x, blockIdx.z, blockIdx.y,
                         blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x,
                         shared_memory);

  task.GetFunctor().Operation1();
  SyncBlock<HIP>::Sync();
  task.GetFunctor().Operation2();
  SyncBlock<HIP>::Sync();
  task.GetFunctor().Operation3();
  SyncBlock<HIP>::Sync();
  task.GetFunctor().Operation4();
  SyncBlock<HIP>::Sync();
  task.GetFunctor().Operation5();
  SyncBlock<HIP>::Sync();
  task.GetFunctor().Operation6();
  SyncBlock<HIP>::Sync();
  task.GetFunctor().Operation7();
  SyncBlock<HIP>::Sync();
  task.GetFunctor().Operation8();
  SyncBlock<HIP>::Sync();
  task.GetFunctor().Operation9();
  SyncBlock<HIP>::Sync();
  task.GetFunctor().Operation10();
}

template <typename Task> MGARDX_KERL void IterKernel(Task task) {
  Byte *shared_memory = SharedMemory<Byte>();

  task.GetFunctor().Init(gridDim.z, gridDim.y, gridDim.x, blockDim.z,
                         blockDim.y, blockDim.x, blockIdx.z, blockIdx.y,
                         blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x,
                         shared_memory);

  task.GetFunctor().Operation1();
  SyncBlock<HIP>::Sync();

  task.GetFunctor().Operation2();
  SyncBlock<HIP>::Sync();

  while (task.GetFunctor().LoopCondition1()) {
    task.GetFunctor().Operation3();
    SyncBlock<HIP>::Sync();
    task.GetFunctor().Operation4();
    SyncBlock<HIP>::Sync();
    task.GetFunctor().Operation5();
    SyncBlock<HIP>::Sync();
    task.GetFunctor().Operation6();
    SyncBlock<HIP>::Sync();
  }

  task.GetFunctor().Operation7();
  SyncBlock<HIP>::Sync();
  task.GetFunctor().Operation8();
  SyncBlock<HIP>::Sync();
  task.GetFunctor().Operation9();
  SyncBlock<HIP>::Sync();
  task.GetFunctor().Operation10();
  SyncBlock<HIP>::Sync();

  while (task.GetFunctor().LoopCondition2()) {
    task.GetFunctor().Operation11();
    SyncBlock<HIP>::Sync();
    task.GetFunctor().Operation12();
    SyncBlock<HIP>::Sync();
    task.GetFunctor().Operation13();
    SyncBlock<HIP>::Sync();
    task.GetFunctor().Operation14();
    SyncBlock<HIP>::Sync();
  }

  task.GetFunctor().Operation15();
  SyncBlock<HIP>::Sync();
  task.GetFunctor().Operation16();
  SyncBlock<HIP>::Sync();
  task.GetFunctor().Operation17();
  SyncBlock<HIP>::Sync();
}

template <typename Task> MGARDX_KERL void HuffmanCLCustomizedKernel(Task task) {
  Byte *shared_memory = SharedMemory<Byte>();

  task.GetFunctor().Init(gridDim.z, gridDim.y, gridDim.x, blockDim.z,
                         blockDim.y, blockDim.x, blockIdx.z, blockIdx.y,
                         blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x,
                         shared_memory);

  task.GetFunctor().Operation1();
  SyncGrid<HIP>::Sync();
  while (task.GetFunctor().LoopCondition1()) {
    task.GetFunctor().Operation2();
    SyncGrid<HIP>::Sync();
    task.GetFunctor().Operation3();
    SyncGrid<HIP>::Sync();
    task.GetFunctor().Operation4();
    SyncGrid<HIP>::Sync();
    task.GetFunctor().Operation5();
    SyncBlock<HIP>::Sync();
    if (task.GetFunctor().BranchCondition1()) {
      while (task.GetFunctor().LoopCondition2()) {
        task.GetFunctor().Operation6();
        SyncBlock<HIP>::Sync();
        task.GetFunctor().Operation7();
        SyncBlock<HIP>::Sync();
        task.GetFunctor().Operation8();
        SyncBlock<HIP>::Sync();
      }
      task.GetFunctor().Operation9();
      SyncGrid<HIP>::Sync();
      task.GetFunctor().Operation10();
      SyncGrid<HIP>::Sync();
    }
    task.GetFunctor().Operation11();
    SyncGrid<HIP>::Sync();
    task.GetFunctor().Operation12();
    SyncGrid<HIP>::Sync();
    task.GetFunctor().Operation13();
    SyncGrid<HIP>::Sync();
    task.GetFunctor().Operation14();
    SyncGrid<HIP>::Sync();
  }
}

#define SINGLE_KERNEL(OPERATION)                                               \
  template <typename Task>                                                     \
  MGARDX_KERL void Single_##OPERATION##_Kernel(Task task) {                    \
    Byte *shared_memory = SharedMemory<Byte>();                                \
    task.GetFunctor().Init(gridDim.z, gridDim.y, gridDim.x, blockDim.z,        \
                           blockDim.y, blockDim.x, blockIdx.z, blockIdx.y,     \
                           blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x,  \
                           shared_memory);                                     \
    task.GetFunctor().OPERATION();                                             \
  }

SINGLE_KERNEL(Operation1);
SINGLE_KERNEL(Operation2);
SINGLE_KERNEL(Operation3);
SINGLE_KERNEL(Operation4);
SINGLE_KERNEL(Operation5);
SINGLE_KERNEL(Operation6);
SINGLE_KERNEL(Operation7);
SINGLE_KERNEL(Operation8);
SINGLE_KERNEL(Operation9);
SINGLE_KERNEL(Operation10);
SINGLE_KERNEL(Operation11);
SINGLE_KERNEL(Operation12);
SINGLE_KERNEL(Operation13);
SINGLE_KERNEL(Operation14);

#undef SINGLE_KERNEL

template <typename Task> MGARDX_KERL void ParallelMergeKernel(Task task) {
  Byte *shared_memory = SharedMemory<Byte>();

  task.GetFunctor().Init(gridDim.z, gridDim.y, gridDim.x, blockDim.z,
                         blockDim.y, blockDim.x, blockIdx.z, blockIdx.y,
                         blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x,
                         shared_memory);

  task.GetFunctor().Operation5();
  SyncBlock<HIP>::Sync();
  while (task.GetFunctor().LoopCondition2()) {
    task.GetFunctor().Operation6();
    SyncBlock<HIP>::Sync();
    task.GetFunctor().Operation7();
    SyncBlock<HIP>::Sync();
    task.GetFunctor().Operation8();
    SyncBlock<HIP>::Sync();
  }
  task.GetFunctor().Operation9();
}

template <typename Task> MGARDX_KERL void HuffmanCWCustomizedKernel(Task task) {
  Byte *shared_memory = SharedMemory<Byte>();

  task.GetFunctor().Init(gridDim.z, gridDim.y, gridDim.x, blockDim.z,
                         blockDim.y, blockDim.x, blockIdx.z, blockIdx.y,
                         blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x,
                         shared_memory);

  task.GetFunctor().Operation1();
  SyncGrid<HIP>::Sync();
  task.GetFunctor().Operation2();
  SyncGrid<HIP>::Sync();
  task.GetFunctor().Operation3();
  SyncGrid<HIP>::Sync();

  while (task.GetFunctor().LoopCondition1()) {
    task.GetFunctor().Operation4();
    SyncGrid<HIP>::Sync();
    task.GetFunctor().Operation5();
    SyncGrid<HIP>::Sync();
    task.GetFunctor().Operation6();
    SyncGrid<HIP>::Sync();
    task.GetFunctor().Operation7();
    SyncGrid<HIP>::Sync();
    task.GetFunctor().Operation8();
    SyncGrid<HIP>::Sync();
  }
  task.GetFunctor().Operation9();
  SyncGrid<HIP>::Sync();
  task.GetFunctor().Operation10();
  SyncGrid<HIP>::Sync();
}

template <> class DeviceSpecification<HIP> {
public:
  MGARDX_CONT
  DeviceSpecification() {
    hipGetDeviceCount(&NumDevices);
    MaxSharedMemorySize = new int[NumDevices];
    WarpSize = new int[NumDevices];
    NumSMs = new int[NumDevices];
    ArchitectureGeneration = new int[NumDevices];
    MaxNumThreadsPerSM = new int[NumDevices];
    MaxNumThreadsPerTB = new int[NumDevices];
    AvailableMemory = new size_t[NumDevices];
    SupportCooperativeGroups = new bool[NumDevices];
    DeviceNames = new std::string[NumDevices];

    for (int d = 0; d < NumDevices; d++) {
      gpuErrchk(hipSetDevice(d));
      int maxbytes;
      hipDeviceGetAttribute(&maxbytes,
                            hipDeviceAttributeMaxSharedMemoryPerBlock, d);
      MaxSharedMemorySize[d] = maxbytes;
      hipDeviceGetAttribute(&WarpSize[d], hipDeviceAttributeWarpSize, d);
      hipDeviceGetAttribute(&NumSMs[d], hipDeviceAttributeMultiprocessorCount,
                            d);
      hipDeviceGetAttribute(&MaxNumThreadsPerSM[d],
                            hipDeviceAttributeMaxThreadsPerMultiProcessor, d);
      hipDeviceGetAttribute(&MaxNumThreadsPerTB[d],
                            hipDeviceAttributeMaxThreadsPerBlock, d);
      SupportCooperativeGroups[d] = false;
      hipDeviceProp_t prop;
      hipGetDeviceProperties(&prop, d);
      ArchitectureGeneration[d] = 1; // default optimized for Volta
      if (prop.major == 7 && prop.minor == 0) {
        ArchitectureGeneration[d] = 1;
      } else if (prop.major == 7 && (prop.minor == 2 || prop.minor == 5)) {
        ArchitectureGeneration[d] = 2;
      }
      MaxNumThreadsPerTB[d] = 32; // Due to a bug in Cooperative Groups in HIP
      WarpSize[d] = 32;
      // DeviceNames[d] = std::string(prop.name); // Not working in HIP
      DeviceNames[d] = std::string("AMD GPU");
    }
  }

  MGARDX_CONT int GetNumDevices() { return NumDevices; }

  MGARDX_CONT int GetMaxSharedMemorySize(int dev_id) {
    return MaxSharedMemorySize[dev_id];
  }

  MGARDX_CONT int GetWarpSize(int dev_id) { return WarpSize[dev_id]; }

  MGARDX_CONT int GetNumSMs(int dev_id) { return NumSMs[dev_id]; }

  MGARDX_CONT int GetArchitectureGeneration(int dev_id) {
    return ArchitectureGeneration[dev_id];
  }

  MGARDX_CONT int GetMaxNumThreadsPerSM(int dev_id) {
    return MaxNumThreadsPerSM[dev_id];
  }

  MGARDX_CONT int GetMaxNumThreadsPerTB(int dev_id) {
    return MaxNumThreadsPerTB[dev_id];
  }

  MGARDX_CONT size_t GetAvailableMemory(int dev_id) {
    gpuErrchk(hipSetDevice(dev_id));
    size_t free, total;
    hipMemGetInfo(&free, &total);
    AvailableMemory[dev_id] = free;
    return AvailableMemory[dev_id];
  }

  MGARDX_CONT bool SupportCG(int dev_id) {
    return SupportCooperativeGroups[dev_id];
  }

  MGARDX_CONT std::string GetDeviceName(int dev_id) {
    return DeviceNames[dev_id];
  }

  MGARDX_CONT
  ~DeviceSpecification() {
    delete[] MaxSharedMemorySize;
    delete[] WarpSize;
    delete[] NumSMs;
    delete[] ArchitectureGeneration;
    delete[] MaxNumThreadsPerSM;
    delete[] MaxNumThreadsPerTB;
    delete[] AvailableMemory;
    delete[] SupportCooperativeGroups;
    delete[] DeviceNames;
  }

  int NumDevices;
  int *MaxSharedMemorySize;
  int *WarpSize;
  int *NumSMs;
  int *ArchitectureGeneration;
  int *MaxNumThreadsPerSM;
  int *MaxNumThreadsPerTB;
  size_t *AvailableMemory;
  bool *SupportCooperativeGroups;
  std::string *DeviceNames;
};

template <> class DeviceQueues<HIP> {
public:
  MGARDX_CONT
  DeviceQueues() {
    hipGetDeviceCount(&NumDevices);
    streams = new hipStream_t *[NumDevices];
    for (int d = 0; d < NumDevices; d++) {
      gpuErrchk(hipSetDevice(d));
      streams[d] = new hipStream_t[MGARDX_NUM_QUEUES];
      for (SIZE i = 0; i < MGARDX_NUM_QUEUES; i++) {
        gpuErrchk(hipStreamCreate(&streams[d][i]));
      }
    }
  }

  MGARDX_CONT hipStream_t GetQueue(int dev_id, SIZE queue_id) {
    return streams[dev_id][queue_id];
  }

  MGARDX_CONT void SyncQueue(int dev_id, SIZE queue_id) {
    hipStreamSynchronize(streams[dev_id][queue_id]);
  }

  MGARDX_CONT void SyncAllQueues(int dev_id) {
    for (SIZE i = 0; i < MGARDX_NUM_QUEUES; i++) {
      gpuErrchk(hipStreamSynchronize(streams[dev_id][i]));
    }
  }

  MGARDX_CONT
  ~DeviceQueues() {
    for (int d = 0; d < NumDevices; d++) {
      gpuErrchk(hipSetDevice(d));
      for (int i = 0; i < MGARDX_NUM_QUEUES; i++) {
        gpuErrchk(hipStreamDestroy(streams[d][i]));
      }
      delete[] streams[d];
    }
    delete[] streams;
    streams = NULL;
  }

  int NumDevices;
  hipStream_t **streams = NULL;
};

template <> class DeviceRuntime<HIP> {
public:
  MGARDX_CONT
  DeviceRuntime() {}

  MGARDX_CONT static int GetDeviceCount() { return DeviceSpecs.NumDevices; }

  MGARDX_CONT static void SelectDevice(SIZE dev_id) {
    gpuErrchk(hipSetDevice(dev_id));
    curr_dev_id = dev_id;
  }

  MGARDX_CONT static hipStream_t GetQueue(SIZE queue_id) {
    gpuErrchk(hipSetDevice(curr_dev_id));
    return queues.GetQueue(curr_dev_id, queue_id);
  }

  MGARDX_CONT static void SyncQueue(SIZE queue_id) {
    gpuErrchk(hipSetDevice(curr_dev_id));
    queues.SyncQueue(curr_dev_id, queue_id);
  }

  MGARDX_CONT static void SyncAllQueues() {
    gpuErrchk(hipSetDevice(curr_dev_id));
    queues.SyncAllQueues(curr_dev_id);
  }

  MGARDX_CONT static void SyncDevice() {
    gpuErrchk(hipSetDevice(curr_dev_id));
    gpuErrchk(hipDeviceSynchronize());
  }

  MGARDX_CONT static std::string GetDeviceName() {
    return DeviceSpecs.GetDeviceName(curr_dev_id);
  }

  MGARDX_CONT static int GetMaxSharedMemorySize() {
    return DeviceSpecs.GetMaxSharedMemorySize(curr_dev_id);
  }

  MGARDX_CONT static int GetWarpSize() {
    return DeviceSpecs.GetWarpSize(curr_dev_id);
  }

  MGARDX_CONT static int GetNumSMs() {
    return DeviceSpecs.GetNumSMs(curr_dev_id);
  }

  MGARDX_CONT static int GetArchitectureGeneration() {
    return DeviceSpecs.GetArchitectureGeneration(curr_dev_id);
  }

  MGARDX_CONT static int GetMaxNumThreadsPerSM() {
    return DeviceSpecs.GetMaxNumThreadsPerSM(curr_dev_id);
  }

  MGARDX_CONT static int GetMaxNumThreadsPerTB() {
    return DeviceSpecs.GetMaxNumThreadsPerTB(curr_dev_id);
  }

  MGARDX_CONT static size_t GetAvailableMemory() {
    return DeviceSpecs.GetAvailableMemory(curr_dev_id);
  }

  MGARDX_CONT static bool SupportCG() {
    return DeviceSpecs.SupportCG(curr_dev_id);
  }

  template <typename FunctorType>
  MGARDX_CONT static int
  GetOccupancyMaxActiveBlocksPerSM(FunctorType functor, int blockSize,
                                   size_t dynamicSMemSize) {
    int numBlocks = 0;

    if constexpr (std::is_base_of<Functor<HIP>, FunctorType>::value) {
      gpuErrchk(hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &numBlocks, Kernel<Task<FunctorType>>, blockSize, dynamicSMemSize));
    } else if constexpr (std::is_base_of<IterFunctor<HIP>,
                                         FunctorType>::value) {
      gpuErrchk(hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &numBlocks, IterKernel<Task<FunctorType>>, blockSize,
          dynamicSMemSize));
    } else if constexpr (std::is_base_of<HuffmanCLCustomizedFunctor<HIP>,
                                         FunctorType>::value) {
      gpuErrchk(hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &numBlocks, HuffmanCLCustomizedKernel<Task<FunctorType>>, blockSize,
          dynamicSMemSize));
    } else if constexpr (std::is_base_of<HuffmanCWCustomizedFunctor<HIP>,
                                         FunctorType>::value) {
      gpuErrchk(hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &numBlocks, HuffmanCWCustomizedKernel<Task<FunctorType>>, blockSize,
          dynamicSMemSize));
    } else {
      std::cout << log::log_err << "GetOccupancyMaxActiveBlocksPerSM Error!\n";
    }
    return numBlocks;
  }

  template <typename FunctorType>
  MGARDX_CONT static void SetMaxDynamicSharedMemorySize(FunctorType functor,
                                                        int maxbytes) {

    if constexpr (std::is_base_of<Functor<HIP>, FunctorType>::value) {
      gpuErrchk(hipFuncSetAttribute((const void *)Kernel<Task<FunctorType>>,
                                    hipFuncAttributeMaxDynamicSharedMemorySize,
                                    maxbytes));
    } else if constexpr (std::is_base_of<IterFunctor<HIP>,
                                         FunctorType>::value) {
      gpuErrchk(hipFuncSetAttribute((const void *)IterKernel<Task<FunctorType>>,
                                    hipFuncAttributeMaxDynamicSharedMemorySize,
                                    maxbytes));
    } else if constexpr (std::is_base_of<HuffmanCLCustomizedFunctor<HIP>,
                                         FunctorType>::value) {
      gpuErrchk(hipFuncSetAttribute(
          (const void *)HuffmanCLCustomizedKernel<Task<FunctorType>>,
          hipFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
    } else if constexpr (std::is_base_of<HuffmanCWCustomizedFunctor<HIP>,
                                         FunctorType>::value) {
      gpuErrchk(hipFuncSetAttribute(
          (const void *)HuffmanCWCustomizedKernel<Task<FunctorType>>,
          hipFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
    } else {
      std::cout << log::log_err << "SetPreferredSharedMemoryCarveout Error!\n";
    }
  }

  MGARDX_CONT
  ~DeviceRuntime() {}

  static int curr_dev_id;
  static DeviceQueues<HIP> queues;
  static bool SyncAllKernelsAndCheckErrors;
  static DeviceSpecification<HIP> DeviceSpecs;
  static bool TimingAllKernels;
  static bool PrintKernelConfig;
};

template <> class MemoryManager<HIP> {
public:
  MGARDX_CONT
  MemoryManager(){};

  template <typename T>
  MGARDX_CONT static void Malloc1D(T *&ptr, SIZE n, int queue_idx) {
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    gpuErrchk(hipMalloc(&ptr, n * sizeof(converted_T)));
    if (DeviceRuntime<HIP>::SyncAllKernelsAndCheckErrors) {
      gpuErrchk(hipDeviceSynchronize());
    }
  }

  template <typename T>
  MGARDX_CONT static void MallocND(T *&ptr, SIZE n1, SIZE n2, SIZE &ld,
                                   int queue_idx) {
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    if (ReduceMemoryFootprint) {
      gpuErrchk(hipMalloc((void **)&ptr, n1 * n2 * sizeof(converted_T)));
      ld = n1;
    } else {
      size_t pitch = 0;
      gpuErrchk(hipMallocPitch((void **)&ptr, &pitch, n1 * sizeof(converted_T),
                               (size_t)n2));
      ld = pitch / sizeof(converted_T);
    }
    if (DeviceRuntime<HIP>::SyncAllKernelsAndCheckErrors) {
      gpuErrchk(hipDeviceSynchronize());
    }
  }

  template <typename T>
  MGARDX_CONT static void MallocManaged1D(T *&ptr, SIZE n, int queue_idx) {
    gpuErrchk(hipMallocManaged(&ptr, n * sizeof(T)));
    if (DeviceRuntime<HIP>::SyncAllKernelsAndCheckErrors) {
      gpuErrchk(hipDeviceSynchronize());
    }
  }

  template <typename T> MGARDX_CONT static void Free(T *ptr) {
    // printf("MemoryManager.Free(%llu)\n", ptr);
    if (ptr == NULL)
      return;
    gpuErrchk(hipFree(ptr));
    if (DeviceRuntime<HIP>::SyncAllKernelsAndCheckErrors) {
      gpuErrchk(hipDeviceSynchronize());
    }
  }

  template <typename T>
  MGARDX_CONT static void Copy1D(T *dst_ptr, const T *src_ptr, SIZE n,
                                 int queue_idx) {
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    hipStream_t stream = DeviceRuntime<HIP>::GetQueue(queue_idx);
    gpuErrchk(hipMemcpyAsync(dst_ptr, src_ptr, n * sizeof(converted_T),
                             hipMemcpyDefault, stream));
    if (DeviceRuntime<HIP>::SyncAllKernelsAndCheckErrors) {
      gpuErrchk(hipDeviceSynchronize());
    }
  }

  template <typename T>
  MGARDX_CONT static void CopyND(T *dst_ptr, SIZE dst_ld, const T *src_ptr,
                                 SIZE src_ld, SIZE n1, SIZE n2, int queue_idx) {
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    hipStream_t stream = DeviceRuntime<HIP>::GetQueue(queue_idx);
    // printf("CopyND: dst: %d, src: %d\n", IsDevicePointer(dst_ptr),
    // IsDevicePointer(src_ptr));
    if (!IsDevicePointer(dst_ptr) && !IsDevicePointer(src_ptr)) {
      for (SIZE i = 0; i < n2; i++) {
        memcpy(dst_ptr + i * dst_ld, src_ptr + i * src_ld,
               n1 * sizeof(converted_T));
      }
    } else {
      gpuErrchk(hipMemcpy2DAsync(dst_ptr, dst_ld * sizeof(converted_T), src_ptr,
                                 src_ld * sizeof(converted_T),
                                 n1 * sizeof(converted_T), n2, hipMemcpyDefault,
                                 stream));
      if (DeviceRuntime<HIP>::SyncAllKernelsAndCheckErrors) {
        gpuErrchk(hipDeviceSynchronize());
      }
    }
  }

  template <typename T>
  MGARDX_CONT static void MallocHost(T *&ptr, SIZE n, int queue_idx) {
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    gpuErrchk(hipMallocHost((void **)&ptr, n * sizeof(converted_T)));
    if (DeviceRuntime<HIP>::SyncAllKernelsAndCheckErrors) {
      gpuErrchk(hipDeviceSynchronize());
    }
  }

  template <typename T> MGARDX_CONT static void FreeHost(T *ptr) {
    if (ptr == NULL)
      return;
    gpuErrchk(hipFreeHost(ptr));
    if (DeviceRuntime<HIP>::SyncAllKernelsAndCheckErrors) {
      gpuErrchk(hipDeviceSynchronize());
    }
  }

  template <typename T>
  MGARDX_CONT static void Memset1D(T *ptr, SIZE n, int value, int queue_idx) {
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    hipStream_t stream = DeviceRuntime<HIP>::GetQueue(queue_idx);
    gpuErrchk(hipMemsetAsync(ptr, value, n * sizeof(converted_T), stream));
  }

  template <typename T>
  MGARDX_CONT static void MemsetND(T *ptr, SIZE ld, SIZE n1, SIZE n2, int value,
                                   int queue_idx) {
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    hipStream_t stream = DeviceRuntime<HIP>::GetQueue(queue_idx);
    gpuErrchk(hipMemset2DAsync(ptr, ld * sizeof(converted_T), value,
                               n1 * sizeof(converted_T), n2, stream));
  }

  template <typename T> MGARDX_CONT static bool IsDevicePointer(T *ptr) {
    hipPointerAttribute_t attr;
    hipPointerGetAttributes(&attr, ptr);
    return attr.memoryType == hipMemoryTypeDevice;
  }

  static bool ReduceMemoryFootprint;
};

#define ALIGN_LEFT 0  // for encoding
#define ALIGN_RIGHT 1 // for decoding

#define Sign_Encoding_Atomic 0
#define Sign_Encoding_Reduce 1
#define Sign_Encoding_Ballot 2

#define Sign_Decoding_Parallel 0

#define BINARY 0
#define NEGABINARY 1

#define Bit_Transpose_Serial_All 0
#define Bit_Transpose_Parallel_B_Serial_b 1
#define Bit_Transpose_Parallel_B_Atomic_b 2
#define Bit_Transpose_Parallel_B_Reduce_b 3
#define Bit_Transpose_Parallel_B_Ballot_b 4
#define Bit_Transpose_TCU 5

#define Warp_Bit_Transpose_Serial_All 0
#define Warp_Bit_Transpose_Parallel_B_Serial_b 1
#define Warp_Bit_Transpose_Serial_B_Atomic_b 2
#define Warp_Bit_Transpose_Serial_B_Reduce_b 3
#define Warp_Bit_Transpose_Serial_B_Ballot_b 4
#define Warp_Bit_Transpose_TCU 5

#define Error_Collecting_Disable -1
#define Error_Collecting_Serial_All 0
#define Error_Collecting_Parallel_Bitplanes_Serial_Error 1
#define Error_Collecting_Parallel_Bitplanes_Atomic_Error 2
#define Error_Collecting_Parallel_Bitplanes_Reduce_Error 3

#define Warp_Error_Collecting_Disable -1
#define Warp_Error_Collecting_Serial_All 0
#define Warp_Error_Collecting_Parallel_Bitplanes_Serial_Error 1
#define Warp_Error_Collecting_Serial_Bitplanes_Atomic_Error 2
#define Warp_Error_Collecting_Serial_Bitplanes_Reduce_Error 3

typedef unsigned long long int uint64_cu;

template <typename T, SIZE nblockx, SIZE nblocky, SIZE nblockz>
struct BlockReduce<T, nblockx, nblocky, nblockz, HIP> {
  typedef hipcub::BlockReduce<T, nblockx, hipcub::BLOCK_REDUCE_WARP_REDUCTIONS,
                              nblocky, nblockz>
      BlockReduceType;
  using TempStorageType = typename BlockReduceType::TempStorage;

  MGARDX_EXEC
  static void Sum(T intput, T &output) {
    __shared__ TempStorageType temp_storage;
    BlockReduceType blockReduce(temp_storage);
    return blockReduce.Sum(intput);
  }

  MGARDX_EXEC
  static void Max(T intput, T &output) {
    __shared__ TempStorageType temp_storage;
    BlockReduceType blockReduce(temp_storage);
    return blockReduce.Reduce(intput, hipcub::Max());
  }
};

template <typename T_org, typename T_trans, SIZE nblockx, SIZE nblocky,
          SIZE nblockz, OPTION ALIGN, OPTION METHOD>
struct BlockBitTranspose<T_org, T_trans, nblockx, nblocky, nblockz, ALIGN,
                         METHOD, HIP> {

  typedef hipcub::WarpReduce<T_trans> WarpReduceType;
  using WarpReduceStorageType = typename WarpReduceType::TempStorage;

  typedef hipcub::BlockReduce<
      T_trans, nblockx, hipcub::BLOCK_REDUCE_WARP_REDUCTIONS, nblocky, nblockz>
      BlockReduceType;
  using BlockReduceStorageType = typename BlockReduceType::TempStorage;

  MGARDX_EXEC
  static void Serial_All(T_org *v, T_trans *tv, SIZE b, SIZE B, SIZE IdX,
                         SIZE IdY) {
    if (IdX == 0 && IdY == 0) {
      // printf("add-in: %llu %u\n", v, v[0]);
      // for (int i = 0; i < b; i++) {
      //   printf("v: %u\n", v[i]);
      // }
      for (SIZE B_idx = 0; B_idx < B; B_idx++) {
        T_trans buffer = 0;
        for (SIZE b_idx = 0; b_idx < b; b_idx++) {
          T_trans bit = (v[b_idx] >> (sizeof(T_org) * 8 - 1 - B_idx)) & 1u;
          if (ALIGN == ALIGN_LEFT) {
            buffer += bit << (sizeof(T_trans) * 8 - 1 - b_idx);
            // if (B_idx == 0) {
            // printf("%u %u %u\n", B_idx, b_idx, bit);
            // print_bits(buffer, sizeof(T_trans)*8, false);
            // printf("\n");
            // }
          } else if (ALIGN == ALIGN_RIGHT) {
            buffer += bit << (b - 1 - b_idx);
            // if (b_idx == 0) printf("%u %u %u\n", B_idx, b_idx, bit);
          } else {
          }
          // if (j == 0 ) {printf("i %u j %u shift %u bit %u\n", i,j,b-1-j,
          // bit); }
        }

        // printf("buffer: %u\n", buffer);

        tv[B_idx] = buffer;
      }
    }
  }

  MGARDX_EXEC
  static void Parallel_B_Serial_b(T_org *v, T_trans *tv, SIZE b, SIZE B,
                                  SIZE IdX, SIZE IdY) {
    if (IdY == 0) {
      for (SIZE B_idx = IdX; B_idx < B; B_idx += 32) {
        T_trans buffer = 0;
        for (SIZE b_idx = 0; b_idx < b; b_idx++) {
          T_trans bit = (v[b_idx] >> (sizeof(T_org) * 8 - 1 - B_idx)) & 1u;
          if (ALIGN == ALIGN_LEFT) {
            buffer += bit << sizeof(T_trans) * 8 - 1 - b_idx;
          } else if (ALIGN == ALIGN_RIGHT) {
            buffer += bit << (b - 1 - b_idx);
            // if (b_idx == 0) printf("%u %u %u\n", B_idx, b_idx, bit);
          } else {
          }
        }
        tv[B_idx] = buffer;
      }
    }
  }

  MGARDX_EXEC
  static void Parallel_B_Atomic_b(T_org *v, T_trans *tv, SIZE b, SIZE B,
                                  SIZE IdX, SIZE IdY) {
    if (IdX < b && IdY < B) {
      SIZE i = IdX;
      for (SIZE B_idx = IdY; B_idx < B; B_idx += 32) {
        for (SIZE b_idx = IdX; b_idx < b; b_idx += 32) {
          T_trans bit = (v[b_idx] >> (sizeof(T_org) * 8 - 1 - B_idx)) & 1u;
          T_trans shifted_bit;
          if (ALIGN == ALIGN_LEFT) {
            shifted_bit = bit << sizeof(T_trans) * 8 - 1 - b_idx;
          } else if (ALIGN == ALIGN_RIGHT) {
            shifted_bit = bit << b - 1 - b_idx;
          } else {
          }
          T_trans *sum = &(tv[B_idx]);
          // atomicAdd_block(sum, shifted_bit);
        }
      }
    }
  }

  MGARDX_EXEC
  static void Parallel_B_Reduction_b(T_org *v, T_trans *tv, SIZE b, SIZE B,
                                     SIZE IdX, SIZE IdY) {

    // __syncthreads(); long long start = clock64();

    __shared__ WarpReduceStorageType warp_storage[32];

    SIZE warp_idx = IdY;
    SIZE lane_idx = IdX;
    SIZE B_idx, b_idx;
    T_trans bit = 0;
    T_trans shifted_bit = 0;
    T_trans sum = 0;

    // __syncthreads(); start = clock64() - start;
    // if (IdY == 0 && IdX == 0) printf("time0: %llu\n", start);
    // __syncthreads(); start = clock64();

    for (SIZE B_idx = IdY; B_idx < B; B_idx += 32) {
      sum = 0;
      for (SIZE b_idx = IdX; b_idx < ((b - 1) / 32 + 1) * 32; b_idx += 32) {
        shifted_bit = 0;
        if (b_idx < b && B_idx < B) {
          bit = (v[b_idx] >> (sizeof(T_org) * 8 - 1 - B_idx)) & 1u;
          if (ALIGN == ALIGN_LEFT) {
            shifted_bit = bit << sizeof(T_trans) * 8 - 1 - b_idx;
          } else if (ALIGN == ALIGN_RIGHT) {
            shifted_bit = bit << b - 1 - b_idx;
          } else {
          }
        }

        // __syncthreads(); start = clock64() - start;
        // if (IdY == 0 && IdX == 0) printf("time1: %llu\n",
        // start);
        // __syncthreads(); start = clock64();

        sum += WarpReduceType(warp_storage[warp_idx]).Sum(shifted_bit);
        // if (B_idx == 32) printf("shifted_bit[%u] %u sum %u\n", b_idx,
        // shifted_bit, sum);
      }
      if (lane_idx == 0) {
        tv[B_idx] = sum;
      }
    }

    // __syncthreads(); start = clock64() - start;
    // if (IdY == 0 && IdX == 0) printf("time2: %llu\n", start);
    // __syncthreads(); start = clock64();
  }

  MGARDX_EXEC
  static void Parallel_B_Ballot_b(T_org *v, T_trans *tv, SIZE b, SIZE B,
                                  SIZE IdX, SIZE IdY) {

    SIZE warp_idx = IdY;
    SIZE lane_idx = IdX;
    SIZE B_idx, b_idx;
    int bit = 0;
    T_trans sum = 0;

    // __syncthreads(); start = clock64() - start;
    // if (IdY == 0 && IdX == 0) printf("time0: %llu\n", start);
    // __syncthreads(); start = clock64();

    for (SIZE B_idx = IdY; B_idx < B; B_idx += 32) {
      sum = 0;
      SIZE shift = 0;
      for (SIZE b_idx = IdX; b_idx < ((b - 1) / 32 + 1) * 32; b_idx += 32) {
        bit = 0;
        if (b_idx < b && B_idx < B) {
          if (ALIGN == ALIGN_LEFT) {
            bit = (v[sizeof(T_trans) * 8 - 1 - b_idx] >>
                   (sizeof(T_org) * 8 - 1 - B_idx)) &
                  1u;
          } else if (ALIGN == ALIGN_RIGHT) {
            bit = (v[b - 1 - b_idx] >> (sizeof(T_org) * 8 - 1 - B_idx)) & 1u;
          } else {
          }
        }

        // __syncthreads(); start = clock64() - start;
        // if (IdY == 0 && IdX == 0) printf("time1: %llu\n",
        // start);
        // __syncthreads(); start = clock64();
        sum += ((T_trans)__ballot(bit)) << shift;
        // sum += WarpReduceType(warp_storage[warp_idx]).Sum(shifted_bit);
        // if (B_idx == 32) printf("shifted_bit[%u] %u sum %u\n", b_idx,
        // shifted_bit, sum);
        shift += 32;
      }
      if (lane_idx == 0) {
        tv[B_idx] = sum;
      }
    }

    // __syncthreads(); start = clock64() - start;
    // if (IdY == 0 && IdX == 0) printf("time2: %llu\n", start);
    // __syncthreads(); start = clock64();

    // __syncthreads(); long long start = clock64();

    // SIZE i = IdX;
    // SIZE B_idx = IdY;
    // SIZE b_idx = IdX;

    // __syncthreads(); start = clock64() - start;
    // if (IdY == 0 && IdX == 0) printf("time0: %llu\n", start);
    // __syncthreads(); start = clock64();

    // int bit = (v[b-1-b_idx] >> (sizeof(T_org)*8 - 1 - B_idx)) & 1u;

    // __syncthreads();
    // start = clock64() - start;
    // if (IdY == 0 && IdX == 0) printf("time1: %llu\n", start);
    // __syncthreads(); start = clock64();

    // printf("b_idx[%u]: bit %d\n", b_idx, bit);
    // unsigned int sum = __ballot_sync (0xffffffff, bit);
    // printf("b_idx[%u]: sum %u\n", b_idx, sum);
    // if (b_idx) tv[B_idx] = sum;

    // __syncthreads(); start = clock64() - start;
    // if (IdY == 0 && IdX == 0) printf("time2: %llu\n", start);
    // __syncthreads(); start = clock64();
  }

  MGARDX_EXEC
  static void Transpose(T_org *v, T_trans *tv, SIZE b, SIZE B, SIZE IdX,
                        SIZE IdY) {
    if (METHOD == Bit_Transpose_Serial_All)
      Serial_All(v, tv, b, B, IdX, IdY);
    else if (METHOD == Bit_Transpose_Parallel_B_Serial_b)
      Parallel_B_Serial_b(v, tv, b, B, IdX, IdY);
    else if (METHOD == Bit_Transpose_Parallel_B_Atomic_b)
      Parallel_B_Atomic_b(v, tv, b, B, IdX, IdY);
    else if (METHOD == Bit_Transpose_Parallel_B_Reduce_b)
      Parallel_B_Reduction_b(v, tv, b, B, IdX, IdY);
    else if (METHOD == Bit_Transpose_Parallel_B_Ballot_b)
      Parallel_B_Ballot_b(v, tv, b, B, IdX, IdY);
    // else { printf("Bit Transpose Wrong Algorithm Type!\n");  }
    // else if (METHOD == 5) TCU(v, tv, b, B, IdX, IdY);
  }
};

template <typename T, typename T_fp, typename T_sfp, typename T_error,
          SIZE nblockx, SIZE nblocky, SIZE nblockz, OPTION METHOD,
          OPTION BinaryType>
struct BlockErrorCollect<T, T_fp, T_sfp, T_error, nblockx, nblocky, nblockz,
                         METHOD, BinaryType, HIP> {

  typedef hipcub::WarpReduce<T_error> WarpReduceType;
  using WarpReduceStorageType = typename WarpReduceType::TempStorage;

  typedef hipcub::BlockReduce<
      T_error, nblockx, hipcub::BLOCK_REDUCE_WARP_REDUCTIONS, nblocky, nblockz>
      BlockReduceType;
  using BlockReduceStorageType = typename BlockReduceType::TempStorage;

  MGARDX_EXEC
  static void Serial_All(T *v, T_error *temp, T_error *errors, SIZE num_elems,
                         SIZE num_bitplanes, SIZE IdX, SIZE IdY) {
    if (IdX == 0 && IdY == 0) {
      for (SIZE elem_idx = 0; elem_idx < num_elems; elem_idx++) {
        T data = v[elem_idx];
        T_fp fp_data = (T_fp)fabs(data);
        T_sfp fps_data = (T_sfp)data;
        T_fp ngb_data = Math<HIP>::binary2negabinary(fps_data);
        T_error mantissa;
        if (BinaryType == BINARY) {
          mantissa = fabs(data) - fp_data;
        } else if (BinaryType == NEGABINARY) {
          mantissa = data - fps_data;
        }
        for (SIZE bitplane_idx = 0; bitplane_idx < num_bitplanes;
             bitplane_idx++) {
          uint64_t mask = (1 << bitplane_idx) - 1;
          T_error diff = 0;
          if (BinaryType == BINARY) {
            diff = (T_error)(fp_data & mask) + mantissa;
          } else if (BinaryType == NEGABINARY) {
            diff = (T_error)Math<HIP>::negabinary2binary(ngb_data & mask) +
                   mantissa;
          }
          errors[num_bitplanes - bitplane_idx] += diff * diff;
          // if (blockIdx.x == 0 && num_bitplanes-bitplane_idx == 2) {
          //   printf("elem error[%u]: %f\n", elem_idx, diff * diff);
          // }
        }
        errors[0] += data * data;
      }
    }
  }

  MGARDX_EXEC
  static void Parallel_Bitplanes_Serial_Error(T *v, T_error *temp,
                                              T_error *errors, SIZE num_elems,
                                              SIZE num_bitplanes, SIZE IdX,
                                              SIZE IdY) {
    SIZE bitplane_idx = IdY * nblockx + IdX;
    if (bitplane_idx < num_bitplanes) {
      for (SIZE elem_idx = 0; elem_idx < num_elems; elem_idx++) {
        T data = v[elem_idx];
        T_fp fp_data = (T_fp)fabs(v[elem_idx]);
        T_sfp fps_data = (T_sfp)data;
        T_fp ngb_data = Math<HIP>::binary2negabinary(fps_data);
        T_error mantissa;
        if (BinaryType == BINARY) {
          mantissa = fabs(data) - fp_data;
        } else if (BinaryType == NEGABINARY) {
          mantissa = data - fps_data;
        }
        uint64_t mask = (1 << bitplane_idx) - 1;
        T_error diff = 0;
        if (BinaryType == BINARY) {
          diff = (T_error)(fp_data & mask) + mantissa;
        } else if (BinaryType == NEGABINARY) {
          diff =
              (T_error)Math<HIP>::negabinary2binary(ngb_data & mask) + mantissa;
        }
        errors[num_bitplanes - bitplane_idx] += diff * diff;
      }
    }
    if (bitplane_idx == 0) {
      for (SIZE elem_idx = 0; elem_idx < num_elems; elem_idx++) {
        T data = v[elem_idx];
        errors[0] += data * data;
      }
    }
  }

  MGARDX_EXEC
  static void Parallel_Bitplanes_Atomic_Error(T *v, T_error *temp,
                                              T_error *errors, SIZE num_elems,
                                              SIZE num_bitplanes, SIZE IdX,
                                              SIZE IdY) {
    for (SIZE elem_idx = IdX; elem_idx < num_elems; elem_idx += nblockx) {
      for (SIZE bitplane_idx = IdY; bitplane_idx < num_bitplanes;
           bitplane_idx += nblocky) {
        T data = v[elem_idx];
        T_fp fp_data = (T_fp)fabs(v[elem_idx]);
        T_sfp fps_data = (T_sfp)data;
        T_fp ngb_data = Math<HIP>::binary2negabinary(fps_data);
        T_error mantissa;
        if (BinaryType == BINARY) {
          mantissa = fabs(data) - fp_data;
        } else if (BinaryType == NEGABINARY) {
          mantissa = data - fps_data;
        }
        uint64_t mask = (1 << bitplane_idx) - 1;
        T_error diff = 0;
        if (BinaryType == BINARY) {
          diff = (T_error)(fp_data & mask) + mantissa;
        } else if (BinaryType == NEGABINARY) {
          diff =
              (T_error)Math<HIP>::negabinary2binary(ngb_data & mask) + mantissa;
        }
        temp[(num_bitplanes - bitplane_idx) * num_elems + elem_idx] =
            diff * diff;
        if (bitplane_idx == 0) {
          temp[elem_idx] = data * data;
        }
      }
    }
    __syncthreads();
    // if (blockIdx.x == 0 && IdX == 0 && IdY == 0) {
    //     for (SIZE elem_idx = 0; elem_idx < num_elems; elem_idx += 1) {
    //       printf("elem error[%u]: %f\n", elem_idx, temp[(2) * num_elems +
    //       elem_idx]);
    //     }
    // }
    for (SIZE bitplane_idx = IdY; bitplane_idx < num_bitplanes + 1;
         bitplane_idx += nblocky) {
      for (SIZE elem_idx = IdX; elem_idx < ((num_elems - 1) / 32 + 1) * 32;
           elem_idx += 32) {
        T_error error = 0;
        if (elem_idx < num_elems) {
          error = temp[(num_bitplanes - bitplane_idx) * num_elems + elem_idx];
        }
        T_error *sum = &(errors[num_bitplanes - bitplane_idx]);
        atomicAdd(sum, error);
      }
    }
  }

  MGARDX_EXEC
  static void Parallel_Bitplanes_Reduce_Error(T *v, T_error *temp,
                                              T_error *errors, SIZE num_elems,
                                              SIZE num_bitplanes, SIZE IdX,
                                              SIZE IdY) {
    for (SIZE elem_idx = IdX; elem_idx < num_elems; elem_idx += nblockx) {
      for (SIZE bitplane_idx = IdY; bitplane_idx < num_bitplanes;
           bitplane_idx += nblocky) {
        T data = v[elem_idx];
        T_fp fp_data = (T_fp)fabs(v[elem_idx]);
        T_sfp fps_data = (T_sfp)data;
        T_fp ngb_data = Math<HIP>::binary2negabinary(fps_data);
        T_error mantissa;
        if (BinaryType == BINARY) {
          mantissa = fabs(data) - fp_data;
        } else if (BinaryType == NEGABINARY) {
          mantissa = data - fps_data;
        }
        uint64_t mask = (1 << bitplane_idx) - 1;
        T_error diff = 0;
        if (BinaryType == BINARY) {
          diff = (T_error)(fp_data & mask) + mantissa;
        } else if (BinaryType == NEGABINARY) {
          diff =
              (T_error)Math<HIP>::negabinary2binary(ngb_data & mask) + mantissa;
        }
        temp[(num_bitplanes - bitplane_idx) * num_elems + elem_idx] =
            diff * diff;
        // if (blockIdx.x == 0 && num_bitplanes - bitplane_idx == 31) {
        //   printf("elem_idx: %u, data: %f, fp_data: %u, mask: %u, mantissa:
        //   %f, diff: %f\n",
        //           elem_idx, data, fp_data, mask, mantissa, diff);
        // }
        if (bitplane_idx == 0) {
          temp[elem_idx] = data * data;
        }
      }
    }
    __syncthreads();

    // if (blockIdx.x == 0 && IdX == 0 && IdY == 0) {
    //   for (SIZE elem_idx = 0; elem_idx < num_elems; elem_idx += 1) {
    //     printf("elem error[%u]: %f\n", elem_idx, temp[(31) * num_elems +
    //     elem_idx]);
    //   }
    // }

    __shared__ WarpReduceStorageType warp_storage[nblocky];

    for (SIZE bitplane_idx = IdY; bitplane_idx < num_bitplanes + 1;
         bitplane_idx += nblocky) {
      T error_sum = 0;
      for (SIZE elem_idx = IdX; elem_idx < ((num_elems - 1) / 32 + 1) * 32;
           elem_idx += 32) {
        T_error error = 0;
        if (elem_idx < num_elems) {
          error = temp[(num_bitplanes - bitplane_idx) * num_elems + elem_idx];
        }
        error_sum += WarpReduceType(warp_storage[IdY]).Sum(error);
        // errors[num_bitplanes - bitplane_idx] +=
        // WarpReduceType(warp_storage[IdY]).Sum(error);
      }
      if (IdX == 0) {
        errors[num_bitplanes - bitplane_idx] = error_sum;
      }
    }
    // __syncthreads();
    // if (blockIdx.x == 0 && IdX == 0 && IdY == 0) {
    //   for (int i = 0; i < num_bitplanes + 1; i++) {
    //     printf("error[%d]: %f\n", i, errors[i]);
    //   }
    // }
  }

  MGARDX_EXEC
  static void Collect(T *v, T_error *temp, T_error *errors, SIZE num_elems,
                      SIZE num_bitplanes, SIZE IdX, SIZE IdY) {
    if (METHOD == Error_Collecting_Serial_All)
      Serial_All(v, temp, errors, num_elems, num_bitplanes, IdX, IdY);
    else if (METHOD == Error_Collecting_Parallel_Bitplanes_Serial_Error)
      Parallel_Bitplanes_Serial_Error(v, temp, errors, num_elems, num_bitplanes,
                                      IdX, IdY);
    else if (METHOD == Error_Collecting_Parallel_Bitplanes_Atomic_Error)
      Parallel_Bitplanes_Atomic_Error(v, temp, errors, num_elems, num_bitplanes,
                                      IdX, IdY);
    else if (METHOD == Error_Collecting_Parallel_Bitplanes_Reduce_Error)
      Parallel_Bitplanes_Reduce_Error(v, temp, errors, num_elems, num_bitplanes,
                                      IdX, IdY);
    // else if (METHOD == Error_Collecting_Disable) {}
    // else { printf("Error Collecting Wrong Algorithm Type!\n");  }
  }
};

template <typename T_org, typename T_trans, OPTION ALIGN, OPTION METHOD, SIZE b,
          SIZE B>
struct WarpBitTranspose<T_org, T_trans, ALIGN, METHOD, b, B, HIP> {

  typedef hipcub::WarpReduce<T_trans> WarpReduceType;
  using WarpReduceStorageType = typename WarpReduceType::TempStorage;

  MGARDX_EXEC static void Serial_All(T_org *v, SIZE inc_v, T_trans *tv,
                                     SIZE inc_tv, SIZE LaneId) {
    // long long start;
    // if (IdX == 0 && IdY == 0) { start = clock64(); }
    if (LaneId == 0) {
      for (SIZE B_idx = 0; B_idx < B; B_idx++) {
        T_trans buffer = 0;
        for (SIZE b_idx = 0; b_idx < b; b_idx++) {
          T_trans bit =
              (v[b_idx * inc_v] >> (sizeof(T_org) * 8 - 1 - B_idx)) & 1u;
          // if (blockIdx.x == 0 )printf("bit: %u\n", bit);
          if (ALIGN == ALIGN_LEFT) {
            buffer += bit << (sizeof(T_trans) * 8 - 1 - b_idx);
          } else if (ALIGN == ALIGN_RIGHT) {
            buffer += bit << (b - 1 - b_idx);
          } else {
          }
        }
        tv[B_idx * inc_tv] = buffer;
      }
    }
    // if (IdX == 0 && IdY == 0) { start = clock64() - start;
    // printf("Serial_All time : %llu\n", start); }
  }

  MGARDX_EXEC static void Parallel_B_Serial_b(T_org *v, SIZE inc_v, T_trans *tv,
                                              SIZE inc_tv, SIZE LaneId) {
    // long long start;
    // if (IdX == 0 && IdY == 0) { start = clock64(); }
    for (SIZE B_idx = LaneId; B_idx < B; B_idx += MGARDX_WARP_SIZE) {
      T_trans buffer = 0;
      for (SIZE b_idx = 0; b_idx < b; b_idx++) {
        T_trans bit =
            (v[b_idx * inc_v] >> (sizeof(T_org) * 8 - 1 - B_idx)) & 1u;
        // if (blockIdx.x == 0 )printf("bit: %u\n", bit);
        if (ALIGN == ALIGN_LEFT) {
          buffer += bit << (sizeof(T_trans) * 8 - 1 - b_idx);
        } else if (ALIGN == ALIGN_RIGHT) {
          buffer += bit << (b - 1 - b_idx);
        } else {
        }
      }
      tv[B_idx * inc_tv] = buffer;
    }
    // if (IdX == 0 && IdY == 0) { start = clock64() - start;
    // printf("Parallel_B_Serial_b time : %llu\n", start); }
  }

  MGARDX_EXEC static void Serial_B_Atomic_b(T_org *v, SIZE inc_v, T_trans *tv,
                                            SIZE inc_tv, SIZE LaneId) {
    // long long start;
    // if (IdX == 0 && IdY == 0) { start = clock64(); }
    for (SIZE B_idx = LaneId; B_idx < B; B_idx += MGARDX_WARP_SIZE) {
      tv[B_idx * inc_tv] = 0;
    }
    for (SIZE B_idx = 0; B_idx < B; B_idx++) {
      for (SIZE b_idx = LaneId; b_idx < b; b_idx += MGARDX_WARP_SIZE) {
        T_trans bit =
            (v[b_idx * inc_v] >> (sizeof(T_org) * 8 - 1 - B_idx)) & 1u;
        T_trans shifted_bit = 0;
        if (ALIGN == ALIGN_LEFT) {
          shifted_bit = bit << (sizeof(T_trans) * 8 - 1 - b_idx);
        } else if (ALIGN == ALIGN_RIGHT) {
          shifted_bit = bit << (b - 1 - b_idx);
        } else {
        }
        T_trans *sum = &(tv[B_idx * inc_tv]);
        atomicAdd(sum, shifted_bit);
      }
    }
    // if (IdX == 0 && IdY == 0) { start = clock64() - start;
    // printf("Serial_B_Atomic_b time : %llu\n", start); }
  }

  MGARDX_EXEC static void Serial_B_Reduce_b(T_org *v, SIZE inc_v, T_trans *tv,
                                            SIZE inc_tv, SIZE LaneId) {
    // long long start;
    // if (IdX == 0 && IdY == 0) { start = clock64(); }
    __shared__ WarpReduceStorageType warp_storage;

    T_trans bit = 0;
    T_trans shifted_bit = 0;
    T_trans sum = 0;
    for (SIZE B_idx = 0; B_idx < B; B_idx++) {
      sum = 0;
      for (SIZE b_idx = LaneId;
           b_idx < ((b - 1) / MGARDX_WARP_SIZE + 1) * MGARDX_WARP_SIZE;
           b_idx += MGARDX_WARP_SIZE) {
        if (b_idx < b) {
          bit = (v[b_idx * inc_v] >> (sizeof(T_org) * 8 - 1 - B_idx)) & 1u;
        }
        shifted_bit = 0;
        if (ALIGN == ALIGN_LEFT) {
          shifted_bit = bit << (sizeof(T_trans) * 8 - 1 - b_idx);
        } else if (ALIGN == ALIGN_RIGHT) {
          shifted_bit = bit << (b - 1 - b_idx);
        } else {
        }
        sum += WarpReduceType(warp_storage).Sum(shifted_bit);
      }
      if (LaneId == 0) {
        tv[B_idx * inc_tv] = sum;
      }
    }
    // if (IdX == 0 && IdY == 0) { start = clock64() - start;
    // printf("Serial_B_Reduce_b time : %llu\n", start); }
  }

  MGARDX_EXEC static void Serial_B_Ballot_b(T_org *v, SIZE inc_v, T_trans *tv,
                                            SIZE inc_tv, SIZE LaneId) {
    // long long start;
    // if (IdX == 0 && IdY == 0) { start = clock64(); }
    T_trans bit = 0;
    T_trans sum = 0;
    for (SIZE B_idx = 0; B_idx < B; B_idx++) {
      sum = 0;
      SIZE shift = 0;
      for (SIZE b_idx = LaneId;
           b_idx < ((b - 1) / MGARDX_WARP_SIZE + 1) * MGARDX_WARP_SIZE;
           b_idx += MGARDX_WARP_SIZE) {
        bit = 0;
        if (b_idx < b) {
          if (ALIGN == ALIGN_LEFT) {
            bit = (v[(sizeof(T_trans) * 8 - 1 - b_idx) * inc_v] >>
                   (sizeof(T_org) * 8 - 1 - B_idx)) &
                  1u;
          } else if (ALIGN == ALIGN_RIGHT) {
            bit = (v[(b - 1 - b_idx) * inc_v] >>
                   (sizeof(T_org) * 8 - 1 - B_idx)) &
                  1u;
          } else {
          }
        }
        sum += ((T_trans)__ballot(bit)) << shift;
        shift += MGARDX_WARP_SIZE;
      }
      if (LaneId == 0) {
        tv[B_idx * inc_tv] = sum;
      }
    }
    // if (IdX == 0 && IdY == 0) { start = clock64() - start;
    // printf("Serial_B_Ballot_b time : %llu\n", start); }
  }

  MGARDX_EXEC static void Transpose(T_org *v, SIZE inc_v, T_trans *tv,
                                    SIZE inc_tv, SIZE LaneId) {
    if (METHOD == Warp_Bit_Transpose_Serial_All) {
      Serial_All(v, inc_v, tv, inc_tv, LaneId);
    } else if (METHOD == Warp_Bit_Transpose_Parallel_B_Serial_b) {
      Parallel_B_Serial_b(v, inc_v, tv, inc_tv, LaneId);
    } else if (METHOD == Warp_Bit_Transpose_Serial_B_Atomic_b) {
      Serial_B_Atomic_b(v, inc_v, tv, inc_tv, LaneId);
    } else if (METHOD == Warp_Bit_Transpose_Serial_B_Reduce_b) {
      Serial_B_Reduce_b(v, inc_v, tv, inc_tv, LaneId);
    } else if (METHOD == Warp_Bit_Transpose_Serial_B_Ballot_b) {
      Serial_B_Ballot_b(v, inc_v, tv, inc_tv, LaneId);
    }
  }
};

template <typename T, typename T_fp, typename T_sfp, typename T_error,
          OPTION METHOD, OPTION BinaryType, SIZE num_elems, SIZE num_bitplanes>
struct WarpErrorCollect<T, T_fp, T_sfp, T_error, METHOD, BinaryType, num_elems,
                        num_bitplanes, HIP> {

  typedef hipcub::WarpReduce<T_error> WarpReduceType;
  using WarpReduceStorageType = typename WarpReduceType::TempStorage;

  MGARDX_EXEC static void Serial_All(T *v, T_error *errors, SIZE LaneId) {
    if (LaneId == 0) {
      for (SIZE elem_idx = 0; elem_idx < num_elems; elem_idx++) {
        T data = v[elem_idx];
        T_fp fp_data = (T_fp)fabs(data);
        T_sfp fps_data = (T_sfp)data;
        T_fp ngb_data = Math<HIP>::binary2negabinary(fps_data);
        T_error mantissa;
        if (BinaryType == BINARY) {
          mantissa = fabs(data) - fp_data;
        } else if (BinaryType == NEGABINARY) {
          mantissa = data - fps_data;
        }

        // printf("fp: %u error: \n", fp_data);
        for (SIZE bitplane_idx = 0; bitplane_idx < num_bitplanes;
             bitplane_idx++) {
          uint64_t mask = (1 << bitplane_idx) - 1;
          T_error diff = 0;
          if (BinaryType == BINARY) {
            diff = (T_error)(fp_data & mask) + mantissa;
          } else if (BinaryType == NEGABINARY) {
            diff = (T_error)Math<HIP>::negabinary2binary(ngb_data & mask) +
                   mantissa;
          }
          errors[num_bitplanes - bitplane_idx] += diff * diff;
          // printf("%f ", diff * diff);
        }
        errors[0] += data * data;
        // printf("%f \n", data * data);
      }
    }
  }

  MGARDX_EXEC static void Parallel_Bitplanes_Serial_Error(T *v, T_error *errors,
                                                          SIZE LaneId) {

    __shared__ WarpReduceStorageType warp_storage;

    for (SIZE bitplane_idx = LaneId; bitplane_idx < num_bitplanes;
         bitplane_idx += MGARDX_WARP_SIZE) {
      for (SIZE elem_idx = 0; elem_idx < num_elems; elem_idx++) {
        T data = v[elem_idx];
        T_fp fp_data = (T_fp)fabs(data);
        T_sfp fps_data = (T_sfp)data;
        T_fp ngb_data = Math<HIP>::binary2negabinary(fps_data);
        T_error mantissa;
        if (BinaryType == BINARY) {
          mantissa = fabs(data) - fp_data;
        } else if (BinaryType == NEGABINARY) {
          mantissa = data - fps_data;
        }

        uint64_t mask = (1 << bitplane_idx) - 1;
        T_error diff = 0;
        if (BinaryType == BINARY) {
          diff = (T_error)(fp_data & mask) + mantissa;
        } else if (BinaryType == NEGABINARY) {
          diff =
              (T_error)Math<HIP>::negabinary2binary(ngb_data & mask) + mantissa;
        }
        errors[num_bitplanes - bitplane_idx] += diff * diff;
      }
    }

    T data = 0;
    for (SIZE elem_idx = LaneId;
         elem_idx < ((num_elems - 1) / MGARDX_WARP_SIZE + 1) * MGARDX_WARP_SIZE;
         elem_idx += MGARDX_WARP_SIZE) {
      if (elem_idx < num_elems) {
        data = v[elem_idx];
      }
      T_error error_sum = WarpReduceType(warp_storage).Sum(data * data);
      if (LaneId == 0)
        errors[0] += error_sum;
    }
  }

  MGARDX_EXEC static void Serial_Bitplanes_Atomic_Error(T *v, T_error *errors,
                                                        SIZE LaneId) {
    T data = 0;
    for (SIZE elem_idx = LaneId;
         elem_idx < ((num_elems - 1) / MGARDX_WARP_SIZE + 1) * MGARDX_WARP_SIZE;
         elem_idx += MGARDX_WARP_SIZE) {
      if (elem_idx < num_elems) {
        data = v[elem_idx];
      } else {
        data = 0;
      }

      T_fp fp_data = (T_fp)fabs(data);
      T_sfp fps_data = (T_sfp)data;
      T_fp ngb_data = Math<HIP>::binary2negabinary(fps_data);
      T_error mantissa;
      if (BinaryType == BINARY) {
        mantissa = fabs(data) - fp_data;
      } else if (BinaryType == NEGABINARY) {
        mantissa = data - fps_data;
      }

      // printf("fp: %u error: \n", fp_data);
      for (SIZE bitplane_idx = 0; bitplane_idx < num_bitplanes;
           bitplane_idx++) {
        uint64_t mask = (1 << bitplane_idx) - 1;
        T_error diff = 0;
        if (BinaryType == BINARY) {
          diff = (T_error)(fp_data & mask) + mantissa;
        } else if (BinaryType == NEGABINARY) {
          diff =
              (T_error)Math<HIP>::negabinary2binary(ngb_data & mask) + mantissa;
        }
        T_error *sum = &(errors[num_bitplanes - bitplane_idx]);
        atomicAdd(sum, diff * diff);
      }
      T_error *sum = &(errors[0]);
      atomicAdd(sum, data * data);
    }
  }

  MGARDX_EXEC static void Serial_Bitplanes_Reduce_Error(T *v, T_error *errors,
                                                        SIZE LaneId) {

    __shared__ WarpReduceStorageType warp_storage;

    T data = 0;
    for (SIZE elem_idx = LaneId;
         elem_idx < ((num_elems - 1) / MGARDX_WARP_SIZE + 1) * MGARDX_WARP_SIZE;
         elem_idx += MGARDX_WARP_SIZE) {
      if (elem_idx < num_elems) {
        data = v[elem_idx];
      } else {
        data = 0;
      }

      T_fp fp_data = (T_fp)fabs(data);
      T_sfp fps_data = (T_sfp)data;
      T_fp ngb_data = Math<HIP>::binary2negabinary(fps_data);
      T_error mantissa;
      if (BinaryType == BINARY) {
        mantissa = fabs(data) - fp_data;
      } else if (BinaryType == NEGABINARY) {
        mantissa = data - fps_data;
      }

      // printf("fp: %u error: \n", fp_data);
      for (SIZE bitplane_idx = 0; bitplane_idx < num_bitplanes;
           bitplane_idx++) {
        uint64_t mask = (1 << bitplane_idx) - 1;
        T_error diff = 0;
        if (BinaryType == BINARY) {
          diff = (T_error)(fp_data & mask) + mantissa;
        } else if (BinaryType == NEGABINARY) {
          diff =
              (T_error)Math<HIP>::negabinary2binary(ngb_data & mask) + mantissa;
        }
        T_error error_sum = WarpReduceType(warp_storage).Sum(diff * diff);
        if (LaneId == 0)
          errors[num_bitplanes - bitplane_idx] += error_sum;
      }
      T_error error_sum = WarpReduceType(warp_storage).Sum(data * data);
      if (LaneId == 0)
        errors[0] += error_sum;
    }
  }

  MGARDX_EXEC static void Collect(T *v, T_error *errors, SIZE LaneId) {
    if (METHOD == Warp_Error_Collecting_Serial_All) {
      Serial_All(v, errors, LaneId);
    }
    if (METHOD == Warp_Error_Collecting_Parallel_Bitplanes_Serial_Error) {
      Parallel_Bitplanes_Serial_Error(v, errors, LaneId);
    }
    if (METHOD == Warp_Error_Collecting_Serial_Bitplanes_Atomic_Error) {
      Serial_Bitplanes_Atomic_Error(v, errors, LaneId);
    }
    if (METHOD == Warp_Error_Collecting_Serial_Bitplanes_Reduce_Error) {
      Serial_Bitplanes_Reduce_Error(v, errors, LaneId);
    }
  }
};

template <typename Task> void HuffmanCLCustomizedNoCGKernel(Task task) {
  // std::cout << "calling HuffmanCLCustomizedNoCGKernel\n";
  dim3 threadsPerBlock(task.GetBlockDimX(), task.GetBlockDimY(),
                       task.GetBlockDimZ());
  dim3 blockPerGrid(task.GetGridDimX(), task.GetGridDimY(), task.GetGridDimZ());
  size_t sm_size = task.GetSharedMemorySize();
  hipStream_t stream = DeviceRuntime<HIP>::GetQueue(task.GetQueueIdx());

  // std::cout << "calling Single_Operation1_Kernel\n";
  Single_Operation1_Kernel<<<blockPerGrid, threadsPerBlock, sm_size, stream>>>(
      task);
  ErrorSyncCheck(hipDeviceSynchronize(), task);

  // std::cout << "calling LoopCondition1\n";
  while (task.GetFunctor().LoopCondition1()) {
    ErrorSyncCheck(hipDeviceSynchronize(), task);

    // std::cout << "calling Single_Operation2_Kernel\n";
    Single_Operation2_Kernel<<<blockPerGrid, threadsPerBlock, sm_size,
                               stream>>>(task);
    ErrorSyncCheck(hipDeviceSynchronize(), task);

    // std::cout << "calling Single_Operation3_Kernel\n";
    Single_Operation3_Kernel<<<blockPerGrid, threadsPerBlock, sm_size,
                               stream>>>(task);
    ErrorSyncCheck(hipDeviceSynchronize(), task);

    // std::cout << "calling Single_Operation4_Kernel\n";
    Single_Operation4_Kernel<<<blockPerGrid, threadsPerBlock, sm_size,
                               stream>>>(task);
    ErrorSyncCheck(hipDeviceSynchronize(), task);

    // std::cout << "calling BranchCondition1\n";
    if (task.GetFunctor().BranchCondition1()) {
      ErrorSyncCheck(hipDeviceSynchronize(), task);

      // std::cout << "calling ParallelMergeKernel\n";
      ParallelMergeKernel<<<blockPerGrid, threadsPerBlock, sm_size, stream>>>(
          task);
      ErrorSyncCheck(hipDeviceSynchronize(), task);

      // std::cout << "calling Single_Operation10_Kernel\n";
      Single_Operation10_Kernel<<<blockPerGrid, threadsPerBlock, sm_size,
                                  stream>>>(task);
      ErrorSyncCheck(hipDeviceSynchronize(), task);
    }

    // std::cout << "calling Single_Operation11_Kernel\n";
    Single_Operation11_Kernel<<<blockPerGrid, threadsPerBlock, sm_size,
                                stream>>>(task);
    ErrorSyncCheck(hipDeviceSynchronize(), task);

    // std::cout << "calling Single_Operation12_Kernel\n";
    Single_Operation12_Kernel<<<blockPerGrid, threadsPerBlock, sm_size,
                                stream>>>(task);
    ErrorSyncCheck(hipDeviceSynchronize(), task);

    // std::cout << "calling Single_Operation13_Kernel\n";
    Single_Operation13_Kernel<<<blockPerGrid, threadsPerBlock, sm_size,
                                stream>>>(task);
    ErrorSyncCheck(hipDeviceSynchronize(), task);

    // std::cout << "calling Single_Operation14_Kernel\n";
    Single_Operation14_Kernel<<<blockPerGrid, threadsPerBlock, sm_size,
                                stream>>>(task);
    ErrorSyncCheck(hipDeviceSynchronize(), task);
  }
}

template <typename Task> void HuffmanCWCustomizedNoCGKernel(Task task) {
  // std::cout << "calling HuffmanCWCustomizedNoCGKernel\n";
  dim3 threadsPerBlock(task.GetBlockDimX(), task.GetBlockDimY(),
                       task.GetBlockDimZ());
  dim3 blockPerGrid(task.GetGridDimX(), task.GetGridDimY(), task.GetGridDimZ());
  size_t sm_size = task.GetSharedMemorySize();
  hipStream_t stream = DeviceRuntime<HIP>::GetQueue(task.GetQueueIdx());

  // std::cout << "calling Single_Operation1_Kernel\n";
  Single_Operation1_Kernel<<<blockPerGrid, threadsPerBlock, sm_size, stream>>>(
      task);
  ErrorSyncCheck(hipDeviceSynchronize(), task);
  // std::cout << "calling Single_Operation2_Kernel\n";
  Single_Operation2_Kernel<<<blockPerGrid, threadsPerBlock, sm_size, stream>>>(
      task);
  ErrorSyncCheck(hipDeviceSynchronize(), task);
  // std::cout << "calling Single_Operation3_Kernel\n";
  Single_Operation3_Kernel<<<blockPerGrid, threadsPerBlock, sm_size, stream>>>(
      task);
  ErrorSyncCheck(hipDeviceSynchronize(), task);

  // std::cout << "calling LoopCondition1\n";
  while (task.GetFunctor().LoopCondition1()) {
    ErrorSyncCheck(hipDeviceSynchronize(), task);

    // std::cout << "calling Single_Operation4_Kernel\n";
    Single_Operation4_Kernel<<<blockPerGrid, threadsPerBlock, sm_size,
                               stream>>>(task);
    ErrorSyncCheck(hipDeviceSynchronize(), task);

    // std::cout << "calling Single_Operation5_Kernel\n";
    Single_Operation5_Kernel<<<blockPerGrid, threadsPerBlock, sm_size,
                               stream>>>(task);
    ErrorSyncCheck(hipDeviceSynchronize(), task);

    // std::cout << "calling Single_Operation6_Kernel\n";
    Single_Operation6_Kernel<<<blockPerGrid, threadsPerBlock, sm_size,
                               stream>>>(task);
    ErrorSyncCheck(hipDeviceSynchronize(), task);

    // std::cout << "calling Single_Operation7_Kernel\n";
    Single_Operation7_Kernel<<<blockPerGrid, threadsPerBlock, sm_size,
                               stream>>>(task);
    ErrorSyncCheck(hipDeviceSynchronize(), task);

    // std::cout << "calling Single_Operation8_Kernel\n";
    Single_Operation8_Kernel<<<blockPerGrid, threadsPerBlock, sm_size,
                               stream>>>(task);
    ErrorSyncCheck(hipDeviceSynchronize(), task);
  }

  // std::cout << "calling Single_Operation9_Kernel\n";
  Single_Operation9_Kernel<<<blockPerGrid, threadsPerBlock, sm_size, stream>>>(
      task);
  ErrorSyncCheck(hipDeviceSynchronize(), task);

  // std::cout << "calling Single_Operation10_Kernel\n";
  Single_Operation10_Kernel<<<blockPerGrid, threadsPerBlock, sm_size, stream>>>(
      task);
  ErrorSyncCheck(hipDeviceSynchronize(), task);
}

template <typename TaskType> class DeviceAdapter<TaskType, HIP> {
public:
  MGARDX_CONT
  DeviceAdapter(){};

  MGARDX_CONT
  int IsResourceEnough(TaskType &task) {
    if (task.GetBlockDimX() * task.GetBlockDimY() * task.GetBlockDimZ() >
        DeviceRuntime<HIP>::GetMaxNumThreadsPerTB()) {
      return THREADBLOCK_TOO_LARGE;
    }
    if (task.GetSharedMemorySize() >
        DeviceRuntime<HIP>::GetMaxSharedMemorySize()) {
      return SHARED_MEMORY_TOO_LARGE;
    }
    return RESOURCE_ENOUGH;
  }

  MGARDX_CONT
  ExecutionReturn Execute(TaskType &task) {
    dim3 threadsPerBlock(task.GetBlockDimX(), task.GetBlockDimY(),
                         task.GetBlockDimZ());
    dim3 blockPerGrid(task.GetGridDimX(), task.GetGridDimY(),
                      task.GetGridDimZ());
    size_t sm_size = task.GetSharedMemorySize();
    // printf("exec config (%d %d %d) (%d %d %d) sm_size: %llu\n",
    // threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z,
    //                 blockPerGrid.x, blockPerGrid.y, blockPerGrid.z, sm_size);
    hipStream_t stream = DeviceRuntime<HIP>::GetQueue(task.GetQueueIdx());

    if (DeviceRuntime<HIP>::PrintKernelConfig) {
      std::cout << log::log_info << task.GetFunctorName() << ": <"
                << task.GetBlockDimX() << ", " << task.GetBlockDimY() << ", "
                << task.GetBlockDimZ() << "> <" << task.GetGridDimX() << ", "
                << task.GetGridDimY() << ", " << task.GetGridDimZ() << ">\n";
    }

    ExecutionReturn ret;
    if (IsResourceEnough(task) != RESOURCE_ENOUGH) {
      if (DeviceRuntime<HIP>::PrintKernelConfig) {
        if (IsResourceEnough(task) == THREADBLOCK_TOO_LARGE) {
          std::cout << log::log_info << "threadblock too large.\n";
        }
        if (IsResourceEnough(task) == SHARED_MEMORY_TOO_LARGE) {
          std::cout << log::log_info << "shared memory too large.\n";
        }
      }
      ret.success = false;
      ret.execution_time = std::numeric_limits<double>::max();
      return ret;
    }

    Timer timer;
    if (DeviceRuntime<HIP>::TimingAllKernels ||
        AutoTuner<HIP>::ProfileKernels) {
      DeviceRuntime<HIP>::SyncDevice();
      timer.start();
    }

    // if constexpr evaluate at compile time otherwise this does not compile
    if constexpr (std::is_base_of<Functor<HIP>,
                                  typename TaskType::Functor>::value) {
      Kernel<<<blockPerGrid, threadsPerBlock, sm_size, stream>>>(task);
    } else if constexpr (std::is_base_of<IterFunctor<HIP>,
                                         typename TaskType::Functor>::value) {
      IterKernel<<<blockPerGrid, threadsPerBlock, sm_size, stream>>>(task);
    } else if constexpr (std::is_base_of<HuffmanCLCustomizedFunctor<HIP>,
                                         typename TaskType::Functor>::value) {
      if (task.GetFunctor().use_CG && DeviceRuntime<HIP>::SupportCG()) {
        void *Args[] = {(void *)&task};
        hipLaunchCooperativeKernel((void *)HuffmanCLCustomizedKernel<TaskType>,
                                   blockPerGrid, threadsPerBlock, Args, sm_size,
                                   stream);
      } else {
        HuffmanCLCustomizedNoCGKernel(task);
      }
    } else if constexpr (std::is_base_of<HuffmanCWCustomizedFunctor<HIP>,
                                         typename TaskType::Functor>::value) {
      if (task.GetFunctor().use_CG && DeviceRuntime<HIP>::SupportCG()) {
        void *Args[] = {(void *)&task};
        hipLaunchCooperativeKernel((void *)HuffmanCWCustomizedKernel<TaskType>,
                                   blockPerGrid, threadsPerBlock, Args, sm_size,
                                   stream);
      } else {
        HuffmanCWCustomizedNoCGKernel(task);
      }
    }
    ErrorAsyncCheck(hipGetLastError(), task);
    gpuErrchk(hipGetLastError());
    if (DeviceRuntime<HIP>::SyncAllKernelsAndCheckErrors) {
      ErrorSyncCheck(hipDeviceSynchronize(), task);
    }

    if (DeviceRuntime<HIP>::TimingAllKernels ||
        AutoTuner<HIP>::ProfileKernels) {
      DeviceRuntime<HIP>::SyncDevice();
      timer.end();
      if (DeviceRuntime<HIP>::TimingAllKernels) {
        timer.print(task.GetFunctorName());
      }
      if (AutoTuner<HIP>::ProfileKernels) {
        ret.success = true;
        ret.execution_time = timer.get();
      }
    }
    return ret;
  }
};

struct AbsMaxOp {
  template <typename T>
  __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    return (fabs(b) > fabs(a)) ? fabs(b) : fabs(a);
  }
};

struct SquareOp {
  template <typename T>
  __device__ __forceinline__ T operator()(const T &a) const {
    return a * a;
  }
};

template <> class DeviceCollective<HIP> {
public:
  MGARDX_CONT
  DeviceCollective(){};

  template <typename T>
  MGARDX_CONT static void Sum(SIZE n, SubArray<1, T, HIP> &v,
                              SubArray<1, T, HIP> &result, int queue_idx) {
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    hipStream_t stream = DeviceRuntime<HIP>::GetQueue(queue_idx);
    bool debug = DeviceRuntime<HIP>::SyncAllKernelsAndCheckErrors;
    hipcub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, v.data(),
                              result.data(), n, stream, debug);
    MemoryManager<HIP>().Malloc1D(d_temp_storage, temp_storage_bytes,
                                  queue_idx);
    hipcub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, v.data(),
                              result.data(), n, stream, debug);
    DeviceRuntime<HIP>::SyncQueue(queue_idx);
    MemoryManager<HIP>().Free(d_temp_storage);
  }

  template <typename T>
  MGARDX_CONT static void AbsMax(SIZE n, SubArray<1, T, HIP> &v,
                                 SubArray<1, T, HIP> &result, int queue_idx) {
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    AbsMaxOp absMaxOp;
    hipStream_t stream = DeviceRuntime<HIP>::GetQueue(queue_idx);
    bool debug = DeviceRuntime<HIP>::SyncAllKernelsAndCheckErrors;
    hipcub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, v.data(),
                                 result.data(), n, absMaxOp, 0, stream, debug);
    MemoryManager<HIP>().Malloc1D(d_temp_storage, temp_storage_bytes,
                                  queue_idx);
    hipcub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, v.data(),
                                 result.data(), n, absMaxOp, 0, stream, debug);
    DeviceRuntime<HIP>::SyncQueue(queue_idx);
    MemoryManager<HIP>().Free(d_temp_storage);
  }

  template <typename T>
  MGARDX_CONT static void SquareSum(SIZE n, SubArray<1, T, HIP> &v,
                                    SubArray<1, T, HIP> &result,
                                    int queue_idx) {
    SquareOp squareOp;
    hipcub::TransformInputIterator<T, SquareOp, T *> transformed_input_iter(
        v.data(), squareOp);
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    hipStream_t stream = DeviceRuntime<HIP>::GetQueue(queue_idx);
    bool debug = DeviceRuntime<HIP>::SyncAllKernelsAndCheckErrors;
    hipcub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                              transformed_input_iter, result.data(), n, stream,
                              debug);
    MemoryManager<HIP>().Malloc1D(d_temp_storage, temp_storage_bytes,
                                  queue_idx);
    hipcub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                              transformed_input_iter, result.data(), n, stream,
                              debug);
    DeviceRuntime<HIP>::SyncQueue(queue_idx);
    MemoryManager<HIP>().Free(d_temp_storage);
  }

  template <typename T>
  MGARDX_CONT static void ScanSumInclusive(SIZE n, SubArray<1, T, HIP> &v,
                                           SubArray<1, T, HIP> &result,
                                           int queue_idx) {
    Byte *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    hipStream_t stream = DeviceRuntime<HIP>::GetQueue(queue_idx);
    bool debug = DeviceRuntime<HIP>::SyncAllKernelsAndCheckErrors;
    hipcub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                     v.data(), result.data(), n);
    MemoryManager<HIP>().Malloc1D(d_temp_storage, temp_storage_bytes,
                                  queue_idx);
    hipcub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                     v.data(), result.data(), n);
    DeviceRuntime<HIP>::SyncQueue(queue_idx);
    MemoryManager<HIP>().Free(d_temp_storage);
  }

  template <typename T>
  MGARDX_CONT static void ScanSumExclusive(SIZE n, SubArray<1, T, HIP> &v,
                                           SubArray<1, T, HIP> &result,
                                           int queue_idx) {
    Byte *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    hipStream_t stream = DeviceRuntime<HIP>::GetQueue(queue_idx);
    bool debug = DeviceRuntime<HIP>::SyncAllKernelsAndCheckErrors;
    hipcub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                     v.data(), result.data(), n);
    MemoryManager<HIP>().Malloc1D(d_temp_storage, temp_storage_bytes,
                                  queue_idx);
    hipcub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                     v.data(), result.data(), n);
    DeviceRuntime<HIP>::SyncQueue(queue_idx);
    MemoryManager<HIP>().Free(d_temp_storage);
  }

  template <typename T>
  MGARDX_CONT static void ScanSumExtended(SIZE n, SubArray<1, T, HIP> &v,
                                          SubArray<1, T, HIP> &result,
                                          int queue_idx) {
    Byte *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    hipStream_t stream = DeviceRuntime<HIP>::GetQueue(queue_idx);
    bool debug = DeviceRuntime<HIP>::SyncAllKernelsAndCheckErrors;
    hipcub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                     v.data(), result.data() + 1, n);
    MemoryManager<HIP>().Malloc1D(d_temp_storage, temp_storage_bytes,
                                  queue_idx);
    hipcub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                     v.data(), result.data() + 1, n);
    T zero = 0;
    MemoryManager<HIP>().Copy1D(result.data(), &zero, 1, queue_idx);
    DeviceRuntime<HIP>::SyncQueue(queue_idx);
    MemoryManager<HIP>().Free(d_temp_storage);
  }

  template <typename KeyT, typename ValueT>
  MGARDX_CONT static void SortByKey(SIZE n, SubArray<1, KeyT, HIP> &keys,
                                    SubArray<1, ValueT, HIP> &values,
                                    int queue_idx) {
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    hipStream_t stream = DeviceRuntime<HIP>::GetQueue(queue_idx);
    bool debug = DeviceRuntime<HIP>::SyncAllKernelsAndCheckErrors;
    Array<1, KeyT, HIP> out_keys({n});
    Array<1, ValueT, HIP> out_values({n});

    hipcub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                       keys.data(), out_keys.data(),
                                       values.data(), out_values.data(), n, 0,
                                       sizeof(KeyT) * 8, stream, debug);
    MemoryManager<HIP>().Malloc1D(d_temp_storage, temp_storage_bytes,
                                  queue_idx);
    hipcub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                       keys.data(), out_keys.data(),
                                       values.data(), out_values.data(), n, 0,
                                       sizeof(KeyT) * 8, stream, debug);
    MemoryManager<HIP>().Copy1D(keys.data(), out_keys.data(), n, queue_idx);
    MemoryManager<HIP>().Copy1D(values.data(), out_values.data(), n, queue_idx);
    DeviceRuntime<HIP>::SyncQueue(queue_idx);
    MemoryManager<HIP>().Free(d_temp_storage);
  }
};

} // namespace mgard_x

#endif