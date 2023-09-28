/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "DeviceAdapter.h"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <iostream>
// #include <mma.h>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

// using namespace nvcuda;
namespace cg = cooperative_groups;

#ifndef MGARD_X_DEVICE_ADAPTER_CUDA_H
#define MGARD_X_DEVICE_ADAPTER_CUDA_H

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

MGARDX_EXEC static float atomicMax(float *address, float val) {
  int *address_as_i = (int *)address;
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = ::atomicCAS(address_as_i, assumed,
                      __float_as_int(::fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

MGARDX_EXEC static double atomicMax(double *address, double val) {
  unsigned long long int *address_as_i = (unsigned long long int *)address;
  unsigned long long int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = ::atomicCAS(address_as_i, assumed,
                      (unsigned long long int)::fmax(val, (double)assumed));
  } while (assumed != old);
  return (double)old;
}

MGARDX_EXEC static uint64_t atomicAdd(uint64_t *address, uint64_t val) {
  unsigned long long int *address_as_i = (unsigned long long int *)address;
  unsigned long long int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = ::atomicCAS(address_as_i, assumed,
                      (unsigned long long int)(val + (uint64_t)assumed));
  } while (assumed != old);
  return (uint64_t)old;
}

MGARDX_EXEC static uint64_t atomicAdd_block(uint64_t *address, uint64_t val) {
  unsigned long long int *address_as_i = (unsigned long long int *)address;
  unsigned long long int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = ::atomicCAS(address_as_i, assumed,
                      (unsigned long long int)(val + (uint64_t)assumed));
  } while (assumed != old);
  return (uint64_t)old;
}

#if defined __CUDA_ARCH__ && __CUDA_ARCH__ < 600
MGARDX_EXEC static double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

namespace mgard_x {

template <typename TaskType>
inline void ErrorAsyncCheckTask(cudaError_t code, TaskType &task,
                                bool abort = true) {
  if (code != cudaSuccess) {
    log::err(std::string(cudaGetErrorString(code)) + " while executing " +
             task.GetFunctorName().c_str() + " with CUDA (Async-check)");
    if (abort)
      exit(code);
  }
}

template <typename TaskType>
inline void ErrorSyncCheckTask(cudaError_t code, TaskType &task,
                               bool abort = true) {
  if (code != cudaSuccess) {
    log::err(std::string(cudaGetErrorString(code)) + " while executing " +
             task.GetFunctorName().c_str() + " with CUDA (Sync-check)");
    if (abort)
      exit(code);
  }
}

inline void ErrorAsyncCheck(cudaError_t code, std::string task,
                            bool abort = true) {
  if (code != cudaSuccess) {
    log::err(std::string(cudaGetErrorString(code)) + " while executing " +
             task.c_str() + " with CUDA (Async-check)");
    if (abort)
      exit(code);
  }
}

inline void ErrorSyncCheck(cudaError_t code, std::string task,
                           bool abort = true) {
  if (code != cudaSuccess) {
    log::err(std::string(cudaGetErrorString(code)) + " while executing " +
             task.c_str() + " with CUDA (Sync-check)");
    if (abort)
      exit(code);
  }
}

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

template <> struct SyncBlock<CUDA> {
  MGARDX_EXEC static void Sync() { __syncthreads(); }
};

template <> struct SyncGrid<CUDA> {
  MGARDX_EXEC static void Sync() { cg::this_grid().sync(); }
};

template <typename T, OPTION MemoryType, OPTION Scope>
struct Atomic<T, MemoryType, Scope, CUDA> {
  MGARDX_EXEC static T Min(T *result, T value) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 600
    if constexpr (Scope == AtomicSystemScope) {
      return atomicMin_system(result, value);
    } else if constexpr (Scope == AtomicDeviceScope) {
      return atomicMin(result, value);
    } else {
      return atomicMin_block(result, value);
    }
#else
    return atomicMin(result, value);
#endif
  }
  MGARDX_EXEC static T Max(T *result, T value) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 600
    if constexpr (Scope == AtomicSystemScope) {
      return atomicMax_system(result, value);
    } else if constexpr (Scope == AtomicDeviceScope) {
      return atomicMax(result, value);
    } else {
      return atomicMax_block(result, value);
    }
#else
    return atomicMax(result, value);
#endif
  }
  MGARDX_EXEC static T Add(T *result, T value) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 600
    if constexpr (Scope == AtomicSystemScope) {
      return atomicAdd_system(result, value);
    } else if constexpr (Scope == AtomicDeviceScope) {
      return atomicAdd(result, value);
    } else {
      return atomicAdd_block(result, value);
    }
#else
    return atomicAdd(result, value);
#endif
  }
};

template <> struct Math<CUDA> {
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

template <typename Task>
MGARDX_KERL void CudaKernel(Task task, THREAD_IDX blockz_offset,
                            THREAD_IDX blocky_offset,
                            THREAD_IDX blockx_offset) {
  Byte *shared_memory = SharedMemory<Byte>();
  task.GetFunctor().Init(task.GetGridDimZ(), task.GetGridDimY(),
                         task.GetGridDimX(), task.GetBlockDimZ(),
                         task.GetBlockDimY(), task.GetBlockDimX(),
                         blockz_offset + blockIdx.z, blocky_offset + blockIdx.y,
                         blockx_offset + blockIdx.x, threadIdx.z, threadIdx.y,
                         threadIdx.x, shared_memory);

  task.GetFunctor().Operation1();
  SyncBlock<CUDA>::Sync();
  task.GetFunctor().Operation2();
  SyncBlock<CUDA>::Sync();
  task.GetFunctor().Operation3();
  SyncBlock<CUDA>::Sync();
  task.GetFunctor().Operation4();
  SyncBlock<CUDA>::Sync();
  task.GetFunctor().Operation5();
  SyncBlock<CUDA>::Sync();
  task.GetFunctor().Operation6();
  SyncBlock<CUDA>::Sync();
  task.GetFunctor().Operation7();
  SyncBlock<CUDA>::Sync();
  task.GetFunctor().Operation8();
  SyncBlock<CUDA>::Sync();
  task.GetFunctor().Operation9();
  SyncBlock<CUDA>::Sync();
  task.GetFunctor().Operation10();
  // SyncBlock<CUDA>::Sync();
  // task.GetFunctor().Operation11();
  // SyncBlock<CUDA>::Sync();
  // task.GetFunctor().Operation12();
  // SyncBlock<CUDA>::Sync();
  // task.GetFunctor().Operation13();
  // SyncBlock<CUDA>::Sync();
  // task.GetFunctor().Operation14();
  // SyncBlock<CUDA>::Sync();
  // task.GetFunctor().Operation15();
  // SyncBlock<CUDA>::Sync();
  // task.GetFunctor().Operation16();
  // SyncBlock<CUDA>::Sync();
  // task.GetFunctor().Operation17();
  // SyncBlock<CUDA>::Sync();
  // task.GetFunctor().Operation18();
  // SyncBlock<CUDA>::Sync();
  // task.GetFunctor().Operation19();
  // SyncBlock<CUDA>::Sync();
  // task.GetFunctor().Operation20();
}

template <typename Task>
MGARDX_KERL void CudaIterKernel(Task task, THREAD_IDX blockz_offset,
                                THREAD_IDX blocky_offset,
                                THREAD_IDX blockx_offset) {
  Byte *shared_memory = SharedMemory<Byte>();

  task.GetFunctor().Init(task.GetGridDimZ(), task.GetGridDimY(),
                         task.GetGridDimX(), task.GetBlockDimZ(),
                         task.GetBlockDimY(), task.GetBlockDimX(),
                         blockz_offset + blockIdx.z, blocky_offset + blockIdx.y,
                         blockx_offset + blockIdx.x, threadIdx.z, threadIdx.y,
                         threadIdx.x, shared_memory);

  task.GetFunctor().Operation1();
  SyncBlock<CUDA>::Sync();

  task.GetFunctor().Operation2();
  SyncBlock<CUDA>::Sync();

  while (task.GetFunctor().LoopCondition1()) {
    task.GetFunctor().Operation3();
    SyncBlock<CUDA>::Sync();
    task.GetFunctor().Operation4();
    SyncBlock<CUDA>::Sync();
    task.GetFunctor().Operation5();
    SyncBlock<CUDA>::Sync();
    task.GetFunctor().Operation6();
    SyncBlock<CUDA>::Sync();
  }

  task.GetFunctor().Operation7();
  SyncBlock<CUDA>::Sync();
  task.GetFunctor().Operation8();
  SyncBlock<CUDA>::Sync();
  task.GetFunctor().Operation9();
  SyncBlock<CUDA>::Sync();
  task.GetFunctor().Operation10();
  SyncBlock<CUDA>::Sync();

  while (task.GetFunctor().LoopCondition2()) {
    task.GetFunctor().Operation11();
    SyncBlock<CUDA>::Sync();
    task.GetFunctor().Operation12();
    SyncBlock<CUDA>::Sync();
    task.GetFunctor().Operation13();
    SyncBlock<CUDA>::Sync();
    task.GetFunctor().Operation14();
    SyncBlock<CUDA>::Sync();
  }

  task.GetFunctor().Operation15();
  SyncBlock<CUDA>::Sync();
  task.GetFunctor().Operation16();
  SyncBlock<CUDA>::Sync();
  task.GetFunctor().Operation17();
  SyncBlock<CUDA>::Sync();
}

template <typename Task>
MGARDX_KERL void CudaHuffmanCLCustomizedKernel(Task task) {
  Byte *shared_memory = SharedMemory<Byte>();

  task.GetFunctor().Init(gridDim.z, gridDim.y, gridDim.x, blockDim.z,
                         blockDim.y, blockDim.x, blockIdx.z, blockIdx.y,
                         blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x,
                         shared_memory);

  task.GetFunctor().Operation1();
  SyncGrid<CUDA>::Sync();
  while (task.GetFunctor().LoopCondition1()) {
    task.GetFunctor().Operation2();
    SyncGrid<CUDA>::Sync();
    task.GetFunctor().Operation3();
    SyncGrid<CUDA>::Sync();
    task.GetFunctor().Operation4();
    SyncGrid<CUDA>::Sync();
    task.GetFunctor().Operation5();
    SyncGrid<CUDA>::Sync();
    if (task.GetFunctor().BranchCondition1()) {
      task.GetFunctor().Operation6();
      SyncBlock<CUDA>::Sync();
      while (task.GetFunctor().LoopCondition2()) {
        task.GetFunctor().Operation7();
        SyncBlock<CUDA>::Sync();
        task.GetFunctor().Operation8();
        SyncBlock<CUDA>::Sync();
        task.GetFunctor().Operation9();
        SyncBlock<CUDA>::Sync();
      }
      task.GetFunctor().Operation10();
      SyncGrid<CUDA>::Sync();
      task.GetFunctor().Operation11();
      SyncGrid<CUDA>::Sync();
    }
    task.GetFunctor().Operation12();
    SyncGrid<CUDA>::Sync();
    task.GetFunctor().Operation13();
    SyncGrid<CUDA>::Sync();
    task.GetFunctor().Operation14();
    SyncGrid<CUDA>::Sync();
    task.GetFunctor().Operation15();
    SyncGrid<CUDA>::Sync();
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
SINGLE_KERNEL(Operation15);

#undef SINGLE_KERNEL

template <typename Task> MGARDX_KERL void CudaParallelMergeKernel(Task task) {
  Byte *shared_memory = SharedMemory<Byte>();

  task.GetFunctor().Init(gridDim.z, gridDim.y, gridDim.x, blockDim.z,
                         blockDim.y, blockDim.x, blockIdx.z, blockIdx.y,
                         blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x,
                         shared_memory);

  task.GetFunctor().Operation6();
  SyncBlock<CUDA>::Sync();
  while (task.GetFunctor().LoopCondition2()) {
    task.GetFunctor().Operation7();
    SyncBlock<CUDA>::Sync();
    task.GetFunctor().Operation8();
    SyncBlock<CUDA>::Sync();
    task.GetFunctor().Operation9();
    SyncBlock<CUDA>::Sync();
  }
  task.GetFunctor().Operation10();
}

template <typename Task>
MGARDX_KERL void CudaHuffmanCWCustomizedKernel(Task task) {
  Byte *shared_memory = SharedMemory<Byte>();

  task.GetFunctor().Init(gridDim.z, gridDim.y, gridDim.x, blockDim.z,
                         blockDim.y, blockDim.x, blockIdx.z, blockIdx.y,
                         blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x,
                         shared_memory);

  task.GetFunctor().Operation1();
  SyncGrid<CUDA>::Sync();
  task.GetFunctor().Operation2();
  SyncGrid<CUDA>::Sync();
  task.GetFunctor().Operation3();
  SyncGrid<CUDA>::Sync();

  while (task.GetFunctor().LoopCondition1()) {
    task.GetFunctor().Operation4();
    SyncGrid<CUDA>::Sync();
    task.GetFunctor().Operation5();
    SyncGrid<CUDA>::Sync();
    task.GetFunctor().Operation6();
    SyncGrid<CUDA>::Sync();
    task.GetFunctor().Operation7();
    SyncGrid<CUDA>::Sync();
    task.GetFunctor().Operation8();
    SyncGrid<CUDA>::Sync();
  }
  task.GetFunctor().Operation9();
  SyncGrid<CUDA>::Sync();
  task.GetFunctor().Operation10();
  SyncGrid<CUDA>::Sync();
}

template <> class DeviceSpecification<CUDA> {
public:
  MGARDX_CONT
  DeviceSpecification() {
    cudaGetDeviceCount(&NumDevices);
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
      gpuErrchk(cudaSetDevice(d));
      int maxbytes;
      int maxbytesOptIn;
      cudaDeviceGetAttribute(&maxbytes, cudaDevAttrMaxSharedMemoryPerBlock, d);
      cudaDeviceGetAttribute(&maxbytesOptIn,
                             cudaDevAttrMaxSharedMemoryPerBlockOptin, d);
      MaxSharedMemorySize[d] = std::max(maxbytes, maxbytesOptIn);
      cudaDeviceGetAttribute(&WarpSize[d], cudaDevAttrWarpSize, d);
      cudaDeviceGetAttribute(&NumSMs[d], cudaDevAttrMultiProcessorCount, d);
      cudaDeviceGetAttribute(&MaxNumThreadsPerSM[d],
                             cudaDevAttrMaxThreadsPerMultiProcessor, d);
      cudaDeviceGetAttribute(&MaxNumThreadsPerTB[d],
                             cudaDevAttrMaxThreadsPerBlock, d);
      SupportCooperativeGroups[d] = true;
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, d);
      DeviceNames[d] = std::string(prop.name);
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
    gpuErrchk(cudaSetDevice(dev_id));
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    AvailableMemory[dev_id] = total;

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

template <> class DeviceQueues<CUDA> {
public:
  MGARDX_CONT
  void Initialize() {
    if (!initialized) {
      log::dbg("Calling DeviceQueues<CUDA>::Initialize");
      cudaGetDeviceCount(&NumDevices);
      streams = new cudaStream_t *[NumDevices];
      for (int d = 0; d < NumDevices; d++) {
        gpuErrchk(cudaSetDevice(d));
        streams[d] = new cudaStream_t[MGARDX_NUM_QUEUES];
        for (SIZE i = 0; i < MGARDX_NUM_QUEUES; i++) {
          gpuErrchk(cudaStreamCreate(&streams[d][i]));
        }
      }
      initialized = true;
    }
  }

  MGARDX_CONT
  void Destroy() {
    if (initialized) {
      log::dbg("Calling DeviceQueues<CUDA>::Destroy");
      for (int d = 0; d < NumDevices; d++) {
        gpuErrchk(cudaSetDevice(d));
        for (int i = 0; i < MGARDX_NUM_QUEUES; i++) {
          gpuErrchk(cudaStreamDestroy(streams[d][i]));
        }
        delete[] streams[d];
      }
      delete[] streams;
      streams = nullptr;
      initialized = false;
    }
  }

  MGARDX_CONT
  DeviceQueues() { Initialize(); }

  MGARDX_CONT cudaStream_t GetQueue(int dev_id, SIZE queue_id) {
    Initialize();
    return streams[dev_id][queue_id];
  }

  MGARDX_CONT void SyncQueue(int dev_id, SIZE queue_id) {
    Initialize();
    cudaStreamSynchronize(streams[dev_id][queue_id]);
  }

  MGARDX_CONT void SyncAllQueues(int dev_id) {
    Initialize();
    for (SIZE i = 0; i < MGARDX_NUM_QUEUES; i++) {
      gpuErrchk(cudaStreamSynchronize(streams[dev_id][i]));
    }
  }

  MGARDX_CONT
  ~DeviceQueues() {}

  int NumDevices;
  cudaStream_t **streams = nullptr;
  bool initialized = false;
};

extern int cuda_dev_id;
#pragma omp threadprivate(cuda_dev_id)

template <> class DeviceRuntime<CUDA> {
public:
  MGARDX_CONT
  DeviceRuntime() {}

  MGARDX_CONT static void Initialize() { queues.Initialize(); }

  MGARDX_CONT static void Finalize() { queues.Destroy(); }

  MGARDX_CONT static int GetDeviceCount() { return DeviceSpecs.NumDevices; }

  MGARDX_CONT static void SelectDevice(SIZE dev_id) {
    gpuErrchk(cudaSetDevice(dev_id));
    cuda_dev_id = dev_id;
  }

  MGARDX_CONT static int GetDevice() {
    gpuErrchk(cudaGetDevice(&cuda_dev_id));
    return cuda_dev_id;
  }

  MGARDX_CONT static cudaStream_t GetQueue(SIZE queue_id) {
    return queues.GetQueue(cuda_dev_id, queue_id);
  }

  MGARDX_CONT static void SyncQueue(SIZE queue_id) {
    queues.SyncQueue(cuda_dev_id, queue_id);
  }

  MGARDX_CONT static void SyncAllQueues() { queues.SyncAllQueues(cuda_dev_id); }

  MGARDX_CONT static void SyncDevice() { gpuErrchk(cudaDeviceSynchronize()); }

  MGARDX_CONT static std::string GetDeviceName() {
    return DeviceSpecs.GetDeviceName(cuda_dev_id);
  }

  MGARDX_CONT static int GetMaxSharedMemorySize() {
    return DeviceSpecs.GetMaxSharedMemorySize(cuda_dev_id);
  }

  MGARDX_CONT static int GetWarpSize() {
    return DeviceSpecs.GetWarpSize(cuda_dev_id);
  }

  MGARDX_CONT static int GetNumSMs() {
    return DeviceSpecs.GetNumSMs(cuda_dev_id);
  }

  MGARDX_CONT static int GetArchitectureGeneration() {
    return DeviceSpecs.GetArchitectureGeneration(cuda_dev_id);
  }

  MGARDX_CONT static int GetMaxNumThreadsPerSM() {
    return DeviceSpecs.GetMaxNumThreadsPerSM(cuda_dev_id);
  }

  MGARDX_CONT static int GetMaxNumThreadsPerTB() {
    return DeviceSpecs.GetMaxNumThreadsPerTB(cuda_dev_id);
  }

  MGARDX_CONT static size_t GetAvailableMemory() {
    return DeviceSpecs.GetAvailableMemory(cuda_dev_id);
  }

  MGARDX_CONT static bool SupportCG() {
    return DeviceSpecs.SupportCG(cuda_dev_id);
  }

  template <typename FunctorType>
  MGARDX_CONT static int
  GetOccupancyMaxActiveBlocksPerSM(FunctorType functor, int blockSize,
                                   size_t dynamicSMemSize) {
    int numBlocks = 0;

    if constexpr (std::is_base_of<Functor<CUDA>, FunctorType>::value) {
      gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &numBlocks, CudaKernel<Task<FunctorType>>, blockSize,
          dynamicSMemSize));
    } else if constexpr (std::is_base_of<IterFunctor<CUDA>,
                                         FunctorType>::value) {
      gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &numBlocks, CudaIterKernel<Task<FunctorType>>, blockSize,
          dynamicSMemSize));
    } else if constexpr (std::is_base_of<HuffmanCLCustomizedFunctor<CUDA>,
                                         FunctorType>::value) {
      gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &numBlocks, CudaHuffmanCLCustomizedKernel<Task<FunctorType>>,
          blockSize, dynamicSMemSize));
    } else if constexpr (std::is_base_of<HuffmanCWCustomizedFunctor<CUDA>,
                                         FunctorType>::value) {
      gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &numBlocks, CudaHuffmanCWCustomizedKernel<Task<FunctorType>>,
          blockSize, dynamicSMemSize));
    } else {
      log::err("GetOccupancyMaxActiveBlocksPerSM Error!");
    }
    return numBlocks;
  }

  template <typename FunctorType>
  MGARDX_CONT static void SetMaxDynamicSharedMemorySize(FunctorType functor,
                                                        int maxbytes) {

    if constexpr (std::is_base_of<Functor<CUDA>, FunctorType>::value) {
      gpuErrchk(cudaFuncSetAttribute(
          CudaKernel<Task<FunctorType>>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
    } else if constexpr (std::is_base_of<IterFunctor<CUDA>,
                                         FunctorType>::value) {
      gpuErrchk(cudaFuncSetAttribute(
          CudaIterKernel<Task<FunctorType>>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
    } else if constexpr (std::is_base_of<HuffmanCLCustomizedFunctor<CUDA>,
                                         FunctorType>::value) {
      gpuErrchk(cudaFuncSetAttribute(
          CudaHuffmanCLCustomizedKernel<Task<FunctorType>>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
    } else if constexpr (std::is_base_of<HuffmanCWCustomizedFunctor<CUDA>,
                                         FunctorType>::value) {
      gpuErrchk(cudaFuncSetAttribute(
          CudaHuffmanCWCustomizedKernel<Task<FunctorType>>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
    } else {
      log::err("SetPreferredSharedMemoryCarveout Error!");
    }
  }

  MGARDX_CONT
  ~DeviceRuntime() {}

  static DeviceQueues<CUDA> queues;
  static bool SyncAllKernelsAndCheckErrors;
  static DeviceSpecification<CUDA> DeviceSpecs;
  static bool TimingAllKernels;
  static bool PrintKernelConfig;
};

template <> class MemoryManager<CUDA> {
public:
  MGARDX_CONT
  MemoryManager(){};

  template <typename T>
  MGARDX_CONT static void Malloc1D(T *&ptr, SIZE n,
                                   int queue_idx = MGARDX_SYNCHRONIZED_QUEUE) {
    log::dbg("Calling MemoryManager<CUDA>::Malloc1D");
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<CUDA>::SyncQueue(queue_idx);
    }
    gpuErrchk(cudaMalloc(&ptr, n * sizeof(T)));
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<CUDA>::SyncQueue(queue_idx);
    }
    if (DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors) {
      DeviceRuntime<CUDA>::SyncDevice();
    }
  }

  template <typename T>
  MGARDX_CONT static void MallocND(T *&ptr, SIZE n1, SIZE n2, SIZE &ld,
                                   int queue_idx = MGARDX_SYNCHRONIZED_QUEUE) {
    log::dbg("Calling MemoryManager<CUDA>::MallocND");
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<CUDA>::SyncQueue(queue_idx);
    }
    if (ReduceMemoryFootprint) {
      gpuErrchk(cudaMalloc(&ptr, n1 * n2 * sizeof(T)));
      ld = n1;
    } else {
      size_t pitch = 0;
      gpuErrchk(cudaMallocPitch(&ptr, &pitch, n1 * sizeof(T), (size_t)n2));
      ld = pitch / sizeof(T);
    }
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<CUDA>::SyncQueue(queue_idx);
    }
    if (DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors) {
      DeviceRuntime<CUDA>::SyncDevice();
    }
  }

  template <typename T>
  MGARDX_CONT static void
  MallocManaged1D(T *&ptr, SIZE n, int queue_idx = MGARDX_SYNCHRONIZED_QUEUE) {
    log::dbg("Calling MemoryManager<CUDA>::MallocManaged1D");
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<CUDA>::SyncQueue(queue_idx);
    }
    gpuErrchk(cudaMallocManaged(&ptr, n * sizeof(T)));
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<CUDA>::SyncQueue(queue_idx);
    }
    if (DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors) {
      DeviceRuntime<CUDA>::SyncDevice();
    }
  }

  template <typename T>
  MGARDX_CONT static void Free(T *ptr,
                               int queue_idx = MGARDX_SYNCHRONIZED_QUEUE) {
    log::dbg("Calling MemoryManager<CUDA>::Free");
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<CUDA>::SyncQueue(queue_idx);
    }
    if (ptr == nullptr)
      return;
    gpuErrchk(cudaFree(ptr));
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<CUDA>::SyncQueue(queue_idx);
    }
    if (DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors) {
      DeviceRuntime<CUDA>::SyncDevice();
    }
  }

  template <typename T>
  MGARDX_CONT static void Copy1D(T *dst_ptr, const T *src_ptr, SIZE n,
                                 int queue_idx = MGARDX_SYNCHRONIZED_QUEUE) {
    log::dbg("Calling MemoryManager<CUDA>::Copy1D");
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<CUDA>::SyncQueue(queue_idx);
    }
    cudaStream_t stream = DeviceRuntime<CUDA>::GetQueue(queue_idx);
    gpuErrchk(cudaMemcpyAsync(dst_ptr, src_ptr, n * sizeof(T),
                              cudaMemcpyDefault, stream));
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<CUDA>::SyncQueue(queue_idx);
    }
    if (DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors) {
      DeviceRuntime<CUDA>::SyncDevice();
    }
  }

  template <typename T>
  MGARDX_CONT static void CopyND(T *dst_ptr, SIZE dst_ld, const T *src_ptr,
                                 SIZE src_ld, SIZE n1, SIZE n2,
                                 int queue_idx = MGARDX_SYNCHRONIZED_QUEUE) {
    log::dbg("Calling MemoryManager<CUDA>::CopyND");
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<CUDA>::SyncQueue(queue_idx);
    }
    cudaStream_t stream = DeviceRuntime<CUDA>::GetQueue(queue_idx);
    gpuErrchk(cudaMemcpy2DAsync(dst_ptr, dst_ld * sizeof(T), src_ptr,
                                src_ld * sizeof(T), n1 * sizeof(T), n2,
                                cudaMemcpyDefault, stream));
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<CUDA>::SyncQueue(queue_idx);
    }
    if (DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors) {
      DeviceRuntime<CUDA>::SyncDevice();
    }
  }

  template <typename T>
  MGARDX_CONT static void
  MallocHost(T *&ptr, SIZE n, int queue_idx = MGARDX_SYNCHRONIZED_QUEUE) {
    log::dbg("Calling MemoryManager<CUDA>::MallocHost");
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<CUDA>::SyncQueue(queue_idx);
    }
    gpuErrchk(cudaMallocHost(&ptr, n * sizeof(T)));
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<CUDA>::SyncQueue(queue_idx);
    }
    if (DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors) {
      DeviceRuntime<CUDA>::SyncDevice();
    }
  }

  template <typename T>
  MGARDX_CONT static void FreeHost(T *ptr,
                                   int queue_idx = MGARDX_SYNCHRONIZED_QUEUE) {
    log::dbg("Calling MemoryManager<CUDA>::FreeHost");
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<CUDA>::SyncQueue(queue_idx);
    }
    if (ptr == nullptr)
      return;
    gpuErrchk(cudaFreeHost(ptr));
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<CUDA>::SyncQueue(queue_idx);
    }
    if (DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors) {
      DeviceRuntime<CUDA>::SyncDevice();
    }
  }

  template <typename T>
  MGARDX_CONT static void Memset1D(T *ptr, SIZE n, int value,
                                   int queue_idx = MGARDX_SYNCHRONIZED_QUEUE) {
    log::dbg("Calling MemoryManager<CUDA>::Memset1D");
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<CUDA>::SyncQueue(queue_idx);
    }
    cudaStream_t stream = DeviceRuntime<CUDA>::GetQueue(queue_idx);
    gpuErrchk(cudaMemsetAsync(ptr, value, n * sizeof(T), stream));
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<CUDA>::SyncQueue(queue_idx);
    }
    if (DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors) {
      DeviceRuntime<CUDA>::SyncDevice();
    }
  }

  template <typename T>
  MGARDX_CONT static void MemsetND(T *ptr, SIZE ld, SIZE n1, SIZE n2, int value,
                                   int queue_idx = MGARDX_SYNCHRONIZED_QUEUE) {
    log::dbg("Calling MemoryManager<CUDA>::MemsetND");
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<CUDA>::SyncQueue(queue_idx);
    }
    cudaStream_t stream = DeviceRuntime<CUDA>::GetQueue(queue_idx);
    gpuErrchk(cudaMemset2DAsync(ptr, ld * sizeof(T), value, n1 * sizeof(T), n2,
                                stream));
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<CUDA>::SyncQueue(queue_idx);
    }
    if (DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors) {
      DeviceRuntime<CUDA>::SyncDevice();
    }
  }

  template <typename T> MGARDX_CONT static bool IsDevicePointer(T *ptr) {
    log::dbg("Calling MemoryManager<CUDA>::IsDevicePointer");
    cudaPointerAttributes attr;
    cudaPointerGetAttributes(&attr, ptr);
    return attr.type == cudaMemoryTypeDevice;
  }

  template <typename T> MGARDX_CONT static int GetPointerDevice(T *ptr) {
    log::dbg("Calling MemoryManager<CUDA>::GetPointerDevice");
    cudaPointerAttributes attr;
    cudaPointerGetAttributes(&attr, ptr);
    return attr.device;
  }

  template <typename T> MGARDX_CONT static bool CheckHostRegister(T *ptr) {
    log::dbg("Calling MemoryManager<CUDA>::CheckHostRegister");
    unsigned int flags;
    cudaHostGetFlags(&flags, (void *)ptr);
    return cudaGetLastError() == cudaSuccess;
  }

  template <typename T> MGARDX_CONT static void HostRegister(T *ptr, SIZE n) {
    log::dbg("Calling MemoryManager<CUDA>::HostRegister");
    if (!CheckHostRegister(ptr)) {
      gpuErrchk(cudaHostRegister((void *)ptr, n * sizeof(T),
                                 cudaHostRegisterPortable));
    }
  }

  template <typename T> MGARDX_CONT static void HostUnregister(T *ptr) {
    log::dbg("Calling MemoryManager<CUDA>::HostUnregister");
    if (CheckHostRegister(ptr)) {
      gpuErrchk(cudaHostUnregister((void *)ptr));
    }
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
struct BlockReduce<T, nblockx, nblocky, nblockz, CUDA> {
  typedef cub::BlockReduce<T, nblockx, cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                           nblocky, nblockz>
      BlockReduceType;
  using TempStorageType = typename BlockReduceType::TempStorage;

  MGARDX_EXEC
  static void Sum(T intput, T &output) {
    __shared__ TempStorageType temp_storage;
    BlockReduceType blockReduce(temp_storage);
    output = blockReduce.Sum(intput);
  }

  MGARDX_EXEC
  static void Max(T intput, T &output) {
    __shared__ TempStorageType temp_storage;
    BlockReduceType blockReduce(temp_storage);
    output = blockReduce.Reduce(intput, cub::Max());
  }
};

template <typename T_org, typename T_trans, SIZE nblockx, SIZE nblocky,
          SIZE nblockz, OPTION ALIGN, OPTION METHOD>
struct BlockBitTranspose<T_org, T_trans, nblockx, nblocky, nblockz, ALIGN,
                         METHOD, CUDA> {

  typedef cub::WarpReduce<T_trans> WarpReduceType;
  using WarpReduceStorageType = typename WarpReduceType::TempStorage;

  typedef cub::BlockReduce<T_trans, nblockx, cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                           nblocky, nblockz>
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
          Atomic<T_trans, AtomicSharedMemory, AtomicBlockScope, CUDA>::Add(
              sum, shifted_bit);
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
    // if (threadIdx.y == 0 && threadIdx.x == 0) printf("time0: %llu\n", start);
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
        // if (threadIdx.y == 0 && threadIdx.x == 0) printf("time1: %llu\n",
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
    // if (threadIdx.y == 0 && threadIdx.x == 0) printf("time2: %llu\n", start);
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
    // if (threadIdx.y == 0 && threadIdx.x == 0) printf("time0: %llu\n", start);
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
        // if (threadIdx.y == 0 && threadIdx.x == 0) printf("time1: %llu\n",
        // start);
        // __syncthreads(); start = clock64();
        sum += ((T_trans)__ballot_sync(0xffffffff, bit)) << shift;
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
    // if (threadIdx.y == 0 && threadIdx.x == 0) printf("time2: %llu\n", start);
    // __syncthreads(); start = clock64();

    // __syncthreads(); long long start = clock64();

    // SIZE i = threadIdx.x;
    // SIZE B_idx = threadIdx.y;
    // SIZE b_idx = threadIdx.x;

    // __syncthreads(); start = clock64() - start;
    // if (threadIdx.y == 0 && threadIdx.x == 0) printf("time0: %llu\n", start);
    // __syncthreads(); start = clock64();

    // int bit = (v[b-1-b_idx] >> (sizeof(T_org)*8 - 1 - B_idx)) & 1u;

    // __syncthreads();
    // start = clock64() - start;
    // if (threadIdx.y == 0 && threadIdx.x == 0) printf("time1: %llu\n", start);
    // __syncthreads(); start = clock64();

    // printf("b_idx[%u]: bit %d\n", b_idx, bit);
    // unsigned int sum = __ballot_sync (0xffffffff, bit);
    // printf("b_idx[%u]: sum %u\n", b_idx, sum);
    // if (b_idx) tv[B_idx] = sum;

    // __syncthreads(); start = clock64() - start;
    // if (threadIdx.y == 0 && threadIdx.x == 0) printf("time2: %llu\n", start);
    // __syncthreads(); start = clock64();
  }

  // MGARDX_EXEC
  // static void TCU(T_org *v, T_trans *tv, SIZE b, SIZE B, SIZE IdX, SIZE IdY)
  // {
  //   __syncthreads();
  //   long long start = clock64();

  //   __shared__ half tile_a[16 * 16];
  //   __shared__ half tile_b[32 * 32];
  //   __shared__ float output[32 * 32];
  //   uint8_t bit;
  //   half shifted_bit;
  //   SIZE i = threadIdx.x;
  //   SIZE B_idx = threadIdx.y;
  //   SIZE b_idx = threadIdx.x;
  //   SIZE warp_idx = threadIdx.y;
  //   SIZE lane_idx = threadIdx.x;

  //   __syncthreads();
  //   start = clock64() - start;
  //   if (IdY == 0 && IdX == 0)
  //     printf("time0: %llu\n", start);

  //   __syncthreads();
  //   start = clock64();
  //   __syncthreads();

  //   if (IdX < B * b) {
  //     uint8_t bit = (v[sizeof(T_trans) * 8 - 1 - b_idx] >>
  //                    (sizeof(T_org) * 8 - 1 - B_idx)) &
  //                   1u;
  //     shifted_bit = bit << (sizeof(T_trans) * 8 - 1 - b_idx) % 8;
  //     tile_b[b_idx * 32 + B_idx] = shifted_bit;
  //     if (i < 8) {
  //       tile_a[i] = 1u;
  //       tile_a[i + 8] = 1u << 8;
  //     }
  //   }
  //   __syncthreads();
  //   start = clock64() - start;
  //   if (IdY == 0 && IdX == 0)
  //     printf("time1: %llu\n", start);

  //   __syncthreads();
  //   start = clock64();
  //   __syncthreads();

  //   if (warp_idx < 4) {
  //     wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
  //     a_frag; wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
  //     wmma::row_major> b_frag; wmma::fragment<wmma::accumulator, 16, 16, 16,
  //     float> c_frag; wmma::load_matrix_sync(a_frag, tile_a, 16);
  //     wmma::load_matrix_sync(
  //         b_frag, tile_b + (warp_idx / 2) * 16 + (warp_idx % 2) * 16, 32);
  //     wmma::fill_fragment(c_frag, 0.0f);
  //     wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  //     wmma::store_matrix_sync(output + (warp_idx / 2) * 16 +
  //                                 (warp_idx % 2) * 16,
  //                             c_frag, 32, wmma::mem_row_major);
  //   }

  //   __syncthreads();
  //   start = clock64() - start;
  //   if (IdY == 0 && IdX == 0)
  //     printf("time2: %llu\n", start);
  // }

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

template <typename T_org, typename T_trans, OPTION ALIGN, OPTION METHOD, SIZE b,
          SIZE B>
struct WarpBitTranspose<T_org, T_trans, ALIGN, METHOD, b, B, CUDA> {

  typedef cub::WarpReduce<T_trans> WarpReduceType;
  using WarpReduceStorageType = typename WarpReduceType::TempStorage;

  MGARDX_EXEC static void Serial_All(T_org *v, SIZE inc_v, T_trans *tv,
                                     SIZE inc_tv, SIZE LaneId) {
    // long long start;
    // if (threadIdx.x == 0 && threadIdx.y == 0) { start = clock64(); }
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
    // if (threadIdx.x == 0 && threadIdx.y == 0) { start = clock64() - start;
    // printf("Serial_All time : %llu\n", start); }
  }

  MGARDX_EXEC static void Parallel_B_Serial_b(T_org *v, SIZE inc_v, T_trans *tv,
                                              SIZE inc_tv, SIZE LaneId) {
    // long long start;
    // if (threadIdx.x == 0 && threadIdx.y == 0) { start = clock64(); }
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
    // if (threadIdx.x == 0 && threadIdx.y == 0) { start = clock64() - start;
    // printf("Parallel_B_Serial_b time : %llu\n", start); }
  }

  MGARDX_EXEC static void Serial_B_Atomic_b(T_org *v, SIZE inc_v, T_trans *tv,
                                            SIZE inc_tv, SIZE LaneId) {
    // long long start;
    // if (threadIdx.x == 0 && threadIdx.y == 0) { start = clock64(); }
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
        Atomic<T_trans, AtomicSharedMemory, AtomicBlockScope, CUDA>::Add(
            sum, shifted_bit);
      }
    }
    // if (threadIdx.x == 0 && threadIdx.y == 0) { start = clock64() - start;
    // printf("Serial_B_Atomic_b time : %llu\n", start); }
  }

  MGARDX_EXEC static void Serial_B_Reduce_b(T_org *v, SIZE inc_v, T_trans *tv,
                                            SIZE inc_tv, SIZE LaneId) {
    // long long start;
    // if (threadIdx.x == 0 && threadIdx.y == 0) { start = clock64(); }
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
    // if (threadIdx.x == 0 && threadIdx.y == 0) { start = clock64() - start;
    // printf("Serial_B_Reduce_b time : %llu\n", start); }
  }

  MGARDX_EXEC static void Serial_B_Ballot_b(T_org *v, SIZE inc_v, T_trans *tv,
                                            SIZE inc_tv, SIZE LaneId) {
    // long long start;
    // if (threadIdx.x == 0 && threadIdx.y == 0) { start = clock64(); }
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
        sum += ((T_trans)__ballot_sync(0xffffffff, bit)) << shift;
        shift += MGARDX_WARP_SIZE;
      }
      if (LaneId == 0) {
        tv[B_idx * inc_tv] = sum;
      }
    }
    // if (threadIdx.x == 0 && threadIdx.y == 0) { start = clock64() - start;
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
          SIZE nblockx, SIZE nblocky, SIZE nblockz, OPTION METHOD,
          OPTION BinaryType>
struct BlockErrorCollect<T, T_fp, T_sfp, T_error, nblockx, nblocky, nblockz,
                         METHOD, BinaryType, CUDA> {

  typedef cub::WarpReduce<T_error> WarpReduceType;
  using WarpReduceStorageType = typename WarpReduceType::TempStorage;

  typedef cub::BlockReduce<T_error, nblockx, cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                           nblocky, nblockz>
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
        T_fp ngb_data = Math<CUDA>::binary2negabinary(fps_data);
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
            diff = (T_error)Math<CUDA>::negabinary2binary(ngb_data & mask) +
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
        T_fp ngb_data = Math<CUDA>::binary2negabinary(fps_data);
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
          diff = (T_error)Math<CUDA>::negabinary2binary(ngb_data & mask) +
                 mantissa;
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
        T_fp ngb_data = Math<CUDA>::binary2negabinary(fps_data);
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
          diff = (T_error)Math<CUDA>::negabinary2binary(ngb_data & mask) +
                 mantissa;
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
        Atomic<T_error, AtomicSharedMemory, AtomicBlockScope, CUDA>::Add(sum,
                                                                         error);
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
        T_fp ngb_data = Math<CUDA>::binary2negabinary(fps_data);
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
          diff = (T_error)Math<CUDA>::negabinary2binary(ngb_data & mask) +
                 mantissa;
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

template <typename T, typename T_fp, typename T_sfp, typename T_error,
          OPTION METHOD, OPTION BinaryType, SIZE num_elems, SIZE num_bitplanes>
struct WarpErrorCollect<T, T_fp, T_sfp, T_error, METHOD, BinaryType, num_elems,
                        num_bitplanes, CUDA> {

  typedef cub::WarpReduce<T_error> WarpReduceType;
  using WarpReduceStorageType = typename WarpReduceType::TempStorage;

  MGARDX_EXEC static void Serial_All(T *v, T_error *errors, SIZE LaneId) {
    if (LaneId == 0) {
      for (SIZE elem_idx = 0; elem_idx < num_elems; elem_idx++) {
        T data = v[elem_idx];
        T_fp fp_data = (T_fp)fabs(data);
        T_sfp fps_data = (T_sfp)data;
        T_fp ngb_data = Math<CUDA>::binary2negabinary(fps_data);
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
            diff = (T_error)Math<CUDA>::negabinary2binary(ngb_data & mask) +
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
        T_fp ngb_data = Math<CUDA>::binary2negabinary(fps_data);
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
          diff = (T_error)Math<CUDA>::negabinary2binary(ngb_data & mask) +
                 mantissa;
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
      T_fp ngb_data = Math<CUDA>::binary2negabinary(fps_data);
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
          diff = (T_error)Math<CUDA>::negabinary2binary(ngb_data & mask) +
                 mantissa;
        }
        T_error *sum = &(errors[num_bitplanes - bitplane_idx]);
        Atomic<T_error, AtomicSharedMemory, AtomicBlockScope, CUDA>::Add(
            sum, diff * diff);
      }
      T_error *sum = &(errors[0]);
      Atomic<T_error, AtomicSharedMemory, AtomicBlockScope, CUDA>::Add(
          sum, data * data);
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
      T_fp ngb_data = Math<CUDA>::binary2negabinary(fps_data);
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
          diff = (T_error)Math<CUDA>::negabinary2binary(ngb_data & mask) +
                 mantissa;
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

template <typename Task> void CudaHuffmanCLCustomizedNoCGKernel(Task task) {
  // std::cout << "calling HuffmanCLCustomizedNoCGKernel\n";
  dim3 threadsPerBlock(task.GetBlockDimX(), task.GetBlockDimY(),
                       task.GetBlockDimZ());
  dim3 blockPerGrid(task.GetGridDimX(), task.GetGridDimY(), task.GetGridDimZ());
  size_t sm_size = task.GetSharedMemorySize();
  cudaStream_t stream = DeviceRuntime<CUDA>::GetQueue(task.GetQueueIdx());

  // std::cout << "calling Single_Operation1_Kernel\n";
  Single_Operation1_Kernel<<<blockPerGrid, threadsPerBlock, sm_size, stream>>>(
      task);
  ErrorSyncCheckTask(cudaDeviceSynchronize(), task);

  // std::cout << "calling LoopCondition1\n";
  while (task.GetFunctor().LoopCondition1()) {
    ErrorSyncCheckTask(cudaDeviceSynchronize(), task);

    // std::cout << "calling Single_Operation2_Kernel\n";
    Single_Operation2_Kernel<<<blockPerGrid, threadsPerBlock, sm_size,
                               stream>>>(task);
    ErrorSyncCheckTask(cudaDeviceSynchronize(), task);

    // std::cout << "calling Single_Operation3_Kernel\n";
    Single_Operation3_Kernel<<<blockPerGrid, threadsPerBlock, sm_size,
                               stream>>>(task);
    ErrorSyncCheckTask(cudaDeviceSynchronize(), task);

    Single_Operation4_Kernel<<<blockPerGrid, threadsPerBlock, sm_size,
                               stream>>>(task);
    ErrorSyncCheckTask(cudaDeviceSynchronize(), task);

    // std::cout << "calling Single_Operation4_Kernel\n";
    Single_Operation5_Kernel<<<blockPerGrid, threadsPerBlock, sm_size,
                               stream>>>(task);
    ErrorSyncCheckTask(cudaDeviceSynchronize(), task);

    // std::cout << "calling BranchCondition1\n";
    if (task.GetFunctor().BranchCondition1()) {
      ErrorSyncCheckTask(cudaDeviceSynchronize(), task);

      // std::cout << "calling CudaParallelMergeKernel\n";
      CudaParallelMergeKernel<<<blockPerGrid, threadsPerBlock, sm_size,
                                stream>>>(task);
      ErrorSyncCheckTask(cudaDeviceSynchronize(), task);

      // std::cout << "calling Single_Operation10_Kernel\n";
      Single_Operation11_Kernel<<<blockPerGrid, threadsPerBlock, sm_size,
                                  stream>>>(task);
      ErrorSyncCheckTask(cudaDeviceSynchronize(), task);
    }

    // std::cout << "calling Single_Operation11_Kernel\n";
    Single_Operation12_Kernel<<<blockPerGrid, threadsPerBlock, sm_size,
                                stream>>>(task);
    ErrorSyncCheckTask(cudaDeviceSynchronize(), task);

    // std::cout << "calling Single_Operation12_Kernel\n";
    Single_Operation13_Kernel<<<blockPerGrid, threadsPerBlock, sm_size,
                                stream>>>(task);
    ErrorSyncCheckTask(cudaDeviceSynchronize(), task);

    // std::cout << "calling Single_Operation13_Kernel\n";
    Single_Operation14_Kernel<<<blockPerGrid, threadsPerBlock, sm_size,
                                stream>>>(task);
    ErrorSyncCheckTask(cudaDeviceSynchronize(), task);

    // std::cout << "calling Single_Operation14_Kernel\n";
    Single_Operation15_Kernel<<<blockPerGrid, threadsPerBlock, sm_size,
                                stream>>>(task);
    ErrorSyncCheckTask(cudaDeviceSynchronize(), task);
  }
}

template <typename Task> void CudaHuffmanCWCustomizedNoCGKernel(Task task) {
  // std::cout << "calling HuffmanCWCustomizedNoCGKernel\n";
  dim3 threadsPerBlock(task.GetBlockDimX(), task.GetBlockDimY(),
                       task.GetBlockDimZ());
  dim3 blockPerGrid(task.GetGridDimX(), task.GetGridDimY(), task.GetGridDimZ());
  size_t sm_size = task.GetSharedMemorySize();
  cudaStream_t stream = DeviceRuntime<CUDA>::GetQueue(task.GetQueueIdx());

  // std::cout << "calling Single_Operation1_Kernel\n";
  Single_Operation1_Kernel<<<blockPerGrid, threadsPerBlock, sm_size, stream>>>(
      task);
  ErrorSyncCheckTask(cudaDeviceSynchronize(), task);
  // std::cout << "calling Single_Operation2_Kernel\n";
  Single_Operation2_Kernel<<<blockPerGrid, threadsPerBlock, sm_size, stream>>>(
      task);
  ErrorSyncCheckTask(cudaDeviceSynchronize(), task);
  // std::cout << "calling Single_Operation3_Kernel\n";
  Single_Operation3_Kernel<<<blockPerGrid, threadsPerBlock, sm_size, stream>>>(
      task);
  ErrorSyncCheckTask(cudaDeviceSynchronize(), task);

  // std::cout << "calling LoopCondition1\n";
  while (task.GetFunctor().LoopCondition1()) {
    ErrorSyncCheckTask(cudaDeviceSynchronize(), task);

    // std::cout << "calling Single_Operation4_Kernel\n";
    Single_Operation4_Kernel<<<blockPerGrid, threadsPerBlock, sm_size,
                               stream>>>(task);
    ErrorSyncCheckTask(cudaDeviceSynchronize(), task);

    // std::cout << "calling Single_Operation5_Kernel\n";
    Single_Operation5_Kernel<<<blockPerGrid, threadsPerBlock, sm_size,
                               stream>>>(task);
    ErrorSyncCheckTask(cudaDeviceSynchronize(), task);

    // std::cout << "calling Single_Operation6_Kernel\n";
    Single_Operation6_Kernel<<<blockPerGrid, threadsPerBlock, sm_size,
                               stream>>>(task);
    ErrorSyncCheckTask(cudaDeviceSynchronize(), task);

    // std::cout << "calling Single_Operation7_Kernel\n";
    Single_Operation7_Kernel<<<blockPerGrid, threadsPerBlock, sm_size,
                               stream>>>(task);
    ErrorSyncCheckTask(cudaDeviceSynchronize(), task);

    // std::cout << "calling Single_Operation8_Kernel\n";
    Single_Operation8_Kernel<<<blockPerGrid, threadsPerBlock, sm_size,
                               stream>>>(task);
    ErrorSyncCheckTask(cudaDeviceSynchronize(), task);
  }

  // std::cout << "calling Single_Operation9_Kernel\n";
  Single_Operation9_Kernel<<<blockPerGrid, threadsPerBlock, sm_size, stream>>>(
      task);
  ErrorSyncCheckTask(cudaDeviceSynchronize(), task);

  // std::cout << "calling Single_Operation10_Kernel\n";
  Single_Operation10_Kernel<<<blockPerGrid, threadsPerBlock, sm_size, stream>>>(
      task);
  ErrorSyncCheckTask(cudaDeviceSynchronize(), task);
}

#define MGARD_CUDA_MAX_GRID_X 2147483647
#define MGARD_CUDA_MAX_GRID_Y 65535
#define MGARD_CUDA_MAX_GRID_Z 65535

template <typename TaskType> class DeviceAdapter<TaskType, CUDA> {
public:
  MGARDX_CONT
  DeviceAdapter(){};

  MGARDX_CONT
  int IsResourceEnough(TaskType &task) {
    if (task.GetBlockDimX() * task.GetBlockDimY() * task.GetBlockDimZ() >
        DeviceRuntime<CUDA>::GetMaxNumThreadsPerTB()) {
      return THREADBLOCK_TOO_LARGE;
    }
    if (task.GetSharedMemorySize() >
        DeviceRuntime<CUDA>::GetMaxSharedMemorySize()) {
      return SHARED_MEMORY_TOO_LARGE;
    }
    return RESOURCE_ENOUGH;
  }

  MGARDX_CONT
  ExecutionReturn Execute(TaskType &task) {

    dim3 threadsPerBlock(task.GetBlockDimX(), task.GetBlockDimY(),
                         task.GetBlockDimZ());
    dim3 blockPerGrid(std::min(task.GetGridDimX(), (IDX)MGARD_CUDA_MAX_GRID_X),
                      std::min(task.GetGridDimY(), (IDX)MGARD_CUDA_MAX_GRID_Y),
                      std::min(task.GetGridDimZ(), (IDX)MGARD_CUDA_MAX_GRID_Z));
    size_t sm_size = task.GetSharedMemorySize();

    cudaStream_t stream = DeviceRuntime<CUDA>::GetQueue(task.GetQueueIdx());

    if (DeviceRuntime<CUDA>::PrintKernelConfig) {
      std::cout << log::log_info << task.GetFunctorName() << ": <"
                << threadsPerBlock.x << ", " << threadsPerBlock.y << ", "
                << threadsPerBlock.z << "> <" << blockPerGrid.x << ", "
                << blockPerGrid.y << ", " << blockPerGrid.z << ">\n";
    }

    ExecutionReturn ret;
    if (IsResourceEnough(task) != RESOURCE_ENOUGH) {
      if (DeviceRuntime<CUDA>::PrintKernelConfig) {
        if (IsResourceEnough(task) == THREADBLOCK_TOO_LARGE) {
          log::info("threadblock too large.");
        }
        if (IsResourceEnough(task) == SHARED_MEMORY_TOO_LARGE) {
          log::info("shared memory too large.");
        }
      }
      ret.success = false;
      ret.execution_time = std::numeric_limits<double>::max();
      return ret;
    }

    Timer timer;
    if (task.GetQueueIdx() == MGARDX_SYNCHRONIZED_QUEUE ||
        DeviceRuntime<CUDA>::TimingAllKernels ||
        AutoTuner<CUDA>::ProfileKernels) {
      DeviceRuntime<CUDA>::SyncDevice();
      timer.start();
    }

    // if constexpr evaluate at compile time otherwise this does not compile
    if constexpr (std::is_base_of<Functor<CUDA>,
                                  typename TaskType::Functor>::value) {
      for (THREAD_IDX blockz_offset = 0; blockz_offset < task.GetGridDimZ();
           blockz_offset += MGARD_CUDA_MAX_GRID_Z) {
        for (THREAD_IDX blocky_offset = 0; blocky_offset < task.GetGridDimY();
             blocky_offset += MGARD_CUDA_MAX_GRID_Y) {
          for (THREAD_IDX blockx_offset = 0; blockx_offset < task.GetGridDimX();
               blockx_offset += MGARD_CUDA_MAX_GRID_X) {
            CudaKernel<<<blockPerGrid, threadsPerBlock, sm_size, stream>>>(
                task, blockz_offset, blocky_offset, blockx_offset);
          }
        }
      }
    } else if constexpr (std::is_base_of<IterFunctor<CUDA>,
                                         typename TaskType::Functor>::value) {
      for (THREAD_IDX blockz_offset = 0; blockz_offset < task.GetGridDimZ();
           blockz_offset += MGARD_CUDA_MAX_GRID_Z) {
        for (THREAD_IDX blocky_offset = 0; blocky_offset < task.GetGridDimY();
             blocky_offset += MGARD_CUDA_MAX_GRID_Y) {
          for (THREAD_IDX blockx_offset = 0; blockx_offset < task.GetGridDimX();
               blockx_offset += MGARD_CUDA_MAX_GRID_X) {
            CudaIterKernel<<<blockPerGrid, threadsPerBlock, sm_size, stream>>>(
                task, blockz_offset, blocky_offset, blockx_offset);
          }
        }
      }
    } else if constexpr (std::is_base_of<HuffmanCLCustomizedFunctor<CUDA>,
                                         typename TaskType::Functor>::value) {
      if (task.GetFunctor().use_CG && DeviceRuntime<CUDA>::SupportCG()) {
        void *Args[] = {(void *)&task};
        cudaLaunchCooperativeKernel(
            (void *)CudaHuffmanCLCustomizedKernel<TaskType>, blockPerGrid,
            threadsPerBlock, Args, sm_size, stream);
      } else {
        CudaHuffmanCLCustomizedNoCGKernel(task);
      }
    } else if constexpr (std::is_base_of<HuffmanCWCustomizedFunctor<CUDA>,
                                         typename TaskType::Functor>::value) {
      if (task.GetFunctor().use_CG && DeviceRuntime<CUDA>::SupportCG()) {
        void *Args[] = {(void *)&task};
        cudaLaunchCooperativeKernel(
            (void *)CudaHuffmanCWCustomizedKernel<TaskType>, blockPerGrid,
            threadsPerBlock, Args, sm_size, stream);
      } else {
        CudaHuffmanCWCustomizedNoCGKernel(task);
      }
    }
    ErrorAsyncCheckTask(cudaGetLastError(), task);
    gpuErrchk(cudaGetLastError());
    if (DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors) {
      ErrorSyncCheckTask(cudaDeviceSynchronize(), task);
    }

    if (task.GetQueueIdx() == MGARDX_SYNCHRONIZED_QUEUE ||
        DeviceRuntime<CUDA>::TimingAllKernels ||
        AutoTuner<CUDA>::ProfileKernels) {
      DeviceRuntime<CUDA>::SyncDevice();
      timer.end();
      if (DeviceRuntime<CUDA>::TimingAllKernels) {
        timer.print(task.GetFunctorName());
      }
      if (AutoTuner<CUDA>::ProfileKernels) {
        ret.success = true;
        ret.execution_time = timer.get();
      }
    }
    return ret;
  }
};

template <> class DeviceLauncher<CUDA> {
public:
  template <typename TaskType>
  MGARDX_CONT int static IsResourceEnough(TaskType &task) {
    if (task.GetBlockDimX() * task.GetBlockDimY() * task.GetBlockDimZ() >
        DeviceRuntime<CUDA>::GetMaxNumThreadsPerTB()) {
      return THREADBLOCK_TOO_LARGE;
    }
    if (task.GetSharedMemorySize() >
        DeviceRuntime<CUDA>::GetMaxSharedMemorySize()) {
      return SHARED_MEMORY_TOO_LARGE;
    }
    return RESOURCE_ENOUGH;
  }

  template <typename TaskType>
  MGARDX_CONT ExecutionReturn static Execute(TaskType &task) {

    dim3 threadsPerBlock(task.GetBlockDimX(), task.GetBlockDimY(),
                         task.GetBlockDimZ());
    dim3 blockPerGrid(std::min(task.GetGridDimX(), (IDX)MGARD_CUDA_MAX_GRID_X),
                      std::min(task.GetGridDimY(), (IDX)MGARD_CUDA_MAX_GRID_Y),
                      std::min(task.GetGridDimZ(), (IDX)MGARD_CUDA_MAX_GRID_Z));
    size_t sm_size = task.GetSharedMemorySize();

    cudaStream_t stream = DeviceRuntime<CUDA>::GetQueue(task.GetQueueIdx());

    if (DeviceRuntime<CUDA>::PrintKernelConfig) {
      std::cout << log::log_info << task.GetFunctorName() << ": <"
                << threadsPerBlock.x << ", " << threadsPerBlock.y << ", "
                << threadsPerBlock.z << "> <" << blockPerGrid.x << ", "
                << blockPerGrid.y << ", " << blockPerGrid.z
                << "> sm: " << sm_size << "\n";
    }

    ExecutionReturn ret;
    if (IsResourceEnough(task) != RESOURCE_ENOUGH) {
      if (DeviceRuntime<CUDA>::PrintKernelConfig) {
        if (IsResourceEnough(task) == THREADBLOCK_TOO_LARGE) {
          log::info("threadblock too large.");
        }
        if (IsResourceEnough(task) == SHARED_MEMORY_TOO_LARGE) {
          log::info("shared memory too large.");
        }
      }
      ret.success = false;
      ret.execution_time = std::numeric_limits<double>::max();
      return ret;
    }

    Timer timer;
    if (task.GetQueueIdx() == MGARDX_SYNCHRONIZED_QUEUE ||
        DeviceRuntime<CUDA>::TimingAllKernels ||
        AutoTuner<CUDA>::ProfileKernels) {
      DeviceRuntime<CUDA>::SyncDevice();
      timer.start();
    }

    // if constexpr evaluate at compile time otherwise this does not compile
    if constexpr (std::is_base_of<Functor<CUDA>,
                                  typename TaskType::Functor>::value) {
      for (THREAD_IDX blockz_offset = 0; blockz_offset < task.GetGridDimZ();
           blockz_offset += MGARD_CUDA_MAX_GRID_Z) {
        for (THREAD_IDX blocky_offset = 0; blocky_offset < task.GetGridDimY();
             blocky_offset += MGARD_CUDA_MAX_GRID_Y) {
          for (THREAD_IDX blockx_offset = 0; blockx_offset < task.GetGridDimX();
               blockx_offset += MGARD_CUDA_MAX_GRID_X) {
            CudaKernel<<<blockPerGrid, threadsPerBlock, sm_size, stream>>>(
                task, blockz_offset, blocky_offset, blockx_offset);
          }
        }
      }
    } else if constexpr (std::is_base_of<IterFunctor<CUDA>,
                                         typename TaskType::Functor>::value) {
      for (THREAD_IDX blockz_offset = 0; blockz_offset < task.GetGridDimZ();
           blockz_offset += MGARD_CUDA_MAX_GRID_Z) {
        for (THREAD_IDX blocky_offset = 0; blocky_offset < task.GetGridDimY();
             blocky_offset += MGARD_CUDA_MAX_GRID_Y) {
          for (THREAD_IDX blockx_offset = 0; blockx_offset < task.GetGridDimX();
               blockx_offset += MGARD_CUDA_MAX_GRID_X) {
            CudaIterKernel<<<blockPerGrid, threadsPerBlock, sm_size, stream>>>(
                task, blockz_offset, blocky_offset, blockx_offset);
          }
        }
      }
    } else if constexpr (std::is_base_of<HuffmanCLCustomizedFunctor<CUDA>,
                                         typename TaskType::Functor>::value) {
      if (task.GetFunctor().use_CG && DeviceRuntime<CUDA>::SupportCG()) {
        void *Args[] = {(void *)&task};
        cudaLaunchCooperativeKernel(
            (void *)CudaHuffmanCLCustomizedKernel<TaskType>, blockPerGrid,
            threadsPerBlock, Args, sm_size, stream);
      } else {
        CudaHuffmanCLCustomizedNoCGKernel(task);
      }
    } else if constexpr (std::is_base_of<HuffmanCWCustomizedFunctor<CUDA>,
                                         typename TaskType::Functor>::value) {
      if (task.GetFunctor().use_CG && DeviceRuntime<CUDA>::SupportCG()) {
        void *Args[] = {(void *)&task};
        cudaLaunchCooperativeKernel(
            (void *)CudaHuffmanCWCustomizedKernel<TaskType>, blockPerGrid,
            threadsPerBlock, Args, sm_size, stream);
      } else {
        CudaHuffmanCWCustomizedNoCGKernel(task);
      }
    }
    ErrorAsyncCheckTask(cudaGetLastError(), task);
    if (DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors) {
      ErrorSyncCheckTask(cudaDeviceSynchronize(), task);
    }

    if (task.GetQueueIdx() == MGARDX_SYNCHRONIZED_QUEUE ||
        DeviceRuntime<CUDA>::TimingAllKernels ||
        AutoTuner<CUDA>::ProfileKernels) {
      DeviceRuntime<CUDA>::SyncDevice();
      timer.end();
      if (DeviceRuntime<CUDA>::TimingAllKernels) {
        timer.print(task.GetFunctorName());
      }
      if (AutoTuner<CUDA>::ProfileKernels) {
        ret.success = true;
        ret.execution_time = timer.get();
      }
    }
    return ret;
  }

  template <typename TaskType>
  MGARDX_CONT static void ConfigTask(TaskType task) {
    typename TaskType::Functor functor;
    int maxbytes = DeviceRuntime<CUDA>::GetMaxSharedMemorySize();
    DeviceRuntime<CUDA>::SetMaxDynamicSharedMemorySize(functor, maxbytes);
  }

  template <typename KernelType>
  MGARDX_CONT static void AutoTune(KernelType kernel, int queue_idx) {
#if MGARD_ENABLE_AUTO_TUNING
    double min_time = std::numeric_limits<double>::max();
    int min_config = 0;
    ExecutionReturn ret;
#define RUN_CONFIG(CONFIG_IDX)                                                 \
  {                                                                            \
    constexpr ExecutionConfig config =                                         \
        GetExecutionConfig<KernelType::NumDim>(CONFIG_IDX);                    \
    auto task =                                                                \
        kernel.template GenTask<config.z, config.y, config.x>(queue_idx);      \
    if constexpr (KernelType::EnableConfig()) {                                \
      ConfigTask(task);                                                        \
    }                                                                          \
    ret = Execute(task);                                                       \
    if (ret.success && min_time > ret.execution_time) {                        \
      min_time = ret.execution_time;                                           \
      min_config = CONFIG_IDX;                                                 \
    }                                                                          \
  }
    RUN_CONFIG(0)
    RUN_CONFIG(1)
    RUN_CONFIG(2)
    RUN_CONFIG(3)
    RUN_CONFIG(4)
    RUN_CONFIG(5)
    RUN_CONFIG(6)
#undef RUN_CONFIG
    if (AutoTuner<CUDA>::WriteToTable) {
      FillAutoTunerTable<KernelType::NumDim, typename KernelType::DataType,
                         CUDA>(std::string(KernelType::Name), min_config);
    }
#else
    log::err("MGARD is not built with auto tuning enabled.");
    exit(-1);
#endif
  }

  template <typename KernelType>
  MGARDX_CONT static void Execute(KernelType kernel, int queue_idx) {
    if constexpr (KernelType::EnableAutoTuning()) {
      constexpr ExecutionConfig config =
          GetExecutionConfig<KernelType::NumDim, typename KernelType::DataType,
                             CUDA>(KernelType::Name);
      auto task =
          kernel.template GenTask<config.z, config.y, config.x>(queue_idx);
      if constexpr (KernelType::EnableConfig()) {
        ConfigTask(task);
      }
      Execute(task);

      if (AutoTuner<CUDA>::ProfileKernels) {
        AutoTune(kernel, queue_idx);
      }
    } else {
      auto task = kernel.GenTask(queue_idx);
      if constexpr (KernelType::EnableConfig()) {
        ConfigTask(task);
      }
      Execute(task);
    }
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

template <> class DeviceCollective<CUDA> {
public:
  MGARDX_CONT
  DeviceCollective(){};

  template <typename T>
  MGARDX_CONT static void Sum(SIZE n, SubArray<1, T, CUDA> v,
                              SubArray<1, T, CUDA> result,
                              Array<1, Byte, CUDA> &workspace,
                              bool workspace_allocated, int queue_idx) {

    Byte *d_temp_storage = workspace_allocated ? workspace.data() : nullptr;
    size_t temp_storage_bytes = workspace_allocated ? workspace.shape(0) : 0;
    cudaStream_t stream = DeviceRuntime<CUDA>::GetQueue(queue_idx);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, v.data(),
                           result.data(), n, stream);
    ErrorAsyncCheck(cudaGetLastError(), "DeviceCollective<CUDA>::Sum");
    if (DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors) {
      ErrorSyncCheck(cudaDeviceSynchronize(), "DeviceCollective<CUDA>::Sum");
    }
    if (!workspace_allocated) {
      workspace.resize({(SIZE)temp_storage_bytes}, queue_idx);
    }
  }

  template <typename T>
  MGARDX_CONT static void AbsMax(SIZE n, SubArray<1, T, CUDA> v,
                                 SubArray<1, T, CUDA> result,
                                 Array<1, Byte, CUDA> &workspace,
                                 bool workspace_allocated, int queue_idx) {

    Byte *d_temp_storage = workspace_allocated ? workspace.data() : nullptr;
    size_t temp_storage_bytes = workspace_allocated ? workspace.shape(0) : 0;
    AbsMaxOp absMaxOp;
    cudaStream_t stream = DeviceRuntime<CUDA>::GetQueue(queue_idx);
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, v.data(),
                              result.data(), n, absMaxOp, static_cast<T>(0),
                              stream);
    ErrorAsyncCheck(cudaGetLastError(), "DeviceCollective<CUDA>::AbsMax");
    if (DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors) {
      ErrorSyncCheck(cudaDeviceSynchronize(), "DeviceCollective<CUDA>::AbsMax");
    }
    if (!workspace_allocated) {
      workspace.resize({(SIZE)temp_storage_bytes}, queue_idx);
    }
  }

  template <typename T>
  MGARDX_CONT static void SquareSum(SIZE n, SubArray<1, T, CUDA> v,
                                    SubArray<1, T, CUDA> result,
                                    Array<1, Byte, CUDA> &workspace,
                                    bool workspace_allocated, int queue_idx) {

    SquareOp squareOp;
    cub::TransformInputIterator<T, SquareOp, T *> transformed_input_iter(
        v.data(), squareOp);
    Byte *d_temp_storage = workspace_allocated ? workspace.data() : nullptr;
    size_t temp_storage_bytes = workspace_allocated ? workspace.shape(0) : 0;
    cudaStream_t stream = DeviceRuntime<CUDA>::GetQueue(queue_idx);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                           transformed_input_iter, result.data(), n, stream);
    ErrorAsyncCheck(cudaGetLastError(), "DeviceCollective<CUDA>::SquareSum");
    if (DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors) {
      ErrorSyncCheck(cudaDeviceSynchronize(),
                     "DeviceCollective<CUDA>::SquareSum");
    }
    if (!workspace_allocated) {
      workspace.resize({(SIZE)temp_storage_bytes}, queue_idx);
    }
  }

  template <typename T>
  MGARDX_CONT static void
  ScanSumInclusive(SIZE n, SubArray<1, T, CUDA> v, SubArray<1, T, CUDA> result,
                   Array<1, Byte, CUDA> &workspace, bool workspace_allocated,
                   int queue_idx) {

    Byte *d_temp_storage = workspace_allocated ? workspace.data() : nullptr;
    size_t temp_storage_bytes = workspace_allocated ? workspace.shape(0) : 0;
    cudaStream_t stream = DeviceRuntime<CUDA>::GetQueue(queue_idx);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, v.data(),
                                  result.data(), n, stream);
    ErrorAsyncCheck(cudaGetLastError(),
                    "DeviceCollective<CUDA>::ScanSumInclusive");
    if (DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors) {
      ErrorSyncCheck(cudaDeviceSynchronize(),
                     "DeviceCollective<CUDA>::ScanSumInclusive");
    }
    if (!workspace_allocated) {
      workspace.resize({(SIZE)temp_storage_bytes}, queue_idx);
    }
  }

  template <typename T>
  MGARDX_CONT static void
  ScanSumExclusive(SIZE n, SubArray<1, T, CUDA> v, SubArray<1, T, CUDA> result,
                   Array<1, Byte, CUDA> &workspace, bool workspace_allocated,
                   int queue_idx) {

    Byte *d_temp_storage = workspace_allocated ? workspace.data() : nullptr;
    size_t temp_storage_bytes = workspace_allocated ? workspace.shape(0) : 0;
    cudaStream_t stream = DeviceRuntime<CUDA>::GetQueue(queue_idx);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, v.data(),
                                  result.data(), n, stream);
    ErrorAsyncCheck(cudaGetLastError(),
                    "DeviceCollective<CUDA>::ScanSumExclusive");
    if (DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors) {
      ErrorSyncCheck(cudaDeviceSynchronize(),
                     "DeviceCollective<CUDA>::ScanSumExclusive");
    }
    if (!workspace_allocated) {
      workspace.resize({(SIZE)temp_storage_bytes}, queue_idx);
    }
  }

  template <typename T>
  MGARDX_CONT static void
  ScanSumExtended(SIZE n, SubArray<1, T, CUDA> v, SubArray<1, T, CUDA> result,
                  Array<1, Byte, CUDA> &workspace, bool workspace_allocated,
                  int queue_idx) {

    Byte *d_temp_storage = workspace_allocated ? workspace.data() : nullptr;
    size_t temp_storage_bytes = workspace_allocated ? workspace.shape(0) : 0;
    cudaStream_t stream = DeviceRuntime<CUDA>::GetQueue(queue_idx);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, v.data(),
                                  result.data() + 1, n, stream);
    ErrorAsyncCheck(cudaGetLastError(),
                    "DeviceCollective<CUDA>::ScanSumExtended");
    if (DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors) {
      ErrorSyncCheck(cudaDeviceSynchronize(),
                     "DeviceCollective<CUDA>::ScanSumExtended");
    }
    if (workspace_allocated) {
      T zero = 0;
      MemoryManager<CUDA>().Copy1D(result.data(), &zero, 1, queue_idx);
    }
    if (!workspace_allocated) {
      workspace.resize({(SIZE)temp_storage_bytes}, queue_idx);
    }
  }

  template <typename KeyT, typename ValueT>
  MGARDX_CONT static void SortByKey(SIZE n, SubArray<1, KeyT, CUDA> in_keys,
                                    SubArray<1, ValueT, CUDA> in_values,
                                    SubArray<1, KeyT, CUDA> out_keys,
                                    SubArray<1, ValueT, CUDA> out_values,
                                    Array<1, Byte, CUDA> &workspace,
                                    bool workspace_allocated, int queue_idx) {
    Byte *d_temp_storage = workspace_allocated ? workspace.data() : nullptr;
    size_t temp_storage_bytes = workspace_allocated ? workspace.shape(0) : 0;
    cudaStream_t stream = DeviceRuntime<CUDA>::GetQueue(queue_idx);
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes, in_keys.data(), out_keys.data(),
        in_values.data(), out_values.data(), n, 0, sizeof(KeyT) * 8, stream);
    ErrorAsyncCheck(cudaGetLastError(), "DeviceCollective<CUDA>::SortByKey");
    if (DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors) {
      ErrorSyncCheck(cudaDeviceSynchronize(),
                     "DeviceCollective<CUDA>::SortByKey");
    }
    if (!workspace_allocated) {
      workspace.resize({(SIZE)temp_storage_bytes}, queue_idx);
    }
  }

  template <typename KeyT, typename ValueT, typename BinaryOpType>
  MGARDX_CONT static void
  ScanOpInclusiveByKey(SubArray<1, SIZE, CUDA> &key,
                       SubArray<1, ValueT, CUDA> &v,
                       SubArray<1, ValueT, CUDA> &result, int queue_idx) {

    thrust::equal_to<KeyT> binary_pred;

    struct ThrustBinaryOp
        : public thrust::binary_function<ValueT, ValueT, ValueT> {
      MGARDX_CONT_EXEC
      ValueT operator()(ValueT x, ValueT y) {
        BinaryOpType op;
        return op(x, y);
      }
    };

    ThrustBinaryOp binary_op;

    thrust::inclusive_scan_by_key(
        thrust::device, thrust::device_ptr<KeyT>(key.data()),
        thrust::device_ptr<KeyT>(key.data() + key.shape(0)),
        thrust::device_ptr<ValueT>(v.data()),
        thrust::device_ptr<ValueT>(result.data()), binary_pred, binary_op);
    ErrorAsyncCheck(cudaGetLastError(),
                    "DeviceCollective<CUDA>::ScanOpInclusiveByKey");
    if (DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors) {
      ErrorSyncCheck(cudaDeviceSynchronize(),
                     "DeviceCollective<CUDA>::ScanOpInclusiveByKey");
    }
  }
};

} // namespace mgard_x

#endif
