/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */


#ifndef MGARD_X_DEVICE_ADAPTER_KOKKOS_H
#define MGARD_X_DEVICE_ADAPTER_KOKKOS_H

#include "DeviceAdapter.h"
#include "Kokkos_Core.hpp"

namespace mgard_x {

#ifdef KOKKOS_ENABLE_CUDA
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
#endif

#ifdef KOKKOS_ENABLE_HIP

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

#endif



template <typename TaskType>
MGARDX_KERL void KOKKOSKernel(TaskType task) {
  typedef Kokkos::TeamPolicy<KOKKOS>::member_type member_type;
  typedef Kokkos::KOKKOS::scratch_memory_space ScratchSpace;
  typedef Kokkos::View<Byte*,ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> ShareMemoryView;
  SIZE league_size = task.GetGridDimZ() * task.GetGridDimY()* task.GetGridDimX();
  SIZE team_size = task.GetBlockDimZ() * task.GetBlockDimY()* task.GetBlockDimX();
  Kokkos::TeamPolicy<KOKKOS> policy (league_size, team_size);

  Kokkos::parallel_for(task.GetFunctorName(),
                       policy.set_scratch_size(0, task.GetSharedMemorySize()),
                       KOKKOS_LAMBDA (member_type team_member) {
    IDX threadx = team_member.team_rank() % task.GetBlockDimX();
    IDX thready = team_member.team_rank() / task.GetBlockDimX();
    IDX threadz = team_member.team_rank() / (task.GetBlockDimX() * task.GetBlockDimY());
    IDX blockx = team_member.league_rank() % task.GetGridDimX();
    IDX blocky = team_member.league_rank() / task.GetGridDimX();
    IDX blockz = team_member.league_rank() / (task.GetGridDimX() * task.GetGridDimY());
    ShareMemoryView shared_memory(team_member.team_scratch(0), team_member.team_size());
    task.GetFunctor().Init(task.GetGridDimZ(), task.GetGridDimY(), task.GetGridDimX(), 
                           task.GetBlockDimZ(), task.GetBlockDimY(), task.GetBlockDimX(),
                           blockz, blocky, blockx, threadz, thready, threadx,
                           shared_memory.data());
    task.Operations1();
    team_member.team_barrier();
    task.Operations2();
    team_member.team_barrier();
    task.Operations3();
    team_member.team_barrier();
    task.Operations4();
    team_member.team_barrier();
    task.Operations5();
    team_member.team_barrier();
    task.Operations6();
    team_member.team_barrier();
    task.Operations7();
    team_member.team_barrier();
    task.Operations8();
    team_member.team_barrier();
    task.Operations9();
    team_member.team_barrier();
    task.Operations10();
    team_member.team_barrier();
    barrier();
  });
}


template <>
class DeviceSpecification<KOKKOS> {
  public:
  MGARDX_CONT
  DeviceSpecification(){

    Kokkos::InitArguments args;
    args.device_id = 0;
    Kokkos::initialize(args);

    NumDevices = 1;
    MaxSharedMemorySize = new int[NumDevices];
    WarpSize = new int[NumDevices];
    NumSMs = new int[NumDevices];
    ArchitectureGeneration = new int[NumDevices];
    MaxNumThreadsPerSM = new int[NumDevices];

    for (int d = 0; d < NumDevices; d++) {
      MaxSharedMemorySize[d] = 1e6;
      WarpSize[d] = 32;
      NumSMs[d] = 80;
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
class DeviceRuntime<KOKKOS> {
  public:
  MGARDX_CONT
  DeviceRuntime(){}

  MGARDX_CONT static void 
  SelectDevice(SIZE dev_id){
    // do not support for now
    curr_dev_id = dev_id;
  }

  MGARDX_CONT static int 
  GetQueue(SIZE queue_id){
    // do not support for now
    return 0;
  }

  MGARDX_CONT static void 
  SyncQueue(SIZE queue_id){
    // do not support for now
    // queues.SyncQueue(curr_dev_id, queue_id);
  }

  MGARDX_CONT static void 
  SyncAllQueues(){
    // do not support for now
    // queues.SyncAllQueues(curr_dev_id);
  }

  MGARDX_CONT static void
  SyncDevice(){
    KOKKOS::impl_static_fence();
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
    int numBlocks = 0;
    // Task<FunctorType> task = Task<FunctorType>(functor, 1, 1, 1, 1, 1, blockSize, dynamicSMemSize, 0);
    // if constexpr (std::is_same<KOKKOS, Kokkos::Cuda>::value) {
    //   if constexpr (std::is_base_of<Functor<CUDA>, FunctorType>::value) {
    //     gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //     &numBlocks, Kernel<Task<FunctorType>>, blockSize, dynamicSMemSize));
    //   } else if constexpr (std::is_base_of<IterFunctor<CUDA>, FunctorType>::value) {
    //     gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //     &numBlocks, IterKernel<Task<FunctorType>>, blockSize, dynamicSMemSize));
    //   } else if constexpr (std::is_base_of<HuffmanCLCustomizedFunctor<CUDA>, FunctorType>::value) {
    //     gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //     &numBlocks, HuffmanCLCustomizedKernel<Task<FunctorType>>, blockSize, dynamicSMemSize));
    //   } else if constexpr (std::is_base_of<HuffmanCWCustomizedFunctor<CUDA>, FunctorType>::value) {
    //     gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //     &numBlocks, HuffmanCWCustomizedKernel<Task<FunctorType>>, blockSize, dynamicSMemSize));
    //   } else {
    //     std::cout << log::log_err << "GetOccupancyMaxActiveBlocksPerSM Error!\n";
    //   }
    // } else if (std::is_same<KOKKOS, Kokkos::Hip>::value) {
    //   if constexpr (std::is_base_of<Functor<HIP>, FunctorType>::value) {
    //     gpuErrchk(hipOccupancyMaxActiveBlocksPerMultiprocessor(
    //     &numBlocks, Kernel<Task<FunctorType>>, blockSize, dynamicSMemSize));
    //   } else if constexpr (std::is_base_of<IterFunctor<HIP>, FunctorType>::value) {
    //     gpuErrchk(hipOccupancyMaxActiveBlocksPerMultiprocessor(
    //     &numBlocks, IterKernel<Task<FunctorType>>, blockSize, dynamicSMemSize));
    //   } else if constexpr (std::is_base_of<HuffmanCLCustomizedFunctor<HIP>, FunctorType>::value) {
    //     gpuErrchk(hipOccupancyMaxActiveBlocksPerMultiprocessor(
    //     &numBlocks, HuffmanCLCustomizedKernel<Task<FunctorType>>, blockSize, dynamicSMemSize));
    //   } else if constexpr (std::is_base_of<HuffmanCWCustomizedFunctor<HIP>, FunctorType>::value) {
    //     gpuErrchk(hipOccupancyMaxActiveBlocksPerMultiprocessor(
    //     &numBlocks, HuffmanCWCustomizedKernel<Task<FunctorType>>, blockSize, dynamicSMemSize));
    //   } else {
    //     std::cout << log::log_err << "GetOccupancyMaxActiveBlocksPerSM Error!\n";
    //   }
    // }

    return numBlocks;
  }

  template <typename FunctorType>
  MGARDX_CONT static void
  SetMaxDynamicSharedMemorySize(FunctorType functor, int maxbytes) {
    // skip for now
  }


  MGARDX_CONT
  ~DeviceRuntime(){
  }

  static int curr_dev_id;
  // static DeviceQueues<KOKKOS> queues;
  static bool SyncAllKernelsAndCheckErrors;
  static DeviceSpecification<KOKKOS> DeviceSpecs;
};


template <>
class MemoryManager<KOKKOS> {
  public:
  MGARDX_CONT
  MemoryManager(){};

  template <typename T>
  MGARDX_CONT static
  void Malloc1D(T *& ptr, SIZE n, int queue_idx) {
    using converted_T = typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    ptr = (T*)Kokkos::kokkos_malloc<KOKKOS::memory_space>(n * sizeof(converted_T));
    if (ptr == NULL) {
      std::cout << log::log_err << "MemoryManager<KOKKOS>::Malloc1D error.\n";
    }
  }

  template <typename T>
  MGARDX_CONT static
  void MallocND(T *& ptr, SIZE n1, SIZE n2, SIZE &ld, int queue_idx) {
    using converted_T = typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    ptr = (T*)Kokkos::kokkos_malloc<KOKKOS::memory_space>(n1 * n2 * sizeof(converted_T));
    ld = n1;
    if (ptr == NULL) {
      std::cout << log::log_err << "MemoryManager<KOKKOS>::MallocND error.\n";
    }
  }

  template <typename T>
  MGARDX_CONT static
  void Free(T * ptr) {
    if (ptr == NULL) return;
    Kokkos::kokkos_free<KOKKOS::memory_space>(ptr);
  }

  template <typename T>
  MGARDX_CONT static
  void Copy1D(T * dst_ptr, const T * src_ptr, SIZE n, int queue_idx) {
    using converted_T = typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    Kokkos::Impl::DeepCopy<KOKKOS::memory_space, KOKKOS::memory_space>(dst_ptr, src_ptr, n * sizeof(converted_T));
  }

  template <typename T>
  MGARDX_CONT static
  void CopyND(T * dst_ptr, SIZE dst_ld, const T * src_ptr, SIZE src_ld, SIZE n1, SIZE n2, int queue_idx) {
    using converted_T = typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    Kokkos::Impl::DeepCopy<KOKKOS::memory_space, KOKKOS::memory_space>(dst_ptr, src_ptr, n1 * n2 * sizeof(converted_T));
  }

  template <typename T>
  MGARDX_CONT static
  void MallocHost(T *& ptr, SIZE n, int queue_idx) {
    using converted_T = typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    ptr = (T*)std::malloc(n * sizeof(converted_T));
    if (ptr == NULL) {
      std::cout << log::log_err << "MemoryManager<KOKKOS>::MallocHost error.\n";
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
    for (SIZE i = 0; i < n; ++i) ptr[i] = value; 
  }

  template <typename T>
  MGARDX_CONT static
  void MemsetND(T * ptr, SIZE ld, SIZE n1, SIZE n2, int value) {
    for (SIZE i = 0; i < n1*n2; ++i) ptr[i] = value; 
  }

  template <typename T>
  MGARDX_CONT static
  bool IsDevicePointer(T * ptr) {
    return true;
  }

  static bool ReduceMemoryFootprint;
};





}

#endif