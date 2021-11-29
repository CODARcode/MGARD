/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */


#ifndef MGARD_X_DEVICE_ADAPTER_KOKKOS_H
#define MGARD_X_DEVICE_ADAPTER_KOKKOS_H

#include "DeviceAdapter.h"
#include"Kokkos_Core.hpp"

namespace mgard_x {

template <>
class DeviceSpecification<KOKKOS> {
  public:
  MGARDX_CONT
  DeviceSpecification(){

    Kokkos::InitArguments args;
    args.device_id = 0;
    Kokkos::initialize(args);

    if (Kokkos::DefaultExecutionSpace == Kokkos::Cuda) {
      printf("Initialize CUDA in Kokkos")
      cudaGetDeviceCount(&NumDevices);
      MaxSharedMemorySize = new int[NumDevices];
      WarpSize = new int[NumDevices];
      NumSMs = new int[NumDevices];
      ArchitectureGeneration = new int[NumDevices];
      MaxNumThreadsPerSM = new int[NumDevices];

      for (int d = 0; d < NumDevices; d++) {
        gpuErrchk(cudaSetDevice(d));
        int maxbytes;
        int maxbytesOptIn;
        cudaDeviceGetAttribute(&maxbytes, cudaDevAttrMaxSharedMemoryPerBlock, d);
        cudaDeviceGetAttribute(&maxbytesOptIn, cudaDevAttrMaxSharedMemoryPerBlockOptin, d);
        MaxSharedMemorySize[d] = std::max(maxbytes, maxbytesOptIn);
        cudaDeviceGetAttribute(&WarpSize[d], cudaDevAttrWarpSize, d);
        cudaDeviceGetAttribute(&NumSMs[d], cudaDevAttrMultiProcessorCount, d);
        cudaDeviceGetAttribute(&MaxNumThreadsPerSM[d], cudaDevAttrMaxThreadsPerMultiProcessor, d);

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, d);
        ArchitectureGeneration[d] = 1; // default optimized for Volta
        if (prop.major == 7 && prop.minor == 0) {
          ArchitectureGeneration[d] = 1;
        } else if (prop.major == 7 && (prop.minor == 2 || prop.minor == 5)) {
          ArchitectureGeneration[d] = 2;
        }
      }
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
    gpuErrchk(cudaSetDevice(dev_id));
    curr_dev_id = dev_id;
  }

  MGARDX_CONT static cudaStream_t 
  GetQueue(SIZE queue_id){
    gpuErrchk(cudaSetDevice(curr_dev_id));
    return queues.GetQueue(curr_dev_id, queue_id);
  }

  MGARDX_CONT static void 
  SyncQueue(SIZE queue_id){
    gpuErrchk(cudaSetDevice(curr_dev_id));
    queues.SyncQueue(curr_dev_id, queue_id);
  }

  MGARDX_CONT static void 
  SyncAllQueues(){
    gpuErrchk(cudaSetDevice(curr_dev_id));
    queues.SyncAllQueues(curr_dev_id);
  }

  MGARDX_CONT static void
  SyncDevice(){
    gpuErrchk(cudaSetDevice(curr_dev_id));
    gpuErrchk(cudaDeviceSynchronize());
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

  // template <typename FunctorType>
  // MGARDX_CONT static int
  // GetOccupancyMaxActiveBlocksPerSM(FunctorType functor, int blockSize, size_t dynamicSMemSize) {
  //   int numBlocks = 0;
  //   Task<FunctorType> task = Task(functor, 1, 1, 1, 
  //               1, 1, blockSize, dynamicSMemSize, 0);

  //   if constexpr (std::is_base_of<Functor<CUDA>, 
  //     FunctorType>::value) {
  //     gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
  //     &numBlocks, 
  //      Kernel<Task<FunctorType>>,
  //      blockSize, 
  //      dynamicSMemSize));
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
  //   return numBlocks;
  // }

  // template <typename FunctorType>
  // MGARDX_CONT static void
  // SetMaxDynamicSharedMemorySize(FunctorType functor, int maxbytes) {
  //   int numBlocks = 0;
  //   Task<FunctorType> task = Task(functor, 1, 1, 1, 1, 1, 1, 0, 0);

  //   if constexpr (std::is_base_of<Functor<CUDA>, 
  //     FunctorType>::value) {
  //     gpuErrchk(cudaFuncSetAttribute( 
  //      Kernel<Task<FunctorType>>,
  //      cudaFuncAttributeMaxDynamicSharedMemorySize,
  //      maxbytes));
  //   } else if constexpr (std::is_base_of<IterFunctor<CUDA>, FunctorType>::value) {
  //     gpuErrchk(cudaFuncSetAttribute( 
  //      IterKernel<Task<FunctorType>>,
  //      cudaFuncAttributeMaxDynamicSharedMemorySize,
  //      maxbytes));
  //   } else if constexpr (std::is_base_of<HuffmanCLCustomizedFunctor<CUDA>, FunctorType>::value) {
  //     gpuErrchk(cudaFuncSetAttribute( 
  //      HuffmanCLCustomizedKernel<Task<FunctorType>>,
  //      cudaFuncAttributeMaxDynamicSharedMemorySize,
  //      maxbytes));
  //   } else if constexpr (std::is_base_of<HuffmanCWCustomizedFunctor<CUDA>, FunctorType>::value) {
  //     gpuErrchk(cudaFuncSetAttribute( 
  //      HuffmanCWCustomizedKernel<Task<FunctorType>>,
  //      cudaFuncAttributeMaxDynamicSharedMemorySize,
  //      maxbytes));
  //   } else {
  //     std::cout << log::log_err << "SetPreferredSharedMemoryCarveout Error!\n";
  //   }
  // }


  MGARDX_CONT
  ~DeviceRuntime(){
  }

  static int curr_dev_id;
  // static DeviceQueues<CUDA> queues;
  static bool SyncAllKernelsAndCheckErrors;
  static DeviceSpecification<KOKKOS> DeviceSpecs;
};

}

#endif