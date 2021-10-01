// /*
//  * Copyright 2021, Oak Ridge National Laboratory.
//  * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
//  * Author: Jieyang Chen (chenj3@ornl.gov)
//  * Date: September 27, 2021
//  */

// #ifndef MGARD_CUDA_DEVICE_ADAPTER_CUDA
// #define MGARD_CUDA_DEVICE_ADAPTER_CUDA

// #include "cuda/CommonInternal.h"
// #include "cuda/DeviceAdapters/DeviceAdapter.h"

// template <class T> struct SharedMemory {
//   __device__ inline operator T *() {
//     extern __shared__ int __smem[];
//     return (T *)__smem;
//   }

//   __device__ inline operator const T *() const {
//     extern __shared__ int __smem[];
//     return (T *)__smem;
//   }
// };


// namespace mgard_cuda {


// template <DIM D, typename T, typename FUNCTOR>
// MGARDm_KERL void kernel (Task<D, T, FUNCTOR> task) {
//   T *shared_memory = SharedMemory<T>();
//   task.__operation1(gridDim.z, gridDim.y, gridDim.x, 
//        blockDim.z, blockDim.y, blockDim.x,
//        blockIdx.z,  blockIdx.y,  blockIdx.x, 
//        threadIdx.z, threadIdx.y, threadIdx.x,
//        shared_memory);
//   SyncThreads<CUDA>();
//   task.__operation2(gridDim.z, gridDim.y, gridDim.x, 
//        blockDim.z, blockDim.y, blockDim.x,
//        blockIdx.z,  blockIdx.y,  blockIdx.x, 
//        threadIdx.z, threadIdx.y, threadIdx.x,
//        shared_memory);
//   SyncThreads<CUDA>();
//   task.__operation3(gridDim.z, gridDim.y, gridDim.x, 
//        blockDim.z, blockDim.y, blockDim.x,
//        blockIdx.z,  blockIdx.y,  blockIdx.x, 
//        threadIdx.z, threadIdx.y, threadIdx.x,
//        shared_memory);
//   SyncThreads<CUDA>();
// }


// template <DIM D, typename T, typename TASK>
// public DeviceAdapter<D, T, TASK, CUDA> {
// public:
//   MGARDm_CONT
//   void Execute(TASK task) {
//     dim3 threadsPerBlock(task.get_nblockx(),
//                          task.get_nblocky(),
//                          task.get_nblockz());
//     dim3 blockPerGrid(task.get_ngridx(),
//                       task.get_ngridy(),
//                       task.get_ngridz());
//     size_t sm_size = task.get_shared_memory_size();
//     // printf("exec config (%d %d %d) (%d %d %d)\n", threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z, 
//     //                 blockPerGrid.x, blockPerGrid.y, blockPerGrid.z);
//     cudaStream_t stream = *(cudaStream_t *)(this->handle.get(task.get_queue_idx()));
//     kernel<D, T><<<blockPerGrid, threadsPerBlock, sm_size, stream>>>(task);

//     gpuErrchk(cudaGetLastError());
//     if (this->handle.sync_and_check_all_kernels) {
//       gpuErrchk(cudaDeviceSynchronize());
//     }

//   }

// };
// }


// #endif