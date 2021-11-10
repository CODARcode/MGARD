/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#ifndef MGARD_CUDA_DEVICE_ADAPTER_CUDA_H
#define MGARD_CUDA_DEVICE_ADAPTER_CUDA_H

#include <cub/cub.cuh>
#include <mma.h>
#include <cooperative_groups.h>
using namespace nvcuda;
namespace cg = cooperative_groups;



template <class T> struct SharedMemory {
  MGARDm_EXEC operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  MGARDm_EXEC operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

static __device__ __inline__ uint32_t __mywarpid(){
  uint32_t warpid = threadIdx.y;
  //asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
  return warpid;
}

static __device__ __inline__ uint32_t __mylaneid(){
  uint32_t laneid = threadIdx.x;
  // asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
  return laneid;
}
 



namespace mgard_cuda {

template<>
struct SyncBlock<CUDA> {
  MGARDm_EXEC static void Sync() {
    __syncthreads();
  }
};

template<>
struct SyncGrid<CUDA> {
  MGARDm_EXEC static void Sync() {
    cg::this_grid().sync();
  }
};

template <> 
struct Atomic<CUDA> {
  template <typename T>
  MGARDm_EXEC static
  void Min(T * result, T value) {
    atomicMin(result, value);
  }
};

template <typename Task>
MGARDm_KERL void kernel() {
}

template <typename Task>
MGARDm_KERL void Kernel(Task task) {
  Byte *shared_memory = SharedMemory<Byte>();
  task.get_functor().Init(gridDim.z, gridDim.y, gridDim.x, 
       blockDim.z, blockDim.y, blockDim.x,
       blockIdx.z,  blockIdx.y,  blockIdx.x, 
       threadIdx.z, threadIdx.y, threadIdx.x,
       shared_memory);

  task.get_functor().Operation1();
  SyncBlock<CUDA>::Sync();
  task.get_functor().Operation2();
  SyncBlock<CUDA>::Sync();
  task.get_functor().Operation3();
  SyncBlock<CUDA>::Sync();
  task.get_functor().Operation4();
  SyncBlock<CUDA>::Sync();
  task.get_functor().Operation5();
}

template <typename Task>
MGARDm_KERL void IterKernel(Task task) {
  Byte *shared_memory = SharedMemory<Byte>();

  task.get_functor().Init(gridDim.z, gridDim.y, gridDim.x, 
       blockDim.z, blockDim.y, blockDim.x,
       blockIdx.z,  blockIdx.y,  blockIdx.x, 
       threadIdx.z, threadIdx.y, threadIdx.x,
       shared_memory);

  task.get_functor().Operation1();
  SyncBlock<CUDA>::Sync();

  while (task.get_functor().LoopCondition1()) {
    task.get_functor().Operation2();
    SyncBlock<CUDA>::Sync();
    task.get_functor().Operation3();
    SyncBlock<CUDA>::Sync();
    task.get_functor().Operation4();
    SyncBlock<CUDA>::Sync();
    task.get_functor().Operation5();
    SyncBlock<CUDA>::Sync();
  }

  task.get_functor().Operation6();
  SyncBlock<CUDA>::Sync();
  task.get_functor().Operation7();
  SyncBlock<CUDA>::Sync();
  task.get_functor().Operation8();
  SyncBlock<CUDA>::Sync();

  task.get_functor().Operation9();
  SyncBlock<CUDA>::Sync();
  
  while (task.get_functor().LoopCondition2()) {
    task.get_functor().Operation10();
    SyncBlock<CUDA>::Sync();
    task.get_functor().Operation11();
    SyncBlock<CUDA>::Sync();
    task.get_functor().Operation12();
    SyncBlock<CUDA>::Sync();
    task.get_functor().Operation13();
    SyncBlock<CUDA>::Sync();
  }

  task.get_functor().Operation14();
  SyncBlock<CUDA>::Sync();
  task.get_functor().Operation15();
  SyncBlock<CUDA>::Sync();
  task.get_functor().Operation16();
  SyncBlock<CUDA>::Sync();
}


template <typename Task>
MGARDm_KERL void HuffmanCLCustomizedKernel(Task task) {
  Byte *shared_memory = SharedMemory<Byte>();

  task.get_functor().Init(gridDim.z, gridDim.y, gridDim.x, 
       blockDim.z, blockDim.y, blockDim.x,
       blockIdx.z,  blockIdx.y,  blockIdx.x, 
       threadIdx.z, threadIdx.y, threadIdx.x,
       shared_memory);

  task.get_functor().Operation1();
  SyncGrid<CUDA>::Sync();
  while (task.get_functor().LoopCondition1()) {
    task.get_functor().Operation2();
    SyncGrid<CUDA>::Sync();
    task.get_functor().Operation3();
    SyncGrid<CUDA>::Sync();
    task.get_functor().Operation4();
    SyncGrid<CUDA>::Sync();
    task.get_functor().Operation5();
    SyncBlock<CUDA>::Sync();
    if (task.get_functor().BranchCondition1()) {
      while (task.get_functor().LoopCondition2()) {
        task.get_functor().Operation6();
        SyncBlock<CUDA>::Sync();
        task.get_functor().Operation7();
        SyncBlock<CUDA>::Sync();
        task.get_functor().Operation8();
        SyncBlock<CUDA>::Sync();
      }
      task.get_functor().Operation9();
      SyncGrid<CUDA>::Sync();
      task.get_functor().Operation10();
      SyncGrid<CUDA>::Sync();
    }
    task.get_functor().Operation11();
    SyncGrid<CUDA>::Sync();
    task.get_functor().Operation12();
    SyncGrid<CUDA>::Sync();
    task.get_functor().Operation13();
    SyncGrid<CUDA>::Sync();
    task.get_functor().Operation14();
    SyncGrid<CUDA>::Sync();
  }
}

template <typename Task>
MGARDm_KERL void HuffmanCWCustomizedKernel(Task task) {
  Byte *shared_memory = SharedMemory<Byte>();

  task.get_functor().Init(gridDim.z, gridDim.y, gridDim.x, 
       blockDim.z, blockDim.y, blockDim.x,
       blockIdx.z,  blockIdx.y,  blockIdx.x, 
       threadIdx.z, threadIdx.y, threadIdx.x,
       shared_memory);

  task.get_functor().Operation1();
  SyncGrid<CUDA>::Sync();
  task.get_functor().Operation2();
  SyncGrid<CUDA>::Sync();
  task.get_functor().Operation3();
  SyncGrid<CUDA>::Sync();

  while (task.get_functor().LoopCondition1()) {
    task.get_functor().Operation4();
    SyncGrid<CUDA>::Sync();
    task.get_functor().Operation5();
    SyncGrid<CUDA>::Sync();
    task.get_functor().Operation6();
    SyncGrid<CUDA>::Sync();
    task.get_functor().Operation7();
    SyncBlock<CUDA>::Sync();
    task.get_functor().Operation8();
    SyncBlock<CUDA>::Sync();
  }
  task.get_functor().Operation9();
  SyncBlock<CUDA>::Sync();
  task.get_functor().Operation10();
  SyncGrid<CUDA>::Sync();
}




template <>
class DeviceSpecification<CUDA> {
  public:
  MGARDm_CONT
  DeviceSpecification(){
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

  MGARDm_CONT int
  GetNumDevices() {
    return NumDevices;
  }

  MGARDm_CONT int
  GetMaxSharedMemorySize(int dev_id) {
    return MaxSharedMemorySize[dev_id];
  }

  MGARDm_CONT int
  GetWarpSize(int dev_id) {
    return WarpSize[dev_id];
  }

  MGARDm_CONT int
  GetNumSMs(int dev_id) {
    return NumSMs[dev_id];
  }

  MGARDm_CONT int
  GetArchitectureGeneration(int dev_id) {
    return ArchitectureGeneration[dev_id];
  }

  MGARDm_CONT int
  GetMaxNumThreadsPerSM(int dev_id) {
    return MaxNumThreadsPerSM[dev_id];
  }

  MGARDm_CONT
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
class DeviceQueues<CUDA> {
  public:
  MGARDm_CONT
  DeviceQueues(){
    cudaGetDeviceCount(&NumDevices);
    streams = new cudaStream_t*[NumDevices];
    for (int d = 0; d < NumDevices; d++) {
      gpuErrchk(cudaSetDevice(d));
      streams[d] = new cudaStream_t[MGARDm_NUM_QUEUES];
      for (SIZE i = 0; i < MGARDm_NUM_QUEUES; i++) {
        gpuErrchk(cudaStreamCreate(&streams[d][i]));
      }
    }
  }

  MGARDm_CONT cudaStream_t 
  GetQueue(int dev_id, SIZE queue_id){
    return streams[dev_id][queue_id];
  }

  MGARDm_CONT void 
  SyncQueue(int dev_id, SIZE queue_id){
    cudaStreamSynchronize(streams[dev_id][queue_id]);
  }

  MGARDm_CONT void 
  SyncAllQueues(int dev_id){
    for (SIZE i = 0; i < MGARDm_NUM_QUEUES; i++) {
      gpuErrchk(cudaStreamSynchronize(streams[dev_id][i]));
    }
  }

  MGARDm_CONT
  ~DeviceQueues(){
    for (int d = 0; d < NumDevices; d++) {
      gpuErrchk(cudaSetDevice(d));
      for (int i = 0; i < MGARDm_NUM_QUEUES; i++) {
        gpuErrchk(cudaStreamDestroy(streams[d][i]));
      }
      delete [] streams[d];
    }
    delete [] streams;
    streams = NULL;
  }
  
  int NumDevices;
  cudaStream_t ** streams = NULL;
};



template <>
class DeviceRuntime<CUDA> {
  public:
  MGARDm_CONT
  DeviceRuntime(){}

  MGARDm_CONT static void 
  SelectDevice(SIZE dev_id){
    gpuErrchk(cudaSetDevice(dev_id));
    curr_dev_id = dev_id;
  }

  MGARDm_CONT static cudaStream_t 
  GetQueue(SIZE queue_id){
    gpuErrchk(cudaSetDevice(curr_dev_id));
    return queues.GetQueue(curr_dev_id, queue_id);
  }

  MGARDm_CONT static void 
  SyncQueue(SIZE queue_id){
    gpuErrchk(cudaSetDevice(curr_dev_id));
    queues.SyncQueue(curr_dev_id, queue_id);
  }

  MGARDm_CONT static void 
  SyncAllQueues(){
    gpuErrchk(cudaSetDevice(curr_dev_id));
    queues.SyncAllQueues(curr_dev_id);
  }

  MGARDm_CONT static void
  SyncDevice(){
    cudaSetDeviceHelper(curr_dev_id);
  }

  MGARDm_CONT static int
  GetMaxSharedMemorySize() {
    return DeviceSpecs.GetMaxSharedMemorySize(curr_dev_id);
  }

  MGARDm_CONT static int
  GetWarpSize() {
    return DeviceSpecs.GetWarpSize(curr_dev_id);
  }

  MGARDm_CONT static int
  GetNumSMs() {
    return DeviceSpecs.GetNumSMs(curr_dev_id);
  }

  MGARDm_CONT static int
  GetArchitectureGeneration() {
    return DeviceSpecs.GetArchitectureGeneration(curr_dev_id);
  }

  MGARDm_CONT static int
  GetMaxNumThreadsPerSM() {
    return DeviceSpecs.GetMaxNumThreadsPerSM(curr_dev_id);
  }

  template <typename FunctorType>
  MGARDm_CONT static int
  GetOccupancyMaxActiveBlocksPerSM(FunctorType functor, int blockSize, size_t dynamicSMemSize) {
    int numBlocks = 0;
    Task<FunctorType> task = Task(functor, 1, 1, 1, 
                1, 1, blockSize, dynamicSMemSize, 0);

    if constexpr (std::is_base_of<Functor<CUDA>, 
      FunctorType>::value) {
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocks, 
       Kernel<Task<FunctorType>>,
       blockSize, 
       dynamicSMemSize);
    } else if constexpr (std::is_base_of<IterFunctor<CUDA>, FunctorType>::value) {
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocks, IterKernel<Task<FunctorType>>, blockSize, dynamicSMemSize);
    } else if constexpr (std::is_base_of<HuffmanCLCustomizedFunctor<CUDA>, FunctorType>::value) {
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocks, HuffmanCLCustomizedKernel<Task<FunctorType>>, blockSize, dynamicSMemSize);
    }
    return numBlocks;
  }

  MGARDm_CONT
  ~DeviceRuntime(){
  }

  static int curr_dev_id;
  static DeviceQueues<CUDA> queues;
  static bool SyncAllKernelsAndCheckErrors;
  static DeviceSpecification<CUDA> DeviceSpecs;
};


template <>
class MemoryManager<CUDA> {
  public:
  MGARDm_CONT
  MemoryManager(){};

  template <typename T>
  MGARDm_CONT
  void Malloc1D(T *& ptr, SIZE n, int queue_idx) {
    gpuErrchk(cudaMalloc(&ptr, n * sizeof(T)));
    if (DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors) {
      gpuErrchk(cudaDeviceSynchronize());
    }
  }

  template <typename T>
  MGARDm_CONT
  void MallocND(T *& ptr, SIZE n1, SIZE n2, SIZE &ld, int queue_idx) {
    if (ReduceMemoryFootprint) {
      gpuErrchk(cudaMalloc(&ptr, n1 * n2 * sizeof(T)));
      ld = n1;
    } else {
      size_t pitch = 0;
      gpuErrchk(cudaMallocPitch(&ptr, &pitch, n1 * sizeof(T), (size_t)n2));
      ld = pitch / sizeof(T);
    }
    if (DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors) {
      gpuErrchk(cudaDeviceSynchronize());
    }
  }

  template <typename T>
  MGARDm_CONT
  void Free(T * ptr) {
    // printf("MemoryManager.Free(%llu)\n", ptr);
    gpuErrchk(cudaFree(ptr));
    if (DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors) {
      gpuErrchk(cudaDeviceSynchronize());
    }
  }

  template <typename T>
  MGARDm_CONT
  void Copy1D(T * dst_ptr, const T * src_ptr, SIZE n, int queue_idx) {
    cudaStream_t stream = DeviceRuntime<CUDA>::GetQueue(queue_idx);
    gpuErrchk(cudaMemcpyAsync(dst_ptr, src_ptr, n*sizeof(T), cudaMemcpyDefault, stream));
    if (DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors) {
      gpuErrchk(cudaDeviceSynchronize());
    }
  }

  template <typename T>
  MGARDm_CONT
  void CopyND(T * dst_ptr, SIZE dst_ld, const T * src_ptr, SIZE src_ld, SIZE n1, SIZE n2, int queue_idx) {
    cudaStream_t stream = DeviceRuntime<CUDA>::GetQueue(queue_idx);
    gpuErrchk(cudaMemcpy2DAsync(dst_ptr, dst_ld * sizeof(T), src_ptr, src_ld * sizeof(T), n1 * sizeof(T), n2,
                              cudaMemcpyDefault, stream));
    if (DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors) {
      gpuErrchk(cudaDeviceSynchronize());
    }
  }

  template <typename T>
  MGARDm_CONT
  void MallocHost(T *& ptr, SIZE n, int queue_idx) {
    gpuErrchk(cudaMallocHost(&ptr, n * sizeof(T)));
    if (DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors) {
      gpuErrchk(cudaDeviceSynchronize());
    }
  }

  template <typename T>
  MGARDm_CONT
  void FreeHost(T * ptr) {
    gpuErrchk(cudaFreeHost(ptr));
    if (DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors) {
      gpuErrchk(cudaDeviceSynchronize());
    }
  }

  static bool ReduceMemoryFootprint;
};


MGARDm_CONT_EXEC
uint64_t binary2negabinary(const int64_t x) {
    return (x + (uint64_t)0xaaaaaaaaaaaaaaaaull) ^ (uint64_t)0xaaaaaaaaaaaaaaaaull;
}

MGARDm_CONT_EXEC
uint32_t binary2negabinary(const int32_t x) {
    return (x + (uint32_t)0xaaaaaaaau) ^ (uint32_t)0xaaaaaaaau;
}

MGARDm_CONT_EXEC
int64_t negabinary2binary(const uint64_t x) {
    return (x ^0xaaaaaaaaaaaaaaaaull) - 0xaaaaaaaaaaaaaaaaull;
}

MGARDm_CONT_EXEC
int32_t negabinary2binary(const uint32_t x) {
    return (x ^0xaaaaaaaau) - 0xaaaaaaaau;
}



template <typename T, SIZE nblockx, SIZE nblocky, SIZE nblockz> 
struct BlockReduce<T, nblockx, nblocky, nblockz, CUDA> {
  typedef cub::BlockReduce<T, nblockx, cub::BLOCK_REDUCE_WARP_REDUCTIONS, nblocky, nblockz> BlockReduceType;
  using TempStorageType = typename BlockReduceType::TempStorage;

  BlockReduceType* blockReduce;

  MGARDm_EXEC
  BlockReduce() {
    __shared__ TempStorageType temp_storage;
    blockReduce = new BlockReduceType(temp_storage);
  }

  MGARDm_EXEC
  ~BlockReduce() {
    delete blockReduce;
  }

  MGARDm_EXEC 
  T Sum(T intput) {
    return blockReduce->Sum(intput);
  }

  MGARDm_EXEC 
  T Max(T intput) {
    return blockReduce->Reduce(intput, cub::Max());
  }
};

#define ALIGN_LEFT 0 // for encoding
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


template <typename T, OPTION METHOD>
struct EncodeSignBits<T, METHOD, CUDA>{
  MGARDm_EXEC 
  T Atomic(T bit, SIZE b_idx) {
    T buffer = 0;
    T shifted_bit;
    shifted_bit = bit << sizeof(T)*8-1-b_idx;
    atomicAdd_block(buffer, shifted_bit);
    return buffer;
  }

  MGARDm_EXEC 
  T Reduction(T bit, SIZE b_idx) {
    T buffer = 0;
    T shifted_bit;
    shifted_bit = bit << sizeof(T)*8-1-b_idx;

    typedef cub::WarpReduce<T> WarpReduceType;
    using WarpReduceStorageType = typename WarpReduceType::TempStorage;
    __shared__ WarpReduceStorageType warp_storage;
    buffer = WarpReduceType(warp_storage).Sum(shifted_bit);
    return buffer;
  }

  MGARDm_EXEC 
  T Ballot(T bit, SIZE b_idx) {
    return (T)__ballot_sync (0xffffffff, (int)bit);
  }


  MGARDm_EXEC 
  T Encode(T bit, SIZE b_idx) {
    if (METHOD == Sign_Encoding_Atomic) return Atomic(bit, b_idx);
    else if (METHOD == Sign_Encoding_Reduce) return Reduction(bit, b_idx);
    else if (METHOD == Sign_Encoding_Ballot) return Ballot(bit, b_idx);
    else { printf("Sign Encoding Wrong Algorithm Type!\n"); }
  }
};


template <typename T, OPTION METHOD>
struct DecodeSignBits<T, METHOD, CUDA>{
  MGARDm_EXEC 
  T Decode(T sign_bitplane, SIZE b_idx) {
    return (sign_bitplane >> (sizeof(T)*8-1-b_idx)) & 1u;
  }
};


template <typename T_org, typename T_trans, SIZE nblockx, SIZE nblocky, SIZE nblockz, OPTION ALIGN, OPTION METHOD> 
struct BlockBitTranspose<T_org, T_trans, nblockx, nblocky, nblockz, ALIGN, METHOD, CUDA> {
  
  typedef cub::WarpReduce<T_trans> WarpReduceType;
  using WarpReduceStorageType = typename WarpReduceType::TempStorage;

  typedef cub::BlockReduce<T_trans, nblockx, cub::BLOCK_REDUCE_WARP_REDUCTIONS, nblocky, nblockz> BlockReduceType;
  using BlockReduceStorageType = typename BlockReduceType::TempStorage;

  MGARDm_EXEC 
  void Serial_All(T_org * v, T_trans * tv, SIZE b, SIZE B) {
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      // printf("add-in: %llu %u\n", v, v[0]);
      // for (int i = 0; i < b; i++) {
      //   printf("v: %u\n", v[i]);
      // }
      for (SIZE B_idx = 0; B_idx < B; B_idx++) {
        T_trans buffer = 0; 
        for (SIZE b_idx = 0; b_idx < b; b_idx++) {
          T_trans bit = (v[b_idx] >> (sizeof(T_org)*8 - 1 - B_idx)) & 1u;
          if (ALIGN == ALIGN_LEFT) {
            buffer += bit << (sizeof(T_trans)*8-1-b_idx); 
            // if (B_idx == 0) {
              // printf("%u %u %u\n", B_idx, b_idx, bit);
              // print_bits(buffer, sizeof(T_trans)*8, false);
              // printf("\n");
            // }
          } else if (ALIGN == ALIGN_RIGHT) {
            buffer += bit << (b-1-b_idx); 
            // if (b_idx == 0) printf("%u %u %u\n", B_idx, b_idx, bit);
          } else { }
          // if (j == 0 ) {printf("i %u j %u shift %u bit %u\n", i,j,b-1-j, bit); }
        }

        // printf("buffer: %u\n", buffer);

        tv[B_idx] = buffer;
      }
    }
  }

  MGARDm_EXEC 
  void Parallel_B_Serial_b(T_org * v, T_trans * tv, SIZE b, SIZE B) {
    if (threadIdx.y == 0) {
      for (SIZE B_idx = threadIdx.x; B_idx < B; B_idx += 32) {
        T_trans buffer = 0; 
        for (SIZE b_idx = 0; b_idx < b; b_idx++) {
          T_trans bit = (v[b_idx] >> (sizeof(T_org)*8 - 1 - B_idx)) & 1u;
          if (ALIGN == ALIGN_LEFT) {
            buffer += bit << sizeof(T_trans)*8-1-b_idx; 
          } else if (ALIGN == ALIGN_RIGHT) {
            buffer += bit << (b-1-b_idx); 
            // if (b_idx == 0) printf("%u %u %u\n", B_idx, b_idx, bit);
          } else { }
        }
        tv[B_idx] = buffer;
      }
    }
  }  

  MGARDm_EXEC 
  void Parallel_B_Atomic_b(T_org * v, T_trans * tv, SIZE b, SIZE B) {
    if (threadIdx.x < b && threadIdx.y < B) {
      SIZE i = threadIdx.x;
      for (SIZE B_idx = threadIdx.y; B_idx < B; B_idx += 32) {
        for (SIZE b_idx = threadIdx.x; b_idx < b; b_idx += 32) {
          T_trans bit = (v[b_idx] >> (sizeof(T_org)*8 - 1 - B_idx)) & 1u;
          T_trans shifted_bit;
          if (ALIGN == ALIGN_LEFT) {
            shifted_bit = bit << sizeof(T_trans)*8-1-b_idx;
          } else if (ALIGN == ALIGN_RIGHT) {
            shifted_bit = bit << b-1-b_idx;
          } else { }
          T_trans * sum = &(tv[B_idx]);
          // atomicAdd_block(sum, shifted_bit);
        }
      }
    }
  }  

  MGARDm_EXEC 
  void Parallel_B_Reduction_b(T_org * v, T_trans * tv, SIZE b, SIZE B) {

    // __syncthreads(); long long start = clock64();

    __shared__ WarpReduceStorageType warp_storage[32];

    SIZE warp_idx = threadIdx.y;
    SIZE lane_idx = threadIdx.x;
    SIZE B_idx, b_idx;
    T_trans bit = 0;
    T_trans shifted_bit = 0;
    T_trans sum = 0;

    // __syncthreads(); start = clock64() - start;
    // if (threadIdx.y == 0 && threadIdx.x == 0) printf("time0: %llu\n", start);
    // __syncthreads(); start = clock64();

    for (SIZE B_idx = threadIdx.y; B_idx < B; B_idx += 32) {
      sum = 0;
      for (SIZE b_idx = threadIdx.x; b_idx < ((b-1)/32+1)*32; b_idx += 32) {
        shifted_bit = 0;
        if (b_idx < b && B_idx < B) {
          bit = (v[b_idx] >> (sizeof(T_org)*8 - 1 - B_idx)) & 1u;
          if (ALIGN == ALIGN_LEFT) {
            shifted_bit = bit << sizeof(T_trans)*8-1-b_idx; 
          } else if (ALIGN == ALIGN_RIGHT) {
            shifted_bit = bit << b-1-b_idx; 
          } else { }
        }

        // __syncthreads(); start = clock64() - start;
        // if (threadIdx.y == 0 && threadIdx.x == 0) printf("time1: %llu\n", start);
        // __syncthreads(); start = clock64();

        sum += WarpReduceType(warp_storage[warp_idx]).Sum(shifted_bit);
        // if (B_idx == 32) printf("shifted_bit[%u] %u sum %u\n", b_idx, shifted_bit, sum);
      }
      if (lane_idx == 0) {
        tv[B_idx] = sum;
      }
    }

    // __syncthreads(); start = clock64() - start;
    // if (threadIdx.y == 0 && threadIdx.x == 0) printf("time2: %llu\n", start);
    // __syncthreads(); start = clock64();
  }  

  MGARDm_EXEC 
  void Parallel_B_Ballot_b(T_org * v, T_trans * tv, SIZE b, SIZE B) {

    SIZE warp_idx = threadIdx.y;
    SIZE lane_idx = threadIdx.x;
    SIZE B_idx, b_idx;
    int bit = 0;
    T_trans sum = 0;
    
    // __syncthreads(); start = clock64() - start;
    // if (threadIdx.y == 0 && threadIdx.x == 0) printf("time0: %llu\n", start);
    // __syncthreads(); start = clock64();


    for (SIZE B_idx = threadIdx.y; B_idx < B; B_idx += 32) {
      sum = 0;
      SIZE shift = 0;
      for (SIZE b_idx = threadIdx.x; b_idx < ((b-1)/32+1)*32; b_idx += 32) {
        bit = 0;
        if (b_idx < b && B_idx < B) {
          if (ALIGN == ALIGN_LEFT) {
            bit = (v[sizeof(T_trans)*8-1-b_idx] >> (sizeof(T_org)*8 - 1 - B_idx)) & 1u;
          } else if (ALIGN == ALIGN_RIGHT) {
            bit = (v[b-1-b_idx] >> (sizeof(T_org)*8 - 1 - B_idx)) & 1u;
          } else { }
        }

        // __syncthreads(); start = clock64() - start;
        // if (threadIdx.y == 0 && threadIdx.x == 0) printf("time1: %llu\n", start);
        // __syncthreads(); start = clock64();
        sum += ((T_trans)__ballot_sync (0xffffffff, bit)) << shift;
        // sum += WarpReduceType(warp_storage[warp_idx]).Sum(shifted_bit);
        // if (B_idx == 32) printf("shifted_bit[%u] %u sum %u\n", b_idx, shifted_bit, sum);
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

  MGARDm_EXEC 
  void TCU(T_org * v, T_trans * tv, SIZE b, SIZE B) {
    __syncthreads();
    long long start = clock64();

    __shared__ half tile_a[16*16];
    __shared__ half tile_b[32*32];
    __shared__ float output[32*32];
    uint8_t bit;
    half shifted_bit;
    SIZE i = threadIdx.x;
    SIZE B_idx = threadIdx.y;
    SIZE b_idx = threadIdx.x;
    SIZE warp_idx = threadIdx.y;
    SIZE lane_idx = threadIdx.x;
    
    __syncthreads();
    start = clock64() - start;
    if (threadIdx.y == 0 && threadIdx.x == 0) printf("time0: %llu\n", start);

    __syncthreads();
    start = clock64();
    __syncthreads();
   
    if (threadIdx.x < B * b) {
      uint8_t bit = (v[sizeof(T_trans)*8-1-b_idx] >> (sizeof(T_org)*8 - 1 - B_idx)) & 1u;
      shifted_bit = bit << (sizeof(T_trans)*8-1-b_idx) % 8; 
      tile_b[b_idx * 32 + B_idx] = shifted_bit;
      if (i < 8) { 
        tile_a[i] = 1u;
        tile_a[i+8] = 1u << 8;
      }
    }
    __syncthreads();
    start = clock64() - start;
    if (threadIdx.y == 0 && threadIdx.x == 0) printf("time1: %llu\n", start);

    __syncthreads();
    start = clock64();
    __syncthreads();
    
    if (warp_idx < 4) { 
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
      wmma::load_matrix_sync(a_frag, tile_a, 16);
      wmma::load_matrix_sync(b_frag, tile_b + (warp_idx/2)*16 + (warp_idx%2)*16, 32);
      wmma::fill_fragment(c_frag, 0.0f);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      wmma::store_matrix_sync(output+ (warp_idx/2)*16 + (warp_idx%2)*16, c_frag, 32, wmma::mem_row_major);
    }

    __syncthreads();
    start = clock64() - start;
    if (threadIdx.y == 0 && threadIdx.x == 0) printf("time2: %llu\n", start);

  }

  MGARDm_EXEC 
  void Transpose(T_org * v, T_trans * tv, SIZE b, SIZE B) {
    if (METHOD == Bit_Transpose_Serial_All)  Serial_All(v, tv, b, B);
    else if (METHOD == Bit_Transpose_Parallel_B_Serial_b) Parallel_B_Serial_b(v, tv, b, B);
    else if (METHOD == Bit_Transpose_Parallel_B_Atomic_b) Parallel_B_Atomic_b(v, tv, b, B);
    else if (METHOD == Bit_Transpose_Parallel_B_Reduce_b) Parallel_B_Reduction_b(v, tv, b, B);
    else if (METHOD == Bit_Transpose_Parallel_B_Ballot_b) Parallel_B_Ballot_b(v, tv, b, B);
    // else { printf("Bit Transpose Wrong Algorithm Type!\n");  }
    // else if (METHOD == 5) TCU(v, tv, b, B);
  }
};


template <typename T, typename T_fp, typename T_sfp, typename T_error, SIZE nblockx, SIZE nblocky, SIZE nblockz, OPTION METHOD, OPTION BinaryType> 
struct ErrorCollect<T, T_fp, T_sfp, T_error, nblockx, nblocky, nblockz, METHOD, BinaryType, CUDA>{

  typedef cub::WarpReduce<T_error> WarpReduceType;
  using WarpReduceStorageType = typename WarpReduceType::TempStorage;

  typedef cub::BlockReduce<T_error, nblockx, cub::BLOCK_REDUCE_WARP_REDUCTIONS, nblocky, nblockz> BlockReduceType;
  using BlockReduceStorageType = typename BlockReduceType::TempStorage;


  MGARDm_EXEC 
  void Serial_All(T * v, T_error * temp, T_error * errors, SIZE num_elems, SIZE num_bitplanes) {
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      for (SIZE elem_idx = 0; elem_idx < num_elems; elem_idx++) {
        T data = v[elem_idx];
        T_fp fp_data = (T_fp) fabs(data);
        T_sfp fps_data = (T_sfp) data;
        T_fp ngb_data = binary2negabinary(fps_data);
        T_error mantissa;
        if (BinaryType == BINARY) {
          mantissa = fabs(data) - fp_data;
        } else if (BinaryType == NEGABINARY) {
          mantissa = data - fps_data;
        }
        for(SIZE bitplane_idx = 0; bitplane_idx < num_bitplanes; bitplane_idx++){
          uint64_t mask = (1 << bitplane_idx) - 1;
          T_error diff = 0;
          if (BinaryType == BINARY) {
            diff = (T_error) (fp_data & mask) + mantissa;
          } else if (BinaryType == NEGABINARY) {
            diff = (T_error) negabinary2binary(ngb_data & mask) + mantissa;
          }
          errors[num_bitplanes-bitplane_idx] += diff * diff;
          // if (blockIdx.x == 0 && num_bitplanes-bitplane_idx == 2) {
          //   printf("elem error[%u]: %f\n", elem_idx, diff * diff);
          // }
        }
        errors[0] += data * data;
      }
    }
  }

  MGARDm_EXEC
  void Parallel_Bitplanes_Serial_Error(T * v, T_error * temp, T_error * errors, SIZE num_elems, SIZE num_bitplanes) {
    SIZE bitplane_idx = threadIdx.y * blockDim.x + threadIdx.x;
    if (bitplane_idx < num_bitplanes) {
      for (SIZE elem_idx = 0; elem_idx < num_elems; elem_idx++) {
        T data = v[elem_idx];
        T_fp fp_data = (T_fp) fabs(v[elem_idx]);
        T_sfp fps_data = (T_sfp) data;
        T_fp ngb_data = binary2negabinary(fps_data);
        T_error mantissa;
        if (BinaryType == BINARY) {
          mantissa = fabs(data) - fp_data;
        } else if (BinaryType == NEGABINARY) {
          mantissa = data - fps_data;
        }
        uint64_t mask = (1 << bitplane_idx) - 1;
        T_error diff = 0;
        if (BinaryType == BINARY) {
          diff = (T_error) (fp_data & mask) + mantissa;
        } else if (BinaryType == NEGABINARY) {
          diff = (T_error) negabinary2binary(ngb_data & mask) + mantissa;
        }
        errors[num_bitplanes-bitplane_idx] += diff * diff;
      }
    }
    if (bitplane_idx == 0) {
      for (SIZE elem_idx = 0; elem_idx < num_elems; elem_idx++) {
        T data = v[elem_idx];
        errors[0] += data * data;
      }
    }
  }

    MGARDm_EXEC
  void Parallel_Bitplanes_Atomic_Error(T * v, T_error * temp, T_error * errors, SIZE num_elems, SIZE num_bitplanes) {
    for (SIZE elem_idx = threadIdx.x; elem_idx < num_elems; elem_idx += blockDim.x) {
      for(SIZE bitplane_idx = threadIdx.y; bitplane_idx < num_bitplanes; bitplane_idx += blockDim.y){
        T data = v[elem_idx];
        T_fp fp_data = (T_fp) fabs(v[elem_idx]);
        T_sfp fps_data = (T_sfp) data;
        T_fp ngb_data = binary2negabinary(fps_data);
        T_error mantissa;
        if (BinaryType == BINARY) {
          mantissa = fabs(data) - fp_data;
        } else if (BinaryType == NEGABINARY) {
          mantissa = data - fps_data;
        }
        uint64_t mask = (1 << bitplane_idx) - 1;
        T_error diff = 0;
        if (BinaryType == BINARY) {
          diff = (T_error) (fp_data & mask) + mantissa;
        } else if (BinaryType == NEGABINARY) {
          diff = (T_error) negabinary2binary(ngb_data & mask) + mantissa;
        }
        temp[(num_bitplanes - bitplane_idx) * num_elems + elem_idx] = diff * diff;
        if (bitplane_idx == 0) {
          temp[elem_idx] = data * data;
        }
      }
    }
    __syncthreads();
    // if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
    //     for (SIZE elem_idx = 0; elem_idx < num_elems; elem_idx += 1) {
    //       printf("elem error[%u]: %f\n", elem_idx, temp[(2) * num_elems + elem_idx]);
    //     }
    // }
    for(SIZE bitplane_idx = threadIdx.y; bitplane_idx < num_bitplanes+1; bitplane_idx += blockDim.y){
      for (SIZE elem_idx = threadIdx.x; elem_idx < ((num_elems-1)/32+1)*32; elem_idx += 32) {
        T_error error = 0;
        if (elem_idx < num_elems) {
          error = temp[(num_bitplanes - bitplane_idx) * num_elems + elem_idx];
        }
        T_error * sum = &(errors[num_bitplanes - bitplane_idx]);
        atomicAdd_block(sum, error);
      }
    }
  }

  MGARDm_EXEC
  void Parallel_Bitplanes_Reduce_Error(T * v, T_error * temp, T_error * errors, SIZE num_elems, SIZE num_bitplanes) {
    for (SIZE elem_idx = threadIdx.x; elem_idx < num_elems; elem_idx += blockDim.x) {
      for(SIZE bitplane_idx = threadIdx.y; bitplane_idx < num_bitplanes; bitplane_idx += blockDim.y){
        T data = v[elem_idx];
        T_fp fp_data = (T_fp) fabs(v[elem_idx]);
        T_sfp fps_data = (T_sfp) data;
        T_fp ngb_data = binary2negabinary(fps_data);
        T_error mantissa;
        if (BinaryType == BINARY) {
          mantissa = fabs(data) - fp_data;
        } else if (BinaryType == NEGABINARY) {
          mantissa = data - fps_data;
        }
        uint64_t mask = (1 << bitplane_idx) - 1;
        T_error diff = 0;
        if (BinaryType == BINARY) {
          diff = (T_error) (fp_data & mask) + mantissa;
        } else if (BinaryType == NEGABINARY) {
          diff = (T_error) negabinary2binary(ngb_data & mask) + mantissa;
        }
        temp[(num_bitplanes - bitplane_idx) * num_elems + elem_idx] = diff * diff;
        // if (blockIdx.x == 0 && num_bitplanes - bitplane_idx == 31) {
        //   printf("elem_idx: %u, data: %f, fp_data: %u, mask: %u, mantissa: %f, diff: %f\n",
        //           elem_idx, data, fp_data, mask, mantissa, diff);
        // }
        if (bitplane_idx == 0) {
          temp[elem_idx] = data * data;
        }
      }
    }
    __syncthreads();

    // if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
    //   for (SIZE elem_idx = 0; elem_idx < num_elems; elem_idx += 1) {
    //     printf("elem error[%u]: %f\n", elem_idx, temp[(31) * num_elems + elem_idx]);
    //   }
    // }


    __shared__ WarpReduceStorageType warp_storage[nblocky];

    for(SIZE bitplane_idx = threadIdx.y; bitplane_idx < num_bitplanes+1; bitplane_idx += blockDim.y){
      T error_sum = 0;
      for (SIZE elem_idx = threadIdx.x; elem_idx < ((num_elems-1)/32+1)*32; elem_idx += 32) {
        T_error error = 0;
        if (elem_idx < num_elems) {
          error = temp[(num_bitplanes - bitplane_idx) * num_elems + elem_idx];
        }
        error_sum += WarpReduceType(warp_storage[threadIdx.y]).Sum(error);
        // errors[num_bitplanes - bitplane_idx] += WarpReduceType(warp_storage[threadIdx.y]).Sum(error);
      }
      if (threadIdx.x == 0) { 
        errors[num_bitplanes - bitplane_idx] = error_sum;
      }
    }
    // __syncthreads();
    // if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
    //   for (int i = 0; i < num_bitplanes + 1; i++) {
    //     printf("error[%d]: %f\n", i, errors[i]);
    //   }
    // }


  }

  MGARDm_EXEC 
  void Collect(T * v, T_error * temp, T_error * errors, SIZE num_elems, SIZE num_bitplanes) {
    if (METHOD == Error_Collecting_Serial_All) Serial_All(v, temp, errors, num_elems, num_bitplanes);
    else if (METHOD == Error_Collecting_Parallel_Bitplanes_Serial_Error) Parallel_Bitplanes_Serial_Error(v, temp, errors, num_elems, num_bitplanes);
    else if (METHOD == Error_Collecting_Parallel_Bitplanes_Atomic_Error) Parallel_Bitplanes_Atomic_Error(v, temp, errors, num_elems, num_bitplanes);
    else if (METHOD == Error_Collecting_Parallel_Bitplanes_Reduce_Error) Parallel_Bitplanes_Reduce_Error(v, temp, errors, num_elems, num_bitplanes);
    // else if (METHOD == Error_Collecting_Disable) {}
    // else { printf("Error Collecting Wrong Algorithm Type!\n");  }
  }
};

template <typename T> 
struct BlockBroadcast<T, CUDA> {
  
  MGARDm_EXEC 
  T Broadcast(T input, SIZE src_threadx, SIZE src_thready, SIZE src_threadz) {
    __shared__ T value[1];
    if (threadIdx.x == src_threadx && 
        threadIdx.y == src_thready &&
        threadIdx.z == src_threadz) {
      value[0] = input;
      // printf("bcast: %u %u\n", input, value[0]);
    }
    __syncthreads();
    // printf("bcast-other[%u]: %u\n", threadIdx.x, value[0]);
    return value[0];
  }
};









template <typename TaskType>
class DeviceAdapter<TaskType, CUDA> {
public:
  MGARDm_CONT
  DeviceAdapter(){};

  MGARDm_CONT
  void Execute(TaskType& task) {
    dim3 threadsPerBlock(task.get_nblockx(),
                         task.get_nblocky(),
                         task.get_nblockz());
    dim3 blockPerGrid(task.get_ngridx(),
                      task.get_ngridy(),
                      task.get_ngridz());
    size_t sm_size = task.get_shared_memory_size();
    // printf("exec config (%d %d %d) (%d %d %d) sm_size: %llu\n", threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z, 
    //                 blockPerGrid.x, blockPerGrid.y, blockPerGrid.z, sm_size);
    cudaStream_t stream = DeviceRuntime<CUDA>::GetQueue(task.get_queue_idx());

    // if constexpr evalute at compile time otherwise this does not compile
    if constexpr (std::is_base_of<Functor<CUDA>, typename TaskType::Functor>::value) {
    // if constexpr (TaskType::Functor::ExecType == SequentialFunctor) {
      Kernel<<<blockPerGrid, threadsPerBlock, sm_size, stream>>>(task);

    } else if constexpr (std::is_base_of<IterFunctor<CUDA>, typename TaskType::Functor>::value) {
    // } else if constexpr (TaskType::Functor::ExecType == IterativeFunctor) {
      IterKernel<<<blockPerGrid, threadsPerBlock, sm_size, stream>>>(task);
    } else if constexpr (std::is_base_of<HuffmanCLCustomizedFunctor<CUDA>, typename TaskType::Functor>::value) {
      void * Args[] = { (void*)&task };
      cudaLaunchCooperativeKernel((void *)HuffmanCLCustomizedKernel<TaskType>,
                              blockPerGrid, threadsPerBlock, Args, sm_size);
    } else if constexpr (std::is_base_of<HuffmanCWCustomizedFunctor<CUDA>, typename TaskType::Functor>::value) {
      void * Args[] = { (void*)&task };
      cudaLaunchCooperativeKernel((void *)HuffmanCWCustomizedKernel<TaskType>,
                              blockPerGrid, threadsPerBlock, Args, sm_size);
    }
    gpuErrchk(cudaGetLastError());
    if (DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors) {
      gpuErrchk(cudaDeviceSynchronize());
    }
  }
};


struct AbsMaxOp
{
    template <typename T>
    __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return (fabs(b) > fabs(a)) ? b : a;
    }
};


template <>
class DeviceCollective<CUDA>{
public:
  MGARDm_CONT
  DeviceCollective(){};

  template <typename T> MGARDm_CONT
  void Sum(SIZE n, SubArray<1, T, CUDA>& v, SubArray<1, T, CUDA>& result, int queue_idx) {
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cudaStream_t stream = DeviceRuntime<CUDA>::GetQueue(queue_idx);
    bool debug = DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, v.data(), result.data(), n, stream, debug);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, v.data(), result.data(), n, stream, debug);
    DeviceRuntime<CUDA>::SyncQueue(queue_idx);
    cudaFree(d_temp_storage);
  }

  template <typename T> MGARDm_CONT
  void AbsMax(SIZE n, SubArray<1, T, CUDA>& v, SubArray<1, T, CUDA>& result, int queue_idx) {
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    AbsMaxOp absMaxOp;
    cudaStream_t stream = DeviceRuntime<CUDA>::GetQueue(queue_idx);
    bool debug = DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors;
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, v.data(), result.data(), n, absMaxOp, 0, stream, debug);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, v.data(), result.data(), n, absMaxOp, 0, stream, debug);
    DeviceRuntime<CUDA>::SyncQueue(queue_idx);
    cudaFree(d_temp_storage);
  }

 template <typename T> MGARDm_CONT
  void ScanSumInclusive(SIZE n, SubArray<1, T, CUDA>& v, SubArray<1, T, CUDA>& result, int queue_idx) {
    Byte     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cudaStream_t stream = DeviceRuntime<CUDA>::GetQueue(queue_idx);
    bool debug = DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, v.data(), result.data(), n);
    MemoryManager<CUDA>().Malloc1D(d_temp_storage, temp_storage_bytes, queue_idx);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, v.data(), result.data(), n);
    MemoryManager<CUDA>().Free(d_temp_storage);
    DeviceRuntime<CUDA>::SyncQueue(queue_idx);
  }

  template <typename T> MGARDm_CONT
  void ScanSumExclusive(SIZE n, SubArray<1, T, CUDA>& v, SubArray<1, T, CUDA>& result, int queue_idx) {
    Byte     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cudaStream_t stream = DeviceRuntime<CUDA>::GetQueue(queue_idx);
    bool debug = DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, v.data(), result.data(), n);
    MemoryManager<CUDA>().Malloc1D(d_temp_storage, temp_storage_bytes, queue_idx);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, v.data(), result.data(), n);
    MemoryManager<CUDA>().Free(d_temp_storage);
    DeviceRuntime<CUDA>::SyncQueue(queue_idx);
  }

  template <typename T> MGARDm_CONT
  void ScanSumExtended(SIZE n, SubArray<1, T, CUDA>& v, SubArray<1, T, CUDA>& result, int queue_idx) {
    Byte     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cudaStream_t stream = DeviceRuntime<CUDA>::GetQueue(queue_idx);
    bool debug = DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, v.data(), result.data()+1, n);
    MemoryManager<CUDA>().Malloc1D(d_temp_storage, temp_storage_bytes, queue_idx);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, v.data(), result.data()+1, n);
    MemoryManager<CUDA>().Free(d_temp_storage);
    T zero = 0;
    MemoryManager<CUDA>().Copy1D(result.data(), &zero, 1, queue_idx);
    DeviceRuntime<CUDA>::SyncQueue(queue_idx);
  }

  template <typename KeyT, typename ValueT> MGARDm_CONT
  void SortByKey(SIZE n, SubArray<1, KeyT, CUDA>& keys, SubArray<1, ValueT, CUDA>& values, 
                 int queue_idx) {
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cudaStream_t stream = DeviceRuntime<CUDA>::GetQueue(queue_idx);
    bool debug = DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors;
    Array<1, KeyT, CUDA> out_keys({n});
    Array<1, ValueT, CUDA> out_values({n});

    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    keys.data(), out_keys.get_dv(), 
                                    values.data(), out_values.get_dv(), n,
                                    0, sizeof(KeyT) * 8, stream, debug);
    MemoryManager<CUDA>().Malloc1D(d_temp_storage, temp_storage_bytes, queue_idx);
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    keys.data(), out_keys.get_dv(), 
                                    values.data(), out_values.get_dv(), n,
                                    0, sizeof(KeyT) * 8, stream, debug);
    MemoryManager<CUDA>().Copy1D(keys.data(), out_keys.get_dv(), n, queue_idx);
    MemoryManager<CUDA>().Copy1D(values.data(), out_values.get_dv(), n, queue_idx);
    DeviceRuntime<CUDA>::SyncQueue(queue_idx);
  }

};












}


#endif