/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "DeviceAdapter.h"
#include <CL/sycl.hpp>

#ifndef MGARD_X_DEVICE_ADAPTER_SYCL_H
#define MGARD_X_DEVICE_ADAPTER_SYCL_H

namespace mgard_x {

using LocalMemory = sycl::accessor<Byte, 1, sycl::access::mode::read_write, sycl::access::target::local>;

// Create an exception sycl::handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e) {
      std::cout << "Failure" << std::endl;
      std::terminate();
    }
  }
};

template <> struct Atomic<SYCL> {
  template <typename T> MGARDX_EXEC static T Min(T *result, T value) {
    using AtomicRef = sycl::ext::oneapi::atomic_ref<T, sycl::memory_order::relaxed, 
          sycl::memory_scope::system, sycl::access::address_space::global_space>;
    return AtomicRef(result[0]).fetch_min(value);

  }
  template <typename T> MGARDX_EXEC static T Max(T *result, T value) {
    using AtomicRef = sycl::ext::oneapi::atomic_ref<T, sycl::memory_order::relaxed, 
          sycl::memory_scope::system, sycl::access::address_space::global_space>;
    return AtomicRef(result[0]).fetch_max(value);
  }
  template <typename T> MGARDX_EXEC static T Add(T *result, T value) {
    using AtomicRef = sycl::ext::oneapi::atomic_ref<T, sycl::memory_order::relaxed, 
          sycl::memory_scope::system, sycl::access::address_space::global_space>;
    return AtomicRef(result[0]) += (value);
  }
};

template <> class DeviceSpecification<SYCL> {
public:
  MGARDX_CONT
  DeviceSpecification() {
    sycl::default_selector d_selector;
    sycl::platform d_platform(d_selector);
    std::vector<sycl::device> d_devices = d_platform.get_devices();
    NumDevices = d_devices.size();
    MaxSharedMemorySize = new int[NumDevices];
    WarpSize = new int[NumDevices];
    NumSMs = new int[NumDevices];
    ArchitectureGeneration = new int[NumDevices];
    MaxNumThreadsPerSM = new int[NumDevices];
    MaxNumThreadsPerTB = new int[NumDevices];
    AvailableMemory = new size_t[NumDevices];
    SupportCooperativeGroups = new bool[NumDevices];

    int d = 0;
    for (auto& device : d_devices) {
      MaxSharedMemorySize[d] = device.get_info<sycl::info::device::local_mem_size>();
      WarpSize[d] = 32;
      NumSMs[d] = device.get_info<sycl::info::device::max_compute_units>();
      MaxNumThreadsPerSM[d] = 1024;
      MaxNumThreadsPerTB[d] = 1024;
      SupportCooperativeGroups[d] = false;
      d++;
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
    AvailableMemory[dev_id] = 1e9;
    return AvailableMemory[dev_id];
  }

  MGARDX_CONT bool SupportCG(int dev_id) {
    return SupportCooperativeGroups[dev_id];
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
};

template <> class DeviceQueues<SYCL> {
public:
  MGARDX_CONT
  DeviceQueues() {
    sycl::default_selector d_selector;
    sycl::platform d_platform(d_selector);
    std::vector<sycl::device> d_devices = d_platform.get_devices();
    NumDevices = d_devices.size();
    queues = new sycl::queue*[NumDevices];
    for (SIZE d = 0; d < NumDevices; d++) {
      queues[d] = new sycl::queue[MGARDX_NUM_QUEUES];
      for (SIZE i = 0; i < MGARDX_NUM_QUEUES; i++) {
        queues[d][i] = sycl::queue(d_selector, exception_handler);
      }
    }
  }

  MGARDX_CONT sycl::queue GetQueue(int dev_id, SIZE queue_id) {
    return queues[dev_id][queue_id];
  }

  MGARDX_CONT void SyncQueue(int dev_id, SIZE queue_id) {
    queues[dev_id][queue_id].wait();
  }

  MGARDX_CONT void SyncAllQueues(int dev_id) {
    for (SIZE i = 0; i < MGARDX_NUM_QUEUES; i++) {
      queues[dev_id][i].wait();
    }
  }

  MGARDX_CONT
  ~DeviceQueues() {
    for (SIZE d = 0; d < NumDevices; d++) {
      delete[] queues[d];
    }
    delete[] queues;
    queues = NULL;
  }

  int NumDevices;
  sycl::queue ** queues = NULL;
};

template <> class DeviceRuntime<SYCL> {
public:
  MGARDX_CONT
  DeviceRuntime() {}

  MGARDX_CONT static int GetDeviceCount() { return DeviceSpecs.NumDevices; }

  MGARDX_CONT static void SelectDevice(SIZE dev_id) {
    curr_dev_id = dev_id;
  }

  MGARDX_CONT static sycl::queue GetQueue(SIZE queue_id) {
    return queues.GetQueue(curr_dev_id, queue_id);
  }

  MGARDX_CONT static void SyncQueue(SIZE queue_id) {
    queues.SyncQueue(curr_dev_id, queue_id);
  }

  MGARDX_CONT static void SyncAllQueues() {
    queues.SyncAllQueues(curr_dev_id);
  }

  MGARDX_CONT static void SyncDevice() {
    queues.SyncAllQueues(curr_dev_id);
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
    return 32;
  }

  template <typename FunctorType>
  MGARDX_CONT static void SetMaxDynamicSharedMemorySize(FunctorType functor,
                                                        int maxbytes) {
    // do nothing
  }

  MGARDX_CONT
  ~DeviceRuntime() {}

  static int curr_dev_id;
  static DeviceQueues<SYCL> queues;
  static bool SyncAllKernelsAndCheckErrors;
  static DeviceSpecification<SYCL> DeviceSpecs;
  static bool TimingAllKernels;
  static bool PrintKernelConfig;
};

template <> class MemoryManager<SYCL> {
public:
  MGARDX_CONT
  MemoryManager(){};

  template <typename T>
  MGARDX_CONT static void Malloc1D(T *&ptr, SIZE n, int queue_idx) {
    sycl::queue q = DeviceRuntime<SYCL>::GetQueue(queue_idx);
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    ptr = malloc_device<converted_T>(n, q);
  }

  template <typename T>
  MGARDX_CONT static void MallocND(T *&ptr, SIZE n1, SIZE n2, SIZE &ld,
                                   int queue_idx) {
    sycl::queue q = DeviceRuntime<SYCL>::GetQueue(queue_idx);
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    ptr = malloc_device<converted_T>(n1 * n2, q);
    ld = n1;
  }

  template <typename T>
  MGARDX_CONT static void MallocManaged1D(T *&ptr, SIZE n, int queue_idx) {
    sycl::queue q = DeviceRuntime<SYCL>::GetQueue(queue_idx);
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    ptr = malloc_shared<converted_T>(n, q);
  }

  template <typename T> MGARDX_CONT static void Free(T *ptr) {
    // printf("MemoryManager.Free(%llu)\n", ptr);
    sycl::queue q = DeviceRuntime<SYCL>::GetQueue(0);
    if (ptr == NULL)
      return;
    sycl::free(ptr, q);
  }

  template <typename T>
  MGARDX_CONT static void Copy1D(T *dst_ptr, const T *src_ptr, SIZE n,
                                 int queue_idx) {
    sycl::queue q = DeviceRuntime<SYCL>::GetQueue(queue_idx);
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    q.memcpy((converted_T*)dst_ptr, (converted_T*)src_ptr, n * sizeof(converted_T));
  }

  template <typename T>
  MGARDX_CONT static void CopyND(T *dst_ptr, SIZE dst_ld, const T *src_ptr,
                                 SIZE src_ld, SIZE n1, SIZE n2, int queue_idx) {
    sycl::queue q = DeviceRuntime<SYCL>::GetQueue(queue_idx);
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    q.memcpy((converted_T*)dst_ptr, (converted_T*)src_ptr, n1 * n2 * sizeof(converted_T));
  }

  template <typename T>
  MGARDX_CONT static void MallocHost(T *&ptr, SIZE n, int queue_idx) {
    sycl::queue q = DeviceRuntime<SYCL>::GetQueue(queue_idx);
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    ptr = malloc_host<converted_T>(n, q);
  }

  template <typename T> MGARDX_CONT static void FreeHost(T *ptr) {
    sycl::queue q = DeviceRuntime<SYCL>::GetQueue(0);
    if (ptr == NULL)
      return;
    sycl::free(ptr, q);
  }

  template <typename T>
  MGARDX_CONT static void Memset1D(T *ptr, SIZE n, int value, int queue_idx) {
    sycl::queue q = DeviceRuntime<SYCL>::GetQueue(queue_idx);
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    q.memset((converted_T*)ptr, value, n * sizeof(converted_T));
  }

  template <typename T>
  MGARDX_CONT static void MemsetND(T *ptr, SIZE ld, SIZE n1, SIZE n2, int value,
                                   int queue_idx) {
    sycl::queue q = DeviceRuntime<SYCL>::GetQueue(queue_idx);
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    q.memset((converted_T*)ptr, value, n1 * n2 * sizeof(converted_T));
  }

  template <typename T> MGARDX_CONT static bool IsDevicePointer(T *ptr) {
    sycl::queue q = DeviceRuntime<SYCL>::GetQueue(0);
    return sycl::get_pointer_type(ptr, q.get_context()) == sycl::usm::alloc::device;
  }

  static bool ReduceMemoryFootprint;
};

template <typename Task>
class Kernel {
public:
  Kernel(Task task, LocalMemory localAccess): task(task), localAccess(localAccess) {}
  void operator()(sycl::nd_item<3> i) const {
    Task my_task = task;
    Byte *shared_memory = localAccess.get_pointer().get();
    my_task.GetFunctor().Init(i.get_global_range(2)/i.get_group_range(2),
                           i.get_global_range(1)/i.get_group_range(1),
                           i.get_global_range(0)/i.get_group_range(0),
                           i.get_group_range(2), i.get_group_range(1), i.get_group_range(0),
                           i.get_group().get_id(2),
                           i.get_group().get_id(1),
                           i.get_group().get_id(0),
                           i.get_local_id(2), i.get_local_id(1), i.get_local_id(0),
                           shared_memory);

    my_task.GetFunctor().Operation1();
    i.barrier();
    my_task.GetFunctor().Operation2();
    i.barrier();
    my_task.GetFunctor().Operation3();
    i.barrier();
    my_task.GetFunctor().Operation4();
    i.barrier();
    my_task.GetFunctor().Operation5();
    i.barrier();
    my_task.GetFunctor().Operation6();
    i.barrier();
    my_task.GetFunctor().Operation7();
    i.barrier();
    my_task.GetFunctor().Operation8();
    i.barrier();
    my_task.GetFunctor().Operation9();
    i.barrier();
    my_task.GetFunctor().Operation10();
  }
private:
  Task task;
  LocalMemory localAccess;
};


template <typename Task>
class IterKernel {
public:
  IterKernel(Task task, LocalMemory localAccess): task(task), localAccess(localAccess) {}
  void operator()(sycl::nd_item<3> i) const {
    Byte *shared_memory = localAccess.get_pointer().get();
    task.GetFunctor().Init(i.get_global_range(2)/i.get_group_range(2),
                           i.get_global_range(1)/i.get_group_range(1),
                           i.get_global_range(0)/i.get_group_range(0),
                           i.get_group_range(2), i.get_group_range(1), i.get_group_range(0),
                           i.get_group().get_id(2),
                           i.get_group().get_id(1),
                           i.get_group().get_id(0),
                           i.get_local_id(2), i.get_local_id(1), i.get_local_id(0),
                           shared_memory);

    task.GetFunctor().Operation1();
    i.barrier();

    task.GetFunctor().Operation2();
    i.barrier();

    while (task.GetFunctor().LoopCondition1()) {
      task.GetFunctor().Operation3();
      i.barrier();
      task.GetFunctor().Operation4();
      i.barrier();
      task.GetFunctor().Operation5();
      i.barrier();
      task.GetFunctor().Operation6();
      i.barrier();
    }

    task.GetFunctor().Operation7();
    i.barrier();
    task.GetFunctor().Operation8();
    i.barrier();
    task.GetFunctor().Operation9();
    i.barrier();
    task.GetFunctor().Operation10();
    i.barrier();

    while (task.GetFunctor().LoopCondition2()) {
      task.GetFunctor().Operation11();
      i.barrier();
      task.GetFunctor().Operation12();
      i.barrier();
      task.GetFunctor().Operation13();
      i.barrier();
      task.GetFunctor().Operation14();
      i.barrier();
    }

    task.GetFunctor().Operation15();
    i.barrier();
    task.GetFunctor().Operation16();
    i.barrier();
    task.GetFunctor().Operation17();
    i.barrier();
  }
private:
  Task task;
  LocalMemory localAccess;
};

#define SINGLE_KERNEL(OPERATION)                                               \
  template <typename Task>                                                     \
  class Single_##OPERATION##_Kernel {                                          \
  public:                                                                      \
    Single_##OPERATION##_Kernel(Task task, LocalMemory localAccess):           \
      task(task), localAccess(localAccess) {}                                  \
    void operator()(sycl::nd_item<3> i) const {                                \
      Byte *shared_memory = localAccess.get_pointer().get();                   \
      task.GetFunctor().Init(i.get_global_range(2)/i.get_group_range(2),       \
                             i.get_global_range(1)/i.get_group_range(1),       \
                             i.get_global_range(0)/i.get_group_range(0),       \
                             i.get_group_range(2),                             \
                             i.get_group_range(1),                             \
                             i.get_group_range(0),                             \
                             i.get_group().get_id(2),                          \
                             i.get_group().get_id(1),                          \
                             i.get_group().get_id(0),                          \
                             i.get_local_id(2),                                \
                             i.get_local_id(1),                                \
                             i.get_local_id(0),                                \
                             shared_memory);                                   \
      task.GetFunctor().OPERATION();                                           \
  }                                                                            \
  private:                                                                     \
    Task task;                                                                 \
    LocalMemory localAccess;                                                   \
  };

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

template <typename Task>
class ParallelMergeKernel {
public:
  ParallelMergeKernel(Task task, LocalMemory localAccess): 
                    task(task), localAccess(localAccess) {}
  void operator()(sycl::nd_item<3> i) const {
    Byte *shared_memory = localAccess.get_pointer().get();
    task.GetFunctor().Init(i.get_global_range(2)/i.get_group_range(2),
                           i.get_global_range(1)/i.get_group_range(1),
                           i.get_global_range(0)/i.get_group_range(0),
                           i.get_group_range(2), i.get_group_range(1), i.get_group_range(0),
                           i.get_group().get_id(2),
                           i.get_group().get_id(1),
                           i.get_group().get_id(0),
                           i.get_local_id(2), i.get_local_id(1), i.get_local_id(0),
                           shared_memory);

    task.GetFunctor().Operation5();
    i.barrier();
    while (task.GetFunctor().LoopCondition2()) {
      task.GetFunctor().Operation6();
      i.barrier();
      task.GetFunctor().Operation7();
      i.barrier();
      task.GetFunctor().Operation8();
      i.barrier();
    }
    task.GetFunctor().Operation9();
  }
private:
  Task task;
  LocalMemory localAccess;
};

template <typename Task> void HuffmanCLCustomizedNoCGKernel(Task task) {
  // std::cout << "calling HuffmanCLCustomizedNoCGKernel\n";
  sycl::range global_threads(task.GetBlockDimX() * task.GetGridDimX(),
                              task.GetBlockDimY() * task.GetGridDimY(),
                              task.GetBlockDimZ() * task.GetGridDimZ());

  sycl::range local_threads(task.GetBlockDimX(),
                            task.GetBlockDimY(),
                            task.GetBlockDimZ());

  size_t sm_size = task.GetSharedMemorySize();

  sycl::queue q = DeviceRuntime<SYCL>::GetQueue(task.GetQueueIdx());

  // std::cout << "calling Single_Operation1_Kernel\n";
  q.submit([&](sycl::handler& h) {
        LocalMemory localAccess{sm_size, h};
        Single_Operation1_Kernel kernel(task, localAccess);
        h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
      });
  DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

  // std::cout << "calling LoopCondition1\n";
  while (task.GetFunctor().LoopCondition1()) {
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

    // std::cout << "calling Single_Operation2_Kernel\n";
    q.submit([&](sycl::handler& h) {
        LocalMemory localAccess{sm_size, h};
        Single_Operation2_Kernel kernel(task, localAccess);
        h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
      });
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

    // std::cout << "calling Single_Operation3_Kernel\n";
    q.submit([&](sycl::handler& h) {
        LocalMemory localAccess{sm_size, h};
        Single_Operation3_Kernel kernel(task, localAccess);
        h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
      });
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

    // std::cout << "calling Single_Operation4_Kernel\n";
    q.submit([&](sycl::handler& h) {
        LocalMemory localAccess{sm_size, h};
        Single_Operation4_Kernel kernel(task, localAccess);
        h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
      });
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

    // std::cout << "calling BranchCondition1\n";
    if (task.GetFunctor().BranchCondition1()) {
      DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

      // std::cout << "calling ParallelMergeKernel\n";
      q.submit([&](sycl::handler& h) {
        LocalMemory localAccess{sm_size, h};
        ParallelMergeKernel kernel(task, localAccess);
        h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
      });
      DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

      // std::cout << "calling Single_Operation10_Kernel\n";
      q.submit([&](sycl::handler& h) {
        LocalMemory localAccess{sm_size, h};
        Single_Operation10_Kernel kernel(task, localAccess);
        h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
      });
      DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());
    }

    // std::cout << "calling Single_Operation11_Kernel\n";
    q.submit([&](sycl::handler& h) {
        LocalMemory localAccess{sm_size, h};
        Single_Operation11_Kernel kernel(task, localAccess);
        h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
      });
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

    // std::cout << "calling Single_Operation12_Kernel\n";
    q.submit([&](sycl::handler& h) {
        LocalMemory localAccess{sm_size, h};
        Single_Operation12_Kernel kernel(task, localAccess);
        h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
      });
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

    // std::cout << "calling Single_Operation13_Kernel\n";
    q.submit([&](sycl::handler& h) {
        LocalMemory localAccess{sm_size, h};
        Single_Operation13_Kernel kernel(task, localAccess);
        h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
      });
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());
    // std::cout << "calling Single_Operation14_Kernel\n";
    q.submit([&](sycl::handler& h) {
        LocalMemory localAccess{sm_size, h};
        Single_Operation14_Kernel kernel(task, localAccess);
        h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
      });
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());
  }
}

template <typename Task> void HuffmanCWCustomizedNoCGKernel(Task task) {
  // std::cout << "calling HuffmanCWCustomizedNoCGKernel\n";
  sycl::range global_threads(task.GetBlockDimX() * task.GetGridDimX(),
                              task.GetBlockDimY() * task.GetGridDimY(),
                              task.GetBlockDimZ() * task.GetGridDimZ());

  sycl::range local_threads(task.GetBlockDimX(),
                            task.GetBlockDimY(),
                            task.GetBlockDimZ());

  size_t sm_size = task.GetSharedMemorySize();

  sycl::queue q = DeviceRuntime<SYCL>::GetQueue(task.GetQueueIdx());

  // std::cout << "calling Single_Operation1_Kernel\n";
  q.submit([&](sycl::handler& h) {
        LocalMemory localAccess{sm_size, h};
        Single_Operation1_Kernel kernel(task, localAccess);
        h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
      });
  DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());
  // std::cout << "calling Single_Operation2_Kernel\n";
  q.submit([&](sycl::handler& h) {
        LocalMemory localAccess{sm_size, h};
        Single_Operation2_Kernel kernel(task, localAccess);
        h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
      });
  DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());
  // std::cout << "calling Single_Operation3_Kernel\n";
  q.submit([&](sycl::handler& h) {
        LocalMemory localAccess{sm_size, h};
        Single_Operation3_Kernel kernel(task, localAccess);
        h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
      });
  DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

  // std::cout << "calling LoopCondition1\n";
  while (task.GetFunctor().LoopCondition1()) {
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

    // std::cout << "calling Single_Operation4_Kernel\n";
    q.submit([&](sycl::handler& h) {
        LocalMemory localAccess{sm_size, h};
        Single_Operation4_Kernel kernel(task, localAccess);
        h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
      });
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

    // std::cout << "calling Single_Operation5_Kernel\n";
    q.submit([&](sycl::handler& h) {
        LocalMemory localAccess{sm_size, h};
        Single_Operation5_Kernel kernel(task, localAccess);
        h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
      });
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

    // std::cout << "calling Single_Operation6_Kernel\n";
    q.submit([&](sycl::handler& h) {
        LocalMemory localAccess{sm_size, h};
        Single_Operation6_Kernel kernel(task, localAccess);
        h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
      });
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

    // std::cout << "calling Single_Operation7_Kernel\n";
    q.submit([&](sycl::handler& h) {
        LocalMemory localAccess{sm_size, h};
        Single_Operation7_Kernel kernel(task, localAccess);
        h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
      });
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

    // std::cout << "calling Single_Operation8_Kernel\n";
    q.submit([&](sycl::handler& h) {
        LocalMemory localAccess{sm_size, h};
        Single_Operation8_Kernel kernel(task, localAccess);
        h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
      });
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());
  }

  // std::cout << "calling Single_Operation9_Kernel\n";
  q.submit([&](sycl::handler& h) {
        LocalMemory localAccess{sm_size, h};
        Single_Operation9_Kernel kernel(task, localAccess);
        h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
      });
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

  // std::cout << "calling Single_Operation10_Kernel\n";
  q.submit([&](sycl::handler& h) {
        LocalMemory localAccess{sm_size, h};
        Single_Operation10_Kernel kernel(task, localAccess);
        h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
      });
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());
}


// template <typename FunctorType

template <typename TaskType> class DeviceAdapter<TaskType, SYCL> {
public:
  // inline constexpr bool sycl::is_device_copyable_v<TaskType> = true;



  MGARDX_CONT
  DeviceAdapter(){};

  MGARDX_CONT
  ExecutionReturn Execute(TaskType &task) {

    sycl::range global_threads(task.GetBlockDimX() * task.GetGridDimX(),
                              task.GetBlockDimY() * task.GetGridDimY(),
                              task.GetBlockDimZ() * task.GetGridDimZ());

    sycl::range local_threads(task.GetBlockDimX(),
                              task.GetBlockDimY(),
                              task.GetBlockDimZ());

    size_t sm_size = task.GetSharedMemorySize();

    sycl::queue q = DeviceRuntime<SYCL>::GetQueue(task.GetQueueIdx());

    if (DeviceRuntime<SYCL>::PrintKernelConfig) {
      std::cout << log::log_info << task.GetFunctorName() << ": <"
                << task.GetBlockDimX() << ", " << task.GetBlockDimY() << ", "
                << task.GetBlockDimZ() << "> <" << task.GetGridDimX() << ", "
                << task.GetGridDimY() << ", " << task.GetGridDimZ() << ">\n";
    }

    Timer timer;
    if (DeviceRuntime<SYCL>::TimingAllKernels ||
        AutoTuner<SYCL>::ProfileKernels) {
      DeviceRuntime<SYCL>::SyncDevice();
      timer.start();
    }

    // if constexpr evaluate at compile time otherwise this does not compile
    if constexpr (std::is_base_of<Functor<SYCL>,
                                  typename TaskType::Functor>::value) {
      q.submit([&](sycl::handler& h) {
        LocalMemory localAccess{sm_size, h};
        Kernel kernel(task, localAccess);
        h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
      });
    } else if constexpr (std::is_base_of<IterFunctor<SYCL>,
                                         typename TaskType::Functor>::value) {
      q.submit([&](sycl::handler& h) {
        LocalMemory localAccess{sm_size, h};
        IterKernel kernel(task, localAccess);
        h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
      });
    } else if constexpr (std::is_base_of<HuffmanCLCustomizedFunctor<SYCL>,
                                         typename TaskType::Functor>::value) {
      HuffmanCLCustomizedNoCGKernel(task);
    } else if constexpr (std::is_base_of<HuffmanCWCustomizedFunctor<SYCL>,
                                         typename TaskType::Functor>::value) {
      HuffmanCWCustomizedNoCGKernel(task);
    }
    if (DeviceRuntime<SYCL>::SyncAllKernelsAndCheckErrors) {
      DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());
    }

    ExecutionReturn ret;

    if (DeviceRuntime<SYCL>::TimingAllKernels ||
        AutoTuner<SYCL>::ProfileKernels) {
      DeviceRuntime<SYCL>::SyncDevice();
      timer.end();
      if (DeviceRuntime<SYCL>::TimingAllKernels) {
        timer.print(task.GetFunctorName());
      }
      if (AutoTuner<SYCL>::ProfileKernels) {
        ret.execution_time = timer.get();
      }
    }
    return ret;
  }
};

template <typename T>
struct AbsMaxOp {
  T operator()(const T &a, const T &b) const {
    return (fabs(b) > fabs(a)) ? fabs(b) : fabs(a);
  }
};

template <typename T>
struct SquareOp {
  T operator()(const T &a) const {
    return a * a;
  }
};

template <> class DeviceCollective<SYCL> {
public:
  MGARDX_CONT
  DeviceCollective(){};

  template <typename T>
  MGARDX_CONT static void Sum(SIZE n, SubArray<1, T, SYCL> &v,
                              SubArray<1, T, SYCL> &result, int queue_idx) {

    sycl::queue q = DeviceRuntime<SYCL>::GetQueue(queue_idx);
    q.submit([&](sycl::handler& h) {
      T * res = result.data();
      T * input = v.data();
      sycl::range global{n};
      sycl::range local{256};
      h.parallel_for(sycl::nd_range{global, local}, sycl::reduction(res, (T)0, sycl::plus<T>()), [=](sycl::nd_item<1> it, auto& res) {
        size_t i = it.get_global_id(0);
        res.combine(input[i]);
      });
    });
    DeviceRuntime<SYCL>::SyncQueue(queue_idx);
  }

  template <typename T>
  MGARDX_CONT static void AbsMax(SIZE n, SubArray<1, T, SYCL> &v,
                                 SubArray<1, T, SYCL> &result, int queue_idx) {
    sycl::queue q = DeviceRuntime<SYCL>::GetQueue(queue_idx);
    q.submit([&](sycl::handler& h) {
      T * res = result.data();
      T * input = v.data();
      sycl::range global{n};
      sycl::range local{256};
      h.parallel_for(sycl::nd_range{global, local}, sycl::reduction(res, (T)0, AbsMaxOp<T>()), [=](sycl::nd_item<1> it, auto& res) {
        size_t i = it.get_global_id(0);
        res.combine(input[i]);
      });
    });
    DeviceRuntime<SYCL>::SyncQueue(queue_idx);
  }

  template <typename T>
  MGARDX_CONT static void SquareSum(SIZE n, SubArray<1, T, SYCL> &v,
                                    SubArray<1, T, SYCL> &result,
                                    int queue_idx) {
    sycl::queue q = DeviceRuntime<SYCL>::GetQueue(queue_idx);
    q.submit([&](sycl::handler& h) {
      T * res = result.data();
      T * input = v.data();
      sycl::range global{n};
      sycl::range local{256};
      h.parallel_for(sycl::nd_range{global, local}, sycl::reduction(res, (T)0, SquareOp<T>()), [=](sycl::nd_item<1> it, auto& res) {
        size_t i = it.get_global_id(0);
        res.combine(input[i]);
      });
    });
    DeviceRuntime<SYCL>::SyncQueue(queue_idx);
  }

  template <typename T>
  MGARDX_CONT static void ScanSumInclusive(SIZE n, SubArray<1, T, SYCL> &v,
                                           SubArray<1, T, SYCL> &result,
                                           int queue_idx) {
  }

  template <typename T>
  MGARDX_CONT static void ScanSumExclusive(SIZE n, SubArray<1, T, SYCL> &v,
                                           SubArray<1, T, SYCL> &result,
                                           int queue_idx) {
  }

  template <typename T>
  MGARDX_CONT static void ScanSumExtended(SIZE n, SubArray<1, T, SYCL> &v,
                                          SubArray<1, T, SYCL> &result,
                                          int queue_idx) {
  }

  template <typename KeyT, typename ValueT>
  MGARDX_CONT static void SortByKey(SIZE n, SubArray<1, KeyT, SYCL> &keys,
                                    SubArray<1, ValueT, SYCL> &values,
                                    int queue_idx) {
  }

  template <typename KeyT, typename ValueT, typename BinaryOpType>
  MGARDX_CONT static void
  ScanOpInclusiveByKey(SubArray<1, SIZE, SYCL> &key,
                       SubArray<1, ValueT, SYCL> &v,
                       SubArray<1, ValueT, SYCL> &result, int queue_idx) {
  }
};


}
#endif