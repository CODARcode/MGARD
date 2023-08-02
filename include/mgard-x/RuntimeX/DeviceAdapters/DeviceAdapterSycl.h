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

#define MGARD_X_SYCL_ND_RANGE_3D 0
#define MGARD_X_SYCL_ND_RANGE_1D 1

namespace mgard_x {

using LocalMemory = sycl::accessor<Byte, 1, sycl::access::mode::read_write,
                                   sycl::access::target::local>;

// Create an exception sycl::handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    } catch (std::exception const &e) {
      std::cout << "Failure" << std::endl;
      std::terminate();
    }
  }
};

template <typename T, OPTION MemoryType, OPTION Scope>
struct Atomic<T, MemoryType, Scope, SYCL> {
  MGARDX_EXEC static T Min(T *result, T value) {
    if constexpr (MemoryType == AtomicGlobalMemory) {
      if constexpr (Scope == AtomicSystemScope) {
        using AtomicRef = sycl::ext::oneapi::atomic_ref<
            T, sycl::memory_order::relaxed, sycl::memory_scope::system,
            sycl::access::address_space::global_space>;
        return AtomicRef(result[0]).fetch_min(value);
      } else if constexpr (Scope == AtomicDeviceScope) {
        using AtomicRef = sycl::ext::oneapi::atomic_ref<
            T, sycl::memory_order::relaxed, sycl::memory_scope::device,
            sycl::access::address_space::global_space>;
        return AtomicRef(result[0]).fetch_min(value);
      } else {
        using AtomicRef = sycl::ext::oneapi::atomic_ref<
            T, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
            sycl::access::address_space::global_space>;
        return AtomicRef(result[0]).fetch_min(value);
      }
    } else {
      if constexpr (Scope == AtomicSystemScope) {
        using AtomicRef = sycl::ext::oneapi::atomic_ref<
            T, sycl::memory_order::relaxed, sycl::memory_scope::system,
            sycl::access::address_space::local_space>;
        return AtomicRef(result[0]).fetch_min(value);
      } else if constexpr (Scope == AtomicDeviceScope) {
        using AtomicRef = sycl::ext::oneapi::atomic_ref<
            T, sycl::memory_order::relaxed, sycl::memory_scope::device,
            sycl::access::address_space::local_space>;
        return AtomicRef(result[0]).fetch_min(value);
      } else {
        using AtomicRef = sycl::ext::oneapi::atomic_ref<
            T, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
            sycl::access::address_space::local_space>;
        return AtomicRef(result[0]).fetch_min(value);
      }
    }
  }
  MGARDX_EXEC static T Max(T *result, T value) {
    if constexpr (MemoryType == AtomicGlobalMemory) {
      if constexpr (Scope == AtomicSystemScope) {
        using AtomicRef = sycl::ext::oneapi::atomic_ref<
            T, sycl::memory_order::relaxed, sycl::memory_scope::system,
            sycl::access::address_space::global_space>;
        return AtomicRef(result[0]).fetch_max(value);
      } else if constexpr (Scope == AtomicDeviceScope) {
        using AtomicRef = sycl::ext::oneapi::atomic_ref<
            T, sycl::memory_order::relaxed, sycl::memory_scope::device,
            sycl::access::address_space::global_space>;
        return AtomicRef(result[0]).fetch_max(value);
      } else {
        using AtomicRef = sycl::ext::oneapi::atomic_ref<
            T, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
            sycl::access::address_space::global_space>;
        return AtomicRef(result[0]).fetch_max(value);
      }
    } else {
      if constexpr (Scope == AtomicSystemScope) {
        using AtomicRef = sycl::ext::oneapi::atomic_ref<
            T, sycl::memory_order::relaxed, sycl::memory_scope::system,
            sycl::access::address_space::local_space>;
        return AtomicRef(result[0]).fetch_max(value);
      } else if constexpr (Scope == AtomicDeviceScope) {
        using AtomicRef = sycl::ext::oneapi::atomic_ref<
            T, sycl::memory_order::relaxed, sycl::memory_scope::device,
            sycl::access::address_space::local_space>;
        return AtomicRef(result[0]).fetch_max(value);
      } else {
        using AtomicRef = sycl::ext::oneapi::atomic_ref<
            T, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
            sycl::access::address_space::local_space>;
        return AtomicRef(result[0]).fetch_max(value);
      }
    }
  }
  MGARDX_EXEC static T Add(T *result, T value) {
    if constexpr (MemoryType == AtomicGlobalMemory) {
      if constexpr (Scope == AtomicSystemScope) {
        using AtomicRef = sycl::ext::oneapi::atomic_ref<
            T, sycl::memory_order::relaxed, sycl::memory_scope::system,
            sycl::access::address_space::global_space>;
        T new_value = AtomicRef(result[0]) += value;
        return new_value - value;
      } else if constexpr (Scope == AtomicDeviceScope) {
        using AtomicRef = sycl::ext::oneapi::atomic_ref<
            T, sycl::memory_order::relaxed, sycl::memory_scope::device,
            sycl::access::address_space::global_space>;
        T new_value = AtomicRef(result[0]) += value;
        return new_value - value;
      } else {
        using AtomicRef = sycl::ext::oneapi::atomic_ref<
            T, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
            sycl::access::address_space::global_space>;
        T new_value = AtomicRef(result[0]) += value;
        return new_value - value;
      }
    } else {
      if constexpr (Scope == AtomicSystemScope) {
        using AtomicRef = sycl::ext::oneapi::atomic_ref<
            T, sycl::memory_order::relaxed, sycl::memory_scope::system,
            sycl::access::address_space::local_space>;
        T new_value = AtomicRef(result[0]) += value;
        return new_value - value;
      } else if constexpr (Scope == AtomicDeviceScope) {
        using AtomicRef = sycl::ext::oneapi::atomic_ref<
            T, sycl::memory_order::relaxed, sycl::memory_scope::device,
            sycl::access::address_space::local_space>;
        T new_value = AtomicRef(result[0]) += value;
        return new_value - value;
      } else {
        using AtomicRef = sycl::ext::oneapi::atomic_ref<
            T, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
            sycl::access::address_space::local_space>;
        T new_value = AtomicRef(result[0]) += value;
        return new_value - value;
      }
    }
  }
};

template <> struct Math<SYCL> {
  template <typename T> MGARDX_EXEC static T Min(T a, T b) {
    if constexpr (std::is_integral<T>::value)
      return sycl::min(a, b);
    else {
      return sycl::fmin(a, b);
    }
  }
  template <typename T> MGARDX_EXEC static T Max(T a, T b) {
    if constexpr (std::is_integral<T>::value)
      return sycl::max(a, b);
    else {
      return sycl::fmax(a, b);
    }
  }
  MGARDX_EXEC static int ffs(unsigned int a) {
    int pos = 0;
    if (a == 0)
      return pos;
    while (!(a & 1)) {
      a >>= 1;
      ++pos;
    }
    return pos + 1;
  }
  MGARDX_EXEC static int ffsll(long long unsigned int a) {
    int pos = 0;
    if (a == 0)
      return pos;
    while (!(a & 1)) {
      a >>= 1;
      ++pos;
    }
    return pos + 1;
  }
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
    DeviceNames = new std::string[NumDevices];

    int d = 0;
    for (auto &device : d_devices) {
      MaxSharedMemorySize[d] =
          device.get_info<sycl::info::device::local_mem_size>();
      WarpSize[d] = MGARDX_WARP_SIZE;
      NumSMs[d] = device.get_info<sycl::info::device::max_compute_units>();
      MaxNumThreadsPerSM[d] =
          device.get_info<sycl::info::device::max_work_group_size>();
      // Larger limit can cause resource insufficient error
      MaxNumThreadsPerTB[d] = std::min(
          1024ul, device.get_info<sycl::info::device::max_work_group_size>());
      AvailableMemory[d] =
          device.get_info<sycl::info::device::global_mem_size>();
      SupportCooperativeGroups[d] = false;
      DeviceNames[d] = std::string(device.get_info<sycl::info::device::name>());
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

template <> class DeviceQueues<SYCL> {
public:
  MGARDX_CONT
  void Initialize() {
    if (!initialized) {
      log::dbg("Calling DeviceQueues<SYCL>::Initialize");
      sycl::default_selector d_selector;
      sycl::platform d_platform(d_selector);
      std::vector<sycl::device> d_devices = d_platform.get_devices();
      NumDevices = d_devices.size();
      queues = new sycl::queue *[NumDevices];
      for (SIZE d = 0; d < NumDevices; d++) {
        queues[d] = new sycl::queue[MGARDX_NUM_QUEUES];
        for (SIZE i = 0; i < MGARDX_NUM_QUEUES; i++) {
          queues[d][i] = sycl::queue(d_devices[d], exception_handler);
        }
      }
      initialized = true;
    }
  }

  MGARDX_CONT
  void Destroy() {
    if (initialized) {
      log::dbg("Calling DeviceQueues<SYCL>::Destroy");
      for (SIZE d = 0; d < NumDevices; d++) {
        delete[] queues[d];
      }
      delete[] queues;
      queues = nullptr;
      initialized = false;
    }
  }

  MGARDX_CONT
  DeviceQueues() { Initialize(); }

  MGARDX_CONT sycl::queue GetQueue(int dev_id, SIZE queue_id) {
    Initialize();
    return queues[dev_id][queue_id];
  }

  MGARDX_CONT void SyncQueue(int dev_id, SIZE queue_id) {
    Initialize();
    queues[dev_id][queue_id].wait();
  }

  MGARDX_CONT void SyncAllQueues(int dev_id) {
    for (SIZE i = 0; i < MGARDX_NUM_QUEUES; i++) {
      Initialize();
      queues[dev_id][i].wait();
    }
  }

  MGARDX_CONT
  ~DeviceQueues() {}

  int initialized = false;
  int NumDevices;
  sycl::queue **queues = nullptr;
};

extern int sycl_dev_id;
#pragma omp threadprivate(sycl_dev_id)

template <> class DeviceRuntime<SYCL> {
public:
  MGARDX_CONT
  DeviceRuntime() {}

  MGARDX_CONT static void Initialize() { queues.Initialize(); }

  MGARDX_CONT static void Finalize() { queues.Destroy(); }

  MGARDX_CONT static int GetDeviceCount() { return DeviceSpecs.NumDevices; }

  MGARDX_CONT static void SelectDevice(SIZE dev_id) { sycl_dev_id = dev_id; }

  MGARDX_CONT static int GetDevice() { return sycl_dev_id; }

  MGARDX_CONT static sycl::queue GetQueue(SIZE queue_id) {
    return queues.GetQueue(sycl_dev_id, queue_id);
  }

  MGARDX_CONT static void SyncQueue(SIZE queue_id) {
    queues.SyncQueue(sycl_dev_id, queue_id);
  }

  MGARDX_CONT static void SyncAllQueues() { queues.SyncAllQueues(sycl_dev_id); }

  MGARDX_CONT static void SyncDevice() { queues.SyncAllQueues(sycl_dev_id); }

  MGARDX_CONT static std::string GetDeviceName() {
    return DeviceSpecs.GetDeviceName(sycl_dev_id);
  }

  MGARDX_CONT static int GetMaxSharedMemorySize() {
    return DeviceSpecs.GetMaxSharedMemorySize(sycl_dev_id);
  }

  MGARDX_CONT static int GetWarpSize() {
    return DeviceSpecs.GetWarpSize(sycl_dev_id);
  }

  MGARDX_CONT static int GetNumSMs() {
    return DeviceSpecs.GetNumSMs(sycl_dev_id);
  }

  MGARDX_CONT static int GetArchitectureGeneration() {
    return DeviceSpecs.GetArchitectureGeneration(sycl_dev_id);
  }

  MGARDX_CONT static int GetMaxNumThreadsPerSM() {
    return DeviceSpecs.GetMaxNumThreadsPerSM(sycl_dev_id);
  }

  MGARDX_CONT static int GetMaxNumThreadsPerTB() {
    return DeviceSpecs.GetMaxNumThreadsPerTB(sycl_dev_id);
  }

  MGARDX_CONT static size_t GetAvailableMemory() {
    return DeviceSpecs.GetAvailableMemory(sycl_dev_id);
  }

  MGARDX_CONT static bool SupportCG() {
    return DeviceSpecs.SupportCG(sycl_dev_id);
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
  MGARDX_CONT static void Malloc1D(T *&ptr, SIZE n,
                                   int queue_idx = MGARDX_SYNCHRONIZED_QUEUE) {
    log::dbg("Calling MemoryManager<SYCL>::Malloc1D");
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<SYCL>::SyncQueue(queue_idx);
    }
    sycl::queue q = DeviceRuntime<SYCL>::GetQueue(queue_idx);
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    ptr = malloc_device<converted_T>(n, q);
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<SYCL>::SyncQueue(queue_idx);
    }
    if (DeviceRuntime<SYCL>::SyncAllKernelsAndCheckErrors) {
      DeviceRuntime<SYCL>::SyncDevice();
    }
  }

  template <typename T>
  MGARDX_CONT static void MallocND(T *&ptr, SIZE n1, SIZE n2, SIZE &ld,
                                   int queue_idx = MGARDX_SYNCHRONIZED_QUEUE) {
    log::dbg("Calling MemoryManager<SYCL>::MallocND");
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<SYCL>::SyncQueue(queue_idx);
    }
    sycl::queue q = DeviceRuntime<SYCL>::GetQueue(queue_idx);
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    ptr = malloc_device<converted_T>(n1 * n2, q);
    ld = n1;
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<SYCL>::SyncQueue(queue_idx);
    }
    if (DeviceRuntime<SYCL>::SyncAllKernelsAndCheckErrors) {
      DeviceRuntime<SYCL>::SyncDevice();
    }
  }

  template <typename T>
  MGARDX_CONT static void
  MallocManaged1D(T *&ptr, SIZE n, int queue_idx = MGARDX_SYNCHRONIZED_QUEUE) {
    log::dbg("Calling MemoryManager<SYCL>::MallocManaged1D");
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<SYCL>::SyncQueue(queue_idx);
    }
    sycl::queue q = DeviceRuntime<SYCL>::GetQueue(queue_idx);
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    ptr = malloc_shared<converted_T>(n, q);
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<SYCL>::SyncQueue(queue_idx);
    }
    if (DeviceRuntime<SYCL>::SyncAllKernelsAndCheckErrors) {
      DeviceRuntime<SYCL>::SyncDevice();
    }
  }

  template <typename T>
  MGARDX_CONT static void Free(T *ptr,
                               int queue_idx = MGARDX_SYNCHRONIZED_QUEUE) {
    log::dbg("Calling MemoryManager<SYCL>::Free");
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<SYCL>::SyncQueue(queue_idx);
    }
    sycl::queue q = DeviceRuntime<SYCL>::GetQueue(queue_idx);
    if (ptr == nullptr)
      return;
    sycl::free(ptr, q);
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<SYCL>::SyncQueue(queue_idx);
    }
    if (DeviceRuntime<SYCL>::SyncAllKernelsAndCheckErrors) {
      DeviceRuntime<SYCL>::SyncDevice();
    }
  }

  template <typename T>
  MGARDX_CONT static void Copy1D(T *dst_ptr, const T *src_ptr, SIZE n,
                                 int queue_idx = MGARDX_SYNCHRONIZED_QUEUE) {
    log::dbg("Calling MemoryManager<SYCL>::Copy1D");
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<SYCL>::SyncQueue(queue_idx);
    }
    sycl::queue q = DeviceRuntime<SYCL>::GetQueue(queue_idx);
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    q.memcpy((converted_T *)dst_ptr, (converted_T *)src_ptr,
             n * sizeof(converted_T));
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<SYCL>::SyncQueue(queue_idx);
    }
    if (DeviceRuntime<SYCL>::SyncAllKernelsAndCheckErrors) {
      DeviceRuntime<SYCL>::SyncDevice();
    }
  }

  template <typename T>
  MGARDX_CONT static void CopyND(T *dst_ptr, SIZE dst_ld, const T *src_ptr,
                                 SIZE src_ld, SIZE n1, SIZE n2,
                                 int queue_idx = MGARDX_SYNCHRONIZED_QUEUE) {
    log::dbg("Calling MemoryManager<SYCL>::CopyND");
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<SYCL>::SyncQueue(queue_idx);
    }
    sycl::queue q = DeviceRuntime<SYCL>::GetQueue(queue_idx);
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    q.memcpy((converted_T *)dst_ptr, (converted_T *)src_ptr,
             n1 * n2 * sizeof(converted_T));
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<SYCL>::SyncQueue(queue_idx);
    }
    if (DeviceRuntime<SYCL>::SyncAllKernelsAndCheckErrors) {
      DeviceRuntime<SYCL>::SyncDevice();
    }
  }

  template <typename T>
  MGARDX_CONT static void
  MallocHost(T *&ptr, SIZE n, int queue_idx = MGARDX_SYNCHRONIZED_QUEUE) {
    log::dbg("Calling MemoryManager<SYCL>::MallocHost");
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<SYCL>::SyncQueue(queue_idx);
    }
    sycl::queue q = DeviceRuntime<SYCL>::GetQueue(queue_idx);
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    ptr = malloc_host<converted_T>(n, q);
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<SYCL>::SyncQueue(queue_idx);
    }
    if (DeviceRuntime<SYCL>::SyncAllKernelsAndCheckErrors) {
      DeviceRuntime<SYCL>::SyncDevice();
    }
  }

  template <typename T>
  MGARDX_CONT static void FreeHost(T *ptr,
                                   int queue_idx = MGARDX_SYNCHRONIZED_QUEUE) {
    log::dbg("Calling MemoryManager<SYCL>::FreeHost");
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<SYCL>::SyncQueue(queue_idx);
    }
    sycl::queue q = DeviceRuntime<SYCL>::GetQueue(queue_idx);
    if (ptr == nullptr)
      return;
    sycl::free(ptr, q);
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<SYCL>::SyncQueue(queue_idx);
    }
    if (DeviceRuntime<SYCL>::SyncAllKernelsAndCheckErrors) {
      DeviceRuntime<SYCL>::SyncDevice();
    }
  }

  template <typename T>
  MGARDX_CONT static void Memset1D(T *ptr, SIZE n, int value,
                                   int queue_idx = MGARDX_SYNCHRONIZED_QUEUE) {
    log::dbg("Calling MemoryManager<SYCL>::Memset1D");
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<SYCL>::SyncQueue(queue_idx);
    }
    sycl::queue q = DeviceRuntime<SYCL>::GetQueue(queue_idx);
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    q.memset((converted_T *)ptr, value, n * sizeof(converted_T));
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<SYCL>::SyncQueue(queue_idx);
    }
    if (DeviceRuntime<SYCL>::SyncAllKernelsAndCheckErrors) {
      DeviceRuntime<SYCL>::SyncDevice();
    }
  }

  template <typename T>
  MGARDX_CONT static void MemsetND(T *ptr, SIZE ld, SIZE n1, SIZE n2, int value,
                                   int queue_idx = MGARDX_SYNCHRONIZED_QUEUE) {
    log::dbg("Calling MemoryManager<SYCL>::MemsetND");
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<SYCL>::SyncQueue(queue_idx);
    }
    sycl::queue q = DeviceRuntime<SYCL>::GetQueue(queue_idx);
    using converted_T =
        typename std::conditional<std::is_same<T, void>::value, Byte, T>::type;
    q.memset((converted_T *)ptr, value, n1 * n2 * sizeof(converted_T));
    if (queue_idx == MGARDX_SYNCHRONIZED_QUEUE) {
      DeviceRuntime<SYCL>::SyncQueue(queue_idx);
    }
    if (DeviceRuntime<SYCL>::SyncAllKernelsAndCheckErrors) {
      DeviceRuntime<SYCL>::SyncDevice();
    }
  }

  template <typename T> MGARDX_CONT static bool IsDevicePointer(T *ptr) {
    log::dbg("Calling MemoryManager<SYCL>::IsDevicePointer");
    for (int i = 0; i < DeviceRuntime<SYCL>::GetDeviceCount(); i++) {
      DeviceRuntime<SYCL>::SelectDevice(i);
      for (int j = 0; j < MGARDX_NUM_QUEUES; j++) {
        sycl::queue q = DeviceRuntime<SYCL>::GetQueue(j);
        if (sycl::get_pointer_type(ptr, q.get_context()) ==
            sycl::usm::alloc::device) {
          return true;
        }
      }
    }
    return false;
  }

  template <typename T> MGARDX_CONT static int GetPointerDevice(T *ptr) {
    log::dbg("Calling MemoryManager<SYCL>::GetPointerDevice");
    sycl::default_selector d_selector;
    sycl::platform d_platform(d_selector);
    std::vector<sycl::device> d_devices = d_platform.get_devices();
    for (int i = 0; i < DeviceRuntime<SYCL>::GetDeviceCount(); i++) {
      DeviceRuntime<SYCL>::SelectDevice(i);
      for (int j = 0; j < MGARDX_NUM_QUEUES; j++) {
        sycl::queue q = DeviceRuntime<SYCL>::GetQueue(j);
        sycl::device ptr_device =
            sycl::get_pointer_device(ptr, q.get_context());
        if (d_devices[i] == ptr_device) {
          return i;
        }
      }
    }
    return 0;
  }

  template <typename T> MGARDX_CONT static bool CheckHostRegister(T *ptr) {
    log::dbg("Calling MemoryManager<SYCL>::CheckHostRegister");
    return false;
  }

  template <typename T> MGARDX_CONT static void HostRegister(T *ptr, SIZE n) {
    log::dbg("Calling MemoryManager<SYCL>::HostRegister");
  }

  template <typename T> MGARDX_CONT static void HostUnregister(T *ptr) {
    log::dbg("Calling MemoryManager<SYCL>::HostUnregister");
  }

  static bool ReduceMemoryFootprint;
};

template <typename FunctorType> class SyclKernel {
public:
  SyclKernel(FunctorType functor, LocalMemory localAccess, THREAD_IDX ngridz,
             THREAD_IDX ngridy, THREAD_IDX ngridx, THREAD_IDX nblockz,
             THREAD_IDX nblocky, THREAD_IDX nblockx)
      : functor(functor), localAccess(localAccess), ngridz(ngridz),
        ngridy(ngridy), ngridx(ngridx), nblockz(nblockz), nblocky(nblocky),
        nblockx(nblockx) {}
  void operator()(sycl::nd_item<3> i) const {
    FunctorType my_functor = functor;
    sycl::local_ptr<Byte> l_ptr = localAccess.get_pointer();
    Byte *shared_memory = l_ptr.get();
#if MGARD_X_SYCL_ND_RANGE_3D
    my_functor.Init(
        i.get_group_range(2), i.get_group_range(1), i.get_group_range(0),
        i.get_global_range(2) / i.get_group_range(2),
        i.get_global_range(1) / i.get_group_range(1),
        i.get_global_range(0) / i.get_group_range(0), i.get_group().get_id(2),
        i.get_group().get_id(1), i.get_group().get_id(0), i.get_local_id(2),
        i.get_local_id(1), i.get_local_id(0), shared_memory);
#endif
#if MGARD_X_SYCL_ND_RANGE_1D
    my_functor.Init(ngridz, ngridy, ngridx, nblockz, nblocky, nblockx,
                    (i.get_group().get_group_id(0) / ngridx) / ngridy,
                    (i.get_group().get_group_id(0) / ngridx) % ngridy,
                    i.get_group().get_group_id(0) % ngridx,
                    (i.get_local_id(0) / nblockx) / nblocky,
                    (i.get_local_id(0) / nblockx) % nblocky,
                    i.get_local_id(0) % nblockx, shared_memory);
#endif
    my_functor.Operation1();
    i.barrier();
    my_functor.Operation2();
    i.barrier();
    my_functor.Operation3();
    i.barrier();
    my_functor.Operation4();
    i.barrier();
    my_functor.Operation5();
    i.barrier();
    my_functor.Operation6();
    i.barrier();
    my_functor.Operation7();
    i.barrier();
    my_functor.Operation8();
    i.barrier();
    my_functor.Operation9();
    i.barrier();
    my_functor.Operation10();
    i.barrier();
    my_functor.Operation11();
    i.barrier();
    my_functor.Operation12();
    i.barrier();
    my_functor.Operation13();
    i.barrier();
    my_functor.Operation14();
    i.barrier();
    my_functor.Operation15();
    i.barrier();
    my_functor.Operation16();
    i.barrier();
    my_functor.Operation17();
    i.barrier();
    my_functor.Operation18();
    i.barrier();
    my_functor.Operation19();
    i.barrier();
    my_functor.Operation20();
  }

private:
  FunctorType functor;
  LocalMemory localAccess;
  THREAD_IDX ngridz;
  THREAD_IDX ngridy;
  THREAD_IDX ngridx;
  THREAD_IDX nblockz;
  THREAD_IDX nblocky;
  THREAD_IDX nblockx;
};

template <typename FunctorType> class SyclIterKernel {
public:
  SyclIterKernel(FunctorType functor, LocalMemory localAccess,
                 THREAD_IDX ngridz, THREAD_IDX ngridy, THREAD_IDX ngridx,
                 THREAD_IDX nblockz, THREAD_IDX nblocky, THREAD_IDX nblockx)
      : functor(functor), localAccess(localAccess), ngridz(ngridz),
        ngridy(ngridy), ngridx(ngridx), nblockz(nblockz), nblocky(nblocky),
        nblockx(nblockx) {}
  void operator()(sycl::nd_item<3> i) const {
    FunctorType my_functor = functor;
    Byte *shared_memory = localAccess.get_pointer().get();
#if MGARD_X_SYCL_ND_RANGE_3D
    my_functor.Init(
        i.get_group_range(2), i.get_group_range(1), i.get_group_range(0),
        i.get_global_range(2) / i.get_group_range(2),
        i.get_global_range(1) / i.get_group_range(1),
        i.get_global_range(0) / i.get_group_range(0), i.get_group().get_id(2),
        i.get_group().get_id(1), i.get_group().get_id(0), i.get_local_id(2),
        i.get_local_id(1), i.get_local_id(0), shared_memory);
#endif
#if MGARD_X_SYCL_ND_RANGE_1D
    my_functor.Init(ngridz, ngridy, ngridx, nblockz, nblocky, nblockx,
                    (i.get_group().get_group_id(0) / ngridx) / ngridy,
                    (i.get_group().get_group_id(0) / ngridx) % ngridy,
                    i.get_group().get_group_id(0) % ngridx,
                    (i.get_local_id(0) / nblockx) / nblocky,
                    (i.get_local_id(0) / nblockx) % nblocky,
                    i.get_local_id(0) % nblockx, shared_memory);
#endif

    my_functor.Operation1();
    i.barrier();

    my_functor.Operation2();
    i.barrier();

    while (my_functor.LoopCondition1()) {
      my_functor.Operation3();
      i.barrier();
      my_functor.Operation4();
      i.barrier();
      my_functor.Operation5();
      i.barrier();
      my_functor.Operation6();
      i.barrier();
    }

    my_functor.Operation7();
    i.barrier();
    my_functor.Operation8();
    i.barrier();
    my_functor.Operation9();
    i.barrier();
    my_functor.Operation10();
    i.barrier();

    while (my_functor.LoopCondition2()) {
      my_functor.Operation11();
      i.barrier();
      my_functor.Operation12();
      i.barrier();
      my_functor.Operation13();
      i.barrier();
      my_functor.Operation14();
      i.barrier();
    }

    my_functor.Operation15();
    i.barrier();
    my_functor.Operation16();
    i.barrier();
    my_functor.Operation17();
    i.barrier();
  }

private:
  FunctorType functor;
  LocalMemory localAccess;
  THREAD_IDX ngridz;
  THREAD_IDX ngridy;
  THREAD_IDX ngridx;
  THREAD_IDX nblockz;
  THREAD_IDX nblocky;
  THREAD_IDX nblockx;
};

#define SINGLE_KERNEL_3D(OPERATION)                                            \
  template <typename FunctorType> class Single_##OPERATION##_Kernel_3D {       \
  public:                                                                      \
    Single_##OPERATION##_Kernel_3D(FunctorType functor,                        \
                                   LocalMemory localAccess)                    \
        : functor(functor), localAccess(localAccess) {}                        \
    void operator()(sycl::nd_item<3> i) const {                                \
      FunctorType my_functor = functor;                                        \
      Byte *shared_memory = localAccess.get_pointer().get();                   \
      my_functor.Init(i.get_group_range(2), i.get_group_range(1),              \
                      i.get_group_range(0),                                    \
                      i.get_global_range(2) / i.get_group_range(2),            \
                      i.get_global_range(1) / i.get_group_range(1),            \
                      i.get_global_range(0) / i.get_group_range(0),            \
                      i.get_group().get_id(2), i.get_group().get_id(1),        \
                      i.get_group().get_id(0), i.get_local_id(2),              \
                      i.get_local_id(1), i.get_local_id(0), shared_memory);    \
      my_functor.OPERATION();                                                  \
      i.barrier();                                                             \
    }                                                                          \
                                                                               \
  private:                                                                     \
    FunctorType functor;                                                       \
    LocalMemory localAccess;                                                   \
  };

SINGLE_KERNEL_3D(Operation1);
SINGLE_KERNEL_3D(Operation2);
SINGLE_KERNEL_3D(Operation3);
SINGLE_KERNEL_3D(Operation4);
SINGLE_KERNEL_3D(Operation5);
SINGLE_KERNEL_3D(Operation6);
SINGLE_KERNEL_3D(Operation7);
SINGLE_KERNEL_3D(Operation8);
SINGLE_KERNEL_3D(Operation9);
SINGLE_KERNEL_3D(Operation10);
SINGLE_KERNEL_3D(Operation11);
SINGLE_KERNEL_3D(Operation12);
SINGLE_KERNEL_3D(Operation13);
SINGLE_KERNEL_3D(Operation14);
SINGLE_KERNEL_3D(Operation15);

#undef SINGLE_KERNEL_3D

#define SINGLE_KERNEL_1D(OPERATION)                                            \
  template <typename FunctorType> class Single_##OPERATION##_Kernel_1D {       \
  public:                                                                      \
    Single_##OPERATION##_Kernel_1D(FunctorType functor,                        \
                                   LocalMemory localAccess, THREAD_IDX ngridz, \
                                   THREAD_IDX ngridy, THREAD_IDX ngridx,       \
                                   THREAD_IDX nblockz, THREAD_IDX nblocky,     \
                                   THREAD_IDX nblockx)                         \
        : functor(functor), localAccess(localAccess), ngridz(ngridz),          \
          ngridy(ngridy), ngridx(ngridx), nblockz(nblockz), nblocky(nblocky),  \
          nblockx(nblockx) {}                                                  \
    void operator()(sycl::nd_item<3> i) const {                                \
      FunctorType my_functor = functor;                                        \
      Byte *shared_memory = localAccess.get_pointer().get();                   \
      my_functor.Init(ngridz, ngridy, ngridx, nblockz, nblocky, nblockx,       \
                      (i.get_group().get_group_id(0) / ngridx) / ngridy,       \
                      (i.get_group().get_group_id(0) / ngridx) % ngridy,       \
                      i.get_group().get_group_id(0) % ngridx,                  \
                      (i.get_local_id(0) / nblockx) / nblocky,                 \
                      (i.get_local_id(0) / nblockx) % nblocky,                 \
                      i.get_local_id(0) % nblockx, shared_memory);             \
      my_functor.OPERATION();                                                  \
      i.barrier();                                                             \
    }                                                                          \
                                                                               \
  private:                                                                     \
    FunctorType functor;                                                       \
    LocalMemory localAccess;                                                   \
    THREAD_IDX ngridz;                                                         \
    THREAD_IDX ngridy;                                                         \
    THREAD_IDX ngridx;                                                         \
    THREAD_IDX nblockz;                                                        \
    THREAD_IDX nblocky;                                                        \
    THREAD_IDX nblockx;                                                        \
  };

SINGLE_KERNEL_1D(Operation1);
SINGLE_KERNEL_1D(Operation2);
SINGLE_KERNEL_1D(Operation3);
SINGLE_KERNEL_1D(Operation4);
SINGLE_KERNEL_1D(Operation5);
SINGLE_KERNEL_1D(Operation6);
SINGLE_KERNEL_1D(Operation7);
SINGLE_KERNEL_1D(Operation8);
SINGLE_KERNEL_1D(Operation9);
SINGLE_KERNEL_1D(Operation10);
SINGLE_KERNEL_1D(Operation11);
SINGLE_KERNEL_1D(Operation12);
SINGLE_KERNEL_1D(Operation13);
SINGLE_KERNEL_1D(Operation14);
SINGLE_KERNEL_1D(Operation15);

#undef SINGLE_KERNEL_1D

template <typename FunctorType> class SyclParallelMergeKernel {
public:
  SyclParallelMergeKernel(FunctorType functor, LocalMemory localAccess,
                          THREAD_IDX ngridz, THREAD_IDX ngridy,
                          THREAD_IDX ngridx, THREAD_IDX nblockz,
                          THREAD_IDX nblocky, THREAD_IDX nblockx)
      : functor(functor), localAccess(localAccess), ngridz(ngridz),
        ngridy(ngridy), ngridx(ngridx), nblockz(nblockz), nblocky(nblocky),
        nblockx(nblockx) {}
  void operator()(sycl::nd_item<3> i) const {
    FunctorType my_functor = functor;
    Byte *shared_memory = localAccess.get_pointer().get();
#if MGARD_X_SYCL_ND_RANGE_3D
    my_functor.Init(
        i.get_group_range(2), i.get_group_range(1), i.get_group_range(0),
        i.get_global_range(2) / i.get_group_range(2),
        i.get_global_range(1) / i.get_group_range(1),
        i.get_global_range(0) / i.get_group_range(0), i.get_group().get_id(2),
        i.get_group().get_id(1), i.get_group().get_id(0), i.get_local_id(2),
        i.get_local_id(1), i.get_local_id(0), shared_memory);
#endif
#if MGARD_X_SYCL_ND_RANGE_1D
    my_functor.Init(ngridz, ngridy, ngridx, nblockz, nblocky, nblockx,
                    (i.get_group().get_group_id(0) / ngridx) / ngridy,
                    (i.get_group().get_group_id(0) / ngridx) % ngridy,
                    i.get_group().get_group_id(0) % ngridx,
                    (i.get_local_id(0) / nblockx) / nblocky,
                    (i.get_local_id(0) / nblockx) % nblocky,
                    i.get_local_id(0) % nblockx, shared_memory);
#endif
    my_functor.Operation6();
    i.barrier();
    while (my_functor.LoopCondition2()) {
      my_functor.Operation7();
      i.barrier();
      my_functor.Operation8();
      i.barrier();
      my_functor.Operation9();
      i.barrier();
    }
    my_functor.Operation10();
  }

private:
  FunctorType functor;
  LocalMemory localAccess;
  THREAD_IDX ngridz;
  THREAD_IDX ngridy;
  THREAD_IDX ngridx;
  THREAD_IDX nblockz;
  THREAD_IDX nblocky;
  THREAD_IDX nblockx;
};

template <typename Task> void SyclHuffmanCLCustomizedNoCGKernel(Task task) {
  // std::cout << "calling SyclHuffmanCLCustomizedNoCGKernel\n";
#if MGARD_X_SYCL_ND_RANGE_3D
  sycl::range global_threads(task.GetBlockDimX() * task.GetGridDimX(),
                             task.GetBlockDimY() * task.GetGridDimY(),
                             task.GetBlockDimZ() * task.GetGridDimZ());

  sycl::range local_threads(task.GetBlockDimX(), task.GetBlockDimY(),
                            task.GetBlockDimZ());
#endif
#if MGARD_X_SYCL_ND_RANGE_1D
  sycl::range global_threads(task.GetBlockDimX() * task.GetGridDimX() *
                                 task.GetBlockDimY() * task.GetGridDimY() *
                                 task.GetBlockDimZ() * task.GetGridDimZ(),
                             1, 1);

  sycl::range local_threads(
      task.GetBlockDimX() * task.GetBlockDimY() * task.GetBlockDimZ(), 1, 1);
#endif
  size_t sm_size = task.GetSharedMemorySize();
  if (sm_size == 0)
    sm_size = 1; // avoid -51 (CL_INVALID_ARG_SIZE) error

  sycl::queue q = DeviceRuntime<SYCL>::GetQueue(task.GetQueueIdx());

  // std::cout << "calling Single_Operation1_Kernel\n";
  q.submit([&](sycl::handler &h) {
    LocalMemory localAccess{sm_size, h};
#if MGARD_X_SYCL_ND_RANGE_3D
    Single_Operation1_Kernel_3D kernel(task.GetFunctor(), localAccess);
#endif
#if MGARD_X_SYCL_ND_RANGE_1D
    Single_Operation1_Kernel_1D kernel(
        task.GetFunctor(), localAccess, task.GetGridDimZ(), task.GetGridDimY(),
        task.GetGridDimX(), task.GetBlockDimZ(), task.GetBlockDimY(),
        task.GetBlockDimX());
#endif
    h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
  });
  DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

  // std::cout << "calling LoopCondition1\n";
  while (task.GetFunctor().LoopCondition1()) {
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

    // std::cout << "calling Single_Operation2_Kernel\n";
    q.submit([&](sycl::handler &h) {
      LocalMemory localAccess{sm_size, h};
#if MGARD_X_SYCL_ND_RANGE_3D
      Single_Operation2_Kernel_3D kernel(task.GetFunctor(), localAccess);
#endif
#if MGARD_X_SYCL_ND_RANGE_1D
      Single_Operation2_Kernel_1D kernel(
          task.GetFunctor(), localAccess, task.GetGridDimZ(),
          task.GetGridDimY(), task.GetGridDimX(), task.GetBlockDimZ(),
          task.GetBlockDimY(), task.GetBlockDimX());
#endif
      h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
    });
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

    // std::cout << "calling Single_Operation3_Kernel\n";
    q.submit([&](sycl::handler &h) {
      LocalMemory localAccess{sm_size, h};
#if MGARD_X_SYCL_ND_RANGE_3D
      Single_Operation3_Kernel_3D kernel(task.GetFunctor(), localAccess);
#endif
#if MGARD_X_SYCL_ND_RANGE_1D
      Single_Operation3_Kernel_1D kernel(
          task.GetFunctor(), localAccess, task.GetGridDimZ(),
          task.GetGridDimY(), task.GetGridDimX(), task.GetBlockDimZ(),
          task.GetBlockDimY(), task.GetBlockDimX());
#endif
      h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
    });
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

    // std::cout << "calling Single_Operation4_Kernel\n";
    q.submit([&](sycl::handler &h) {
      LocalMemory localAccess{sm_size, h};
#if MGARD_X_SYCL_ND_RANGE_3D
      Single_Operation4_Kernel_3D kernel(task.GetFunctor(), localAccess);
#endif
#if MGARD_X_SYCL_ND_RANGE_1D
      Single_Operation4_Kernel_1D kernel(
          task.GetFunctor(), localAccess, task.GetGridDimZ(),
          task.GetGridDimY(), task.GetGridDimX(), task.GetBlockDimZ(),
          task.GetBlockDimY(), task.GetBlockDimX());
#endif
      h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
    });
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

    // std::cout << "calling Single_Operation4_Kernel\n";
    q.submit([&](sycl::handler &h) {
      LocalMemory localAccess{sm_size, h};
#if MGARD_X_SYCL_ND_RANGE_3D
      Single_Operation5_Kernel_3D kernel(task.GetFunctor(), localAccess);
#endif
#if MGARD_X_SYCL_ND_RANGE_1D
      Single_Operation5_Kernel_1D kernel(
          task.GetFunctor(), localAccess, task.GetGridDimZ(),
          task.GetGridDimY(), task.GetGridDimX(), task.GetBlockDimZ(),
          task.GetBlockDimY(), task.GetBlockDimX());
#endif
      h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
    });
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

    // std::cout << "calling BranchCondition1\n";
    if (task.GetFunctor().BranchCondition1()) {
      DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

      // std::cout << "calling SyclParallelMergeKernel\n";
      q.submit([&](sycl::handler &h) {
        LocalMemory localAccess{sm_size, h};
        SyclParallelMergeKernel kernel(
            task.GetFunctor(), localAccess, task.GetGridDimZ(),
            task.GetGridDimY(), task.GetGridDimX(), task.GetBlockDimZ(),
            task.GetBlockDimY(), task.GetBlockDimX());
        h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
      });
      DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

      // std::cout << "calling Single_Operation10_Kernel\n";
      q.submit([&](sycl::handler &h) {
        LocalMemory localAccess{sm_size, h};
#if MGARD_X_SYCL_ND_RANGE_3D
        Single_Operation11_Kernel_3D kernel(task.GetFunctor(), localAccess);
#endif
#if MGARD_X_SYCL_ND_RANGE_1D
        Single_Operation11_Kernel_1D kernel(
            task.GetFunctor(), localAccess, task.GetGridDimZ(),
            task.GetGridDimY(), task.GetGridDimX(), task.GetBlockDimZ(),
            task.GetBlockDimY(), task.GetBlockDimX());
#endif
        h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
      });
      DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());
    }

    // std::cout << "calling Single_Operation11_Kernel\n";
    q.submit([&](sycl::handler &h) {
      LocalMemory localAccess{sm_size, h};
#if MGARD_X_SYCL_ND_RANGE_3D
      Single_Operation12_Kernel_3D kernel(task.GetFunctor(), localAccess);
#endif
#if MGARD_X_SYCL_ND_RANGE_1D
      Single_Operation12_Kernel_1D kernel(
          task.GetFunctor(), localAccess, task.GetGridDimZ(),
          task.GetGridDimY(), task.GetGridDimX(), task.GetBlockDimZ(),
          task.GetBlockDimY(), task.GetBlockDimX());
#endif
      h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
    });
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

    // std::cout << "calling Single_Operation12_Kernel\n";
    q.submit([&](sycl::handler &h) {
      LocalMemory localAccess{sm_size, h};
#if MGARD_X_SYCL_ND_RANGE_3D
      Single_Operation13_Kernel_3D kernel(task.GetFunctor(), localAccess);
#endif
#if MGARD_X_SYCL_ND_RANGE_1D
      Single_Operation13_Kernel_1D kernel(
          task.GetFunctor(), localAccess, task.GetGridDimZ(),
          task.GetGridDimY(), task.GetGridDimX(), task.GetBlockDimZ(),
          task.GetBlockDimY(), task.GetBlockDimX());
#endif
      h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
    });
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

    // std::cout << "calling Single_Operation13_Kernel\n";
    q.submit([&](sycl::handler &h) {
      LocalMemory localAccess{sm_size, h};
#if MGARD_X_SYCL_ND_RANGE_3D
      Single_Operation14_Kernel_3D kernel(task.GetFunctor(), localAccess);
#endif
#if MGARD_X_SYCL_ND_RANGE_1D
      Single_Operation14_Kernel_1D kernel(
          task.GetFunctor(), localAccess, task.GetGridDimZ(),
          task.GetGridDimY(), task.GetGridDimX(), task.GetBlockDimZ(),
          task.GetBlockDimY(), task.GetBlockDimX());
#endif
      h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
    });
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());
    // std::cout << "calling Single_Operation14_Kernel\n";
    q.submit([&](sycl::handler &h) {
      LocalMemory localAccess{sm_size, h};
#if MGARD_X_SYCL_ND_RANGE_3D
      Single_Operation15_Kernel_3D kernel(task.GetFunctor(), localAccess);
#endif
#if MGARD_X_SYCL_ND_RANGE_1D
      Single_Operation15_Kernel_1D kernel(
          task.GetFunctor(), localAccess, task.GetGridDimZ(),
          task.GetGridDimY(), task.GetGridDimX(), task.GetBlockDimZ(),
          task.GetBlockDimY(), task.GetBlockDimX());
#endif
      h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
    });
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());
  }
}

template <typename Task> void SyclHuffmanCWCustomizedNoCGKernel(Task task) {
  // std::cout << "calling SyclHuffmanCWCustomizedNoCGKernel\n";
#if MGARD_X_SYCL_ND_RANGE_3D
  sycl::range global_threads(task.GetBlockDimX() * task.GetGridDimX(),
                             task.GetBlockDimY() * task.GetGridDimY(),
                             task.GetBlockDimZ() * task.GetGridDimZ());

  sycl::range local_threads(task.GetBlockDimX(), task.GetBlockDimY(),
                            task.GetBlockDimZ());
#endif
#if MGARD_X_SYCL_ND_RANGE_1D
  sycl::range global_threads(task.GetBlockDimX() * task.GetGridDimX() *
                                 task.GetBlockDimY() * task.GetGridDimY() *
                                 task.GetBlockDimZ() * task.GetGridDimZ(),
                             1, 1);

  sycl::range local_threads(
      task.GetBlockDimX() * task.GetBlockDimY() * task.GetBlockDimZ(), 1, 1);
#endif
  size_t sm_size = task.GetSharedMemorySize();
  if (sm_size == 0)
    sm_size = 1; // avoid -51 (CL_INVALID_ARG_SIZE) error

  sycl::queue q = DeviceRuntime<SYCL>::GetQueue(task.GetQueueIdx());

  // std::cout << "calling Single_Operation1_Kernel\n";
  q.submit([&](sycl::handler &h) {
    LocalMemory localAccess{sm_size, h};
#if MGARD_X_SYCL_ND_RANGE_3D
    Single_Operation1_Kernel_3D kernel(task.GetFunctor(), localAccess);
#endif
#if MGARD_X_SYCL_ND_RANGE_1D
    Single_Operation1_Kernel_1D kernel(
        task.GetFunctor(), localAccess, task.GetGridDimZ(), task.GetGridDimY(),
        task.GetGridDimX(), task.GetBlockDimZ(), task.GetBlockDimY(),
        task.GetBlockDimX());
#endif
    h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
  });
  DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());
  // std::cout << "calling Single_Operation2_Kernel\n";
  q.submit([&](sycl::handler &h) {
    LocalMemory localAccess{sm_size, h};
#if MGARD_X_SYCL_ND_RANGE_3D
    Single_Operation2_Kernel_3D kernel(task.GetFunctor(), localAccess);
#endif
#if MGARD_X_SYCL_ND_RANGE_1D
    Single_Operation2_Kernel_1D kernel(
        task.GetFunctor(), localAccess, task.GetGridDimZ(), task.GetGridDimY(),
        task.GetGridDimX(), task.GetBlockDimZ(), task.GetBlockDimY(),
        task.GetBlockDimX());
#endif
    h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
  });
  DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());
  // std::cout << "calling Single_Operation3_Kernel\n";
  q.submit([&](sycl::handler &h) {
    LocalMemory localAccess{sm_size, h};
#if MGARD_X_SYCL_ND_RANGE_3D
    Single_Operation3_Kernel_3D kernel(task.GetFunctor(), localAccess);
#endif
#if MGARD_X_SYCL_ND_RANGE_1D
    Single_Operation3_Kernel_1D kernel(
        task.GetFunctor(), localAccess, task.GetGridDimZ(), task.GetGridDimY(),
        task.GetGridDimX(), task.GetBlockDimZ(), task.GetBlockDimY(),
        task.GetBlockDimX());
#endif
    h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
  });
  DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

  // std::cout << "calling LoopCondition1\n";
  while (task.GetFunctor().LoopCondition1()) {
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

    // std::cout << "calling Single_Operation4_Kernel\n";
    q.submit([&](sycl::handler &h) {
      LocalMemory localAccess{sm_size, h};
#if MGARD_X_SYCL_ND_RANGE_3D
      Single_Operation4_Kernel_3D kernel(task.GetFunctor(), localAccess);
#endif
#if MGARD_X_SYCL_ND_RANGE_1D
      Single_Operation4_Kernel_1D kernel(
          task.GetFunctor(), localAccess, task.GetGridDimZ(),
          task.GetGridDimY(), task.GetGridDimX(), task.GetBlockDimZ(),
          task.GetBlockDimY(), task.GetBlockDimX());
#endif
      h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
    });
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

    // std::cout << "calling Single_Operation5_Kernel\n";
    q.submit([&](sycl::handler &h) {
      LocalMemory localAccess{sm_size, h};
#if MGARD_X_SYCL_ND_RANGE_3D
      Single_Operation5_Kernel_3D kernel(task.GetFunctor(), localAccess);
#endif
#if MGARD_X_SYCL_ND_RANGE_1D
      Single_Operation5_Kernel_1D kernel(
          task.GetFunctor(), localAccess, task.GetGridDimZ(),
          task.GetGridDimY(), task.GetGridDimX(), task.GetBlockDimZ(),
          task.GetBlockDimY(), task.GetBlockDimX());
#endif
      h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
    });
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

    // std::cout << "calling Single_Operation6_Kernel\n";
    q.submit([&](sycl::handler &h) {
      LocalMemory localAccess{sm_size, h};
#if MGARD_X_SYCL_ND_RANGE_3D
      Single_Operation6_Kernel_3D kernel(task.GetFunctor(), localAccess);
#endif
#if MGARD_X_SYCL_ND_RANGE_1D
      Single_Operation6_Kernel_1D kernel(
          task.GetFunctor(), localAccess, task.GetGridDimZ(),
          task.GetGridDimY(), task.GetGridDimX(), task.GetBlockDimZ(),
          task.GetBlockDimY(), task.GetBlockDimX());
#endif
      h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
    });
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

    // std::cout << "calling Single_Operation7_Kernel\n";
    q.submit([&](sycl::handler &h) {
      LocalMemory localAccess{sm_size, h};
#if MGARD_X_SYCL_ND_RANGE_3D
      Single_Operation7_Kernel_3D kernel(task.GetFunctor(), localAccess);
#endif
#if MGARD_X_SYCL_ND_RANGE_1D
      Single_Operation7_Kernel_1D kernel(
          task.GetFunctor(), localAccess, task.GetGridDimZ(),
          task.GetGridDimY(), task.GetGridDimX(), task.GetBlockDimZ(),
          task.GetBlockDimY(), task.GetBlockDimX());
#endif
      h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
    });
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

    // std::cout << "calling Single_Operation8_Kernel\n";
    q.submit([&](sycl::handler &h) {
      LocalMemory localAccess{sm_size, h};
#if MGARD_X_SYCL_ND_RANGE_3D
      Single_Operation8_Kernel_3D kernel(task.GetFunctor(), localAccess);
#endif
#if MGARD_X_SYCL_ND_RANGE_1D
      Single_Operation8_Kernel_1D kernel(
          task.GetFunctor(), localAccess, task.GetGridDimZ(),
          task.GetGridDimY(), task.GetGridDimX(), task.GetBlockDimZ(),
          task.GetBlockDimY(), task.GetBlockDimX());
#endif
      h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
    });
    DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());
  }

  // std::cout << "calling Single_Operation9_Kernel\n";
  q.submit([&](sycl::handler &h) {
    LocalMemory localAccess{sm_size, h};
#if MGARD_X_SYCL_ND_RANGE_3D
    Single_Operation9_Kernel_3D kernel(task.GetFunctor(), localAccess);
#endif
#if MGARD_X_SYCL_ND_RANGE_1D
    Single_Operation9_Kernel_1D kernel(
        task.GetFunctor(), localAccess, task.GetGridDimZ(), task.GetGridDimY(),
        task.GetGridDimX(), task.GetBlockDimZ(), task.GetBlockDimY(),
        task.GetBlockDimX());
#endif
    h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
  });
  DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());

  // std::cout << "calling Single_Operation10_Kernel\n";
  q.submit([&](sycl::handler &h) {
    LocalMemory localAccess{sm_size, h};
#if MGARD_X_SYCL_ND_RANGE_3D
    Single_Operation10_Kernel_3D kernel(task.GetFunctor(), localAccess);
#endif
#if MGARD_X_SYCL_ND_RANGE_1D
    Single_Operation10_Kernel_1D kernel(
        task.GetFunctor(), localAccess, task.GetGridDimZ(), task.GetGridDimY(),
        task.GetGridDimX(), task.GetBlockDimZ(), task.GetBlockDimY(),
        task.GetBlockDimX());
#endif
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
  int IsResourceEnough(TaskType &task) {
    if (task.GetBlockDimX() * task.GetBlockDimY() * task.GetBlockDimZ() >
        DeviceRuntime<SYCL>::GetMaxNumThreadsPerTB()) {
      return THREADBLOCK_TOO_LARGE;
    }
    if (task.GetSharedMemorySize() >
        DeviceRuntime<SYCL>::GetMaxSharedMemorySize()) {
      return SHARED_MEMORY_TOO_LARGE;
    }
    return RESOURCE_ENOUGH;
  }

  MGARDX_CONT
  ExecutionReturn Execute(TaskType &task) {
#if MGARD_X_SYCL_ND_RANGE_3D
    sycl::range global_threads(task.GetBlockDimX() * task.GetGridDimX(),
                               task.GetBlockDimY() * task.GetGridDimY(),
                               task.GetBlockDimZ() * task.GetGridDimZ());

    sycl::range local_threads(task.GetBlockDimX(), task.GetBlockDimY(),
                              task.GetBlockDimZ());
#endif
#if MGARD_X_SYCL_ND_RANGE_1D
    sycl::range global_threads(task.GetBlockDimX() * task.GetGridDimX() *
                                   task.GetBlockDimY() * task.GetGridDimY() *
                                   task.GetBlockDimZ() * task.GetGridDimZ(),
                               1, 1);

    sycl::range local_threads(
        task.GetBlockDimX() * task.GetBlockDimY() * task.GetBlockDimZ(), 1, 1);
#endif
    size_t sm_size = task.GetSharedMemorySize();
    if (sm_size == 0)
      sm_size = 1; // avoid -51 (CL_INVALID_ARG_SIZE) error

    sycl::queue q = DeviceRuntime<SYCL>::GetQueue(task.GetQueueIdx());

    if (DeviceRuntime<SYCL>::PrintKernelConfig) {
#if MGARD_X_SYCL_ND_RANGE_3D
      std::cout << log::log_info << task.GetFunctorName() << ": <"
                << task.GetBlockDimX() << ", " << task.GetBlockDimY() << ", "
                << task.GetBlockDimZ() << "> <" << task.GetGridDimX() << ", "
                << task.GetGridDimY() << ", " << task.GetGridDimZ() << ">\n";
#endif
#if MGARD_X_SYCL_ND_RANGE_1D
      std::cout << log::log_info << task.GetFunctorName() << ": <"
                << task.GetBlockDimX() * task.GetBlockDimY() *
                       task.GetBlockDimZ()
                << ", " << 1 << ", " << 1 << "> <"
                << task.GetGridDimX() * task.GetGridDimY() * task.GetGridDimZ()
                << ", " << 1 << ", " << 1 << ">\n";
#endif
    }

    ExecutionReturn ret;
    if (IsResourceEnough(task) != RESOURCE_ENOUGH) {
      if (DeviceRuntime<SYCL>::PrintKernelConfig) {
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
        DeviceRuntime<SYCL>::TimingAllKernels ||
        AutoTuner<SYCL>::ProfileKernels) {
      DeviceRuntime<SYCL>::SyncDevice();
      timer.start();
    }

    // if constexpr evaluate at compile time otherwise this does not compile
    if constexpr (std::is_base_of<Functor<SYCL>,
                                  typename TaskType::Functor>::value) {
      q.submit([&](sycl::handler &h) {
        LocalMemory localAccess{sm_size, h};
        SyclKernel kernel(task.GetFunctor(), localAccess, task.GetGridDimZ(),
                          task.GetGridDimY(), task.GetGridDimX(),
                          task.GetBlockDimZ(), task.GetBlockDimY(),
                          task.GetBlockDimX());
        h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
      });
    } else if constexpr (std::is_base_of<IterFunctor<SYCL>,
                                         typename TaskType::Functor>::value) {
      q.submit([&](sycl::handler &h) {
        LocalMemory localAccess{sm_size, h};
        SyclIterKernel kernel(task.GetFunctor(), localAccess,
                              task.GetGridDimZ(), task.GetGridDimY(),
                              task.GetGridDimX(), task.GetBlockDimZ(),
                              task.GetBlockDimY(), task.GetBlockDimX());
        h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
      });
    } else if constexpr (std::is_base_of<HuffmanCLCustomizedFunctor<SYCL>,
                                         typename TaskType::Functor>::value) {
      SyclHuffmanCLCustomizedNoCGKernel(task);
    } else if constexpr (std::is_base_of<HuffmanCWCustomizedFunctor<SYCL>,
                                         typename TaskType::Functor>::value) {
      SyclHuffmanCWCustomizedNoCGKernel(task);
    }
    if (DeviceRuntime<SYCL>::SyncAllKernelsAndCheckErrors) {
      DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());
    }

    if (task.GetQueueIdx() == MGARDX_SYNCHRONIZED_QUEUE ||
        DeviceRuntime<SYCL>::TimingAllKernels ||
        AutoTuner<SYCL>::ProfileKernels) {
      DeviceRuntime<SYCL>::SyncDevice();
      timer.end();
      if (DeviceRuntime<SYCL>::TimingAllKernels) {
        timer.print(task.GetFunctorName());
      }
      if (AutoTuner<SYCL>::ProfileKernels) {
        ret.success = true;
        ret.execution_time = timer.get();
      }
    }
    return ret;
  }
};

template <> class DeviceLauncher<SYCL> {
public:
  template <typename TaskType>
  MGARDX_CONT int static IsResourceEnough(TaskType &task) {
    if (task.GetBlockDimX() * task.GetBlockDimY() * task.GetBlockDimZ() >
        DeviceRuntime<SYCL>::GetMaxNumThreadsPerTB()) {
      return THREADBLOCK_TOO_LARGE;
    }
    if (task.GetSharedMemorySize() >
        DeviceRuntime<SYCL>::GetMaxSharedMemorySize()) {
      return SHARED_MEMORY_TOO_LARGE;
    }
    return RESOURCE_ENOUGH;
  }

  template <typename TaskType>
  MGARDX_CONT ExecutionReturn static Execute(TaskType &task) {
#if MGARD_X_SYCL_ND_RANGE_3D
    sycl::range global_threads(task.GetBlockDimX() * task.GetGridDimX(),
                               task.GetBlockDimY() * task.GetGridDimY(),
                               task.GetBlockDimZ() * task.GetGridDimZ());

    sycl::range local_threads(task.GetBlockDimX(), task.GetBlockDimY(),
                              task.GetBlockDimZ());
#endif
#if MGARD_X_SYCL_ND_RANGE_1D
    sycl::range global_threads(task.GetBlockDimX() * task.GetGridDimX() *
                                   task.GetBlockDimY() * task.GetGridDimY() *
                                   task.GetBlockDimZ() * task.GetGridDimZ(),
                               1, 1);

    sycl::range local_threads(
        task.GetBlockDimX() * task.GetBlockDimY() * task.GetBlockDimZ(), 1, 1);
#endif
    size_t sm_size = task.GetSharedMemorySize();
    if (sm_size == 0)
      sm_size = 1; // avoid -51 (CL_INVALID_ARG_SIZE) error

    sycl::queue q = DeviceRuntime<SYCL>::GetQueue(task.GetQueueIdx());

    if (DeviceRuntime<SYCL>::PrintKernelConfig) {
#if MGARD_X_SYCL_ND_RANGE_3D
      std::cout << log::log_info << task.GetFunctorName() << ": <"
                << task.GetBlockDimX() << ", " << task.GetBlockDimY() << ", "
                << task.GetBlockDimZ() << "> <" << task.GetGridDimX() << ", "
                << task.GetGridDimY() << ", " << task.GetGridDimZ() << ">\n";
#endif
#if MGARD_X_SYCL_ND_RANGE_1D
      std::cout << log::log_info << task.GetFunctorName() << ": <"
                << task.GetBlockDimX() * task.GetBlockDimY() *
                       task.GetBlockDimZ()
                << ", " << 1 << ", " << 1 << "> <"
                << task.GetGridDimX() * task.GetGridDimY() * task.GetGridDimZ()
                << ", " << 1 << ", " << 1 << ">\n";
#endif
    }

    ExecutionReturn ret;
    if (IsResourceEnough(task) != RESOURCE_ENOUGH) {
      if (DeviceRuntime<SYCL>::PrintKernelConfig) {
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
        DeviceRuntime<SYCL>::TimingAllKernels ||
        AutoTuner<SYCL>::ProfileKernels) {
      DeviceRuntime<SYCL>::SyncDevice();
      timer.start();
    }

    // if constexpr evaluate at compile time otherwise this does not compile
    if constexpr (std::is_base_of<Functor<SYCL>,
                                  typename TaskType::Functor>::value) {
      q.submit([&](sycl::handler &h) {
        LocalMemory localAccess{sm_size, h};
        SyclKernel kernel(task.GetFunctor(), localAccess, task.GetGridDimZ(),
                          task.GetGridDimY(), task.GetGridDimX(),
                          task.GetBlockDimZ(), task.GetBlockDimY(),
                          task.GetBlockDimX());
        h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
      });
    } else if constexpr (std::is_base_of<IterFunctor<SYCL>,
                                         typename TaskType::Functor>::value) {
      q.submit([&](sycl::handler &h) {
        LocalMemory localAccess{sm_size, h};
        SyclIterKernel kernel(task.GetFunctor(), localAccess,
                              task.GetGridDimZ(), task.GetGridDimY(),
                              task.GetGridDimX(), task.GetBlockDimZ(),
                              task.GetBlockDimY(), task.GetBlockDimX());
        h.parallel_for(sycl::nd_range{global_threads, local_threads}, kernel);
      });
    } else if constexpr (std::is_base_of<HuffmanCLCustomizedFunctor<SYCL>,
                                         typename TaskType::Functor>::value) {
      SyclHuffmanCLCustomizedNoCGKernel(task);
    } else if constexpr (std::is_base_of<HuffmanCWCustomizedFunctor<SYCL>,
                                         typename TaskType::Functor>::value) {
      SyclHuffmanCWCustomizedNoCGKernel(task);
    }
    if (DeviceRuntime<SYCL>::SyncAllKernelsAndCheckErrors) {
      DeviceRuntime<SYCL>::SyncQueue(task.GetQueueIdx());
    }

    if (task.GetQueueIdx() == MGARDX_SYNCHRONIZED_QUEUE ||
        DeviceRuntime<SYCL>::TimingAllKernels ||
        AutoTuner<SYCL>::ProfileKernels) {
      DeviceRuntime<SYCL>::SyncDevice();
      timer.end();
      if (DeviceRuntime<SYCL>::TimingAllKernels) {
        timer.print(task.GetFunctorName());
      }
      if (AutoTuner<SYCL>::ProfileKernels) {
        ret.success = true;
        ret.execution_time = timer.get();
      }
    }
    return ret;
  }

  template <typename TaskType>
  MGARDX_CONT static void ConfigTask(TaskType task) {
    typename TaskType::Functor functor;
    int maxbytes = DeviceRuntime<SYCL>::GetMaxSharedMemorySize();
    DeviceRuntime<SYCL>::SetMaxDynamicSharedMemorySize(functor, maxbytes);
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
    ret = Execute(task);                                                       \
    if constexpr (KernelType::EnableConfig()) {                                \
      ConfigTask(task);                                                        \
    }                                                                          \
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
    if (AutoTuner<SYCL>::WriteToTable) {
      FillAutoTunerTable<KernelType::NumDim, typename KernelType::DataType,
                         SYCL>(std::string(KernelType::Name), min_config);
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
                             SYCL>(KernelType::Name);
      auto task =
          kernel.template GenTask<config.z, config.y, config.x>(queue_idx);
      if constexpr (KernelType::EnableConfig()) {
        ConfigTask(task);
      }
      Execute(task);

      if (AutoTuner<SYCL>::ProfileKernels) {
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

template <typename T> struct AbsMaxOp {
  T operator()(const T &a, const T &b) const {
    return (fabs(b) > fabs(a)) ? fabs(b) : fabs(a);
  }
};

template <typename T> struct SquareOp {
  T operator()(const T &a) const { return a * a; }
};

template <> class DeviceCollective<SYCL> {
public:
  MGARDX_CONT
  DeviceCollective(){};

  template <typename T>
  MGARDX_CONT static void Sum(SIZE n, SubArray<1, T, SYCL> v,
                              SubArray<1, T, SYCL> result,
                              Array<1, Byte, SYCL> &workspace,
                              bool workspace_allocated, int queue_idx) {

    if (workspace_allocated) {
      sycl::queue q = DeviceRuntime<SYCL>::GetQueue(queue_idx);
      q.submit([&](sycl::handler &h) {
        T *res = result.data();
        T *input = v.data();
        h.parallel_for(
            sycl::range{n}, sycl::reduction(res, (T)0, sycl::plus<T>()),
            [=](sycl::id<1> i, auto &res) { res.combine(input[i]); });
      });
      DeviceRuntime<SYCL>::SyncDevice();
    } else {
      workspace.resize({(SIZE)1}, queue_idx);
    }
  }

  template <typename T>
  MGARDX_CONT static void AbsMax(SIZE n, SubArray<1, T, SYCL> v,
                                 SubArray<1, T, SYCL> result,
                                 Array<1, Byte, SYCL> &workspace,
                                 bool workspace_allocated, int queue_idx) {

    if (workspace_allocated) {
      sycl::queue q = DeviceRuntime<SYCL>::GetQueue(queue_idx);
      q.submit([&](sycl::handler &h) {
        T *res = result.data();
        T *input = v.data();
        h.parallel_for(
            sycl::range{n}, sycl::reduction(res, (T)0, AbsMaxOp<T>()),
            [=](sycl::id<1> i, auto &res) { res.combine(input[i]); });
      });
      DeviceRuntime<SYCL>::SyncDevice();
    } else {
      workspace.resize({(SIZE)1}, queue_idx);
    }
  }

  template <typename T>
  MGARDX_CONT static void SquareSum(SIZE n, SubArray<1, T, SYCL> v,
                                    SubArray<1, T, SYCL> result,
                                    Array<1, Byte, SYCL> &workspace,
                                    bool workspace_allocated, int queue_idx) {

    if (workspace_allocated) {
      sycl::queue q = DeviceRuntime<SYCL>::GetQueue(queue_idx);
      q.submit([&](sycl::handler &h) {
        T *res = result.data();
        T *input = v.data();
        h.parallel_for(sycl::range{n},
                       sycl::reduction(res, (T)0, sycl::plus<T>()),
                       [=](sycl::id<1> i, auto &res) {
                         res.combine(input[i] * input[i]);
                       });
      });
      DeviceRuntime<SYCL>::SyncDevice();
    } else {
      workspace.resize({(SIZE)1}, queue_idx);
    }
  }

  template <typename T>
  MGARDX_CONT static void
  ScanSumInclusive(SIZE n, SubArray<1, T, SYCL> v, SubArray<1, T, SYCL> result,
                   Array<1, Byte, SYCL> &workspace, bool workspace_allocated,
                   int queue_idx) {}

  template <typename T>
  MGARDX_CONT static void
  ScanSumExclusive(SIZE n, SubArray<1, T, SYCL> v, SubArray<1, T, SYCL> result,
                   Array<1, Byte, SYCL> &workspace, bool workspace_allocated,
                   int queue_idx) {}

  template <typename T>
  MGARDX_CONT static void
  ScanSumExtended(SIZE n, SubArray<1, T, SYCL> v, SubArray<1, T, SYCL> result,
                  Array<1, Byte, SYCL> &workspace, bool workspace_allocated,
                  int queue_idx) {}

  template <typename KeyT, typename ValueT>
  MGARDX_CONT static void SortByKey(SIZE n, SubArray<1, KeyT, SYCL> in_keys,
                                    SubArray<1, ValueT, SYCL> in_values,
                                    SubArray<1, KeyT, SYCL> out_keys,
                                    SubArray<1, ValueT, SYCL> out_values,
                                    Array<1, Byte, SYCL> &workspace,
                                    bool workspace_allocated, int queue_idx) {

    if (workspace_allocated) {
      KeyT *keys_array = new KeyT[n];
      ValueT *values_array = new ValueT[n];
      MemoryManager<SYCL>::Copy1D(keys_array, in_keys.data(), n, 0);
      MemoryManager<SYCL>::Copy1D(values_array, in_values.data(), n, 0);
      DeviceRuntime<SYCL>::SyncQueue(0);

      std::vector<std::pair<KeyT, ValueT>> data(n);
      for (SIZE i = 0; i < n; ++i) {
        data[i] = std::pair<KeyT, ValueT>(keys_array[i], values_array[i]);
      }
      std::stable_sort(data.begin(), data.end(),
                       KeyValueComparator<KeyT, ValueT>{});
      for (SIZE i = 0; i < n; ++i) {
        keys_array[i] = data[i].first;
        values_array[i] = data[i].second;
      }
      MemoryManager<SYCL>::Copy1D(out_keys.data(), keys_array, n, 0);
      MemoryManager<SYCL>::Copy1D(out_values.data(), values_array, n, 0);
      DeviceRuntime<SYCL>::SyncDevice();
      delete[] keys_array;
      delete[] values_array;
    } else {
      workspace.resize({(SIZE)1}, queue_idx);
    }
  }

  template <typename KeyT, typename ValueT, typename BinaryOpType>
  MGARDX_CONT static void
  ScanOpInclusiveByKey(SubArray<1, SIZE, SYCL> &key,
                       SubArray<1, ValueT, SYCL> &v,
                       SubArray<1, ValueT, SYCL> &result, int queue_idx) {}
};

} // namespace mgard_x
#endif
