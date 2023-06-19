/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_ARRAY_HPP
#define MGARD_X_ARRAY_HPP
// #include "Common.h"

// #include "CommonInternal.h"
#include <algorithm>
#include <vector>

#include "Array.h"

// #include "../DeviceAdapters/DeviceAdapter.h"

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
Array<D, T, DeviceType>::Array() {
  initialize(std::vector<SIZE>(D, 1));
}

template <DIM D, typename T, typename DeviceType>
Array<D, T, DeviceType>::Array(std::vector<SIZE> shape, bool pitched,
                               bool managed, int queue_idx) {
  initialize(shape);
  allocate(pitched, managed, queue_idx);
}

template <DIM D, typename T, typename DeviceType>
Array<D, T, DeviceType>::Array(std::vector<SIZE> shape, T *dv) {
  initialize(shape);
  __shape_allocation = shape;
  __ldvs_allocation = shape;
  device_allocated = true;
  external_allocation = true;
  this->dv = dv;
}

template <DIM D, typename T, typename DeviceType>
void Array<D, T, DeviceType>::initialize(std::vector<SIZE> shape) {
  if (shape.size() != D) {
    std::cerr << log::log_err << "Number of dimensions mismatch ("
              << shape.size() << "!=" << D
              << "). mgard_x::Array not initialized!\n";
    exit(-1);
  }
  dev_id = DeviceRuntime<DeviceType>::GetDevice();
  __shape = shape;
  free();
  __ldvs = __shape;
  linearized_width = 1;
  for (DIM d = 0; d < D - 1; d++) {
    linearized_width *= __shape[d];
  }
  host_allocated = false;
  device_allocated = false;
  external_allocation = false;
  pitched = false;
  managed = false;
}

template <DIM D, typename T, typename DeviceType>
void Array<D, T, DeviceType>::allocate(bool pitched, bool managed,
                                       int queue_idx) {
  this->pitched = pitched && !MemoryManager<DeviceType>::ReduceMemoryFootprint;
  this->managed = managed;
  if (this->pitched) {
    if (!this->managed) {
      SIZE ld = 0;
      MemoryManager<DeviceType>::MallocND(dv, __shape[D - 1], linearized_width,
                                          ld, queue_idx);
      __ldvs[D - 1] = ld;
    } else {
      std::cerr << log::log_err
                << "Does not support managed memory in pitched mode.\n";
    }
  } else {
    if (!this->managed) {
      MemoryManager<DeviceType>::Malloc1D(dv, __shape[D - 1] * linearized_width,
                                          queue_idx);
    } else {
      MemoryManager<DeviceType>::MallocManaged1D(
          dv, __shape[D - 1] * linearized_width, queue_idx);
    }
  }
  __shape_allocation = __shape;
  __ldvs_allocation = __ldvs;
  device_allocated = true;
  external_allocation = false;
}

template <DIM D, typename T, typename DeviceType>
void Array<D, T, DeviceType>::copy(const Array<D, T, DeviceType> &array,
                                   int queue_idx) {
  initialize(array.__shape);
  if (array.device_allocated) {
    allocate(array.pitched, array.managed, queue_idx);
    MemoryManager<DeviceType>::CopyND(dv, __ldvs[D - 1], array.dv,
                                      array.__ldvs[D - 1], array.__shape[D - 1],
                                      array.linearized_width, queue_idx);
  }
  if (array.host_allocated) {
    hostCopy(array.keepHostCopy, queue_idx);
  }
}

template <DIM D, typename T, typename DeviceType>
void Array<D, T, DeviceType>::move(Array<D, T, DeviceType> &&array) {
  initialize(array.__shape);
  this->dev_id = array.dev_id;
  this->pitched = array.pitched;
  this->managed = array.managed;
  if (array.device_allocated) {
    this->dv = array.dv;
    this->__ldvs = array.__ldvs;
    this->__shape_allocation = array.__shape_allocation;
    this->__ldvs_allocation = array.__ldvs_allocation;
    this->device_allocated = true;
    array.device_allocated = false;
    this->external_allocation = array.external_allocation;
    array.dv = nullptr;
  }
  if (array.host_allocated) {
    this->hv = array.hv;
    this->host_allocated = true;
    array.host_allocated = false;
  }
}

template <DIM D, typename T, typename DeviceType>
void Array<D, T, DeviceType>::memset(int value, int queue_idx) {
  if (this->pitched) {
    MemoryManager<DeviceType>::MemsetND(dv, __ldvs[D - 1], __shape[D - 1],
                                        linearized_width, value, queue_idx);
  } else {
    MemoryManager<DeviceType>::Memset1D(dv, __ldvs[D - 1] * linearized_width,
                                        value, queue_idx);
  }
}

template <DIM D, typename T, typename DeviceType>
void Array<D, T, DeviceType>::free(int queue_idx) {
  if (device_allocated && !external_allocation) {
    MemoryManager<DeviceType>::Free(dv, queue_idx);
    device_allocated = false;
    dv = nullptr;
  }
  if (host_allocated && !keepHostCopy) {
    MemoryManager<DeviceType>::FreeHost(hv, queue_idx);
    host_allocated = false;
    hv = nullptr;
  }
}

template <DIM D, typename T, typename DeviceType>
Array<D, T, DeviceType>::Array(const Array<D, T, DeviceType> &array) {
  this->copy(array);
}

template <DIM D, typename T, typename DeviceType>
Array<D, T, DeviceType> &
Array<D, T, DeviceType>::operator=(const Array<D, T, DeviceType> &array) {
  // printf("Array operator =\n");
  this->copy(array);
  return *this;
}

template <DIM D, typename T, typename DeviceType>
Array<D, T, DeviceType> &
Array<D, T, DeviceType>::operator=(Array<D, T, DeviceType> &&array) {
  // printf("Array move = \n");
  this->move(std::move(array));
  return *this;
}

template <DIM D, typename T, typename DeviceType>
Array<D, T, DeviceType>::Array(Array<D, T, DeviceType> &&array) {
  // printf("Array move\n");
  this->move(std::move(array));
}

template <DIM D, typename T, typename DeviceType>
Array<D, T, DeviceType>::~Array() {
  this->free();
}

template <DIM D, typename T, typename DeviceType>
void Array<D, T, DeviceType>::load(const T *data, SIZE ld, int queue_idx) {
  log::dbg("Calling Array::load");
  if (ld == 0) {
    ld = __shape[D - 1];
  }
  MemoryManager<DeviceType>::CopyND(dv, __ldvs[D - 1], data, ld, __shape[D - 1],
                                    linearized_width, queue_idx);
}

template <DIM D, typename T, typename DeviceType>
T *Array<D, T, DeviceType>::hostCopy(bool keep, int queue_idx) {
  log::dbg("Calling Array::hostCopy");
  if (!device_allocated) {
    std::cout << log::log_err << "device buffer not initialized.\n";
    exit(-1);
  }
  if (!host_allocated) {
    MemoryManager<DeviceType>::MallocHost(hv, __shape[D - 1] * linearized_width,
                                          queue_idx);
    host_allocated = true;
  }
  MemoryManager<DeviceType>::CopyND(hv, __shape[D - 1], dv, __ldvs[D - 1],
                                    __shape[D - 1], linearized_width,
                                    queue_idx);
  keepHostCopy = keep;
  return hv;
}

template <DIM D, typename T, typename DeviceType>
T *Array<D, T, DeviceType>::data(SIZE &ld) {
  if (!device_allocated) {
    std::cout << log::log_err << "device buffer not initialized.\n";
    exit(-1);
  }
  ld = __ldvs[D - 1];
  return dv;
}

template <DIM D, typename T, typename DeviceType>
SIZE &Array<D, T, DeviceType>::shape(DIM d) {
  return __shape[d];
}

template <DIM D, typename T, typename DeviceType>
std::vector<SIZE> &Array<D, T, DeviceType>::shape() {
  return __shape;
}

template <DIM D, typename T, typename DeviceType>
SIZE Array<D, T, DeviceType>::totalNumElems() {
  SIZE total_num_elems = 1;
  for (DIM d = 0; d < D; d++) {
    total_num_elems *= __shape[d];
  }
  return total_num_elems;
}

template <DIM D, typename T, typename DeviceType>
T *Array<D, T, DeviceType>::data() {
  if (!device_allocated) {
    std::cout << log::log_err << "device buffer not initialized.\n";
    exit(-1);
  }
  return dv;
}

template <DIM D, typename T, typename DeviceType>
T *Array<D, T, DeviceType>::dataHost() {
  if (!host_allocated) {
    std::cout << log::log_err << "host buffer not initialized.\n";
    exit(-1);
  }
  return hv;
}

template <DIM D, typename T, typename DeviceType>
SIZE Array<D, T, DeviceType>::ld(DIM d) {
  return __ldvs[d];
}

template <DIM D, typename T, typename DeviceType>
bool Array<D, T, DeviceType>::isPitched() {
  return pitched;
}

template <DIM D, typename T, typename DeviceType>
bool Array<D, T, DeviceType>::isManaged() {
  return managed;
}

template <DIM D, typename T, typename DeviceType>
int Array<D, T, DeviceType>::resideDevice() {
  return dev_id;
}

template <DIM D, typename T, typename DeviceType>
bool Array<D, T, DeviceType>::hasDeviceAllocation() {
  return device_allocated;
}

template <DIM D, typename T, typename DeviceType>
bool Array<D, T, DeviceType>::hasHostAllocation() {
  return host_allocated;
}

template <DIM D, typename T, typename DeviceType>
void Array<D, T, DeviceType>::resize(std::vector<SIZE> shape, int queue_idx) {
  bool inplace_resizable = false;
  if (device_allocated) {
    if (!isPitched()) {
      // check total number of elements
      SIZE original_num_elems = 1;
      SIZE new_num_elems = 1;
      for (DIM d = 0; d < D; d++) {
        original_num_elems *= __shape_allocation[d];
        new_num_elems *= shape[d];
      }
      if (original_num_elems >= new_num_elems) {
        // We can reuse existing allocation
        inplace_resizable = true;
        __shape = shape;
        __ldvs = __shape;
        linearized_width = 1;
        for (DIM d = 0; d < D - 1; d++) {
          linearized_width *= __shape[d];
        }
      }
    } else {
      bool shape_compatiable = true;
      for (DIM d = 0; d < D; d++) {
        if (__shape_allocation[d] < shape[d]) {
          shape_compatiable = false;
          break;
        }
      }
      if (shape_compatiable) {
        // We can reuse existing allocation
        inplace_resizable = true;
        __shape = shape;
        linearized_width = 1;
        for (DIM d = 0; d < D - 1; d++) {
          linearized_width *= __shape[d];
        }
      }
    }
  }
  // If cannot reuse existing allocation or there is no existing allocation
  if (!inplace_resizable) {
    initialize(shape);
    allocate(isPitched(), isManaged(), queue_idx);
  }
}

} // namespace mgard_x

#endif