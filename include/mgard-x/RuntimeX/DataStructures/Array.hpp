/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
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

template <DIM D, typename T, typename DeviceType, bool Pitched, bool Managed>
Array<D, T, DeviceType, Pitched, Managed>::Array() {
  initialize(std::vector<SIZE>(D, 1));
}

template <DIM D, typename T, typename DeviceType, bool Pitched, bool Managed>
Array<D, T, DeviceType, Pitched, Managed>::Array(std::vector<SIZE> shape,
                                                 int queue_idx) {
  initialize(shape);
  allocate(queue_idx);
}

template <DIM D, typename T, typename DeviceType, bool Pitched, bool Managed>
Array<D, T, DeviceType, Pitched, Managed>::Array(std::vector<SIZE> shape,
                                                 T *dv) {
  initialize(shape);
  __shape_allocation = shape;
  __ldvs_allocation = shape;
  device_allocated = true;
  external_allocation = true;
  this->dv = dv;
}

template <DIM D, typename T, typename DeviceType, bool Pitched, bool Managed>
void Array<D, T, DeviceType, Pitched, Managed>::initialize(
    std::vector<SIZE> shape) {
  if (shape.size() != D) {
    throw std::runtime_error(
        "Number of dimensions mismatch (" + std::to_string(shape.size()) +
        "!=" + std::to_string(D) + "). mgard_x::Array not initialized!");
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
}

template <DIM D, typename T, typename DeviceType, bool Pitched, bool Managed>
void Array<D, T, DeviceType, Pitched, Managed>::allocate(int queue_idx) {
  if (Pitched) {
    if (!Managed) {
      SIZE ld = 0;
      MemoryManager<DeviceType>::MallocND(dv, __shape[D - 1], linearized_width,
                                          ld, queue_idx);
      __ldvs[D - 1] = ld;
    } else {
      std::cerr << log::log_err
                << "Does not support managed memory in pitched mode.\n";
    }
  } else {
    if (!Managed) {
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

template <DIM D, typename T, typename DeviceType, bool Pitched, bool Managed>
void Array<D, T, DeviceType, Pitched, Managed>::copy(
    const Array<D, T, DeviceType, Pitched, Managed> &array, int queue_idx) {
  initialize(array.__shape);
  if (array.device_allocated) {
    allocate(queue_idx);
    MemoryManager<DeviceType>::Copy2D(dv, __ldvs[D - 1], array.dv,
                                      array.__ldvs[D - 1], array.__shape[D - 1],
                                      array.linearized_width, queue_idx);
  }
  if (array.host_allocated) {
    hostCopy(array.keepHostCopy, queue_idx);
  }
}

template <DIM D, typename T, typename DeviceType, bool Pitched, bool Managed>
void Array<D, T, DeviceType, Pitched, Managed>::move(
    Array<D, T, DeviceType, Pitched, Managed> &&array) {
  initialize(array.__shape);
  this->dev_id = array.dev_id;
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

template <DIM D, typename T, typename DeviceType, bool Pitched, bool Managed>
void Array<D, T, DeviceType, Pitched, Managed>::memset(int value,
                                                       int queue_idx) {
  if (Pitched) {
    MemoryManager<DeviceType>::MemsetND(dv, __ldvs[D - 1], __shape[D - 1],
                                        linearized_width, value, queue_idx);
  } else {
    MemoryManager<DeviceType>::Memset1D(dv, __ldvs[D - 1] * linearized_width,
                                        value, queue_idx);
  }
}

template <DIM D, typename T, typename DeviceType, bool Pitched, bool Managed>
void Array<D, T, DeviceType, Pitched, Managed>::free(int queue_idx) {
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

template <DIM D, typename T, typename DeviceType, bool Pitched, bool Managed>
Array<D, T, DeviceType, Pitched, Managed>::Array(
    const Array<D, T, DeviceType, Pitched, Managed> &array) {
  this->copy(array);
}

template <DIM D, typename T, typename DeviceType, bool Pitched, bool Managed>
Array<D, T, DeviceType, Pitched, Managed> &
Array<D, T, DeviceType, Pitched, Managed>::operator=(
    const Array<D, T, DeviceType, Pitched, Managed> &array) {
  // printf("Array operator =\n");
  this->copy(array);
  return *this;
}

template <DIM D, typename T, typename DeviceType, bool Pitched, bool Managed>
Array<D, T, DeviceType, Pitched, Managed> &
Array<D, T, DeviceType, Pitched, Managed>::operator=(
    Array<D, T, DeviceType, Pitched, Managed> &&array) {
  // printf("Array move = \n");
  this->move(std::move(array));
  return *this;
}

template <DIM D, typename T, typename DeviceType, bool Pitched, bool Managed>
Array<D, T, DeviceType, Pitched, Managed>::Array(
    Array<D, T, DeviceType, Pitched, Managed> &&array) {
  // printf("Array move\n");
  this->move(std::move(array));
}

template <DIM D, typename T, typename DeviceType, bool Pitched, bool Managed>
Array<D, T, DeviceType, Pitched, Managed>::~Array() {
  this->free();
}

template <DIM D, typename T, typename DeviceType, bool Pitched, bool Managed>
void Array<D, T, DeviceType, Pitched, Managed>::load(const T *data, SIZE ld,
                                                     int queue_idx) {
  log::dbg("Calling Array::load");
  if (ld == 0) {
    ld = __shape[D - 1];
  }
  MemoryManager<DeviceType>::Copy2D(dv, __ldvs[D - 1], data, ld, __shape[D - 1],
                                    linearized_width, queue_idx);
}
template <DIM D, typename T, typename DeviceType, bool Pitched, bool Managed>
void Array<D, T, DeviceType, Pitched, Managed>::hostAllocate(bool keep,
                                                             int queue_idx) {
  log::dbg("Calling Array::hostAllocate");
  if (!device_allocated) {
    throw std::runtime_error("device buffer not initialized.");
  }
  if (!host_allocated) {
    MemoryManager<DeviceType>::MallocHost(hv, __shape[D - 1] * linearized_width,
                                          queue_idx);
    host_allocated = true;
  }
  keepHostCopy = keep;
}

template <DIM D, typename T, typename DeviceType, bool Pitched, bool Managed>
T *Array<D, T, DeviceType, Pitched, Managed>::hostCopy(bool keep,
                                                       int queue_idx) {
  log::dbg("Calling Array::hostCopy");
  if (!device_allocated) {
    throw std::runtime_error("device buffer not initialized.");
  }
  if (!host_allocated) {
    MemoryManager<DeviceType>::MallocHost(hv, __shape[D - 1] * linearized_width,
                                          queue_idx);
    host_allocated = true;
  }
  MemoryManager<DeviceType>::Copy2D(hv, __shape[D - 1], dv, __ldvs[D - 1],
                                    __shape[D - 1], linearized_width,
                                    queue_idx);
  keepHostCopy = keep;
  return hv;
}

template <DIM D, typename T, typename DeviceType, bool Pitched, bool Managed>
T *Array<D, T, DeviceType, Pitched, Managed>::data(SIZE &ld) {
  if (!device_allocated) {
    throw std::runtime_error("device buffer not initialized.");
  }
  ld = __ldvs[D - 1];
  return dv;
}

template <DIM D, typename T, typename DeviceType, bool Pitched, bool Managed>
SIZE &Array<D, T, DeviceType, Pitched, Managed>::shape(DIM d) {
  return __shape[d];
}

template <DIM D, typename T, typename DeviceType, bool Pitched, bool Managed>
std::vector<SIZE> &Array<D, T, DeviceType, Pitched, Managed>::shape() {
  return __shape;
}

template <DIM D, typename T, typename DeviceType, bool Pitched, bool Managed>
SIZE Array<D, T, DeviceType, Pitched, Managed>::totalNumElems() {
  SIZE total_num_elems = 1;
  for (DIM d = 0; d < D; d++) {
    total_num_elems *= __shape[d];
  }
  return total_num_elems;
}

template <DIM D, typename T, typename DeviceType, bool Pitched, bool Managed>
T *Array<D, T, DeviceType, Pitched, Managed>::data() {
  if (!device_allocated) {
    throw std::runtime_error("device buffer not initialized.");
  }
  return dv;
}

template <DIM D, typename T, typename DeviceType, bool Pitched, bool Managed>
T *Array<D, T, DeviceType, Pitched, Managed>::dataHost() {
  if (!host_allocated) {
    throw std::runtime_error("host buffer not initialized.");
  }
  return hv;
}

template <DIM D, typename T, typename DeviceType, bool Pitched, bool Managed>
SIZE Array<D, T, DeviceType, Pitched, Managed>::ld(DIM d) {
  return __ldvs[d];
}

template <DIM D, typename T, typename DeviceType, bool Pitched, bool Managed>
bool Array<D, T, DeviceType, Pitched, Managed>::isPitched() {
  return Pitched;
}

template <DIM D, typename T, typename DeviceType, bool Pitched, bool Managed>
bool Array<D, T, DeviceType, Pitched, Managed>::isManaged() {
  return Managed;
}

template <DIM D, typename T, typename DeviceType, bool Pitched, bool Managed>
int Array<D, T, DeviceType, Pitched, Managed>::resideDevice() {
  return dev_id;
}

template <DIM D, typename T, typename DeviceType, bool Pitched, bool Managed>
bool Array<D, T, DeviceType, Pitched, Managed>::hasDeviceAllocation() {
  return device_allocated;
}

template <DIM D, typename T, typename DeviceType, bool Pitched, bool Managed>
bool Array<D, T, DeviceType, Pitched, Managed>::hasHostAllocation() {
  return host_allocated;
}

template <DIM D, typename T, typename DeviceType, bool Pitched, bool Managed>
bool Array<D, T, DeviceType, Pitched, Managed>::resize(std::vector<SIZE> shape,
                                                       int queue_idx) {
  if (!device_allocated) {
    initialize(shape);
    allocate(queue_idx);
    return false;
  } else {
    bool inplace_resizable = false;
    if (Pitched) {
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
        // Keep the leading dimensions in step with the shape, as initialize()
        // and the pitched branch above both do. A non-pitched array is dense,
        // and callers rely on that: several of them derive a stride from the
        // shape rather than asking the array for its ld (see the
        // setLd(fine_shapes[0][d]) sites in
        // BlockLocalHierarchyDataRefactor). Leaving a shrunk array with the
        // wider ld of its previous shape leaves those two strides disagreeing
        // over the same buffer, which silently corrupts the data.
        __ldvs = __shape;
        linearized_width = 1;
        for (DIM d = 0; d < D - 1; d++) {
          linearized_width *= __shape[d];
        }
      }
    }
    // If cannot reuse existing allocation or there is no existing allocation
    if (!inplace_resizable) {
      initialize(shape);
      allocate(queue_idx);
    }
    return inplace_resizable;
  }
}

} // namespace mgard_x

#endif