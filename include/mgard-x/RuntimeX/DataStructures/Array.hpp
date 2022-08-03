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
Array<D, T, DeviceType>::Array(std::vector<SIZE> _shape, bool pitched,
                               bool managed, int queue_idx) {
  initialize(_shape);
  allocate(pitched, managed, queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void Array<D, T, DeviceType>::initialize(std::vector<SIZE> shape) {
  if (shape.size() != D) {
    std::cerr << log::log_err << "Number of dimensions mismatch ("
              << shape.size() << "!=" << D
              << "). mgard_x::Array not initialized!\n";
    exit(-1);
  }
  shape_org = shape;
  __shape = shape;
  free();
  D_padded = D;
  if (D < 3) {
    D_padded = 3;
  }
  if (D % 2 == 0) {
    D_padded = D + 1;
  }
  D_pad = D_padded - D;
  // padding dimensions
  for (DIM d = 0; d < D_pad; d++) {
    __shape.insert(__shape.begin(), 1);
  }
  __ldvs = __shape;

  _shape = __shape;
  _ldvs = __shape;
  std::reverse(_shape.begin(), _shape.end());
  std::reverse(_ldvs.begin(), _ldvs.end());
  linearized_depth = 1;
  for (DIM d = 2; d < D_padded; d++) {
    linearized_depth *= _shape[d];
  }
  linearized_width = 1;
  for (DIM d = 0; d < D_padded-1; d++) {
    linearized_width *= __shape[d];
  }
  host_allocated = false;
  device_allocated = false;
  pitched = false;
  managed = false;
}

template <DIM D, typename T, typename DeviceType>
void Array<D, T, DeviceType>::allocate(bool pitched, bool managed, int queue_idx) {
  this->pitched = pitched;
  this->managed = managed;
  if (this->pitched) {
    if (!this->managed) {
      SIZE ld = 0;
      MemoryManager<DeviceType>::MallocND(
          dv, __shape[D_padded-1], linearized_width, ld, queue_idx);
      _ldvs[0] = ld;
      __ldvs[D_padded-1] = ld;
    } else {
      std::cerr << log::log_err
                << "Does not support managed memory in pitched mode.\n";
    }
  } else {
    if (!this->managed) {
      MemoryManager<DeviceType>::Malloc1D(
          dv, __shape[D_padded-1] * linearized_width, queue_idx);
    } else {
      MemoryManager<DeviceType>::MallocManaged1D(
          dv, __shape[D_padded-1] * linearized_width, queue_idx);
    }
  }
  device_allocated = true;
}

template <DIM D, typename T, typename DeviceType>
void Array<D, T, DeviceType>::copy(const Array<D, T, DeviceType> &array, int queue_idx) {
  initialize(array.shape_org);
  allocate(array.pitched, array.managed, queue_idx);
  MemoryManager<DeviceType>::CopyND(
      dv, __ldvs[D_padded-1], array.dv, array.__ldvs[array.D_padded-1], array.__shape[D_padded-1],
      array.linearized_width, queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void Array<D, T, DeviceType>::move(Array<D, T, DeviceType> &&array) {
  initialize(array.shape_org);
  this->pitched = array.pitched;
  this->managed = array.managed;
  if (array.device_allocated) {
    this->dv = array.dv;
    this->_ldvs = array._ldvs;
    this->__ldvs = array.__ldvs;
    this->device_allocated = true;
    array.device_allocated = false;
    array.dv = NULL;
  }
}

template <DIM D, typename T, typename DeviceType>
void Array<D, T, DeviceType>::memset(int value, int queue_idx) {
  if (this->pitched) {
    MemoryManager<DeviceType>::MemsetND(
        dv, __ldvs[D_padded-1], __shape[D_padded-1],
        linearized_width, value, queue_idx);
  } else {
    MemoryManager<DeviceType>::Memset1D(
        dv, __ldvs[D_padded-1] * linearized_width,
        value, queue_idx);
  }
  // DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
}

template <DIM D, typename T, typename DeviceType>
void Array<D, T, DeviceType>::free(int queue_idx) {
  if (device_allocated) {
    MemoryManager<DeviceType>::Free(dv, queue_idx);
    device_allocated = false;
    dv = NULL;
  }
  if (host_allocated && !keepHostCopy) {
    MemoryManager<DeviceType>::FreeHost(hv, queue_idx);
    host_allocated = false;
    hv = NULL;
  }
}

template <DIM D, typename T, typename DeviceType>
Array<D, T, DeviceType>::Array(const Array<D, T, DeviceType> &array) {
  this->copy(array);
}

template <DIM D, typename T, typename DeviceType>
Array<D, T, DeviceType> &Array<D, T, DeviceType>::
operator=(const Array<D, T, DeviceType> &array) {
  // printf("Array operator =\n");
  this->copy(array);
  return *this;
}

template <DIM D, typename T, typename DeviceType>
Array<D, T, DeviceType> &Array<D, T, DeviceType>::
operator=(Array<D, T, DeviceType> &&array) {
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
  if (ld == 0) {
    ld = _shape[0];
  }
  MemoryManager<DeviceType>::CopyND(dv, __ldvs[D_padded-1], data, ld, __shape[D_padded-1],
                                    linearized_width, queue_idx);

  DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
}

template <DIM D, typename T, typename DeviceType>
T *Array<D, T, DeviceType>::hostCopy(bool keep, int queue_idx) {
  if (!device_allocated) {
    std::cout << log::log_err << "device buffer not initialized.\n";
    exit(-1);
  }
  if (!host_allocated) {
    MemoryManager<DeviceType>::MallocHost(
        hv, __shape[D_padded-1] * linearized_width, queue_idx);
    host_allocated = true;
  }
  MemoryManager<DeviceType>::CopyND(hv, __shape[D_padded-1], dv, __ldvs[D_padded-1], __shape[D_padded-1],
                                     linearized_width, queue_idx);
  DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
  keepHostCopy = keep;
  return hv;
}

template <DIM D, typename T, typename DeviceType>
T *Array<D, T, DeviceType>::data(SIZE &ld) {
  ld = __ldvs[D_padded-1];
  return dv;
}

template <DIM D, typename T, typename DeviceType>
std::vector<SIZE> &Array<D, T, DeviceType>::shape() {
  return _shape;
}

template <DIM D, typename T, typename DeviceType>
SIZE Array<D, T, DeviceType>::shape(DIM d) {
  return __shape[d + D_pad];
}


template <DIM D, typename T, typename DeviceType>
T *Array<D, T, DeviceType>::data() {
  return dv;
}

template <DIM D, typename T, typename DeviceType>
std::vector<SIZE> Array<D, T, DeviceType>::ld() {
  return _ldvs;
}

template <DIM D, typename T, typename DeviceType>
SIZE Array<D, T, DeviceType>::ld(DIM d) {
  return __ldvs[d + D_pad];
}

template <DIM D, typename T, typename DeviceType>
bool Array<D, T, DeviceType>::isPitched() {
  return pitched;
}

template <DIM D, typename T, typename DeviceType>
bool Array<D, T, DeviceType>::isManaged() {
  return managed;
}

} // namespace mgard_x

#endif