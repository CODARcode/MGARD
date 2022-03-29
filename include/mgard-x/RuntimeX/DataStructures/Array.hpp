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
  this->host_allocated = false;
  this->device_allocated = false;
}

template <DIM D, typename T, typename DeviceType>
Array<D, T, DeviceType>::Array(std::vector<SIZE> _shape, bool pitched,
                               bool managed) {
  this->host_allocated = false;
  this->device_allocated = false;
  this->pitched = pitched;
  this->managed = managed;
  std::reverse(_shape.begin(), _shape.end());
  this->_shape = _shape;
  int ret = check_shape<D>(_shape);
  if (ret == -1) {
    std::cerr << log::log_err << "Number of dimensions mismatch (" << D
              << " != " << _shape.size()
              << "). mgard_x::Array not "
                 "initialized!\n";
    return;
  }

  this->D_padded = D;
  if (D < 3) {
    this->D_padded = 3;
  }
  if (D % 2 == 0) {
    this->D_padded = D + 1;
  }
  // padding dimensions
  for (int d = this->_shape.size(); d < D_padded; d++) {
    this->_shape.push_back(1);
  }
  this->linearized_depth = 1;
  for (int i = 2; i < D_padded; i++) {
    this->linearized_depth *= this->_shape[i];
  }

  if (this->pitched) {
    if (!this->managed) {
      SIZE ld = 0;
      MemoryManager<DeviceType>().MallocND(
          this->dv, this->_shape[0], this->_shape[1] * this->linearized_depth,
          ld, 0);
      this->_ldvs.push_back(ld);
      for (int i = 1; i < D_padded; i++) {
        this->_ldvs.push_back(this->_shape[i]);
      }
    } else {
      std::cerr << log::log_err
                << "Does not support managed memory in pitched mode.\n";
    }
  } else {
    if (!this->managed) {
      MemoryManager<DeviceType>().Malloc1D(
          this->dv, this->_shape[0] * this->_shape[1] * this->linearized_depth,
          0);
      for (int i = 0; i < D_padded; i++) {
        this->_ldvs.push_back(this->_shape[i]);
      }
    } else {
      MemoryManager<DeviceType>().MallocManaged1D(
          this->dv, this->_shape[0] * this->_shape[1] * this->linearized_depth,
          0);
      for (int i = 0; i < D_padded; i++) {
        this->_ldvs.push_back(this->_shape[i]);
      }
    }
  }

  this->device_allocated = true;
}

template <DIM D, typename T, typename DeviceType>
void Array<D, T, DeviceType>::copy(const Array<D, T, DeviceType> &array) {
  this->free();
  this->_shape = array._shape;
  this->pitched = array.pitched;
  this->D_padded = array.D_padded;
  this->linearized_depth = array.linearized_depth;
  this->_ldvs.resize(D_padded);

  if (this->pitched) {
    SIZE ld = 0;
    MemoryManager<DeviceType>().MallocND(
        this->dv, this->_shape[0], this->_shape[1] * this->linearized_depth, ld,
        0);
    this->_ldvs[0] = ld;
    for (int i = 1; i < D_padded; i++) {
      this->_ldvs[i] = this->_shape[i];
    }
  } else {
    MemoryManager<DeviceType>().Malloc1D(
        this->dv, this->_shape[0] * this->_shape[1] * this->linearized_depth,
        0);
    for (int i = 0; i < D_padded; i++) {
      this->_ldvs[i] = this->_shape[i];
    }
  }

  MemoryManager<DeviceType>().CopyND(
      this->dv, this->_ldvs[0], array.dv, array._ldvs[0], array._shape[0],
      array._shape[1] * this->linearized_depth, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  this->device_allocated = true;
}

template <DIM D, typename T, typename DeviceType>
void Array<D, T, DeviceType>::move(Array<D, T, DeviceType> &&array) {
  this->free();
  this->_shape = array._shape;
  this->pitched = array.pitched;
  this->D_padded = array.D_padded;
  this->linearized_depth = array.linearized_depth;

  if (array.device_allocated) {
    this->dv = array.dv;
    this->_ldvs = array._ldvs;
    this->device_allocated = true;
    array.device_allocated = false;
    array.dv = NULL;
  }
}

template <DIM D, typename T, typename DeviceType>
void Array<D, T, DeviceType>::memset(int value) {
  if (this->pitched) {
    MemoryManager<DeviceType>().MemsetND(
        this->dv, this->_ldvs[0], this->_shape[0],
        this->_shape[1] * this->linearized_depth, value, 0);
  } else {
    MemoryManager<DeviceType>().Memset1D(
        this->dv, this->_shape[0] * this->_shape[1] * this->linearized_depth,
        value, 0);
  }
  DeviceRuntime<DeviceType>::SyncQueue(0);
}

template <DIM D, typename T, typename DeviceType>
void Array<D, T, DeviceType>::free() {
  if (device_allocated) {
    MemoryManager<DeviceType>().Free(dv);
    device_allocated = false;
    dv = NULL;
  }
  if (host_allocated && !keepHostCopy) {
    MemoryManager<DeviceType>().FreeHost(hv);
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
void Array<D, T, DeviceType>::load(const T *data, SIZE ld) {
  if (ld == 0) {
    ld = _shape[0];
  }
  MemoryManager<DeviceType>().CopyND(dv, _ldvs[0], data, ld, _shape[0],
                                     _shape[1] * linearized_depth, 0);

  DeviceRuntime<DeviceType>::SyncQueue(0);
}

template <DIM D, typename T, typename DeviceType>
T *Array<D, T, DeviceType>::hostCopy(bool keep) {
  if (!device_allocated) {
    std::cout << log::log_err << "device buffer not initialized.\n";
    exit(-1);
  }
  if (!host_allocated) {
    MemoryManager<DeviceType>().MallocHost(
        hv, _shape[0] * _shape[1] * linearized_depth, 0);
    host_allocated = true;
  }
  MemoryManager<DeviceType>().CopyND(hv, _shape[0], dv, _ldvs[0], _shape[0],
                                     _shape[1] * linearized_depth, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  keepHostCopy = keep;
  return hv;
}

template <DIM D, typename T, typename DeviceType>
T *Array<D, T, DeviceType>::data(SIZE &ld) {
  ld = _ldvs[0];
  return dv;
}

template <DIM D, typename T, typename DeviceType>
std::vector<SIZE> &Array<D, T, DeviceType>::shape() {
  return _shape;
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
bool Array<D, T, DeviceType>::is_pitched() {
  return pitched;
}
} // namespace mgard_x

#endif