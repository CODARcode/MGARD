/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_ARRAY_HPP
#define MGARD_X_ARRAY_HPP
// #include "Common.h"

// #include "CommonInternal.h"
#include <vector>
#include <algorithm>

#include "Array.h"


// #include "../DeviceAdapters/DeviceAdapter.h"

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
Array<D, T, DeviceType>::Array() {
  // printf("Array allocate empty\n");
  this->host_allocated = false;
  this->device_allocated = false;
}

template <DIM D, typename T, typename DeviceType>
Array<D, T, DeviceType>::Array(std::vector<SIZE> shape, bool pitched) {
  // printf("Array allocate\n");
  this->host_allocated = false;
  this->device_allocated = false;
  this->pitched = pitched;
  std::reverse(shape.begin(), shape.end());
  this->shape = shape;
  int ret = check_shape<D>(shape);
  if (ret == -1) {
    std::cerr << log::log_err
              << "Number of dimensions mismatch (" << D << " != " << shape.size() << "). mgard_x::Array not "
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
  for (int d = this->shape.size(); d < D_padded; d++) {
    this->shape.push_back(1);
  }
  this->linearized_depth = 1;
  for (int i = 2; i < D_padded; i++) {
    this->linearized_depth *= this->shape[i];
  }

  if (this->pitched) {
    SIZE ld = 0;
    MemoryManager<DeviceType>().MallocND(
                      this->dv, this->shape[0], this->shape[1] * this->linearized_depth,
                      ld, 0);
    this->ldvs_h.push_back(ld);
    for (int i = 1; i < D_padded; i++) {
      this->ldvs_h.push_back(this->shape[i]);
    }
  } else {
    MemoryManager<DeviceType>().Malloc1D(
                      this->dv, this->shape[0] * this->shape[1] * this->linearized_depth,
                      0);
    for (int i = 0; i < D_padded; i++) {
      this->ldvs_h.push_back(this->shape[i]);
    }

  }

  MemoryManager<DeviceType>().Malloc1D(
                      this->ldvs_d, this->ldvs_h.size(),
                      0);
  
  MemoryManager<DeviceType>().Copy1D(
                      this->ldvs_d, this->ldvs_h.data(),
                        this->ldvs_h.size(),
                      0);

  this->device_allocated = true;
}

template <DIM D, typename T, typename DeviceType> 
Array<D, T, DeviceType>::Array(const Array<D, T, DeviceType> &array) {
  // printf("Array copy 1\n");
  this->host_allocated = false;
  this->device_allocated = false;
  this->shape = array.shape;
  this->pitched = array.pitched;
  this->D_padded = D;
  if (D < 3) {
    this->D_padded = 3;
  }
  if (D % 2 == 0) {
    this->D_padded = D + 1;
  }
  // padding dimensions
  for (int d = this->shape.size(); d < this->D_padded; d++) {
    this->shape.push_back(1);
  }

  this->linearized_depth = 1;
  for (int i = 2; i < this->D_padded; i++) {
    this->linearized_depth *= this->shape[i];
  }

  if (this->pitched) {
    SIZE ld = 0;
    MemoryManager<DeviceType>().MallocND(
                      this->dv, this->shape[0], this->shape[1] * this->linearized_depth,
                      ld, 0);
    this->ldvs_h.push_back(ld);
    for (int i = 1; i < D_padded; i++) {
      this->ldvs_h.push_back(this->shape[i]);
    }
  } else {
    MemoryManager<DeviceType>().Malloc1D(
                      this->dv, this->shape[0] * this->shape[1] * this->linearized_depth,
                      0);
    for (int i = 0; i < D_padded; i++) {
      this->ldvs_h.push_back(this->shape[i]);
    }
  }
  
  MemoryManager<DeviceType>().Malloc1D(
                      this->ldvs_d, this->ldvs_h.size(),
                      0);
  MemoryManager<DeviceType>().Copy1D(
                      this->ldvs_d, this->ldvs_h.data(),
                        this->ldvs_h.size(),
                      0);

  MemoryManager<DeviceType>().CopyND(
          this->dv, this->ldvs_h[0], array.dv, array.ldvs_h[0], 
          array.shape[0], array.shape[1]*this->linearized_depth, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  this->device_allocated = true;
}

template <DIM D, typename T, typename DeviceType> 
Array<D, T, DeviceType>::Array(Array<D, T, DeviceType> &array) {
  // printf("Array copy2\n");
  this->host_allocated = false;
  this->device_allocated = false;
  this->shape = array.shape;
  this->pitched = array.pitched;
  this->D_padded = D;
  if (D < 3) {
    this->D_padded = 3;
  }
  if (D % 2 == 0) {
    this->D_padded = D + 1;
  }
  // padding dimensions
  for (int d = this->shape.size(); d < this->D_padded; d++) {
    this->shape.push_back(1);
  }

  this->linearized_depth = 1;
  for (int i = 2; i < this->D_padded; i++) {
    this->linearized_depth *= this->shape[i];
  }

  if (this->pitched) {
    SIZE ld = 0;
    MemoryManager<DeviceType>().MallocND(
                      this->dv, this->shape[0], this->shape[1] * this->linearized_depth,
                      ld, 0);
    this->ldvs_h.push_back(ld);
    for (int i = 1; i < D_padded; i++) {
      this->ldvs_h.push_back(this->shape[i]);
    }
  } else {
    MemoryManager<DeviceType>().Malloc1D(
                      this->dv, this->shape[0] * this->shape[1] * this->linearized_depth,
                      0);
    for (int i = 0; i < D_padded; i++) {
      this->ldvs_h.push_back(this->shape[i]);
    }
  }
  
  MemoryManager<DeviceType>().Malloc1D(
                      this->ldvs_d, this->ldvs_h.size(),
                      0);
  MemoryManager<DeviceType>().Copy1D(
                      this->ldvs_d, this->ldvs_h.data(),
                        this->ldvs_h.size(),
                      0);
  MemoryManager<DeviceType>().CopyND(
          this->dv, this->ldvs_h[0], array.dv, array.ldvs_h[0], 
          array.shape[0], array.shape[1]*this->linearized_depth, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  this->device_allocated = true;
}

template <DIM D, typename T, typename DeviceType> 
void Array<D, T, DeviceType>::memset(int value) {
  if (this->pitched) {
    MemoryManager<DeviceType>().MemsetND(this->dv, this->ldvs_h[0], 
                                         this->shape[0], 
                                         this->shape[1] * this->linearized_depth,
                                         value, 0);
  } else {
    MemoryManager<DeviceType>().Memset1D(this->dv,
                                         this->shape[0] *
                                         this->shape[1] * this->linearized_depth,
                                         value, 0);
  }
  DeviceRuntime<DeviceType>::SyncQueue(0);
}

template <DIM D, typename T, typename DeviceType> 
Array<D, T, DeviceType>& Array<D, T, DeviceType>::operator = (const Array<D, T, DeviceType> &array) {
  // printf("Array operator =\n");
  this->host_allocated = false;
  this->device_allocated = false;
  this->shape = array.shape;
  this->pitched = array.pitched;
  this->D_padded = D;
  if (D < 3) {
    this->D_padded = 3;
  }
  if (D % 2 == 0) {
    this->D_padded = D + 1;
  }
  // padding dimensions
  for (int d = this->shape.size(); d < this->D_padded; d++) {
    this->shape.push_back(1);
  }

  this->linearized_depth = 1;
  for (int i = 2; i < this->D_padded; i++) {
    this->linearized_depth *= this->shape[i];
  }

  if (this->pitched) {
    SIZE ld = 0;
    MemoryManager<DeviceType>().MallocND(
                      this->dv, this->shape[0], this->shape[1] * this->linearized_depth,
                      ld, 0);
    this->ldvs_h.push_back(ld);
    for (int i = 1; i < D_padded; i++) {
      this->ldvs_h.push_back(this->shape[i]);
    }
  } else {
    MemoryManager<DeviceType>().Malloc1D(
                      this->dv, this->shape[0] * this->shape[1] * this->linearized_depth,
                      0);
    for (int i = 0; i < D_padded; i++) {
      this->ldvs_h.push_back(this->shape[i]);
    }
  }
  
  MemoryManager<DeviceType>().Malloc1D(
                      this->ldvs_d, this->ldvs_h.size(),
                      0);
  MemoryManager<DeviceType>().Copy1D(
                      this->ldvs_d, this->ldvs_h.data(),
                        this->ldvs_h.size(),
                      0);
  MemoryManager<DeviceType>().CopyND(
          this->dv, this->ldvs_h[0], array.dv, array.ldvs_h[0], 
          array.shape[0], array.shape[1]*this->linearized_depth, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  this->device_allocated = true;
  return *this;
}


template <DIM D, typename T, typename DeviceType> 
Array<D, T, DeviceType>::Array(Array<D, T, DeviceType> && array) {
  this->host_allocated = false;
  this->device_allocated = false;
  this->shape = array.shape;
  this->D_padded = D;
  if (D < 3) {
    this->D_padded = 3;
  }
  if (D % 2 == 0) {
    this->D_padded = D + 1;
  }
  // padding dimensions
  for (int d = this->shape.size(); d < this->D_padded; d++) {
    this->shape.push_back(1);
  }

  this->linearized_depth = 1;
  for (int i = 2; i < this->D_padded; i++) {
    this->linearized_depth *= this->shape[i];
  }
  DeviceRuntime<DeviceType>::SyncQueue(0);
  this->ldvs_h = array.ldvs_h;
  this->ldvs_d = array.ldvs_d;
  this->dv = array.dv;

  array.device_allocated = false;
  this->device_allocated = true;
}

template <DIM D, typename T, typename DeviceType>
Array<D, T, DeviceType>::~Array() {
  if (device_allocated) {
    MemoryManager<DeviceType>().Free(ldvs_d);
    MemoryManager<DeviceType>().Free(dv);
  }
  if (host_allocated) {
    MemoryManager<DeviceType>().FreeHost(hv);
  }

}

template <DIM D, typename T, typename DeviceType>
void Array<D, T, DeviceType>::loadData(const T *data, SIZE ld) {
  if (ld == 0) {
    ld = shape[0];
  }
  MemoryManager<DeviceType>().CopyND(
                  dv, ldvs_h[0], data, ld,
                  shape[0], shape[1] * linearized_depth, 0);

  DeviceRuntime<DeviceType>::SyncQueue(0);
}

template <DIM D, typename T, typename DeviceType> 
T *Array<D, T, DeviceType>::getDataHost() {
  if (!device_allocated) { 
    std::cout << log::log_err << "device buffer not initialized.\n";
    exit(-1);
  }
  if (!host_allocated) {
    MemoryManager<DeviceType>().MallocHost(
                hv, shape[0] * shape[1] * linearized_depth, 0);
    host_allocated = true;
  }
  MemoryManager<DeviceType>().CopyND(
    hv, shape[0], dv, ldvs_h[0], shape[0], shape[1] * linearized_depth, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  return hv;
}

template <DIM D, typename T, typename DeviceType> 
T *Array<D, T, DeviceType>::getDataDevice(SIZE &ld) {
  ld = ldvs_h[0];
  return dv;
}

template <DIM D, typename T, typename DeviceType> 
std::vector<SIZE>& Array<D, T, DeviceType>::getShape() {
  return shape;
}

template <DIM D, typename T, typename DeviceType> 
T *Array<D, T, DeviceType>::get_dv() { return dv; }

template <DIM D, typename T, typename DeviceType> 
std::vector<SIZE> Array<D, T, DeviceType>::get_ldvs_h() {
  return ldvs_h;
}

template <DIM D, typename T, typename DeviceType> 
SIZE *Array<D, T, DeviceType>::get_ldvs_d() {
  return ldvs_d;
}

template <DIM D, typename T, typename DeviceType> 
bool Array<D, T, DeviceType>::is_pitched() {
  return pitched;
}
}

#endif