/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#ifndef MGARD_CUDA_ARRAY_HPP
#define MGARD_CUDA_ARRAY_HPP
#include "Common.h"

#include "CommonInternal.h"
#include <vector>

#include "Array.h"

namespace mgard_cuda {

template <DIM D, typename T>
Array<D, T>::Array() {
  this->host_allocated = false;
  this->device_allocated = false;
}

template <DIM D, typename T>
Array<D, T>::Array(std::vector<SIZE> shape, bool pitched) {
  this->host_allocated = false;
  this->device_allocated = false;
  this->pitched = pitched;
  std::reverse(shape.begin(), shape.end());
  this->shape = shape;
  int ret = check_shape<D>(shape);
  if (ret == -1) {
    std::cerr << log::log_err
              << "Number of dimensions mismatch (" << D << " != " << shape.size() << "). mgard_cuda::Array not "
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
  Handle<1, float> handle;
  if (this->pitched) {
    size_t dv_pitch;
    cudaMalloc3DHelper(handle, (void **)&(this->dv), &dv_pitch,
                       this->shape[0] * sizeof(T), this->shape[1],
                       this->linearized_depth);
    this->ldvs_h.push_back((SIZE)dv_pitch / sizeof(T));
    for (int i = 1; i < D_padded; i++) {
      this->ldvs_h.push_back(this->shape[i]);
    }
  } else {
    cudaMallocHelper(handle, (void **)&(this->dv), this->shape[0] * this->shape[1] *
                       this->linearized_depth * sizeof(T));
    for (int i = 0; i < D_padded; i++) {
      this->ldvs_h.push_back(this->shape[i]);
    }

  }
  
  cudaMallocHelper(handle, (void **)&(this->ldvs_d), this->ldvs_h.size() * sizeof(SIZE));
  cudaMemcpyAsyncHelper(handle, this->ldvs_d, this->ldvs_h.data(),
                        this->ldvs_h.size() * sizeof(SIZE), AUTO, 0);

  // printf("Array constructured: %u %u %u %u %u %u\n", 
  //                         this->shape[0], this->shape[1], this->shape[2],
  //                         this->ldvs_h[0], this->ldvs_h[1], this->ldvs_h[2]);
  this->device_allocated = true;
}

template <DIM D, typename T> Array<D, T>::Array(const Array<D, T> &array) {
  
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
  Handle<1, float> handle;
  if (this->pitched) {
    size_t dv_pitch;
    cudaMalloc3DHelper(handle, (void **)&(this->dv), &dv_pitch,
                       this->shape[0] * sizeof(T), this->shape[1],
                       this->linearized_depth);
    this->ldvs_h.push_back((SIZE)dv_pitch / sizeof(T));
    for (int i = 1; i < D_padded; i++) {
      this->ldvs_h.push_back(this->shape[i]);
    }
  } else {
    cudaMallocHelper(handle, (void **)&(this->dv), this->shape[0] * this->shape[1] *
                       this->linearized_depth * sizeof(T));
    for (int i = 0; i < D_padded; i++) {
      this->ldvs_h.push_back(this->shape[i]);
    }
  }
  
  cudaMallocHelper(handle, (void **)&(this->ldvs_d), this->ldvs_h.size() * sizeof(SIZE));
  cudaMemcpyAsyncHelper(handle, this->ldvs_d, this->ldvs_h.data(),
                        this->ldvs_h.size() * sizeof(SIZE), AUTO, 0);
  cudaMemcpy3DAsyncHelper(
      handle, this->dv, this->ldvs_h[0] * sizeof(T), this->shape[0] * sizeof(T),
      this->shape[1], array.dv, array.ldvs_h[0] * sizeof(T),
      array.shape[0] * sizeof(T), array.shape[1], this->shape[0] * sizeof(T),
      this->shape[1], this->linearized_depth, AUTO, 0);
  this->device_allocated = true;

  // printf("Array copy1 %llu -> %llu\n", array.dv, this->dv);
}

template <DIM D, typename T> Array<D, T>::Array(Array<D, T> &array) {
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
  Handle<1, float> handle;
  if (this->pitched) {
    size_t dv_pitch;
    cudaMalloc3DHelper(handle, (void **)&(this->dv), &dv_pitch,
                       this->shape[0] * sizeof(T), this->shape[1],
                       this->linearized_depth);
    this->ldvs_h.push_back((SIZE)dv_pitch / sizeof(T));
    for (int i = 1; i < D_padded; i++) {
      this->ldvs_h.push_back(this->shape[i]);
    }
  } else {
    cudaMallocHelper(handle, (void **)&(this->dv), this->shape[0] * this->shape[1] *
                       this->linearized_depth * sizeof(T));
    for (int i = 0; i < D_padded; i++) {
      this->ldvs_h.push_back(this->shape[i]);
    }
  }
  
  cudaMallocHelper(handle, (void **)&(this->ldvs_d), this->ldvs_h.size() * sizeof(SIZE));
  cudaMemcpyAsyncHelper(handle, this->ldvs_d, this->ldvs_h.data(),
                        this->ldvs_h.size() * sizeof(SIZE), AUTO, 0);
  cudaMemcpy3DAsyncHelper(
      handle, this->dv, this->ldvs_h[0] * sizeof(T), this->shape[0] * sizeof(T),
      this->shape[1], array.dv, array.ldvs_h[0] * sizeof(T),
      array.shape[0] * sizeof(T), array.shape[1], this->shape[0] * sizeof(T),
      this->shape[1], this->linearized_depth, AUTO, 0);
  this->device_allocated = true;
  // printf("Array copy2 %llu -> %llu\n", array.dv, this->dv);
}


template <DIM D, typename T> Array<D, T>& Array<D, T>::operator = (const Array<D, T> &array) {
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
  Handle<1, float> handle;
  if (this->pitched) {
    size_t dv_pitch;
    cudaMalloc3DHelper(handle, (void **)&(this->dv), &dv_pitch,
                       this->shape[0] * sizeof(T), this->shape[1],
                       this->linearized_depth);
    this->ldvs_h.push_back((SIZE)dv_pitch / sizeof(T));
    for (int i = 1; i < D_padded; i++) {
      this->ldvs_h.push_back(this->shape[i]);
    }
  } else {
    cudaMallocHelper(handle, (void **)&(this->dv), this->shape[0] * this->shape[1] *
                       this->linearized_depth * sizeof(T));
    for (int i = 0; i < D_padded; i++) {
      this->ldvs_h.push_back(this->shape[i]);
    }
  }
  
  cudaMallocHelper(handle, (void **)&(this->ldvs_d), this->ldvs_h.size() * sizeof(SIZE));
  cudaMemcpyAsyncHelper(handle, this->ldvs_d, this->ldvs_h.data(),
                        this->ldvs_h.size() * sizeof(SIZE), AUTO, 0);
  cudaMemcpy3DAsyncHelper(
      handle, this->dv, this->ldvs_h[0] * sizeof(T), this->shape[0] * sizeof(T),
      this->shape[1], array.dv, array.ldvs_h[0] * sizeof(T),
      array.shape[0] * sizeof(T), array.shape[1], this->shape[0] * sizeof(T),
      this->shape[1], this->linearized_depth, AUTO, 0);
  this->device_allocated = true;
  // printf("Array copy3 %llu -> %llu\n", array.dv, this->dv);
  return *this;
}


template <DIM D, typename T> Array<D, T>::Array(Array<D, T> && array) {
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
  this->ldvs_h = array.ldvs_h;
  this->ldvs_d = array.ldvs_d;
  this->dv = array.dv;
  array.device_allocated = false;
  this->device_allocated = true;
  // printf("Array move %llu -> %llu\n", array.dv, this->dv);
}

template <DIM D, typename T> Array<D, T>::~Array() {
  // printf("Array deleted %llu\n", this->dv);
  if (device_allocated) {
    cudaFreeHelper(ldvs_d);
    cudaFreeHelper(dv);
  }
  if (host_allocated) {
    cudaFreeHostHelper(hv);
  }

}

template <DIM D, typename T>
void Array<D, T>::loadData(const T *data, SIZE ld) {
  if (ld == 0) {
    ld = shape[0];
  }
  Handle<1, float> handle;
  cudaMemcpy3DAsyncHelper(handle, dv, ldvs_h[0] * sizeof(T),
                          shape[0] * sizeof(T), shape[1], data, ld * sizeof(T),
                          shape[0] * sizeof(T), shape[1], shape[0] * sizeof(T),
                          shape[1], linearized_depth, AUTO, 0);
  handle.sync(0);
}

template <DIM D, typename T> T *Array<D, T>::getDataHost() {
  Handle<1, float> handle;
  if (!host_allocated) {
    cudaMallocHostHelper((void **)&hv,
                         sizeof(T) * shape[0] * shape[1] * linearized_depth);
    host_allocated = true;
  }
  cudaMemcpy3DAsyncHelper(
      handle, hv, shape[0] * sizeof(T), shape[0] * sizeof(T), shape[1], dv,
      ldvs_h[0] * sizeof(T), shape[0] * sizeof(T), shape[1],
      shape[0] * sizeof(T), shape[1], linearized_depth, AUTO, 0);
  handle.sync(0);
  return hv;
}

template <DIM D, typename T> T *Array<D, T>::getDataDevice(SIZE &ld) {
  ld = ldvs_h[0];
  return dv;
}

template <DIM D, typename T> std::vector<SIZE> Array<D, T>::getShape() {
  return shape;
}

template <DIM D, typename T> T *Array<D, T>::get_dv() { return dv; }

template <DIM D, typename T> std::vector<SIZE> Array<D, T>::get_ldvs_h() {
  return ldvs_h;
}

template <DIM D, typename T> SIZE *Array<D, T>::get_ldvs_d() {
  return ldvs_d;
}
}

#endif