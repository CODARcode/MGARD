/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include <algorithm>

#include "cuda/Array.h"
#include "cuda/Common.h"
#include "cuda/CommonInternal.h"

namespace mgard_cuda {

template <uint32_t D, typename T>
Array<D, T>::Array(std::vector<size_t> shape) {
  this->host_allocated = false;
  this->device_allocated = false;
  std::reverse(shape.begin(), shape.end());
  this->shape = shape;
  int ret = check_shape<D>(shape);
  if (ret == -1) {
    std::cerr << log_err
              << "Number of dimensions mismatch. mgard_cuda::Array not "
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
  size_t dv_pitch;
  cudaMalloc3DHelper((void **)&(this->dv), &dv_pitch,
                     this->shape[0] * sizeof(T), this->shape[1],
                     this->linearized_depth);
  this->ldvs_h.push_back(dv_pitch / sizeof(T));
  for (int i = 1; i < D_padded; i++) {
    this->ldvs_h.push_back(this->shape[i]);
  }
  Handle<1, float> handle;
  cudaMallocHelper((void **)&(this->ldvs_d), this->ldvs_h.size() * sizeof(int));
  cudaMemcpyAsyncHelper(handle, this->ldvs_d, this->ldvs_h.data(),
                        this->ldvs_h.size() * sizeof(int), AUTO, 0);

  this->device_allocated = true;
}

template <uint32_t D, typename T> Array<D, T>::Array(Array<D, T> &array) {
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
  size_t dv_pitch;
  cudaMalloc3DHelper((void **)&dv, &dv_pitch, this->shape[0] * sizeof(T),
                     this->shape[1], linearized_depth);
  ldvs_h.push_back(dv_pitch / sizeof(T));
  for (int i = 1; i < this->D_padded; i++) {
    this->ldvs_h.push_back(this->shape[i]);
  }
  Handle<1, float> handle;

  cudaMallocHelper((void **)&(this->ldvs_d), this->ldvs_h.size() * sizeof(int));
  cudaMemcpyAsyncHelper(handle, this->ldvs_d, this->ldvs_h.data(),
                        this->ldvs_h.size() * sizeof(int), AUTO, 0);

  cudaMemcpy3DAsyncHelper(
      handle, this->dv, this->ldvs_h[0] * sizeof(T), this->shape[0] * sizeof(T),
      this->shape[1], array.dv, this->ldvs_h[0] * sizeof(T),
      this->shape[0] * sizeof(T), this->shape[1], this->shape[0] * sizeof(T),
      this->shape[1], this->linearized_depth, AUTO, 0);

  this->device_allocated = true;
}

template <uint32_t D, typename T> Array<D, T>::~Array() {
  if (device_allocated) {
    cudaFreeHelper(ldvs_d);
    cudaFreeHelper(dv);
  }
  if (host_allocated) {
    delete[] hv;
  }
}

template <uint32_t D, typename T>
void Array<D, T>::loadData(T *data, size_t ld) {
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

template <uint32_t D, typename T> T *Array<D, T>::getDataHost() {
  Handle<1, float> handle;
  if (!host_allocated) {
    cudaMallocHostHelper((void **)&hv,
                         sizeof(T) * shape[0] * shape[1] * linearized_depth);
  }
  cudaMemcpy3DAsyncHelper(
      handle, hv, shape[0] * sizeof(T), shape[0] * sizeof(T), shape[1], dv,
      ldvs_h[0] * sizeof(T), shape[0] * sizeof(T), shape[1],
      shape[0] * sizeof(T), shape[1], linearized_depth, AUTO, 0);
  handle.sync(0);
  return hv;
}

template <uint32_t D, typename T> T *Array<D, T>::getDataDevice(size_t &ld) {
  ld = ldvs_h[0];
  return dv;
}

template <uint32_t D, typename T> std::vector<size_t> Array<D, T>::getShape() {
  return shape;
}

template <uint32_t D, typename T> T *Array<D, T>::get_dv() { return dv; }

template <uint32_t D, typename T> std::vector<int> Array<D, T>::get_ldvs_h() {
  return ldvs_h;
}

template <uint32_t D, typename T> int *Array<D, T>::get_ldvs_d() {
  return ldvs_d;
}

template class Array<1, double>;
template class Array<1, float>;
template class Array<2, double>;
template class Array<2, float>;
template class Array<3, double>;
template class Array<3, float>;
template class Array<4, double>;
template class Array<4, float>;
template class Array<5, double>;
template class Array<5, float>;

template class Array<1, unsigned char>;

} // namespace mgard_cuda