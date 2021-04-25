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

template <typename T, uint32_t D>
Array<T, D>::Array(std::vector<size_t> shape) {
  this->host_allocated = false;
  this->device_allocated = false;
  std::reverse(shape.begin(), shape.end());
  this->shape = shape;
  if (shape.size() != D) {
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
  Handle<float, 1> handle;
  cudaMallocHelper((void **)&(this->ldvs_d), this->ldvs_h.size() * sizeof(int));
  cudaMemcpyAsyncHelper(handle, this->ldvs_d, this->ldvs_h.data(),
                        this->ldvs_h.size() * sizeof(int), AUTO, 0);

  this->device_allocated = true;
}

template <typename T, uint32_t D> Array<T, D>::Array(Array<T, D> &array) {
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
  Handle<float, 1> handle;

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

template <typename T, uint32_t D> Array<T, D>::~Array() {
  if (device_allocated) {
    cudaFreeHelper(ldvs_d);
    cudaFreeHelper(dv);
  }
  if (host_allocated) {
    delete[] hv;
  }
}

template <typename T, uint32_t D>
void Array<T, D>::loadData(T *data, size_t ld) {
  if (ld == 0) {
    ld = shape[0];
  }
  Handle<float, 1> handle;
  cudaMemcpy3DAsyncHelper(handle, dv, ldvs_h[0] * sizeof(T),
                          shape[0] * sizeof(T), shape[1], data, ld * sizeof(T),
                          shape[0] * sizeof(T), shape[1], shape[0] * sizeof(T),
                          shape[1], linearized_depth, AUTO, 0);
  handle.sync(0);
}

template <typename T, uint32_t D> T *Array<T, D>::getDataHost() {
  Handle<float, 1> handle;
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

template <typename T, uint32_t D> T *Array<T, D>::getDataDevice(size_t &ld) {
  ld = ldvs_h[0];
  return dv;
}

template <typename T, uint32_t D> std::vector<size_t> Array<T, D>::getShape() {
  return shape;
}

template <typename T, uint32_t D> T *Array<T, D>::get_dv() { return dv; }

template <typename T, uint32_t D> std::vector<int> Array<T, D>::get_ldvs_h() {
  return ldvs_h;
}

template <typename T, uint32_t D> int *Array<T, D>::get_ldvs_d() {
  return ldvs_d;
}

template class Array<double, 1>;
template class Array<float, 1>;
template class Array<double, 2>;
template class Array<float, 2>;
template class Array<double, 3>;
template class Array<float, 3>;
template class Array<double, 4>;
template class Array<float, 4>;
template class Array<double, 5>;
template class Array<float, 5>;

template class Array<unsigned char, 1>;

} // namespace mgard_cuda