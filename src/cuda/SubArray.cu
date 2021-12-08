/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: Jul 10, 2021
 */

#include <algorithm>

#include "cuda/CommonInternal.h"

#include "cuda/SubArray.h"

namespace mgard_cuda {

template <DIM D, typename T> SubArray<D, T>::SubArray() {
  lddv1 = 1;
  lddv2 = 1;
}

template <DIM D, typename T> SubArray<D, T>::SubArray(Array<D, T> &array) {
  this->shape = array.getShape();
  this->dv = array.get_dv();
  this->ldvs_h = array.get_ldvs_h();
  this->ldvs_d = array.get_ldvs_d();
  lddv1 = ldvs_h[0];
  lddv2 = ldvs_h[1];
}

template <DIM D, typename T>
SubArray<D, T>::SubArray(std::vector<SIZE> shape, T *dv,
                         std::vector<SIZE> ldvs_h, SIZE *ldvs_d) {
  this->shape = shape;
  this->dv = dv;
  this->ldvs_h = ldvs_h;
  this->ldvs_d = ldvs_d;
  lddv1 = ldvs_h[0];
  lddv2 = ldvs_h[1];
}

template <DIM D, typename T>
SubArray<D, T>::SubArray(std::vector<SIZE> shape, T *dv) {
  this->shape = shape;
  this->dv = dv;
  this->lddv1 = shape[0];
  if (D > 1) {
    this->lddv2 = shape[1];
  } else {
    this->lddv2 = 1;
  }
}

template <DIM D, typename T>
SubArray<D, T>::SubArray(SubArray<D, T> &subArray) {
  this->shape = subArray.shape;
  this->dv = subArray.dv;
  this->ldvs_h = subArray.ldvs_h;
  this->ldvs_d = subArray.ldvs_d;

  this->lddv1 = subArray.lddv1;
  this->lddv2 = subArray.lddv2;

  this->projected_dim0 = subArray.projected_dim0;
  this->projected_dim1 = subArray.projected_dim1;
  this->projected_dim2 = subArray.projected_dim2;
}

template <DIM D, typename T>
SubArray<D, T>::SubArray(const SubArray<D, T> &subArray) {
  this->shape = subArray.shape;
  this->dv = subArray.dv;
  this->ldvs_h = subArray.ldvs_h;
  this->ldvs_d = subArray.ldvs_d;

  this->lddv1 = subArray.lddv1;
  this->lddv2 = subArray.lddv2;

  this->projected_dim0 = subArray.projected_dim0;
  this->projected_dim1 = subArray.projected_dim1;
  this->projected_dim2 = subArray.projected_dim2;
}

template <DIM D, typename T>
SubArray<D, T> &SubArray<D, T>::operator=(const SubArray<D, T> &subArray) {
  this->shape = subArray.shape;
  this->dv = subArray.dv;
  this->ldvs_h = subArray.ldvs_h;
  this->ldvs_d = subArray.ldvs_d;

  this->lddv1 = subArray.lddv1;
  this->lddv2 = subArray.lddv2;

  this->projected_dim0 = subArray.projected_dim0;
  this->projected_dim1 = subArray.projected_dim1;
  this->projected_dim2 = subArray.projected_dim2;
  return *this;
}

template <DIM D, typename T>
void SubArray<D, T>::offset(std::vector<SIZE> idx) {
  dv += get_idx(ldvs_h, idx);
}

template <DIM D, typename T>
void SubArray<D, T>::resize(std::vector<SIZE> shape) {
  this->shape = shape;
}

template <DIM D, typename T>
void SubArray<D, T>::offset(SIZE dim, SIZE offset_value) {
  std::vector<SIZE> idx(D, 0);
  idx[dim] = offset_value;
  dv += get_idx(ldvs_h, idx);
}

template <DIM D, typename T>
void SubArray<D, T>::resize(SIZE dim, SIZE new_size) {
  shape[dim] = new_size;
}

template <DIM D, typename T>
void SubArray<D, T>::project(DIM dim0, DIM dim1, DIM dim2) {
  projected_dim0 = dim0;
  projected_dim1 = dim1;
  projected_dim2 = dim2;
  lddv1 = 1, lddv2 = 1;
  for (DIM d = projected_dim0; d < projected_dim1; d++) {
    lddv1 *= ldvs_h[d];
  }
  for (DIM d = projected_dim1; d < projected_dim2; d++) {
    lddv2 *= ldvs_h[d];
  }
}

// template <DIM D, typename T>
// MGARDm_EXEC
// T* SubArray<D, T>::operator()(IDX z, IDX y, IDX x) {
//   return dv + lddv2 * lddv1 * z + lddv1 * y + x;
// }

// template <DIM D> __forceinline__ __device__ LENGTH get_idx(SIZE *lds, SIZE
// *idx) {
//   LENGTH curr_stride = 1;
//   LENGTH ret_idx = 0;
//   for (DIM i = 0; i < D; i++) {
//     ret_idx += idx[i] * curr_stride;
//     curr_stride *= lds[i];
//   }
//   return ret_idx;
// }

template <DIM D, typename T> SubArray<D, T>::~SubArray() {
  // nothing needs to be released
}

template class SubArray<1, double>;
template class SubArray<1, float>;
template class SubArray<2, double>;
template class SubArray<2, float>;
template class SubArray<3, double>;
template class SubArray<3, float>;
template class SubArray<4, double>;
template class SubArray<4, float>;
template class SubArray<5, double>;
template class SubArray<5, float>;

template class SubArray<1, bool>;

template class SubArray<1, uint8_t>;
template class SubArray<1, uint16_t>;
template class SubArray<1, uint32_t>;
template class SubArray<1, uint64_t>;

template class SubArray<2, uint8_t>;
template class SubArray<2, uint16_t>;
template class SubArray<2, uint32_t>;
template class SubArray<2, uint64_t>;

template class SubArray<1, unsigned long long>;

// template class SubArray<1, QUANTIZED_INT>;
// template class SubArray<2, QUANTIZED_INT>;
// template class SubArray<3, QUANTIZED_INT>;
// template class SubArray<4, QUANTIZED_INT>;
// template class SubArray<5, QUANTIZED_INT>;

} // namespace mgard_cuda