/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: Jul 20, 2021
 */

#ifndef MGARD_CUDA_SUBARRAY_HPP
#define MGARD_CUDA_SUBARRAY_HPP
#include "Common.h"
#include "CommonInternal.h"
#include <vector>
#include <iostream>

namespace mgard_cuda {

template <DIM D, typename T, typename DeviceType> class SubArray {
public:
  SubArray();
  SubArray(Array<D, T, DeviceType> &array, bool get_host_pointer = false);
  SubArray(std::vector<SIZE> shape, T * dv, std::vector<SIZE> ldvs_h, SIZE * ldvs_d);
  SubArray(std::vector<SIZE> shape, T * dv);
  SubArray(SubArray<D, T, DeviceType> &subArray);
  SubArray(const SubArray<D, T, DeviceType> &subArray);
  SubArray<D, T, DeviceType>& operator = (const SubArray<D, T, DeviceType> &subArray);
  void offset(std::vector<SIZE> idx);
  void resize(std::vector<SIZE> shape);
  void offset(DIM dim, SIZE offset_value);
  void resize(DIM dim, SIZE new_size);
  void project(DIM dim0, DIM dim1, DIM dim2);
 
  MGARDm_CONT_EXEC
  T* operator()(SIZE * idx) {
    LENGTH curr_stride = 1;
    LENGTH offset = 0;
    for (DIM i = 0; i < D; i++) {
      offset += idx[i] * curr_stride;
#ifdef MGARDm_COMPILE_EXEC
      curr_stride *= ldvs_d[i];
#else
      curr_stride *= ldvs_h[i];
#endif
    }
    return dv + offset;
  }

  MGARDm_CONT_EXEC
  T* operator()(IDX z, IDX y, IDX x) {
    return dv + lddv2 * lddv1 * z + lddv1 * y + x;
  }
  MGARDm_CONT_EXEC
  T* operator()(IDX y, IDX x) {
    return dv + lddv1 * y + x;
  }
  MGARDm_CONT_EXEC
  T* operator()(IDX x) {
    return dv + x;
  }

  MGARDm_CONT_EXEC
  T* offset(IDX z, IDX y, IDX x) {
    dv += lddv2 * lddv1 * z + lddv1 * y + x;
  }
  MGARDm_CONT_EXEC
  T* offset(IDX y, IDX x) {
    dv += lddv1 * y + x;
  }
  MGARDm_CONT_EXEC
  T* offset(IDX x) {
    dv += x;
  }

  MGARDm_CONT_EXEC
  bool isNull() {
    return dv == NULL;
  }
  MGARDm_CONT_EXEC
  T * data() {
    return dv;
  }

  MGARDm_CONT
  T * dataHost() {
    if (!has_host_pointer) {
      std::cerr << log::log_err
                << "Host pointer not initialized!\n";
      exit(-1);
    }
    return v;
  }

  ~SubArray();

  T *dv; // device pointer
  T *v; // host pointer
  bool has_host_pointer;
  std::vector<SIZE> ldvs_h;
  SIZE *ldvs_d;
  std::vector<SIZE> shape;
  DIM projected_dim0;
  DIM projected_dim1;
  DIM projected_dim2;
  SIZE lddv1;
  SIZE lddv2;
  using DataType = T;
  static const DIM NumDims = D;
};


template <DIM D, typename T, typename DeviceType>
SubArray<D, T, DeviceType>::SubArray() {
  lddv1 = 1;
  lddv2 = 1;
}

template <DIM D, typename T, typename DeviceType>
SubArray<D, T, DeviceType>::SubArray(Array<D, T, DeviceType> &array, bool get_host_pointer) {
  this->shape  = array.getShape();
  this->dv     = array.get_dv();
  this->ldvs_h = array.get_ldvs_h();
  this->ldvs_d = array.get_ldvs_d();
  lddv1 = ldvs_h[0];
  lddv2 = ldvs_h[1];
  if (get_host_pointer) {
    this->v = array.getDataHost();
    has_host_pointer = true;
  }
}

template <DIM D, typename T, typename DeviceType>
SubArray<D, T, DeviceType>::SubArray(std::vector<SIZE> shape, T * dv, std::vector<SIZE> ldvs_h, SIZE * ldvs_d) {
  this->shape  = shape;
  this->dv     = dv;
  this->ldvs_h = ldvs_h;
  this->ldvs_d = ldvs_d;
  lddv1 = ldvs_h[0];
  lddv2 = ldvs_h[1];
}

template <DIM D, typename T, typename DeviceType>
SubArray<D, T, DeviceType>::SubArray(std::vector<SIZE> shape, T * dv) {
  this->shape  = shape;
  this->dv     = dv;
}


template <DIM D, typename T, typename DeviceType> 
SubArray<D, T, DeviceType>::SubArray(SubArray<D, T, DeviceType> &subArray) {
  this->shape  = subArray.shape;
  this->dv     = subArray.dv;
  this->ldvs_h = subArray.ldvs_h;
  this->ldvs_d = subArray.ldvs_d;

  this->lddv1 = subArray.lddv1;
  this->lddv2 = subArray.lddv2;

  this->projected_dim0 = subArray.projected_dim0;
  this->projected_dim1 = subArray.projected_dim1;
  this->projected_dim2 = subArray.projected_dim2;

  if (subArray.has_host_pointer) {
    this->has_host_pointer = true;
    this->v = subArray.v;
  }


}

template <DIM D, typename T, typename DeviceType> 
SubArray<D, T, DeviceType>::SubArray(const SubArray<D, T, DeviceType> &subArray) {
  this->shape  = subArray.shape;
  this->dv     = subArray.dv;
  this->ldvs_h = subArray.ldvs_h;
  this->ldvs_d = subArray.ldvs_d;

  this->lddv1 = subArray.lddv1;
  this->lddv2 = subArray.lddv2;

  this->projected_dim0 = subArray.projected_dim0;
  this->projected_dim1 = subArray.projected_dim1;
  this->projected_dim2 = subArray.projected_dim2;

  if (subArray.has_host_pointer) {
    this->has_host_pointer = true;
    this->v = subArray.v;
  }

}

template <DIM D, typename T, typename DeviceType> 
SubArray<D, T, DeviceType>& SubArray<D, T, DeviceType>::operator = (const SubArray<D, T, DeviceType> &subArray) {
  this->shape  = subArray.shape;
  this->dv     = subArray.dv;
  this->ldvs_h = subArray.ldvs_h;
  this->ldvs_d = subArray.ldvs_d;

  this->lddv1 = subArray.lddv1;
  this->lddv2 = subArray.lddv2;

  this->projected_dim0 = subArray.projected_dim0;
  this->projected_dim1 = subArray.projected_dim1;
  this->projected_dim2 = subArray.projected_dim2;

  if (subArray.has_host_pointer) {
    this->has_host_pointer = true;
    this->v = subArray.v;
  }
  
  return *this;
}

template <DIM D, typename T, typename DeviceType> 
void SubArray<D, T, DeviceType>::offset(std::vector<SIZE> idx) {
  dv += get_idx(ldvs_h, idx);
}

template <DIM D, typename T, typename DeviceType> 
void SubArray<D, T, DeviceType>::resize(std::vector<SIZE> shape) {
  this->shape = shape;
}

template <DIM D, typename T, typename DeviceType> 
void SubArray<D, T, DeviceType>::offset(DIM dim, SIZE offset_value) {
  std::vector<SIZE> idx(D, 0);
  idx[dim] = offset_value;
  dv += get_idx(ldvs_h, idx);
}

template <DIM D, typename T, typename DeviceType> 
void SubArray<D, T, DeviceType>::resize(DIM dim, SIZE new_size) {
  shape[dim] = new_size;
}

template <DIM D, typename T, typename DeviceType> 
void SubArray<D, T, DeviceType>::project(DIM dim0, DIM dim1, DIM dim2) {
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


// template <DIM D, typename T, typename DeviceType> 
// MGARDm_EXEC 
// T* SubArray<D, T, DeviceType>::operator()(IDX z, IDX y, IDX x) {
//   return dv + lddv2 * lddv1 * z + lddv1 * y + x;
// }

// template <DIM D> __forceinline__ __device__ LENGTH get_idx(SIZE *lds, SIZE *idx) {
//   LENGTH curr_stride = 1;
//   LENGTH ret_idx = 0;
//   for (DIM i = 0; i < D; i++) {
//     ret_idx += idx[i] * curr_stride;
//     curr_stride *= lds[i];
//   }
//   return ret_idx;
// }



template <DIM D, typename T, typename DeviceType> 
SubArray<D, T, DeviceType>::~SubArray() {
  // nothing needs to be released
}

} // namespace mgard_cuda
#endif