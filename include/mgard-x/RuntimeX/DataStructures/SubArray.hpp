/*
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: Jul 20, 2021
 */

#ifndef MGARD_X_SUBARRAY_HPP
#define MGARD_X_SUBARRAY_HPP

#include <assert.h>
#include <iostream>
#include <vector>

namespace mgard_x {

template <DIM D, typename T, typename DeviceType> class SubArray {
public:
  MGARDX_CONT_EXEC
  SubArray();

  MGARDX_CONT
  SubArray(Array<D, T, DeviceType> &array);

  MGARDX_CONT
  SubArray(std::vector<SIZE> shape, T *dv);

  MGARDX_CONT_EXEC
  void initialize();

  MGARDX_CONT_EXEC
  T *data() { return this->dv; }

  MGARDX_CONT_EXEC
  void setData(T *dv) { this->dv = dv; }

  MGARDX_CONT_EXEC
  bool hasHostData() { return has_host_pointer; };

  MGARDX_CONT
  T *dataHost() {
    if (!has_host_pointer) {
      std::cerr << log::log_err << "Host pointer not initialized!\n";
      exit(-1);
    }
    return v;
  }

  MGARDX_CONT
  void setDataHost(T *v) {
    this->has_host_pointer = true;
    this->v = v;
  }

  MGARDX_CONT_EXEC
  void setShape(DIM d, SIZE n) {
    if (d >= D) {
      return;
    }
    __shape[d] = n;
  }

  MGARDX_CONT_EXEC
  SIZE shape(DIM d) const {
    if (d >= D)
      return 1;
    return __shape[d];
  }

  MGARDX_CONT_EXEC
  SIZE ld(DIM d) const {
    if (d >= D)
      return 1;
    return __ldvs[d];
  }

  MGARDX_CONT_EXEC
  void setLd(DIM d, SIZE ld) { __ldvs[d] = ld; }

  MGARDX_CONT_EXEC
  bool isPitched() { return this->pitched; }

  MGARDX_CONT_EXEC
  void setPitched(bool pitched) { this->pitched = pitched; }

  MGARDX_CONT_EXEC
  SIZE lddv1() const { return __lddv1; }

  MGARDX_CONT_EXEC
  SIZE lddv2() const { return __lddv2; }

  void offset(std::vector<SIZE> idx);

  MGARDX_CONT
  void resize(std::vector<SIZE> shape);

  MGARDX_CONT
  void offset_dim(DIM dim, SIZE offset_value);

  MGARDX_CONT
  void resize(DIM dim, SIZE new_size);

  MGARDX_CONT
  void project(DIM dim_slowest, DIM dim_medium, DIM dim_fastest);

  MGARDX_CONT
  SubArray<1, T, DeviceType> Linearize();

  MGARDX_CONT
  SubArray<3, T, DeviceType> Slice3D(DIM d1, DIM d2, DIM d3);

  MGARDX_CONT_EXEC
  T &operator[](SIZE idx[D]) {
    SIZE curr_stride = 1;
    SIZE offset = 0;
    for (int d = D - 1; d >= 0; d--) {
      offset += idx[d] * curr_stride;
      curr_stride *= __ldvs[d];
    }
    return dv[offset];
  }

  MGARDX_CONT_EXEC
  T *operator()(IDX l, IDX z, IDX y, IDX x) {
    return dv + __ldvs[1] * __ldvs[2] * __ldvs[3] * l +
           __ldvs[2] * __ldvs[3] * z + __ldvs[3] * y + x;
  }
  MGARDX_CONT_EXEC
  T *operator()(IDX z, IDX y, IDX x) {
    return dv + __lddv2 * __lddv1 * z + __lddv1 * y + x;
  }
  MGARDX_CONT_EXEC
  T *operator()(IDX y, IDX x) { return dv + __lddv1 * y + x; }
  MGARDX_CONT_EXEC
  T *operator()(IDX x) { return dv + x; }

  MGARDX_EXEC void offset(SIZE idx[D]) {
    ptr_offset += calc_offset(idx);
    dv += calc_offset(idx);
  }

  MGARDX_EXEC void offset_3d(IDX z, IDX y, IDX x) {
    ptr_offset += __lddv2 * __lddv1 * z + __lddv1 * y + x;
    dv += __lddv2 * __lddv1 * z + __lddv1 * y + x;
  }
  MGARDX_EXEC void offset_2d(IDX y, IDX x) {
    ptr_offset += __lddv1 * y + x;
    dv += __lddv1 * y + x;
  }
  MGARDX_EXEC void offset_1d(IDX x) {
    ptr_offset += x;
    dv += x;
  }

  MGARDX_EXEC void reset_offset() {
    dv -= ptr_offset;
    ptr_offset = 0;
  }

  MGARDX_CONT_EXEC
  bool isNull() { return dv == nullptr; }

  using DataType = T;
  using DevType = DeviceType;
  static const DIM NumDims = D;

private:
  T *dv; // device pointer
  T *v;  // host pointer
  bool has_host_pointer;

  SIZE __ldvs[D];
  SIZE __shape[D];

  DIM projected_dim_fastest;
  DIM projected_dim_medium;
  DIM projected_dim_slowest;

  SIZE __lddv1;
  SIZE __lddv2;

  bool pitched;
  bool managed;

  SIZE ptr_offset;

  MGARDX_CONT_EXEC
  SIZE calc_offset(SIZE idx[D]) {
    SIZE curr_stride = 1;
    SIZE offset = 0;
    for (int d = D - 1; d >= 0; d--) {
      offset += idx[d] * curr_stride;
      curr_stride *= __ldvs[d];
    }
    return offset;
  }
};

template <DIM D, typename T, typename DeviceType>
MGARDX_CONT_EXEC void SubArray<D, T, DeviceType>::initialize() {
  dv = nullptr;
  v = nullptr;
  has_host_pointer = false;
  for (DIM d = 0; d < D; d++) {
    __ldvs[d] = 1;
    __shape[d] = 1;
  }

  projected_dim_fastest = D - 1;
  projected_dim_medium = D - 2;
  projected_dim_slowest = D - 3;
  __lddv1 = 1;
  __lddv2 = 1;

  pitched = false;
  managed = false;

  ptr_offset = 0;
}

template <DIM D, typename T, typename DeviceType>
MGARDX_CONT_EXEC SubArray<D, T, DeviceType>::SubArray() {
  initialize();
}

template <DIM D, typename T, typename DeviceType>
MGARDX_CONT
SubArray<D, T, DeviceType>::SubArray(Array<D, T, DeviceType> &array) {
  initialize();
  dv = array.data();
  for (DIM d = 0; d < D; d++) {
    __shape[d] = array.shape(d);
    __ldvs[d] = array.ld(d);
  }
  __lddv1 = __ldvs[D - 1];
  if (D > 1)
    __lddv2 = __ldvs[D - 2];
  if (array.hasHostAllocation()) {
    v = array.dataHost();
    has_host_pointer = true;
  }
  pitched = array.isPitched();
  managed = array.isManaged();
}

// TODO: update shape
template <DIM D, typename T, typename DeviceType>
MGARDX_CONT SubArray<D, T, DeviceType>::SubArray(std::vector<SIZE> shape,
                                                 T *dv) {
  initialize();
  this->dv = dv;
  for (DIM d = 0; d < D; d++) {
    this->__shape[d] = shape[d];
    this->__ldvs[d] = shape[d];
  }
  __lddv1 = __ldvs[D - 1];
  if (D > 1)
    __lddv2 = __ldvs[D - 2];
}

template <DIM D, typename T, typename DeviceType>
MGARDX_CONT SubArray<1, T, DeviceType> SubArray<D, T, DeviceType>::Linearize() {
  SubArray<1, T, DeviceType> subArray;
  if (!pitched) {
    SIZE linearized_shape = 1;
    for (DIM d = 0; d < D; d++)
      linearized_shape *= this->__shape[d];
    subArray.setData(data());
    subArray.setShape(0, linearized_shape);
    subArray.setLd(0, linearized_shape);
    subArray.project(0, 1, 2);

    if (has_host_pointer) {
      subArray.setDataHost(dataHost());
    }
    subArray.setPitched(isPitched());
  } else {
    std::cout << log::log_err
              << "Linearized pitched SubArray not implemented!\n";
    exit(-1);
  }
  return subArray;
}

template <DIM D, typename T, typename DeviceType>
MGARDX_CONT SubArray<3, T, DeviceType>
SubArray<D, T, DeviceType>::Slice3D(DIM d2, DIM d1, DIM d0) {
  // d2 is slowest dim.
  // d0 is fastest dim.
  if (D < 3) {
    std::cout << log::log_err << "calling Slice3D on SubArray with " << D
              << "D data.\n";
    exit(-1);
  }
  SubArray<3, T, DeviceType> subArray;
  subArray.setShape(2, __shape[d0]);
  subArray.setShape(1, __shape[d1]);
  subArray.setShape(0, __shape[d2]);
  subArray.setData(dv);
  subArray.setLd(2, __ldvs[d0]);
  subArray.setLd(1, __ldvs[d1]);
  subArray.setLd(0, __ldvs[d2]);
  subArray.project(d2, d1, d0);

  if (has_host_pointer) {
    subArray.setDataHost(v);
  }
  subArray.setPitched(pitched);
  return subArray;
}

template <DIM D, typename T, typename DeviceType>
MGARDX_CONT void SubArray<D, T, DeviceType>::offset(std::vector<SIZE> idx) {
  if (idx.size() < D) {
    std::cerr << log::log_err << "SubArray::resize insufficient idx length.\n";
  }
  // In case shape.size > D;
  DIM skip_dim = idx.size() - D;
  SIZE _idx[D];
  for (DIM d = 0; d < D; d++)
    _idx[d] = idx[skip_dim + d];
  dv += calc_offset(_idx);
}

template <DIM D, typename T, typename DeviceType>
MGARDX_CONT void SubArray<D, T, DeviceType>::resize(std::vector<SIZE> shape) {
  if (shape.size() < D) {
    std::cerr << log::log_err
              << "SubArray::resize insufficient shape length.\n";
  }
  // In case shape.size > D;
  DIM skip_dim = shape.size() - D;
  for (DIM d = 0; d < D; d++) {
    __shape[d] = shape[skip_dim + d];
  }
}

template <DIM D, typename T, typename DeviceType>
MGARDX_CONT void SubArray<D, T, DeviceType>::offset_dim(DIM dim,
                                                        SIZE offset_value) {
  if (dim >= D)
    return;
  SIZE idx[D];
  for (DIM d = 0; d < D; d++) {
    idx[d] = 0;
  }
  idx[dim] = offset_value;
  dv += calc_offset(idx);
}

template <DIM D, typename T, typename DeviceType>
MGARDX_CONT void SubArray<D, T, DeviceType>::resize(DIM dim, SIZE new_size) {
  if (dim >= D)
    return;
  __shape[dim] = new_size;
}

template <DIM D, typename T, typename DeviceType>
MGARDX_CONT void SubArray<D, T, DeviceType>::project(DIM dim_slowest,
                                                     DIM dim_medium,
                                                     DIM dim_fastest) {
  projected_dim_slowest = dim_slowest;
  projected_dim_medium = dim_medium;
  projected_dim_fastest = dim_fastest;
  if (projected_dim_slowest >= D)
    projected_dim_slowest = 0;
  if (projected_dim_medium >= D)
    projected_dim_medium = 0;
  __lddv1 = 1, __lddv2 = 1;
  for (int d = projected_dim_fastest; d > projected_dim_medium; d--) {
    __lddv1 *= __ldvs[d];
  }
  for (int d = projected_dim_medium; d > projected_dim_slowest; d--) {
    __lddv2 *= __ldvs[d];
  }
}

} // namespace mgard_x
#endif
