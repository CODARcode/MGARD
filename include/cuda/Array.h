/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#ifndef MGARD_CUDA_ARRAY
#define MGARD_CUDA_ARRAY
#include "Common.h"
#include <vector>

namespace mgard_cuda {

template <DIM D, typename T, typename DeviceType> class Array {
public:
  Array();
  Array(std::vector<SIZE> shape, bool pitched = true);
  Array(const Array &array);
  Array(Array &array);
  Array& operator = (const Array &array);
  Array(Array && array);
  ~Array();
  void loadData(const T *data, SIZE ld = 0);
  T *getDataHost();
  T *getDataDevice(SIZE &ld);
  std::vector<SIZE> getShape();
  T *get_dv();
  std::vector<SIZE> get_ldvs_h();
  SIZE *get_ldvs_d();
  bool is_pitched();

private:
  DIM D_padded;
  bool pitched;
  T *dv;
  T *hv;
  bool device_allocated;
  bool host_allocated;
  std::vector<SIZE> ldvs_h;
  SIZE *ldvs_d;
  std::vector<SIZE> shape;
  SIZE linearized_depth;
};



} // namespace mgard_cuda
#endif