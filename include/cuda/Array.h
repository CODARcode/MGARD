/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGARD_CUDA_ARRAY
#define MGARD_CUDA_ARRAY
#include "Common.h"
#include <vector>

namespace mgard_cuda {

template <DIM D, typename T> class Array {
public:
  Array();
  Array(std::vector<SIZE> shape);
  Array(const Array &array);
  Array(Array &array);
  Array &operator=(const Array &array);
  Array(Array &&array);
  ~Array();
  void loadData(const T *data, SIZE ld = 0);
  T *getDataHost();
  T *getDataDevice(SIZE &ld);
  std::vector<SIZE> getShape();
  T *get_dv();
  std::vector<SIZE> get_ldvs_h();
  SIZE *get_ldvs_d();

private:
  DIM D_padded;
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