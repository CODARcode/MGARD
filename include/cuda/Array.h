/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGARD_CUDA_ARRAY
#define MGARD_CUDA_ARRAY
#include <stdint.h>
#include <vector>

namespace mgard_cuda {

template <typename T, uint32_t D> class Array {
public:
  Array(std::vector<size_t> shape);
  Array(Array &array);
  ~Array();
  void loadData(T *data, size_t ld = 0);
  T *getDataHost();
  T *getDataDevice(size_t &ld);
  std::vector<size_t> getShape();
  T *get_dv();
  std::vector<int> get_ldvs_h();
  int *get_ldvs_d();

private:
  int D_padded;
  T *dv;
  T *hv;
  bool device_allocated;
  bool host_allocated;
  std::vector<int> ldvs_h;
  int *ldvs_d;
  std::vector<size_t> shape;
  size_t linearized_depth;
};

} // namespace mgard_cuda
#endif