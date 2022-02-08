/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_ARRAY
#define MGARD_X_ARRAY
// #include "Common.h"
#include <vector>

namespace mgard_x {

template <DIM D, typename T, typename DeviceType> class Array {
public:
  Array();
  Array(std::vector<SIZE> shape, bool pitched = true, bool managed = false);
  Array(const Array &array);
  Array(Array &array);
  Array &operator=(const Array &array);
  Array(Array &&array);
  ~Array();
  void memset(int value);
  void loadData(const T *data, SIZE ld = 0);
  T *getDataHost();
  T *getDataDevice(SIZE &ld);
  std::vector<SIZE> &getShape();
  T *get_dv();
  std::vector<SIZE> get_ldvs_h();
  SIZE *get_ldvs_d();
  bool is_pitched();

private:
  DIM D_padded;
  bool pitched;
  bool managed;
  T *dv = NULL;
  T *hv = NULL;
  bool device_allocated;
  bool host_allocated;
  std::vector<SIZE> ldvs_h;
  SIZE *ldvs_d;
  std::vector<SIZE> shape;
  SIZE linearized_depth;
};

} // namespace mgard_x
#endif