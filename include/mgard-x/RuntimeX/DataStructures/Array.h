/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
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
  void copy(const Array &array);
  void move(Array &&array);
  void memset(int value);
  void free();
  Array(const Array &array);
  Array &operator=(const Array &array);
  Array &operator=(Array &&array);
  Array(Array &&array);
  ~Array();
  void load(const T *data, SIZE ld = 0);
  T *hostCopy(bool keep = false);
  T *data(SIZE &ld);
  std::vector<SIZE> &shape();
  T *data();
  std::vector<SIZE> ld();
  bool is_pitched();

private:
  DIM D_padded;
  bool pitched;
  bool managed;
  bool keepHostCopy = false;
  T *dv = nullptr;
  T *hv = nullptr;
  bool device_allocated = false;
  bool host_allocated = false;
  std::vector<SIZE> _ldvs;
  std::vector<SIZE> _shape;
  SIZE linearized_depth;
};

} // namespace mgard_x
#endif