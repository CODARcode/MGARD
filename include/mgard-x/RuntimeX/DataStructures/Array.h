/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_ARRAY
#define MGARD_X_ARRAY

#include <vector>

namespace mgard_x {

template <DIM D, typename T, typename DeviceType> class Array {
public:
  Array();
  Array(std::vector<SIZE> shape, bool pitched = true, bool managed = false,
        int queue_idx = MGARDX_SYNCHRONIZED_QUEUE);
  Array(std::vector<SIZE> shape, T *dv);
  void initialize(std::vector<SIZE> shape);
  void allocate(bool pitched, bool managed,
                int queue_idx = MGARDX_SYNCHRONIZED_QUEUE);
  void copy(const Array &array, int queue_idx = MGARDX_SYNCHRONIZED_QUEUE);
  void move(Array &&array);
  void memset(int value, int queue_idx = MGARDX_SYNCHRONIZED_QUEUE);
  void free(int queue_idx = MGARDX_SYNCHRONIZED_QUEUE);
  Array(const Array &array);
  Array &operator=(const Array &array);
  Array &operator=(Array &&array);
  Array(Array &&array);
  ~Array();
  void load(const T *data, SIZE ld = 0,
            int queue_idx = MGARDX_SYNCHRONIZED_QUEUE);
  T *hostCopy(bool keep = false, int queue_idx = MGARDX_SYNCHRONIZED_QUEUE);
  T *data(SIZE &ld);
  SIZE &shape(DIM d);
  std::vector<SIZE> &shape();
  SIZE totalNumElems();
  T *data();
  T *dataHost();
  SIZE ld(DIM d);
  bool isPitched();
  bool isManaged();
  bool hasDeviceAllocation();
  bool hasHostAllocation();
  int resideDevice();
  void resize(std::vector<SIZE> shape,
              int queue_idx = MGARDX_SYNCHRONIZED_QUEUE);

private:
  int dev_id;
  bool pitched;
  bool managed;
  bool keepHostCopy = false;
  T *dv = nullptr;
  T *hv = nullptr;
  bool device_allocated = false;
  bool host_allocated = false;
  bool external_allocation = false;
  std::vector<SIZE> __ldvs;
  std::vector<SIZE> __shape;
  std::vector<SIZE> __ldvs_allocation;
  std::vector<SIZE> __shape_allocation;
  SIZE linearized_width;
};

} // namespace mgard_x
#endif