/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_CELL
#define MGARD_X_CELL

namespace mgard_x {


template <typename T, typename DeviceType>
class Cell {
public:
SIZE index[3];
SIZE size[3];
T value[8];

MGARDX_CONT void
Print() const {
  std::cout << "(";
  for (DIM d = 0; d < 3; d++) {
    std::cout << index[d];
    if (d < 3 - 1) std::cout << ", ";
  }
  std::cout << ") -> (";
  for (DIM d = 0; d < 3; d++) {
    std::cout << size[d];
    if (d < 3 - 1) std::cout << ", ";
  }
  std::cout << ")\n";
}

};

}

#endif