/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_EDGE
#define MGARD_X_EDGE

namespace mgard_x {

#define NONE 0
#define LEAD 1
#define REGULAR 2


template <DIM D, typename T, typename DeviceType>
class Edge {
public:
SIZE start_index[D];
SIZE end_index[D];
T start_value;
T end_value; 

// role of this edge in its cell
int role;

MGARDX_CONT void
Print() const {
  std::cout << "(";
  for (DIM d = 0; d < D; d++) {
    std::cout << start_index[d];
    if (d < D - 1) std::cout << ", ";
  }
  std::cout << ") -> (";
  for (DIM d = 0; d < D; d++) {
    std::cout << end_index[d];
    if (d < D - 1) std::cout << ", ";
  }
  std::cout << ")\n";
}

MGARDX_CONT
bool operator ==(const Edge<D, T, DeviceType> &edge) const {
  for (DIM d = 0; d < D; d++) {
    if (start_index[d] != edge.start_index[d] ||
        end_index[d] != edge.end_index[d]) {
      return false;
    }
  }
  return true;
}

};

}

#endif