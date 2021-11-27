/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_CHECK_ENDIANESS_HPP
#define MGARD_X_CHECK_ENDIANESS_HPP

namespace mgard_x {
enum endiness_type CheckEndianess() {
  int i = 1;
  char *p = (char *)&i;
  if (p[0] == 1) {
    return endiness_type::Little_Endian;
  } else {
    return endiness_type::Big_Endian;
  }
}

}
#endif