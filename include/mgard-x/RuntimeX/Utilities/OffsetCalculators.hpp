/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_OFFSET_CALCULATORS_H
#define MGARD_X_OFFSET_CALCULATORS_H

namespace mgard_x {

MGARDX_CONT_EXEC LENGTH get_idx(const SIZE ld1, const SIZE ld2, const SIZE z,
                                const SIZE y, const SIZE x) {
  return ld2 * ld1 * z + ld1 * y + x;
}

// for 3D+
MGARDX_CONT_EXEC LENGTH get_idx(const LENGTH ld1, const LENGTH ld2,
                                const SIZE z, const SIZE y, const SIZE x) {
  return ld2 * ld1 * z + ld1 * y + x;
}

// leading dimension first
MGARDX_CONT LENGTH get_idx(std::vector<SIZE> lds, std::vector<SIZE> idx) {
  LENGTH curr_stride = 1;
  LENGTH ret_idx = 0;
  for (DIM i = 0; i < idx.size(); i++) {
    ret_idx += idx[i] * curr_stride;
    curr_stride *= lds[i];
  }
  return ret_idx;
}

template <DIM D> MGARDX_CONT_EXEC LENGTH get_idx(SIZE *lds, SIZE *idx) {
  LENGTH curr_stride = 1;
  LENGTH ret_idx = 0;
  for (DIM i = 0; i < D; i++) {
    ret_idx += idx[i] * curr_stride;
    curr_stride *= lds[i];
  }
  return ret_idx;
}

MGARDX_CONT std::vector<SIZE> gen_idx(DIM D, DIM curr_dim_r, DIM curr_dim_c,
                                      DIM curr_dim_f, SIZE idx_r, SIZE idx_c,
                                      SIZE idx_f) {
  std::vector<SIZE> idx(D, 0);
  idx[curr_dim_r] = idx_r;
  idx[curr_dim_c] = idx_c;
  idx[curr_dim_f] = idx_f;
  return idx;
}

MGARDX_CONT_EXEC int div_roundup(SIZE a, SIZE b) { return (a - 1) / b + 1; }

template <typename T1, typename T2> MGARDX_CONT_EXEC SIZE roundup(T2 a) {
  return ((a - 1) / sizeof(T1) + 1) * sizeof(T1);
}

} // namespace mgard_x

#endif