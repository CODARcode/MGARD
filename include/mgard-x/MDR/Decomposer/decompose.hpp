#ifndef _MGARD_DECOMPOSE_HPP
#define _MGARD_DECOMPOSE_HPP

#include "utils.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace MGARD {

using namespace std;

template <class T> class Decomposer {
public:
  Decomposer(bool use_sz_ = true) { use_sz = use_sz_; };
  ~Decomposer() {
    if (data_buffer)
      free(data_buffer);
    if (correction_buffer)
      free(correction_buffer);
    if (load_v_buffer)
      free(load_v_buffer);
  };
  // return levels
  int decompose(T *data_, const vector<size_t> &dims, size_t target_level,
                bool hierarchical = false) {
    data = data_;
    size_t num_elements = 1;
    for (const auto &d : dims) {
      num_elements *= d;
    }
    data_buffer_size = num_elements * sizeof(T);
    int max_level = log2(*min_element(dims.begin(), dims.end()));
    if (target_level > max_level)
      target_level = max_level;
    init(dims);
    current_dims.resize(dims.size());
    if (dims.size() == 1) {
      size_t h = 1;
      size_t n = dims[0];
      for (int i = 0; i < target_level; i++) {
        hierarchical ? decompose_level_1D_with_hierarchical_basis(data, n, h)
                     : decompose_level_1D(data, n, h);
        n = (n >> 1) + 1;
        h <<= 1;
      }
    } else if (dims.size() == 2) {
      size_t h = 1;
      size_t n1 = dims[0];
      size_t n2 = dims[1];
      for (int i = 0; i < target_level; i++) {
        hierarchical ? decompose_level_2D_with_hierarchical_basis(data, n1, n2,
                                                                  (T)h, dims[1])
                     : decompose_level_2D(data, n1, n2, (T)h, dims[1]);
        n1 = (n1 >> 1) + 1;
        n2 = (n2 >> 1) + 1;
        h <<= 1;
      }
    } else if (dims.size() == 3) {
      size_t h = 1;
      size_t n1 = dims[0];
      size_t n2 = dims[1];
      size_t n3 = dims[2];
      for (int i = 0; i < target_level; i++) {
        current_dims[0] = (n1 >> 1) + 1;
        current_dims[1] = (n2 >> 1) + 1;
        current_dims[2] = (n3 >> 1) + 1;
        hierarchical ? decompose_level_3D_with_hierarchical_basis(
                           data, n1, n2, n3, (T)h, dims[1] * dims[2], dims[2])
                     : decompose_level_3D(data, n1, n2, n3, (T)h,
                                          dims[1] * dims[2], dims[2]);
        n1 = (n1 >> 1) + 1;
        n2 = (n2 >> 1) + 1;
        n3 = (n3 >> 1) + 1;
        h <<= 1;
      }
    }
    return target_level;
  }

private:
  unsigned int default_batch_size = 32;
  size_t data_buffer_size = 0;
  bool use_sz = true;
  T *data = NULL;        // pointer to the original data
  T *data_buffer = NULL; // buffer for reordered data
  T *load_v_buffer = NULL;
  T *correction_buffer = NULL;
  vector<size_t> current_dims;

  void init(const vector<size_t> &dims) {
    size_t buffer_size = default_batch_size *
                         (*max_element(dims.begin(), dims.end())) * sizeof(T);
    // cerr << "buffer_size = " << buffer_size << endl;
    if (data_buffer)
      free(data_buffer);
    if (correction_buffer)
      free(correction_buffer);
    if (load_v_buffer)
      free(load_v_buffer);
    data_buffer = (T *)malloc(data_buffer_size);
    correction_buffer = (T *)malloc(buffer_size);
    load_v_buffer = (T *)malloc(buffer_size);
  }
  // reorder the data to put all the coefficient to the back
  void data_reorder_1D(const T *data_pos, size_t n_nodal, size_t n_coeff,
                       T *nodal_buffer, T *coeff_buffer) {
    T *nodal_pos = nodal_buffer;
    T *coeff_pos = coeff_buffer;
    T const *cur_data_pos = data_pos;
    for (int i = 0; i < n_coeff; i++) {
      *(nodal_pos++) = *(cur_data_pos++);
      *(coeff_pos++) = *(cur_data_pos++);
    }
    *(nodal_pos++) = *(cur_data_pos++);
    if (n_nodal == n_coeff + 2) {
      // if even, add a nodal value such that the interpolant
      // of the last two nodal values equal to the last coefficient
      *nodal_pos = 2 * cur_data_pos[0] - nodal_pos[-1];
    }
  }
  // compute the difference between original value
  // and interpolant (I - PI_l)Q_l
  // overwrite the data in N_l \ N_(l-1) in place
  void compute_interpolant_difference_1D(size_t n_coeff, const T *nodal_buffer,
                                         T *coeff_buffer) {
    for (int i = 0; i < n_coeff; i++) {
      coeff_buffer[i] -= (nodal_buffer[i] + nodal_buffer[i + 1]) / 2;
    }
  }
  void add_correction(size_t n_nodal, T *nodal_buffer) {
    for (int i = 0; i < n_nodal; i++) {
      nodal_buffer[i] += correction_buffer[i];
    }
  }
  // decompose a level with n element and the given stride
  // to a level with n/2 element
  void decompose_level_1D(T *data_pos, size_t n, T h, bool nodal_row = true) {
    size_t n_nodal = (n >> 1) + 1;
    size_t n_coeff = n - n_nodal;
    T *nodal_buffer = data_buffer;
    T *coeff_buffer = data_buffer + n_nodal;
    data_reorder_1D(data_pos, n_nodal, n_coeff, nodal_buffer, coeff_buffer);
    compute_interpolant_difference_1D(n_coeff, nodal_buffer, coeff_buffer);
    if (nodal_row)
      compute_load_vector_nodal_row(load_v_buffer, n_nodal, n_coeff, h,
                                    coeff_buffer);
    else
      compute_load_vector_coeff_row(load_v_buffer, n_nodal, n_coeff, h,
                                    nodal_buffer, coeff_buffer);
    compute_correction(correction_buffer, n_nodal, h, load_v_buffer);
    add_correction(n_nodal, nodal_buffer);
    memcpy(data_pos, data_buffer, n * sizeof(T));
  }
  void decompose_level_1D_with_hierarchical_basis(T *data_pos, size_t n, T h,
                                                  bool nodal_row = true) {
    size_t n_nodal = (n >> 1) + 1;
    size_t n_coeff = n - n_nodal;
    T *nodal_buffer = data_buffer;
    T *coeff_buffer = data_buffer + n_nodal;
    data_reorder_1D(data_pos, n_nodal, n_coeff, nodal_buffer, coeff_buffer);
    compute_interpolant_difference_1D(n_coeff, nodal_buffer, coeff_buffer);
  }
  /*
          2D decomposition
  */
  // reorder the data to put all the coefficient to the back
  /*
          oxoxo		oooxx		oooxx
          xxxxx	(1)	xxxxx	(2)	oooxx
          oxoxo	=>	oooxx	=>	oooxx
          xxxxx		xxxxx		xxxxx
          oxoxo		oooxx		xxxxx
  */
  void data_reorder_2D(T *data_pos, size_t n1, size_t n2, size_t stride) {
    size_t n1_nodal = (n1 >> 1) + 1;
    size_t n1_coeff = n1 - n1_nodal;
    size_t n2_nodal = (n2 >> 1) + 1;
    size_t n2_coeff = n2 - n2_nodal;
    T *cur_data_pos = data_pos;
    T *nodal_pos = data_buffer;
    T *coeff_pos = data_buffer + n2_nodal;
    // do reorder (1)
    for (int i = 0; i < n1; i++) {
      data_reorder_1D(cur_data_pos, n2_nodal, n2_coeff, nodal_pos, coeff_pos);
      memcpy(cur_data_pos, data_buffer, n2 * sizeof(T));
      cur_data_pos += stride;
    }
    if (!(n1 & 1)) {
      // n1 is even, change the last coeff row into nodal row
      cur_data_pos -= stride;
      for (int j = 0; j < n2; j++) {
        cur_data_pos[j] = 2 * cur_data_pos[j] - cur_data_pos[-stride + j];
      }
    }
    // do reorder (2)
    // TODO: change to online processing for memory saving
    switch_rows_2D_by_buffer(data_pos, data_buffer, n1, n2, stride);
  }
  // compute the difference between original value
  // and interpolant (I - PI_l)Q_l for the coefficient rows in 2D
  // overwrite the data in N_l \ N_(l-1) in place
  // Note: interpolant difference in the nodal rows have already been computed
  void compute_interpolant_difference_2D_vertical(T *data_pos, size_t n1,
                                                  size_t n2, size_t stride) {
    size_t n1_nodal = (n1 >> 1) + 1;
    size_t n1_coeff = n1 - n1_nodal;
    size_t n2_nodal = (n2 >> 1) + 1;
    size_t n2_coeff = n2 - n2_nodal;
    bool even_n2 = !(n2 & 1);
    T *n1_nodal_data = data_pos;
    T *n1_coeff_data = data_pos + n1_nodal * stride;
    for (int i = 0; i < n1_coeff; i++) {
      const T *nodal_pos = n1_nodal_data + i * stride;
      T *coeff_pos = n1_coeff_data + i * stride;
      // TODO: optimize average computation
      T *nodal_coeff_pos = coeff_pos; // coeffcients in nodal rows
      T *coeff_coeff_pos =
          coeff_pos + n2_nodal; // coefficients in coeffcients rows
      for (int j = 0; j < n2_coeff; j++) {
        // coefficients in nodal columns
        *(nodal_coeff_pos++) -= (nodal_pos[j] + nodal_pos[stride + j]) / 2;
        // coefficients in centers
        *(coeff_coeff_pos++) -=
            (nodal_pos[j] + nodal_pos[j + 1] + nodal_pos[stride + j] +
             nodal_pos[stride + j + 1]) /
            4;
      }
      // compute the last (or second last if n2 is even) nodal column
      *(nodal_coeff_pos++) -=
          (nodal_pos[n2_coeff] + nodal_pos[stride + n2_coeff]) / 2;
      if (even_n2) {
        // compute the last nodal column
        *(nodal_coeff_pos++) -=
            (nodal_pos[n2_coeff + 1] + nodal_pos[stride + n2_coeff + 1]) / 2;
      }
    }
  }
  void compute_interpolant_difference_2D(T *data_pos, size_t n1, size_t n2,
                                         size_t stride) {
    size_t n1_nodal = (n1 >> 1) + 1;
    size_t n1_coeff = n1 - n1_nodal;
    size_t n2_nodal = (n2 >> 1) + 1;
    size_t n2_coeff = n2 - n2_nodal;
    // compute horizontal difference
    const T *nodal_pos = data_pos;
    T *coeff_pos = data_pos + n2_nodal;
    for (int i = 0; i < n1_nodal; i++) {
      compute_interpolant_difference_1D(n2_coeff, nodal_pos, coeff_pos);
      nodal_pos += stride, coeff_pos += stride;
    }
    // compute vertical difference
    compute_interpolant_difference_2D_vertical(data_pos, n1, n2, stride);
  }
  // decompose n1 x n2 data into coarse level (n1/2 x n2/2)
  void decompose_level_2D(T *data_pos, size_t n1, size_t n2, T h,
                          size_t stride) {
    // cerr << "decompose, h = " << h << endl;
    size_t n1_nodal = (n1 >> 1) + 1;
    size_t n1_coeff = n1 - n1_nodal;
    size_t n2_nodal = (n2 >> 1) + 1;
    size_t n2_coeff = n2 - n2_nodal;
    data_reorder_2D(data_pos, n1, n2, stride);
    compute_interpolant_difference_2D(data_pos, n1, n2, stride);
    vector<T> w1(n1_nodal);
    vector<T> b1(n1_nodal);
    vector<T> w2(n2_nodal);
    vector<T> b2(n2_nodal);
    precompute_w_and_b(w1.data(), b1.data(), n1_nodal);
    precompute_w_and_b(w2.data(), b2.data(), n2_nodal);
    compute_correction_2D(data_pos, data_buffer, load_v_buffer, n1, n2,
                          n1_nodal, h, stride, w1.data(), b1.data(), w2.data(),
                          b2.data(), default_batch_size);
    apply_correction_batched(data_pos, data_buffer, n1_nodal, stride, n2_nodal,
                             true);
  }
  void decompose_level_2D_with_hierarchical_basis(T *data_pos, size_t n1,
                                                  size_t n2, T h,
                                                  size_t stride) {
    data_reorder_2D(data_pos, n1, n2, stride);
    compute_interpolant_difference_2D(data_pos, n1, n2, stride);
  }
  /*
          2D reorder + vertical reorder
  */
  void data_reorder_3D(T *data_pos, size_t n1, size_t n2, size_t n3,
                       size_t dim0_stride, size_t dim1_stride) {
    size_t n1_nodal = (n1 >> 1) + 1;
    size_t n1_coeff = n1 - n1_nodal;
    size_t n2_nodal = (n2 >> 1) + 1;
    size_t n2_coeff = n2 - n2_nodal;
    size_t n3_nodal = (n3 >> 1) + 1;
    size_t n3_coeff = n3 - n3_nodal;
    T *cur_data_pos = data_pos;
    // do 2D reorder
    for (int i = 0; i < n1; i++) {
      data_reorder_2D(cur_data_pos, n2, n3, dim1_stride);
      cur_data_pos += dim0_stride;
    }
    if (!(n1 & 1)) {
      // n1 is even, change the last coeff plane into nodal plane
      cur_data_pos -= dim0_stride;
      for (int j = 0; j < n2; j++) {
        for (int k = 0; k < n3; k++) {
          cur_data_pos[k] =
              2 * cur_data_pos[k] - cur_data_pos[-dim0_stride + k];
        }
        cur_data_pos += dim1_stride;
      }
    }
    cur_data_pos = data_pos;
    // reorder vertically
    for (int j = 0; j < n2; j++) {
      switch_rows_2D_by_buffer(cur_data_pos, data_buffer, n1, n3, dim0_stride);
      cur_data_pos += dim1_stride;
    }
  }
  /*
          2D computation + vertical computation for coefficient plane
  */
  void compute_interpolant_difference_3D(T *data_pos, size_t n1, size_t n2,
                                         size_t n3, size_t dim0_stride,
                                         size_t dim1_stride) {
    size_t n1_nodal = (n1 >> 1) + 1;
    size_t n1_coeff = n1 - n1_nodal;
    size_t n2_nodal = (n2 >> 1) + 1;
    size_t n2_coeff = n2 - n2_nodal;
    size_t n3_nodal = (n3 >> 1) + 1;
    size_t n3_coeff = n3 - n3_nodal;
    bool even_n2 = (!(n2 & 1));
    bool even_n3 = (!(n3 & 1));
    T *cur_data_pos = data_pos;
    for (int i = 0; i < n1_nodal; i++) {
      compute_interpolant_difference_2D(cur_data_pos, n2, n3, dim1_stride);
      cur_data_pos += dim0_stride;
    }
    // compute vertically
    const T *nodal_pos = data_pos;
    T *coeff_pos = data_pos + n1_nodal * dim0_stride;
    for (int i = 0; i < n1_coeff; i++) {
      // iterate throught coefficient planes along n1
      /*
              data in the coefficient plane
              xxxxx		xxx xx
              xxxxx		xxx	coeff_nodal_nonal	xx
         coeff_nodal_coeff xxxxx	=>	xxx
         xx xxxxx xxxxx		xxx	coeff_coeff_nodal	xx
         coeff_coeff_coeff xxx						xx
      */
      const T *nodal_nodal_nodal_pos = nodal_pos;
      T *coeff_nodal_nodal_pos = coeff_pos;
      T *coeff_nodal_coeff_pos = coeff_pos + n3_nodal;
      T *coeff_coeff_nodal_pos = coeff_pos + n2_nodal * dim1_stride;
      T *coeff_coeff_coeff_pos = coeff_coeff_nodal_pos + n3_nodal;
      // TODO: optimize average computation
      for (int j = 0; j < n2_coeff; j++) {
        for (int k = 0; k < n3_coeff; k++) {
          // coeff_nodal_nonal
          coeff_nodal_nodal_pos[k] -= (nodal_nodal_nodal_pos[k] +
                                       nodal_nodal_nodal_pos[dim0_stride + k]) /
                                      2;
          // coeff_nodal_coeff
          coeff_nodal_coeff_pos[k] -=
              (nodal_nodal_nodal_pos[k] +
               nodal_nodal_nodal_pos[dim0_stride + k] +
               nodal_nodal_nodal_pos[k + 1] +
               nodal_nodal_nodal_pos[dim0_stride + k + 1]) /
              4;
          // coeff_coeff_nodal
          coeff_coeff_nodal_pos[k] -=
              (nodal_nodal_nodal_pos[k] +
               nodal_nodal_nodal_pos[dim0_stride + k] +
               nodal_nodal_nodal_pos[k + dim1_stride] +
               nodal_nodal_nodal_pos[dim0_stride + k + dim1_stride]) /
              4;
          // coeff_coeff_coeff
          coeff_coeff_coeff_pos[k] -=
              (nodal_nodal_nodal_pos[k] +
               nodal_nodal_nodal_pos[dim0_stride + k] +
               nodal_nodal_nodal_pos[k + 1] +
               nodal_nodal_nodal_pos[dim0_stride + k + 1] +
               nodal_nodal_nodal_pos[k + dim1_stride] +
               nodal_nodal_nodal_pos[dim0_stride + k + dim1_stride] +
               nodal_nodal_nodal_pos[k + dim1_stride + 1] +
               nodal_nodal_nodal_pos[dim0_stride + k + dim1_stride + 1]) /
              8;
        }
        // compute the last (or second last if n3 is even) coeff_*_nodal column
        coeff_nodal_nodal_pos[n3_coeff] -=
            (nodal_nodal_nodal_pos[n3_coeff] +
             nodal_nodal_nodal_pos[dim0_stride + n3_coeff]) /
            2;
        coeff_coeff_nodal_pos[n3_coeff] -=
            (nodal_nodal_nodal_pos[n3_coeff] +
             nodal_nodal_nodal_pos[dim0_stride + n3_coeff] +
             nodal_nodal_nodal_pos[n3_coeff + dim1_stride] +
             nodal_nodal_nodal_pos[dim0_stride + n3_coeff + dim1_stride]) /
            4;
        if (even_n3) {
          // compute the last coeff_*_nodal column if n3 is even
          coeff_nodal_nodal_pos[n3_coeff + 1] -=
              (nodal_nodal_nodal_pos[n3_coeff + 1] +
               nodal_nodal_nodal_pos[dim0_stride + n3_coeff + 1]) /
              2;
          coeff_coeff_nodal_pos[n3_coeff + 1] -=
              (nodal_nodal_nodal_pos[n3_coeff + 1] +
               nodal_nodal_nodal_pos[dim0_stride + n3_coeff + 1] +
               nodal_nodal_nodal_pos[n3_coeff + 1 + dim1_stride] +
               nodal_nodal_nodal_pos[dim0_stride + n3_coeff + 1 +
                                     dim1_stride]) /
              4;
        }
        coeff_nodal_nodal_pos += dim1_stride;
        coeff_nodal_coeff_pos += dim1_stride;
        coeff_coeff_nodal_pos += dim1_stride;
        coeff_coeff_coeff_pos += dim1_stride;
        nodal_nodal_nodal_pos += dim1_stride;
      }
      // compute the last (or second last if n2 is even) coeff_nodal_* row
      {
        for (int k = 0; k < n3_coeff; k++) {
          coeff_nodal_nodal_pos[k] -= (nodal_nodal_nodal_pos[k] +
                                       nodal_nodal_nodal_pos[dim0_stride + k]) /
                                      2;
          coeff_nodal_coeff_pos[k] -=
              (nodal_nodal_nodal_pos[k] +
               nodal_nodal_nodal_pos[dim0_stride + k] +
               nodal_nodal_nodal_pos[k + 1] +
               nodal_nodal_nodal_pos[dim0_stride + k + 1]) /
              4;
        }
        coeff_nodal_nodal_pos[n3_coeff] -=
            (nodal_nodal_nodal_pos[n3_coeff] +
             nodal_nodal_nodal_pos[dim0_stride + n3_coeff]) /
            2;
        if (even_n3) {
          coeff_nodal_nodal_pos[n3_coeff + 1] -=
              (nodal_nodal_nodal_pos[n3_coeff + 1] +
               nodal_nodal_nodal_pos[dim0_stride + n3_coeff + 1]) /
              2;
        }
        coeff_nodal_nodal_pos += dim1_stride;
        coeff_nodal_coeff_pos += dim1_stride;
        coeff_coeff_nodal_pos += dim1_stride;
        coeff_coeff_coeff_pos += dim1_stride;
        nodal_nodal_nodal_pos += dim1_stride;
      }
      if (even_n2) {
        // compute the last coeff_nodal_* row if n2 is even
        for (int k = 0; k < n3_coeff; k++) {
          coeff_nodal_nodal_pos[k] -= (nodal_nodal_nodal_pos[k] +
                                       nodal_nodal_nodal_pos[dim0_stride + k]) /
                                      2;
          coeff_nodal_coeff_pos[k] -=
              (nodal_nodal_nodal_pos[k] +
               nodal_nodal_nodal_pos[dim0_stride + k] +
               nodal_nodal_nodal_pos[k + 1] +
               nodal_nodal_nodal_pos[dim0_stride + k + 1]) /
              4;
        }
        coeff_nodal_nodal_pos[n3_coeff] -=
            (nodal_nodal_nodal_pos[n3_coeff] +
             nodal_nodal_nodal_pos[dim0_stride + n3_coeff]) /
            2;
        if (even_n3) {
          coeff_nodal_nodal_pos[n3_coeff + 1] -=
              (nodal_nodal_nodal_pos[n3_coeff + 1] +
               nodal_nodal_nodal_pos[dim0_stride + n3_coeff + 1]) /
              2;
        }
      }
      nodal_pos += dim0_stride;
      coeff_pos += dim0_stride;
    }
  }
  // decompse n1 x n2 x n3 data into coarse level (n1/2 x n2/2 x n3/2)
  void decompose_level_3D(T *data_pos, size_t n1, size_t n2, size_t n3, T h,
                          size_t dim0_stride, size_t dim1_stride) {
    data_reorder_3D(data_pos, n1, n2, n3, dim0_stride, dim1_stride);
    compute_interpolant_difference_3D(data_pos, n1, n2, n3, dim0_stride,
                                      dim1_stride);
    size_t n1_nodal = (n1 >> 1) + 1;
    size_t n2_nodal = (n2 >> 1) + 1;
    size_t n3_nodal = (n3 >> 1) + 1;
    compute_correction_3D(data_pos, data_buffer, load_v_buffer, n1, n2, n3,
                          n1_nodal, h, dim0_stride, dim1_stride,
                          default_batch_size);
    T *nodal_pos = data_pos;
    const T *correction_pos = data_buffer;
    for (int i = 0; i < n1_nodal; i++) {
      apply_correction_batched(nodal_pos, correction_pos, n2_nodal, dim1_stride,
                               n3_nodal, true);
      nodal_pos += dim0_stride;
      correction_pos += n2_nodal * n3_nodal;
    }
  }
  void decompose_level_3D_with_hierarchical_basis(T *data_pos, size_t n1,
                                                  size_t n2, size_t n3, T h,
                                                  size_t dim0_stride,
                                                  size_t dim1_stride) {
    data_reorder_3D(data_pos, n1, n2, n3, dim0_stride, dim1_stride);
    compute_interpolant_difference_3D(data_pos, n1, n2, n3, dim0_stride,
                                      dim1_stride);
  }
};

} // namespace MGARD

#endif