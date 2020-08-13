#include "cuda/mgard_cuda_common.h"

namespace mgard_cuda {

template <typename T>
void restriction_1(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nr,
                   int nc, int row_stride, int col_stride, int *dirow,
                   int *dicol, T *dcoords_x, T *dv, int lddv, int queue_idx);

template <typename T>
void restriction_2(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nr,
                   int nc, int row_stride, int col_stride, int *dirow,
                   int *dicol, T *dcoords_y, T *dv, int lddv, int queue_idx);

template <typename T>
void restriction_1_cpt(mgard_cuda_handle<T> &handle, int nr, int nc,
                       int row_stride, int col_stride, T *ddist_x, T *dv,
                       int lddv, int queue_idx);

template <typename T>
void restriction_2_cpt(mgard_cuda_handle<T> &handle, int nr, int nc,
                       int row_stride, int col_stride, T *ddist_y, T *dv,
                       int lddv, int queue_idx);

template <typename T>
void restriction_first_1(mgard_cuda_handle<T> &handle, int nrow, int ncol,
                         int nr, int nc, int row_stride, int col_stride,
                         int *dirow, int *dicol_p, T *ddist_c, T *dv, int lddv,
                         int queue_idx);

template <typename T>
void restriction_first_2(mgard_cuda_handle<T> &handle, int nrow, int ncol,
                         int nr, int nc, int row_stride, int col_stride,
                         int *irow_p, int *icol, T *dist_r, T *dv, int lddv,
                         int queue_idx);
} // namespace mgard_cuda