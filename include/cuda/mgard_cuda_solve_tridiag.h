#include "cuda/mgard_cuda_common.h"

namespace mgard_cuda {

template <typename T>
void solve_tridiag_1(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nr,
                     int nc, int row_stride, int col_stride, int *dirow,
                     int *dicol, T *dcoords_x, T *dv, int lddv, int queue_idx);

template <typename T>
void solve_tridiag_2(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nr,
                     int nc, int row_stride, int col_stride, int *dirow,
                     int *dicol, T *dcoords_y, T *dv, int lddv, int queue_idx);

template <typename T>
void calc_am_bm(mgard_cuda_handle<T> &handle, int n, T *ddist, T *am, T *bm,
                int queue_idx);

template <typename T>
void solve_tridiag_forward_1_cpt(mgard_cuda_handle<T> &handle, int nr, int nc,
                                 int row_stride, int col_stride, T *bm, T *dv,
                                 int lddv, int queue_idx);

template <typename T>
void solve_tridiag_backward_1_cpt(mgard_cuda_handle<T> &handle, int nr, int nc,
                                  int row_stride, int col_stride, T *ddist_x,
                                  T *am, T *dv, int lddv, int queue_idx);

template <typename T>
void solve_tridiag_1_cpt(mgard_cuda_handle<T> &handle, int nr, int nc,
                         int row_stride, int col_stride, T *ddist_x, T *am,
                         T *bm, T *dv, int lddv, int queue_idx);

template <typename T>
void solve_tridiag_forward_2_cpt(mgard_cuda_handle<T> &handle, int nr, int nc,
                                 int row_stride, int col_stride, T *bm, T *dv,
                                 int lddv, int queue_idx);

template <typename T>
void solve_tridiag_backward_2_cpt(mgard_cuda_handle<T> &handle, int nr, int nc,
                                  int row_stride, int col_stride, T *ddist_y,
                                  T *am, T *dv, int lddv, int queue_idx);

template <typename T>
void solve_tridiag_2_cpt(mgard_cuda_handle<T> &handle, int nr, int nc,
                         int row_stride, int col_stride, T *ddist_y, T *am,
                         T *bm, T *dv, int lddv, int queue_idx);

} // namespace mgard_cuda