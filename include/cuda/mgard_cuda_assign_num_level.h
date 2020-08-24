#include "cuda/mgard_cuda_common.h"

namespace mgard_cuda {

template <typename T>
void assign_num_level(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nr,
                      int nc, int row_stride, int col_stride, int *dirow,
                      int *dicol, T num, T *dv, int lddv, int queue_idx);

template <typename T>
void assign_num_level_cpt(mgard_cuda_handle<T> &handle, int nr, int nc,
                          int row_stride, int col_stride, T num, T *dv,
                          int lddv, int queue_idx);

template <typename T>
void assign_num_level(mgard_cuda_handle<T> &handle, int nrow, int ncol,
                      int nfib, int nr, int nc, int nf, int row_stride,
                      int col_stride, int fib_stride, int *irow, int *icol,
                      int *ifib, T num, T *dwork, int lddwork1, int lddwork2,
                      int queue_idx);

template <typename T>
void assign_num_level_cpt(mgard_cuda_handle<T> &handle, int nr, int nc, int nf,
                          int row_stride, int col_stride, int fib_stride, T num,
                          T *dwork, int lddwork1, int lddwork2, int queue_idx);

} // namespace mgard_cuda
