#include "cuda/mgard_cuda_common.h"

namespace mgard_cuda {

template <typename T> __device__ T _dist(T *dcoord, double x, double y);

template <typename T>
void org_to_pow2p1(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nfib,
                   int nr, int nc, int nf, int *dirow, int *dicol, int *difib,
                   T *dv, int lddv1, int lddv2, T *dcv, int lddcv1, int lddcv2,
                   int queue_idx);

template <typename T>
void org_to_pow2p1(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nr,
                   int nc, int *dirow, int *dicol, T *dv, int lddv, T *dcv,
                   int lddcv, int queue_idx);

template <typename T>
void org_to_pow2p1(mgard_cuda_handle<T> &handle, int nrow, int nr, int *dirow,
                   T *dv, T *dcv, int queue_idx);

template <typename T>
void pow2p1_to_org(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nfib,
                   int nr, int nc, int nf, int *dirow, int *dicol, int *difib,
                   T *dcv, int lddcv1, int lddcv2, T *dv, int lddv1, int lddv2,
                   int queue_idx);

template <typename T>
void pow2p1_to_org(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nr,
                   int nc, int *dirow, int *dicol, T *dcv, int lddcv, T *dv,
                   int lddv, int queue_idx);

template <typename T>
void pow2p1_to_org(mgard_cuda_handle<T> &handle, int nrow, int nr, int *dirow,
                   T *dcv, T *dv, int queue_idx);

template <typename T>
void pow2p1_to_cpt(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nfib,
                   int row_stride, int col_stride, int fib_stride, T *dv,
                   int lddv1, int lddv2, T *dcv, int lddcv1, int lddcv2,
                   int queue_idx);

template <typename T>
void pow2p1_to_cpt(mgard_cuda_handle<T> &handle, int nrow, int ncol,
                   int row_stride, int col_stride, T *dv, int lddv, T *dcv,
                   int lddcv, int queue_idx);

template <typename T>
void pow2p1_to_cpt_num_assign(mgard_cuda_handle<T> &handle, int nrow, int ncol,
                              int row_stride, int col_stride, T val, T *dv,
                              int lddv, T *dcv, int lddcv, int queue_idx);

template <typename T>
void pow2p1_to_cpt(mgard_cuda_handle<T> &handle, int nrow, int row_stride,
                   T *dv, T *dcv, int queue_idx);

template <typename T>
void cpt_to_pow2p1(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nfib,
                   int row_stride, int col_stride, int fib_stride, T *dcv,
                   int lddcv1, int lddcv2, T *dv, int lddv1, int lddv2,
                   int queue_idx);

template <typename T>
void cpt_to_pow2p1(mgard_cuda_handle<T> &handle, int nrow, int ncol,
                   int row_stride, int col_stride, T *dcv, int lddcv, T *dv,
                   int lddv, int queue_idx);

template <typename T>
void cpt_to_pow2p1_add(mgard_cuda_handle<T> &handle, int nrow, int ncol,
                       int row_stride1, int col_stride1, int row_stride2,
                       int col_stride2, T *dcv, int lddcv, T *dv, int lddv,
                       int queue_idx);

template <typename T>
void cpt_to_pow2p1_subtract(mgard_cuda_handle<T> &handle, int nrow, int ncol,
                            int row_stride1, int col_stride1, int row_stride2,
                            int col_stride2, T *dcv, int lddcv, T *dv, int lddv,
                            int queue_idx);

template <typename T>
void cpt_to_pow2p1(mgard_cuda_handle<T> &handle, int nrow, int row_stride,
                   T *dcv, T *dv, int queue_idx);

template <typename T>
void calc_cpt_dist(mgard_cuda_handle<T> &handle, int nrow, int row_stride,
                   T *dcoord, T *ddist, int queue_idx);

} // namespace mgard_cuda