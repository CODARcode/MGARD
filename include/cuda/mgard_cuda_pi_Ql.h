#include "cuda/mgard_cuda_common.h"

namespace mgard_cuda {

template <typename T>
void pi_Ql(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nr, int nc,
           int row_stride, int col_stride, int *dirow, int *dicol, T *dcoords_y,
           T *dcoords_x, T *dv, int lddv, int queue_idx);

template <typename T>
void pi_Ql_cpt(mgard_cuda_handle<T> &handle, int nr, int nc, int row_stride,
               int col_stride, T *ddist_y, T *ddist_x, T *dv, int lddv,
               int queue_idx);

template <typename T>
void pi_Ql_first_1(mgard_cuda_handle<T> &handle, const int nrow, const int ncol,
                   const int nr, const int nc, int *dirow, int *dicol_p,
                   T *ddist_r, T *ddist_c, T *dv, int lddv, int queue_idx);
template <typename T>
void pi_Ql_first_2(mgard_cuda_handle<T> &handle, const int nrow, const int ncol,
                   const int nr, const int nc, int *dirow_p, int *dicol,
                   T *ddist_r, T *ddist_c, T *dv, int lddv, int queue_idx);
template <typename T>
void pi_Ql_first_12(mgard_cuda_handle<T> &handle, const int nrow,
                    const int ncol, const int nr, const int nc, int *dirow_p,
                    int *dicol_p, T *ddist_r, T *ddist_c, T *dv, int lddv,
                    int queue_idx);

template <typename T>
void pi_Ql_cpt(mgard_cuda_handle<T> &handle, int nr, int nc, int nf,
               int row_stride, int col_stride, int fib_stride, T *ddist_r,
               T *ddist_c, T *ddist_f, T *dv, int lddv1, int lddv2,
               int queue_idx);

template <typename T>
void pi_Ql_first_1(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nfib,
                   int nr, int nc, int nf, int *dirow, int *dicol, int *difib_p,
                   T *ddist_r, T *ddist_c, T *ddist_f, T *dv, int lddv1,
                   int lddv2, int queue_idx);

template <typename T>
void pi_Ql_first_2(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nfib,
                   int nr, int nc, int nf, int *dirow, int *dicolP, int *difib,
                   T *ddist_r, T *ddist_c, T *ddist_f, T *dv, int lddv1,
                   int lddv2, int queue_idx);

template <typename T>
void pi_Ql_first_3(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nfib,
                   int nr, int nc, int nf, int *dirowP, int *dicol, int *difib,
                   T *ddist_r, T *ddist_c, T *ddist_f, T *dv, int lddv1,
                   int lddv2, int queue_idx);

template <typename T>
void pi_Ql_first_12(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nfib,
                    int nr, int nc, int nf, int *dirow, int *dicol_p,
                    int *difib_p, T *ddist_r, T *ddist_c, T *ddist_f, T *dv,
                    int lddv1, int lddv2, int queue_idx);

template <typename T>
void pi_Ql_first_13(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nfib,
                    int nr, int nc, int nf, int *dirow_p, int *dicol,
                    int *difib_p, T *ddist_r, T *ddist_c, T *ddist_f, T *dv,
                    int lddv1, int lddv2, int queue_idx);

template <typename T>
void pi_Ql_first_23(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nfib,
                    int nr, int nc, int nf, int *dirow_p, int *dicol_p,
                    int *difib, T *ddist_r, T *ddist_c, T *ddist_f, T *dv,
                    int lddv1, int lddv2, int queue_idx);

template <typename T>
void pi_Ql_first_123(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nfib,
                     int nr, int nc, int nf, int *dirow_p, int *dicol_p,
                     int *difib_p, T *ddist_r, T *ddist_c, T *ddist_f, T *dv,
                     int lddv1, int lddv2, int queue_idx);
} // namespace mgard_cuda