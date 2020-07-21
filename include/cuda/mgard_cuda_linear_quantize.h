#include "cuda/mgard_cuda_common.h"
namespace mgard_cuda {

template <typename T>
void linear_quantize(mgard_cuda_handle<T> &handle, int nrow, int ncol, T norm,
                     T tol, T *dv, int lddv, int *dwork, int lddwork,
                     int queue_idx);

template <typename T>
void linear_dequantize(mgard_cuda_handle<T> &handle, int nrow, int ncol, T *dv,
                       int lddv, int *dwork, int lddwork, int queue_idx);

template <typename T>
void linear_quantize(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nfib,
                     T norm, T tol, T *dv, int lddv1, int lddv2, int *dwork,
                     int lddwork1, int lddwork2, int queue_idx);

template <typename T>
void linear_dequantize(mgard_cuda_handle<T> &handle, int nrow, int ncol,
                       int nfib, T *dv, int lddv1, int lddv2, int *dwork,
                       int lddwork1, int lddwork2, int queue_idx);
} // namespace mgard_cuda