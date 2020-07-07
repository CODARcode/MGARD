#include "cuda/mgard_cuda_common.h"
namespace mgard_cuda {

template <typename T>
void 
subtract_level(mgard_cuda_handle<T> & handle, 
               int nrow,       int ncol, 
               int nr,         int nc,
               int row_stride, int col_stride,
               int * dirow,    int * dicol,
               T * dv,    int lddv, 
               T * dwork, int lddwork,
               int queue_idx);

template <typename T>
void 
subtract_level_cpt(mgard_cuda_handle<T> & handle, 
                   int nrow,       int ncol, 
                   int row_stride, int col_stride,
                   T * dv,    int lddv, 
                   T * dwork, int lddwork,
                   int queue_idx);

template <typename T>
void 
subtract_level(mgard_cuda_handle<T> & handle, 
               int nrow,       int ncol, int nfib,
               int nr,         int nc,         int nf,
               int row_stride, int col_stride, int fib_stride,
               int * dirow,    int * dicol, int * difib,
               T * dv,    int lddv1,      int lddv2,
               T * dwork, int lddwork1,   int lddwork2,
               int queue_idx);

template <typename T>
void 
subtract_level_cpt(mgard_cuda_handle<T> & handle, 
                   int nr,         int nc,         int nf,
                   int row_stride, int col_stride, int fib_stride,
                   T * dv,    int lddv1,      int lddv2,
                   T * dwork, int lddwork1,   int lddwork2,
                   int queue_idx);



}