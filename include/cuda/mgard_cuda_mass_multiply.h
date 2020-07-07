#include "cuda/mgard_cuda_common.h"

namespace mgard_cuda {

template <typename T>
void 
mass_multiply_1(mgard_cuda_handle<T> & handle, 
                int nrow,       int ncol,
                int nr,         int nc,
                int row_stride, int col_stride,
                int * dirow,    int * dicol,
                T * dcoords_x,
                T * dv,    int lddv,
                int queue_idx);

template <typename T>
void 
mass_multiply_2(mgard_cuda_handle<T> & handle, 
                int nrow,       int ncol,
               int nr,         int nc,
               int row_stride, int col_stride,
               int * dirow,    int * dicol,
               T * dcoords_y,
               T * dv,    int lddv,
               int queue_idx);

template <typename T>
void 
mass_multiply_1_cpt(mgard_cuda_handle<T> & handle,
                    int nr,         int nc,
                    int row_stride, int col_stride,
                    T * ddist_x,
                    T * dv,    int lddv,
                    int queue_idx);

template <typename T>
void 
mass_multiply_2_cpt(mgard_cuda_handle<T> & handle, 
                    int nr,         int nc,
                    int row_stride, int col_stride,
                    T * ddist_y,
                    T * dv,    int lddv,
                    int queue_idx);

// template <typename T>
// mgard_cuda_ret 
// mass_mult_l_row_cuda_sm_pf(int nrow,       int ncol,
//                      int nr,         int nc,
//                      int row_stride, int col_stride,
//                      int * dirow,    int * dicol,
//                      T * dv,    int lddv,
//                      T * dcoords_x,
//                      int B, int ghost_col,
//                      mgard_cuda_handle<T> & handle, 
//                      int queue_idx, bool profile);
// template <typename T>
// mgard_cuda_ret 
// mass_mult_l_row_cuda_sm_pf(int nrow,       int ncol,
//                      int nr,         int nc,
//                      int row_stride, int col_stride,
//                      int * dirow,    int * dicol,
//                      T * dv,    int lddv,
//                      T * dcoords_x,
//                      int B, int ghost_col,
//                      mgard_cuda_handle<T> & handle, 
//                      int queue_idx, bool profile);
}
