#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"


namespace mgard_2d {
namespace mgard_gen {

template <typename T>
mgard_cuda_ret
solve_tridiag_M_l_row_cuda(int nrow,       int ncol,
                           int nr,         int nc,
                           int row_stride, int col_stride,
                           int * dirow,    int * dicol,
                           T * dv,     int lddv, 
                           T * dcoords_x,
                           int B, mgard_cuda_handle & handle, 
                           int queue_idx, bool profile);

template <typename T>
mgard_cuda_ret
solve_tridiag_M_l_col_cuda(int nrow,       int ncol,
                           int nr,         int nc,
                           int row_stride, int col_stride,
                           int * dirow,    int * dicol,
                           T * dv,    int lddv, 
                           T * dcoords_y,
                           int B, mgard_cuda_handle & handle, 
                           int queue_idx, bool profile);


template <typename T>
mgard_cuda_ret
calc_am_bm(int n, T * am, T * bm, T * ddist,
           int B, mgard_cuda_handle & handle, 
           int queue_idx, bool profile);


template <typename T>
mgard_cuda_ret 
solve_tridiag_M_l_row_forward_cuda_sm(int nr,         int nc,
                                      int row_stride, int col_stride,
                                      T * bm,
                                      T * dv,    int lddv,
                                      int B, int ghost_col,
                                      mgard_cuda_handle & handle, 
                                      int queue_idx, bool profile);

template <typename T>
mgard_cuda_ret 
solve_tridiag_M_l_row_backward_cuda_sm(int nr,         int nc,
                                      int row_stride, int col_stride,
                                      T * am,    T * ddist_x,
                                      T * dv,    int lddv,
                                      int B, int ghost_col,
                                      mgard_cuda_handle & handle, 
                                      int queue_idx, bool profile);

template <typename T>
mgard_cuda_ret 
solve_tridiag_M_l_row_cuda_sm(int nr,         int nc,
                              int row_stride, int col_stride,
                              T * dv,    int lddv,
                              T * ddist_x,
                              T * am, T * bm,
                              int B, int ghost_col,
                              mgard_cuda_handle & handle, 
                              int queue_idx, bool profile);


template <typename T>
mgard_cuda_ret 
solve_tridiag_M_l_col_forward_cuda_sm(int nr,         int nc,
                                      int row_stride, int col_stride,
                                      T * bm,
                                      T * dv,    int lddv,
                                      int B, int ghost_row,
                                      mgard_cuda_handle & handle, 
                                      int queue_idx, bool profile);

template <typename T>
mgard_cuda_ret 
solve_tridiag_M_l_col_backward_cuda_sm(int nr,         int nc,
                                      int row_stride, int col_stride,
                                      T * am,    T * ddist_y,
                                      T * dv,    int lddv,
                                      int B, int ghost_row,
                                      mgard_cuda_handle & handle, 
                                      int queue_idx, bool profile);

template <typename T>
mgard_cuda_ret 
solve_tridiag_M_l_col_cuda_sm(int nr,         int nc,
                              int row_stride, int col_stride,
                              T * dv,    int lddv,
                              T * ddist_y,
                              T * am, T * bm,
                              int B, int ghost_row,
                              mgard_cuda_handle & handle, 
                              int queue_idx, bool profile);

}
}