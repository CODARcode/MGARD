#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_3d_cuda_cpt_l2_sm.h"
#include "mgard_cuda_helper.h"

namespace mgard_gen {

template <typename T>
mgard_cuda_ret 
pi_Ql3D_first_fib(int nrow,        int ncol,         int nfib,
                  int nr,          int nc,           int nf, 
                  int * dirow,     int * dicol,      int * difibP,
                  T * dv,      int lddv1,        int lddv2, 
                  T * ddist_r, T * ddist_c, T * ddist_f,
                  int B, mgard_cuda_handle & handle, 
                  int queue_idx, bool profile);
template <typename T>
mgard_cuda_ret 
pi_Ql3D_first_col(int nrow,        int ncol,         int nfib,
                  int nr,          int nc,           int nf, 
                  int * dirow,     int * dicolP,      int * difib,
                  T * dv,      int lddv1,        int lddv2, 
                  T * ddist_r, T * ddist_c, T * ddist_f,
                  int B, mgard_cuda_handle & handle, 
                  int queue_idx, bool profile);
template <typename T>
mgard_cuda_ret 
pi_Ql3D_first_row(int nrow,        int ncol,         int nfib,
                  int nr,          int nc,           int nf, 
                  int * dirowP,     int * dicol,      int * difib,
                  T * dv,      int lddv1,        int lddv2, 
                  T * ddist_r, T * ddist_c, T * ddist_f,
                  int B, mgard_cuda_handle & handle, 
                  int queue_idx, bool profile);
template <typename T>
mgard_cuda_ret 
pi_Ql3D_first_fib_col(int nrow,        int ncol,         int nfib,
                  int nr,          int nc,           int nf, 
                  int * dirow,     int * dicolP,      int * difibP,
                  T * dv,      int lddv1,        int lddv2, 
                  T * ddist_r, T * ddist_c, T * ddist_f,
                  int B, mgard_cuda_handle & handle, 
                  int queue_idx, bool profile);
template <typename T>
mgard_cuda_ret 
pi_Ql3D_first_fib_row(int nrow,        int ncol,         int nfib,
                  int nr,          int nc,           int nf, 
                  int * dirowP,     int * dicol,      int * difibP,
                  T * dv,      int lddv1,        int lddv2, 
                  T * ddist_r, T * ddist_c, T * ddist_f,
                  int B, mgard_cuda_handle & handle, 
                  int queue_idx, bool profile);
template <typename T>
mgard_cuda_ret 
pi_Ql3D_first_col_row(int nrow,        int ncol,         int nfib,
                  int nr,          int nc,           int nf, 
                  int * dirowP,     int * dicolP,      int * difib,
                  T * dv,      int lddv1,        int lddv2, 
                  T * ddist_r, T * ddist_c, T * ddist_f,
                  int B, mgard_cuda_handle & handle, 
                  int queue_idx, bool profile);
template <typename T>
mgard_cuda_ret 
pi_Ql3D_first_fib_col_row(int nrow,        int ncol,         int nfib,
                  int nr,          int nc,           int nf, 
                  int * dirowP,     int * dicolP,      int * difibP,
                  T * dv,      int lddv1,        int lddv2, 
                  T * ddist_r, T * ddist_c, T * ddist_f,
                  int B, mgard_cuda_handle & handle, 
                  int queue_idx, bool profile);

}