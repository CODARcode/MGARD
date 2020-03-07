#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"


namespace mgard_2d {
namespace mgard_gen {
mgard_cuda_ret
restriction_l_row_cuda_sm(int nr,         int nc,
                          int row_stride, int col_stride, 
                          double * dv,    int lddv,
                          double * ddist_x,
                          int B, int ghost_col);
mgard_cuda_ret 
restriction_l_col_cuda_sm(int nr,         int nc,
                          int row_stride, int col_stride,
                          double * dv,    int lddv,
                          double * ddist_y,
                          int B, int ghost_row);

mgard_cuda_ret
restriction_l_row_cuda_sm_pf(int nrow,       int ncol,
                     int nr,         int nc,
                     int row_stride, int col_stride,
                     int * dirow,    int * dicol,
                     double * dv,    int lddv,
                     double * dcoords_x,
                     int B, int ghost_col);
}
}