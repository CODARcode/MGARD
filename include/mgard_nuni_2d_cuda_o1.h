#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"


namespace mgard_2d {
namespace mgard_gen {

mgard_cuda_ret 
compact_to_2k_plus_1(int nrow,     int ncol,
                     int nr,       int nc,
                     int * dirow,  int * dicol,
                     double * dv,  int lddv,
                     double * dcv, int lddcv);

mgard_cuda_ret 
restore_from_2k_plus_1(int nrow,     int ncol,
                       int nr,       int nc,
                       int * dirow,  int * dicol,
                       double * dcv, int lddcv,
                       double * dv,  int lddv);


mgard_cuda_ret 
original_to_compacted_cuda(int nrow,      int ncol, 
                          int row_stride, int col_stride,
                          double * dv,    int lddv, 
                          double * dcv,   int lddcv);

mgard_cuda_ret
compacted_to_original_cuda(int nrow, int ncol, 
                           int row_stride, int col_stride, 
                           double * dcv, int lddcv, 
                           double * dv, int lddv);

mgard_cuda_ret 
mass_mult_l_row_cuda_o1_config(int nrow,       int ncol,
                     int nr,         int nc,
                     int row_stride, int col_stride,
                     int * dirow,    int * dicol,
                     double * dv,    int lddv,
                     double * dcoords_x,
                     int B, int ghost_col);

mgard_cuda_ret 
mass_mult_l_row_cuda_o2_config(int nrow,       int ncol,
                     int nr,         int nc,
                     int row_stride, int col_stride,
                     int * dirow,    int * dicol,
                     double * dv,    int lddv,
                     double * dcoords_x,
                     int B, int ghost_col);

mgard_cuda_ret 
mass_mult_l_row_cuda_o1(int nrow,       int ncol,
                     int nr,         int nc,
                     int row_stride, int col_stride,
                     int * dirow,    int * dicol,
                     double * dv,    int lddv,
                     double * dcoords_x);
  
void 
refactor_2D_cuda_o1(const int l_target,
                    const int nrow,     const int ncol,
                    const int nr,       const int nc, 
                    int * dirow,        int * dicol,
                    int * dirowP,       int * dicolP,
                    double * dv,        int lddv, 
                    double * dwork,     int lddwork,
                    double * dcoords_x, double * dcoords_y);

void 
refactor_2D_cuda_o2(const int l_target,
                    const int nrow,     const int ncol,
                    const int nr,       const int nc, 
                    int * dirow,        int * dicol,
                    int * dirowP,       int * dicolP,
                    double * dv,        int lddv, 
                    double * dwork,     int lddwork,
                    double * dcoords_x, double * dcoords_y);


}
}