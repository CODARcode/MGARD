#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_3d_cuda.h"
#include "mgard_cuda_helper.h"

namespace mgard_gen {

mgard_cuda_ret 
copy_level_l_cuda_cpt(int nf,         int nr,         int nc,
                      int fib_stride, int row_stride, int col_stride,
                      double * dv,    int lddv1,      int lddv2,
                      double * dwork, int lddwork1,   int lddwork2);

}