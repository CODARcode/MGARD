#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_3d_cuda_cpt_l2_sm.h"
#include "mgard_cuda_helper.h"

namespace mgard_gen {

template <typename T>
mgard_cuda_ret 
add_level_l_cuda_cpt(int nr,         int nc,         int nf,
                     int row_stride, int col_stride, int fib_stride,
                     T * dv,    int lddv1,      int lddv2,
                     T * dwork, int lddwork1,   int lddwork2,
                     int B, mgard_cuda_handle & handle, 
                     int queue_idx, bool profile);

}