#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_3d_cuda_cpt_l2_sm.h"
#include "mgard_cuda_helper.h"

namespace mgard_gen {

template <typename T>
mgard_cuda_ret 
assign_num_level_l_cuda_cpt(int nr,         int nc,         int nf,
                            int row_stride, int col_stride, int fib_stride, 
                            T * dwork, int lddwork1,   int lddwork2,
                            T num, 
                            int B, mgard_cuda_handle & handle, 
                            int queue_idx, bool profile);

}