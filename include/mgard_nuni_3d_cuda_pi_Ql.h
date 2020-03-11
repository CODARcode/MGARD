#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_3d_cuda_cpt_l2_sm.h"
#include "mgard_cuda_helper.h"

namespace mgard_gen {

template <typename T>
mgard_cuda_ret 
pi_Ql_cuda_cpt_sm(int nr,           int nc,           int nf, 
                  int row_stride,   int col_stride,   int fib_stride, 
                  T * dv,      int lddv1,        int lddv2, 
                  T * ddist_r, T * ddist_c, T * ddist_f,
                  int B);
}