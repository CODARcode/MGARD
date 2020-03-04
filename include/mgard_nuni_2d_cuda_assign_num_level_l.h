#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"


namespace mgard_2d {
namespace mgard_gen {

mgard_cuda_ret 
assign_num_level_l_cuda_l2_sm(int nr,             int nc,
                              int row_stride,     int col_stride,
                              double * dv,        int lddv,
                              double num);

}
}