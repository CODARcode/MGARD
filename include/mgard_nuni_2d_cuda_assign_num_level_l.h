#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"


namespace mgard_2d {
namespace mgard_gen {

template <typename T>
mgard_cuda_ret 
assign_num_level_l_cuda(int nrow,           int ncol,
                        int nr,             int nc,
                        int row_stride,     int col_stride,
                        int * dirow,        int * dicol,
                        T * dv,        int lddv,
                        T num, 
                        int B, mgard_cuda_handle & handle, 
                        int queue_idx, bool profile);

template <typename T>
mgard_cuda_ret 
assign_num_level_l_cuda_l2_sm(int nr,             int nc,
                              int row_stride,     int col_stride,
                              T * dv,        int lddv,
                              T num,
                              int B, mgard_cuda_handle & handle, 
                              int queue_idx, bool profile);

}
}