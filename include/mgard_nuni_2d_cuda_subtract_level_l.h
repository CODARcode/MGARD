#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"


namespace mgard_2d {
namespace mgard_gen {

template <typename T>
mgard_cuda_ret 
subtract_level_l_cuda(int nrow,       int ncol, 
                      int nr,         int nc,
                      int row_stride, int col_stride,
                      int * dirow,    int * dicol,
                      T * dv,    int lddv, 
                      T * dwork, int lddwork,
                      int B, mgard_cuda_handle & handle, 
                      int queue_idx, bool profile);

}
}