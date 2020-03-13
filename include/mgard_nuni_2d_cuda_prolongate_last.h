#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"


namespace mgard_2d {
namespace mgard_gen {


template <typename T>
mgard_cuda_ret 
prolongate_last_row_cuda(int nrow,       int ncol, 
                         int nr,         int nc,
                         int row_stride, int col_stride,
                         int * dirow,    int * dicolP,
                         T * dv,    int lddv,
                         T * dcoords_x,
                         int B, mgard_cuda_handle & handle, 
                         int queue_idx, bool profile);

template <typename T>
mgard_cuda_ret 
prolongate_last_col_cuda(int nrow,       int ncol,
                         int nr,         int nc,
                         int row_stride, int col_stride,
                         int * dirowP,   int * dicol, 
                         T * dv,    int lddv,
                         T * dcoords_y,
                         int B, mgard_cuda_handle & handle, 
                         int queue_idx, bool profile);

}
}