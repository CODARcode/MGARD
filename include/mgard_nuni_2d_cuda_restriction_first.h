#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"


namespace mgard_2d {
namespace mgard_gen {

template <typename T>
mgard_cuda_ret 
restriction_first_row_cuda(int nrow,       int ncol, 
                           int row_stride, int * dicolP, int nc,
                           T * dv,    int lddv,
                           T * dcoords_x);

template <typename T>
mgard_cuda_ret 
restriction_first_col_cuda(int nrow,       int ncol, 
                           int * dirowP, int nr, int col_stride,
                           T * dv,    int lddv,
                           T * dcoords_y);

}
}