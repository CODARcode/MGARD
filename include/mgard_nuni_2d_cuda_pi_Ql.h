#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"


namespace mgard_2d {
namespace mgard_gen {

template <typename T>
mgard_cuda_ret 
pi_Ql_cuda(int nrow,           int ncol,
           int nr,             int nc,
           int row_stride,     int col_stride,
           int * dirow,        int * dicol,
           T * dv,        int lddv, 
           T * dcoords_x, T * dcoords_y);

template <typename T>
mgard_cuda_ret 
pi_Ql_cuda_sm(int nr,         int nc,
              int row_stride, int col_stride,
              T * dv,    int lddv,
              T * ddist_x, T * ddist_y,
              int B);

}
}