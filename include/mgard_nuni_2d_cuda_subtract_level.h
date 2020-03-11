#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"

namespace mgard_2d {
namespace mgard_cannon {

template <typename T>
mgard_cuda_ret 
subtract_level_cuda(int nrow,       int ncol, 
                    int row_stride, int col_stride,
                    T * dv,    int lddv, 
                    T * dwork, int lddwork);

}
}