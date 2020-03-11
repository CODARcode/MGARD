#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"


namespace mgard_2d {
namespace mgard_gen {

template <typename T>
mgard_cuda_ret 
pi_Ql_first_cuda(const int nrow,     const int ncol,
                 const int nr,       const int nc,
                 int * dirow,        int * dicol,
                 int * dirowP,       int * dicolP,
                 T * dcoords_x,     T * dcoords_y,
                 T * dv,           const int lddv);

}
}