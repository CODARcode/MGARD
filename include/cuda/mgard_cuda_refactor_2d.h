#include "cuda/mgard_cuda_common.h"
namespace mgard_cuda {

template <typename T>
void 
refactor_2D_cuda(mgard_cuda_handle<T> & handle, T * dv, int lddv);

// template <typename T> 
// mgard_cuda_ret 
// refactor_2D_cuda_v1(mgard_cuda_handle<T> & handle, T * dv, int lddv);

// template <typename T> 
// mgard_cuda_ret 
// refactor_2D_cuda_v2(mgard_cuda_handle<T> & handle, T * dv, int lddv);

// template <typename T>
// mgard_cuda_ret
// refactor_2D_cuda_v3(mgard_cuda_handle<T> & handle, T * dv, int lddv);

template <typename T>
void
refactor_2D_cuda_cpt(mgard_cuda_handle<T> & handle, T * dv, int lddv);

}
