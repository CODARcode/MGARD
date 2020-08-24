#include "cuda/mgard_cuda_common.h"

namespace mgard_cuda {

template <typename T>
void prep_2D_cuda(mgard_cuda_handle<T> &handle, T *dv, int lddv);

template <typename T>
void prep_2D_cuda_cpt(mgard_cuda_handle<T> &handle, T *dv, int lddv);

} // namespace mgard_cuda