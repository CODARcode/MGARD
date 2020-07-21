#include "cuda/mgard_cuda_common.h"
namespace mgard_cuda {

template <typename T>
void prep_3D_cuda_cpt(mgard_cuda_handle<T> &handle, T *dv, int lddv1,
                      int lddv2);
}