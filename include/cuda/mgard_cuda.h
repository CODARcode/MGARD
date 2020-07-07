#include "cuda/mgard_cuda_common.h"


namespace mgard_cuda {

template <typename T>
unsigned char *
refactor_qz_cuda(mgard_cuda_handle<T> & handle, T * u, int &outsize, T tol);

template <typename T>
T *
recompose_udq_cuda(mgard_cuda_handle<T> & handle, unsigned char *data, int data_len);

template <typename T>
unsigned char *
refactor_qz_2D_cuda (mgard_cuda_handle<T> & handle, T *u, int &outsize, T tol);

template <typename T>
T * 
recompose_udq_2D_cuda(mgard_cuda_handle<T> & handle, unsigned char *data, int data_len);
}
