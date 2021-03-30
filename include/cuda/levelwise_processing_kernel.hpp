#include "cuda/levelwise_processing_kernel.h"
namespace mgard_cuda {

template <typename T, int D, int R, int C, int F, int OP>
__global__ void _lwpk(int * shape,
                      T *dv, int * ldvs, T *dwork, int * ldws) {

  size_t threadId =  (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
  int * sm = SharedMemory<int>();
  int * shape_sm = sm;
  int * ldvs_sm = shape_sm + D;
  int * ldws_sm = ldvs_sm + D;

  if (threadId < D) {
    shape_sm[threadId] = shape[threadId];
    ldvs_sm[threadId] = ldvs[threadId];
    ldws_sm[threadId] = ldws[threadId];
  }
  __syncthreads();

  int idx[D];
  int firstD = div_roundup(shape_sm[0], F);

  int bidx = blockIdx.x;
  idx[0] = (bidx % firstD) * F + threadIdx.x;

  // printf("firstD %d idx[0] %d\n", firstD, idx[0]);

  bidx /= firstD;
  if (D >= 2) idx[1] = blockIdx.y * blockDim.y + threadIdx.y;
  if (D >= 3) idx[2] = blockIdx.z * blockDim.z + threadIdx.z;

  for (int d = 3; d < D; d++) {
    idx[d] = bidx % shape_sm[d];
    bidx /= shape_sm[d];
  }
  // int z = blockIdx.z * blockDim.z + threadIdx.z;
  // int y = blockIdx.y * blockDim.y + threadIdx.y;
  // int x = blockIdx.z * blockDim.z + threadIdx.z;
  bool in_range = true;
  for (int d = 0; d < D; d++) {
    if (idx[d] >= shape_sm[d]) in_range = false;
  }
  if (in_range) {
    // printf("%d %d %d %d\n", idx[3], idx[2], idx[1], idx[0]);
    if (OP == COPY) dwork[get_idx<D>(ldws, idx)] = dv[get_idx<D>(ldvs, idx)];
    if (OP == ADD) dwork[get_idx<D>(ldws, idx)] += dv[get_idx<D>(ldvs, idx)];
    if (OP == SUBTRACT) dwork[get_idx<D>(ldws, idx)] -= dv[get_idx<D>(ldvs, idx)];
  }
}

template <typename T, int D, int R, int C, int F, int OP>
void lwpk_adaptive_launcher(mgard_cuda_handle<T, D> &handle, thrust::device_vector<int> shape,
                   T *dv, thrust::device_vector<int> ldvs, T *dwork, thrust::device_vector<int> ldws,
                   int queue_idx) {

  int total_thread_z = shape[2];
  int total_thread_y = shape[1];
  int total_thread_x = shape[0];
  // linearize other dimensions
  int tbz = R;
  int tby = C;
  int tbx = F;
  int gridz = ceil((float)total_thread_z / tbz);
  int gridy = ceil((float)total_thread_y / tby);
  int gridx = ceil((float)total_thread_x / tbx);
  for (int d = 3; d < D; d++) { gridx *= shape[d]; }

  // printf("exec: %d %d %d %d %d %d\n", tbx, tby, tbz, gridx, gridy, gridz);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);
  size_t sm_size = (D * 3) * sizeof(int);
  _lwpk<T, D, R, C, F, OP><<<blockPerGrid, threadsPerBlock, sm_size,
                   *(cudaStream_t *)handle.get(queue_idx)>>>(
                    thrust::raw_pointer_cast(shape.data()), 
                    dv, thrust::raw_pointer_cast(ldvs.data()), 
                    dwork, thrust::raw_pointer_cast(ldws.data()));

  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template <typename T, int D, int OP>
void lwpk(mgard_cuda_handle<T, D> &handle, thrust::device_vector<int> shape,
                   T *dv, thrust::device_vector<int> ldvs, T *dwork, thrust::device_vector<int> ldws,
                   int queue_idx) {
#define COPYLEVEL(R, C, F) {lwpk_adaptive_launcher<T, D, R, C, F, OP>(\
                                  handle, shape,\
                                  dv, ldvs, dwork, ldws,\
                                  queue_idx);}
  if (D >= 3) { COPYLEVEL(4, 4, 4) }
  if (D == 2) { COPYLEVEL(1, 4, 4) }
  if (D == 1) { COPYLEVEL(1, 1, 8) }

#undef COPYLEVEL

}

template <typename T, int D, int R, int C, int F, int OP>
void lwpk_adaptive_launcher(mgard_cuda_handle<T, D> &handle, 
                    int * shape_h, int * shape_d,
                    T *dv, int * ldvs, 
                    T *dwork, int * ldws,
                   int queue_idx) {

  int total_thread_z = shape_h[2];
  int total_thread_y = shape_h[1];
  int total_thread_x = shape_h[0];
  // linearize other dimensions
  int tbz = R;
  int tby = C;
  int tbx = F;
  int gridz = ceil((float)total_thread_z / tbz);
  int gridy = ceil((float)total_thread_y / tby);
  int gridx = ceil((float)total_thread_x / tbx);
  for (int d = 3; d < D; d++) { gridx *= shape_h[d]; }

  // printf("exec: %d %d %d %d %d %d\n", tbx, tby, tbz, gridx, gridy, gridz);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);
  size_t sm_size = (D * 3) * sizeof(int);
  _lwpk<T, D, R, C, F, OP><<<blockPerGrid, threadsPerBlock, sm_size,
                   *(cudaStream_t *)handle.get(queue_idx)>>>(
                    shape_d, 
                    dv, ldvs, 
                    dwork, ldws);

  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template <typename T, int D, int OP>
void lwpk(mgard_cuda_handle<T, D> &handle, 
          int * shape_h, int * shape_d,
          T *dv, int * ldvs, 
          T *dwork, int * ldws,
          int queue_idx) {
#define COPYLEVEL(R, C, F) {lwpk_adaptive_launcher<T, D, R, C, F, OP>(\
                                  handle, shape_h, shape_d,\
                                  dv, ldvs,\
                                  dwork, ldws,\
                                  queue_idx);}
  if (D >= 3) { COPYLEVEL(4, 4, 4) }
  if (D == 2) { COPYLEVEL(1, 4, 4) }
  if (D == 1) { COPYLEVEL(1, 1, 8) }

#undef COPYLEVEL

}

}