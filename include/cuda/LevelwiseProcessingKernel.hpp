/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGRAD_CUDA_LEVELWISE_PROCESSING_KERNEL_TEMPLATE
#define MGRAD_CUDA_LEVELWISE_PROCESSING_KERNEL_TEMPLATE

#include "CommonInternal.h"

#include "LevelwiseProcessingKernel.h"
namespace mgard_cuda {

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, int OP>
__global__ void _lwpk(SIZE *shape, T *dv, SIZE *ldvs, T *dwork, SIZE *ldws) {

  size_t threadId = (threadIdx.z * (blockDim.x * blockDim.y)) +
                    (threadIdx.y * blockDim.x) + threadIdx.x;
  SIZE *sm = SharedMemory<SIZE>();
  SIZE *shape_sm = sm;
  SIZE *ldvs_sm = shape_sm + D;
  SIZE *ldws_sm = ldvs_sm + D;

  if (threadId < D) {
    shape_sm[threadId] = shape[threadId];
    ldvs_sm[threadId] = ldvs[threadId];
    ldws_sm[threadId] = ldws[threadId];
  }
  __syncthreads();

  SIZE idx[D];
  SIZE firstD = div_roundup(shape_sm[0], F);

  SIZE bidx = blockIdx.x;
  idx[0] = (bidx % firstD) * F + threadIdx.x;

  // printf("firstD %d idx[0] %d\n", firstD, idx[0]);

  bidx /= firstD;
  if (D >= 2)
    idx[1] = blockIdx.y * blockDim.y + threadIdx.y;
  if (D >= 3)
    idx[2] = blockIdx.z * blockDim.z + threadIdx.z;

  for (DIM d = 3; d < D; d++) {
    idx[d] = bidx % shape_sm[d];
    bidx /= shape_sm[d];
  }
  // int z = blockIdx.z * blockDim.z + threadIdx.z;
  // int y = blockIdx.y * blockDim.y + threadIdx.y;
  // int x = blockIdx.z * blockDim.z + threadIdx.z;
  bool in_range = true;
  for (DIM d = 0; d < D; d++) {
    if (idx[d] >= shape_sm[d])
      in_range = false;
  }
  if (in_range) {
    // printf("%d %d %d %d\n", idx[3], idx[2], idx[1], idx[0]);
    if (OP == COPY)
      dwork[get_idx<D>(ldws, idx)] = dv[get_idx<D>(ldvs, idx)];
    if (OP == ADD)
      dwork[get_idx<D>(ldws, idx)] += dv[get_idx<D>(ldvs, idx)];
    if (OP == SUBTRACT)
      dwork[get_idx<D>(ldws, idx)] -= dv[get_idx<D>(ldvs, idx)];
  }
}

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, int OP>
void lwpk_adaptive_launcher(Handle<D, T> &handle, SIZE *shape_h, SIZE *shape_d,
                            T *dv, SIZE *ldvs, T *dwork, SIZE *ldws,
                            int queue_idx) {

  SIZE total_thread_z = shape_h[2];
  SIZE total_thread_y = shape_h[1];
  SIZE total_thread_x = shape_h[0];
  // linearize other dimensions
  SIZE tbz = R;
  SIZE tby = C;
  SIZE tbx = F;
  SIZE gridz = ceil((double)total_thread_z / tbz);
  SIZE gridy = ceil((double)total_thread_y / tby);
  SIZE gridx = ceil((double)total_thread_x / tbx);
  for (DIM d = 3; d < D; d++) {
    gridx *= shape_h[d];
  }

  // printf("exec: %d %d %d %d %d %d\n", tbx, tby, tbz, gridx, gridy, gridz);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);
  size_t sm_size = (D * 3) * sizeof(SIZE);
  _lwpk<D, T, R, C, F, OP><<<blockPerGrid, threadsPerBlock, sm_size,
                             *(cudaStream_t *)handle.get(queue_idx)>>>(
      shape_d, dv, ldvs, dwork, ldws);

  gpuErrchk(cudaGetLastError());
  if (handle.sync_and_check_all_kernels) {
    gpuErrchk(cudaDeviceSynchronize());
  }
}

template <DIM D, typename T, int OP>
void lwpk(Handle<D, T> &handle, SIZE *shape_h, SIZE *shape_d, T *dv, SIZE *ldvs,
          T *dwork, SIZE *ldws, int queue_idx) {
#define COPYLEVEL(R, C, F)                                                     \
  {                                                                            \
    lwpk_adaptive_launcher<D, T, R, C, F, OP>(handle, shape_h, shape_d, dv,    \
                                              ldvs, dwork, ldws, queue_idx);   \
  }
  if (D >= 3) {
    COPYLEVEL(4, 4, 4)
  }
  if (D == 2) {
    COPYLEVEL(1, 4, 4)
  }
  if (D == 1) {
    COPYLEVEL(1, 1, 8)
  }

#undef COPYLEVEL
}

} // namespace mgard_cuda

#endif