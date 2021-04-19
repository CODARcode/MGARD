/* 
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */


#ifndef MGRAD_CUDA_ITERATIVE_PROCESSING_KERNEL_TEMPLATE
#define MGRAD_CUDA_ITERATIVE_PROCESSING_KERNEL_TEMPLATE

#include "iterative_processing_kernel.h"
namespace mgard_cuda {

template <typename T, int D, int R, int C, int F, int G>
__global__ void _ipk_1( int * shape, int * shape_c,
                        int * ldvs, int * ldws,
                        int processed_n, int * processed_dims, 
                        int curr_dim_r, int curr_dim_c, int curr_dim_f,
                        T * am, T *bm,
                        T * dist_f, T *v, int ldv1, int ldv2) {

  bool debug = false;
  if (blockIdx.z == 0 && blockIdx.y == 0 && blockIdx.x == 0 && 
  threadIdx.z == 0 && threadIdx.y == 0 ) debug = false;

  bool debug2 = false;
  if (threadIdx.z == 0 && threadIdx.y == 0 && threadIdx.x == 0 ) debug2 = false;

  size_t threadId =  (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;


  
  T *sm = SharedMemory<T>();
  int ldsm1 = F + G;
  int ldsm2 = C;
  T *vec_sm = sm;
  T *bm_sm = sm + R * ldsm1 * ldsm2;
  T *dist_sm = bm_sm + ldsm1;
  int *shape_sm               = (int *)(dist_sm + ldsm1);
  int *shape_c_sm             = shape_sm + D;
  int *processed_dims_sm      = shape_c_sm + D;
  int *ldvs_sm                = processed_dims_sm + D;
  int *ldws_sm                = ldvs_sm + D;
  int idx[D];
  if (threadId < D) {
    shape_sm[threadId] = shape[threadId];
    shape_c_sm[threadId] = shape_c[threadId];
    ldvs_sm[threadId] = ldvs[threadId];
    ldws_sm[threadId] = ldws[threadId];
  }
  if (threadId < processed_n) {
    processed_dims_sm[threadId] = processed_dims[threadId];
  }
  __syncthreads();

  for (int d = 0; d < D; d ++) idx[d] = 0;

  int nr = shape_sm[curr_dim_r];
  int nc = shape_sm[curr_dim_c];
  int nf_c = shape_c_sm[curr_dim_f];

  if (D < 3) nr = 1;
  if (D < 2) nc = 1;

  int bidx = blockIdx.x;
  int firstD = div_roundup(nc, C);
  int blockId = bidx % firstD;
  // if (debug2 && blockIdx.x != blockId) printf("blockIdx.x %d nc %d blockDim.x %d firstD: %d blockId %d\n", blockIdx.x, nc, blockDim.x, firstD, blockId);
  bidx /= firstD;

  for (int d = 0; d < D; d++) {
    if (d != curr_dim_r && d != curr_dim_c && d != curr_dim_f) {
      int t = shape_sm[d];
      for (int k = 0; k < processed_n; k++) {
        if (d == processed_dims[k]) {
          t = shape_c_sm[d];
        }
      }
      idx[d] = bidx % t;
      bidx /= t;
    }
  }

  size_t other_offset_v = get_idx<D>(ldvs_sm, idx);
  v = v + other_offset_v;

  if (debug2) {
    printf("ipk1 idx: %d %d %d\n", idx[2], idx[1], idx[0]);
    printf("ipk1 other_offset_v: %llu\n", other_offset_v);
  }

  int c_gl = blockId * C;
  int r_gl = blockIdx.y * R;
  int f_gl = threadIdx.x;

  int c_sm = threadIdx.x;
  int r_sm = threadIdx.y;
  int f_sm = threadIdx.x;

  T *vec = v + get_idx(ldv1, ldv2, r_gl, c_gl, 0);



  register T prev_vec_sm = 0.0;

  int c_rest = min(C, nc - blockId * C);
  int r_rest = min(R, nr - blockIdx.y * R);

  int f_rest = nf_c;
  int f_ghost = min(nf_c, G);
  int f_main = F;

  // printf("r_sm: %d, r_rest: %d, c_sm: %d, c_rest: %d f_sm: %d, f_rest %d , nf_c: %d\n", r_sm, r_rest, c_sm, c_rest, f_sm, f_rest, nf_c);

  // printf("test %f", vec_sm[get_idx(ldsm1, ldsm2, 0, 1, 0)]);
  /* Load first ghost */
  if (r_sm < r_rest && f_sm < f_ghost) {
    for (int i = 0; i < c_rest; i++) {
      vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] = vec[get_idx(ldv1, ldv2, r_sm, i, f_gl)];
      // if (r_sm == 0) printf("r0_stride = %d, vec_sm[%d] = %f\n", r0_stride,
      // i, vec_sm[i * ldsm + c_sm]);
    }    
    if (r_sm == 0) bm_sm[f_sm] = bm[f_gl];
  }
  
  f_rest -= f_ghost;
  __syncthreads();

  while (f_rest > F - f_ghost) {
    // if (c_gl == 0 && c_sm == 0 && r_gl == 0 && r_sm == 0) printf("%d %d\n", f_rest, F - f_ghost);
    f_main = min(F, f_rest);
    if (r_sm < r_rest && f_sm < f_main) {
      for (int i = 0; i < c_rest; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm + f_ghost)] =
            vec[get_idx(ldv1, ldv2, r_sm, i, f_gl + f_ghost)];
      }
      if (r_sm == 0) bm_sm[f_sm + f_ghost] = bm[f_gl + f_ghost];
    }
    
    __syncthreads();

    
    /* Computation of v in parallel*/
    if (r_sm < r_rest && c_sm < c_rest) {
      // if (r_gl == 0 && c_gl == 0 && r_sm == 0 && c_sm == 0) printf("%f + %f * %f -> %f\n", 
      //                                               vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)],
      //                                               prev_vec_sm, bm_sm[0], 
      //                                               vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]+prev_vec_sm * bm_sm[0]);
      
// #ifdef MGARD_CUDA_FMA
//       vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] = 
//       __fma_rn(prev_vec_sm, bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);
// #else
//       vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] -= prev_vec_sm * bm_sm[0];
// #endif
      vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] = 
        tridiag_forward(prev_vec_sm, bm_sm[0], 
          vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);
      
      //#pragma unroll 32
      for (int i = 1; i < F; i++) {

        // if (c_gl == 0 && c_sm == 1 && blockIdx.x == 0 && blockIdx.y == 0 && r_sm == 0) {
        //   printf("%f + %f * %f -> %f(%d %d %d)\n", 
        //           vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)],
        //           vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)], bm_sm[i], 
        //           vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]-
        //           vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)] * bm_sm[i],
        //           r_sm, c_sm, i);
        // }
// #ifdef MGARD_CUDA_FMA
//         vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)] =
//         __fma_rn(vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)], 
//           bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]);
// #else
//         vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)] -= 
//         vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)] * bm_sm[i];
// #endif
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)] = 
          tridiag_forward(vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)], 
            bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]);
        // printf("calc[%d]: %f\n", i, vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]);
        // if (r_gl == 0 && c_gl == 0) 
        //   printf("out[%d %d %d] %f\n", r_sm, c_sm, i, vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]);
        
      }

      /* Store last v */
      prev_vec_sm = vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, F - 1)];
    }
    __syncthreads();

    /* flush results to v */
    if (r_sm < r_rest && f_sm < F) {
      for (int i = 0; i < c_rest; i++) {
        // if (blockIdx.x == 0 && blockIdx.y == 0 && r_sm == 0 && i == 1) {
        //   printf("store [%d %d %d] %f<-%f [%d %d %d]\n", 
        //     r_sm, i, f_gl, vec[get_idx(ldv1, ldv2, r_sm, i, f_gl)], vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)], r_sm, i, f_sm);
        // }
        vec[get_idx(ldv1, ldv2, r_sm, i, f_gl)] = 
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)];
        // if (blockIdx.x == 0 && blockIdx.y == 0 && r_sm == 0 && i == 1) {
        //   printf("store [%d %d %d] %f<-%f [%d %d %d]\n", 
        //     r_sm, i, f_gl, vec[get_idx(ldv1, ldv2, r_sm, i, f_gl)], vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)], r_sm, i, f_sm);
        // }
      }
    }
    __syncthreads();

    /* Update unloaded col */
    f_rest -= f_main;

    /* Advance c */
    f_gl += F;

    /* Copy next ghost to main */
    f_ghost = min(G, f_main - (F - G));
    if (r_sm < r_rest && f_sm < f_ghost) {
      for (int i = 0; i < c_rest; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] = 
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm + F)];
      }
      if (r_sm == 0) bm_sm[f_sm] = bm_sm[f_sm + blockDim.x];
    }
    __syncthreads();
  } //end of while

  /* Load all rest col */
  if (r_sm < r_rest && f_sm < f_rest) {
    for (int i = 0; i < c_rest; i++) {
      vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm + f_ghost)] =
          vec[get_idx(ldv1, ldv2, r_sm, i, f_gl + f_ghost)];
    }
    if (r_sm == 0) bm_sm[f_sm + f_ghost] = bm[f_gl + f_ghost];
  }
  
  __syncthreads();

  /* Only 1 col remain */
  if (f_ghost + f_rest == 1) {
    if (r_sm < r_rest && c_sm < c_rest) {
// #ifdef MGARD_CUDA_FMA
//       vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] = 
//       __fma_rn(prev_vec_sm, bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);
// #else
//       vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] -= prev_vec_sm * bm_sm[0];
// #endif
      vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] = 
        tridiag_forward(prev_vec_sm, bm_sm[0], 
          vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);
      // printf ("prev_vec_sm = %f\n", prev_vec_sm );
      // printf ("vec_sm[r_sm * ldsm + 0] = %f\n", vec_sm[r_sm * ldsm + 0] );
    }
    //__syncthreads();

  } else {
    if (r_sm < r_rest && c_sm < c_rest) {
// #ifdef MGARD_CUDA_FMA
//       vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] = 
//       __fma_rn(prev_vec_sm, bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);
// #else
//       vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] -= prev_vec_sm * bm_sm[0];
// #endif
      vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] = 
        tridiag_forward(prev_vec_sm, bm_sm[0], 
          vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);
      for (int i = 1; i < f_ghost + f_rest; i++) {
// #ifdef MGARD_CUDA_FMA 
//         vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)] =
//         __fma_rn(vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)], bm_sm[i], 
//           vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]);
// #else
//         vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)] -= 
//         vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)] * bm_sm[i];
// #endif
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)] = 
          tridiag_forward(vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)], 
            bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]);
      }
    }
  }
  __syncthreads();
  /* flush results to v */
  if (r_sm < r_rest && f_sm < f_ghost + f_rest) {
    for (int i = 0; i < c_rest; i++) {
      vec[get_idx(ldv1, ldv2, r_sm, i, f_gl)] = vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)];
      // printf("c_stride = %d, c_sm = %d, vec_sm = %f, vec[%d] =
      // %f\n",c_stride, c_sm, vec_sm[r_sm * ldsm + 0],i * row_stride * lddv +
      // c_stride, vec[i * row_stride * lddv + c_stride]);
    }
  }
  __syncthreads();

  /* backward */
  T * am_sm = bm_sm;
  f_rest = nf_c;
  f_ghost = min(nf_c, G);
  f_main = F;
  f_gl = threadIdx.x;
  prev_vec_sm = 0.0;

  /* Load first ghost */
  if (r_sm < r_rest && f_sm < f_ghost) {
    for (int i = 0; i < c_rest; i++) {
      vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] = vec[get_idx(ldv1, ldv2, r_sm, i, (nf_c - 1) - f_gl)];
      // if (r_sm == 0) printf("r0_stride = %d, vec_sm[%d] = %f\n", r0_stride,
      // i, vec_sm[i * ldsm + c_sm]);
    }
  }
  if (r_sm == 0){
    am_sm[f_sm] = am[(nf_c - 1) - f_gl];
    dist_sm[f_sm] = dist_f[(nf_c - 1) - f_gl];// * -1;
  }
  f_rest -= f_ghost;
  __syncthreads();

  while (f_rest > F - f_ghost) {
    f_main = min(F, f_rest);
    if (r_sm < r_rest && f_sm < f_main) {
      for (int i = 0; i < c_rest; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm + f_ghost)] =
            vec[get_idx(ldv1, ldv2, r_sm, i, (nf_c - 1) - f_gl - f_ghost)];
      }
    }
    if (r_sm == 0) {
      am_sm[f_sm + f_ghost] = am[(nf_c - 1) - f_gl - f_ghost];
      dist_sm[f_sm + f_ghost] = dist_f[(nf_c - 1) - f_gl - f_ghost];// * -1;
    }
    __syncthreads();

    /* Computation of v in parallel*/
    if (r_sm < r_rest && c_sm < c_rest) {
      // vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] = 
      // (vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] - dist_sm[0] * prev_vec_sm) / am_sm[0];
      // if (r_gl == 0 && c_gl == 0 && r_sm == 0 && c_sm == 0) 
      //     printf("(%f + %f * %f) * %f -> %f\n", 
      //             vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)],
      //             dist_sm[0], prev_vec_sm, am_sm[0],
      //             (vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] - dist_sm[0] * prev_vec_sm) / am_sm[0]);

// #ifdef MGARD_CUDA_FMA
//       vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] = 
//       __fma_rn(dist_sm[0], prev_vec_sm, 
//         vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]) * am_sm[0];
// #else
//       vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] = 
//       (vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] - dist_sm[0] * prev_vec_sm) / am_sm[0];
// #endif
      vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] = 
        tridiag_backward(prev_vec_sm, dist_sm[0], am_sm[0], 
          vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);
      //#pragma unroll 32
      for (int i = 1; i < F; i++) {
        // if (r_gl == 0 && c_gl == 0 && r_sm == 0 && c_sm == 0) 
        //   printf("(%f + %f * %f) * %f -> %f\n", 
        //           vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)],
        //           dist_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i-1)],
        //           (vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)] - dist_sm[i] * vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i-1)]) * am_sm[i]);

// #ifdef MGARD_CUDA_FMA
//         vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)] = 
//       __fma_rn(dist_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)], 
//         vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]) * am_sm[i];
// #else
//       vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)] = 
//         (vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)] - 
//           dist_sm[i] * vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)]) / am_sm[i];
// #endif 
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)] = 
        tridiag_backward(vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)],
         dist_sm[i], am_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]);
      }
      /* Store last v */
      prev_vec_sm = vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, blockDim.x - 1)];
    }
    __syncthreads();

    /* flush results to v */
    if (r_sm < r_rest && f_sm < F) {
      for (int i = 0; i < c_rest; i++) {
        vec[get_idx(ldv1, ldv2, r_sm, i, (nf_c - 1) - f_gl)] = 
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)];
      }
    }
    __syncthreads();

    /* Update unloaded col */
    f_rest -= f_main;

    /* Advance c */
    f_gl += F;

    /* Copy next ghost to main */
    f_ghost = min(G, f_main - (F - G));
    if (r_sm < r_rest && f_sm < f_ghost) {
      for (int i = 0; i < c_rest; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] = 
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm + F)];
      }
      if (r_sm == 0) {
        am_sm[f_sm] = am_sm[f_sm + F];
        dist_sm[f_sm] = dist_sm[f_sm + F];
      }
    }
    __syncthreads();
  } //end of while

  /* Load all rest col */
  if (r_sm < r_rest && f_sm < f_rest) {
    for (int i = 0; i < c_rest; i++) {
      vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm + f_ghost)] =
          vec[get_idx(ldv1, ldv2, r_sm, i, (nf_c - 1) - f_gl - f_ghost)];
    }
  }
  if (r_sm == 0){
    am_sm[f_sm + f_ghost] = am[(nf_c - 1) - f_gl - f_ghost];
    dist_sm[f_sm + f_ghost] = dist_f[(nf_c - 1) - f_gl - f_ghost];
  }
  __syncthreads();

  /* Only 1 col remain */
  if (f_ghost + f_rest == 1) {
    if (r_sm < r_rest && c_sm < c_rest) {
// #ifdef MGARD_CUDA_FMA
//       vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] = 
//       __fma_rn(dist_sm[0], prev_vec_sm, 
//         vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]) * am_sm[0];
// #else
//       vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] = 
//       (vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] - dist_sm[0] * prev_vec_sm) / am_sm[0];
// #endif
      vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] = 
        tridiag_backward(prev_vec_sm, dist_sm[0], am_sm[0], 
          vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);
      // printf ("prev_vec_sm = %f\n", prev_vec_sm );
      // printf ("vec_sm[r_sm * ldsm + 0] = %f\n", vec_sm[r_sm * ldsm + 0] );
    }
    //__syncthreads();

  } else {
    if (r_sm < r_rest && c_sm < c_rest) {
      
// #ifdef MGARD_CUDA_FMA
//       vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] = 
//       __fma_rn(dist_sm[0], prev_vec_sm, 
//         vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]) * am_sm[0];
// #else
//       vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] = 
//       (vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] - dist_sm[0] * prev_vec_sm) / am_sm[0];
// #endif
      vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] = 
        tridiag_backward(prev_vec_sm, dist_sm[0], am_sm[0], 
          vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);
      for (int i = 1; i < f_ghost + f_rest; i++) {
// #ifdef MGARD_CUDA_FMA
//       vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)] = 
//       __fma_rn(dist_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)], 
//         vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]) * am_sm[i];
// #else   
//       vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)] = 
//         (vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)] - 
//           vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)] * dist_sm[i]) / am_sm[i];
// #endif   
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)] = 
        tridiag_backward(vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)],
         dist_sm[i], am_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]);
      }
    }
  }
  __syncthreads();
  /* flush results to v */
  if (r_sm < r_rest && f_sm < f_ghost + f_rest) {
    for (int i = 0; i < c_rest; i++) {
      vec[get_idx(ldv1, ldv2, r_sm, i, (nf_c - 1) - f_gl)] = vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)];
      // printf("c_stride = %d, c_sm = %d, vec_sm = %f, vec[%d] =
      // %f\n",c_stride, c_sm, vec_sm[r_sm * ldsm + 0],i * row_stride * lddv +
      // c_stride, vec[i * row_stride * lddv + c_stride]);
    }
  }
  __syncthreads();
}


template <typename T, int D, int R, int C, int F, int G>
void ipk_1_adaptive_launcher( mgard_cuda_handle<T, D> &handle, 
                              thrust::device_vector<int> shape, 
                              thrust::device_vector<int> shape_c, 
                              thrust::device_vector<int> ldvs, 
                              thrust::device_vector<int> ldws,
                              thrust::device_vector<int> processed_dims,
                              int curr_dim_r, int curr_dim_c, int curr_dim_f,
                              T* am, T *bm, T * ddist_f, T *dv,
                              int lddv1, int lddv2, int queue_idx) {

  int nr = shape[curr_dim_r];
  int nc = shape[curr_dim_c];
  int nf_c = shape_c[curr_dim_f];

  int total_thread_x = nc;
  int total_thread_y = nr;
  int total_thread_z = 1;
  int tbx, tby, tbz, gridx, gridy, gridz;
  dim3 threadsPerBlock, blockPerGrid;
  size_t sm_size;

  tbx = std::max(C, std::min(C, total_thread_x)); 
  tby = std::max(R, std::min(R, total_thread_y)); 
  tbz = 1;
  sm_size = (R * C + 2) * (F + G) * sizeof(T); 
  sm_size += (D * 5) * sizeof(int); 
  gridx = ceil((float)total_thread_x / tbx); 
  gridy = ceil((float)total_thread_y / tby); 
  gridz = 1; 
  // printf("ipk 1 total_thread_x %d tbx %d\n", total_thread_x, tbx);
  
  for (int d = 0; d < D; d++) {
    if (d != curr_dim_f && d != curr_dim_c && d != curr_dim_r) {
      int t = shape[d];
      for (int k = 0; k < processed_dims.size(); k++) {
        if (d == processed_dims[k]) {
          t = shape_c[d];
        }
      }
      gridx *= t;
    }
  }
  // printf("ipk_1 exec config (%d %d %d) (%d %d %d)\n", F, tby, tbz, gridx, gridy, gridz);
  threadsPerBlock = dim3(F, tby, tbz); 
  blockPerGrid = dim3(gridx, gridy, gridz); 

  // printf("ipk_1 exec config (%d %d %d) (%d %d %d)\n", tbx, tby, tbz, gridx, gridy, gridz);
  _ipk_1<T, D, R, C, F, G><<<blockPerGrid, threadsPerBlock, sm_size, 
                                 *(cudaStream_t *)handle.get(queue_idx)>>>( 
                                                thrust::raw_pointer_cast(shape.data()), 
                                                thrust::raw_pointer_cast(shape_c.data()), 
                                                thrust::raw_pointer_cast(ldvs.data()), 
                                                thrust::raw_pointer_cast(ldws.data()),
                                                processed_dims.size(), 
                                                thrust::raw_pointer_cast(processed_dims.data()), 
                                                curr_dim_r, curr_dim_c, curr_dim_f, 
                                                am, bm, ddist_f, dv, lddv1, lddv2); 
  gpuErrchk(cudaGetLastError()); 
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
  // std::cout << "test\n";
}


template <typename T, int D>
void ipk_1( mgard_cuda_handle<T, D> &handle, 
            thrust::device_vector<int> shape, 
            thrust::device_vector<int> shape_c, 
            thrust::device_vector<int> ldvs, 
            thrust::device_vector<int> ldws,
            thrust::device_vector<int> processed_dims,
            int curr_dim_r, int curr_dim_c, int curr_dim_f,
            T* am, T *bm, T * ddist_f, T *dv,
            int lddv1, int lddv2, int queue_idx, int config) {

#define IPK(R, C, F, G) { ipk_1_adaptive_launcher<T, D, R, C, F, G>(\
                                  handle,\
                                  shape, shape_c, ldvs, ldws, processed_dims,\
                                  curr_dim_r, curr_dim_c, curr_dim_f,\
                                  am, bm, ddist_f, dv,\
                                  lddv1, lddv2, queue_idx);}
  bool profile = false;
#ifdef MGARD_CUDA_KERNEL_PROFILE 
  profile = true;
#endif
  if (D >= 3) {
    if (profile || config == 6) { IPK(2, 2, 128, 2) }
    if (profile || config == 5) { IPK(2, 2, 64, 2) }
    if (profile || config == 4) { IPK(2, 2, 32, 2) }
    if (profile || config == 3) { IPK(4, 4, 16, 4) }
    if (profile || config == 2) { IPK(8, 8, 8, 4) }
    if (profile || config == 1) { IPK(4, 4, 4, 4) }
    if (profile || config == 0) { IPK(2, 2, 2, 2) }
  } else if (D == 2){
    if (profile || config == 6) { IPK(1, 2, 128, 2) }
    if (profile || config == 5) { IPK(1, 2, 64, 2) }
    if (profile || config == 4) { IPK(1, 2, 32, 2) }
    if (profile || config == 3) { IPK(1, 4, 16, 4) }
    if (profile || config == 2) { IPK(1, 8, 8, 4) }
    if (profile || config == 1) { IPK(1, 4, 4, 4) }
    if (profile || config == 0) { IPK(1, 2, 4, 2) }
  } else if (D == 1) {
    if (profile || config == 6) { IPK(1, 1, 128, 2) }
    if (profile || config == 5) { IPK(1, 1, 64, 2) }
    if (profile || config == 4) { IPK(1, 1, 32, 2) }
    if (profile || config == 3) { IPK(1, 1, 16, 4) }
    if (profile || config == 2) { IPK(1, 1, 8, 4) }
    if (profile || config == 1) { IPK(1, 1, 8, 4) }
    if (profile || config == 0) { IPK(1, 1, 8, 2) }
  }
#undef IPK
}

template <typename T, int D, int R, int C, int F, int G>
void ipk_1_adaptive_launcher( mgard_cuda_handle<T, D> &handle, 
                              int * shape_h, int * shape_c_h, int * shape_d, int * shape_c_d, 
                              int * ldvs, int * ldws,
                              int processed_n, int * processed_dims_h, int * processed_dims_d,
                              int curr_dim_r, int curr_dim_c, int curr_dim_f,
                              T* am, T *bm, T * ddist_f, T *dv,
                              int lddv1, int lddv2, int queue_idx) {

  int nr = shape_h[curr_dim_r];
  int nc = shape_h[curr_dim_c];
  int nf_c = shape_c_h[curr_dim_f];

  int total_thread_x = nc;
  int total_thread_y = nr;
  int total_thread_z = 1;
  int tbx, tby, tbz, gridx, gridy, gridz;
  dim3 threadsPerBlock, blockPerGrid;
  size_t sm_size;

  tbx = std::max(C, std::min(C, total_thread_x)); 
  tby = std::max(R, std::min(R, total_thread_y)); 
  tbz = 1;
  sm_size = (R * C + 2) * (F + G) * sizeof(T); 
  sm_size += (D * 5) * sizeof(int); 
  gridx = ceil((float)total_thread_x / tbx); 
  gridy = ceil((float)total_thread_y / tby); 
  gridz = 1; 
  // printf("ipk 1 total_thread_x %d tbx %d\n", total_thread_x, tbx);
  
  for (int d = 0; d < D; d++) {
    if (d != curr_dim_f && d != curr_dim_c && d != curr_dim_r) {
      int t = shape_h[d];
      for (int k = 0; k < processed_n; k++) {
        if (d == processed_dims_h[k]) {
          t = shape_c_h[d];
        }
      }
      gridx *= t;
    }
  }
  // printf("ipk_1 exec config (%d %d %d) (%d %d %d)\n", F, tby, tbz, gridx, gridy, gridz);
  threadsPerBlock = dim3(F, tby, tbz); 
  blockPerGrid = dim3(gridx, gridy, gridz); 

  // printf("ipk_1 exec config (%d %d %d) (%d %d %d)\n", tbx, tby, tbz, gridx, gridy, gridz);
  _ipk_1<T, D, R, C, F, G><<<blockPerGrid, threadsPerBlock, sm_size, 
                                 *(cudaStream_t *)handle.get(queue_idx)>>>( 
                                                shape_d, shape_c_d, ldvs, ldws,
                                                processed_n, processed_dims_d,  
                                                curr_dim_r, curr_dim_c, curr_dim_f, 
                                                am, bm, ddist_f, dv, lddv1, lddv2); 
  gpuErrchk(cudaGetLastError()); 
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
  // std::cout << "test\n";
}


template <typename T, int D>
void ipk_1( mgard_cuda_handle<T, D> &handle, 
            int * shape_h, int * shape_c_h, int * shape_d, int * shape_c_d, 
            int * ldvs, int * ldws,
            int processed_n, int * processed_dims_h, int * processed_dims_d,
            int curr_dim_r, int curr_dim_c, int curr_dim_f,
            T* am, T *bm, T * ddist_f, T *dv,
            int lddv1, int lddv2, int queue_idx, int config) {

#define IPK(R, C, F, G) { ipk_1_adaptive_launcher<T, D, R, C, F, G>(\
                                  handle,\
                                  shape_h, shape_c_h, shape_d, shape_c_d, ldvs, ldws,\
                                  processed_n, processed_dims_h, processed_dims_d,\ 
                                  curr_dim_r, curr_dim_c, curr_dim_f,\
                                  am, bm, ddist_f, dv,\
                                  lddv1, lddv2, queue_idx);}
  bool profile = false;
#ifdef MGARD_CUDA_KERNEL_PROFILE 
  profile = true;
#endif
  if (D >= 3) {
    if (profile || config == 6) { IPK(2, 2, 128, 2) }
    if (profile || config == 5) { IPK(2, 2, 64, 2) }
    if (profile || config == 4) { IPK(2, 2, 32, 2) }
    if (profile || config == 3) { IPK(4, 4, 16, 4) }
    if (profile || config == 2) { IPK(8, 8, 8, 4) }
    if (profile || config == 1) { IPK(4, 4, 4, 4) }
    if (profile || config == 0) { IPK(2, 2, 2, 2) }
  } else if (D == 2){
    if (profile || config == 6) { IPK(1, 2, 128, 2) }
    if (profile || config == 5) { IPK(1, 2, 64, 2) }
    if (profile || config == 4) { IPK(1, 2, 32, 2) }
    if (profile || config == 3) { IPK(1, 4, 16, 4) }
    if (profile || config == 2) { IPK(1, 8, 8, 4) }
    if (profile || config == 1) { IPK(1, 4, 4, 4) }
    if (profile || config == 0) { IPK(1, 2, 4, 2) }
  } else if (D == 1) {
    if (profile || config == 6) { IPK(1, 1, 128, 2) }
    if (profile || config == 5) { IPK(1, 1, 64, 2) }
    if (profile || config == 4) { IPK(1, 1, 32, 2) }
    if (profile || config == 3) { IPK(1, 1, 16, 4) }
    if (profile || config == 2) { IPK(1, 1, 8, 4) }
    if (profile || config == 1) { IPK(1, 1, 8, 4) }
    if (profile || config == 0) { IPK(1, 1, 8, 2) }
  }
#undef IPK
}

template <typename T, int D, int R, int C, int F, int G>
__global__ void _ipk_2(int * shape, int * shape_c,
                        int * ldvs, int * ldws,
                        int processed_n, int * processed_dims, 
                        int curr_dim_r, int curr_dim_c, int curr_dim_f,
                        T * am, T *bm, T * dist_c, T *v, int ldv1, int ldv2) {

  bool debug = false;
  if (blockIdx.z == 0 && blockIdx.y == 0 && blockIdx.x == 0 && 
  threadIdx.z == 0 && threadIdx.y == 0 ) debug = false;

  bool debug2 = false;
  if (threadIdx.z == 0 && threadIdx.y == 0 && threadIdx.x == 0 ) debug2 = false;

  size_t threadId =  (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;


  T *sm = SharedMemory<T>();
  int ldsm1 = F;
  int ldsm2 = C + G;
  T *vec_sm = sm;
  T *bm_sm = sm + R * ldsm1 * ldsm2;
  T *dist_sm = bm_sm + ldsm2;
  int *shape_sm               = (int *)(dist_sm + ldsm2);
  int *shape_c_sm             = shape_sm + D;
  int *processed_dims_sm      = shape_c_sm + D;
  int *ldvs_sm                = processed_dims_sm + D;
  int *ldws_sm                = ldvs_sm + D;
  int idx[D];
  if (threadId < D) {
    shape_sm[threadId] = shape[threadId];
    shape_c_sm[threadId] = shape_c[threadId];
    ldvs_sm[threadId] = ldvs[threadId];
    ldws_sm[threadId] = ldws[threadId];
  }
  if (threadId < processed_n) {
    processed_dims_sm[threadId] = processed_dims[threadId];
  }
  __syncthreads();

  for (int d = 0; d < D; d ++) idx[d] = 0;

  int nr = shape_sm[curr_dim_r];
  int nc_c = shape_c_sm[curr_dim_c];
  int nf_c = shape_c_sm[curr_dim_f];

  if (D < 3) nr = 1;

  int bidx = blockIdx.x;
  int firstD = div_roundup(nf_c, blockDim.x);
  int blockId = bidx % firstD;

  bidx /= firstD;

  for (int d = 0; d < D; d++) {
    if (d != curr_dim_r && d != curr_dim_c && d != curr_dim_f) {
      int t = shape_sm[d];
      for (int k = 0; k < processed_n; k++) {
        if (d == processed_dims[k]) {
          t = shape_c_sm[d];
        }
      }
      idx[d] = bidx % t;
      bidx /= t;
    }
  }

  size_t other_offset_v = get_idx<D>(ldvs_sm, idx);
  v = v + other_offset_v;


  int f_gl = blockId * F;
  int r_gl = blockIdx.y * R;
  int c_gl = 0;

  int f_sm = threadIdx.x;
  int r_sm = threadIdx.y;
  int c_sm = threadIdx.x;

  T *vec = v + get_idx(ldv1, ldv2, r_gl, 0, f_gl);

  register T prev_vec_sm = 0.0;

  int f_rest = min(F, nf_c - blockId * F);
  int r_rest = min(R, nr - blockIdx.y * R);

  // if (blockIdx.x == 1 && blockIdx.y == 0 && f_sm == 0 && r_sm == 0) {
  //   printf("f_rest: %d r_rest: %d\n", f_rest, r_rest);
  // }

  int c_rest = nc_c;
  int c_ghost = min(nc_c, G);
  int c_main = C;

  /* Load first ghost */
  if (r_sm < r_rest && f_sm < f_rest) {
    for (int i = 0; i < c_ghost; i++) {
      vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] = 
        vec[get_idx(ldv1, ldv2, r_sm, c_gl + i, f_sm)];
      // if (r_sm == 0) printf("r0_stride = %d, vec_sm[%d] = %f\n", r0_stride,
      // i, vec_sm[i * ldsm + c_sm]);
    }
  }
  if (r_sm == 0 && c_sm < c_ghost) bm_sm[c_sm] = bm[c_gl + c_sm];
  c_rest -= c_ghost;
  __syncthreads();

  while (c_rest > C - c_ghost) {
    // printf("%d %d %d\n", c_rest, C, c_ghost);
    c_main = min(C, c_rest);
    if (r_sm < r_rest && f_sm < f_rest) {
      for (int i = 0; i < c_main; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i + c_ghost, f_sm)] =
            vec[get_idx(ldv1, ldv2, r_sm, c_gl + i + c_ghost, f_sm)];
      }
    }
    if (r_sm == 0 && c_sm < c_main) bm_sm[c_sm + c_ghost] = bm[c_gl + c_sm + c_ghost];
    __syncthreads();

    /* Computation of v in parallel*/
    if (r_sm < r_rest && f_sm < f_rest) {
      
// #ifdef MGARD_CUDA_FMA
//       vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] = 
//       __fma_rn(prev_vec_sm, bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)]);
// #else
//       vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] -= prev_vec_sm * bm_sm[0];
// #endif
      vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] = 
        tridiag_forward(prev_vec_sm, bm_sm[0], 
          vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)]);

      for (int i = 1; i < C; i++) {
// #ifdef MGARD_CUDA_FMA
//         vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] = 
//       __fma_rn(vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)], bm_sm[i], 
//         vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);
// #else
//       // if (blockIdx.x == 1 && blockIdx.y == 0 && f_sm == 0 && r_sm == 0) {
//       //   printf("calc: %f %f %f -> %f \n", vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)],
//       //     vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)], bm_sm[i],  vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] -
//       //    vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)] * bm_sm[i]);
//       // }

//       vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] -=
//          vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)] * bm_sm[i];
// #endif
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] = 
          tridiag_forward(vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)], 
            bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);
      }
      /* Store last v */
      prev_vec_sm = vec_sm[get_idx(ldsm1, ldsm2, r_sm, C - 1, f_sm)];
    }
    __syncthreads();

    /* flush results to v */
    if (r_sm < r_rest && f_sm < f_rest) {
      for (int i = 0; i < C; i++) {
        // if (blockIdx.x == 1 && blockIdx.y == 0 && f_sm == 0 && r_sm == 0) {
        //   printf("store: %f\n", vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);
        // }
        vec[get_idx(ldv1, ldv2, r_sm, c_gl + i, f_sm)] = 
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)];
      }
    }
    __syncthreads();

    /* Update unloaded col */
    c_rest -= c_main;

    /* Advance c */
    c_gl += C;

    /* Copy next ghost to main */
    c_ghost = min(G, c_main - (C - G));
    if (r_sm < r_rest && f_sm < f_rest) {
      for (int i = 0; i < c_ghost; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] = 
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i + C, f_sm)];
      }
    }
    if (r_sm == 0 && c_sm < c_ghost) bm_sm[c_sm] = bm_sm[c_sm + C];
    __syncthreads();

  } // end of while

  /* Load all rest col */
  if (r_sm < r_rest && f_sm < f_rest) {
    for (int i = 0; i < c_rest; i++) {
      vec_sm[get_idx(ldsm1, ldsm2, r_sm, i + c_ghost, f_sm)] =
          vec[get_idx(ldv1, ldv2, r_sm, c_gl + i + c_ghost, f_sm)];
    }
  }
  if (r_sm == 0 && c_sm < c_rest) bm_sm[c_sm + c_ghost] = bm[c_gl + c_sm + c_ghost];
  __syncthreads();

  /* Only 1 col remain */
  if (c_ghost + c_rest == 1) {
    if (r_sm < r_rest && f_sm < f_rest) {
      // vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] -= prev_vec_sm * bm_sm[0];
// #ifdef MGARD_CUDA_FMA
//       vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] = 
//       __fma_rn(prev_vec_sm, bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)]);
// #else
//       vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] -= prev_vec_sm * bm_sm[0];
// #endif
      vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] = 
        tridiag_forward(prev_vec_sm, bm_sm[0], 
          vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)]);
      // printf ("prev_vec_sm = %f\n", prev_vec_sm );
      // printf ("vec_sm[r_sm * ldsm + 0] = %f\n", vec_sm[r_sm * ldsm + 0] );
    }
    //__syncthreads();

  } else {
    if (r_sm < r_rest && f_sm < f_rest) {
// #ifdef MGARD_CUDA_FMA
//       vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] = 
//       __fma_rn(prev_vec_sm, bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)]);
// #else
//       vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] -= prev_vec_sm * bm_sm[0];
// #endif
      vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] = 
        tridiag_forward(prev_vec_sm, bm_sm[0], 
          vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)]);
      for (int i = 1; i < c_ghost + c_rest; i++) {
// #ifdef MGARD_CUDA_FMA
//         vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] = 
//       __fma_rn(vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)], bm_sm[i], 
//         vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);
// #else
//       vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] -= 
//         vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)] * bm_sm[i];
// #endif
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] = 
          tridiag_forward(vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)], 
            bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);
      }
    }
  }
  __syncthreads();
  /* flush results to v */
  if (r_sm < r_rest && f_sm < f_rest) {
    for (int i = 0; i < c_ghost + c_rest; i++) {
      vec[get_idx(ldv1, ldv2, r_sm, c_gl + i, f_sm)] = 
      vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)];
      // printf("c_stride = %d, c_sm = %d, vec_sm = %f, vec[%d] =
      // %f\n",c_stride, c_sm, vec_sm[r_sm * ldsm + 0],i * row_stride * lddv +
      // c_stride, vec[i * row_stride * lddv + c_stride]);
    }
  }
  __syncthreads();

  /* backward */
  T * am_sm = bm_sm;
  c_rest = nc_c;
  c_ghost = min(nc_c, G);
  c_main = C;
  c_gl = 0;
  prev_vec_sm = 0.0;

  if (f_gl+f_sm == 0 && r_gl+r_sm == 0 && idx[3] == 0) debug = false;
  if (debug) printf("block id: (%d %d %d) thread id: (%d %d %d)\n", 
          blockIdx.x, blockIdx.y, blockIdx.z,
          threadIdx.x, threadIdx.y, threadIdx.z);

  /* Load first ghost */
  if (r_sm < r_rest && f_sm < f_rest) {
    for (int i = 0; i < c_ghost; i++) {
      vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] = 
      vec[get_idx(ldv1, ldv2, r_sm, (nc_c - 1) - (c_gl + i), f_sm)];
      if (debug) printf("load vec_sm[%d] = %f\n", get_idx(ldsm1, ldsm2, r_sm, i, f_sm), 
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);
    }
  }
  if (r_sm == 0 && c_sm < c_ghost){
    am_sm[c_sm] = am[(nc_c - 1) - (c_gl + c_sm)];
    dist_sm[c_sm] = dist_c[(nc_c - 1) - (c_gl + c_sm) ];
  }
  c_rest -= c_ghost;
  __syncthreads();

  while (c_rest > C - c_ghost) {
    // printf("%d %d %d\n", c_rest, C, c_ghost);
    c_main = min(C, c_rest);
    if (r_sm < r_rest && f_sm < f_rest) {
      for (int i = 0; i < c_main; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i + c_ghost, f_sm)] =
            vec[get_idx(ldv1, ldv2, r_sm, (nc_c - 1) - (c_gl + i + c_ghost), f_sm)];
        if (debug) printf("load vec_sm[%d] = %f\n", get_idx(ldsm1, ldsm2, r_sm, i + c_ghost, f_sm), 
                  vec_sm[get_idx(ldsm1, ldsm2, r_sm, i + c_ghost, f_sm)]);

      }
    }
    if (r_sm == 0 && c_sm < c_main) {
      am_sm[c_sm + c_ghost] = am[(nc_c - 1) - (c_gl + c_sm + c_ghost)];
      dist_sm[c_sm + c_ghost] = dist_c[(nc_c - 1) - (c_gl + c_sm + c_ghost)];
    }
    __syncthreads();

    // if (r_gl == 0 && f_gl == 0 && r_sm == 0 && f_sm == 0) printf("*****test\n");
    /* Computation of v in parallel*/
    if (r_sm < r_rest && f_sm < f_rest) {
// #ifdef MGARD_CUDA_FMA
//       vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] =
//       __fma_rn(dist_sm[0], prev_vec_sm, vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)]) *
//       am_sm[0];
// #else
//       // if (r_gl == 0 && f_gl == 0 && r_sm == 0 && f_sm == 0) 
//       //     printf("(%f + %f * %f) * %f -> %f\n", 
//       //             vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)],
//       //             dist_sm[0], prev_vec_sm, am_sm[0],
//       //             (vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] - dist_sm[0] * prev_vec_sm) / am_sm[0]);
//       vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] = 
//       (vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] - dist_sm[0] * prev_vec_sm) / am_sm[0];
// #endif
      vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, c_sm)] = 
        tridiag_backward(prev_vec_sm, dist_sm[0], am_sm[0], 
          vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, c_sm)]);
      if (debug) printf("calc vec_sm[%d] = %f\n", get_idx(ldsm1, ldsm2, r_sm, 0, f_sm), 
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)]);

      for (int i = 1; i < C; i++) {      
// #ifdef MGARD_CUDA_FMA
//         vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] =
//       __fma_rn(dist_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)], 
//         vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)]) * am_sm[i];
// #else
//       // if (r_gl == 0 && f_gl == 0 && r_sm == 0 && f_sm == 0) 
//       //   printf("(%f + %f * %f) * %f -> %f\n", 
//       //             vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)],
//       //             dist_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, i-1, f_sm)], am_sm[i],
//       //             (vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] -
//       //   dist_sm[i] * vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)]) / am_sm[i]);


//       vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] =
//          (vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] -
//          dist_sm[i] * vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)]) / am_sm[i];


// #endif
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] = 
        tridiag_backward(vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)],
         dist_sm[i], am_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);

        if (debug) printf("calc vec_sm[%d] = %f\n", get_idx(ldsm1, ldsm2, r_sm, i, f_sm), 
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);
      }

      /* Store last v */
      prev_vec_sm = vec_sm[get_idx(ldsm1, ldsm2, r_sm, C - 1, f_sm)];
    }
    __syncthreads();

    /* flush results to v */
    if (r_sm < r_rest && f_sm < f_rest) {
      for (int i = 0; i < C; i++) {
        vec[get_idx(ldv1, ldv2, r_sm, (nc_c - 1) - (c_gl + i), f_sm)] = 
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)];
      }
    }
    __syncthreads();

    /* Update unloaded col */
    c_rest -= c_main;

    /* Advance c */
    c_gl += C;

    /* Copy next ghost to main */
    c_ghost = min(G, c_main - (C - G));
    if (r_sm < r_rest && f_sm < f_rest) {
      for (int i = 0; i < c_ghost; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] = 
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i + C, f_sm)];
      }
    }
    if (r_sm == 0 && c_sm < c_ghost) {
      am_sm[c_sm] = am_sm[c_sm + C];
      dist_sm[c_sm] = dist_sm[c_sm + C];
    }
    __syncthreads();

  } // end of while

  // Load all rest col 
  if (r_sm < r_rest && f_sm < f_rest) {
    for (int i = 0; i < c_rest; i++) {
      vec_sm[get_idx(ldsm1, ldsm2, r_sm, i + c_ghost, f_sm)] =
          vec[get_idx(ldv1, ldv2, r_sm, (nc_c - 1) - (c_gl + i + c_ghost), f_sm)];

      if (debug) printf("load ec_sm[%d] = %f\n", get_idx(ldsm1, ldsm2, r_sm, i + c_ghost, f_sm), 
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i + c_ghost, f_sm)]);
    }
  }
  if (r_sm == 0 && c_sm < c_rest) {
    am_sm[c_sm + c_ghost] = am[(nc_c - 1) - (c_gl + c_sm + c_ghost)];
    dist_sm[c_sm + c_ghost] = dist_c[(nc_c - 1) - (c_gl + c_sm + c_ghost)];
  }
  __syncthreads();

  /* Only 1 col remain */
  if (c_ghost + c_rest == 1) {
    if (r_sm < r_rest && f_sm < f_rest) {
// #ifdef MGARD_CUDA_FMA
//       vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] =
//       __fma_rn(dist_sm[0], prev_vec_sm, vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)]) *
//       am_sm[0];
// #else
//       // if (r_gl == 0 && f_gl == 0 && r_sm == 0 && f_sm == 0) 
//       //     printf("(%f + %f * %f) * %f -> %f\n", 
//       //             vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)],
//       //             dist_sm[0], prev_vec_sm, am_sm[0],
//       //             (vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] - dist_sm[0] * prev_vec_sm) / am_sm[0]);
//       vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] =
//       (vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] - dist_sm[0] * prev_vec_sm) / am_sm[0];
// #endif
      vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, c_sm)] = 
        tridiag_backward(prev_vec_sm, dist_sm[0], am_sm[0], 
          vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, c_sm)]);
        if (debug) printf("calc vec_sm[%d] = %f\n", get_idx(ldsm1, ldsm2, r_sm, 0, f_sm), 
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)]);
      // printf ("prev_vec_sm = %f\n", prev_vec_sm );
      // printf ("vec_sm[r_sm * ldsm + 0] = %f\n", vec_sm[r_sm * ldsm + 0] );
    }
    //__syncthreads();

  } else {
    if (r_sm < r_rest && f_sm < f_rest) {
// #ifdef MGARD_CUDA_FMA
//       vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] =
//       __fma_rn(dist_sm[0], prev_vec_sm, vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)]) *
//       am_sm[0];
// #else
//       // if (r_gl == 0 && f_gl == 0 && r_sm == 0 && f_sm == 0) 
//       //     printf("(%f + %f * %f) * %f -> %f\n", 
//       //             vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)],
//       //             dist_sm[0], prev_vec_sm, am_sm[0],
//       //             (vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] - dist_sm[0] * prev_vec_sm) / am_sm[0]);
//       vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] =
//       (vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] - dist_sm[0] * prev_vec_sm) / am_sm[0];
// #endif
      vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, c_sm)] = 
        tridiag_backward(prev_vec_sm, dist_sm[0], am_sm[0], 
          vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, c_sm)]);
        if (debug) printf("calc vec_sm[%d] = %f\n", get_idx(ldsm1, ldsm2, r_sm, 0, f_sm), 
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)]);
      for (int i = 1; i < c_ghost + c_rest; i++) {
       
// #ifdef MGARD_CUDA_FMA
//         vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] =
//       __fma_rn(dist_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)], 
//         vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)]) * am_sm[i];
// #else
//       // if (r_gl == 0 && f_gl == 0 && r_sm == 0 && f_sm == 0) 
//       //   printf("(%f + %f * %f) * %f -> %f\n", 
//       //             vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)],
//       //             dist_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, i-1, f_sm)], am_sm[i],
//       //             (vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] -
//       //    dist_sm[i] * vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)]) / am_sm[i]);
//        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] =
//          (vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] -
//          dist_sm[i] * vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)]) / am_sm[i];
// #endif
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] = 
        tridiag_backward(vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)],
         dist_sm[i], am_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);
        if (debug) printf("calc vec_sm[%d] = %f\n", get_idx(ldsm1, ldsm2, r_sm, i, f_sm), 
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);
      }
    }
  }
  __syncthreads();
  /* flush results to v */
  if (r_sm < r_rest && f_sm < f_rest) {
    for (int i = 0; i < c_ghost + c_rest; i++) {
      vec[get_idx(ldv1, ldv2, r_sm, (nc_c - 1) - (c_gl + i), f_sm)] = 
      vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)];
      // printf("c_stride = %d, c_sm = %d, vec_sm = %f, vec[%d] =
      // %f\n",c_stride, c_sm, vec_sm[r_sm * ldsm + 0],i * row_stride * lddv +
      // c_stride, vec[i * row_stride * lddv + c_stride]);
    }
  }
  __syncthreads();
}

template <typename T, int D, int R, int C, int F, int G>
void ipk_2_adaptive_launcher( mgard_cuda_handle<T, D> &handle, 
                              thrust::device_vector<int> shape, 
                              thrust::device_vector<int> shape_c, 
                              thrust::device_vector<int> ldvs, 
                              thrust::device_vector<int> ldws,
                              thrust::device_vector<int> processed_dims,
                              int curr_dim_r, int curr_dim_c, int curr_dim_f,
                              T* am, T *bm, T * ddist_c, T *dv,
                              int lddv1, int lddv2, int queue_idx) {
  int nr = shape[curr_dim_r];
  int nc_c = shape_c[curr_dim_c];
  int nf_c = shape_c[curr_dim_f];

  int total_thread_x = nf_c;
  int total_thread_y = nr;
  int total_thread_z = 1;
  int tbx, tby, tbz, gridx, gridy, gridz;
  dim3 threadsPerBlock, blockPerGrid;
  size_t sm_size;

  tbx = std::max(F, std::min(F, total_thread_x)); 
  tby = std::max(R, std::min(R, total_thread_y)); 
  tbz = 1; 
  sm_size = (R * F + 2) * (C + G) * sizeof(T); 
  sm_size += (D * 5) * sizeof(int); 
  gridx = ceil((float)total_thread_x / tbx); 
  gridy = ceil((float)total_thread_y / tby); 
  gridz = 1; 
  for (int d = 0; d < D; d++) {
    if (d != curr_dim_f && d != curr_dim_c && d != curr_dim_r) {
      int t = shape[d];
      for (int k = 0; k < processed_dims.size(); k++) {
        if (d == processed_dims[k]) {
          t = shape_c[d];
        }
      }
      gridx *= t;
    }
  }

  threadsPerBlock = dim3(tbx, tby, tbz); 
  blockPerGrid = dim3(gridx, gridy, gridz); 
  _ipk_2<T, D, R, C, F, G><<<blockPerGrid, threadsPerBlock, sm_size, 
                                 *(cudaStream_t *)handle.get(queue_idx)>>>( 
                                    thrust::raw_pointer_cast(shape.data()), 
                                    thrust::raw_pointer_cast(shape_c.data()), 
                                    thrust::raw_pointer_cast(ldvs.data()), 
                                    thrust::raw_pointer_cast(ldws.data()),
                                    processed_dims.size(), 
                                    thrust::raw_pointer_cast(processed_dims.data()), 
                                    curr_dim_r, curr_dim_c, curr_dim_f, 
                                    am, bm, ddist_c, dv, lddv1, lddv2); 
  gpuErrchk(cudaGetLastError()); 
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}


template <typename T, int D>
void ipk_2( mgard_cuda_handle<T, D> &handle, 
            thrust::device_vector<int> shape, 
            thrust::device_vector<int> shape_c, 
            thrust::device_vector<int> ldvs, 
            thrust::device_vector<int> ldws,
            thrust::device_vector<int> processed_dims,
            int curr_dim_r, int curr_dim_c, int curr_dim_f, 
            T* am, T *bm, T * ddist_c, T *dv,
            int lddv1, int lddv2, int queue_idx, int config) {


#define IPK(R, C, F, G) {ipk_2_adaptive_launcher<T, D, R, C, F, G>(\
                                handle,\
                                shape, shape_c, ldvs, ldws, processed_dims,\
                                curr_dim_r, curr_dim_c, curr_dim_f,\
                                am, bm, ddist_c, dv,\
                                lddv1, lddv2, queue_idx);}
  bool profile = false;
#ifdef MGARD_CUDA_KERNEL_PROFILE  
  profile = true;
#endif
  if (D >= 3) {
    if (profile || config == 6) { IPK(2, 2, 128, 2) }
    if (profile || config == 5) { IPK(2, 2, 64, 2) }
    if (profile || config == 4) { IPK(2, 2, 32, 2) }
    if (profile || config == 3) { IPK(4, 4, 16, 4) }
    if (profile || config == 2) { IPK(8, 8, 8, 4) }
    if (profile || config == 1) { IPK(4, 4, 4, 4) }
    if (profile || config == 0) { IPK(2, 2, 2, 2) }
  } else if(D == 2) {
    if (profile || config == 6) { IPK(1, 2, 128, 2) }
    if (profile || config == 5) { IPK(1, 2, 64, 2) }
    if (profile || config == 4) { IPK(1, 2, 32, 2) }
    if (profile || config == 3) { IPK(1, 4, 16, 4) }
    if (profile || config == 2) { IPK(1, 8, 8, 4) }
    if (profile || config == 1) { IPK(1, 4, 4, 4) }
    if (profile || config == 0) { IPK(1, 2, 4, 2) }
  } else {
    printf("Error: solve_tridiag_2_cpt is only for 3D and 2D data\n");
  }
#undef IPK

}

template <typename T, int D, int R, int C, int F, int G>
void ipk_2_adaptive_launcher( mgard_cuda_handle<T, D> &handle, 
                              int * shape_h, int * shape_c_h, int * shape_d, int * shape_c_d, 
                              int * ldvs, int * ldws,
                              int processed_n, int * processed_dims_h, int * processed_dims_d,
                              int curr_dim_r, int curr_dim_c, int curr_dim_f,
                              T* am, T *bm, T * ddist_c, T *dv,
                              int lddv1, int lddv2, int queue_idx) {
  int nr = shape_h[curr_dim_r];
  int nc_c = shape_c_h[curr_dim_c];
  int nf_c = shape_c_h[curr_dim_f];

  int total_thread_x = nf_c;
  int total_thread_y = nr;
  int total_thread_z = 1;
  int tbx, tby, tbz, gridx, gridy, gridz;
  dim3 threadsPerBlock, blockPerGrid;
  size_t sm_size;

  tbx = std::max(F, std::min(F, total_thread_x)); 
  tby = std::max(R, std::min(R, total_thread_y)); 
  tbz = 1; 
  sm_size = (R * F + 2) * (C + G) * sizeof(T); 
  sm_size += (D * 5) * sizeof(int); 
  gridx = ceil((float)total_thread_x / tbx); 
  gridy = ceil((float)total_thread_y / tby); 
  gridz = 1; 
  for (int d = 0; d < D; d++) {
    if (d != curr_dim_f && d != curr_dim_c && d != curr_dim_r) {
      int t = shape_h[d];
      for (int k = 0; k < processed_n; k++) {
        if (d == processed_dims_h[k]) {
          t = shape_c_h[d];
        }
      }
      gridx *= t;
    }
  }

  threadsPerBlock = dim3(tbx, tby, tbz); 
  blockPerGrid = dim3(gridx, gridy, gridz); 
  _ipk_2<T, D, R, C, F, G><<<blockPerGrid, threadsPerBlock, sm_size, 
                                 *(cudaStream_t *)handle.get(queue_idx)>>>( 
                                    shape_d, shape_c_d, ldvs, ldws,
                                    processed_n, processed_dims_d,  
                                    curr_dim_r, curr_dim_c, curr_dim_f, 
                                    am, bm, ddist_c, dv, lddv1, lddv2); 
  gpuErrchk(cudaGetLastError()); 
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}


template <typename T, int D>
void ipk_2( mgard_cuda_handle<T, D> &handle, 
            int * shape_h, int * shape_c_h, int * shape_d, int * shape_c_d, 
            int * ldvs, int * ldws,
            int processed_n, int * processed_dims_h, int * processed_dims_d,
            int curr_dim_r, int curr_dim_c, int curr_dim_f, 
            T* am, T *bm, T * ddist_c, T *dv,
            int lddv1, int lddv2, int queue_idx, int config) {


#define IPK(R, C, F, G) {ipk_2_adaptive_launcher<T, D, R, C, F, G>(\
                                handle,\
                                shape_h, shape_c_h, shape_d, shape_c_d, ldvs, ldws,\
                                processed_n, processed_dims_h, processed_dims_d,\ 
                                curr_dim_r, curr_dim_c, curr_dim_f,\
                                am, bm, ddist_c, dv,\
                                lddv1, lddv2, queue_idx);}
  bool profile = false;
#ifdef MGARD_CUDA_KERNEL_PROFILE  
  profile = true;
#endif
  if (D >= 3) {
    if (profile || config == 6) { IPK(2, 2, 128, 2) }
    if (profile || config == 5) { IPK(2, 2, 64, 2) }
    if (profile || config == 4) { IPK(2, 2, 32, 2) }
    if (profile || config == 3) { IPK(4, 4, 16, 4) }
    if (profile || config == 2) { IPK(8, 8, 8, 4) }
    if (profile || config == 1) { IPK(4, 4, 4, 4) }
    if (profile || config == 0) { IPK(2, 2, 2, 2) }
  } else if(D == 2) {
    if (profile || config == 6) { IPK(1, 2, 128, 2) }
    if (profile || config == 5) { IPK(1, 2, 64, 2) }
    if (profile || config == 4) { IPK(1, 2, 32, 2) }
    if (profile || config == 3) { IPK(1, 4, 16, 4) }
    if (profile || config == 2) { IPK(1, 8, 8, 4) }
    if (profile || config == 1) { IPK(1, 4, 4, 4) }
    if (profile || config == 0) { IPK(1, 2, 4, 2) }
  } else {
    printf("Error: solve_tridiag_2_cpt is only for 3D and 2D data\n");
  }
#undef IPK

}

template <typename T, int D, int R, int C, int F, int G>
__global__ void _ipk_3( int * shape, int * shape_c,
                        int * ldvs, int * ldws,
                        int processed_n, int * processed_dims, 
                        int curr_dim_r, int curr_dim_c, int curr_dim_f,
                        T * am, T *bm, T * dist_r, T *v, int ldv1, int ldv2) {

  // bool debug = false;
  // if (blockIdx.z == 0 && blockIdx.y == 0 && blockIdx.x == 0 && 
  // threadIdx.z == 0 && threadIdx.y == 0 ) debug = false;

  // bool debug2 = false;
  // if (threadIdx.z == 0 && threadIdx.y == 0 && threadIdx.x == 0 ) debug2 = false;

  size_t threadId =  (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;


  


  
  T *sm = SharedMemory<T>();
  int ldsm1 = F;
  int ldsm2 = C;
  T *vec_sm = sm;
  T *bm_sm = sm + (R + G) * ldsm1 * ldsm2;
  T *dist_sm = bm_sm + (R + G);
  int *shape_sm               = (int *)(dist_sm + R + G);
  int *shape_c_sm             = shape_sm + D;
  int *processed_dims_sm      = shape_c_sm + D;
  int *ldvs_sm                = processed_dims_sm + D;
  int *ldws_sm                = ldvs_sm + D;
  int idx[D];
  if (threadId < D) {
    shape_sm[threadId] = shape[threadId];
    shape_c_sm[threadId] = shape_c[threadId];
    ldvs_sm[threadId] = ldvs[threadId];
    ldws_sm[threadId] = ldws[threadId];
  }
  if (threadId < processed_n) {
    processed_dims_sm[threadId] = processed_dims[threadId];
  }
  __syncthreads();

  for (int d = 0; d < D; d ++) idx[d] = 0;

  int nr_c = shape_c_sm[curr_dim_r];
  int nc_c = shape_c_sm[curr_dim_c];
  int nf_c = shape_c_sm[curr_dim_f];

  int bidx = blockIdx.x;
  int firstD = div_roundup(nf_c, blockDim.x);
  int blockId = bidx % firstD;

  bidx /= firstD;

  for (int d = 0; d < D; d++) {
    if (d != curr_dim_r && d != curr_dim_c && d != curr_dim_f) {
      int t = shape_sm[d];
      for (int k = 0; k < processed_n; k++) {
        if (d == processed_dims[k]) {
          t = shape_c_sm[d];
        }
      }
      idx[d] = bidx % t;
      bidx /= t;
    }
  }

  size_t other_offset_v = get_idx<D>(ldvs_sm, idx);
  v = v + other_offset_v;

  int f_gl = blockId * F;
  int c_gl = blockIdx.y * C;
  int r_gl = 0;

  int f_sm = threadIdx.x;
  int c_sm = threadIdx.y;
  int r_sm = threadIdx.x;

  T *vec = v + get_idx(ldv1, ldv2, 0, c_gl, f_gl);

  register T prev_vec_sm = 0.0;

  int f_rest = min(F, nf_c - blockId * F);
  int c_rest = min(C, nc_c - blockIdx.y * C);

  int r_rest = nr_c;
  int r_ghost = min(nr_c, G);
  int r_main = R;

  // if (f_gl + f_sm == 32 && c_gl + c_sm == 1 ) debug = false;

  /* Load first ghost */
  if (c_sm < c_rest && f_sm < f_rest) {
    for (int i = 0; i < r_ghost; i++) {
      vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] = 
        vec[get_idx(ldv1, ldv2, r_gl + i, c_sm, f_sm)];
        // if (debug) printf("load first sm[%d] %f [%d]\n", i, 
        //             vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)], r_gl + i);
    }
  }

  if (c_sm == 0 && r_sm < r_ghost) bm_sm[r_sm] = bm[r_gl + r_sm];
  r_rest -= r_ghost;
  __syncthreads();

  while (r_rest > R - r_ghost) {
    r_main = min(R, r_rest);
    if (c_sm < c_rest && f_sm < f_rest) {
      for (int i = 0; i < r_main; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, i + r_ghost, c_sm, f_sm)] =
            vec[get_idx(ldv1, ldv2, r_gl + i + r_ghost, c_sm, f_sm)];
       // if (debug) printf("load ghost sm[%d] %f [%d]\n", i + r_ghost, 
       //              vec_sm[get_idx(ldsm1, ldsm2, i + r_ghost, c_sm, f_sm)], 
       //              r_gl + i + r_ghost);
      }
    }
    if (c_sm == 0 && r_sm < r_main) bm_sm[r_sm + r_ghost] = bm[r_gl + r_sm + r_ghost];
    __syncthreads();

    /* Computation of v in parallel*/
    if (c_sm < c_rest && f_sm < f_rest) {
      
// #ifdef MGARD_CUDA_FMA
//       vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] = 
//       __fma_rn(prev_vec_sm, bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);
// #else
//       vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] -= prev_vec_sm * bm_sm[0];
// #endif
      // if (debug) printf("compute sm[%d] %f <- %f %f %f\n", 0, 
      //               tridiag_forward(prev_vec_sm, bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]),
      //               prev_vec_sm, bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);

      vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] = 
        tridiag_forward(prev_vec_sm, bm_sm[0], 
          vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);
        
      for (int i = 1; i < R; i++) {
// #ifdef MGARD_CUDA_FMA
//         vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] = 
//       __fma_rn(vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)], bm_sm[i], 
//         vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]);
// #else
//         vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] -=
//          vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)] * bm_sm[i];
// #endif

        // if (debug) printf("compute sm[%d] %f <- %f %f %f\n", i, 
        //             tridiag_forward(vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)],
        //              bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]),
        //             vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)],
        //              bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]);

        vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] = 
          tridiag_forward(vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)], 
            bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]);
      }

      /* Store last v */
      prev_vec_sm = vec_sm[get_idx(ldsm1, ldsm2, R - 1, c_sm, f_sm)];
    }
    __syncthreads();

    /* flush results to v */
    if (c_sm < c_rest && f_sm < f_rest) {
      for (int i = 0; i < R; i++) {
        vec[get_idx(ldv1, ldv2, r_gl + i, c_sm, f_sm)] = 
        vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)];
        // if (debug) printf("store[%d] %f [%d]\n", r_gl + i, 
        //   vec[get_idx(ldv1, ldv2, r_gl + i, c_sm, f_sm)], i);
      }
    }
    __syncthreads();

    // /* Update unloaded col */
    r_rest -= r_main;

    /* Advance c */
    r_gl += R;

    /* Copy next ghost to main */
    r_ghost = min(G, r_main - (R - G));
    if (c_sm < c_rest && f_sm < f_rest) {
      for (int i = 0; i < r_ghost; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] = 
        vec_sm[get_idx(ldsm1, ldsm2, i + R, c_sm, f_sm)];

        // if (debug) printf("copy next ghost[%d] %f [%d]\n", i, 
        //   vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)], i+R);

      }
    }
    if (c_sm == 0 && r_sm < r_ghost) bm_sm[r_sm] = bm_sm[r_sm + R];
    __syncthreads();

  } //end of while

  /* Load all rest col */
  if (c_sm < c_rest && f_sm < f_rest) {
    for (int i = 0; i < r_rest; i++) {
      vec_sm[get_idx(ldsm1, ldsm2, i + r_ghost, c_sm, f_sm)] =
          vec[get_idx(ldv1, ldv2, r_gl + i + r_ghost, c_sm, f_sm)];

      // if (debug) printf("load ghost-rest sm[%d] %f [%d]\n", i + r_ghost, 
      //               vec_sm[get_idx(ldsm1, ldsm2, i + r_ghost, c_sm, f_sm)], 
      //               r_gl + i + r_ghost);
    }
  }

  if (c_sm == 0 && r_sm < r_rest) bm_sm[r_sm + r_ghost] = bm[r_gl + r_sm + r_ghost];

  __syncthreads();

  /* Only 1 col remain */
  if (r_ghost + r_rest == 1) {
    if (c_sm < c_rest && f_sm < f_rest) {
      
// #ifdef MGARD_CUDA_FMA
//       vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] = 
//       __fma_rn(prev_vec_sm, bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);
// #else
//       vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] -= prev_vec_sm * bm_sm[0];
// #endif
      // if (debug) printf("compute-rest sm[%d] %f <- %f %f %f\n", 0, 
      //               tridiag_forward(prev_vec_sm, bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]),
      //               prev_vec_sm, bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);

      vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] = 
        tridiag_forward(prev_vec_sm, bm_sm[0], 
          vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);
      // printf ("prev_vec_sm = %f\n", prev_vec_sm );
      // printf ("vec_sm[r_sm * ldsm + 0] = %f\n", vec_sm[r_sm * ldsm + 0] );
    }
    //__syncthreads();

  } else {
    if (c_sm < c_rest && f_sm < f_rest) {     
// #ifdef MGARD_CUDA_FMA
//       vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] = 
//       __fma_rn(prev_vec_sm, bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);
// #else
//       vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] -= prev_vec_sm * bm_sm[0];
// #endif

      // if (debug) printf("compute-rest sm[%d] %f <- %f %f %f\n", 0, 
      //               tridiag_forward(prev_vec_sm, bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]),
      //               prev_vec_sm, bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);

      vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] = 
        tridiag_forward(prev_vec_sm, bm_sm[0], 
          vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);
      for (int i = 1; i < r_ghost + r_rest; i++) {
// #ifdef MGARD_CUDA_FMA
//         vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] = 
//       __fma_rn(vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)], bm_sm[i], 
//         vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]);
// #else
//       vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] -= 
//         vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)] * bm_sm[i];
// #endif
        // if (debug) printf("compute-rest sm[%d] %f <- %f %f %f\n", i, 
        //             tridiag_forward(vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)],
        //              bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]),
        //             vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)],
        //              bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]);

        vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] = 
          tridiag_forward(vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)], 
            bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]);
      }
    }
  }
  __syncthreads();
  /* flush results to v */
  if (c_sm < c_rest && f_sm < f_rest) {
    for (int i = 0; i < r_ghost + r_rest; i++) {
      

      vec[get_idx(ldv1, ldv2, r_gl + i, c_sm, f_sm)] = 
      vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)];

      // if (debug) printf("store-rest[%d] %f [%d]\n", r_gl + i, 
      //     vec[get_idx(ldv1, ldv2, r_gl + i, c_sm, f_sm)], i);
      // printf("c_stride = %d, c_sm = %d, vec_sm = %f, vec[%d] =
      // %f\n",c_stride, c_sm, vec_sm[r_sm * ldsm + 0],i * row_stride * lddv +
      // c_stride, vec[i * row_stride * lddv + c_stride]);
    }
  }
  __syncthreads();

  /* backward */
  T * am_sm = bm_sm;
  r_rest = nr_c;
  r_ghost = min(nr_c, G);
  r_main = R;
  r_gl = 0;
  prev_vec_sm = 0.0;

  /* Load first ghost */
  if (c_sm < c_rest && f_sm < f_rest) {
    for (int i = 0; i < r_ghost; i++) {
      vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] = 
        vec[get_idx(ldv1, ldv2, (nr_c - 1) - (r_gl + i), c_sm, f_sm)];

        // if (debug) printf("load first sm[%d] %f [%d]\n", i, 
        //             vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)], (nr_c - 1) - (r_gl + i));

      // if (r_sm == 0) printf("r0_stride = %d, vec_sm[%d] = %f\n", r0_stride,
      // i, vec_sm[i * ldsm + c_sm]);
    }
  }

  if (c_sm == 0 && r_sm < r_ghost) {
    am_sm[r_sm] = am[(nr_c - 1) - (r_gl + r_sm)];
    dist_sm[r_sm] = dist_r[(nr_c - 1) - (r_gl + r_sm)];
  }
  r_rest -= r_ghost;
  __syncthreads();

  while (r_rest > R - r_ghost) {
    r_main = min(R, r_rest);
    if (c_sm < c_rest && f_sm < f_rest) {
      for (int i = 0; i < r_main; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, i + r_ghost, c_sm, f_sm)] =
            vec[get_idx(ldv1, ldv2, (nr_c - 1) - (r_gl + i + r_ghost), c_sm, f_sm)];
        // if (debug) printf("load ghost sm[%d] %f [%d]\n", i + r_ghost, 
        //             vec_sm[get_idx(ldsm1, ldsm2, i + r_ghost, c_sm, f_sm)], 
        //             (nr_c - 1) - (r_gl + i + r_ghost));

      }
    }
    if (c_sm == 0 && r_sm < r_main) {
      am_sm[r_sm + r_ghost] = am[(nr_c - 1) - (r_gl + r_sm + r_ghost)];
      dist_sm[r_sm + r_ghost] = dist_r[(nr_c - 1) - (r_gl + r_sm + r_ghost)];
    }
    __syncthreads();

    /* Computation of v in parallel*/
    if (c_sm < c_rest && f_sm < f_rest) {
// #ifdef MGARD_CUDA_FMA
//       vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] =
//       __fma_rn(dist_sm[0], prev_vec_sm, vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]) * am_sm[0];
// #else
//       vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] = 
//       (vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] - dist_sm[0] * prev_vec_sm) / am_sm[0];
// #endif

       // if (debug) printf("compute sm[%d] %f <- %f %f %f %f\n", 0, 
       //              tridiag_backward(prev_vec_sm, dist_sm[0], am_sm[0], 
       //    vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]),
       //              prev_vec_sm, dist_sm[0], am_sm[0], 
       //    vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);

      vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] = 
        tridiag_backward(prev_vec_sm, dist_sm[0], am_sm[0], 
          vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);
      for (int i = 1; i < R; i++) {
        
// #ifdef MGARD_CUDA_FMA
//         vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] =
//       __fma_rn(dist_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)], 
//         vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]) * am_sm[i];
// #else
//       vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] =
//         (vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] - 
//          dist_sm[i] * vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)]) / am_sm[i];
// #endif
        
        // if (debug) printf("compute sm[%d] %f <- %f %f %f %f\n", i, 
        //             tridiag_backward(vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)],
        //  dist_sm[i], am_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]),
        //             vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)],
        //  dist_sm[i], am_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]);

        vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] = 
        tridiag_backward(vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)],
         dist_sm[i], am_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]);
      }

      /* Store last v */
      prev_vec_sm = vec_sm[get_idx(ldsm1, ldsm2, R - 1, c_sm, f_sm)];
    }
    __syncthreads();

    /* flush results to v */
    if (c_sm < c_rest && f_sm < f_rest) {
      for (int i = 0; i < R; i++) {
        // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        //   printf("%d %d %d (%f) <- %d %d %d\n", (nr - 1) - (r_gl + i), c_sm, f_sm,
        //           vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)], i, c_sm, f_sm);
        // }
        vec[get_idx(ldv1, ldv2, (nr_c - 1) - (r_gl + i), c_sm, f_sm)] = 
        vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)];

        // if (debug) printf("store[%d] %f [%d]\n", (nr_c - 1) - (r_gl + i), 
        //   vec[get_idx(ldv1, ldv2, (nr_c - 1) - (r_gl + i), c_sm, f_sm)], i);


      }
    }
    __syncthreads();

    // /* Update unloaded col */
    r_rest -= r_main;

    /* Advance c */
    r_gl += R;

    /* Copy next ghost to main */
    r_ghost = min(G, r_main - (R - G));
    if (c_sm < c_rest && f_sm < f_rest) {
      for (int i = 0; i < r_ghost; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] = 
        vec_sm[get_idx(ldsm1, ldsm2, i + R, c_sm, f_sm)];

         // if (debug) printf("copy next ghost[%d] %f [%d]\n", i, 
         //  vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)], i+R);

      }
    }
    if (c_sm == 0 && r_sm < r_ghost) {
      am_sm[r_sm] = am_sm[r_sm + R];
      dist_sm[r_sm] = dist_sm[r_sm + R];
    }
    __syncthreads();

  } //end of while

  /* Load all rest col */
  if (c_sm < c_rest && f_sm < f_rest) {
    for (int i = 0; i < r_rest; i++) {
      vec_sm[get_idx(ldsm1, ldsm2, i + r_ghost, c_sm, f_sm)] =
          vec[get_idx(ldv1, ldv2, (nr_c - 1) - (r_gl + i + r_ghost), c_sm, f_sm)];

      // if (debug) printf("load ghost-rest sm[%d] %f [%d]\n", i + r_ghost, 
      //               vec_sm[get_idx(ldsm1, ldsm2, i + r_ghost, c_sm, f_sm)], 
      //               (nr_c - 1) - (r_gl + i + r_ghost));

    }
  }
  if (c_sm == 0 && r_sm < r_rest) {
    am_sm[r_sm + r_ghost] = am[(nr_c - 1) - (r_gl + r_sm + r_ghost)];
    dist_sm[r_sm + r_ghost] = dist_r[(nr_c - 1) - (r_gl + r_sm + r_ghost)];
  }
  __syncthreads();

  /* Only 1 col remain */
  if (r_ghost + r_rest == 1) {
    if (c_sm < c_rest && f_sm < f_rest) {
// #ifdef MGARD_CUDA_FMA
//       vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] =
//       __fma_rn(dist_sm[0], prev_vec_sm, vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]) * am_sm[0];
// #else
//       vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] = 
//       (vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] - dist_sm[0] * prev_vec_sm) / am_sm[0];
// #endif
      // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
      //   printf("backward 1 (%f) %f %f %f %f\n", tridiag_backward(prev_vec_sm, dist_sm[0], am_sm[0], 
      //     vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]), prev_vec_sm, dist_sm[0], am_sm[0], 
      //     vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);

      // }
      // if (debug) printf("compute sm[%d] %f <- %f %f %f %f\n", 0, 
      //               tridiag_backward(prev_vec_sm, dist_sm[0], am_sm[0], 
      //     vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]),
      //               prev_vec_sm, dist_sm[0], am_sm[0], 
      //     vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);

      vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] = 
        tridiag_backward(prev_vec_sm, dist_sm[0], am_sm[0], 
          vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);
      // printf ("prev_vec_sm = %f\n", prev_vec_sm );
      // printf ("vec_sm[r_sm * ldsm + 0] = %f\n", vec_sm[r_sm * ldsm + 0] );
    }
    //__syncthreads();

  } else {
    if (c_sm < c_rest && f_sm < f_rest) {
// #ifdef MGARD_CUDA_FMA
//       vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] =
//       __fma_rn(dist_sm[0], prev_vec_sm, vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]) * am_sm[0];
// #else
//       vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] = 
//       (vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] - dist_sm[0] * prev_vec_sm) / am_sm[0];
// #endif
      // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
      //   printf("backward 1 (%f) %f %f %f %f\n", tridiag_backward(prev_vec_sm, dist_sm[0], am_sm[0], 
      //     vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]), prev_vec_sm, dist_sm[0], am_sm[0], 
      //     vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);

      // }

      // if (debug) printf("compute sm[%d] %f <- %f %f %f %f\n", 0, 
      //               tridiag_backward(prev_vec_sm, dist_sm[0], am_sm[0], 
      //     vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]),
      //               prev_vec_sm, dist_sm[0], am_sm[0], 
      //     vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);

      vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] = 
        tridiag_backward(prev_vec_sm, dist_sm[0], am_sm[0], 
          vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);
      for (int i = 1; i < r_ghost + r_rest; i++) {
        
// #ifdef MGARD_CUDA_FMA
//         vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] =
//       __fma_rn(dist_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)], 
//         vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]) * am_sm[i];
// #else
//         vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] =
//         (vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] - 
//          dist_sm[i] * vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)]) / am_sm[i];
// #endif
      //   if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
      //   printf("backward R=%d (%f) %f %f %f %f\n", i, tridiag_backward(vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)],
      //    dist_sm[i], am_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]), vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)],
      //    dist_sm[i], am_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]);

      // }

        // if (debug) printf("compute sm[%d] %f <- %f %f %f %f\n", i, 
        //             tridiag_backward(vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)],
        //  dist_sm[i], am_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]),
        //             vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)],
        //  dist_sm[i], am_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]);

        vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] = 
        tridiag_backward(vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)],
         dist_sm[i], am_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]);
      }
    }
  }
  __syncthreads();
  /* flush results to v */
  if (c_sm < c_rest && f_sm < f_rest) {
    for (int i = 0; i < r_ghost + r_rest; i++) {
      vec[get_idx(ldv1, ldv2, (nr_c - 1) - (r_gl + i), c_sm, f_sm)] = 
      vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)];

      // if (debug) printf("store-rest[%d] %f [%d]\n", (nr_c - 1) - (r_gl + i), 
      //     vec[get_idx(ldv1, ldv2, (nr_c - 1) - (r_gl + i), c_sm, f_sm)], i);

      // printf("c_stride = %d, c_sm = %d, vec_sm = %f, vec[%d] =
      // %f\n",c_stride, c_sm, vec_sm[r_sm * ldsm + 0],i * row_stride * lddv +
      // c_stride, vec[i * row_stride * lddv + c_stride]);
    }
  }
  __syncthreads();
}

template <typename T, int D, int R, int C, int F, int G>
void ipk_3_adaptive_launcher( mgard_cuda_handle<T, D> &handle, 
                              thrust::device_vector<int> shape, 
                              thrust::device_vector<int> shape_c, 
                              thrust::device_vector<int> ldvs, 
                              thrust::device_vector<int> ldws,
                              thrust::device_vector<int> processed_dims,
                              int curr_dim_r, int curr_dim_c, int curr_dim_f,
                              T* am, T *bm, T * ddist_r, T *dv,
                              int lddv1, int lddv2, int queue_idx) {

  // printf("am: ");
  // print_matrix_cuda(1, nr, am, nr);
  // printf("bm: ");
  // print_matrix_cuda(1, nr, bm, nr);

  int nr_c = shape_c[curr_dim_r];
  int nc_c = shape_c[curr_dim_c];
  int nf_c = shape_c[curr_dim_f];

  int total_thread_x = nf_c;
  int total_thread_y = nc_c;
  int total_thread_z = 1;
  int tbx, tby, tbz, gridx, gridy, gridz;
  dim3 threadsPerBlock, blockPerGrid;
  size_t sm_size;

  tbx = std::max(F, std::min(F, total_thread_x));
  tby = std::max(C, std::min(C, total_thread_y));
  tbz = 1; 
  sm_size = (C * F + 2) * (R + G) * sizeof(T); 
  sm_size += (D * 5) * sizeof(int); 
  gridx = ceil((float)total_thread_x / tbx); 
  gridy = ceil((float)total_thread_y / tby); 
  gridz = 1; 
  for (int d = 0; d < D; d++) {
    if (d != curr_dim_f && d != curr_dim_c && d != curr_dim_r) {
      int t = shape[d];
      for (int k = 0; k < processed_dims.size(); k++) {
        if (d == processed_dims[k]) {
          t = shape_c[d];
        }
      }
      gridx *= t;
    }
  }
  threadsPerBlock = dim3(tbx, tby, tbz); 
  blockPerGrid = dim3(gridx, gridy, gridz); 
  _ipk_3<T, D, R, C, F, G><<<blockPerGrid, threadsPerBlock, sm_size, 
                                 *(cudaStream_t *)handle.get(queue_idx)>>>( 
                                  thrust::raw_pointer_cast(shape.data()), 
                                  thrust::raw_pointer_cast(shape_c.data()), 
                                  thrust::raw_pointer_cast(ldvs.data()), 
                                  thrust::raw_pointer_cast(ldws.data()),
                                  processed_dims.size(), 
                                  thrust::raw_pointer_cast(processed_dims.data()), 
                                  curr_dim_r, curr_dim_c, curr_dim_f,   
                                  am, bm, ddist_r, dv, lddv1, lddv2); 
  gpuErrchk(cudaGetLastError()); 
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}


template <typename T, int D>
void ipk_3( mgard_cuda_handle<T, D> &handle, 
            thrust::device_vector<int> shape, 
            thrust::device_vector<int> shape_c, 
            thrust::device_vector<int> ldvs, 
            thrust::device_vector<int> ldws,
            thrust::device_vector<int> processed_dims,
            int curr_dim_r, int curr_dim_c, int curr_dim_f, 
            T* am, T *bm, T * ddist_r, T *dv,
            int lddv1, int lddv2, int queue_idx, int config) {
  
#define IPK(R, C, F, G) {ipk_3_adaptive_launcher<T, D, R, C, F, G>(\
                                handle,\
                                shape, shape_c, ldvs, ldws, processed_dims,\
                                curr_dim_r, curr_dim_c, curr_dim_f,\
                                am, bm, ddist_r, dv,\
                                lddv1, lddv2, queue_idx);}

  bool profile = false;
#ifdef MGARD_CUDA_KERNEL_PROFILE  
  profile = true;
#endif
  if (D >= 3) {
    if (profile || config == 6) { IPK(2, 2, 128, 2) }
    if (profile || config == 5) { IPK(2, 2, 64, 2) }
    if (profile || config == 4) { IPK(2, 2, 32, 2) }
    if (profile || config == 3) { IPK(2, 2, 16, 2) }
    if (profile || config == 2) { IPK(8, 8, 8, 4) }
    if (profile || config == 1) { IPK(4, 4, 4, 4) }
    if (profile || config == 0) { IPK(2, 2, 2, 2) }

    // IPK(8, 8, 8, 4); 
  } else {
    printf("Error: solve_tridiag_3_cpt is only for 3D data\n");
  }
#undef IPK
}

template <typename T, int D, int R, int C, int F, int G>
void ipk_3_adaptive_launcher( mgard_cuda_handle<T, D> &handle, 
                              int * shape_h, int * shape_c_h, int * shape_d, int * shape_c_d, 
                              int * ldvs, int * ldws,
                              int processed_n, int * processed_dims_h, int * processed_dims_d,
                              int curr_dim_r, int curr_dim_c, int curr_dim_f,
                              T* am, T *bm, T * ddist_r, T *dv,
                              int lddv1, int lddv2, int queue_idx) {

  // printf("am: ");
  // print_matrix_cuda(1, nr, am, nr);
  // printf("bm: ");
  // print_matrix_cuda(1, nr, bm, nr);

  int nr_c = shape_c_h[curr_dim_r];
  int nc_c = shape_c_h[curr_dim_c];
  int nf_c = shape_c_h[curr_dim_f];

  int total_thread_x = nf_c;
  int total_thread_y = nc_c;
  int total_thread_z = 1;
  int tbx, tby, tbz, gridx, gridy, gridz;
  dim3 threadsPerBlock, blockPerGrid;
  size_t sm_size;

  tbx = std::max(F, std::min(F, total_thread_x));
  tby = std::max(C, std::min(C, total_thread_y));
  tbz = 1; 
  sm_size = (C * F + 2) * (R + G) * sizeof(T); 
  sm_size += (D * 5) * sizeof(int); 
  gridx = ceil((float)total_thread_x / tbx); 
  gridy = ceil((float)total_thread_y / tby); 
  gridz = 1; 
  for (int d = 0; d < D; d++) {
    if (d != curr_dim_f && d != curr_dim_c && d != curr_dim_r) {
      int t = shape_h[d];
      for (int k = 0; k < processed_n; k++) {
        if (d == processed_dims_h[k]) {
          t = shape_c_h[d];
        }
      }
      gridx *= t;
    }
  }
  threadsPerBlock = dim3(tbx, tby, tbz); 
  blockPerGrid = dim3(gridx, gridy, gridz); 
  // printf("ipk_1 exec config (%d %d %d) (%d %d %d)\n", tbx, tby, tbz, gridx, gridy, gridz);
  _ipk_3<T, D, R, C, F, G><<<blockPerGrid, threadsPerBlock, sm_size, 
                                 *(cudaStream_t *)handle.get(queue_idx)>>>( 
                                  shape_d, shape_c_d, ldvs, ldws,
                                  processed_n, processed_dims_d,  
                                  curr_dim_r, curr_dim_c, curr_dim_f,   
                                  am, bm, ddist_r, dv, lddv1, lddv2); 
  gpuErrchk(cudaGetLastError()); 
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}


template <typename T, int D>
void ipk_3( mgard_cuda_handle<T, D> &handle, 
            int * shape_h, int * shape_c_h, int * shape_d, int * shape_c_d, 
            int * ldvs, int * ldws,
            int processed_n, int * processed_dims_h, int * processed_dims_d,
            int curr_dim_r, int curr_dim_c, int curr_dim_f, 
            T* am, T *bm, T * ddist_r, T *dv,
            int lddv1, int lddv2, int queue_idx, int config) {
  
#define IPK(R, C, F, G) {ipk_3_adaptive_launcher<T, D, R, C, F, G>(\
                                handle,\
                                shape_h, shape_c_h, shape_d, shape_c_d, ldvs, ldws, \
                                processed_n, processed_dims_h, processed_dims_d,\ 
                                curr_dim_r, curr_dim_c, curr_dim_f,\
                                am, bm, ddist_r, dv,\
                                lddv1, lddv2, queue_idx);}

  bool profile = false;
#ifdef MGARD_CUDA_KERNEL_PROFILE  
  profile = true;
#endif
  if (D >= 3) {
    if (profile || config == 6) { IPK(2, 2, 128, 2) }
    if (profile || config == 5) { IPK(2, 2, 64, 2) }
    if (profile || config == 4) { IPK(2, 2, 32, 2) }
    if (profile || config == 3) { IPK(2, 2, 16, 2) }
    if (profile || config == 2) { IPK(8, 8, 8, 4) }
    if (profile || config == 1) { IPK(4, 4, 4, 4) }
    if (profile || config == 0) { IPK(2, 2, 2, 2) }
      // IPK(2, 2, 64, 2)
  } else {
    printf("Error: solve_tridiag_3_cpt is only for 3D data\n");
  }
#undef IPK
}

}

#endif