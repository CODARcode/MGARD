/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#ifndef MGRAD_CUDA_LINEAR_QUANTIZATION_TEMPLATE
#define MGRAD_CUDA_LINEAR_QUANTIZATION_TEMPLATE

#include "CommonInternal.h"

#include "LinearQuantization.h"

namespace mgard_cuda {

template <DIM D, typename T>
void calc_quantizers(Handle<D, T> &handle, T *quantizers, Metadata m,
                     bool reciprocal) {

  double abs_tol = m.tol;
  if (m.ebtype == error_bound_type::REL) {
    abs_tol *= m.norm;
  }

  // printf("tol %f, l_target %d, D %d\n", tol, l_target, D);

  abs_tol *= 2;

  // original
  // tol /= l_target + 2;
  // for (int l = 0; l < l_target+1; l++) {
  //   quantizers[l] = tol;
  // }
  // printf("l_target %d\n", l_target);

  // levelwise
  // tol *= 2;
  // T C2 = 1 + 3*std::sqrt(3)/4;
  // T c = std::sqrt(std::pow(2, D));
  // T cc = (1 - c) / (1 - std::pow(c, l_target+1));
  // T level_eb = cc * tol / C2;

  // for (int l = 0; l < l_target+1; l++) {
  //   quantizers[l] = level_eb;
  //   level_eb *= c;
  // }

  // s = 0;

  // levelwise with s
  // tol *= 2;
  // T C2 = 1 + 3 * std::sqrt(3) / 4;
  // T c = std::sqrt(std::pow(2, D - 2 * s));
  // T cc = (1 - c) / (1 - std::pow(c, l_target + 1));
  // T level_eb = cc * tol / C2;

  // for (int l = 0; l < l_target + 1; l++) {
  //   quantizers[l] = level_eb;
  //   // T c = std::sqrt(std::pow(2, 2*s*l + D * (l_target - l)));
  //   level_eb *= c;
  //   if (reciprocal)
  //     quantizers[l] = 1.0f / quantizers[l];
  // }

  if (m.ntype == norm_type::L_Inf) {

    // printf("quantizers: ");
    for (int l = 0; l < m.l_target + 1; l++) {
      // ben
      quantizers[l] = (abs_tol) / ((m.l_target + 1) * (1 + std::pow(3, D)));
      // xin
      // quantizers[l] = (tol) / ((l_target + 1) * (1 + 3 * std::sqrt(3) / 4));

      // printf("%f ", quantizers[l]);
      if (reciprocal)
        quantizers[l] = 1.0f / quantizers[l];
    }
    // printf("\n");

  } else if (m.ntype == norm_type::L_2) { // s != inf
    // xin - uniform
    // T C2 = 1 + 3 * std::sqrt(3) / 4;
    // T c = std::sqrt(std::pow(2, D - 2 * s));
    // T cc = (1 - c) / (1 - std::pow(c, l_target + 1));
    // T level_eb = cc * tol / C2;
    // for (int l = 0; l < l_target + 1; l++) {
    //   quantizers[l] = level_eb;
    //   // T c = std::sqrt(std::pow(2, 2*s*l + D * (l_target - l)));
    //   level_eb *= c;
    //   if (reciprocal)
    //     quantizers[l] = 1.0f / quantizers[l];
    // }

    // ben - uniform
    // printf("quantizers: ");

    size_t dof = 1;
    for (int d = 0; d < D; d++)
      dof *= handle.dofs[d][0];
    // printf("tol: %f, dof: %llu\n", tol, dof);
    // printf ("dof = %llu\n", dof);
    for (int l = 0; l < m.l_target + 1; l++) {

      quantizers[l] = (abs_tol) / (std::exp2(m.s * l) * std::sqrt(dof));

      // printf("l %d, vol: %f quantizer: %f \n", l, std::pow(2, (l_target - l)
      // * D), quantizers[l]);

      // printf("tol: %f quant: %e \n", tol, quantizers[l]);
      if (reciprocal)
        quantizers[l] = 1.0f / quantizers[l];
    }
    // printf("\n");
  }

  // print quantizers
  // printf("quantizers: ");
  // for (int l = 0; l < l_target+1; l++) {
  //   printf("%f ", 1.0f/quantizers[l]);
  // }
  // printf("\n");
}

template <DIM D, typename T, int R, int C, int F, bool CALC_VOL>
__global__ void
_levelwise_linear_quantize(SIZE *shapes, SIZE l_target, T *quantizers,
                           T *volumes, SIZE ldvolumes, T *dv, SIZE *ldvs,
                           QUANTIZED_INT *dwork, SIZE *ldws, bool prep_huffmam,
                           SIZE dict_size, SIZE *shape, LENGTH *outlier_count,
                           LENGTH *outlier_idx, QUANTIZED_INT *outliers) {

  size_t threadId = (threadIdx.z * (blockDim.x * blockDim.y)) +
                    (threadIdx.y * blockDim.x) + threadIdx.x;
  T *smT = SharedMemory<T>();
  T *quantizers_sm = smT;
  smT += l_target + 1;

  T *volumes_0 = smT;
  if (CALC_VOL)
    smT += blockDim.x * (l_target + 1);
  T *volumes_1 = smT;
  if (CALC_VOL)
    smT += blockDim.y * (l_target + 1);
  T *volumes_2 = smT;
  if (CALC_VOL)
    smT += blockDim.z * (l_target + 1);
  T *volumes_3_plus = smT;
  if (CALC_VOL && D > 3)
    smT += (D - 3) * (l_target + 1);

  SIZE *smInt = (SIZE *)smT;
  SIZE *ldvs_sm = smInt;
  smInt += D;
  SIZE *ldws_sm = smInt;
  smInt += D;
  SIZE *shape_sm = smInt;
  smInt += D;
  SIZE *shapes_sm = smInt;
  smInt += D * (l_target + 2);

  if (threadId < l_target + 1) {
    quantizers_sm[threadId] = quantizers[threadId];
  }
  if (threadId < D) {
    ldvs_sm[threadId] = ldvs[threadId];
    ldws_sm[threadId] = ldws[threadId];
    shape_sm[threadId] = shape[threadId];
  }
  if (threadId < D * (l_target + 2)) {
    shapes_sm[threadId] = shapes[threadId];
    // printf ("D: %d l_target+2: %d load shapes[%llu]: %d\n", D, l_target+2,
    // threadId, shapes_sm[threadId]);
  }

  __syncthreads();

  // determine global idx
  SIZE idx[D];  // thread global idx
  SIZE idx0[D]; // block global idx

  SIZE firstD = div_roundup(shapes_sm[l_target + 1], F);

  SIZE bidx = blockIdx.x;
  idx[0] = (bidx % firstD) * F + threadIdx.x;
  idx0[0] = (bidx % firstD) * F;

  // printf("shapes_sm[l_target+1]: %d firstD %d idx[0] %d\n",
  // shapes_sm[l_target+1], firstD, idx[0]);

  bidx /= firstD;
  if (D >= 2) {
    idx[1] = blockIdx.y * blockDim.y + threadIdx.y;
    idx0[1] = blockIdx.y * blockDim.y;
  }
  if (D >= 3) {
    idx[2] = blockIdx.z * blockDim.z + threadIdx.z;
    idx0[2] = blockIdx.z * blockDim.z;
  }

  for (int d = 3; d < D; d++) {
    idx[d] = bidx % shapes_sm[(l_target + 2) * d + l_target + 1];
    idx0[d] = idx[d];
    bidx /= shapes_sm[(l_target + 2) * d + l_target + 1];
  }

  if (CALC_VOL) {
    // cache volumes
    for (int l = 0; l < l_target + 1; l++) {
      // volumes 0
      if (threadId < blockDim.x &&
          idx0[0] + threadId < shapes_sm[(l_target + 2) * 0 + l_target + 1]) {
        volumes_0[l * blockDim.x + threadId] =
            volumes[(0 * (l_target + 1) + l) * ldvolumes + idx0[0] + threadId];
        // printf("load %f\n", volumes[(0 * (l_target + 1) + l) * ldvolumes +
        // idx0[0] + threadId]);
      }
      if (D >= 2) {
        // volumes 1
        if (threadId < blockDim.y &&
            idx0[1] + threadId < shapes_sm[(l_target + 2) * 1 + l_target + 1]) {
          volumes_1[l * blockDim.y + threadId] =
              volumes[(1 * (l_target + 1) + l) * ldvolumes + idx0[1] +
                      threadId];
        }
      }
      if (D >= 3) {
        // volumes 2
        if (threadId < blockDim.z &&
            idx0[2] + threadId < shapes_sm[(l_target + 2) * 2 + l_target + 1]) {
          volumes_2[l * blockDim.z + threadId] =
              volumes[(2 * (l_target + 1) + l) * ldvolumes + idx0[2] +
                      threadId];
        }
      }
    }

    if (D >= 4) {
      if (threadId < 1) {
        for (int d = 3; d < D; d++) {
          for (int l = 0; l < l_target + 1; l++) {
            volumes_3_plus[(d - 3) * (l_target + 1) + l] =
                volumes[(d * (l_target + 1) + l) * ldvolumes + idx[d]];
          }
        }
      }
    }
  }

  // if (blockIdx.y == 0 && blockIdx.x == 0 && blockIdx.z == 0 && threadId == 0)
  // {
  //   printf("volumes_0: ");
  //   for (int l = 0; l < l_target+1; l++) {
  //     printf("l = %d\n", l);
  //     for (int i = 0; i < min(blockDim.x, shapes_sm[(l_target + 2) * 0 +
  //     l_target + 1]) ; i++) {
  //       printf("%f ", volumes_0[l * blockDim.x + i]);
  //     }
  //     printf("\n");
  //   }
  //   printf("\n");
  //   if (D >= 2) {
  //     printf("volumes_1: ");
  //     for (int l = 0; l < l_target+1; l++) {
  //       printf("l = %d\n", l);
  //       for (int i = 0; i < min(blockDim.y, shapes_sm[(l_target + 2) * 1 +
  //       l_target + 1]); i++) {
  //         printf("%f ", volumes_1[l * blockDim.y + i]);
  //       }
  //       printf("\n");
  //     }

  //     printf("\n");
  //   }
  //   if (D >= 3) {
  //     printf("volumes_2: ");
  //     for (int l = 0; l < l_target+1; l++) {
  //       printf("l = %d\n", l);
  //       for (int i = 0; i < min(blockDim.z, shapes_sm[(l_target + 2) * 2 +
  //       l_target + 1]); i++) {
  //         printf("%f ", volumes_2[l * blockDim.y + i]);
  //       }
  //       printf("\n");
  //     }
  //   }
  // }

  __syncthreads();

  int level = 0;
  for (DIM d = 0; d < D; d++) {
    long long unsigned int l_bit = 0l;
    for (SIZE l = 0; l < l_target + 1; l++) {
      int bit = (idx[d] >= shapes_sm[(l_target + 2) * d + l]) &&
                (idx[d] < shapes_sm[(l_target + 2) * d + l + 1]);
      l_bit += bit << l;
      // printf("idx: %d %d d: %d l_bit: %llu\n", idx[1], idx[0], d, l_bit);
    }
    level = max(level, __ffsll(l_bit));
  }
  level = level - 1;

  bool in_range = true;
  for (DIM d = 0; d < D; d++) {
    if (idx[d] >= shapes_sm[(l_target + 2) * d + l_target + 1])
      in_range = false;
  }

  // printf("idx %llu, level: %d, in_range: %d idx[0]: shape_sm: %d\n",
  // get_idx<D>(shape_sm, idx), level, in_range, shapes_sm[(l_target+2) * 0 +
  // l_target+1]);

  if (level >= 0 && level <= l_target && in_range) {
    T t = dv[get_idx<D>(ldvs, idx)];
    T volume = 1;
    if (CALC_VOL) {
      volume *= volumes_0[level * blockDim.x + threadIdx.x];
      if (D >= 2) {
        volume *= volumes_1[level * blockDim.y + threadIdx.y];
      }
      if (D >= 3) {
        volume *= volumes_2[level * blockDim.z + threadIdx.z];
      }
      if (D >= 4) {
        for (int d = 3; d < D; d++) {
          volume *= volumes_3_plus[(d - 3) * (l_target + 1) + level];
        }
      }
      if (sizeof(T) == sizeof(double))
        volume = sqrt(volume);
      else if (sizeof(T) == sizeof(float))
        volume = sqrtf(volume);
    }
    // printf("l: %d, vol %f(%f*%f*%f), quantizers_sm: %f, quantizers: %f,
    // before: %f, quantized: %d\n", level, volume,
    //   volumes_0[level * blockDim.x + threadIdx.x], volumes_1[level *
    //   blockDim.y + threadIdx.y], volumes_2[level * blockDim.z + threadIdx.z],
    //   quantizers_sm[level],
    //   (quantizers_sm[level] / volume), t, (int)copysign(0.5 + fabs(t /(
    //   quantizers_sm[level] / volume)), t));

    QUANTIZED_INT quantized_data =
        copysign(0.5 + fabs(t / (quantizers_sm[level] * volume)), t);
    // QUANTIZED_INT quantized_data = copysign(0.5 + fabs(t /
    // (quantizers_sm[level] / volume) ), t); printf("dv[%llu] %f
    // quantizers[%d]%f -> dw[%llu]%d \n",
    //       get_idx<D>(ldvs, idx), t,
    //       level, quantizers_sm[level],
    //       get_idx<D>(ldws, idx), quantized_data+dict_size / 2);

    if (prep_huffmam) {
      quantized_data += dict_size / 2;
      if (quantized_data >= 0 && quantized_data < dict_size) {
        // do nothing
      } else {
        LENGTH i = atomicAdd(outlier_count, (LENGTH)1);
        outlier_idx[i] = get_idx<D>(shape_sm, idx);
        outliers[i] = quantized_data;
        quantized_data = 0;
      }
      // if (get_idx<D>(shape_sm, idx) < quant_meta_size_ratio) {
      //   size_t i = atomicAdd((unsigned long long int*)outlier_count,
      //   (unsigned long long int)1); outlier_idx[i] = get_idx<D>(shape_sm,
      //   idx);
      // }
    }

    dwork[get_idx<D>(ldws, idx)] = quantized_data;
  }
}

template <DIM D, typename T, int R, int C, int F>
void levelwise_linear_quantize_adaptive_launcher(
    Handle<D, T> &handle, SIZE *shapes, SIZE l_target, T *volumes,
    SIZE ldvolumes, Metadata m, T *dv, SIZE *ldvs, QUANTIZED_INT *dwork,
    SIZE *ldws, bool prep_huffmam, SIZE *shape, LENGTH *outlier_count,
    LENGTH *outlier_idx, QUANTIZED_INT *outliers, int queue_idx) {

  T *quantizers = new T[l_target + 1];
  calc_quantizers(handle, quantizers, m, false);
  cudaMemcpyAsyncHelper(handle, handle.quantizers, quantizers,
                        sizeof(T) * (l_target + 1), H2D, queue_idx);
  // printf("norm: %f, tol: %f, s: %f, dict_size: %d\n", m.norm, m.tol, m.s,
  // m.dict_size);
  int total_thread_z = handle.dofs[2][0];
  int total_thread_y = handle.dofs[1][0];
  int total_thread_x = handle.dofs[0][0];
  // linearize other dimensions
  int tbz = R;
  int tby = C;
  int tbx = F;
  int gridz = ceil((float)total_thread_z / tbz);
  int gridy = ceil((float)total_thread_y / tby);
  int gridx = ceil((float)total_thread_x / tbx);
  for (int d = 3; d < D; d++) {
    gridx *= handle.dofs[d][0];
  }

  // printf("exec: %d %d %d %d %d %d\n", tbx, tby, tbz, gridx, gridy, gridz);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);
  // ldvs + ldws + shape
  size_t sm_size = (D * 3) * sizeof(SIZE);
  // quantizer
  sm_size += (l_target + 1) * sizeof(T);
  // ranges
  sm_size += (l_target + 2) * D * sizeof(SIZE);
  // volumes
  sm_size += tbx * (l_target + 1) * sizeof(T);
  sm_size += tby * (l_target + 1) * sizeof(T);
  sm_size += tbz * (l_target + 1) * sizeof(T);
  if (D > 3)
    sm_size += (D - 3) * (l_target + 1) * sizeof(T);
  // printf("sm_size: %llu\n", sm_size);
  if (m.ntype == norm_type::L_Inf) {
    _levelwise_linear_quantize<D, T, R, C, F, false>
        <<<blockPerGrid, threadsPerBlock, sm_size,
           *(cudaStream_t *)handle.get(queue_idx)>>>(
            shapes, l_target, handle.quantizers, volumes, ldvolumes, dv, ldvs,
            dwork, ldws, prep_huffmam, m.dict_size, shape, outlier_count,
            outlier_idx, outliers);
  } else if (m.ntype == norm_type::L_2) {
    _levelwise_linear_quantize<D, T, R, C, F, true>
        <<<blockPerGrid, threadsPerBlock, sm_size,
           *(cudaStream_t *)handle.get(queue_idx)>>>(
            shapes, l_target, handle.quantizers, volumes, ldvolumes, dv, ldvs,
            dwork, ldws, prep_huffmam, m.dict_size, shape, outlier_count,
            outlier_idx, outliers);
  } else {
    std::cout << log::log_err << "unsupported norm type!\n";
    exit(-1);
  }

  gpuErrchk(cudaGetLastError());
  if (handle.sync_and_check_all_kernels) {
    gpuErrchk(cudaDeviceSynchronize());
  }
}

template <DIM D, typename T>
void levelwise_linear_quantize(Handle<D, T> &handle, SIZE *shapes,
                               SIZE l_target, T *volumes, SIZE ldvolumes,
                               Metadata m, T *dv, SIZE *ldvs,
                               QUANTIZED_INT *dwork, SIZE *ldws,
                               bool prep_huffmam, SIZE *shape,
                               LENGTH *outlier_count, LENGTH *outlier_idx,
                               QUANTIZED_INT *outliers, int queue_idx) {
#define QUANTIZE(R, C, F)                                                      \
  {                                                                            \
    levelwise_linear_quantize_adaptive_launcher<D, T, R, C, F>(                \
        handle, shapes, l_target, volumes, ldvolumes, m, dv, ldvs, dwork,      \
        ldws, prep_huffmam, shape, outlier_count, outlier_idx, outliers,       \
        queue_idx);                                                            \
  }

  if (D >= 3) {
    QUANTIZE(4, 4, 16)
  }
  if (D == 2) {
    QUANTIZE(1, 4, 32)
  }
  if (D == 1) {
    QUANTIZE(1, 1, 64)
  }

#undef QUANTIZE
}

template <DIM D, typename T, int R, int C, int F, bool CALC_VOL>
__global__ void _levelwise_linear_dequantize(
    SIZE *shapes, SIZE l_target, T *quantizers, T *volumes, SIZE ldvolumes,
    QUANTIZED_INT *dv, SIZE *ldvs, T *dwork, SIZE *ldws, bool prep_huffmam,
    SIZE dict_size, LENGTH outlier_count, LENGTH *outlier_idx,
    QUANTIZED_INT *outliers) {

  LENGTH threadId = (threadIdx.z * (blockDim.x * blockDim.y)) +
                    (threadIdx.y * blockDim.x) + threadIdx.x;
  LENGTH blockId = (blockIdx.z * (gridDim.x * gridDim.y)) +
                   (blockIdx.y * gridDim.x) + blockIdx.x;
  LENGTH gloablId = blockId * blockDim.x * blockDim.y * blockDim.z + threadId;

  T *smT = SharedMemory<T>();
  T *quantizers_sm = smT;
  smT += l_target + 1;
  T *volumes_0 = smT;
  if (CALC_VOL)
    smT += blockDim.x * (l_target + 1);
  T *volumes_1 = smT;
  if (CALC_VOL)
    smT += blockDim.y * (l_target + 1);
  T *volumes_2 = smT;
  if (CALC_VOL)
    smT += blockDim.z * (l_target + 1);
  T *volumes_3_plus = smT;
  if (CALC_VOL && D > 3)
    smT += (D - 3) * (l_target + 1);

  SIZE *smInt = (SIZE *)smT;
  SIZE *ldvs_sm = smInt;
  smInt += D;
  SIZE *ldws_sm = smInt;
  smInt += D;
  SIZE *shape_sm = smInt;
  smInt += D;
  SIZE *shapes_sm = smInt;
  smInt += D * (l_target + 2);

  if (threadId < l_target + 1) {
    quantizers_sm[threadId] = quantizers[threadId];
  }
  if (threadId < D) {
    ldvs_sm[threadId] = ldvs[threadId];
    ldws_sm[threadId] = ldws[threadId];
  }
  if (threadId < D * (l_target + 2)) {
    shapes_sm[threadId] = shapes[threadId];
  }

  __syncthreads();

  // bool debug = false;
  // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
  //     threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
  //   debug = true;
  // for (int d = 0; d < D; d ++) {
  //   printf("shapes_sm[%d]\n", d);
  //   for (int l = 0; l < l_target + 1; l++) {
  //     printf("%d ", shapes_sm[(l_target+1) * d + l]);
  //   }
  //   printf("\n");
  // }
  // }
  // __syncthreads();

  // determine global idx
  SIZE idx[D];  // thread global idx
  SIZE idx0[D]; // block global idx

  SIZE firstD = div_roundup(shapes_sm[l_target + 1], F);

  SIZE bidx = blockIdx.x;
  idx[0] = (bidx % firstD) * F + threadIdx.x;
  idx0[0] = (bidx % firstD) * F;

  // printf("shapes_sm[l_target+1]: %d firstD %d idx[0] %d\n",
  // shapes_sm[l_target+1], firstD, idx[0]);

  bidx /= firstD;
  if (D >= 2) {
    idx[1] = blockIdx.y * blockDim.y + threadIdx.y;
    idx0[1] = blockIdx.y * blockDim.y;
  }
  if (D >= 3) {
    idx[2] = blockIdx.z * blockDim.z + threadIdx.z;
    idx0[2] = blockIdx.z * blockDim.z;
  }

  for (DIM d = 3; d < D; d++) {
    idx[d] = bidx % shapes_sm[(l_target + 2) * d + l_target + 1];
    idx0[d] = idx[d];
    bidx /= shapes_sm[(l_target + 2) * d + l_target + 1];
  }

  if (CALC_VOL) {
    // cache volumes
    for (SIZE l = 0; l < l_target + 1; l++) {
      // volumes 0
      if (threadId < blockDim.x &&
          idx0[0] + threadId < shapes_sm[(l_target + 2) * 0 + l_target + 1]) {
        // printf("%d < %d[%d, %d, %d]\n", idx0[0] + (int)threadId,
        //   shapes_sm[(l_target + 2) * 0 + l_target + 1],
        //   l_target, (l_target + 2) * 0 + l_target + 1, l_target + 2);
        volumes_0[l * blockDim.x + threadId] =
            volumes[(0 * (l_target + 1) + l) * ldvolumes + idx0[0] + threadId];
        // printf("load %f\n", volumes_0[l * blockDim.x + threadId]);
      }
      if (D >= 2) {
        // volumes 1
        if (threadId < blockDim.y &&
            idx0[1] + threadId < shapes_sm[(l_target + 2) * 1 + l_target + 1]) {
          volumes_1[l * blockDim.y + threadId] =
              volumes[(1 * (l_target + 1) + l) * ldvolumes + idx0[1] +
                      threadId];
        }
      }
      if (D >= 3) {
        // volumes 2
        if (threadId < blockDim.z &&
            idx0[2] + threadId < shapes_sm[(l_target + 2) * 2 + l_target + 1]) {
          volumes_2[l * blockDim.z + threadId] =
              volumes[(2 * (l_target + 1) + l) * ldvolumes + idx0[2] +
                      threadId];
        }
      }
    }

    if (D >= 4) {
      if (threadId < 1) {
        for (DIM d = 3; d < D; d++) {
          for (SIZE l = 0; l < l_target + 1; l++) {
            volumes_3_plus[(d - 3) * (l_target + 1) + l] =
                volumes[(d * (l_target + 1) + l) * ldvolumes + idx[d]];
          }
        }
      }
    }
  }

  // if (blockIdx.y == 0 && blockIdx.x == 0 && threadId == 0) {
  //   printf("volumes_0: ");
  //   for (int l = 0; l < l_target+1; l++) {
  //     printf("l = %d\n", l);
  //     for (int i = 0; i < min(blockDim.x, shapes_sm[(l_target + 2) * 0 +
  //     l_target + 1]) ; i++) {
  //       printf("%f ", volumes_0[l * blockDim.x + i]);
  //     }
  //     printf("\n");
  //   }
  //   printf("\n");
  //   printf("volumes_1: ");
  //   for (int l = 0; l < l_target+1; l++) {
  //     printf("l = %d\n", l);
  //     for (int i = 0; i < min(blockDim.y, shapes_sm[(l_target + 2) * 1 +
  //     l_target + 1]); i++) {
  //       printf("%f ", volumes_1[l * blockDim.y + i]);
  //     }
  //     printf("\n");
  //   }

  // }

  __syncthreads();

  int level = 0;
  for (DIM d = 0; d < D; d++) {
    long long unsigned int l_bit = 0l;
    for (SIZE l = 0; l < l_target + 1; l++) {
      int bit = (idx[d] >= shapes_sm[(l_target + 2) * d + l]) &&
                (idx[d] < shapes_sm[(l_target + 2) * d + l + 1]);
      l_bit += bit << l;
      // printf("idx: %d %d d: %d l_bit: %llu\n", idx[1], idx[0], d, l_bit);
    }
    level = max(level, __ffsll(l_bit));
  }

  bool in_range = true;
  for (DIM d = 0; d < D; d++) {
    if (idx[d] >= shapes_sm[(l_target + 2) * d + l_target + 1])
      in_range = false;
  }

  level = level - 1;
  if (level >= 0 && level <= l_target && in_range) {
    // printf("%d %d %d %d\n", idx[3], idx[2], idx[1], idx[0]);
    // printf("idx: %d %d l: %d\n", idx[1], idx[0], level);
    QUANTIZED_INT quantized_data = dv[get_idx<D>(ldvs, idx)];
    T volume = 1;
    if (CALC_VOL) {
      volume *= volumes_0[level * blockDim.x + threadIdx.x];
      if (D >= 2)
        volume *= volumes_1[level * blockDim.y + threadIdx.y];
      if (D >= 3)
        volume *= volumes_2[level * blockDim.z + threadIdx.z];
      if (D >= 4) {
        for (int d = 3; d < D; d++) {
          volume *= volumes_3_plus[(d - 3) * (l_target + 1) + level];
        }
      }
      if (sizeof(T) == sizeof(double))
        volume = sqrt(volume);
      else if (sizeof(T) == sizeof(float))
        volume = sqrtf(volume);
    }

    if (prep_huffmam) {
      quantized_data -= dict_size / 2;
    }

    // printf("%d %d %d %d %d %d vol %f (%f * %f * %f), dequantizers: %f,
    // before: %d, dequantized: %f\n", blockIdx.z, blockIdx.y, blockIdx.x,
    // threadIdx.z, threadIdx.y, threadIdx.x, volume,
    //   volumes_0[level * blockDim.x + threadIdx.x], volumes_1[level *
    //   blockDim.y + threadIdx.y], volumes_2[level * blockDim.z + threadIdx.z],
    //   quantizers_sm[level] / volume, quantized_data, (quantizers_sm[level] /
    //   volume) * (T)quantized_data);
    dwork[get_idx<D>(ldws, idx)] =
        (quantizers_sm[level] * volume) * (T)quantized_data;
    // dwork[get_idx<D>(ldws, idx)] = (quantizers_sm[level] / volume) *
    // (T)quantized_data; dwork[get_idx<D>(ldws, idx)] = (T)dv[get_idx<D>(ldvs,
    // idx)];

    // printf("dw[%llu] %d dequantizers[%d]%f -> dw[%llu]%f \n",
    // get_idx<D>(ldvs, idx),
    //       quantized_data, level, quantizers_sm[level], get_idx<D>(ldws, idx),
    //       quantizers_sm[level] * (T)quantized_data);
  }

  // //outliers
  // if (gloablId < outlier_count) {
  //   size_t linerized_idx = outlier_idx[gloablId];
  //   for (int d = 0; d < D; d++) {
  //     idx[d] = linerized_idx % shapes_sm[(l_target+2) * d+l_target+1];
  //     linerized_idx /= shapes_sm[(l_target+2) * d+l_target+1];
  //   }
  //   int outliter = outliers[gloablId];
  //   outliter -= dict_size / 2;

  //   level = 0;
  //   for (int d = 0; d < D; d++) {
  //     long long unsigned int l_bit = 0l;
  //     for (int l = 0; l < l_target+1; l++) {
  //       int bit = (idx[d] >= shapes_sm[(l_target+2) * d + l]) && (idx[d] <
  //       shapes_sm[(l_target+2) * d + l+1]); l_bit += bit << l;
  //       // printf("idx: %d %d d: %d l_bit: %llu\n", idx[1], idx[0], d,
  //       l_bit);
  //     }
  //     level = max(level, __ffsll(l_bit));
  //   }
  //   level = level - 1;

  //   dwork[get_idx<D>(ldws, idx)] = quantizers_sm[level] * (T)outliter;

  //   // printf("outliter: dw[%llu] %d dequantizers[%d]%f -> dw[%llu]%f \n",
  //   get_idx<D>(ldvs, idx),
  //   //       outliter, level, quantizers_sm[level], get_idx<D>(ldws, idx),
  //   quantizers_sm[level] * (T)outliter);

  // }
}

template <DIM D, typename T, int R, int C, int F, bool CALC_VOL>
__global__ void _levelwise_linear_dequantize_outliers(
    SIZE *shapes, SIZE l_target, T *quantizers, T *volumes, SIZE ldvolumes,
    QUANTIZED_INT *dv, SIZE *ldvs, T *dwork, SIZE *ldws, SIZE dict_size,
    LENGTH outlier_count, LENGTH *outlier_idx, QUANTIZED_INT *outliers) {

  size_t threadId = (threadIdx.z * (blockDim.x * blockDim.y)) +
                    (threadIdx.y * blockDim.x) + threadIdx.x;
  size_t blockId = (blockIdx.z * (gridDim.x * gridDim.y)) +
                   (blockIdx.y * gridDim.x) + blockIdx.x;
  size_t gloablId = blockId * blockDim.x * blockDim.y * blockDim.z + threadId;

  T *sm = SharedMemory<T>();
  T *quantizers_sm = sm;
  sm += l_target + 1;

  SIZE *sm_size = (SIZE *)sm;
  SIZE *ldvs_sm = sm_size;
  sm_size += D;
  SIZE *ldws_sm = sm_size;
  sm_size += D;
  SIZE *shapes_sm = sm_size;
  sm_size += D * (l_target + 2);

  if (threadId < l_target + 1) {
    quantizers_sm[threadId] = quantizers[threadId];
  }
  if (threadId < D) {
    ldvs_sm[threadId] = ldvs[threadId];
    ldws_sm[threadId] = ldws[threadId];
  }
  if (threadId < D * (l_target + 2)) {
    shapes_sm[threadId] = shapes[threadId];
  }

  __syncthreads();
  SIZE idx[D]; // thread global idx

  // outliers
  if (gloablId < outlier_count) {
    size_t linerized_idx = outlier_idx[gloablId];
    for (DIM d = 0; d < D; d++) {
      idx[d] = linerized_idx % shapes_sm[(l_target + 2) * d + l_target + 1];
      linerized_idx /= shapes_sm[(l_target + 2) * d + l_target + 1];
    }
    QUANTIZED_INT outliter = outliers[gloablId];
    outliter -= dict_size / 2;

    int level = 0;
    for (DIM d = 0; d < D; d++) {
      long long unsigned int l_bit = 0l;
      for (SIZE l = 0; l < l_target + 1; l++) {
        int bit = (idx[d] >= shapes_sm[(l_target + 2) * d + l]) &&
                  (idx[d] < shapes_sm[(l_target + 2) * d + l + 1]);
        l_bit += bit << l;
        // printf("idx: %d %d d: %d l_bit: %llu\n", idx[1], idx[0], d, l_bit);
      }
      level = max(level, __ffsll(l_bit));
    }
    level = level - 1;

    T volume = 1;

    if (CALC_VOL) {
      for (DIM d = 0; d < D; d++) {
        volume *= volumes[(d * (l_target + 1) + level) * ldvolumes + idx[d]];
      }
      if (sizeof(T) == sizeof(double))
        volume = sqrt(volume);
      else if (sizeof(T) == sizeof(float))
        volume = sqrtf(volume);
    }
    dwork[get_idx<D>(ldws, idx)] =
        (quantizers_sm[level] * volume) * (T)outliter;
    // dwork[get_idx<D>(ldws, idx)] = (quantizers_sm[level] / volume) *
    // (T)outliter;

    // printf("outliter: dw[%llu] %d dequantizers[%d]%f -> dw[%llu]%f \n",
    // get_idx<D>(ldvs, idx),
    //       outliter, level, quantizers_sm[level], get_idx<D>(ldws, idx),
    //       quantizers_sm[level] * (T)outliter);
  }
}

template <DIM D, typename T, int R, int C, int F>
void levelwise_linear_dequantize_adaptive_launcher(
    Handle<D, T> &handle, SIZE *shapes, SIZE l_target, T *volumes,
    SIZE ldvolumes, Metadata m, QUANTIZED_INT *dv, SIZE *ldvs, T *dwork,
    SIZE *ldws, bool prep_huffman, LENGTH outlier_count, LENGTH *outlier_idx,
    QUANTIZED_INT *outliers, int queue_idx) {

  // printf("norm: %f, tol: %f, s: %f, dict_size: %d\n", m.norm, m.tol, m.s,
  // m.dict_size);

  T *quantizers = new T[l_target + 1];
  calc_quantizers(handle, quantizers, m, false);
  cudaMemcpyAsyncHelper(handle, handle.quantizers, quantizers,
                        sizeof(T) * (l_target + 1), H2D, queue_idx);

  SIZE total_thread_z = handle.dofs[2][0];
  SIZE total_thread_y = handle.dofs[1][0];
  SIZE total_thread_x = handle.dofs[0][0];
  // linearize other dimensions
  SIZE tbz = R;
  SIZE tby = C;
  SIZE tbx = F;
  SIZE gridz = ceil((float)total_thread_z / tbz);
  SIZE gridy = ceil((float)total_thread_y / tby);
  SIZE gridx = ceil((float)total_thread_x / tbx);
  for (DIM d = 3; d < D; d++) {
    gridx *= handle.dofs[d][0];
  }

  // printf("exec: %d %d %d %d %d %d\n", tbx, tby, tbz, gridx, gridy, gridz);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);
  size_t sm_size = (D * 3) * sizeof(SIZE);
  sm_size += (l_target + 1) * sizeof(T);
  sm_size += (l_target + 2) * D * sizeof(SIZE);
  sm_size += tbx * (l_target + 1) * sizeof(T);
  sm_size += tby * (l_target + 1) * sizeof(T);
  sm_size += tbz * (l_target + 1) * sizeof(T);
  if (D > 3)
    sm_size += (D - 3) * (l_target + 1) * sizeof(T);

  if (m.ntype == norm_type::L_Inf) {
    _levelwise_linear_dequantize<D, T, R, C, F, false>
        <<<blockPerGrid, threadsPerBlock, sm_size,
           *(cudaStream_t *)handle.get(queue_idx)>>>(
            shapes, l_target, handle.quantizers, volumes, ldvolumes, dv, ldvs,
            dwork, ldws, prep_huffman, m.dict_size, outlier_count, outlier_idx,
            outliers);
    if (prep_huffman) {
      _levelwise_linear_dequantize_outliers<D, T, R, C, F, false>
          <<<blockPerGrid, threadsPerBlock, sm_size,
             *(cudaStream_t *)handle.get(queue_idx)>>>(
              shapes, l_target, handle.quantizers, volumes, ldvolumes, dv, ldvs,
              dwork, ldws, m.dict_size, outlier_count, outlier_idx, outliers);
    }
  } else if (m.ntype == norm_type::L_2) {
    _levelwise_linear_dequantize<D, T, R, C, F, true>
        <<<blockPerGrid, threadsPerBlock, sm_size,
           *(cudaStream_t *)handle.get(queue_idx)>>>(
            shapes, l_target, handle.quantizers, volumes, ldvolumes, dv, ldvs,
            dwork, ldws, prep_huffman, m.dict_size, outlier_count, outlier_idx,
            outliers);
    if (prep_huffman) {
      _levelwise_linear_dequantize_outliers<D, T, R, C, F, true>
          <<<blockPerGrid, threadsPerBlock, sm_size,
             *(cudaStream_t *)handle.get(queue_idx)>>>(
              shapes, l_target, handle.quantizers, volumes, ldvolumes, dv, ldvs,
              dwork, ldws, m.dict_size, outlier_count, outlier_idx, outliers);
    }
  } else {
    std::cout << log::log_err << "unsupported norm type!\n";
    exit(-1);
  }
  gpuErrchk(cudaGetLastError());
  if (handle.sync_and_check_all_kernels) {
    gpuErrchk(cudaDeviceSynchronize());
  }
}

template <DIM D, typename T>
void levelwise_linear_dequantize(Handle<D, T> &handle, SIZE *shapes,
                                 SIZE l_target, T *volumes, SIZE ldvolumes,
                                 Metadata m, QUANTIZED_INT *dv, SIZE *ldvs,
                                 T *dwork, SIZE *ldws, bool prep_huffmam,
                                 LENGTH outlier_count, LENGTH *outlier_idx,
                                 QUANTIZED_INT *outliers, int queue_idx) {
#define DEQUANTIZE(R, C, F)                                                    \
  {                                                                            \
    levelwise_linear_dequantize_adaptive_launcher<D, T, R, C, F>(              \
        handle, shapes, l_target, volumes, ldvolumes, m, dv, ldvs, dwork,      \
        ldws, prep_huffmam, outlier_count, outlier_idx, outliers, queue_idx);  \
  }

  if (D >= 3) {
    DEQUANTIZE(4, 4, 16)
  }
  if (D == 2) {
    DEQUANTIZE(1, 4, 32)
  }
  if (D == 1) {
    DEQUANTIZE(1, 1, 64)
  }

#undef DEQUANTIZE
}

} // namespace mgard_cuda

#endif