#include <chrono>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda/Common.h"
#include "cuda/CommonInternal.h"

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

#include "cuda/GridProcessingKernel.h"
#include "cuda/GridProcessingKernel3D.h"
#include "cuda/IterativeProcessingKernel.h"
#include "cuda/IterativeProcessingKernel3D.h"
#include "cuda/LevelwiseProcessingKernel.h"
#include "cuda/LinearProcessingKernel.h"
#include "cuda/LinearProcessingKernel3D.h"

#include "cuda/DataRefactoring.h"
#include "cuda/LinearQuantization.h"
// #include "cuda/Testing/ReorderToolsGPU.hpp"
// #include "cuda/Testing/ReorderToolsCPU.hpp"

enum data_type { SINGLE, DOUBLE };

template <mgard_x::DIM D, typename T>
T *generate_data(std::vector<mgard_x::SIZE> shape) {
  mgard_x::LENGTH linearized_n = 1;
  for (mgard_x::DIM d = 0; d < D; d++)
    linearized_n *= shape[d];
  T *in_buff = new T[linearized_n];
  for (int i = 0; i < linearized_n; i++) {
    in_buff[i] = rand() % 10 + 1;
  }
  return in_buff;
}

template <mgard_x::DIM D, typename T>
void test_reorder(std::vector<mgard_x::SIZE> shape) {
  T *original = generate_data<D, T>(shape);
  mgard_x::Handle<D, T> handle(shape);
  mgard_x::Array<D, T> gpu_original(shape);
  mgard_x::Array<D, T> gpu_reordered(shape);
  mgard_x::Array<D, T> gpu_restored(shape);
  gpu_original.loadData(original);
  mgard_x::SubArray org_array(gpu_original);
  mgard_x::SubArray reo_array(gpu_reordered);
  mgard_x::SubArray rev_array(gpu_restored);
  // ReorderGPU(handle, org_array, reo_array, 0);
  // ReverseReorderGPU(handle, reo_array, rev_array, 0);
  printf("org_array:\n");
  mgard_x::print_matrix_cuda(shape[1], shape[0], org_array.dv,
                             org_array.ldvs_h[0]);
  printf("reo_array:\n");
  mgard_x::print_matrix_cuda(shape[1], shape[0], reo_array.dv,
                             reo_array.ldvs_h[0]);
  printf("rev_array:\n");
  mgard_x::print_matrix_cuda(shape[1], shape[0], rev_array.dv,
                             rev_array.ldvs_h[0]);
}

template <mgard_x::DIM D, typename T>
void test_quantization(std::vector<mgard_x::SIZE> shape) {
  T *original = generate_data<D, T>(shape);
  mgard_x::Handle<D, T> handle(shape);
  mgard_x::Array<D, T> gpu_original(shape);
  mgard_x::Array<D, T> gpu_reordered(shape);
  mgard_x::Array<D, T> gpu_restored(shape);
  gpu_original.loadData(original);
  mgard_x::SubArray org_array(gpu_original);
  mgard_x::SubArray reo_array(gpu_reordered);
  mgard_x::SubArray rev_array(gpu_restored);
  // ReorderGPU(handle, org_array, reo_array, 0);
  printf("org_array:\n");
  mgard_x::print_matrix_cuda(shape[1], shape[0], org_array.dv,
                             org_array.ldvs_h[0]);
  printf("reo_array:\n");
  mgard_x::print_matrix_cuda(shape[1], shape[0], reo_array.dv,
                             reo_array.ldvs_h[0]);

  mgard_x::LENGTH quantized_count =
      handle.dofs[0][0] * handle.dofs[1][0] * handle.linearized_depth;
  mgard_x::QUANTIZED_INT *dqv;
  mgard_x::cudaMallocHelper(
      (void **)&dqv,
      (handle.dofs[0][0] * handle.dofs[1][0] * handle.linearized_depth) *
          sizeof(mgard_x::QUANTIZED_INT));

  std::vector<mgard_x::SIZE> ldqvs_h(handle.D_padded);
  ldqvs_h[0] = handle.dofs[0][0];
  for (int i = 1; i < handle.D_padded; i++) {
    ldqvs_h[i] = handle.dofs[i][0];
  }
  mgard_x::SIZE *ldqvs_d;
  mgard_x::cudaMallocHelper((void **)&ldqvs_d,
                            handle.D_padded * sizeof(mgard_x::SIZE));
  mgard_x::cudaMemcpyAsyncHelper(handle, ldqvs_d, ldqvs_h.data(),
                                 handle.D_padded * sizeof(mgard_x::SIZE),
                                 mgard_x::H2D, 0);

  mgard_x::quant_meta<T> m;
  m.norm = 1;
  m.s = 0;
  m.tol = 0.1;
  m.dict_size = 4096;
  m.enable_lz4 = true;
  m.l_target = handle.l_target;
  m.gpu_lossless = false;
  bool prep_huffman = false;

  mgard_x::LENGTH estimate_outlier_count = (double)handle.dofs[0][0] *
                                           handle.dofs[1][0] *
                                           handle.linearized_depth * 1;
  // printf("estimate_outlier_count: %llu\n", estimate_outlier_count);
  // mgard_x::LENGTH *outlier_count_d;
  // mgard_x::LENGTH *outlier_idx_d;
  // mgard_x::QUANTIZED_INT *outliers;
  // mgard_x::cudaMallocHelper((void **)&outliers, estimate_outlier_count *
  // sizeof(mgard_x::QUANTIZED_INT)); mgard_x::cudaMallocHelper((void
  // **)&outlier_count_d, sizeof(mgard_x::LENGTH));
  // mgard_x::cudaMallocHelper((void **)&outlier_idx_d,
  //                  estimate_outlier_count * sizeof(mgard_x::LENGTH));
  // mgard_x::LENGTH zero = 0, outlier_count, *outlier_idx_h;
  // mgard_x::cudaMemcpyAsyncHelper(handle, outlier_count_d, &zero,
  // sizeof(mgard_x::LENGTH), mgard_x::H2D, 0);

  // mgard_x::levelwise_linear_quantize<D, T>(
  //     handle, handle.ranges_d, handle.l_target, handle.volumes,
  //     handle.ldvolumes, m, reo_array.dv, reo_array.ldvs_d, dqv, ldqvs_d,
  //     prep_huffman, handle.shapes_d[0], outlier_count_d, outlier_idx_d,
  //     outliers, 0);

  // printf("before quantization:\n");
  // mgard_x::print_matrix_cuda(shape[1], shape[0],
  //   reo_array.dv, handle.dofs[0][0]);

  // printf("after quantization:\n");
  // mgard_x::print_matrix_cuda(shape[1], shape[0],
  //   dqv,  handle.dofs[0][0]);

  // T * cpu_reordered = new T[handle.dofs[0][0] * handle.dofs[1][0] *
  // handle.linearized_depth]; std::array<std::size_t, D> array_shape;
  // std::copy(shape.begin(), shape.end(), array_shape.begin());
  // mgard::TensorMeshHierarchy<D, T> hierarchy(array_shape)
  // mgard::shuffle(hierarchy, original, cpu_reordered);

  // using Qntzr = mgard::TensorMultilevelCoefficientQuantizer<D, T, long int>;
  // const Qntzr quantizer(hierarchy, s, tolerance);
  // using It = typename Qntzr::iterator;

  // const mgard::RangeSlice<It> quantized_range = quantizer(u);
  // // printf("get RangeSlice\n");
  // const std::vector<long int> quantized(quantized_range.begin(),
  //                                            quantized_range.end());
}

template <int D, typename T>
void tests_dimensions(std::vector<mgard_x::SIZE> shape) {
  // test_reorder<D, T>(shape);
  test_quantization<D, T>(shape);
}

template <typename T>
void tests_precision(int D, std::vector<mgard_x::SIZE> shape) {
  if (D == 1) {
    tests_dimensions<1, T>(shape);
  }
  if (D == 2) {
    tests_dimensions<2, T>(shape);
  }
  if (D == 3) {
    tests_dimensions<3, T>(shape);
  }
  if (D == 4) {
    tests_dimensions<4, T>(shape);
  }
  if (D == 5) {
    tests_dimensions<5, T>(shape);
  }
}

int main(int argc, char *argv[]) {

  enum data_type dtype;
  int i = 1;
  char *dt = argv[i++];
  if (strcmp(dt, "s") == 0)
    dtype = SINGLE;
  else if (strcmp(dt, "d") == 0)
    dtype = DOUBLE;

  std::vector<mgard_x::SIZE> shape;
  int D = atoi(argv[i++]);
  for (mgard_x::DIM d = 0; d < D; d++) {
    shape.push_back(atoi(argv[i++]));
  }

  if (dtype == SINGLE) {
    tests_precision<float>(D, shape);
  } else if (dtype == DOUBLE) {
    tests_precision<double>(D, shape);
  }
  return 0;
}