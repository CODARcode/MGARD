/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <vector>

#include "cuda/CommonInternal.h"

#include "cuda/CompressionWorkflow.h"

#include "cuda/MemoryManagement.h"

#include "cuda/DataRefactoring.h"
#include "cuda/LinearQuantization.h"
#include "cuda/LosslessCompression.h"

namespace mgard_cuda {

bool verify(const void *compressed_data, size_t compressed_size) {
  char signature[SIGNATURE_SIZE + 1];
  if (compressed_size < sizeof(signature))
    return false;
  SIZE meta_size = *(SIZE *)compressed_data;
  Metadata meta;
  meta.Deserialize((SERIALIZED_TYPE *)compressed_data, meta_size);
  std::memcpy(signature, meta.signature, SIGNATURE_SIZE);
  signature[SIGNATURE_SIZE] = '\0';
  if (strcmp(signature, SIGNATURE) == 0) {
    return true;
  } else {
    return false;
  }
}

enum data_type infer_type(const void *compressed_data, size_t compressed_size) {
  if (!verify(compressed_data, compressed_size)) {
    std::cout << mgard_cuda::log::log_err << "cannot verify the data!\n";
    exit(-1);
  }
  SIZE meta_size = *(SIZE *)compressed_data;
  Metadata meta;
  meta.Deserialize((SERIALIZED_TYPE *)compressed_data, meta_size);
  return meta.dtype;
}

std::vector<SIZE> infer_shape(const void *compressed_data,
                              size_t compressed_size) {
  if (!verify(compressed_data, compressed_size)) {
    std::cout << mgard_cuda::log::log_err << "cannot verify the data!\n";
    exit(-1);
  }
  SIZE meta_size = *(SIZE *)compressed_data;
  Metadata meta;
  meta.Deserialize((SERIALIZED_TYPE *)compressed_data, meta_size);
  std::vector<SIZE> shape(meta.total_dims);
  for (DIM d = 0; d < meta.total_dims; d++) {
    shape[d] = (SIZE)(*(meta.shape + d));
  }
  return shape;
}

template <DIM D, typename T>
void compress(std::vector<SIZE> shape, T tol, T s, enum error_bound_type mode,
              const void *original_data, void *&compressed_data,
              size_t &compressed_size, Config config, bool isAllocated) {
  Handle<D, T> handle(shape, config);
  mgard_cuda::Array<D, T> in_array(shape);
  in_array.loadData((const T *)original_data);
  Array<1, unsigned char> compressed_array =
      compress(handle, in_array, mode, tol, s);
  compressed_size = compressed_array.getShape()[0];
  if (!isAllocated)
    compressed_data = (void *)std::malloc(compressed_size);
  std::memcpy(compressed_data, compressed_array.getDataHost(), compressed_size);
}

template <DIM D, typename T>
void decompress(std::vector<SIZE> shape, const void *compressed_data,
                size_t compressed_size, void *&decompressed_data, Config config,
                bool isAllocated) {
  size_t original_size = 1;
  for (int i = 0; i < D; i++)
    original_size *= shape[i];
  Handle<D, T> handle(shape, config);
  std::vector<SIZE> compressed_shape(1);
  compressed_shape[0] = compressed_size;
  Array<1, unsigned char> compressed_array(compressed_shape);
  compressed_array.loadData((const unsigned char *)compressed_data);
  Array<D, T> out_array = decompress(handle, compressed_array);
  if (!isAllocated)
    decompressed_data = (void *)std::malloc(original_size * sizeof(T));
  std::memcpy(decompressed_data, out_array.getDataHost(),
              original_size * sizeof(T));
}

void compress(std::vector<SIZE> shape, data_type T, double tol, double s,
              enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size, Config config,
              bool isAllocated) {
  if (shape.size() == 1) {
    if (T == data_type::Double) {
      mgard_cuda::compress<1, double>(shape, tol, s, mode, original_data,
                                      compressed_data, compressed_size, config,
                                      isAllocated);
    } else {
      mgard_cuda::compress<1, float>(shape, tol, s, mode, original_data,
                                     compressed_data, compressed_size, config,
                                     isAllocated);
    }
  } else if (shape.size() == 2) {
    if (T == data_type::Double) {
      mgard_cuda::compress<2, double>(shape, tol, s, mode, original_data,
                                      compressed_data, compressed_size, config,
                                      isAllocated);
    } else {
      mgard_cuda::compress<2, float>(shape, tol, s, mode, original_data,
                                     compressed_data, compressed_size, config,
                                     isAllocated);
    }
  } else if (shape.size() == 3) {
    if (T == data_type::Double) {
      mgard_cuda::compress<3, double>(shape, tol, s, mode, original_data,
                                      compressed_data, compressed_size, config,
                                      isAllocated);
    } else {
      mgard_cuda::compress<3, float>(shape, tol, s, mode, original_data,
                                     compressed_data, compressed_size, config,
                                     isAllocated);
    }
  } else if (shape.size() == 4) {
    if (T == data_type::Double) {
      mgard_cuda::compress<4, double>(shape, tol, s, mode, original_data,
                                      compressed_data, compressed_size, config,
                                      isAllocated);
    } else {
      mgard_cuda::compress<4, float>(shape, tol, s, mode, original_data,
                                     compressed_data, compressed_size, config,
                                     isAllocated);
    }
  } else if (shape.size() == 5) {
    if (T == data_type::Double) {
      mgard_cuda::compress<5, double>(shape, tol, s, mode, original_data,
                                      compressed_data, compressed_size, config,
                                      isAllocated);
    } else {
      mgard_cuda::compress<5, float>(shape, tol, s, mode, original_data,
                                     compressed_data, compressed_size, config,
                                     isAllocated);
    }
  }
}

void decompress(const void *compressed_data, size_t compressed_size,
                void *&decompressed_data, Config config, bool isAllocated) {
  enum data_type T = infer_type(compressed_data, compressed_size);
  std::vector<SIZE> shape = infer_shape(compressed_data, compressed_size);
  if (shape.size() == 1) {
    if (T == data_type::Double) {
      mgard_cuda::decompress<1, double>(shape, compressed_data, compressed_size,
                                        decompressed_data, config, isAllocated);
    } else {
      mgard_cuda::decompress<1, float>(shape, compressed_data, compressed_size,
                                       decompressed_data, config, isAllocated);
    }
  } else if (shape.size() == 2) {
    if (T == data_type::Double) {
      mgard_cuda::decompress<2, double>(shape, compressed_data, compressed_size,
                                        decompressed_data, config, isAllocated);
    } else {
      mgard_cuda::decompress<2, float>(shape, compressed_data, compressed_size,
                                       decompressed_data, config, isAllocated);
    }
  } else if (shape.size() == 3) {
    if (T == data_type::Double) {
      mgard_cuda::decompress<3, double>(shape, compressed_data, compressed_size,
                                        decompressed_data, config, isAllocated);
    } else {
      mgard_cuda::decompress<3, float>(shape, compressed_data, compressed_size,
                                       decompressed_data, config, isAllocated);
    }
  } else if (shape.size() == 4) {
    if (T == data_type::Double) {
      mgard_cuda::decompress<4, double>(shape, compressed_data, compressed_size,
                                        decompressed_data, config, isAllocated);
    } else {
      mgard_cuda::decompress<4, float>(shape, compressed_data, compressed_size,
                                       decompressed_data, config, isAllocated);
    }
  } else if (shape.size() == 5) {
    if (T == data_type::Double) {
      mgard_cuda::decompress<5, double>(shape, compressed_data, compressed_size,
                                        decompressed_data, config, isAllocated);
    } else {
      mgard_cuda::decompress<5, float>(shape, compressed_data, compressed_size,
                                       decompressed_data, config, isAllocated);
    }
  }
}

#define KERNELS(D, T)                                                          \
  template void compress<D, T>(                                                \
      std::vector<SIZE> shape, T tol, T s, enum error_bound_type mode,         \
      const void *original_data, void *&compressed_data,                       \
      size_t &compressed_size, Config config, bool isAllocated);               \
  template void decompress<D, T>(                                              \
      std::vector<SIZE> shape, const void *compressed_data,                    \
      size_t compressed_size, void *&decompressed_data, Config config,         \
      bool isAllocated);

KERNELS(1, double)
KERNELS(1, float)
KERNELS(2, double)
KERNELS(2, float)
KERNELS(3, double)
KERNELS(3, float)
KERNELS(4, double)
KERNELS(4, float)
KERNELS(5, double)
KERNELS(5, float)
#undef KERNELS

} // namespace mgard_cuda
