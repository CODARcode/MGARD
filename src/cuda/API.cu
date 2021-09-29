/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
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

// #include "cuda/CompressionWorkflow.h"
#include "compress_cuda.hpp"

#include "cuda/MemoryManagement.h"

#include "cuda/DataRefactoring.h"
#include "cuda/LinearQuantization.h"
#include "cuda/LosslessCompression.h"

namespace mgard_cuda {

bool verify(const void *compressed_data, size_t compressed_size) {
  char magic_word[MAGIC_WORD_SIZE + 1];
  if (compressed_size < sizeof(magic_word))
    return false;
  SIZE meta_size = *(SIZE *)compressed_data;
  Metadata meta;
  meta.Deserialize((SERIALIZED_TYPE *)compressed_data, meta_size);
  std::memcpy(magic_word, meta.magic_word, MAGIC_WORD_SIZE);
  magic_word[MAGIC_WORD_SIZE] = '\0';
  if (strcmp(magic_word, MAGIC_WORD) == 0) {
    return true;
  } else {
    return false;
  }
}

enum data_type infer_type(const void *compressed_data, size_t compressed_size) {
  if (!verify(compressed_data, compressed_size)) {
    std::cout << log::log_err << "cannot verify the data!\n";
    exit(-1);
  }
  Metadata meta;
  SIZE meta_size = *(SIZE *)compressed_data + meta.metadata_size_offset();
  meta.Deserialize((SERIALIZED_TYPE *)compressed_data, meta_size);
  return meta.dtype;
}

std::vector<SIZE> infer_shape(const void *compressed_data,
                              size_t compressed_size) {
  if (!verify(compressed_data, compressed_size)) {
    std::cout << log::log_err << "cannot verify the data!\n";
    exit(-1);
  }
  Metadata meta;
  uint32_t meta_size =
      *(uint32_t *)compressed_data + meta.metadata_size_offset();
  meta.Deserialize((SERIALIZED_TYPE *)compressed_data, meta_size);
  std::vector<SIZE> shape(meta.total_dims);
  for (DIM d = 0; d < meta.total_dims; d++) {
    shape[d] = (SIZE)(*(meta.shape + d));
  }
  return shape;
}

enum data_structure_type infer_data_structure(const void *compressed_data,
                                              size_t compressed_size) {
  if (!verify(compressed_data, compressed_size)) {
    std::cout << log::log_err << "cannot verify the data!\n";
    exit(-1);
  }
  Metadata meta;
  uint32_t meta_size =
      *(uint32_t *)compressed_data + meta.metadata_size_offset();
  meta.Deserialize((SERIALIZED_TYPE *)compressed_data, meta_size);
  return meta.dstype;
}

std::string infer_nonuniform_coords_file(const void *compressed_data,
                                         size_t compressed_size) {
  if (!verify(compressed_data, compressed_size)) {
    std::cout << log::log_err << "cannot verify the data!\n";
    exit(-1);
  }
  Metadata meta;
  uint32_t meta_size =
      *(uint32_t *)compressed_data + meta.metadata_size_offset();
  meta.Deserialize((SERIALIZED_TYPE *)compressed_data, meta_size);
  return std::string(meta.nonuniform_coords_file);
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
void compress(std::vector<SIZE> shape, T tol, T s, enum error_bound_type mode,
              const void *original_data, void *&compressed_data,
              size_t &compressed_size, Config config, bool isAllocated,
              std::vector<T *> coords) {
  Handle<D, T> handle(shape, coords, config);
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

void compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol,
              double s, enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size, Config config,
              bool isAllocated) {
  if (dtype == data_type::Float) {
    if (D == 1) {
      compress<1, float>(shape, tol, s, mode, original_data, compressed_data,
                         compressed_size, config, isAllocated);
    } else if (D == 2) {
      compress<2, float>(shape, tol, s, mode, original_data, compressed_data,
                         compressed_size, config, isAllocated);
    } else if (D == 3) {
      compress<3, float>(shape, tol, s, mode, original_data, compressed_data,
                         compressed_size, config, isAllocated);
    } else if (D == 4) {
      compress<4, float>(shape, tol, s, mode, original_data, compressed_data,
                         compressed_size, config, isAllocated);
    } else if (D == 5) {
      compress<5, float>(shape, tol, s, mode, original_data, compressed_data,
                         compressed_size, config, isAllocated);
    } else {
      std::cout << log::log_err
                << "do not support higher than five dimentions!\n";
      exit(-1);
    }
  } else if (dtype == data_type::Double) {
    if (D == 1) {
      compress<1, double>(shape, tol, s, mode, original_data, compressed_data,
                          compressed_size, config, isAllocated);
    } else if (D == 2) {
      compress<2, double>(shape, tol, s, mode, original_data, compressed_data,
                          compressed_size, config, isAllocated);
    } else if (D == 3) {
      compress<3, double>(shape, tol, s, mode, original_data, compressed_data,
                          compressed_size, config, isAllocated);
    } else if (D == 4) {
      compress<4, double>(shape, tol, s, mode, original_data, compressed_data,
                          compressed_size, config, isAllocated);
    } else if (D == 5) {
      compress<5, double>(shape, tol, s, mode, original_data, compressed_data,
                          compressed_size, config, isAllocated);
    } else {
      std::cout << log::log_err
                << "do not support higher than five dimentions!\n";
      exit(-1);
    }
  } else {
    std::cout << log::log_err
              << "do not support types other than double and float!\n";
    exit(-1);
  }
}

void compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol,
              double s, enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size, Config config,
              bool isAllocated, std::vector<const Byte *> coords) {
  if (dtype == data_type::Float) {
    std::vector<float *> float_coords;
    for (auto &coord : coords)
      float_coords.push_back((float *)coord);
    if (D == 1) {
      compress<1, float>(shape, tol, s, mode, original_data, compressed_data,
                         compressed_size, config, isAllocated, float_coords);
    } else if (D == 2) {
      compress<2, float>(shape, tol, s, mode, original_data, compressed_data,
                         compressed_size, config, isAllocated, float_coords);
    } else if (D == 3) {
      compress<3, float>(shape, tol, s, mode, original_data, compressed_data,
                         compressed_size, config, isAllocated, float_coords);
    } else if (D == 4) {
      compress<4, float>(shape, tol, s, mode, original_data, compressed_data,
                         compressed_size, config, isAllocated, float_coords);
    } else if (D == 5) {
      compress<5, float>(shape, tol, s, mode, original_data, compressed_data,
                         compressed_size, config, isAllocated, float_coords);
    } else {
      std::cout << log::log_err
                << "do not support higher than five dimentions!\n";
      exit(-1);
    }
  } else if (dtype == data_type::Double) {
    std::vector<double *> double_coords;
    for (auto &coord : coords)
      double_coords.push_back((double *)coord);
    if (D == 1) {
      compress<1, double>(shape, tol, s, mode, original_data, compressed_data,
                          compressed_size, config, isAllocated, double_coords);
    } else if (D == 2) {
      compress<2, double>(shape, tol, s, mode, original_data, compressed_data,
                          compressed_size, config, isAllocated, double_coords);
    } else if (D == 3) {
      compress<3, double>(shape, tol, s, mode, original_data, compressed_data,
                          compressed_size, config, isAllocated, double_coords);
    } else if (D == 4) {
      compress<4, double>(shape, tol, s, mode, original_data, compressed_data,
                          compressed_size, config, isAllocated, double_coords);
    } else if (D == 5) {
      compress<5, double>(shape, tol, s, mode, original_data, compressed_data,
                          compressed_size, config, isAllocated, double_coords);
    } else {
      std::cout << log::log_err
                << "do not support higher than five dimentions!\n";
      exit(-1);
    }
  } else {
    std::cout << log::log_err
              << "do not support types other than double and float!\n";
    exit(-1);
  }
}

void decompress(const void *compressed_data, size_t compressed_size,
                void *&decompressed_data, Config config, bool isAllocated) {
  std::vector<mgard_cuda::SIZE> shape =
      mgard_cuda::infer_shape(compressed_data, compressed_size);
  mgard_cuda::data_type dtype =
      mgard_cuda::infer_type(compressed_data, compressed_size);

  if (dtype == data_type::Float) {
    if (shape.size() == 1) {
      decompress<1, float>(shape, compressed_data, compressed_size,
                           decompressed_data, config, isAllocated);
    } else if (shape.size() == 2) {
      decompress<2, float>(shape, compressed_data, compressed_size,
                           decompressed_data, config, isAllocated);
    } else if (shape.size() == 3) {
      decompress<3, float>(shape, compressed_data, compressed_size,
                           decompressed_data, config, isAllocated);
    } else if (shape.size() == 4) {
      decompress<4, float>(shape, compressed_data, compressed_size,
                           decompressed_data, config, isAllocated);
    } else if (shape.size() == 5) {
      decompress<5, float>(shape, compressed_data, compressed_size,
                           decompressed_data, config, isAllocated);
    } else {
      std::cout << log::log_err
                << "do not support higher than five dimentions!\n";
      exit(-1);
    }
  } else if (dtype == data_type::Double) {
    if (shape.size() == 1) {
      decompress<1, double>(shape, compressed_data, compressed_size,
                            decompressed_data, config, isAllocated);
    } else if (shape.size() == 2) {
      decompress<2, double>(shape, compressed_data, compressed_size,
                            decompressed_data, config, isAllocated);
    } else if (shape.size() == 3) {
      decompress<3, double>(shape, compressed_data, compressed_size,
                            decompressed_data, config, isAllocated);
    } else if (shape.size() == 4) {
      decompress<4, double>(shape, compressed_data, compressed_size,
                            decompressed_data, config, isAllocated);
    } else if (shape.size() == 5) {
      decompress<5, double>(shape, compressed_data, compressed_size,
                            decompressed_data, config, isAllocated);
    } else {
      std::cout << log::log_err
                << "do not support higher than five dimentions!\n";
      exit(-1);
    }
  } else {
    std::cout << log::log_err
              << "do not support types other than double and float!\n";
    exit(-1);
  }
}

} // namespace mgard_cuda
