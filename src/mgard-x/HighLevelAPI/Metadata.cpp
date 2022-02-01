/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

// #include "compress_cuda.hpp"
#include "mgard-x/Hierarchy.h"
#include "mgard-x/Metadata.hpp"
#include "mgard-x/RuntimeX/RuntimeXPublic.h"

namespace mgard_x {

bool verify(const void *compressed_data, size_t compressed_size) {
  char magic_word[MAGIC_WORD_SIZE + 1];
  if (compressed_size < sizeof(magic_word))
    return false;
  Metadata meta;
  meta.Deserialize((SERIALIZED_TYPE *)compressed_data);
  std::memcpy(magic_word, meta.magic_word, MAGIC_WORD_SIZE);
  magic_word[MAGIC_WORD_SIZE] = '\0';
  if (strcmp(magic_word, MAGIC_WORD) == 0) {
    return true;
  } else {
    return false;
  }
}

enum data_type infer_data_type(const void *compressed_data,
                               size_t compressed_size) {
  if (!verify(compressed_data, compressed_size)) {
    std::cout << log::log_err << "cannot verify the data!\n";
    exit(-1);
  }
  Metadata meta;
  meta.Deserialize((SERIALIZED_TYPE *)compressed_data);
  return meta.dtype;
}

std::vector<SIZE> infer_shape(const void *compressed_data,
                              size_t compressed_size) {
  if (!verify(compressed_data, compressed_size)) {
    std::cout << log::log_err << "cannot verify the data!\n";
    exit(-1);
  }

  Metadata meta;
  meta.Deserialize((SERIALIZED_TYPE *)compressed_data);
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
  meta.Deserialize((SERIALIZED_TYPE *)compressed_data);
  return meta.dstype;
}

template <typename T>
std::vector<T *> infer_coords(const void *compressed_data,
                              size_t compressed_size) {
  if (!verify(compressed_data, compressed_size)) {
    std::cout << log::log_err << "cannot verify the data!\n";
    exit(-1);
  }
  Metadata meta;
  meta.Deserialize((SERIALIZED_TYPE *)compressed_data);
  std::vector<SIZE> shape(meta.total_dims);
  for (DIM d = 0; d < meta.total_dims; d++) {
    shape[d] = (SIZE)(*(meta.shape + d));
  }
  std::vector<T *> coords(meta.total_dims);
  for (DIM d = 0; d < meta.total_dims; d++) {
    coords[d] = (T *)std::malloc(shape[d] * sizeof(T));
    std::memcpy(coords[d], meta.coords[d], shape[d] * sizeof(T));
  }
  return coords;
}

template std::vector<float *> infer_coords(const void *compressed_data,
                                           size_t compressed_size);
template std::vector<double *> infer_coords(const void *compressed_data,
                                            size_t compressed_size);

std::string infer_nonuniform_coords_file(const void *compressed_data,
                                         size_t compressed_size) {
  if (!verify(compressed_data, compressed_size)) {
    std::cout << log::log_err << "cannot verify the data!\n";
    exit(-1);
  }
  Metadata meta;
  meta.Deserialize((SERIALIZED_TYPE *)compressed_data);
  return std::string(meta.nonuniform_coords_file);
}

bool infer_domain_decomposed(const void *compressed_data,
                             size_t compressed_size) {
  if (!verify(compressed_data, compressed_size)) {
    std::cout << log::log_err << "cannot verify the data!\n";
    exit(-1);
  }
  Metadata meta;
  meta.Deserialize((SERIALIZED_TYPE *)compressed_data);
  return meta.domain_decomposed;
}

} // namespace mgard_x