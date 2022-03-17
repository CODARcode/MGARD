/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
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

// bool verify(const void *compressed_data, size_t compressed_size) {
//   if (compressed_size < SIGNATURE_SIZE)
//     return false;
//   Metadata meta;
//   meta.Deserialize((SERIALIZED_TYPE *)compressed_data);
//   for (size_t i = 0; i < SIGNATURE_SIZE; i++) {
//     if (meta.signature[i] != meta.mgard_signature[i]) {
//       return false;
//     }
//   }
//   return true;
// }

// enum data_type infer_data_type(const void *compressed_data,
//                                size_t compressed_size) {
//   if (!verify(compressed_data, compressed_size)) {
//     std::cout << log::log_err << "cannot verify the data!\n";
//     exit(-1);
//   }
//   Metadata meta;
//   meta.Deserialize((SERIALIZED_TYPE *)compressed_data);
//   return meta.dtype;
// }

// std::vector<SIZE> infer_shape(const void *compressed_data,
//                               size_t compressed_size) {
//   if (!verify(compressed_data, compressed_size)) {
//     std::cout << log::log_err << "cannot verify the data!\n";
//     exit(-1);
//   }

//   Metadata meta;
//   meta.Deserialize((SERIALIZED_TYPE *)compressed_data);
//   std::vector<SIZE> shape(meta.total_dims);
//   for (DIM d = 0; d < meta.total_dims; d++) {
//     shape[d] = (SIZE)meta.shape[d];
//   }
//   return shape;
// }

// enum data_structure_type infer_data_structure(const void *compressed_data,
//                                               size_t compressed_size) {
//   if (!verify(compressed_data, compressed_size)) {
//     std::cout << log::log_err << "cannot verify the data!\n";
//     exit(-1);
//   }
//   Metadata meta;
//   meta.Deserialize((SERIALIZED_TYPE *)compressed_data);
//   return meta.dstype;
// }

// template <typename T>
// std::vector<T *> infer_coords(const void *compressed_data,
//                               size_t compressed_size) {
//   if (!verify(compressed_data, compressed_size)) {
//     std::cout << log::log_err << "cannot verify the data!\n";
//     exit(-1);
//   }
//   Metadata meta;
//   meta.Deserialize((SERIALIZED_TYPE *)compressed_data);
//   std::vector<SIZE> shape(meta.total_dims);
//   for (DIM d = 0; d < meta.total_dims; d++) {
//     shape[d] = (SIZE)meta.shape[d];
//   }
//   std::vector<T *> coords(meta.total_dims);
//   for (DIM d = 0; d < meta.total_dims; d++) {
//     coords[d] = (T *)std::malloc(shape[d] * sizeof(T));
//     for (SIZE i = 0; i < shape[d]; i++) {
//       coords[d][i] = (T)meta.coords[d][i];
//     }
//   }
//   return coords;
// }

// template std::vector<float *> infer_coords(const void *compressed_data,
//                                            size_t compressed_size);
// template std::vector<double *> infer_coords(const void *compressed_data,
//                                             size_t compressed_size);

// std::string infer_nonuniform_coords_file(const void *compressed_data,
//                                          size_t compressed_size) {
//   if (!verify(compressed_data, compressed_size)) {
//     std::cout << log::log_err << "cannot verify the data!\n";
//     exit(-1);
//   }
//   Metadata meta;
//   meta.Deserialize((SERIALIZED_TYPE *)compressed_data);
//   return std::string(meta.nonuniform_coords_file);
// }

// bool infer_domain_decomposed(const void *compressed_data,
//                              size_t compressed_size) {
//   if (!verify(compressed_data, compressed_size)) {
//     std::cout << log::log_err << "cannot verify the data!\n";
//     exit(-1);
//   }
//   Metadata meta;
//   meta.Deserialize((SERIALIZED_TYPE *)compressed_data);
//   return meta.domain_decomposed;
// }

} // namespace mgard_x