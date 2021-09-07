/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */
#include "cuda/CommonInternal.h"

#include "cuda/Metadata.h"

namespace mgard_cuda {

SERIALIZED_TYPE *Metadata::Serialize(SIZE &total_size) {
  total_size = 0;
  total_size += sizeof(SIZE);                  // total_size
  total_size += SIGNATURE_SIZE * sizeof(char); // signature
  total_size += sizeof(enum data_type);        // data_type
  total_size += sizeof(DIM);                   // total_dims
  for (DIM d = 0; d < total_dims; d++) {       // shape
    total_size += sizeof(SIZE);
  }
  total_size += sizeof(SIZE);   // l_targer
  total_size += sizeof(bool);   // gpu_lossless
  total_size += sizeof(bool);   // enable_lz4
  total_size += sizeof(SIZE);   // dict_size
  total_size += sizeof(double); // norm
  total_size += sizeof(double); // tol
  total_size += sizeof(double); // s
  SERIALIZED_TYPE *serialized_data = (SERIALIZED_TYPE *)std::malloc(total_size);
  SERIALIZED_TYPE *p = serialized_data;

  std::memcpy(p, &total_size, sizeof(SIZE));
  p += sizeof(SIZE);
  std::memcpy(p, &signature, SIGNATURE_SIZE * sizeof(char));
  p += SIGNATURE_SIZE * sizeof(char);
  std::memcpy(p, &dtype, sizeof(enum data_type));
  p += sizeof(enum data_type);
  std::memcpy(p, &total_dims, sizeof(DIM));
  p += sizeof(DIM);
  for (DIM d = 0; d < total_dims; d++) { // shape
    std::memcpy(p, shape + d, sizeof(SIZE));
    p += sizeof(SIZE);
  }
  std::memcpy(p, &l_target, sizeof(SIZE));
  p += sizeof(SIZE);
  std::memcpy(p, &gpu_lossless, sizeof(bool));
  p += sizeof(bool);
  std::memcpy(p, &enable_lz4, sizeof(bool));
  p += sizeof(bool);
  std::memcpy(p, &dict_size, sizeof(SIZE));
  p += sizeof(SIZE);
  std::memcpy(p, &norm, sizeof(double));
  p += sizeof(double);
  std::memcpy(p, &tol, sizeof(double));
  p += sizeof(double);
  std::memcpy(p, &s, sizeof(double));
  p += sizeof(double);
  return serialized_data;
}

void Metadata::Deserialize(SERIALIZED_TYPE *serialized_data, SIZE total_size) {
  SERIALIZED_TYPE *p = serialized_data;
  SIZE size;
  std::memcpy(&size, p, sizeof(SIZE));
  p += sizeof(SIZE);
  if (total_size != size) {
    printf("Error: metadata size mismatch!\n");
    return;
  }
  std::memcpy(&signature, p, SIGNATURE_SIZE * sizeof(char));
  p += SIGNATURE_SIZE * sizeof(char);
  std::memcpy(&dtype, p, sizeof(enum data_type));
  p += sizeof(enum data_type);
  std::memcpy(&total_dims, p, sizeof(DIM));
  p += sizeof(DIM);
  shape = new SIZE[total_dims];
  for (DIM d = 0; d < total_dims; d++) { // shape
    std::memcpy(shape + d, p, sizeof(SIZE));
    p += sizeof(SIZE);
  }
  std::memcpy(&l_target, p, sizeof(SIZE));
  p += sizeof(SIZE);
  std::memcpy(&gpu_lossless, p, sizeof(bool));
  p += sizeof(bool);
  std::memcpy(&enable_lz4, p, sizeof(bool));
  p += sizeof(bool);
  std::memcpy(&dict_size, p, sizeof(SIZE));
  p += sizeof(SIZE);
  std::memcpy(&norm, p, sizeof(double));
  p += sizeof(double);
  std::memcpy(&tol, p, sizeof(double));
  p += sizeof(double);
  std::memcpy(&s, p, sizeof(double));
  p += sizeof(double);
}

} // namespace mgard_cuda