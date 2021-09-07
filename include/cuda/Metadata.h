/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGRAD_CUDA_METADATA
#define MGRAD_CUDA_METADATA

#define SIGNATURE_SIZE 19
#define SIGNATURE "COMPRESSED_BY_MGARD"

namespace mgard_cuda {

struct Metadata {
  char signature[SIGNATURE_SIZE + 1] = SIGNATURE;
  enum data_type dtype;
  DIM total_dims = 0;
  SIZE *shape;
  SIZE l_target;
  bool gpu_lossless;
  bool enable_lz4;
  SIZE dict_size;
  double norm;
  double tol;
  double s;

public:
  SERIALIZED_TYPE *Serialize(SIZE &total_size);
  void Deserialize(SERIALIZED_TYPE *serialized_data, SIZE total_size);
  ~Metadata() {
    if (total_dims) {
      // delete [] shape;
    }
  }
};
} // namespace mgard_cuda

#endif