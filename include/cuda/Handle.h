/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGRAD_CUDA_HANDLE
#define MGRAD_CUDA_HANDLE

namespace mgard_cuda {

struct mgard_cuda_config {
  int l_target;
  int huff_dict_size;
  int huff_block_size;
  bool enable_lz4;
  int lz4_block_size;
};

template <typename T, uint32_t D> struct Handle {

  /* for Internal use only */
  Handle();

  /* for general users */
  Handle(std::vector<size_t> shape);
  Handle(std::vector<size_t> shape, std::vector<T *> coords);
  Handle(std::vector<size_t> shape, mgard_cuda_config config);
  Handle(std::vector<size_t> shape, std::vector<T *> coords,
         mgard_cuda_config config);

  ~Handle();

  void allocate_workspace();
  void free_workspace();
  void *get(int i);
  void sync(int i);
  void sync_all();

  /* CUDA env */
  void *queues;
  int num_of_queues;

  /* Refactoring env */
  int l_target;
  int D_padded;
  std::vector<std::vector<int>> dofs;
  std::vector<int *> shapes_h;
  std::vector<int *> shapes_d;
  std::vector<T *> coords;
  std::vector<std::vector<T *>> dist;
  std::vector<std::vector<T *>> ratio;
  std::vector<std::vector<T *>> am;
  std::vector<std::vector<T *>> bm;
  size_t linearized_depth;
  size_t padded_linearized_depth;

  T *quantizers;
  int huff_dict_size;
  int huff_block_size;
  bool enable_lz4;
  int lz4_block_size;

  int *processed_n;
  int **processed_dims_h;
  int **processed_dims_d;

  T *dw;
  int lddw1, lddw2;
  std::vector<int> ldws_h;
  int *ldws_d;

  T *db;
  int lddb1, lddb2;
  std::vector<int> ldbs_h;
  int *ldbs_d;

  int ***auto_tuning_cc;
  int ***auto_tuning_mr1, ***auto_tuning_ts1;
  int ***auto_tuning_mr2, ***auto_tuning_ts2;
  int ***auto_tuning_mr3, ***auto_tuning_ts3;
  int arch, precision;

private:
  void padding_dimensions(std::vector<size_t> &shape, std::vector<T *> &coords);

  void create_queues();
  void destroy_queues();

  std::vector<T *> create_uniform_coords(std::vector<size_t> shape);
  bool uniform_coords_created = false;

  int num_arch = 3;
  int num_precision = 2;
  int num_range = 9;
  void init_auto_tuning_table();
  void destroy_auto_tuning_table();
  bool auto_tuning_table_created = false;

  void init(std::vector<size_t> shape, std::vector<T *> coords,
            mgard_cuda_config config);
  void destroy();
  bool initialized = false;

  mgard_cuda_config default_config();
};

} // namespace mgard_cuda

#endif