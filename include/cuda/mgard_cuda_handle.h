/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include <numeric>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

#ifndef MGRAD_CUDA_HANDLE
#define MGRAD_CUDA_HANDLE

namespace mgard {

struct mgard_cuda_config {
  int l_target;
  int huff_dict_size;
  int huff_block_size;
  bool enable_lz4;
  int lz4_block_size;
};

template <typename T, int D> struct mgard_cuda_handle {

  /* for Internal use only */
  mgard_cuda_handle();

  /* for general users */
  mgard_cuda_handle(std::vector<size_t> shape);
  mgard_cuda_handle(std::vector<size_t> shape, std::vector<T *> coords);
  mgard_cuda_handle(std::vector<size_t> shape, mgard_cuda_config config);
  mgard_cuda_handle(std::vector<size_t> shape, std::vector<T *> coords,
                    mgard_cuda_config config);

  ~mgard_cuda_handle();

  // int get_lindex(const int n, const int no, const int i);
  void allocate_workspace();
  void free_workspace();
  void *get(int i);
  int get_queues(int dr, int dc, int df);
  void sync(int i);
  void sync_all();
  void destory();

  /* CUDA env */
  void *queues;
  int num_of_queues;
  int queue_per_block;
  int num_gpus;
  int *dev_ids;

  /* Refactoring env */
  int B;
  int opt;
  int l_target;
  int local_l_target;
  int nrow, ncol, nfib, nr, nc, nf;
  T *dcoords_r, *dcoords_c, *dcoords_f;

  std::vector<int> nr_l;
  std::vector<int> nc_l;
  std::vector<int> nf_l;

  int D_padded;
  std::vector<std::vector<int>> dofs;
  std::vector<int *> shapes_h;
  std::vector<int *> shapes_d;
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

  int *dirow, *dicol, *difib;
  int *dirow_p, *dicol_p, *difib_p;
  int *dirow_a, *dicol_a, *difib_a;
  T *ddist_r, *ddist_c, *ddist_f;
  T **ddist_r_l, **ddist_c_l, **ddist_f_l;
  T *dratio_r, *dratio_c, *dratio_f;
  T **dratio_r_l, **dratio_c_l, **dratio_f_l;
  T *dwork;
  int lddwork, lddwork1, lddwork2;
  T **dcwork_2d_rf, **dcwork_2d_cf;
  int *lddcwork_2d_rf, *lddcwork_2d_cf;
  T **am_row, **bm_row, **am_col, **bm_col, **am_fib, **bm_fib;
  T **am_row_l, **bm_row_l, **am_col_l, **bm_col_l, **am_fib_l, **bm_fib_l;

  int *div_srow, *div_scol, *div_sfib;
  int *div_nrow, *div_ncol, *div_nfib;
  int **div_sr_l, **div_sc_l, **div_sf_l;
  int **div_nr_l, **div_nc_l, **div_nf_l;

  int max_local_nrow, max_local_ncol, max_local_nfib;

  int ***dev_assign;

  T *dw;
  int lddw1, lddw2;
  thrust::device_vector<int> ldws;
  int *ldws_h;
  int *ldws_d;

  T *db;
  int lddb1, lddb2;
  thrust::device_vector<int> ldbs;
  int *ldbs_h;
  int *ldbs_d;

  bool *huffman_flags;
  thrust::device_vector<int> ldhuffman_flags;

  T ****mdw;
  int ***ldmdw1, ***ldmdw2;
  int **swr_l, **swc_l, **swf_l;
  int **ewr_l, **ewc_l, **ewf_l;

  T ****mdsf;
  int ***ldmdsf1, ***ldmdsf2;
  T ****mdsc;
  int ***ldmdsc1, ***ldmdsc2;
  T ****mdsr;
  int ***ldmdsr1, ***ldmdsr2;

  T ****mdef;
  int ***ldmdef1, ***ldmdef2;
  T ****mdec;
  int ***ldmdec1, ***ldmdec2;
  T ****mder;
  int ***ldmder1, ***ldmder2;

  int **local_srow, **local_scol, **local_sfib;
  int **local_nrow, **local_ncol, **local_nfib;
  int ***local_sr_l, ***local_sc_l, ***local_sf_l;
  int ***local_nr_l, ***local_nc_l, ***local_nf_l;

  int **local_dirow, **local_dicol, **local_difib;
  int **local_dirow_p, **local_dicol_p, **local_difib_p;
  int **local_dirow_a, **local_dicol_a, **local_difib_a;
  T **local_dcoords_r, **local_dcoords_c, **local_dcoords_f;
  T **local_dccoords_r, **local_dccoords_c, **local_dccoords_f;

  T ***local_ddist_r_l, ***local_ddist_c_l, ***local_ddist_f_l;
  T ***local_dratio_r_l, ***local_dratio_c_l, ***local_dratio_f_l;
  T ***local_am_row_l, ***local_bm_row_l, ***local_am_col_l, ***local_bm_col_l,
      ***local_am_fib_l, ***local_bm_fib_l;

  int ***auto_tuning_cc;
  int ***auto_tuning_mr1, ***auto_tuning_ts1;
  int ***auto_tuning_mr2, ***auto_tuning_ts2;
  int ***auto_tuning_mr3, ***auto_tuning_ts3;
  int arch, precision;

  void *cascaded_compress_temp_space;
  size_t cascaded_compress_temp_size;
  size_t cascaded_compress_max_out_size;
  void *cascaded_decompress_temp_space;
  size_t cascaded_decompress_temp_size;
  int size_ratio;

private:
  void padding_dimensions(std::vector<size_t> &shape, std::vector<T *> &coords);
  void init_auto_tuning_table();
  void init(std::vector<size_t> shape, std::vector<T *> coords,
            mgard_cuda_config config);
  mgard_cuda_config default_config();
};

} // namespace mgard

#endif