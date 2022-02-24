/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#include "Hierarchy.h"
#include "RuntimeX/RuntimeX.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#ifndef MGARD_X_HANDLE_HPP
#define MGARD_X_HANDLE_HPP

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
void Hierarchy<D, T, DeviceType>::coord_to_dist(SIZE dof, T *coord, T *dist) {
  if (dof <= 1)
    return;
  // printf("coord_to_dist\n");
  T *h_coord = new T[dof];
  T *h_dist = new T[dof];
  for (int i = 0; i < dof; i++)
    h_dist[i] = 0.0;
  // cudaMemcpyAsyncHelper(*this, h_coord, coord, dof * sizeof(T), AUTO, 0);
  MemoryManager<DeviceType>::Copy1D(h_coord, coord, dof, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  // this->sync(0);
  for (int i = 0; i < dof - 1; i++) {
    h_dist[i] = h_coord[i + 1] - h_coord[i];
  }
  if (dof != 2 && dof % 2 == 0) {
    T last_dist = h_dist[dof - 2];
    h_dist[dof - 2] = last_dist / 2.0;
    h_dist[dof - 1] = last_dist / 2.0;
  }
  // cudaMemcpyAsyncHelper(*this, dist, h_dist, dof * sizeof(T), AUTO, 0);
  // this->sync(0);
  MemoryManager<DeviceType>::Copy1D(dist, h_dist, dof, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);

  delete[] h_coord;
  delete[] h_dist;
}

template <DIM D, typename T, typename DeviceType>
void Hierarchy<D, T, DeviceType>::dist_to_ratio(SIZE dof, T *dist, T *ratio) {
  if (dof <= 1)
    return;
  // printf("dist_to_ratio %llu\n", dof);
  T *h_dist = new T[dof];
  T *h_ratio = new T[dof];
  for (int i = 0; i < dof; i++)
    h_ratio[i] = 0.0;
  // cudaMemcpyAsyncHelper(*this, h_dist, dist, dof * sizeof(T), AUTO, 0);
  // this->sync(0);
  MemoryManager<DeviceType>::Copy1D(h_dist, dist, dof, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  for (int i = 0; i < dof - 2; i++) {
    h_ratio[i] = h_dist[i] / (h_dist[i + 1] + h_dist[i]);
    // printf("dof: %llu ratio: %f\n", dof, h_ratio[i]);
  }
  if (dof % 2 == 0) {
    h_ratio[dof - 2] = h_dist[dof - 2] / (h_dist[dof - 1] + h_dist[dof - 2]);
    // printf("dof: %llu ratio: %f\n", dof, h_ratio[dof - 2]);
  }
  // cudaMemcpyAsyncHelper(*this, ratio, h_ratio, dof * sizeof(T), AUTO, 0);
  // this->sync(0);
  MemoryManager<DeviceType>::Copy1D(ratio, h_ratio, dof, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  delete[] h_dist;
  delete[] h_ratio;
}

template <DIM D, typename T, typename DeviceType>
void Hierarchy<D, T, DeviceType>::reduce_dist(SIZE dof, T *dist, T *dist2) {
  if (dof <= 1)
    return;
  // printf("reduce_dist\n");
  SIZE dof2 = dof / 2 + 1;
  T *h_dist = new T[dof];
  T *h_dist2 = new T[dof2];
  for (int i = 0; i < dof2; i++)
    h_dist2[i] = 0.0;
  // cudaMemcpyAsyncHelper(*this, h_dist, dist, dof * sizeof(T), AUTO, 0);
  // this->sync(0);
  MemoryManager<DeviceType>::Copy1D(h_dist, dist, dof, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  for (int i = 0; i < dof2 - 1; i++) {
    h_dist2[i] = h_dist[i * 2] + h_dist[i * 2 + 1];
  }
  if (dof2 != 2 && dof2 % 2 == 0) {
    T last_dist = h_dist2[dof2 - 2];
    h_dist2[dof2 - 2] = last_dist / 2.0;
    h_dist2[dof2 - 1] = last_dist / 2.0;
  }
  // cudaMemcpyAsyncHelper(*this, dist2, h_dist2, dof2 * sizeof(T), AUTO, 0);
  // this->sync(0);
  MemoryManager<DeviceType>::Copy1D(dist2, h_dist2, dof2, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  delete[] h_dist;
  delete[] h_dist2;
}

template <DIM D, typename T, typename DeviceType>
void Hierarchy<D, T, DeviceType>::calc_am_bm(SIZE dof, T *dist, T *am, T *bm) {
  T *h_dist = new T[dof];
  T *h_am = new T[dof + 1];
  T *h_bm = new T[dof + 1];
  for (int i = 0; i < dof + 1; i++) {
    h_am[i] = 0.0;
    h_bm[i] = 0.0;
  }
  // cudaMemcpyAsyncHelper(*this, h_dist, dist, dof * sizeof(T), AUTO, 0);
  // this->sync(0);
  MemoryManager<DeviceType>::Copy1D(h_dist, dist, dof, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  h_bm[0] = 2 * h_dist[0] / 6;
  h_am[0] = 0.0;

  for (int i = 1; i < dof - 1; i++) {
    T a_j = h_dist[i - 1] / 6;
    T w = a_j / h_bm[i - 1];
    h_bm[i] = 2 * (h_dist[i - 1] + h_dist[i]) / 6 - w * a_j;
    h_am[i] = a_j;
  }
  T a_j = h_dist[dof - 2] / 6;
  T w = a_j / h_bm[dof - 2];
  h_bm[dof - 1] = 2 * h_dist[dof - 2] / 6 - w * a_j;
  h_am[dof - 1] = a_j;
#ifdef MGARD_X_FMA
  for (int i = 0; i < dof + 1; i++) {
    h_am[i] = -1 * h_am[i];
    h_bm[i] = 1 / h_bm[i];
  }
#endif
  // cudaMemcpyAsyncHelper(*this, am, h_am, dof * sizeof(T), AUTO, 0);
  // cudaMemcpyAsyncHelper(*this, bm+1, h_bm, dof * sizeof(T), AUTO, 0); //add
  // offset

  T one = 1;
  // cudaMemcpyAsyncHelper(*this, bm, &one, sizeof(T), AUTO, 0); //add offset
  T zero = 0;
  // cudaMemcpyAsyncHelper(*this, am+dof, &zero, sizeof(T), AUTO, 0);

  MemoryManager<DeviceType>::Copy1D(am, h_am, dof, 0);
  MemoryManager<DeviceType>::Copy1D(bm + 1, h_bm, dof, 0);
  MemoryManager<DeviceType>::Copy1D(bm, &one, 1, 0);
  MemoryManager<DeviceType>::Copy1D(am + dof, &zero, 1, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);

  // this->sync(0);
  delete[] h_dist;
  delete[] h_am;
  delete[] h_bm;
}

template <DIM D, typename T, typename DeviceType>
void Hierarchy<D, T, DeviceType>::calc_volume(SIZE dof, T *dist, T *volume) {
  T *h_dist = new T[dof];
  T *h_volume = new T[dof];
  for (int i = 0; i < dof; i++) {
    h_volume[i] = 0.0;
  }
  // cudaMemcpyAsyncHelper(*this, h_dist, dist, dof * sizeof(T), AUTO, 0);
  // this->sync(0);
  MemoryManager<DeviceType>::Copy1D(h_dist, dist, dof, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  if (dof == 2) {
    h_volume[0] = h_dist[0] / 2;
    h_volume[1] = h_dist[0] / 2;
  } else {
    int node_coeff_div = dof / 2 + 1;
    h_volume[0] = h_dist[0] / 2;
    for (int i = 1; i < dof - 1; i++) {
      if (i % 2 == 0) { // node
        h_volume[i / 2] = (h_dist[i - 1] + h_dist[i]) / 2;
      } else { // coeff
        h_volume[node_coeff_div + i / 2] = (h_dist[i - 1] + h_dist[i]) / 2;
      }
    }
    if (dof % 2 != 0) {
      h_volume[node_coeff_div - 1] = h_dist[dof - 2] / 2;
    } else {
      h_volume[node_coeff_div - 1] = h_dist[dof - 1] / 2;
    }
  }

  for (int i = 0; i < dof; i++) {
    h_volume[i] = 1.0 / h_volume[i];
  }
  // cudaMemcpyAsyncHelper(*this, volume, h_volume, dof * sizeof(T), AUTO, 0);
  // this->sync(0);
  MemoryManager<DeviceType>::Copy1D(volume, h_volume, dof, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  delete[] h_dist;
  delete[] h_volume;
}

template <DIM D, typename T, typename DeviceType>
void Hierarchy<D, T, DeviceType>::init(std::vector<SIZE> shape,
                                       std::vector<T *> coords) {

  this->shape = shape;
  // determine dof
  for (DIM i = 0; i < shape.size(); i++) {
    std::vector<SIZE> curr_dofs;
    int n = shape[i];
    while (n > 2) {
      curr_dofs.push_back(n);
      n = n / 2 + 1;
    }
    if (shape[i] > 1)
      curr_dofs.push_back(2);
    dofs.push_back(curr_dofs);
  }

  linearized_depth = 1;
  for (int i = 2; i < shape.size(); i++) {
    linearized_depth *= shape[i];
  }

  // // workspace (assume 3d and above)
  // padded_linearized_depth = 1;
  // for (int i = 2; i < D; i++) {
  //   padded_linearized_depth *= (shape[i] + 2);
  // }

  for (SIZE i = 1; i < shape.size(); i++) {
    if (shape[i] == 1) {
      for (SIZE l = 0; l < dofs[0].size(); l++) {
        dofs[i].push_back(1);
      }
    }
  }

  // determine l target
  SIZE nlevel = dofs[0].size();
  for (int i = 1; i < shape.size(); i++) {
    nlevel = std::min(nlevel, (SIZE)dofs[i].size());
  }
  l_target = nlevel - 1;
  // if (config.l_target != -1) {
  // l_target = std::min(nlevel - 1, config.l_target);
  // }

  // shapes
  for (int l = 0; l < l_target + 1; l++) {
    SIZE *curr_shape_h = new SIZE[D_padded];
    for (int d = 0; d < D_padded; d++) {
      curr_shape_h[d] = dofs[d][l];
    }

    Array<1, SIZE, DeviceType> shape_array({D_padded});
    shape_array.loadData(curr_shape_h);
    shapes.push_back(shape_array);
    delete[] curr_shape_h;
  }

  // ranges
  SIZE *ranges_h = new SIZE[D * (l_target + 2)];
  for (int d = 0; d < D; d++) {
    ranges_h[d * (l_target + 2)] = 0;
    for (int l = 1; l < l_target + 2; l++) {
      ranges_h[d * (l_target + 2) + l] = dofs[d][l_target + 1 - l];
    }
  }
  ranges = Array<1, SIZE, DeviceType>({D * (l_target + 2)});
  ranges.loadData(ranges_h);
  delete[] ranges_h;

  {
    processed_n = new DIM[D];

    std::vector<DIM> tmp(0);
    for (int d = 0; d < D; d++) {
      processed_n[d] = tmp.size();
      // processed_dims_h[d] = new DIM[processed_n[d]];

      processed_dims[d] = Array<1, DIM, DeviceType>({(SIZE)tmp.size()});
      processed_dims[d].loadData(tmp.data());
      tmp.push_back(d);
    }
  }
  {
    std::vector<DIM> tmp(0);
    for (int i = 3; i < D; i++) {
      tmp.push_back(i);
    }
    unprocessed_n = new DIM[tmp.size()];

    //+1 is used for storing empty status
    for (int d = 0; d < (int)D - 3 + 1; d++) {
      unprocessed_n[d] = tmp.size();
      unprocessed_dims[d] = Array<1, DIM, DeviceType>({(SIZE)tmp.size()});
      unprocessed_dims[d].loadData(tmp.data());
      tmp.pop_back();
    }
  }

  // handle coords
  this->coords_h = coords;
  for (int i = 0; i < shape.size(); i++) {
    Array<1, T, DeviceType> coord({shape[i]});
    coord.loadData(coords[i]);
    this->coords.push_back(coord);
  }

  // calculate dist and ratio
  for (int i = 0; i < shape.size(); i++) {
    // std::vector<T *> curr_ddist_l, curr_dratio_l;
    std::vector<Array<1, T, DeviceType>> curr_ddist_array_l,
        curr_dratio_array_l;
    Array<1, T, DeviceType> curr_ddist0_array({dofs[i][0]});
    Array<1, T, DeviceType> curr_dratio0_array({dofs[i][0]});
    curr_ddist_array_l.push_back(curr_ddist0_array);
    curr_dratio_array_l.push_back(curr_dratio0_array);

    coord_to_dist(dofs[i][0], this->coords[i].get_dv(),
                  curr_ddist_array_l[0].get_dv());
    dist_to_ratio(dofs[i][0], curr_ddist_array_l[0].get_dv(),
                  curr_dratio_array_l[0].get_dv());

    // for l = 1 ... l_target
    for (int l = 1; l < l_target + 1; l++) {
      Array<1, T, DeviceType> curr_ddist_array({dofs[i][l]});
      Array<1, T, DeviceType> curr_dratio_array({dofs[i][l]});
      curr_ddist_array_l.push_back(curr_ddist_array);
      curr_dratio_array_l.push_back(curr_dratio_array);
      reduce_dist(dofs[i][l - 1], curr_ddist_array_l[l - 1].get_dv(),
                  curr_ddist_array_l[l].get_dv());
      dist_to_ratio(dofs[i][l], curr_ddist_array_l[l].get_dv(),
                    curr_dratio_array_l[l].get_dv());
    }
    this->dist_array.push_back(curr_ddist_array_l);
    this->ratio_array.push_back(curr_dratio_array_l);
  }

  // volume for quantization
  SIZE volumes_width = 0;
  for (int d = 0; d < D; d++) {
    volumes_width = std::max(volumes_width, dofs[d][0]);
  }

  volumes_array = Array<2, T, DeviceType>({D * (l_target + 1), volumes_width});
  SubArray<2, T, DeviceType> volumes_subarray(volumes_array);
  for (int d = 0; d < D; d++) {
    for (int l = 0; l < l_target + 1; l++) {
      calc_volume(dofs[d][l], dist_array[d][l].get_dv(),
                  volumes_subarray((d * (l_target + 1) + (l_target - l)), 0));
    }
  }

  for (DIM i = 0; i < D; i++) {
    // std::vector<T *> curr_am_l, curr_bm_l;
    std::vector<Array<1, T, DeviceType>> curr_am_l_array, curr_bm_l_array;
    for (SIZE l = 0; l < l_target + 1; l++) {
      Array<1, T, DeviceType> curr_am_array({dofs[i][l] + 1});
      Array<1, T, DeviceType> curr_bm_array({dofs[i][l] + 1});
      curr_am_array.memset(0);
      curr_bm_array.memset(0);

      curr_am_l_array.push_back(curr_am_array);
      curr_bm_l_array.push_back(curr_bm_array);

      calc_am_bm(dofs[i][l], dist_array[i][l].get_dv(),
                 curr_am_l_array[l].get_dv(), curr_bm_l_array[l].get_dv());
    }

    am_array.push_back(curr_am_l_array);
    bm_array.push_back(curr_bm_l_array);
  }

  // dev_type = config.dev_type;
  // dev_id = config.dev_id;
  // lossless = config.lossless;
  // huff_dict_size = config.huff_dict_size;
  // huff_block_size = config.huff_block_size;
  // lz4_block_size = config.lz4_block_size;
  // zstd_compress_level = config.zstd_compress_level;
  // reduce_memory_footprint = config.reduce_memory_footprint;
  // profile_kernels = config.profile_kernels;
  // sync_and_check_all_kernels = config.sync_and_check_all_kernels;
  // timing = config.timing;
  initialized = true;
}

template <DIM D, typename T, typename DeviceType>
void Hierarchy<D, T, DeviceType>::destroy() {

  delete[] processed_n;
  delete[] unprocessed_n;

  if (uniform_coords_created) {
    for (int d = 0; d < D; d++) {
      delete[] this->coords_h[d];
    }
    uniform_coords_created = false;
  }
}

template <DIM D, typename T, typename DeviceType>
void Hierarchy<D, T, DeviceType>::padding_dimensions(std::vector<SIZE> &shape,
                                                     std::vector<T *> &coords) {
  D_padded = D;
  if (D < 3) {
    D_padded = 3;
  }
  if (D % 2 == 0) {
    D_padded = D + 1;
  }
  // padding dimensions
  for (int d = shape.size(); d < D_padded; d++) {
    shape.push_back(1);
    T *curr_coords = new T[shape[d]];
    for (int i = 0; i < shape[d]; i++) {
      curr_coords[i] = (T)i;
    }
    coords.push_back(curr_coords);
  }
}

template <DIM D, typename T, typename DeviceType>
std::vector<T *>
Hierarchy<D, T, DeviceType>::create_uniform_coords(std::vector<SIZE> shape,
                                                   int uniform_coord_mode) {

  std::vector<T *> coords(D);
  for (int d = 0; d < D; d++) {
    T *curr_coords = new T[shape[d]];
    for (int i = 0; i < shape[d]; i++) {
      // 0...n-1
      if (uniform_coord_mode == 0) {
        curr_coords[i] = (T)i;
      } else if (uniform_coord_mode == 1) {
        // 0...1
        curr_coords[i] = (T)i / (shape[d] - 1);
      } else {
        std::cout << log::log_err << "wrong uniform coordinates mode(" << uniform_coord_mode <<") !\n";
        exit(-1);
      }
    }
    coords[d] = curr_coords;
  }
  uniform_coords_created = true;
  return coords;
}

template <typename T> T roundup(T a, T b) {
  return ((double)(a - 1) / b + 1) * b;
}

template <typename T> void printShape(std::string name, std::vector<T> shape) {
  std::cout << log::log_info << name << ": ";
  for (DIM d = 0; d < shape.size(); d++) {
    std::cout << shape[d] << " ";
  }
  std::cout << std::endl;
}

template <DIM D, typename T, typename DeviceType>
size_t
Hierarchy<D, T, DeviceType>::estimate_memory_usgae(std::vector<SIZE> shape) {
  size_t estimate_memory_usgae = 0;
  size_t total_elem = 1;
  for (DIM d = 0; d < D; d++) {
    total_elem *= shape[d];
  }
  size_t pitch_size = 1;
  if (!MemoryManager<DeviceType>::ReduceMemoryFootprint) { // pitch is enable
    T *dummy;
    SIZE ld;
    MemoryManager<DeviceType>::MallocND(dummy, 1, 1, ld, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
    MemoryManager<DeviceType>::Free(dummy);
    pitch_size = ld * sizeof(T);
  }

  size_t hierarchy_space = 0;
  // space need for hiearachy
  int nlevel = std::numeric_limits<int>::max();
  for (DIM i = 0; i < shape.size(); i++) {
    int n = shape[i];
    int l = 0;
    while (n > 2) {
      n = n / 2 + 1;
      l++;
    }
    nlevel = std::min(nlevel, l);
  }
  nlevel--;
  for (DIM d = 0; d < D; d++) {
    hierarchy_space += shape[d] * 2 * sizeof(T); // dist
    hierarchy_space += shape[d] * 2 * sizeof(T); // ratio
  }
  SIZE max_dim = 0;
  for (DIM d = 0; d < D; d++) {
    max_dim = std::max(max_dim, shape[d]);
  }

  hierarchy_space +=
      D * (nlevel + 1) * roundup(max_dim * sizeof(T), pitch_size); // volume
  for (DIM d = 0; d < D; d++) {
    hierarchy_space += shape[d] * 2 * sizeof(T); // am
    hierarchy_space += shape[d] * 2 * sizeof(T); // bm
  }

  size_t input_space = roundup(shape[D - 1] * sizeof(T), pitch_size);
  for (DIM d = 0; d < D - 1; d++) {
    input_space *= shape[d];
  }

  size_t norm_workspace = roundup(shape[D - 1] * sizeof(T), pitch_size);
  for (DIM d = 0; d < D - 1; d++) {
    norm_workspace *= shape[d];
  }
  estimate_memory_usgae = std::max(
      estimate_memory_usgae, hierarchy_space + input_space + norm_workspace);

  size_t decomposition_workspace =
      roundup(shape[D - 1] * sizeof(T), pitch_size);
  for (DIM d = 0; d < D - 1; d++) {
    decomposition_workspace *= shape[d];
  }
  estimate_memory_usgae =
      std::max(estimate_memory_usgae,
               hierarchy_space + input_space + decomposition_workspace);

  size_t quantization_workspace =
      sizeof(QUANTIZED_INT) * total_elem + // quantized
      sizeof(LENGTH) * total_elem +        // outlier index
      sizeof(QUANTIZED_INT) * total_elem;  // outlier

  estimate_memory_usgae =
      std::max(estimate_memory_usgae,
               hierarchy_space + input_space + quantization_workspace);

  size_t huffman_workspace =
      sizeof(QUANTIZED_INT) *
      total_elem; // fix-length encoding
                  // space taken by codebook generation is ignored

  estimate_memory_usgae = std::max(
      estimate_memory_usgae, hierarchy_space + input_space +
                                 quantization_workspace + huffman_workspace);

  double estimated_output_ratio = 0.5;

  return estimate_memory_usgae + (double)input_space * estimated_output_ratio;
}

template <DIM D, typename T, typename DeviceType>
bool Hierarchy<D, T, DeviceType>::need_domain_decomposition(
    std::vector<SIZE> shape) {
  // std::cout << log::log_info << "Estimated device memory usage/available "
  //                            << (double)estimate_memory_usgae(shape)/1e9
  //                            << "GB/" <<
  //                            DeviceRuntime<DeviceType>::GetAvailableMemory()/1e9
  //                            << "GB.\n";
  return estimate_memory_usgae(shape) >=
         DeviceRuntime<DeviceType>::GetAvailableMemory();
}

template <DIM D, typename T, typename DeviceType>
void Hierarchy<D, T, DeviceType>::domain_decomposition_strategy(
    std::vector<SIZE> shape) {
  // determine max dimension
  DIM max_dim = 0;
  for (DIM d = 0; d < D; d++) {
    if (shape[d] > max_dim) {
      max_dim = shape[d];
      domain_decomposed_dim = d;
    }
  }

  // domain decomposition strategy
  std::vector<SIZE> chunck_shape = shape;
  while (need_domain_decomposition(chunck_shape)) {
    chunck_shape[domain_decomposed_dim] /= 2;
  }
  domain_decomposed_size = chunck_shape[domain_decomposed_dim];
}

template <DIM D, typename T, typename DeviceType>
void Hierarchy<D, T, DeviceType>::domain_decompose(std::vector<SIZE> shape,
                                                   int uniform_coord_mode) {
  if (domain_decomposed_size < 3) {
    std::cerr << log::log_err
              << "need domain decomposition with reduce dimension.\n"
              << "This feature is not implemented. Please contact Jieyang Chen "
                 "(chenj3@ornl.gov).\n";
    exit(-1);
  }
  domain_decomposed = true;

  std::vector<SIZE> chunck_shape = shape;
  chunck_shape[domain_decomposed_dim] = domain_decomposed_size;
  for (SIZE i = 0; i < shape[domain_decomposed_dim];
       i += chunck_shape[domain_decomposed_dim]) {
    // printShape("Decomposed domain " +
    // std::to_string(hierarchy_chunck.size()), chunck_shape);
    hierarchy_chunck.push_back(
        Hierarchy<D, T, DeviceType>(chunck_shape, uniform_coord_mode));
  }

  SIZE leftover_dim_size =
      shape[domain_decomposed_dim] % chunck_shape[domain_decomposed_dim];
  if (leftover_dim_size != 0) {
    std::vector<SIZE> leftover_shape = shape;
    leftover_shape[domain_decomposed_dim] = leftover_dim_size;
    // printShape("Decomposed domain " +
    // std::to_string(hierarchy_chunck.size()), leftover_shape);
    hierarchy_chunck.push_back(
        Hierarchy<D, T, DeviceType>(leftover_shape, uniform_coord_mode));
  }
}

template <DIM D, typename T, typename DeviceType>
void Hierarchy<D, T, DeviceType>::domain_decompose(std::vector<SIZE> shape,
                                                   std::vector<T *> &coords) {
  if (domain_decomposed_size < 3) {
    std::cerr << log::log_err
              << "need domain decomposition with reduce dimension.\n"
              << "This feature is not implemented. Please contact Jieyang Chen "
                 "(chenj3@ornl.gov).\n";
    exit(-1);
  }
  domain_decomposed = true;
  std::vector<SIZE> chunck_shape = shape;
  chunck_shape[domain_decomposed_dim] = domain_decomposed_size;
  std::vector<T *> chunck_coords = coords;
  for (SIZE i = 0; i < shape[domain_decomposed_dim];
       i += chunck_shape[domain_decomposed_dim]) {
    T *decompose_dim_coord = new T[chunck_shape[domain_decomposed_dim]];
    MemoryManager<DeviceType>::Copy1D(decompose_dim_coord,
                                      coords[domain_decomposed_dim] + i,
                                      chunck_shape[domain_decomposed_dim], 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
    for (SIZE j = 0; j < chunck_shape[domain_decomposed_dim]; j++)
      decompose_dim_coord[j] -= decompose_dim_coord[0];
    chunck_coords[domain_decomposed_dim] = decompose_dim_coord;
    hierarchy_chunck.push_back(
        Hierarchy<D, T, DeviceType>(chunck_shape, chunck_coords));
    delete[] decompose_dim_coord;
  }
  SIZE leftover_dim_size =
      shape[domain_decomposed_dim] % chunck_shape[domain_decomposed_dim];
  if (leftover_dim_size != 0) {
    std::vector<SIZE> leftover_shape = shape;
    leftover_shape[domain_decomposed_dim] = leftover_dim_size;
    std::vector<T *> leftover_coords = coords;
    T *decompose_dim_coord = new T[leftover_dim_size];
    MemoryManager<DeviceType>::Copy1D(
        decompose_dim_coord,
        coords[domain_decomposed_dim] +
            (shape[domain_decomposed_dim] - leftover_dim_size),
        leftover_dim_size, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
    for (SIZE j = 0; j < leftover_dim_size; j++)
      decompose_dim_coord[j] -= decompose_dim_coord[0];
    leftover_coords[domain_decomposed_dim] = decompose_dim_coord;
    hierarchy_chunck.push_back(
        Hierarchy<D, T, DeviceType>(leftover_shape, leftover_coords));
    delete[] decompose_dim_coord;
  }
}

// This constructor is for internal use only
template <DIM D, typename T, typename DeviceType>
Hierarchy<D, T, DeviceType>::Hierarchy() {}

template <DIM D, typename T, typename DeviceType>
Hierarchy<D, T, DeviceType>::Hierarchy(std::vector<SIZE> shape,
                                       int uniform_coord_mode) {
  if (!need_domain_decomposition(shape)) {
    // Config config;
    shape_org = shape;
    std::reverse(shape.begin(), shape.end());
    int ret = check_shape<D>(shape);
    if (ret == -1) {
      std::cerr << log::log_err
                << "Number of dimensions mismatch. mgard_x::Hierarchy not "
                   "initialized!\n";
      exit(-1);
    }
    if (ret == -2) {
      std::cerr << log::log_err
                << "Size of any dimension cannot be smaller than 3. "
                   "mgard_x::Hierarchy not "
                   "initialized!\n";
      exit(-1);
    }
    dstype = data_structure_type::Cartesian_Grid_Uniform;
    std::vector<T *> coords = create_uniform_coords(shape, uniform_coord_mode);
    padding_dimensions(shape, coords);
    init(shape, coords);
  } else { // need domain decomposition
    // std::cout << log::log_info << "Need domain decomposition.\n";
    domain_decomposition_strategy(shape);
    domain_decompose(shape, uniform_coord_mode);
  }
}

template <DIM D, typename T, typename DeviceType>
Hierarchy<D, T, DeviceType>::Hierarchy(std::vector<SIZE> shape,
                                       std::vector<T *> coords) {
  if (!need_domain_decomposition(shape)) {
    // Config config;
    shape_org = shape;
    std::reverse(shape.begin(), shape.end());
    std::reverse(coords.begin(), coords.end());
    int ret = check_shape<D>(shape);
    if (ret == -1) {
      std::cerr << log::log_err
                << "Number of dimensions mismatch. mgard_x::Hanlde not "
                   "initialized!\n";
      return;
    }
    if (ret == -2) {
      std::cerr << log::log_err
                << "Size of any dimensions cannot be smaller than 3. "
                   "mgard_x::Hanlde not "
                   "initialized!\n";
    }

    dstype = data_structure_type::Cartesian_Grid_Non_Uniform;
    padding_dimensions(shape, coords);
    init(shape, coords);
  } else {
    // std::cout << log::log_info << "Need domain decomposition.\n";
    domain_decomposition_strategy(shape);
    domain_decompose(shape, coords);
  }
}

template <DIM D, typename T, typename DeviceType>
Hierarchy<D, T, DeviceType>::Hierarchy(std::vector<SIZE> shape,
                                       DIM domain_decomposed_dim,
                                       SIZE domain_decomposed_size,
                                       int uniform_coord_mode) {
  // std::cout << log::log_info << "Domain decomposition was used during
  // compression\n";
  this->domain_decomposed_dim = domain_decomposed_dim;
  this->domain_decomposed_size = domain_decomposed_size;
  domain_decompose(shape, uniform_coord_mode);
}

template <DIM D, typename T, typename DeviceType>
Hierarchy<D, T, DeviceType>::Hierarchy(std::vector<SIZE> shape,
                                       DIM domain_decomposed_dim,
                                       SIZE domain_decomposed_size,
                                       std::vector<T *> coords) {
  // std::cout << log::log_info << "Domain decomposition was used during
  // compression\n";
  this->domain_decomposed_dim = domain_decomposed_dim;
  this->domain_decomposed_size = domain_decomposed_size;
  domain_decompose(shape, coords);
}

template <DIM D, typename T, typename DeviceType>
Hierarchy<D, T, DeviceType>::Hierarchy(const Hierarchy &hierarchy) {
  l_target = hierarchy.l_target;
  D_padded = hierarchy.D_padded;
  shape_org = hierarchy.shape_org;
  shape = hierarchy.shape;
  dofs = hierarchy.dofs;

  shapes = hierarchy.shapes;
  ranges = hierarchy.ranges;
  coords = hierarchy.coords;

  dist_array = hierarchy.dist_array;
  ratio_array = hierarchy.ratio_array;
  volumes_array = hierarchy.volumes_array;

  am_array = hierarchy.am_array;
  bm_array = hierarchy.bm_array;
  linearized_depth = hierarchy.linearized_depth;
  dstype = hierarchy.dstype;

  processed_n = new DIM[D];
  unprocessed_n = new DIM[D];
  for (DIM d = 0; d < D; d++) {
    processed_n[d] = hierarchy.processed_n[d];
    unprocessed_n[d] = hierarchy.unprocessed_n[d];
  }

  for (int d = 0; d < (int)D - 3 + 1; d++) {
    unprocessed_dims[d] = hierarchy.unprocessed_dims[d];
  }
  domain_decomposed = hierarchy.domain_decomposed;
  domain_decomposed_dim = hierarchy.domain_decomposed_dim;
  domain_decomposed_size = hierarchy.domain_decomposed_size;
}

// template <DIM D, typename T, typename DeviceType>
// Hierarchy<D, T, DeviceType>::Hierarchy(std::vector<SIZE> shape, Config
// config, int uniform_coord_mode) {
//   if (!need_domain_decomposition(shape)) {
//     shape_org = shape;
//     std::reverse(shape.begin(), shape.end());
//     std::vector<T *> coords = create_uniform_coords(shape,
//     uniform_coord_mode); int ret = check_shape<D>(shape); if (ret == -1) {
//       std::cerr << log::log_err
//                 << "Number of dimensions mismatch. mgard_x::Hanlde not "
//                    "initialized!\n";
//       return;
//     }
//     if (ret == -2) {
//       std::cerr << log::log_err
//                 << "Size of any dimensions cannot be smaller than 3. "
//                    "mgard_x::Hanlde not "
//                    "initialized!\n";
//     }

//     dstype = data_structure_type::Cartesian_Grid_Uniform;
//     padding_dimensions(shape, coords);
//     init(shape, coords, config);
//   } else {
//     std::cout << log::log_info << "Need domain decomposition.\n";
//     domain_decomposed = true;
//     DIM decompose_dim;
//     std::vector<SIZE> chunck_shape = domain_decomposition(shape,
//     decompose_dim); for (SIZE i = 0; i < shape[decompose_dim]; i +=
//     chunck_shape[decompose_dim]) {
//       hierarchy_chunck.push_back(Hierarchy<D, T, DeviceType>(chunck_shape));
//     }
//     SIZE leftover_dim_size = shape[decompose_dim] %
//     chunck_shape[decompose_dim]; if (leftover_dim_size != 0) {
//       std::vector<SIZE> leftover_shape = shape;
//       leftover_shape[decompose_dim] = leftover_dim_size;
//       hierarchy_chunck.push_back(Hierarchy<D, T,
//       DeviceType>(leftover_shape));
//     }
//   }
// }

// template <DIM D, typename T, typename DeviceType>
// Hierarchy<D, T, DeviceType>::Hierarchy(std::vector<SIZE> shape, std::vector<T
// *> coords,
//                      Config config) {
//   if (!need_domain_decomposition(shape)) {
//     shape_org = shape;
//     std::reverse(shape.begin(), shape.end());
//     std::reverse(coords.begin(), coords.end());
//     int ret = check_shape<D>(shape);
//     if (ret == -1) {
//       std::cerr << log::log_err
//                 << "Number of dimensions mismatch. mgard_x::Hanlde not "
//                    "initialized!\n";
//       return;
//     }
//     if (ret == -2) {
//       std::cerr << log::log_err
//                 << "Size of any dimensions cannot be smaller than 3. "
//                    "mgard_x::Hanlde not "
//                    "initialized!\n";
//     }

//     dstype = data_structure_type::Cartesian_Grid_Non_Uniform;
//     padding_dimensions(shape, coords);
//     init(shape, coords, config);
//   } else {
//     std::cout << log::log_info << "Need domain decomposition.\n";
//     domain_decomposed = true;
//     DIM decompose_dim;
//     std::vector<SIZE> chunck_shape = domain_decomposition(shape,
//     decompose_dim); std::vector<T *> chunck_coords = coords; for (SIZE i = 0;
//     i < shape[decompose_dim]; i += chunck_shape[decompose_dim]) {
//       T * decompose_dim_coord = new T[chunck_shape[decompose_dim]];
//       MemoryManager<DeviceType>::Copy1D(decompose_dim_coord,
//       coords[decompose_dim] + i, chunck_shape[decompose_dim], 0);
//       DeviceRuntime<DeviceType>::SyncQueue(0);
//       for (SIZE j = 0; j < chunck_shape[decompose_dim]; j++)
//       decompose_dim_coord[j] -= decompose_dim_coord[0];
//       chunck_coords[decompose_dim] = decompose_dim_coord;
//       hierarchy_chunck.push_back(Hierarchy<D, T, DeviceType>(chunck_shape,
//       chunck_coords, config)); delete [] decompose_dim_coord;
//     }
//     SIZE leftover_dim_size = shape[decompose_dim] %
//     chunck_shape[decompose_dim]; if (leftover_dim_size != 0) {
//       std::vector<SIZE> leftover_shape = shape;
//       leftover_shape[decompose_dim] = leftover_dim_size;
//       std::vector<T *> leftover_coords = coords;
//       T * decompose_dim_coord = new T[leftover_dim_size];
//       MemoryManager<DeviceType>::Copy1D(decompose_dim_coord,
//       coords[decompose_dim] + (shape[decompose_dim] - leftover_dim_size),
//                             leftover_dim_size, 0);
//       DeviceRuntime<DeviceType>::SyncQueue(0);
//       for (SIZE j = 0; j < leftover_dim_size; j++) decompose_dim_coord[j] -=
//       decompose_dim_coord[0]; leftover_coords[decompose_dim] =
//       decompose_dim_coord; hierarchy_chunck.push_back(Hierarchy<D, T,
//       DeviceType>(leftover_shape, leftover_coords, config)); delete []
//       decompose_dim_coord;
//     }
//   }
// }

template <DIM D, typename T, typename DeviceType>
Hierarchy<D, T, DeviceType>::~Hierarchy() {
  if (initialized) {
    destroy();
  }
}

} // namespace mgard_x

#endif