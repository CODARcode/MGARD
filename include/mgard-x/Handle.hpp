/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#include "RuntimeX/RuntimeX.h"
#include "Handle.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#ifndef MGARD_X_HANDLE_HPP
#define MGARD_X_HANDLE_HPP

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
void Handle<D, T, DeviceType>::coord_to_dist(SIZE dof, T * coord, T * dist) {
  if (dof <= 1) return;
  // printf("coord_to_dist\n");
  T * h_coord = new T[dof];
  T * h_dist = new T[dof];
  for (int i = 0; i < dof; i ++)  h_dist[i] = 0.0;
  // cudaMemcpyAsyncHelper(*this, h_coord, coord, dof * sizeof(T), AUTO, 0);
  MemoryManager<CUDA>::Copy1D(h_coord, coord, dof, 0);
  DeviceRuntime<CUDA>::SyncQueue(0);
  // this->sync(0);
  for (int i = 0; i < dof - 1; i++) {
    h_dist[i] = h_coord[i+1] - h_coord[i];
  }
  if (dof !=2 && dof % 2 == 0) {
    T last_dist = h_dist[dof-2];
    h_dist[dof-2] = last_dist / 2.0;
    h_dist[dof-1] = last_dist / 2.0;
  }
  // cudaMemcpyAsyncHelper(*this, dist, h_dist, dof * sizeof(T), AUTO, 0);
  // this->sync(0);
  MemoryManager<CUDA>::Copy1D(dist, h_dist, dof, 0);
  DeviceRuntime<CUDA>::SyncQueue(0);

  delete [] h_coord;
  delete [] h_dist;
}

template <DIM D, typename T, typename DeviceType>
void Handle<D, T, DeviceType>::dist_to_ratio(SIZE dof, T * dist, T * ratio) {
  if (dof <= 1) return;
  // printf("dist_to_ratio %llu\n", dof);
  T * h_dist = new T[dof];
  T * h_ratio = new T[dof];
  for (int i = 0; i < dof; i ++)  h_ratio[i] = 0.0;
  // cudaMemcpyAsyncHelper(*this, h_dist, dist, dof * sizeof(T), AUTO, 0);
  // this->sync(0);
  MemoryManager<CUDA>::Copy1D(h_dist, dist, dof, 0);
  DeviceRuntime<CUDA>::SyncQueue(0);
  for (int i = 0; i < dof - 2; i++) {
    h_ratio[i] = h_dist[i] / (h_dist[i+1] + h_dist[i]);
    // printf("dof: %llu ratio: %f\n", dof, h_ratio[i]);
  }
  if (dof % 2 == 0) {
    h_ratio[dof - 2] = h_dist[dof - 2] / (h_dist[dof - 1] + h_dist[dof - 2]);
    // printf("dof: %llu ratio: %f\n", dof, h_ratio[dof - 2]);
  }
  // cudaMemcpyAsyncHelper(*this, ratio, h_ratio, dof * sizeof(T), AUTO, 0);
  // this->sync(0);
  MemoryManager<CUDA>::Copy1D(ratio, h_ratio, dof, 0);
  DeviceRuntime<CUDA>::SyncQueue(0);
  delete [] h_dist;
  delete [] h_ratio;
}

template <DIM D, typename T, typename DeviceType>
void Handle<D, T, DeviceType>::reduce_dist(SIZE dof, T * dist, T * dist2) {
  if (dof <= 1) return;
  // printf("reduce_dist\n");
  SIZE dof2 = dof / 2 + 1;
  T * h_dist = new T[dof];
  T * h_dist2 = new T[dof2];
  for (int i = 0; i < dof2; i ++)  h_dist2[i] = 0.0;
  // cudaMemcpyAsyncHelper(*this, h_dist, dist, dof * sizeof(T), AUTO, 0);
  // this->sync(0); 
  MemoryManager<CUDA>::Copy1D(h_dist, dist, dof, 0);
  DeviceRuntime<CUDA>::SyncQueue(0); 
  for (int i = 0; i < dof2 - 1; i++) {
    h_dist2[i] = h_dist[i*2] + h_dist[i*2+1]; 
  }
  if (dof2 !=2 && dof2 % 2 == 0) {
    T last_dist = h_dist2[dof2-2];
    h_dist2[dof2-2] = last_dist / 2.0;
    h_dist2[dof2-1] = last_dist / 2.0;
  }
  // cudaMemcpyAsyncHelper(*this, dist2, h_dist2, dof2 * sizeof(T), AUTO, 0);
  // this->sync(0);
  MemoryManager<CUDA>::Copy1D(dist2, h_dist2, dof2, 0);
  DeviceRuntime<CUDA>::SyncQueue(0); 
  delete [] h_dist;
  delete [] h_dist2;
}

template <DIM D, typename T, typename DeviceType>
void Handle<D, T, DeviceType>::calc_am_bm(SIZE dof, T * dist, T * am, T * bm) {
  T * h_dist = new T[dof];
  T * h_am = new T[dof+1];
  T * h_bm = new T[dof+1];
  for (int i = 0; i < dof+1; i ++) { h_am[i] = 0.0; h_bm[i] = 0.0; }
  // cudaMemcpyAsyncHelper(*this, h_dist, dist, dof * sizeof(T), AUTO, 0);
  // this->sync(0);  
  MemoryManager<CUDA>::Copy1D(h_dist, dist, dof, 0);
  DeviceRuntime<CUDA>::SyncQueue(0); 
  h_bm[0] = 2 * h_dist[0] / 6;
  h_am[0] = 0.0;

  for (int i = 1; i < dof-1; i++) {
    T a_j = h_dist[i-1] / 6;
    T w = a_j / h_bm[i-1];
    h_bm[i] = 2 * (h_dist[i-1] + h_dist[i]) / 6 - w * a_j;
    h_am[i] = a_j;
  }
  T a_j = h_dist[dof-2] / 6;
  T w = a_j / h_bm[dof-2];
  h_bm[dof-1] = 2 * h_dist[dof-2] / 6 - w * a_j;
  h_am[dof-1] = a_j;
  #ifdef MGARD_X_FMA
  for (int i = 0; i < dof+1; i ++) { h_am[i] = -1 * h_am[i]; h_bm[i] = 1 / h_bm[i]; }
  #endif
  // cudaMemcpyAsyncHelper(*this, am, h_am, dof * sizeof(T), AUTO, 0);
  // cudaMemcpyAsyncHelper(*this, bm+1, h_bm, dof * sizeof(T), AUTO, 0); //add offset
  
  
  T one = 1;
  // cudaMemcpyAsyncHelper(*this, bm, &one, sizeof(T), AUTO, 0); //add offset
  T zero = 0;
  // cudaMemcpyAsyncHelper(*this, am+dof, &zero, sizeof(T), AUTO, 0);

  MemoryManager<CUDA>::Copy1D(am, h_am, dof, 0);
  MemoryManager<CUDA>::Copy1D(bm+1, h_bm, dof, 0);
  MemoryManager<CUDA>::Copy1D(bm, &one, 1, 0);
  MemoryManager<CUDA>::Copy1D(am+dof, &zero, 1, 0);
  DeviceRuntime<CUDA>::SyncQueue(0); 

  // this->sync(0);
  delete [] h_dist;
  delete [] h_am;
  delete [] h_bm;
}

template <DIM D, typename T, typename DeviceType>
void Handle<D, T, DeviceType>::calc_volume(SIZE dof, T * dist, T * volume) {
  T * h_dist = new T[dof];
  T * h_volume = new T[dof];
  for (int i = 0; i < dof; i ++) { h_volume[i] = 0.0; }
  // cudaMemcpyAsyncHelper(*this, h_dist, dist, dof * sizeof(T), AUTO, 0);
  // this->sync(0);  
  MemoryManager<CUDA>::Copy1D(h_dist, dist, dof, 0);
  DeviceRuntime<CUDA>::SyncQueue(0); 
  if (dof == 2) {
    h_volume[0] = h_dist[0] / 2;
    h_volume[1] = h_dist[0] / 2;
  } else {
    int node_coeff_div = dof / 2 + 1;
    h_volume[0] = h_dist[0] / 2;
    for (int i = 1; i < dof-1; i++) {
      if (i % 2 == 0) { //node
        h_volume[i/2] = (h_dist[i-1] + h_dist[i]) / 2;
      } else { //coeff
        h_volume[node_coeff_div+i/2] = (h_dist[i-1] + h_dist[i]) / 2;
      }
    }
    if (dof % 2 != 0) {
      h_volume[node_coeff_div-1] = h_dist[dof-2] / 2;
    } else {
      h_volume[node_coeff_div-1] = h_dist[dof-1] / 2;
    }
  }

  for (int i = 0; i < dof; i ++) { h_volume[i] = 1.0/h_volume[i]; }
  // cudaMemcpyAsyncHelper(*this, volume, h_volume, dof * sizeof(T), AUTO, 0);
  // this->sync(0);
  MemoryManager<CUDA>::Copy1D(volume, h_volume, dof, 0);
  DeviceRuntime<CUDA>::SyncQueue(0); 
  delete [] h_dist;
  delete [] h_volume;
}


template <DIM D, typename T, typename DeviceType>
void Handle<D, T, DeviceType>::init(std::vector<SIZE> shape, std::vector<T *> coords,
                        Config config) {

  this->shape = shape;
  // determine dof
  for (DIM i = 0; i < shape.size(); i++) {
    std::vector<SIZE> curr_dofs;
    int n = shape[i];
    while (n > 2) {
      curr_dofs.push_back(n);
      n = n / 2 + 1;
    }
    if (shape[i] > 1) curr_dofs.push_back(2);
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
  if (config.l_target != -1) {
    l_target = std::min(nlevel - 1, config.l_target);
  }

  // shapes
  for (int l = 0; l < l_target + 1; l++) {
    SIZE *curr_shape_h = new SIZE[D_padded];
    for (int d = 0; d < D_padded; d++) {
      curr_shape_h[d] = dofs[d][l];
    }

    Array<1, SIZE, CUDA> shape_array({D_padded});
    shape_array.loadData(curr_shape_h);
    shapes.push_back(shape_array);
    delete [] curr_shape_h;
  }

  //ranges
  ranges_h = new SIZE[D * (l_target + 2)];
  for (int d = 0; d < D; d++) {
    ranges_h[d * (l_target + 2)] = 0;
    for (int l = 1; l < l_target + 2; l++) {
      ranges_h[d * (l_target + 2) + l] =
          dofs[d][l_target + 1 - l];
    }
  }
  ranges = Array<1, SIZE, CUDA>({D * (l_target + 2)});
  ranges.loadData(ranges_h);

  {
    processed_n = new DIM[D];
    thrust::device_vector<DIM> tmp(0);
    for (int d = 0; d < D; d++) {
      processed_n[d] = tmp.size();
      // processed_dims_h[d] = new DIM[processed_n[d]];

      processed_dims[d] = Array<1, DIM, CUDA>({(SIZE)tmp.size()});
      processed_dims[d].loadData(thrust::raw_pointer_cast(tmp.data()));
      tmp.push_back(d);
    }
  }
  {
    thrust::device_vector<DIM> tmp(0);
    for (int i = 3; i < D; i++) {
      tmp.push_back(i);
    }
    unprocessed_n = new DIM[tmp.size()];

    //+1 is used for storing empty status
    for (int d = 0; d < (int)D-3+1; d++) {
      unprocessed_n[d] = tmp.size();
      unprocessed_dims[d] = Array<1, DIM, CUDA>({(SIZE)tmp.size()});
      unprocessed_dims[d].loadData(thrust::raw_pointer_cast(tmp.data()));
      tmp.pop_back();
    }
  }

  // handle coords
  this->coords_h = coords;
  for (int i = 0; i < shape.size(); i++) {
    Array<1, T, CUDA> coord({shape[i]});
    coord.loadData(coords[i]);
    this->coords.push_back(coord);
  }

  // calculate dist and ratio
  for (int i = 0; i < shape.size(); i++) {
    // std::vector<T *> curr_ddist_l, curr_dratio_l;
    std::vector<Array<1, T, CUDA>> curr_ddist_array_l, curr_dratio_array_l;
    Array<1, T, CUDA> curr_ddist0_array({dofs[i][0]});
    Array<1, T, CUDA> curr_dratio0_array({dofs[i][0]});
    curr_ddist_array_l.push_back(curr_ddist0_array);
    curr_dratio_array_l.push_back(curr_dratio0_array);

    coord_to_dist(dofs[i][0], this->coords[i].get_dv(), curr_ddist_array_l[0].get_dv());
    dist_to_ratio(dofs[i][0], curr_ddist_array_l[0].get_dv(), curr_dratio_array_l[0].get_dv());

    // for l = 1 ... l_target
    for (int l = 1; l < l_target + 1; l++) {
      Array<1, T, CUDA> curr_ddist_array({dofs[i][l]});
      Array<1, T, CUDA> curr_dratio_array({dofs[i][l]});
      curr_ddist_array_l.push_back(curr_ddist_array);
      curr_dratio_array_l.push_back(curr_dratio_array);
      reduce_dist(dofs[i][l-1], curr_ddist_array_l[l - 1].get_dv(), curr_ddist_array_l[l].get_dv());
      dist_to_ratio(dofs[i][l], curr_ddist_array_l[l].get_dv(), curr_dratio_array_l[l].get_dv());
    }
    dist_array.push_back(curr_ddist_array_l);
    ratio_array.push_back(curr_dratio_array_l);
  }

  //volume for quantization
  SIZE volumes_width = 0;
  for (int d = 0; d < D; d++) {
    volumes_width = std::max(volumes_width, dofs[d][0]); 
  }

  volumes_array = Array<2, T, CUDA>({D * (l_target+1), volumes_width});
  SubArray<2, T, CUDA> volumes_subarray(volumes_array);
  for (int d = 0; d < D; d++) {
    for (int l = 0; l < l_target + 1; l++) {
      calc_volume(dofs[d][l], dist_array[d][l].get_dv(), volumes_subarray((d*(l_target + 1) + (l_target-l)), 0));
    }
  }


  for (DIM i = 0; i < D; i++) {
    // std::vector<T *> curr_am_l, curr_bm_l;
    std::vector<Array<1, T, CUDA>> curr_am_l_array, curr_bm_l_array;
    for (SIZE l = 0; l < l_target+1; l++) {
      Array<1, T, CUDA> curr_am_array({dofs[i][l]+1});
      Array<1, T, CUDA> curr_bm_array({dofs[i][l]+1});
      curr_am_array.memset(0);
      curr_bm_array.memset(0);

      curr_am_l_array.push_back(curr_am_array);
      curr_bm_l_array.push_back(curr_bm_array);

      calc_am_bm(dofs[i][l], dist_array[i][l].get_dv(), curr_am_l_array[l].get_dv(), curr_bm_l_array[l].get_dv());
    }

    am_array.push_back(curr_am_l_array);
    bm_array.push_back(curr_bm_l_array);
  }

  lossless = config.lossless;
  huff_dict_size = config.huff_dict_size;
  huff_block_size = config.huff_block_size;
  lz4_block_size = config.lz4_block_size;
  reduce_memory_footprint = config.reduce_memory_footprint;
  profile_kernels = config.profile_kernels;
  sync_and_check_all_kernels = config.sync_and_check_all_kernels;
  timing = config.timing;
  initialized = true;
}


template <DIM D, typename T, typename DeviceType> void Handle<D, T, DeviceType>::destroy() {

  delete[] processed_n;
  delete[] unprocessed_n;

  if (uniform_coords_created) {
    for (int d = 0; d < D; d++) {
      delete [] this->coords_h[d];
    }
    uniform_coords_created = false;
  }
}

template <DIM D, typename T, typename DeviceType>
void Handle<D, T, DeviceType>::padding_dimensions(std::vector<SIZE> &shape,
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
Handle<D, T, DeviceType>::create_uniform_coords(std::vector<SIZE> shape, int mode) {

  
  std::vector<T *> coords(D);
  for (int d = 0; d < D; d++) {
    T *curr_coords = new T[shape[d]];
    for (int i = 0; i < shape[d]; i++) {
      // 0...n-1
      if (mode == 0) {
        // printf("create_uniform_coords %d\n", mode);
        curr_coords[i] = (T)i;
      } else if (mode == 1) {
        //0...1
        curr_coords[i] = (T)i / (shape[d]-1);
      } else {
        std::cout << log::log_err << "wrong uniform coordinates mode!\n";
        exit(-1);
      }
    }
    coords[d] = curr_coords;
  }
  uniform_coords_created = true;
  return coords;
}

// This constructor is for internal use only
template <DIM D, typename T, typename DeviceType> Handle<D, T, DeviceType>::Handle() {}

template <DIM D, typename T, typename DeviceType>
Handle<D, T, DeviceType>::Handle(std::vector<SIZE> shape) {
  Config config;
  shape_org = shape;
  std::reverse(shape.begin(), shape.end());
  int ret = check_shape<D>(shape);
  if (ret == -1) {
    std::cerr << log::log_err
              << "Number of dimensions mismatch. mgard_x::Handle not "
                 "initialized!\n";
    exit(-1);
  }
  if (ret == -2) {
    std::cerr << log::log_err
              << "Size of any dimension cannot be smaller than 3. "
                 "mgard_x::Handle not "
                 "initialized!\n";
    exit(-1);
  }
  dstype = data_structure_type::Cartesian_Grid_Uniform;
  std::vector<T *> coords = create_uniform_coords(shape, 0);
  padding_dimensions(shape, coords);
  init(shape, coords, config);
}

template <DIM D, typename T, typename DeviceType>
Handle<D, T, DeviceType>::Handle(std::vector<SIZE> shape, std::vector<T *> coords) {
  Config config;
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
  init(shape, coords, config);
}

template <DIM D, typename T, typename DeviceType>
Handle<D, T, DeviceType>::Handle(std::vector<SIZE> shape, Config config) {
  shape_org = shape;
  std::reverse(shape.begin(), shape.end());
  std::vector<T *> coords = create_uniform_coords(shape, config.uniform_coord_mode);
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

  dstype = data_structure_type::Cartesian_Grid_Uniform;
  padding_dimensions(shape, coords);
  init(shape, coords, config);
}

template <DIM D, typename T, typename DeviceType>
Handle<D, T, DeviceType>::Handle(std::vector<SIZE> shape, std::vector<T *> coords,
                     Config config) {
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
  init(shape, coords, config);
}

template <DIM D, typename T, typename DeviceType> Handle<D, T, DeviceType>::~Handle() {
  if (initialized) {
    destroy();
  }
}

} // namespace mgard_x

#endif