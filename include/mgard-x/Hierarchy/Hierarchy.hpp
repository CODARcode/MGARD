/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../RuntimeX/RuntimeX.h"
#include "Hierarchy.h"

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <limits>
#include <vector>

#ifndef MGARD_X_HIERARCHY_HPP
#define MGARD_X_HIERARCHY_HPP

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
void Hierarchy<D, T, DeviceType>::calc_volume(SIZE dof, T *dist, T *volume,
                                              bool reciprocal) {
  T *h_dist = new T[dof];
  T *h_volume = new T[dof];
  for (int i = 0; i < dof; i++) {
    h_volume[i] = 0.0;
  }
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

  if (reciprocal) {
    for (int i = 0; i < dof; i++) {
      h_volume[i] = 1.0 / h_volume[i];
    }
  }
  MemoryManager<DeviceType>::Copy1D(volume, h_volume, dof, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  delete[] h_dist;
  delete[] h_volume;
}

template <DIM D, typename T, typename DeviceType>
void Hierarchy<D, T, DeviceType>::init(std::vector<SIZE> shape,
                                       std::vector<T *> coords,
                                       SIZE max_larget_level) {

  this->shape = shape;

  std::vector<std::vector<SIZE>> shape_level;
  for (int d = 0; d < D; d++) {
    std::vector<SIZE> curr_shape_level;
    SIZE n = shape[d];
    while (n > 2) {
      curr_shape_level.push_back(n);
      n = n / 2 + 1;
    }
    curr_shape_level.push_back(2);
    shape_level.push_back(curr_shape_level);
  }

  SIZE nlevel = shape_level[0].size();
  for (DIM d = 1; d < D; d++) {
    nlevel = std::min(nlevel, (SIZE)shape_level[d].size());
  }

  _l_target = nlevel - 1;
  _l_target = std::min(_l_target, max_larget_level);

  for (int l = 0; l < _l_target + 1; l++) {
    std::vector<SIZE> curr_level_shape(D);
    Array<1, SIZE, DeviceType> curr_level_shape_array({D});
    assert(shape_level.size() == D);
    for (int d = 0; d < D; d++) {
      curr_level_shape[d] = shape_level[d][_l_target - l];
    }
    curr_level_shape_array.load(curr_level_shape.data());
    curr_level_shape_array.hostCopy();
    _level_shape.push_back(curr_level_shape);
    _level_shape_array.push_back(curr_level_shape_array);
  }

  {
    _total_num_elems = 1;
    for (int d = D - 1; d >= 0; d--) {
      _total_num_elems *= _level_shape[_l_target][d];
    }
    _linearized_width = 1;
    for (int d = D - 2; d >= 0; d--) {
      _linearized_width *= _level_shape[_l_target][d];
    }
  }

  { // Ranges
    SIZE *ranges_h_org = new SIZE[(_l_target + 2) * D];
    // beginning of the range
    for (int d = 0; d < D; d++) {
      ranges_h_org[d] = 0;
    }
    for (int l = 0; l < _l_target + 1; l++) {
      for (int d = 0; d < D; d++) {
        // Use l+1 bacause the first is reserved for 0 (beginning of the range)
        ranges_h_org[(l + 1) * D + d] = _level_shape[l][d];
      }
    }
    bool pitched = false;
    _level_ranges = Array<2, SIZE, DeviceType>({_l_target + 2, D}, pitched);
    _level_ranges.load(ranges_h_org);
    _level_ranges.hostCopy(); // keeping a copy on the host
    delete[] ranges_h_org;
  }

  {
    SIZE max_width = 0;
    for (DIM d = 0; d < D; d++) {
      max_width = std::max(max_width, _level_shape[_l_target][d]);
    }
    int *_level_marks_h = new int[D * max_width];
    for (DIM d = 0; d < D; d++) {
      int curr_level = 0;
      int i = 0;
      for (SIZE l = 0; l < _l_target + 1; l++) {
        for (; i < _level_shape[l][d]; i++) {
          _level_marks_h[d * max_width + i] = l;
        }
      }
    }
    bool pitched = false;
    _level_marks = Array<2, int, DeviceType>({D, max_width}, pitched);
    _level_marks.load(_level_marks_h);
    // PrintSubarray("_level_marks", SubArray(_level_marks));
    delete[] _level_marks_h;
  }

  { // Coords
    for (int d = 0; d < D; d++) {
      Array<1, T, DeviceType> coord({shape[d]});
      coord.load(coords[d]);
      _coords_org.push_back(coord);
    }
  }

  { // calculate dist and ratio
    _dist_array =
        std::vector<std::vector<Array<1, T, DeviceType>>>(_l_target + 1);
    _ratio_array =
        std::vector<std::vector<Array<1, T, DeviceType>>>(_l_target + 1);
    for (SIZE l = 0; l < _l_target + 1; l++) {
      _dist_array[l] = std::vector<Array<1, T, DeviceType>>(D);
      _ratio_array[l] = std::vector<Array<1, T, DeviceType>>(D);
    }
    // For the finest level: l = _l_target;
    for (DIM d = 0; d < D; d++) {
      _dist_array[_l_target][d] =
          Array<1, T, DeviceType>({_level_shape[_l_target][d]});
      _ratio_array[_l_target][d] =
          Array<1, T, DeviceType>({_level_shape[_l_target][d]});
      coord_to_dist(_level_shape[_l_target][d], this->_coords_org[d].data(),
                    _dist_array[_l_target][d].data());
      dist_to_ratio(_level_shape[_l_target][d],
                    _dist_array[_l_target][d].data(),
                    _ratio_array[_l_target][d].data());
    }

    // for l = 1 ... _l_target
    for (int l = _l_target - 1; l >= 0; l--) {
      for (DIM d = 0; d < D; d++) {
        _dist_array[l][d] = Array<1, T, DeviceType>({_level_shape[l][d]});
        _ratio_array[l][d] = Array<1, T, DeviceType>({_level_shape[l][d]});

        reduce_dist(_level_shape[l + 1][d], _dist_array[l + 1][d].data(),
                    _dist_array[l][d].data());
        dist_to_ratio(_level_shape[l][d], _dist_array[l][d].data(),
                      _ratio_array[l][d].data());
      }
    }
  }

  { // volume for quantization
    SIZE volumes_width = 0;
    for (DIM d = 0; d < D; d++) {
      volumes_width = std::max(volumes_width, _level_shape[_l_target][d]);
    }

    bool pitched = false;
    _level_volumes =
        Array<3, T, DeviceType>({_l_target + 1, D, volumes_width}, pitched);
    _level_volumes_reciprocal =
        Array<3, T, DeviceType>({_l_target + 1, D, volumes_width}, pitched);
    SubArray<3, T, DeviceType> volumes_subarray(_level_volumes);
    SubArray<3, T, DeviceType> volumes_reciprocal_subarray(
        _level_volumes_reciprocal);
    for (SIZE l = 0; l < _l_target + 1; l++) {
      for (DIM d = 0; d < D; d++) {
        calc_volume(_level_shape[l][d], _dist_array[l][d].data(),
                    volumes_subarray(l, d, 0), false);
        calc_volume(_level_shape[l][d], _dist_array[l][d].data(),
                    volumes_reciprocal_subarray(l, d, 0), true);
      }
    }
  }

  { // am and bm
    _am_array =
        std::vector<std::vector<Array<1, T, DeviceType>>>(_l_target + 1);
    _bm_array =
        std::vector<std::vector<Array<1, T, DeviceType>>>(_l_target + 1);
    for (SIZE l = 0; l < _l_target + 1; l++) {
      _am_array[l] = std::vector<Array<1, T, DeviceType>>(D);
      _bm_array[l] = std::vector<Array<1, T, DeviceType>>(D);
    }
    for (SIZE l = 0; l < _l_target + 1; l++) {
      for (DIM d = 0; d < D; d++) {
        _am_array[l][d] = Array<1, T, DeviceType>({_level_shape[l][d] + 1});
        _bm_array[l][d] = Array<1, T, DeviceType>({_level_shape[l][d] + 1});
        _am_array[l][d].memset(0);
        _bm_array[l][d].memset(0);
        calc_am_bm(_level_shape[l][d], _dist_array[l][d].data(),
                   _am_array[l][d].data(), _bm_array[l][d].data());
      }
    }
  }

  if (D >= 4) {
    std::vector<DIM> tmp(0);
    for (int d = D - 1; d >= 0; d--) {
      _processed_n[D - 1 - d] = tmp.size();
      _processed_dims[D - 1 - d] =
          Array<1, DIM, DeviceType>({(SIZE)tmp.size()});
      _processed_dims[D - 1 - d].load(tmp.data());
      _processed_dims[D - 1 - d].hostCopy(); // keep a host copy
      tmp.push_back(d);
    }
  }

  if (D >= 4) {
    std::vector<DIM> tmp(0);
    for (int d = 0; d < D - 3; d++) {
      tmp.push_back(d);
    }
    // Extra padding needed in for loop below.
    tmp.push_back(0);

    //+1 is used for storing empty status
    for (int d = 0; d < (int)D - 3 + 1; d++) {
      tmp.pop_back();
      _unprocessed_n[d] = tmp.size();
      _unprocessed_dims[d] = Array<1, DIM, DeviceType>({(SIZE)tmp.size()});
      _unprocessed_dims[d].load(tmp.data());
    }
  }

  {
    _level_num_elems.resize(_l_target + 1);
    SIZE prev_num_elems = 0;
    for (int level_idx = 0; level_idx < _l_target + 1; level_idx++) {
      SIZE curr_num_elems = 1;
      for (DIM d = 0; d < D; d++) {
        curr_num_elems *= level_shape(level_idx, d);
      }
      _level_num_elems[level_idx] = curr_num_elems - prev_num_elems;
      prev_num_elems = curr_num_elems;
    }
  }

  dummy_array = Array<1, T, DeviceType>({1});

  initialized = true;
}

template <DIM D, typename T, typename DeviceType>
size_t
Hierarchy<D, T, DeviceType>::EstimateMemoryFootprint(std::vector<SIZE> shape) {

  size_t estimate_memory_usgae = 0;
  Array<1, T, DeviceType> array_with_pitch({1});
  SIZE pitch_size = array_with_pitch.ld(0) * sizeof(T);

  this->shape = shape;
  std::vector<std::vector<SIZE>> shape_level;
  for (int d = 0; d < D; d++) {
    std::vector<SIZE> curr_shape_level;
    SIZE n = shape[d];
    while (n > 2) {
      curr_shape_level.push_back(n);
      n = n / 2 + 1;
    }
    curr_shape_level.push_back(2);
    shape_level.push_back(curr_shape_level);
  }

  SIZE nlevel = shape_level[0].size();
  for (DIM d = 1; d < D; d++) {
    nlevel = std::min(nlevel, (SIZE)shape_level[d].size());
  }

  _l_target = nlevel - 1;

  for (int l = 0; l < _l_target + 1; l++) {
    std::vector<SIZE> curr_level_shape(D);
    estimate_memory_usgae += roundup((SIZE)(D * sizeof(SIZE)), pitch_size);
    assert(shape_level.size() == D);
    for (int d = 0; d < D; d++) {
      curr_level_shape[d] = shape_level[d][_l_target - l];
    }
    _level_shape.push_back(curr_level_shape);
  }

  {
    _total_num_elems = 1;
    for (int d = D - 1; d >= 0; d--) {
      _total_num_elems *= _level_shape[_l_target][d];
    }
    _linearized_width = 1;
    for (int d = D - 2; d >= 0; d--) {
      _linearized_width *= _level_shape[_l_target][d];
    }
  }

  { // Ranges
    estimate_memory_usgae += (_l_target + 2) * D * sizeof(SIZE);
  }

  { // Coords
    for (int d = 0; d < D; d++) {
      estimate_memory_usgae +=
          roundup((SIZE)(shape[d] * sizeof(T)), pitch_size);
    }
  }

  { // calculate dist and ratio
    for (DIM d = 0; d < D; d++) {
      estimate_memory_usgae +=
          roundup((SIZE)(_level_shape[_l_target][d] * sizeof(T)), pitch_size);
      estimate_memory_usgae +=
          roundup((SIZE)(_level_shape[_l_target][d] * sizeof(T)), pitch_size);
    }

    // for l = 1 ... _l_target
    for (int l = _l_target - 1; l >= 0; l--) {
      for (DIM d = 0; d < D; d++) {
        estimate_memory_usgae +=
            roundup((SIZE)(_level_shape[l][d] * sizeof(T)), pitch_size);
        estimate_memory_usgae +=
            roundup((SIZE)(_level_shape[l][d] * sizeof(T)), pitch_size);
      }
    }
  }

  { // volume for quantization
    SIZE volumes_width = 0;
    for (DIM d = 0; d < D; d++) {
      volumes_width = std::max(volumes_width, _level_shape[_l_target][d]);
    }
    estimate_memory_usgae += (_l_target + 1) * D * volumes_width * sizeof(T);
    estimate_memory_usgae += (_l_target + 1) * D * volumes_width * sizeof(T);
  }

  { // am and bm
    for (SIZE l = 0; l < _l_target + 1; l++) {
      for (DIM d = 0; d < D; d++) {
        estimate_memory_usgae +=
            roundup((SIZE)(_level_shape[l][d] * sizeof(T)), pitch_size);
        estimate_memory_usgae +=
            roundup((SIZE)(_level_shape[l][d] * sizeof(T)), pitch_size);
      }
    }
  }

  {
    _level_num_elems.resize(_l_target + 1);
    SIZE prev_num_elems = 0;
    for (int level_idx = 0; level_idx < _l_target + 1; level_idx++) {
      SIZE curr_num_elems = 1;
      for (DIM d = 0; d < D; d++) {
        curr_num_elems *= level_shape(level_idx, d);
      }
      _level_num_elems[level_idx] = curr_num_elems - prev_num_elems;
      prev_num_elems = curr_num_elems;
    }
  }

  return estimate_memory_usgae;
}

template <DIM D, typename T, typename DeviceType>
SIZE Hierarchy<D, T, DeviceType>::total_num_elems() {
  return _total_num_elems;
}

template <DIM D, typename T, typename DeviceType>
SIZE Hierarchy<D, T, DeviceType>::level_num_elems(SIZE level) {
  if (level > _l_target + 1) {
    log::err("Hierarchy::level_num_elems level out of bound.");
    exit(-1);
  }
  return _level_num_elems[level];
}

template <DIM D, typename T, typename DeviceType>
SIZE Hierarchy<D, T, DeviceType>::linearized_width() {
  return _linearized_width;
}

template <DIM D, typename T, typename DeviceType>
SIZE Hierarchy<D, T, DeviceType>::l_target() {
  return _l_target;
}

template <DIM D, typename T, typename DeviceType>
std::vector<SIZE> Hierarchy<D, T, DeviceType>::level_shape(SIZE level) {
  if (level > _l_target + 1) {
    log::err("Hierarchy::level_shape level out of bound.");
    exit(-1);
  }
  return _level_shape[level];
}

template <DIM D, typename T, typename DeviceType>
SIZE Hierarchy<D, T, DeviceType>::level_shape(SIZE level, DIM dim) {
  if (level > _l_target + 1) {
    log::err("Hierarchy::level_shape level out of bound.");
    exit(-1);
  }
  if (dim >= D)
    return 1;
  return _level_shape[level][dim];
}

template <DIM D, typename T, typename DeviceType>
Array<1, SIZE, DeviceType> &
Hierarchy<D, T, DeviceType>::level_shape_array(SIZE level) {
  if (level > _l_target + 1) {
    log::err("Hierarchy::level_shape_array level out of bound.");
    exit(-1);
  }
  return _level_shape_array[level];
}

template <DIM D, typename T, typename DeviceType>
Array<1, T, DeviceType> &Hierarchy<D, T, DeviceType>::dist(SIZE level,
                                                           DIM dim) {
  if (level > _l_target + 1) {
    log::err("Hierarchy::dist level out of bound.");
    exit(-1);
  }
  if (dim >= D)
    return dummy_array;
  return _dist_array[level][dim];
}

template <DIM D, typename T, typename DeviceType>
Array<1, T, DeviceType> &Hierarchy<D, T, DeviceType>::ratio(SIZE level,
                                                            DIM dim) {
  if (level > _l_target + 1) {
    log::err("Hierarchy::ratio level out of bound.");
    exit(-1);
  }
  if (dim >= D)
    return dummy_array;
  return _ratio_array[level][dim];
}

template <DIM D, typename T, typename DeviceType>
Array<1, T, DeviceType> &Hierarchy<D, T, DeviceType>::am(SIZE level, DIM dim) {
  if (level > _l_target + 1) {
    log::err("Hierarchy::am level out of bound.");
    exit(-1);
  }
  if (dim >= D)
    return dummy_array;
  return _am_array[level][dim];
}

template <DIM D, typename T, typename DeviceType>
Array<1, T, DeviceType> &Hierarchy<D, T, DeviceType>::bm(SIZE level, DIM dim) {
  if (level > _l_target + 1) {
    log::err("Hierarchy::bm level out of bound.");
    exit(-1);
  }
  if (dim >= D)
    return dummy_array;
  return _bm_array[level][dim];
}

template <DIM D, typename T, typename DeviceType>
Array<1, DIM, DeviceType> &
Hierarchy<D, T, DeviceType>::processed(SIZE idx, DIM &processed_n) {
  if (idx >= D) {
    log::err("Hierarchy::processed idx out of bound.");
    exit(-1);
  }
  processed_n = _processed_n[idx];
  return _processed_dims[idx];
}

template <DIM D, typename T, typename DeviceType>
Array<1, DIM, DeviceType> &
Hierarchy<D, T, DeviceType>::unprocessed(SIZE idx, DIM &processed_n) {
  if (idx >= D) {
    log::err("Hierarchy::unprocessed idx out of bound.");
    exit(-1);
  }
  processed_n = _unprocessed_n[idx];
  return _unprocessed_dims[idx];
}

template <DIM D, typename T, typename DeviceType>
Array<2, SIZE, DeviceType> &Hierarchy<D, T, DeviceType>::level_ranges() {
  return _level_ranges;
}

template <DIM D, typename T, typename DeviceType>
Array<2, int, DeviceType> &Hierarchy<D, T, DeviceType>::level_marks() {
  return _level_marks;
}

template <DIM D, typename T, typename DeviceType>
Array<3, T, DeviceType> &
Hierarchy<D, T, DeviceType>::level_volumes(bool reciprocal) {
  if (!reciprocal) {
    return _level_volumes;
  } else {
    return _level_volumes_reciprocal;
  }
}

template <DIM D, typename T, typename DeviceType>
data_structure_type Hierarchy<D, T, DeviceType>::data_structure() {
  return dstype;
}

template <DIM D, typename T, typename DeviceType>
void Hierarchy<D, T, DeviceType>::destroy() {
  initialized = false;
}

template <DIM D, typename T, typename DeviceType>
std::vector<T *>
Hierarchy<D, T, DeviceType>::create_uniform_coords(std::vector<SIZE> shape,
                                                   bool normalize_coordinates) {

  std::vector<T *> coords(D);
  for (int d = 0; d < D; d++) {
    T *curr_coords = new T[shape[d]];
    for (int i = 0; i < shape[d]; i++) {
      // 0...n-1
      if (!normalize_coordinates) {
        curr_coords[i] = (T)i;
      } else {
        // 0...1
        curr_coords[i] = (T)i / (shape[d] - 1);
      }
    }
    coords[d] = curr_coords;
  }
  uniform_coords_created = true;
  return coords;
}

template <typename T> void printShape(std::string name, std::vector<T> shape) {
  std::cout << log::log_info << name << ": ";
  for (DIM d = 0; d < shape.size(); d++) {
    std::cout << shape[d] << " ";
  }
  std::cout << std::endl;
}

template <DIM D, typename T, typename DeviceType>
bool Hierarchy<D, T, DeviceType>::is_initialized() {
  return initialized;
}

template <DIM D, typename T, typename DeviceType>
bool Hierarchy<D, T, DeviceType>::can_reuse(std::vector<SIZE> shape) {
  if (data_structure() == data_structure_type::Cartesian_Grid_Non_Uniform) {
    return false;
  }
  for (DIM d = 0; d < D; d++) {
    if (level_shape(l_target(), d) != shape[d]) {
      return false;
    }
  }
  return true;
}

// This constructor is for internal use only
template <DIM D, typename T, typename DeviceType>
Hierarchy<D, T, DeviceType>::Hierarchy() {}

template <DIM D, typename T, typename DeviceType>
Hierarchy<D, T, DeviceType>::Hierarchy(std::vector<SIZE> shape, Config config) {
  int ret = check_shape<D>(shape);
  if (ret == -1) {
    log::err(
        "Number of dimensions mismatch. mgard_x::Hierarchy not initialized!");
    exit(-1);
  }
  if (ret == -2) {
    log::err("Size of any dimension cannot be smaller than 3. "
             "mgard_x::Hierarchy not initialized!");
    std::stringstream ss;
    for (DIM d = 0; d < D; d++)
      ss << shape[d] << " ";
    log::err("Input shape: " + ss.str());
    exit(-1);
  }
  dstype = data_structure_type::Cartesian_Grid_Uniform;
  std::vector<T *> coords =
      create_uniform_coords(shape, config.normalize_coordinates);
  init(shape, coords, config.max_larget_level);
  assert(uniform_coords_created);
  assert(coords.size() == D);
  for (int d = 0; d < D; d++)
    delete[] coords[d];
}

template <DIM D, typename T, typename DeviceType>
Hierarchy<D, T, DeviceType>::Hierarchy(std::vector<SIZE> shape,
                                       std::vector<T *> coords, Config config) {
  int ret = check_shape<D>(shape);
  if (ret == -1) {
    log::err(
        "Number of dimensions mismatch. mgard_x::Hierarchy not initialized!");
    exit(-1);
  }
  if (ret == -2) {
    log::err("Size of any dimension cannot be smaller than 3. "
             "mgard_x::Hierarchy not initialized!");
    exit(-1);
  }

  dstype = data_structure_type::Cartesian_Grid_Non_Uniform;
  init(shape, coords, config.max_larget_level);
}

template <DIM D, typename T, typename DeviceType>
Hierarchy<D, T, DeviceType>::Hierarchy(const Hierarchy &hierarchy) {
  _l_target = hierarchy._l_target;
  shape = hierarchy.shape;
  _total_num_elems = hierarchy._total_num_elems;
  _level_num_elems = hierarchy._level_num_elems;
  _linearized_width = hierarchy._linearized_width;
  _coords_org = hierarchy._coords_org;
  _dist_array = hierarchy._dist_array;
  _ratio_array = hierarchy._ratio_array;
  _level_volumes = hierarchy._level_volumes;
  _level_volumes_reciprocal = hierarchy._level_volumes_reciprocal;
  _am_array = hierarchy._am_array;
  _bm_array = hierarchy._bm_array;
  _level_shape = hierarchy._level_shape;
  _level_shape_array = hierarchy._level_shape_array;
  _level_ranges = hierarchy._level_ranges;
  _level_marks = hierarchy._level_marks;
  dstype = hierarchy.dstype;
  if (D >= 4) {
    for (DIM d = 0; d < D; d++) {
      _processed_n[d] = hierarchy._processed_n[d];
      _unprocessed_n[d] = hierarchy._unprocessed_n[d];
      _processed_dims[d] = hierarchy._processed_dims[d];
      _unprocessed_dims[d] = hierarchy._unprocessed_dims[d];
    }
  }
  dummy_array = hierarchy.dummy_array;
}

template <DIM D, typename T, typename DeviceType>
Hierarchy<D, T, DeviceType>::~Hierarchy() {
  if (initialized) {
    destroy();
  }
}

} // namespace mgard_x

#endif
