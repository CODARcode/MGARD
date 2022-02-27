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

#include "compress_x.hpp"
#include "mgard-x/Hierarchy.hpp"
#include "mgard-x/Metadata.hpp"
#include "mgard-x/RuntimeX/RuntimeX.h"

#include "Utilities/CheckEndianess.h"

#ifndef MGARD_X_HIGH_LEVEL_API_HPP
#define MGARD_X_HIGH_LEVEL_API_HPP

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
void domain_decompose(T *data, std::vector<T *> &decomposed_data,
                      std::vector<SIZE> shape, DIM domain_decomposed_dim,
                      SIZE domain_decomposed_size) {
  // {
  //   Array<D, T, DeviceType> data_array({shape});
  //   data_array.load(data);
  //   PrintSubarray("Input", SubArray(data_array));
  // }

  std::vector<SIZE> chunck_shape = shape;
  chunck_shape[domain_decomposed_dim] = domain_decomposed_size;
  std::vector<SIZE> rev_shape = shape;
  std::reverse(rev_shape.begin(), rev_shape.end());
  DIM rev_domain_decomposed_dim = D - 1 - domain_decomposed_dim;
  SIZE elem_per_chunk = 1;
  for (DIM d = 0; d < chunck_shape.size(); d++) {
    elem_per_chunk *= chunck_shape[d];
  }
  // printf("domain_decomposed_dim: %u\n", domain_decomposed_dim);
  // printf("domain_decomposed_size: %u\n", domain_decomposed_size);
  // printf("rev_domain_decomposed_dim: %u\n", rev_domain_decomposed_dim);
  // printShape("rev_shape", rev_shape);
  // printf("elem_per_chunk: %u\n", elem_per_chunk);
  for (SIZE i = 0; i < shape[domain_decomposed_dim];
       i += domain_decomposed_size) {
    T *chunck_data;
    MemoryManager<DeviceType>::MallocHost(chunck_data, elem_per_chunk, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);

    SIZE dst_ld = domain_decomposed_size;
    if (rev_domain_decomposed_dim >= 1) {
      for (DIM d = 0; d < rev_domain_decomposed_dim; d++) {
        dst_ld *= rev_shape[d];
      }
    }

    SIZE src_ld = 1;
    for (DIM d = 0; d < rev_domain_decomposed_dim + 1; d++) {
      src_ld *= rev_shape[d];
    }
    SIZE n1 = dst_ld;
    SIZE n2 = 1;
    for (DIM d = rev_domain_decomposed_dim + 1; d < D; d++) {
      n2 *= rev_shape[d];
    }

    T *src_ptr = data + n1 * (i / domain_decomposed_size);

    // std::cout << "src_ptr:" << src_ptr << " "
    //           << "src_ld:" << src_ld << " "
    //           << "chunck_data:" << chunck_data << " "
    //           << "dst_ld:" << dst_ld << " "
    //           << "n1:" << n1 << " "
    //           << "n2:" << n2 << "\n";

    MemoryManager<DeviceType>::CopyND(chunck_data, dst_ld, src_ptr, src_ld, n1,
                                      n2, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
    decomposed_data.push_back(chunck_data);
    // {
    //   Array<D, T, DeviceType> data_array({chunck_shape});
    //   data_array.load(chunck_data);
    //   PrintSubarray("chunck_data", SubArray(data_array));
    // }
  }
  SIZE leftover_dim_size =
      shape[domain_decomposed_dim] % domain_decomposed_size;
  if (leftover_dim_size != 0) {
    std::vector<SIZE> leftover_shape = shape;
    leftover_shape[domain_decomposed_dim] = leftover_dim_size;
    SIZE elem_leftover = 1;
    for (DIM d = 0; d < leftover_shape.size(); d++) {
      elem_leftover *= leftover_shape[d];
    }
    T *leftover_data;
    MemoryManager<DeviceType>::MallocHost(leftover_data, elem_leftover, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);

    SIZE dst_ld = leftover_dim_size;
    if (rev_domain_decomposed_dim >= 1) {
      for (DIM d = 0; d < rev_domain_decomposed_dim; d++) {
        dst_ld *= rev_shape[d];
      }
    }
    SIZE src_ld = 1;
    for (DIM d = 0; d < rev_domain_decomposed_dim + 1; d++) {
      src_ld *= rev_shape[d];
    }
    SIZE n1 = dst_ld;
    SIZE n2 = 1;
    for (DIM d = rev_domain_decomposed_dim + 1; d < D; d++) {
      n2 *= rev_shape[d];
    }

    T *src_ptr = data + n1 * decomposed_data.size();

    MemoryManager<DeviceType>::CopyND(leftover_data, dst_ld, src_ptr, src_ld,
                                      n1, n2, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
    decomposed_data.push_back(leftover_data);
  }
}

template <DIM D, typename T, typename DeviceType>
void domain_recompose(std::vector<T *> decomposed_data, T *data,
                      std::vector<SIZE> shape, DIM domain_decomposed_dim,
                      SIZE domain_decomposed_size) {
  std::vector<SIZE> chunck_shape = shape;
  chunck_shape[domain_decomposed_dim] = domain_decomposed_size;
  std::vector<SIZE> rev_shape = shape;
  std::reverse(rev_shape.begin(), rev_shape.end());
  DIM rev_domain_decomposed_dim = D - 1 - domain_decomposed_dim;
  SIZE elem_per_chunk = 1;
  for (DIM d = 0; d < chunck_shape.size(); d++) {
    elem_per_chunk *= chunck_shape[d];
  }
  for (SIZE i = 0; i < shape[domain_decomposed_dim];
       i += domain_decomposed_size) {
    T *chunck_data = decomposed_data[i / domain_decomposed_size];

    SIZE dst_ld = domain_decomposed_size;
    if (rev_domain_decomposed_dim >= 1) {
      for (DIM d = 0; d < rev_domain_decomposed_dim; d++) {
        dst_ld *= rev_shape[d];
      }
    }

    SIZE src_ld = 1;
    for (DIM d = 0; d < rev_domain_decomposed_dim + 1; d++) {
      src_ld *= rev_shape[d];
    }
    SIZE n1 = dst_ld;
    SIZE n2 = 1;
    for (DIM d = rev_domain_decomposed_dim + 1; d < D; d++) {
      // printf("d: %u\n", d);
      n2 *= rev_shape[d];
    }

    T *src_ptr = data + n1 * (i / domain_decomposed_size);

    // std::cout << "src_ptr:" << src_ptr << " "
    //           << "src_ld:" << src_ld << " "
    //           << "chunck_data:" << chunck_data << " "
    //           << "dst_ld:" << dst_ld << " "
    //           << "n1:" << n1 << " "
    //           << "n2:" << n2 << "\n";
    MemoryManager<DeviceType>::CopyND(src_ptr, src_ld, chunck_data, dst_ld, n1,
                                      n2, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);

    // {
    //   Array<D, T, DeviceType> data_array({chunck_shape});
    //   data_array.load(chunck_data);
    //   PrintSubarray("chunck_data", SubArray(data_array));
    // }
  }
  SIZE leftover_dim_size =
      shape[domain_decomposed_dim] % domain_decomposed_size;
  if (leftover_dim_size != 0) {
    std::vector<SIZE> leftover_shape = shape;
    leftover_shape[domain_decomposed_dim] = leftover_dim_size;
    SIZE elem_leftover = 1;
    for (DIM d = 0; d < leftover_shape.size(); d++) {
      elem_leftover *= leftover_shape[d];
    }
    T *leftover_data = decomposed_data[decomposed_data.size() - 1];

    SIZE dst_ld = leftover_dim_size;
    if (rev_domain_decomposed_dim >= 1) {
      for (DIM d = 0; d < rev_domain_decomposed_dim; d++) {
        dst_ld *= rev_shape[d];
      }
    }
    SIZE src_ld = 1;
    for (DIM d = 0; d < rev_domain_decomposed_dim + 1; d++) {
      src_ld *= rev_shape[d];
    }
    SIZE n1 = dst_ld;
    SIZE n2 = 1;
    for (DIM d = rev_domain_decomposed_dim + 1; d < D; d++) {
      n2 *= rev_shape[d];
    }

    T *src_ptr = data + n1 * decomposed_data.size();

    MemoryManager<DeviceType>::CopyND(src_ptr, src_ld, leftover_data, dst_ld,
                                      n1, n2, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
  }

  // {
  //   Array<D, T, DeviceType> data_array({shape});
  //   data_array.load(data);
  //   PrintSubarray("Output", SubArray(data_array));
  // }
}

template <DIM D, typename T, typename DeviceType>
T calc_norm_decomposed(std::vector<T *> decomposed_data, T s,
                       std::vector<SIZE> shape, DIM domain_decomposed_dim,
                       SIZE domain_decomposed_size) {
  T norm;
  Array<1, T, DeviceType> norm_array({1});
  SubArray<1, T, DeviceType> norm_subarray(norm_array);
  std::vector<SIZE> chunck_shape = shape;
  chunck_shape[domain_decomposed_dim] = domain_decomposed_size;
  SIZE elem_per_chunk = 1;
  for (DIM d = 0; d < chunck_shape.size(); d++) {
    elem_per_chunk *= chunck_shape[d];
  }
  for (SIZE i = 0; i < shape[domain_decomposed_dim];
       i += domain_decomposed_size) {
    Array<1, T, DeviceType> chunck_in_array({elem_per_chunk});
    chunck_in_array.load(decomposed_data[i / domain_decomposed_size]);
    SubArray chunck_in_subarray(chunck_in_array);
    if (s == std::numeric_limits<T>::infinity()) {
      DeviceCollective<DeviceType>::AbsMax(elem_per_chunk, chunck_in_subarray,
                                           norm_subarray, 0);
      DeviceRuntime<DeviceType>::SyncQueue(0);
      norm = std::max(norm, norm_array.hostCopy()[0]);
    } else {
      DeviceCollective<DeviceType>::SquareSum(
          elem_per_chunk, chunck_in_subarray, norm_subarray, 0);
      DeviceRuntime<DeviceType>::SyncQueue(0);
      norm += norm_array.hostCopy()[0];
    }
  }
  SIZE leftover_dim_size =
      shape[domain_decomposed_dim] % domain_decomposed_size;
  if (leftover_dim_size != 0) {
    std::vector<SIZE> leftover_shape = shape;
    leftover_shape[domain_decomposed_dim] = leftover_dim_size;
    SIZE elem_leftover = 1;
    for (DIM d = 0; d < leftover_shape.size(); d++) {
      elem_leftover *= leftover_shape[d];
    }
    Array<1, T, DeviceType> leftover_array({elem_leftover});
    leftover_array.load(decomposed_data[decomposed_data.size() - 1]);
    SubArray leftover_subarray(leftover_array);
    if (s == std::numeric_limits<T>::infinity()) {
      DeviceCollective<DeviceType>::AbsMax(elem_leftover, leftover_subarray,
                                           norm_subarray, 0);
      DeviceRuntime<DeviceType>::SyncQueue(0);
      norm = std::max(norm, norm_array.hostCopy()[0]);
    } else {
      DeviceCollective<DeviceType>::SquareSum(elem_leftover, leftover_subarray,
                                              norm_subarray, 0);
      DeviceRuntime<DeviceType>::SyncQueue(0);
      norm += norm_array.hostCopy()[0];
    }
  }

  if (s != std::numeric_limits<T>::infinity()) {
    norm = std::sqrt(norm);
  }

  return norm;
}

template <DIM D, typename T, typename DeviceType>
void compress(std::vector<SIZE> shape, T tol, T s, enum error_bound_type type,
              const void *original_data, void *&compressed_data,
              size_t &compressed_size, Config config,
              bool output_pre_allocated) {

  Hierarchy<D, T, DeviceType> hierarchy(shape, config.uniform_coord_mode);

  Metadata m;
  if (std::is_same<DeviceType, Serial>::value) {
    m.ptype = processor_type::X_Serial;
  } else if (std::is_same<DeviceType, CUDA>::value) {
    m.ptype = processor_type::X_CUDA;
  } else if (std::is_same<DeviceType, HIP>::value) {
    m.ptype = processor_type::X_HIP;
  }
  m.ebtype = type;
  m.tol = tol;
  if (s == std::numeric_limits<T>::infinity()) {
    m.ntype = norm_type::L_Inf;
    m.s = s;
  } else {
    m.ntype = norm_type::L_2;
    m.s = s;
  }
  // m.l_target = hierarchy.l_target;
  if (s != std::numeric_limits<T>::infinity()) {
    m.decomposition = decomposition_type::MultiDim;
  }
  m.decomposition = config.decomposition;
  m.reorder = config.reorder;
  m.ltype = config.lossless;
  m.huff_dict_size = config.huff_dict_size;
  m.huff_block_size = config.huff_block_size;
#ifndef MGARDX_COMPILE_CUDA
  if (config.lossless == lossless_type::Huffman_LZ4) {
    config.lossless = lossless_type::Huffman_Zstd;
    m.ltype = config.lossless; // update matadata
  }
#endif
  m.dtype =
      std::is_same<T, double>::value ? data_type::Double : data_type::Float;
  m.etype = CheckEndianess();
  m.dstype = data_structure_type::Cartesian_Grid_Uniform;
  m.total_dims = D;
  m.shape = new uint64_t[D];
  for (int d = 0; d < D; d++) {
    m.shape[d] = (uint64_t)shape[d];
  }

  if (!hierarchy.domain_decomposed) {
    mgard_x::Array<D, T, DeviceType> in_array(shape);
    in_array.load((const T *)original_data);
    T norm = 1;
    // PrintSubarray("in_array", SubArray(in_array));
    Array<1, Byte, DeviceType> lossless_compressed_array =
        compress<D, T, DeviceType>(hierarchy, in_array, type, tol, s, norm,
                                   config);
    SubArray lossless_compressed_subarray(lossless_compressed_array);
    // compressed_size = compressed_array.getShape()[0];

    if (type == error_bound_type::REL) {
      m.norm = norm;
    }
    m.domain_decomposed = false;

    uint32_t metadata_size;

    SERIALIZED_TYPE *serizalied_meta = m.Serialize(metadata_size);
    delete[] m.shape;

    // SIZE outsize = 0;
    SIZE byte_offset = 0;
    advance_with_align<SERIALIZED_TYPE>(byte_offset, metadata_size);
    advance_with_align<SIZE>(byte_offset, 1);
    align_byte_offset<uint64_t>(byte_offset); // for zero copy when deserialize
    advance_with_align<Byte>(byte_offset,
                             lossless_compressed_subarray.getShape(0));

    compressed_size = byte_offset;
    DeviceRuntime<DeviceType>::SyncDevice();

    Array<1, unsigned char, DeviceType> compressed_array(
        {(SIZE)compressed_size});
    SubArray compressed_subarray(compressed_array);

    SERIALIZED_TYPE *buffer = compressed_array.get_dv();
    void *buffer_p = (void *)buffer;
    byte_offset = 0;

    SerializeArray<SERIALIZED_TYPE>(compressed_subarray, serizalied_meta,
                                    metadata_size, byte_offset);
    SIZE lossless_size = lossless_compressed_subarray.getShape(0);
    SerializeArray<SIZE>(compressed_subarray, &lossless_size, 1, byte_offset);

    align_byte_offset<uint64_t>(byte_offset);
    SerializeArray<Byte>(compressed_subarray,
                         lossless_compressed_subarray.data(),
                         lossless_compressed_subarray.getShape(0), byte_offset);

    free(serizalied_meta);

    if (MemoryManager<DeviceType>::IsDevicePointer(original_data)) {
      if (!output_pre_allocated) {
        MemoryManager<DeviceType>::Malloc1D(compressed_data, compressed_size,
                                            0);
      }
      MemoryManager<DeviceType>::Copy1D(compressed_data,
                                        (void *)compressed_array.get_dv(),
                                        compressed_size, 0);
      DeviceRuntime<DeviceType>::SyncQueue(0);
    } else {
      if (!output_pre_allocated) {
        compressed_data = (unsigned char *)malloc(compressed_size);
      }
      memcpy(compressed_data, compressed_array.hostCopy(), compressed_size);
    }
  } else { // chunck by chunck
    if (config.timing) {
      std::cout << log::log_info
                << "Insufficient device memory for compressing the whole "
                   "dataset at once. "
                << "Orignial domain is decomposed into "
                << hierarchy.hierarchy_chunck.size() << " sub-domains\n";
    }
    std::vector<T *> decomposed_data;
    domain_decompose<D, T, DeviceType>((T *)original_data, decomposed_data,
                                       shape, hierarchy.domain_decomposed_dim,
                                       hierarchy.domain_decomposed_size);
    std::vector<SIZE> lossless_compressed_size;
    std::vector<Byte *> lossless_compressed_data;
    T norm = 1;
    if (type == error_bound_type::REL) {
      norm = calc_norm_decomposed<D, T, DeviceType>(
          decomposed_data, s, shape, hierarchy.domain_decomposed_dim,
          hierarchy.domain_decomposed_size);
      if (config.timing) {
        if (s == std::numeric_limits<T>::infinity()) {
          std::cout << log::log_info << "L_inf norm: " << norm << std::endl;
        } else {
          std::cout << log::log_info << "L_2 norm: " << norm << std::endl;
        }
      }
    }
    if (s != std::numeric_limits<T>::infinity()) {
      tol = std::sqrt((tol * tol) / hierarchy.hierarchy_chunck.size());
      if (config.timing) {
        std::cout << log::log_info << "local bound: " << tol << "\n";
      }
    }
    for (SIZE i = 0; i < hierarchy.hierarchy_chunck.size(); i++) {
      if (config.timing) {
        std::cout << log::log_info << "Compressing decomposed domain (" << i
                  << ") - ( ";
        for (DIM d = 0; d < D; d++)
          std::cout << hierarchy.hierarchy_chunck[i].shape_org[d] << " ";
        std::cout << ")\n";
      }
      Array<D, T, DeviceType> chunck_in_array(
          hierarchy.hierarchy_chunck[i].shape_org);
      chunck_in_array.load((const T *)decomposed_data[i]);
      // PrintSubarray("chunck_in_array", SubArray(chunck_in_array));
      Array<1, Byte, DeviceType> lossless_compressed_array =
          compress<D, T, DeviceType>(hierarchy.hierarchy_chunck[i],
                                     chunck_in_array, type, tol, s, norm,
                                     config);
      lossless_compressed_size.push_back(
          lossless_compressed_array.getShape()[0]);
      Byte *temp = NULL;
      MemoryManager<DeviceType>::MallocHost(
          temp, lossless_compressed_array.getShape()[0], 0);
      // std::cout << "temp: " << temp
      // << "lossless_compressed_array.get_dv(): " <<
      // lossless_compressed_array.get_dv() ;
      // << "lossless_compressed_array.getShape()[0]: " <<
      // lossless_compressed_array.getShape()[0] << "\n" << std::flush;
      // temp = new Byte[lossless_compressed_array.getShape()[0]];
      MemoryManager<DeviceType>::Copy1D(
          temp, lossless_compressed_array.get_dv(),
          lossless_compressed_array.getShape()[0], 0);
      DeviceRuntime<DeviceType>::SyncQueue(0);
      lossless_compressed_data.push_back(temp);
    }

    // Seralize
    if (type == error_bound_type::REL) {
      m.norm = norm;
    }
    m.domain_decomposed = true;
    m.ddtype = domain_decomposition_type::MaxDim;
    m.domain_decomposed_dim = hierarchy.domain_decomposed_dim;
    m.domain_decomposed_size = hierarchy.domain_decomposed_size;

    uint32_t metadata_size;
    SERIALIZED_TYPE *serizalied_meta = m.Serialize(metadata_size);

    SIZE byte_offset = 0;
    advance_with_align<SERIALIZED_TYPE>(byte_offset, metadata_size);
    for (uint32_t i = 0; i < hierarchy.hierarchy_chunck.size(); i++) {
      advance_with_align<SIZE>(byte_offset, 1);
      align_byte_offset<uint64_t>(
          byte_offset); // for zero copy when deserialize
      advance_with_align<Byte>(byte_offset, lossless_compressed_size[i]);
    }

    compressed_size = byte_offset;
    DeviceRuntime<DeviceType>::SyncDevice();

    if (!output_pre_allocated) {
      compressed_data = (unsigned char *)malloc(compressed_size);
    }

    byte_offset = 0;
    Serialize<SERIALIZED_TYPE, DeviceType>(
        (Byte *)compressed_data, serizalied_meta, metadata_size, byte_offset);
    for (uint32_t i = 0; i < hierarchy.hierarchy_chunck.size(); i++) {
      Serialize<SIZE, DeviceType>((Byte *)compressed_data,
                                  lossless_compressed_size.data() + i, 1,
                                  byte_offset);
      align_byte_offset<uint64_t>(byte_offset);
      Serialize<Byte, DeviceType>((Byte *)compressed_data,
                                  lossless_compressed_data[i],
                                  lossless_compressed_size[i], byte_offset);
    }
    free(serizalied_meta);
    for (SIZE i = 0; i < hierarchy.hierarchy_chunck.size(); i++) {
      MemoryManager<DeviceType>::FreeHost(lossless_compressed_data[i]);
      MemoryManager<DeviceType>::FreeHost(decomposed_data[i]);
      // delete [] lossless_compressed_data[i];
    }
  }
}

template <DIM D, typename T, typename DeviceType>
void compress(std::vector<SIZE> shape, T tol, T s, enum error_bound_type type,
              const void *original_data, void *&compressed_data,
              size_t &compressed_size, Config config, std::vector<T *> coords,
              bool output_pre_allocated) {
  Hierarchy<D, T, DeviceType> hierarchy(shape, coords);

  Metadata m;
  if (std::is_same<DeviceType, Serial>::value) {
    m.ptype = processor_type::X_Serial;
  } else if (std::is_same<DeviceType, CUDA>::value) {
    m.ptype = processor_type::X_CUDA;
  } else if (std::is_same<DeviceType, HIP>::value) {
    m.ptype = processor_type::X_HIP;
  }
  m.ebtype = type;
  m.tol = tol;
  if (s == std::numeric_limits<T>::infinity()) {
    m.ntype = norm_type::L_Inf;
    m.s = s;
  } else {
    m.ntype = norm_type::L_2;
    m.s = s;
  }
  // m.l_target = hierarchy.l_target;
  if (s != std::numeric_limits<T>::infinity()) {
    m.decomposition = decomposition_type::MultiDim;
  }
  m.decomposition = config.decomposition;
  m.ltype = config.lossless;
  m.huff_dict_size = config.huff_dict_size;
  m.huff_block_size = config.huff_block_size;
  m.reorder = config.reorder;
#ifndef MGARDX_COMPILE_CUDA
  if (config.lossless == lossless_type::Huffman_LZ4) {
    config.lossless = lossless_type::Huffman_Zstd;
    m.ltype = config.lossless; // update matadata
  }
#endif
  m.dtype =
      std::is_same<T, double>::value ? data_type::Double : data_type::Float;
  m.etype = CheckEndianess();
  m.dstype = data_structure_type::Cartesian_Grid_Non_Uniform;
  m.total_dims = D;
  m.shape = new uint64_t[D];
  for (int d = 0; d < D; d++) {
    m.shape[d] = (uint64_t)shape[d];
  }
  m.cltype = coordinate_location::Embedded;
  for (int d = 0; d < D; d++) {
    m.coords.push_back((Byte *)coords[d]);
  }

  if (!hierarchy.domain_decomposed) {
    mgard_x::Array<D, T, DeviceType> in_array(shape);
    in_array.load((const T *)original_data);
    T norm = 1;
    Array<1, Byte, DeviceType> lossless_compressed_array =
        compress<D, T, DeviceType>(hierarchy, in_array, type, tol, s, norm,
                                   config);
    SubArray lossless_compressed_subarray(lossless_compressed_array);
    // compressed_size = compressed_array.getShape()[0];

    if (type == error_bound_type::REL) {
      m.norm = norm;
    }
    m.domain_decomposed = false;

    uint32_t metadata_size;

    SERIALIZED_TYPE *serizalied_meta = m.Serialize(metadata_size);
    delete[] m.shape;

    // SIZE outsize = 0;
    SIZE byte_offset = 0;
    advance_with_align<SERIALIZED_TYPE>(byte_offset, metadata_size);
    advance_with_align<SIZE>(byte_offset, 1);
    align_byte_offset<uint64_t>(byte_offset); // for zero copy when deserialize
    advance_with_align<Byte>(byte_offset,
                             lossless_compressed_subarray.getShape(0));

    compressed_size = byte_offset;
    DeviceRuntime<DeviceType>::SyncDevice();

    Array<1, unsigned char, DeviceType> compressed_array(
        {(SIZE)compressed_size});
    SubArray compressed_subarray(compressed_array);

    SERIALIZED_TYPE *buffer = compressed_array.get_dv();
    void *buffer_p = (void *)buffer;
    byte_offset = 0;

    SerializeArray<SERIALIZED_TYPE>(compressed_subarray, serizalied_meta,
                                    metadata_size, byte_offset);
    SIZE lossless_size = lossless_compressed_subarray.getShape(0);
    SerializeArray<SIZE>(compressed_subarray, &lossless_size, 1, byte_offset);

    align_byte_offset<uint64_t>(byte_offset);
    SerializeArray<Byte>(compressed_subarray,
                         lossless_compressed_subarray.data(),
                         lossless_compressed_subarray.getShape(0), byte_offset);

    free(serizalied_meta);

    if (MemoryManager<DeviceType>::IsDevicePointer(original_data)) {
      if (!output_pre_allocated) {
        MemoryManager<DeviceType>::Malloc1D(compressed_data, compressed_size,
                                            0);
      }
      MemoryManager<DeviceType>::Copy1D(compressed_data,
                                        (void *)compressed_array.get_dv(),
                                        compressed_size, 0);
      DeviceRuntime<DeviceType>::SyncQueue(0);
    } else {
      if (!output_pre_allocated) {
        compressed_data = (unsigned char *)malloc(compressed_size);
      }
      memcpy(compressed_data, compressed_array.hostCopy(), compressed_size);
    }
  } else { // chunck by chunck
    if (config.timing) {
      std::cout << log::log_info
                << "Insufficient device memory for compressing the whole "
                   "dataset at once. "
                << "Orignial domain is decomposed into "
                << hierarchy.hierarchy_chunck.size() << " sub-domains\n";
    }
    std::vector<T *> decomposed_data;
    domain_decompose<D, T, DeviceType>((T *)original_data, decomposed_data,
                                       shape, hierarchy.domain_decomposed_dim,
                                       hierarchy.domain_decomposed_size);
    std::vector<SIZE> lossless_compressed_size;
    std::vector<Byte *> lossless_compressed_data;
    T norm = 1;
    if (type == error_bound_type::REL) {
      norm = calc_norm_decomposed<D, T, DeviceType>(
          decomposed_data, s, shape, hierarchy.domain_decomposed_dim,
          hierarchy.domain_decomposed_size);
      if (config.timing) {
        if (s == std::numeric_limits<T>::infinity()) {
          std::cout << log::log_info << "L_inf norm: " << norm << std::endl;
        } else {
          std::cout << log::log_info << "L_2 norm: " << norm << std::endl;
        }
      }
    }
    if (s != std::numeric_limits<T>::infinity()) {
      tol = std::sqrt((tol * tol) / hierarchy.hierarchy_chunck.size());
      if (config.timing) {
        std::cout << log::log_info << "local bound: " << tol << "\n";
      }
    }
    for (SIZE i = 0; i < hierarchy.hierarchy_chunck.size(); i++) {
      if (config.timing) {
        std::cout << log::log_info << "Compressing decomposed domain (" << i
                  << ") - ( ";
        for (DIM d = 0; d < D; d++)
          std::cout << hierarchy.hierarchy_chunck[i].shape_org[d] << " ";
        std::cout << ")\n";
      }
      Array<D, T, DeviceType> chunck_in_array(
          hierarchy.hierarchy_chunck[i].shape_org);
      chunck_in_array.load((const T *)decomposed_data[i]);
      Array<1, Byte, DeviceType> lossless_compressed_array =
          compress<D, T, DeviceType>(hierarchy.hierarchy_chunck[i],
                                     chunck_in_array, type, tol, s, norm,
                                     config);
      lossless_compressed_size.push_back(
          lossless_compressed_array.getShape()[0]);
      Byte *temp = NULL;
      MemoryManager<DeviceType>::MallocHost(
          temp, lossless_compressed_array.getShape()[0], 0);
      MemoryManager<DeviceType>::Copy1D(
          temp, lossless_compressed_array.get_dv(),
          lossless_compressed_array.getShape()[0], 0);
      DeviceRuntime<DeviceType>::SyncQueue(0);
      lossless_compressed_data.push_back(temp);
    }

    // Serialize
    if (type == error_bound_type::REL) {
      m.norm = norm;
    }
    m.domain_decomposed = true;
    m.ddtype = domain_decomposition_type::MaxDim;
    m.domain_decomposed_dim = hierarchy.domain_decomposed_dim;
    m.domain_decomposed_size = hierarchy.domain_decomposed_size;

    uint32_t metadata_size;
    SERIALIZED_TYPE *serizalied_meta = m.Serialize(metadata_size);

    SIZE byte_offset = 0;
    advance_with_align<SERIALIZED_TYPE>(byte_offset, metadata_size);
    for (uint32_t i = 0; i < hierarchy.hierarchy_chunck.size(); i++) {
      advance_with_align<SIZE>(byte_offset, 1);
      align_byte_offset<uint64_t>(
          byte_offset); // for zero copy when deserialize
      advance_with_align<Byte>(byte_offset, lossless_compressed_size[i]);
    }

    compressed_size = byte_offset;
    DeviceRuntime<DeviceType>::SyncDevice();

    if (!output_pre_allocated) {
      compressed_data = (unsigned char *)malloc(compressed_size);
    }

    byte_offset = 0;
    Serialize<SERIALIZED_TYPE, DeviceType>(
        (Byte *)compressed_data, serizalied_meta, metadata_size, byte_offset);
    for (uint32_t i = 0; i < hierarchy.hierarchy_chunck.size(); i++) {
      Serialize<SIZE, DeviceType>((Byte *)compressed_data,
                                  lossless_compressed_size.data() + i, 1,
                                  byte_offset);
      align_byte_offset<uint64_t>(byte_offset);
      Serialize<Byte, DeviceType>((Byte *)compressed_data,
                                  lossless_compressed_data[i],
                                  lossless_compressed_size[i], byte_offset);
    }
    free(serizalied_meta);
    for (SIZE i = 0; i < hierarchy.hierarchy_chunck.size(); i++) {
      MemoryManager<DeviceType>::FreeHost(lossless_compressed_data[i]);
      MemoryManager<DeviceType>::FreeHost(decomposed_data[i]);
    }
  }
}

template <DIM D, typename T, typename DeviceType>
void decompress(std::vector<SIZE> shape, const void *compressed_data,
                size_t compressed_size, void *&decompressed_data, Config config,
                bool output_pre_allocated) {
  size_t original_size = 1;
  for (int i = 0; i < D; i++)
    original_size *= shape[i];

  if (!infer_domain_decomposed(compressed_data, compressed_size)) {
    Hierarchy<D, T, DeviceType> hierarchy(shape, config.uniform_coord_mode);
    std::vector<SIZE> compressed_shape(1);
    compressed_shape[0] = compressed_size;
    Array<1, unsigned char, DeviceType> compressed_array(compressed_shape);
    compressed_array.load((const unsigned char *)compressed_data);

    // Deserialize
    SubArray compressed_subarray(compressed_array);
    // SIZE byte_offset = 0;

    Metadata m;
    // uint32_t metadata_size;
    // SERIALIZED_TYPE * metadata_size_prt = (SERIALIZED_TYPE*)&metadata_size;
    // byte_offset = m.metadata_size_offset();
    // DeserializeArray<SERIALIZED_TYPE>(compressed_subarray, metadata_size_prt,
    // sizeof(uint32_t), byte_offset, false); SERIALIZED_TYPE * serizalied_meta
    // = (SERIALIZED_TYPE *)std::malloc(metadata_size); byte_offset = 0;
    // DeserializeArray<SERIALIZED_TYPE>(compressed_subarray, serizalied_meta,
    // metadata_size, byte_offset, false);

    m.Deserialize((SERIALIZED_TYPE *)compressed_data);

    if (m.etype != CheckEndianess()) {
      std::cout << log::log_err
                << "This data was compressed on a machine with different "
                   "endianess!\n";
      exit(-1);
    }

    if (strcmp(m.magic_word, MAGIC_WORD) != 0) {
      std::cout << log::log_err
                << "This data was not compressed with MGARD or corrupted!\n";
      exit(-1);
    }

    if (m.ptype == processor_type::GPU_CUDA) {
      std::cout << log::log_err
                << "This data was compressed with legacy CUDA compressor!\n";
      exit(-1);
    }

    if (m.ptype != processor_type::X_Serial &&
        m.ptype != processor_type::X_CUDA && m.ptype != processor_type::X_HIP) {
      std::cout << log::log_err
                << "This data was not compressed with legacy MGARD-CUDA "
                   "compressor or MGARD-X compressor!\n";
      exit(-1);
    }

    config.decomposition = m.decomposition;
    config.lossless = m.ltype;
    config.huff_dict_size = m.huff_dict_size;
    config.huff_block_size = m.huff_block_size;
    config.reorder = m.reorder;
    // printf("config.reorder: %d\n", config.reorder);

    SIZE lossless_size;
    SIZE *lossless_size_ptr = &lossless_size;
    Byte *lossless_data;
    SIZE byte_offset = m.metadata_size;
    DeserializeArray<SIZE>(compressed_subarray, lossless_size_ptr, 1,
                           byte_offset, false);
    align_byte_offset<uint64_t>(byte_offset);
    DeserializeArray<Byte>(compressed_subarray, lossless_data, lossless_size,
                           byte_offset, true);

    Array<1, unsigned char, DeviceType> lossless_compressed_array(
        {lossless_size});
    lossless_compressed_array.load((const unsigned char *)lossless_data);
    // SubArray<1, Byte, DeviceType> lossless_compressed_subarray({(SIZE)
    // lossless_size}, lossless_data);

    config.lossless = m.ltype;

    Array<D, T, DeviceType> out_array =
        decompress<D, T, DeviceType>(hierarchy, lossless_compressed_array,
                                     m.ebtype, m.tol, m.s, m.norm, config);

    // PrintSubarray("out_array", SubArray(out_array));

    if (MemoryManager<DeviceType>::IsDevicePointer(compressed_data)) {
      if (!output_pre_allocated) {
        MemoryManager<DeviceType>::Malloc1D(decompressed_data,
                                            original_size * sizeof(T), 0);
      }
      MemoryManager<DeviceType>::Copy1D(decompressed_data,
                                        (void *)out_array.get_dv(),
                                        original_size * sizeof(T), 0);
      DeviceRuntime<DeviceType>::SyncQueue(0);
    } else {
      if (!output_pre_allocated) {
        decompressed_data = (T *)malloc(original_size * sizeof(T));
      }
      memcpy(decompressed_data, out_array.hostCopy(),
             original_size * sizeof(T));
    }
  } else { // domain decomposition
    // deserialized
    Metadata m;
    // uint32_t metadata_size = m.get_metadata_size
    // uint32_t * metadata_size_prt = &metadata_size;
    // SIZE byte_offset = m.metadata_size_offset();
    // Deserialize<uint32_t, DeviceType>((Byte*)compressed_data,
    // metadata_size_prt, 1, byte_offset); printf("decompress metadata_size:
    // %u\n", metadata_size); SERIALIZED_TYPE * serizalied_meta =
    // (SERIALIZED_TYPE *)std::malloc(metadata_size); byte_offset = 0;
    // Deserialize<SERIALIZED_TYPE, DeviceType>((Byte*)compressed_data,
    // serizalied_meta, metadata_size, byte_offset);

    m.Deserialize((SERIALIZED_TYPE *)compressed_data);

    if (m.etype != CheckEndianess()) {
      std::cout << log::log_err
                << "This data was compressed on a machine with different "
                   "endianess!\n";
      exit(-1);
    }

    if (strcmp(m.magic_word, MAGIC_WORD) != 0) {
      std::cout << log::log_err
                << "This data was not compressed with MGARD or corrupted!\n";
      exit(-1);
    }

    if (m.ptype == processor_type::GPU_CUDA) {
      std::cout << log::log_err
                << "This data was compressed with legacy CUDA compressor!\n";
      exit(-1);
    }

    if (m.ptype != processor_type::X_Serial &&
        m.ptype != processor_type::X_CUDA && m.ptype != processor_type::X_HIP) {
      std::cout << log::log_err
                << "This data was not compressed with legacy MGARD-CUDA "
                   "compressor or MGARD-X compressor!\n";
      exit(-1);
    }

    config.decomposition = m.decomposition;
    config.lossless = m.ltype;
    config.huff_dict_size = m.huff_dict_size;
    config.huff_block_size = m.huff_block_size;
    config.reorder = m.reorder;

    std::vector<SIZE> lossless_compressed_size;
    std::vector<Byte *> lossless_compressed_data;
    std::vector<T *> decomposed_data;

    Hierarchy<D, T, DeviceType> hierarchy(shape, m.domain_decomposed_dim,
                                          m.domain_decomposed_size,
                                          config.uniform_coord_mode);

    if (config.timing) {
      std::cout << log::log_info << "The original domain was decomposed into "
                << hierarchy.hierarchy_chunck.size()
                << " sub-domains during compression.\n";
    }

    SIZE byte_offset = m.metadata_size;
    for (uint32_t i = 0; i < hierarchy.hierarchy_chunck.size(); i++) {
      SIZE temp_size;
      Byte *temp_data;
      Deserialize<SIZE, DeviceType>((Byte *)compressed_data, &temp_size, 1,
                                    byte_offset);
      MemoryManager<DeviceType>::MallocHost(temp_data, temp_size, 0);
      DeviceRuntime<DeviceType>::SyncQueue(0);
      align_byte_offset<uint64_t>(byte_offset);
      Deserialize<Byte, DeviceType>((Byte *)compressed_data, temp_data,
                                    temp_size, byte_offset);
      lossless_compressed_size.push_back(temp_size);
      lossless_compressed_data.push_back(temp_data);
    }

    if (m.s != std::numeric_limits<T>::infinity()) {
      m.tol = std::sqrt((m.tol * m.tol) / hierarchy.hierarchy_chunck.size());
      if (config.timing) {
        std::cout << log::log_info << "local bound: " << m.tol << "\n";
      }
    }

    // decompress
    for (uint32_t i = 0; i < hierarchy.hierarchy_chunck.size(); i++) {
      if (config.timing) {
        std::cout << log::log_info << "Decompressing decomposed domain (" << i
                  << ") - ( ";
        for (DIM d = 0; d < D; d++)
          std::cout << hierarchy.hierarchy_chunck[i].shape_org[d] << " ";
        std::cout << ")\n";
      }
      Array<1, Byte, DeviceType> lossless_compressed_array(
          {lossless_compressed_size[i]});
      lossless_compressed_array.load(lossless_compressed_data[i]);
      Array<D, T, DeviceType> out_array = decompress<D, T, DeviceType>(
          hierarchy.hierarchy_chunck[i], lossless_compressed_array, m.ebtype,
          m.tol, m.s, m.norm, config);

      SIZE linearized_width = 1;
      for (DIM d = 1; d < D; d++)
        linearized_width *= out_array.getShape()[d];
      T *decompressed_data_chunck = NULL;
      MemoryManager<DeviceType>::MallocHost(
          decompressed_data_chunck, out_array.getShape()[0] * linearized_width,
          0);
      MemoryManager<DeviceType>::CopyND(
          decompressed_data_chunck, out_array.getShape()[0], out_array.get_dv(),
          out_array.get_ldvs_h()[0], out_array.getShape()[0], linearized_width,
          0);
      DeviceRuntime<DeviceType>::SyncQueue(0);
      decomposed_data.push_back(decompressed_data_chunck);

      // {
      //   Array<D, T, DeviceType> data_array({5, 5, 5});
      //   data_array.load(decompressed_data_chunck);
      //   PrintSubarray("decompressed_data_chunck", SubArray(data_array));
      // }
    }

    if (!output_pre_allocated) {
      decompressed_data = (T *)malloc(original_size * sizeof(T));
    }

    // domain recomposition
    domain_recompose<D, T, DeviceType>(decomposed_data, (T *)decompressed_data,
                                       shape, hierarchy.domain_decomposed_dim,
                                       hierarchy.domain_decomposed_size);

    for (SIZE i = 0; i < hierarchy.hierarchy_chunck.size(); i++) {
      MemoryManager<DeviceType>::FreeHost(lossless_compressed_data[i]);
      MemoryManager<DeviceType>::FreeHost(decomposed_data[i]);
    }
  }
}

template <DIM D, typename T, typename DeviceType>
void decompress(std::vector<SIZE> shape, const void *compressed_data,
                size_t compressed_size, void *&decompressed_data,
                std::vector<T *> coords, Config config,
                bool output_pre_allocated) {
  size_t original_size = 1;
  for (int i = 0; i < D; i++) {
    original_size *= shape[i];
  }

  if (!infer_domain_decomposed(compressed_data, compressed_size)) {

    Hierarchy<D, T, DeviceType> hierarchy(shape, coords);
    std::vector<SIZE> compressed_shape(1);
    compressed_shape[0] = compressed_size;
    Array<1, unsigned char, DeviceType> compressed_array(compressed_shape);
    compressed_array.load((const unsigned char *)compressed_data);

    // Deserialize
    SubArray compressed_subarray(compressed_array);
    // SIZE byte_offset = 0;

    Metadata m;
    // uint32_t metadata_size;
    // SERIALIZED_TYPE * metadata_size_prt = (SERIALIZED_TYPE*)&metadata_size;
    // byte_offset = m.metadata_size_offset();
    // DeserializeArray<SERIALIZED_TYPE>(compressed_subarray, metadata_size_prt,
    // sizeof(uint32_t), byte_offset, false); SERIALIZED_TYPE * serizalied_meta
    // = (SERIALIZED_TYPE *)std::malloc(metadata_size); byte_offset = 0;
    // DeserializeArray<SERIALIZED_TYPE>(compressed_subarray, serizalied_meta,
    // metadata_size, byte_offset, false);

    m.Deserialize((SERIALIZED_TYPE *)compressed_data);

    if (m.etype != CheckEndianess()) {
      std::cout << log::log_err
                << "This data was compressed on a machine with different "
                   "endianess!\n";
      exit(-1);
    }

    if (strcmp(m.magic_word, MAGIC_WORD) != 0) {
      std::cout << log::log_err
                << "This data was not compressed with MGARD or corrupted!\n";
      exit(-1);
    }

    if (m.ptype == processor_type::GPU_CUDA) {
      std::cout << log::log_err
                << "This data was compressed with legacy CUDA compressor!\n";
      exit(-1);
    }

    if (m.ptype != processor_type::X_Serial &&
        m.ptype != processor_type::X_CUDA && m.ptype != processor_type::X_HIP) {
      std::cout << log::log_err
                << "This data was not compressed with legacy MGARD-CUDA "
                   "compressor or MGARD-X compressor!\n";
      exit(-1);
    }

    config.decomposition = m.decomposition;
    config.lossless = m.ltype;
    config.huff_dict_size = m.huff_dict_size;
    config.huff_block_size = m.huff_block_size;
    config.reorder = m.reorder;

    SIZE lossless_size;
    SIZE *lossless_size_ptr = &lossless_size;
    Byte *lossless_data;
    SIZE byte_offset = m.metadata_size;
    DeserializeArray<SIZE>(compressed_subarray, lossless_size_ptr, 1,
                           byte_offset, false);
    align_byte_offset<uint64_t>(byte_offset);
    DeserializeArray<Byte>(compressed_subarray, lossless_data, lossless_size,
                           byte_offset, true);

    SubArray<1, Byte, DeviceType> lossless_compressed_subarray(
        {(SIZE)lossless_size}, lossless_data);

    Array<D, T, DeviceType> out_array = decompress<D, T, DeviceType>(
        hierarchy, compressed_array, m.ebtype, m.tol, m.s, m.norm, config);

    if (MemoryManager<DeviceType>::IsDevicePointer(compressed_data)) {
      if (!output_pre_allocated) {
        MemoryManager<DeviceType>::Malloc1D(decompressed_data,
                                            original_size * sizeof(T), 0);
      }
      MemoryManager<DeviceType>::Copy1D(decompressed_data,
                                        (void *)out_array.get_dv(),
                                        original_size * sizeof(T), 0);
      DeviceRuntime<DeviceType>::SyncQueue(0);

    } else {
      if (!output_pre_allocated) {
        decompressed_data = (T *)malloc(original_size * sizeof(T));
      }
      memcpy(decompressed_data, out_array.hostCopy(),
             original_size * sizeof(T));
    }
  } else { // domain decomposed
    // deserialized
    Metadata m;
    // uint32_t metadata_size;
    // uint32_t * metadata_size_prt = &metadata_size;
    // SIZE byte_offset = m.metadata_size_offset();
    // Deserialize<uint32_t, DeviceType>((Byte*)compressed_data,
    // metadata_size_prt, 1, byte_offset); SERIALIZED_TYPE * serizalied_meta =
    // (SERIALIZED_TYPE *)std::malloc(metadata_size); byte_offset = 0;
    // Deserialize<SERIALIZED_TYPE, DeviceType>((Byte*)compressed_data,
    // serizalied_meta, metadata_size, byte_offset);

    m.Deserialize((SERIALIZED_TYPE *)compressed_data);

    if (m.etype != CheckEndianess()) {
      std::cout << log::log_err
                << "This data was compressed on a machine with different "
                   "endianess!\n";
      exit(-1);
    }

    if (strcmp(m.magic_word, MAGIC_WORD) != 0) {
      std::cout << log::log_err
                << "This data was not compressed with MGARD or corrupted!\n";
      exit(-1);
    }

    if (m.ptype == processor_type::GPU_CUDA) {
      std::cout << log::log_err
                << "This data was compressed with legacy CUDA compressor!\n";
      exit(-1);
    }

    if (m.ptype != processor_type::X_Serial &&
        m.ptype != processor_type::X_CUDA && m.ptype != processor_type::X_HIP) {
      std::cout << log::log_err
                << "This data was not compressed with legacy MGARD-CUDA "
                   "compressor or MGARD-X compressor!\n";
      exit(-1);
    }

    config.decomposition = m.decomposition;
    config.lossless = m.ltype;
    config.huff_dict_size = m.huff_dict_size;
    config.huff_block_size = m.huff_block_size;
    config.reorder = m.reorder;

    std::vector<SIZE> lossless_compressed_size;
    std::vector<Byte *> lossless_compressed_data;
    std::vector<T *> decomposed_data;

    Hierarchy<D, T, DeviceType> hierarchy(shape, m.domain_decomposed_dim,
                                          m.domain_decomposed_size, coords);

    if (config.timing) {
      std::cout << log::log_info << "The original domain was decomposed into "
                << hierarchy.hierarchy_chunck.size()
                << " sub-domains during compression.\n";
    }

    SIZE byte_offset = m.metadata_size;
    for (uint32_t i = 0; i < hierarchy.hierarchy_chunck.size(); i++) {
      SIZE temp_size;
      Byte *temp_data;
      Deserialize<SIZE, DeviceType>((Byte *)compressed_data, &temp_size, 1,
                                    byte_offset);
      MemoryManager<DeviceType>::MallocHost(temp_data, temp_size, 0);
      DeviceRuntime<DeviceType>::SyncQueue(0);
      align_byte_offset<uint64_t>(byte_offset);
      Deserialize<Byte, DeviceType>((Byte *)compressed_data, temp_data,
                                    temp_size, byte_offset);
      lossless_compressed_size.push_back(temp_size);
      lossless_compressed_data.push_back(temp_data);
    }

    if (m.s != std::numeric_limits<T>::infinity()) {
      m.tol = std::sqrt((m.tol * m.tol) / hierarchy.hierarchy_chunck.size());
      if (config.timing) {
        std::cout << log::log_info << "local bound: " << m.tol << "\n";
      }
    }

    // decompress
    for (uint32_t i = 0; i < hierarchy.hierarchy_chunck.size(); i++) {
      if (config.timing) {
        std::cout << log::log_info << "Decompressing decomposed domain (" << i
                  << ") - ( ";
        for (DIM d = 0; d < D; d++)
          std::cout << hierarchy.hierarchy_chunck[i].shape_org[d] << " ";
        std::cout << ")\n";
      }
      Array<1, Byte, DeviceType> lossless_compressed_array(
          {lossless_compressed_size[i]});
      lossless_compressed_array.load(lossless_compressed_data[i]);
      Array<D, T, DeviceType> out_array = decompress<D, T, DeviceType>(
          hierarchy.hierarchy_chunck[i], lossless_compressed_array, m.ebtype,
          m.tol, m.s, m.norm, config);
      SIZE linearized_width = 1;
      for (DIM d = 1; d < D; d++)
        linearized_width *= out_array.getShape()[d];
      T *decompressed_data_chunck = NULL;
      MemoryManager<DeviceType>::MallocHost(
          decompressed_data_chunck, out_array.getShape()[0] * linearized_width,
          0);
      MemoryManager<DeviceType>::CopyND(
          decompressed_data_chunck, out_array.getShape()[0], out_array.get_dv(),
          out_array.get_ldvs_h()[0], out_array.getShape()[0], linearized_width,
          0);
      DeviceRuntime<DeviceType>::SyncQueue(0);
      decomposed_data.push_back(decompressed_data_chunck);
    }

    if (!output_pre_allocated) {
      decompressed_data = (T *)malloc(original_size * sizeof(T));
    }

    // domain recomposition
    domain_recompose<D, T, DeviceType>(decomposed_data, (T *)decompressed_data,
                                       shape, hierarchy.domain_decomposed_dim,
                                       hierarchy.domain_decomposed_size);

    for (SIZE i = 0; i < hierarchy.hierarchy_chunck.size(); i++) {
      MemoryManager<DeviceType>::FreeHost(lossless_compressed_data[i]);
      MemoryManager<DeviceType>::FreeHost(decomposed_data[i]);
    }
  }
}

template <typename DeviceType> void BeginAutoTuning() {
  AutoTuner<DeviceType>::ProfileKernels = true;
}

template <typename DeviceType> void EndAutoTuning() {
  AutoTuner<DeviceType>::ProfileKernels = false;
}

} // namespace mgard_x

#endif