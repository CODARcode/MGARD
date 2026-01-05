/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../Config/Config.h"
#include "../RuntimeX/RuntimeX.h"
#include "../Utilities/Types.h"
#include "MGARDConfig.hpp"
#include <cstring>
#include <zlib.h>

#ifndef MGARD_X_METADATA
#define MGARD_X_METADATA

namespace mgard_x {

struct MetadataBase {
  // about MGARD software
  uint8_t software_version[3];
  uint8_t file_version[3];
  uint32_t metadata_size = 0;
  uint32_t metadata_crc32 = 0;

  enum endiness_type etype;

  // about data
  enum data_type dtype;
  enum data_structure_type dstype;
  uint64_t total_dims = 0;
  std::vector<uint64_t> shape;
  char *nonuniform_coords_file;
  std::vector<std::vector<double>> coords;

  enum decomposition_type decomposition;
  uint32_t l_target = 0;
  uint32_t reorder = 0;

  bool domain_decomposed = false;
  enum domain_decomposition_type ddtype;
  uint8_t domain_decomposed_dim;
  uint64_t domain_decomposed_size;

  enum operation_type otype;

  // about MDR
  enum bitplane_encoding_type betype;
  uint64_t number_bitplanes;

  // about compression
  enum error_bound_type ebtype;
  double norm = 0; // optional
  double tol = 0;
  enum norm_type ntype;
  double s = 0; // optional

  enum lossless_type ltype;
  uint32_t huff_dict_size = 0;  // optional (for Huffman)
  uint32_t huff_block_size = 0; // optional (for Huffman)

  enum processor_type ptype;

  void InitializeConfig(Config &config);
  void PrintSummary();

  std::vector<SERIALIZED_TYPE> Serialize();
  void Deserialize(const std::vector<SERIALIZED_TYPE> &serialized_data);

protected:
  static uint64_t SerializePreambleSize();
  uint64_t DeserializeSize(std::vector<SERIALIZED_TYPE>::const_iterator &iter);
};

template <typename DeviceType> struct Metadata : MetadataBase {
  using Mem = MemoryManager<DeviceType>;
  template <typename T>
  void FillForCompression(enum error_bound_type ebtype, T tol, T s, T norm,
                          enum decomposition_type decomposition,
                          uint32_t reorder, enum lossless_type ltype,
                          uint32_t huff_dict_size, uint32_t huff_block_size,
                          std::vector<SIZE> shape, bool domain_decomposed,
                          domain_decomposition_type ddtype,
                          uint8_t domain_decomposed_dim,
                          uint64_t domain_decomposed_size) {

    otype = operation_type::Compression;
    if (std::is_same<DeviceType, SERIAL>::value) {
      this->ptype = processor_type::X_SERIAL;
    } else if (std::is_same<DeviceType, OPENMP>::value) {
      this->ptype = processor_type::X_OPENMP;
    } else if (std::is_same<DeviceType, CUDA>::value) {
      this->ptype = processor_type::X_CUDA;
    } else if (std::is_same<DeviceType, HIP>::value) {
      this->ptype = processor_type::X_HIP;
    } else if (std::is_same<DeviceType, SYCL>::value) {
      this->ptype = processor_type::X_SYCL;
    }
    this->ebtype = ebtype;
    this->tol = (double)tol;
    if (s == std::numeric_limits<T>::infinity()) {
      this->ntype = norm_type::L_Inf;
      this->s = (double)s;
    } else {
      this->ntype = norm_type::L_2;
      this->s = (double)s;
    }
    this->norm = norm;
    this->decomposition = decomposition;
    this->reorder = reorder;
    this->ltype = ltype;
    this->huff_dict_size = huff_dict_size;
    this->huff_block_size = huff_block_size;
    this->dtype =
        std::is_same<T, double>::value ? data_type::Double : data_type::Float;
    this->dstype = data_structure_type::Cartesian_Grid_Uniform;
    this->total_dims = shape.size();
    this->shape = std::vector<uint64_t>(this->total_dims);
    for (int d = 0; d < this->total_dims; d++) {
      this->shape[d] = (uint64_t)shape[d];
    }
    this->domain_decomposed = domain_decomposed;
    this->ddtype = ddtype;
    this->domain_decomposed_dim = domain_decomposed_dim;
    this->domain_decomposed_size = domain_decomposed_size;
  }

  template <typename T>
  void
  FillForCompression(enum error_bound_type ebtype, T tol, T s, T norm,
                     enum decomposition_type decomposition, uint32_t reorder,
                     enum lossless_type ltype, uint32_t huff_dict_size,
                     uint32_t huff_block_size, std::vector<SIZE> shape,
                     bool domain_decomposed, domain_decomposition_type ddtype,
                     uint8_t domain_decomposed_dim,
                     uint64_t domain_decomposed_size, std::vector<T *> coords) {
    FillForCompression(ebtype, tol, s, norm, decomposition, reorder, ltype,
                       huff_dict_size, huff_block_size, shape,
                       domain_decomposed, ddtype, domain_decomposed_dim,
                       domain_decomposed_size);
    for (int d = 0; d < this->total_dims; d++) {
      std::vector<double> coord(shape[d]);
      T *coord_h = new T[shape[d]];
      MemoryManager<DeviceType>::Copy1D(coord_h, coords[d], shape[d]);
      for (SIZE i = 0; i < shape[d]; i++) {
        coord[i] = (double)coord_h[i];
      }
      this->coords.push_back(coord);
      delete coord_h;
    }
    this->dstype = data_structure_type::Cartesian_Grid_Non_Uniform;
  }

  template <typename T>
  void FillForMDR(T norm, enum decomposition_type decomposition,
                  enum lossless_type ltype, uint32_t huff_dict_size,
                  uint32_t huff_block_size, std::vector<SIZE> shape,
                  bool domain_decomposed, domain_decomposition_type ddtype,
                  uint8_t domain_decomposed_dim,
                  uint64_t domain_decomposed_size, uint64_t number_bitplanes) {
    otype = operation_type::MDR;
    if (std::is_same<DeviceType, SERIAL>::value) {
      this->ptype = processor_type::X_SERIAL;
    } else if (std::is_same<DeviceType, OPENMP>::value) {
      this->ptype = processor_type::X_OPENMP;
    } else if (std::is_same<DeviceType, CUDA>::value) {
      this->ptype = processor_type::X_CUDA;
    } else if (std::is_same<DeviceType, HIP>::value) {
      this->ptype = processor_type::X_HIP;
    } else if (std::is_same<DeviceType, SYCL>::value) {
      this->ptype = processor_type::X_SYCL;
    }
    this->norm = norm;
    this->decomposition = decomposition;
    this->ltype = ltype;
    this->huff_dict_size = huff_dict_size;
    this->huff_block_size = huff_block_size;
    this->dtype =
        std::is_same<T, double>::value ? data_type::Double : data_type::Float;
    this->dstype = data_structure_type::Cartesian_Grid_Uniform;
    this->total_dims = shape.size();
    this->shape = std::vector<uint64_t>(this->total_dims);
    for (int d = 0; d < this->total_dims; d++) {
      this->shape[d] = (uint64_t)shape[d];
    }
    this->domain_decomposed = domain_decomposed;
    this->ddtype = ddtype;
    this->domain_decomposed_dim = domain_decomposed_dim;
    this->domain_decomposed_size = domain_decomposed_size;
    this->betype = bitplane_encoding_type::GroupedBitplaneEncoding;
    this->number_bitplanes = number_bitplanes;
  }

  template <typename T>
  void FillForMDR(T norm, enum decomposition_type decomposition,
                  enum lossless_type ltype, uint32_t huff_dict_size,
                  uint32_t huff_block_size, std::vector<SIZE> shape,
                  bool domain_decomposed, domain_decomposition_type ddtype,
                  uint8_t domain_decomposed_dim,
                  uint64_t domain_decomposed_size, uint64_t number_bitplanes,
                  std::vector<T *> coords) {
    FillForMDR(norm, decomposition, ltype, huff_dict_size, huff_block_size,
               shape, domain_decomposed, ddtype, domain_decomposed_dim,
               domain_decomposed_size, number_bitplanes);
    for (int d = 0; d < this->total_dims; d++) {
      std::vector<double> coord(shape[d]);
      T *coord_h = new T[shape[d]];
      MemoryManager<DeviceType>::Copy1D(coord_h, coords[d], shape[d]);
      for (SIZE i = 0; i < shape[d]; i++) {
        coord[i] = (double)coord_h[i];
      }
      this->coords.push_back(coord);
      delete coord_h;
    }
    this->dstype = data_structure_type::Cartesian_Grid_Non_Uniform;
  }

  SERIALIZED_TYPE *Serialize(uint32_t &total_size) {
    std::vector<SERIALIZED_TYPE> data_h = MetadataBase::Serialize();
    total_size = data_h.size();
    SERIALIZED_TYPE *data_d;
    Mem::Malloc1D(data_d, total_size, 0);
    Mem::Copy1D(data_d, data_h.data(), total_size, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
    return data_d;
  }
  void Deserialize(const SERIALIZED_TYPE *serialized_data) {
    // Do a partial deserialize to get the size of the buffer.
    std::vector<SERIALIZED_TYPE> data_h(MetadataBase::SerializePreambleSize());
    Mem::Copy1D(data_h.data(), serialized_data, data_h.size(), 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
    auto data_h_iter = data_h.cbegin();
    std::uint64_t total_size = MetadataBase::DeserializeSize(data_h_iter);

    data_h.resize(total_size);
    Mem::Copy1D(data_h.data(), serialized_data, total_size, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
    MetadataBase::Deserialize(data_h);
  }
};

bool verify(const void *compressed_data, size_t compressed_size);
enum data_type infer_data_type(const void *compressed_data,
                               size_t compressed_size);
std::vector<SIZE> infer_shape(const void *compressed_data,
                              size_t compressed_size);
enum data_structure_type infer_data_structure(const void *compressed_data,
                                              size_t compressed_size);
template <typename T>
std::vector<T *> infer_coords(const void *compressed_data,
                              size_t compressed_size);

std::string infer_nonuniform_coords_file(const void *compressed_data,
                                         size_t compressed_size);

bool infer_domain_decomposed(const void *compressed_data,
                             size_t compressed_size);

} // namespace mgard_x

#endif