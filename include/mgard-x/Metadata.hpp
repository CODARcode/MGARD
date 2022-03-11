/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#include "MGARDConfig.hpp"
#include "Types.h"
#include <cstring>

#ifndef MGARD_X_METADATA
#define MGARD_X_METADATA

#define SIGNATURE "MGARD"
#define SIGNATURE_SIZE 5

namespace mgard_x {

struct Metadata {
  // about MGARD software
  char mgard_signature[SIGNATURE_SIZE+1] = SIGNATURE;
  std::vector<char> signature;
  uint8_t software_version[3];
  uint8_t file_version[3];
  uint32_t metadata_size;
  uint32_t metadata_crc32;
  enum processor_type ptype;

  // about compression
  enum error_bound_type ebtype;
  double norm; // optional
  double tol;
  enum norm_type ntype;
  double s; // optional
  enum decomposition_type decomposition;
  uint32_t l_target;
  uint32_t reorder;
  enum lossless_type ltype;
  uint32_t huff_dict_size;  // optional (for Huffman)
  uint32_t huff_block_size; // optional (for Huffman)

  // about data
  enum data_type dtype;
  enum endiness_type etype;
  enum data_structure_type dstype;
  uint64_t total_dims = 0;
  std::vector<uint64_t> shape;
  char *nonuniform_coords_file;
  std::vector<std::vector<double>> coords;
  bool domain_decomposed = false;
  enum domain_decomposition_type ddtype;
  uint8_t domain_decomposed_dim;
  uint64_t domain_decomposed_size;

public:
  SERIALIZED_TYPE *Serialize(uint32_t &total_size) {
    return SerializeAll(total_size);
  }
  void Deserialize(SERIALIZED_TYPE *serialized_data) {
    DeserializeAll(serialized_data);
  }

private:
  SERIALIZED_TYPE *SerializeAll(uint32_t &total_size) {
        signature = std::vector<char>(SIGNATURE_SIZE);
    for (size_t i = 0; i < SIGNATURE_SIZE; i++) {
      signature[i] = mgard_signature[i];
    }

    total_size = 0;

    // about MGARD software
    total_size += SIGNATURE_SIZE;
    total_size += sizeof(software_version);
    total_size += sizeof(file_version);
    total_size += sizeof(metadata_size);
    total_size += sizeof(metadata_crc32);
    total_size += sizeof(ptype);

    // about compression
    total_size += sizeof(ebtype);
    if (ebtype == error_bound_type::REL) {
      total_size += sizeof(norm); // norm
    }
    total_size += sizeof(tol); // tol
    total_size += sizeof(ntype);
    // if (ntype == norm_type::L_2) {
    total_size += sizeof(s); // s
    //}
    total_size += sizeof(decomposition);
    total_size += sizeof(l_target); // l_target;
    total_size += sizeof(reorder);
    total_size += sizeof(ltype);
    if (ltype == lossless_type::Huffman ||
        ltype == lossless_type::Huffman_LZ4 ||
        ltype == lossless_type::Huffman_Zstd) {
      total_size += sizeof(huff_dict_size);  // dict size
      total_size += sizeof(huff_block_size); // block size
    }

    // about data
    total_size += sizeof(dtype);
    total_size += sizeof(etype);
    total_size += sizeof(dstype);
    total_size += sizeof(total_dims);            // total_dims;
    total_size += sizeof(shape[0]) * total_dims; // shape;
    if (dstype == data_structure_type::Cartesian_Grid_Non_Uniform) {
      size_t coord_size = 0;
      for (DIM d = 0; d < total_dims; d++) {
        coord_size += shape[d] * sizeof(double);
      }
      total_size += coord_size;
    }

    total_size += sizeof(domain_decomposed);
    if (domain_decomposed) {
      total_size += sizeof(ddtype);
      total_size += sizeof(domain_decomposed_dim);
      total_size += sizeof(domain_decomposed_size);
    }

    // initialize some fields
    metadata_size = total_size;

    software_version[0] = MGARD_VERSION_MAJOR;
    software_version[1] = MGARD_VERSION_MINOR;
    software_version[2] = MGARD_VERSION_PATCH;

    file_version[0] = MGARD_FILE_VERSION_MAJOR;
    file_version[1] = MGARD_FILE_VERSION_MINOR;
    file_version[2] = MGARD_FILE_VERSION_PATCH;

    // to be replaced with actual CRC-32 checksum
    metadata_crc32 = 0;

    // start serializing
    SERIALIZED_TYPE *serialized_data =
        (SERIALIZED_TYPE *)std::malloc(total_size);
    SERIALIZED_TYPE *p = serialized_data;

    SerializeSignature(p);
    Serialize(software_version, p);
    Serialize(file_version, p);
    Serialize(metadata_size, p);
    Serialize(metadata_crc32, p);
    Serialize(ptype, p);

    Serialize(ebtype, p);
    if (ebtype == error_bound_type::REL) {
      Serialize(norm, p);
    }
    Serialize(tol, p);
    Serialize(ntype, p);
    // if (ntype == norm_type::L_2) {
    Serialize(s, p);
    //}
    Serialize(decomposition, p);
    Serialize(l_target, p);
    Serialize(reorder, p);
    Serialize(ltype, p);
    if (ltype == lossless_type::Huffman ||
        ltype == lossless_type::Huffman_LZ4 ||
        ltype == lossless_type::Huffman_Zstd) {
      Serialize(huff_dict_size, p);
      Serialize(huff_block_size, p);
    }

    Serialize(dtype, p);
    Serialize(etype, p);
    Serialize(dstype, p);
    Serialize(total_dims, p);
    SerializeShape(shape, p);
    if (dstype == data_structure_type::Cartesian_Grid_Non_Uniform) {
      SerializeCoords(coords, p);
    }

    Serialize(domain_decomposed, p);
    if (domain_decomposed) {
      Serialize(ddtype, p);
      Serialize(domain_decomposed_dim, p);
      Serialize(domain_decomposed_size, p);
    }
    return serialized_data;
  }

  void DeserializeAll(SERIALIZED_TYPE *serialized_data) {
    SERIALIZED_TYPE *p = serialized_data;

    DeserializeSignature(p);
    Deserialize(software_version, p);
    Deserialize(file_version, p);
    Deserialize(metadata_size, p);
    Deserialize(metadata_crc32, p);
    Deserialize(ptype, p);

    Deserialize(ebtype, p);
    if (ebtype == error_bound_type::REL) {
      Deserialize(norm, p);
    }
    Deserialize(tol, p);
    Deserialize(ntype, p);
    // if (ntype == norm_type::L_2) {
    Deserialize(s, p);
    //}
    Deserialize(decomposition, p);
    Deserialize(l_target, p);
    Deserialize(reorder, p);
    Deserialize(ltype, p);
    if (ltype == lossless_type::Huffman ||
        ltype == lossless_type::Huffman_LZ4 ||
        ltype == lossless_type::Huffman_Zstd) {
      Deserialize(huff_dict_size, p);
      Deserialize(huff_block_size, p);
    }

    Deserialize(dtype, p);
    Deserialize(etype, p);
    Deserialize(dstype, p);
    Deserialize(total_dims, p);
    DeserializeShape(shape, p);
    if (dstype == data_structure_type::Cartesian_Grid_Non_Uniform) {
      DeserializeCoords(coords, p);
    }

    Deserialize(domain_decomposed, p);
    if (domain_decomposed) {
      Deserialize(ddtype, p);
      Deserialize(domain_decomposed_dim, p);
      Deserialize(domain_decomposed_size, p);
    }
  }

  template <typename T> void Serialize(T &item, SERIALIZED_TYPE *&p) {
    std::memcpy(p, &item, sizeof(item));
    p += sizeof(item);
  }

  void SerializeSignature(SERIALIZED_TYPE *&p) {
    std::memcpy(p, signature.data(), SIGNATURE_SIZE);
    p += SIGNATURE_SIZE;
  }

  void SerializeShape(std::vector<uint64_t>& shape, SERIALIZED_TYPE *&p) {
    std::memcpy(p, shape.data(), sizeof(uint64_t) * total_dims);
    p += sizeof(uint64_t) * total_dims;
  }

  void SerializeCoords(std::vector<std::vector<double>> &coords,
                 SERIALIZED_TYPE *&p) {
    for (size_t d = 0; d < coords.size(); d++) {
      std::memcpy(p, coords[d].data(), sizeof(double) * shape[d]);
      p += sizeof(double) * shape[d];
    }
  }

  template <typename T> void Deserialize(T &item, SERIALIZED_TYPE *&p) {
    std::memcpy(&item, p, sizeof(item));
    p += sizeof(item);
  }

  void DeserializeSignature(SERIALIZED_TYPE *&p) {
    signature = std::vector<char>(SIGNATURE_SIZE);
    std::memcpy(signature.data(), p, SIGNATURE_SIZE);
    p += SIGNATURE_SIZE;
  }

  void DeserializeShape(std::vector<uint64_t>& shape, SERIALIZED_TYPE *&p) {
    shape = std::vector<uint64_t>(total_dims);
    std::memcpy(shape.data(), p, sizeof(uint64_t) * total_dims);
    p += sizeof(uint64_t) * total_dims;
  }

  void DeserializeCoords(std::vector<std::vector<double>> &coords, 
                         SERIALIZED_TYPE *&p) {
    coords = std::vector<std::vector<double>>(total_dims);
    for (size_t d = 0; d < total_dims; d++) {
      coords[d] = std::vector<double>(shape[d]);
      std::memcpy(coords[d].data(), p, sizeof(double) * shape[d]);
      p += sizeof(double) * shape[d];
    }
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