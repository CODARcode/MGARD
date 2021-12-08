/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#include "MGARDConfig.hpp"
#include "Types.h"

#ifndef MGARD_X_METADATA
#define MGARD_X_METADATA


#include <cstring>

#define MAGIC_WORD "MGARD"
#define MAGIC_WORD_SIZE 5

namespace mgard_x {

struct Metadata {
  // about MGARD software
  char magic_word[MAGIC_WORD_SIZE+1] = MAGIC_WORD;
  uint8_t software_version[3];
  uint8_t file_version[3];
  uint32_t metadata_size;
  uint32_t metadata_crc32;
  enum processor_type ptype;

  // about compression
  enum error_bound_type ebtype;
  double norm; //optional
  double tol;
  enum norm_type ntype;
  double s; // optional
  uint32_t l_target;
  enum lossless_type ltype;
  uint32_t huff_dict_size; // optional (for GPU_Huffman)
  uint32_t huff_block_size; // optional (for GPU_Huffman)

  // about data
  enum data_type dtype;
  enum endiness_type etype;
  enum data_structure_type dstype;
  uint8_t total_dims = 0;
  uint64_t * shape;
  enum coordinate_location cltype;
  char * nonuniform_coords_file;
  std::vector<Byte *> coords;
  
  public:
  SERIALIZED_TYPE *Serialize(uint32_t &total_size) {
    total_size = 0;

    // about MGARD software
    total_size += sizeof(char) * strlen(magic_word);
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
    if (ntype == norm_type::L_2) {
      total_size += sizeof(s); // s
    }
    total_size += sizeof(l_target); //l_target;
    total_size += sizeof(ltype);
    if (ltype == lossless_type::Huffman || 
        ltype == lossless_type::Huffman_LZ4 ||
        ltype == lossless_type::Huffman_Zstd) {
      total_size += sizeof(huff_dict_size); // dict size
      total_size += sizeof(huff_block_size); // block size
    }

    // about data
    total_size += sizeof(dtype);
    total_size += sizeof(etype);
    total_size += sizeof(dstype);
    total_size += sizeof(total_dims); // total_dims;
    total_size += sizeof(shape[0]) * total_dims; // shape;
    if (dstype == data_structure_type::Cartesian_Grid_Non_Uniform) {
      total_size += sizeof(cltype);
      if (cltype == coordinate_location::Embedded) {
        size_t coord_size = 0;
        for (DIM d = 0; d < total_dims; d ++) {
          if (dtype == data_type::Float) {
            coord_size += shape[d] * sizeof(float);
          } else if (dtype == data_type::Double) {
            coord_size += shape[d] * sizeof(double);
          }
        }
        total_size += coord_size;
      } else if (cltype == coordinate_location::External) {
        total_size += sizeof(char) * strlen(nonuniform_coords_file);
      }
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
    SERIALIZED_TYPE * serialized_data = (SERIALIZED_TYPE *)std::malloc(total_size);
    SERIALIZED_TYPE * p = serialized_data;
    Serialize(&magic_word[0], p);
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
    if (ntype == norm_type::L_2) {
      Serialize(s, p);
    }
    Serialize(l_target, p);
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
    Serialize(shape, total_dims, p);
    if (dstype == data_structure_type::Cartesian_Grid_Non_Uniform) {
      Serialize(cltype, p);
      if (cltype == coordinate_location::Embedded) {
        Serialize(coords, shape, dtype, p);
      } else if (cltype == coordinate_location::External){
        Serialize(nonuniform_coords_file, p);
      }
    }
    self_initialized = false;
    return serialized_data;
  }
  void Deserialize(SERIALIZED_TYPE * serialized_data, 
                              uint32_t &total_size) {
    SERIALIZED_TYPE * p = serialized_data;

    Deserialize(&magic_word[0], p);
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
    if (ntype == norm_type::L_2) {
      Deserialize(s, p);
    }
    Deserialize(l_target, p);
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
    shape = new uint64_t[total_dims];
    Deserialize(shape, total_dims, p);

    if (dstype == data_structure_type::Cartesian_Grid_Non_Uniform) {
      // printf("Deserialize Non_Uniform\n");
      Deserialize(cltype, p);
      if (cltype == coordinate_location::Embedded) {
        coords = std::vector<Byte *>(total_dims);
        Deserialize(coords, shape, dtype, p);
      } else if (cltype == coordinate_location::External) {
        Deserialize(nonuniform_coords_file, p);
      }
    }
    total_size = p - serialized_data;
    self_initialized = true;
  }
  size_t metadata_size_offset() {
    size_t offset = 0;
    offset += strlen(magic_word);
    offset += sizeof(software_version);
    offset += sizeof(file_version);
    return offset;
  }
  ~Metadata() {
    if (self_initialized) {
      delete[] shape;
      if (dstype == data_structure_type::Cartesian_Grid_Non_Uniform) {
        for (size_t d = 0; d < total_dims; d++) {
          delete[] coords[d];
        }
      }
    }
  }

  private:

  template <typename T>
  void Serialize(T &item, SERIALIZED_TYPE * &p) {
    std::memcpy(p, &item, sizeof(item)); 
    p += sizeof(item);
  }
  void Serialize(char *item, SERIALIZED_TYPE * &p) {
    std::memcpy(p, item, strlen(item)); 
    p += strlen(item);
  } 
  template <typename T, typename N>
  void Serialize(T *&item, N n, SERIALIZED_TYPE * &p) {
    std::memcpy(p, item, sizeof(T) * n); 
    p += sizeof(T) * n;
  } 

  void Serialize(std::vector<Byte *>& coords, uint64_t * shape, enum data_type dtype, SERIALIZED_TYPE * &p) {
    for (size_t i = 0; i < coords.size(); i++) {
      if (dtype == data_type::Float) {
        Serialize(coords[i], shape[i] * sizeof(float), p);
      } else if (dtype == data_type::Double) {
        Serialize(coords[i], shape[i] * sizeof(double), p);
      }
    }
  } 

  template <typename T>
  void Deserialize(T &item, SERIALIZED_TYPE * &p) {
    std::memcpy(&item, p, sizeof(item)); 
    p += sizeof(item);
  }
  void Deserialize(char *item, SERIALIZED_TYPE * &p) {
    std::memcpy(item, p, strlen(item)); 
    p += strlen(item);
  } 
  template <typename T, typename N>
  void Deserialize(T *&item, N n, SERIALIZED_TYPE * &p) {
    std::memcpy(item, p, sizeof(T) * n); 
    p += sizeof(T) * n;
  } 

  void Deserialize(std::vector<Byte *>& coords, uint64_t * shape, enum data_type dtype, SERIALIZED_TYPE * &p) {
    for (size_t i = 0; i < coords.size(); i++) {
      if (dtype == data_type::Float) {
        coords[i] = (Byte *)std::malloc(shape[i] * sizeof(float));
        Deserialize(coords[i], shape[i] * sizeof(float), p);
      } else if (dtype == data_type::Double) {
        coords[i] = (Byte *)std::malloc(shape[i] * sizeof(double));
        Deserialize(coords[i], shape[i] * sizeof(double), p);
      }
    }
  }

  bool self_initialized;
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


}

#endif