/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#ifndef MGRAD_CUDA_METADATA
#define MGRAD_CUDA_METADATA

#define MAGIC_WORD "MGARD"
#define MAGIC_WORD_SIZE 5

namespace mgard_cuda {

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
  uint32_t dict_size; // optional (for GPU_Huffman)

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
  SERIALIZED_TYPE *Serialize(uint32_t &total_size);
  void Deserialize(SERIALIZED_TYPE * serialized_data, 
                              uint32_t &total_size);
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
}

#endif