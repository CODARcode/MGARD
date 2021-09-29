/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */
#include "cuda/CommonInternal.h"

#include "cuda/Metadata.h"

#include "MGARDConfig.hpp"


namespace mgard_cuda {

SERIALIZED_TYPE *Metadata::Serialize(uint32_t &total_size) {
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
  total_size += sizeof(l_target); // l_target;
  total_size += sizeof(ltype);
  if (ltype == lossless_type::GPU_Huffman ||
      ltype == lossless_type::GPU_Huffman_LZ4) {
    total_size += sizeof(dict_size); // dict size
  }

  // about data
  total_size += sizeof(dtype);
  total_size += sizeof(etype);
  total_size += sizeof(dstype);
  total_size += sizeof(total_dims);            // total_dims;
  total_size += sizeof(shape[0]) * total_dims; // shape;
  if (dstype == data_structure_type::Cartesian_Grid_Non_Uniform) {
    total_size += sizeof(cltype);
    if (cltype == coordinate_location::Embedded) {
      size_t coord_size = 0;
      for (DIM d = 0; d < total_dims; d++) {
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
  SERIALIZED_TYPE *serialized_data = (SERIALIZED_TYPE *)std::malloc(total_size);
  SERIALIZED_TYPE *p = serialized_data;
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
  if (ltype == lossless_type::GPU_Huffman ||
      ltype == lossless_type::GPU_Huffman_LZ4) {
    Serialize(dict_size, p);
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
    } else if (cltype == coordinate_location::External) {
      Serialize(nonuniform_coords_file, p);
    }
  }

  return serialized_data;
}

void Metadata::Deserialize(SERIALIZED_TYPE *serialized_data,
                           uint32_t &total_size) {
  SERIALIZED_TYPE *p = serialized_data;

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
  if (ltype == lossless_type::GPU_Huffman ||
      ltype == lossless_type::GPU_Huffman_LZ4) {
    Deserialize(dict_size, p);
  }

  Deserialize(dtype, p);
  Deserialize(etype, p);
  Deserialize(dstype, p);
  Deserialize(total_dims, p);
  shape = new uint64_t[total_dims];
  Deserialize(shape, total_dims, p);
  if (dstype == data_structure_type::Cartesian_Grid_Non_Uniform) {
    Deserialize(cltype, p);
    if (cltype == coordinate_location::Embedded) {
      Deserialize(coords, shape, dtype, p);
    } else if (cltype == coordinate_location::External) {
      Deserialize(nonuniform_coords_file, p);
    }
  }
  total_size = p - serialized_data;
}

} // namespace mgard_cuda