/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Kenneth Moreland (morelandkd@ornl.gov)
 * Date: January 5, 2026
 */

#include "mgard-x/Metadata/Metadata.hpp"
#include "format.hpp"
#include "proto/mgard.pb.h"

#include <google/protobuf/io/coded_stream.h>

using mgard_x::SERIALIZED_TYPE;

using SerializedIter = std::vector<SERIALIZED_TYPE>::const_iterator;

namespace {

template <typename Int> bool big_endian() {
  static_assert(std::is_integral<Int>::value,
                "can only check endianness of integral types");
  const Int n = 1;
  return not *reinterpret_cast<unsigned char const *>(&n);
}

uint32_t ComputeCRC32(const std::vector<SERIALIZED_TYPE> &data,
                      std::size_t start = 0) {
  // `crc32_z` takes a `z_size_t`.
  if (data.size() - start > std::numeric_limits<z_size_t>::max()) {
    std::cout << mgard_x::log::log_err
              << "buffer is too large (size would overflow.\n";
  }
  uLong crc32_ = crc32_z(0, Z_NULL, 0);
  crc32_ = crc32_z(crc32_, static_cast<const Bytef *>(data.data() + start),
                   data.size() - start);
  return crc32_;
}

std::vector<SERIALIZED_TYPE> SerializeProtoBuf(mgard::pb::Header &header) {
  size_t header_size = header.ByteSize();
  std::vector<SERIALIZED_TYPE> header_bytes(header_size);
  header.SerializeToArray(header_bytes.data(), header_size);
  return header_bytes;
}

mgard::pb::Header
DeserializeProtoBuf(const std::vector<SERIALIZED_TYPE> header_bytes,
                    uint64_t offset) {
  uint64_t header_size = header_bytes.size() - offset;

  // The `CodedInputStream` constructor takes an `int`.
  if (header_size > std::numeric_limits<int>::max()) {
    std::cout << mgard_x::log::log_err
              << "header is too large (size would overflow).\n";
  }
  mgard::pb::Header header;
  google::protobuf::io::CodedInputStream stream(
      static_cast<google::protobuf::uint8 const *>(header_bytes.data() +
                                                   offset),
      header_size);
  if (not header.ParseFromCodedStream(&stream)) {
    throw std::runtime_error("header parsing encountered read or format error");
  }
  if (not stream.ConsumedEntireMessage()) {
    throw std::runtime_error("part of header left unparsed");
  }
  return header;
}

template <typename T>
void Serialize(const T &item, std::vector<SERIALIZED_TYPE> &vec) {
  if constexpr (std::is_integral<T>::value) {
    T in = item;
    for (int i = 0; i < sizeof(T); i++) {
      vec.push_back(in);
      in = in >> 8;
    }
  } else {
    const SERIALIZED_TYPE *in =
        reinterpret_cast<const SERIALIZED_TYPE *>(&item);
    vec.insert(vec.end(), in, in + sizeof(item));
  }
}

template <typename T> void Deserialize(T &item, SerializedIter &iter) {
  if constexpr (std::is_integral<T>::value) {
    T out = 0;
    for (int i = sizeof(T) - 1; i >= 0; i--) {
      out = out << 8;
      out = out + *(iter + i);
    }
    item = out;
  } else {
    SERIALIZED_TYPE *out = reinterpret_cast<SERIALIZED_TYPE *>(&item);
    std::copy(iter, iter + sizeof(T), out);
  }
  iter += sizeof(T);
}

void SerializeSignature(std::vector<SERIALIZED_TYPE> &vec) {
  for (char c : mgard::SIGNATURE) {
    Serialize(c, vec);
  }
}

// This function does not assign to a signature data member. Instead, it just
// checks that the deserialized signature matches `mgard::SIGNATURE`.
void DeserializeSignature(SerializedIter &iter) {
  for (const char c : mgard::SIGNATURE) {
    char c_;
    Deserialize(c_, iter);
    if (c_ != c) {
      std::cout << mgard_x::log::log_err << "signature mismatch.\n";
      exit(-1);
    }
  }
}

void SerializeBytes(const std::vector<SERIALIZED_TYPE> &data,
                    std::vector<SERIALIZED_TYPE> &vec) {
  vec.insert(vec.end(), data.begin(), data.end());
}

} // anonymous namespace

namespace mgard_x {

void MetadataBase::InitializeConfig(Config &config) {
  config.domain_decomposition = ddtype;
  config.decomposition = decomposition;
  config.lossless = ltype;
  config.huff_dict_size = huff_dict_size;
  config.huff_block_size = huff_block_size;
  config.reorder = reorder;
}

void MetadataBase::PrintSummary() {
  std::cout << "=======Metadata Summary=======\n";
  std::cout << "Signature: ";
  for (const char c : mgard::SIGNATURE)
    std::cout << c;
  std::cout << "\n";
  std::cout << "MGARD version: " << (int)software_version[0] << "."
            << (int)software_version[1] << "." << (int)software_version[2]
            << "\n";
  std::cout << "File format version: " << (int)file_version[0] << "."
            << (int)file_version[1] << "." << (int)file_version[2] << "\n";
  std::cout << "Metadata size: " << metadata_size << "\n";
  std::cout << "Metadata crc32: " << metadata_crc32 << "\n";
  std::cout << "Endiness: ";
  if (etype == endiness_type::Big_Endian) {
    std::cout << "Big Endian\n";
  } else {
    std::cout << "Little Endian\n";
  }
  std::cout << "Data type: ";
  if (dtype == data_type::Float) {
    std::cout << "Float\n";
  } else if (dtype == data_type::Double) {
    std::cout << "Double\n";
  }
  std::cout << "Topology: ";
  if (dstype == data_structure_type::Cartesian_Grid_Uniform) {
    std::cout << "Uniform Grid\n";
  } else if (dstype == data_structure_type::Cartesian_Grid_Non_Uniform) {
    std::cout << "Non-uniform Grid\n";
  }
  std::cout << "Shape: ";
  for (uint64_t &c : shape)
    std::cout << c << " ";
  std::cout << "\n";
  std::cout << "Function Decomposition: ";
  if (decomposition == decomposition_type::MultiDim) {
    std::cout << "MultiDim\n";
  } else if (decomposition == decomposition_type::SingleDim) {
    std::cout << "SingleDim\n";
  }
  std::cout << "Reorder: " << reorder << "\n";
  std::cout << "Domain Decomposition: ";
  if (domain_decomposed) {
    if (ddtype == domain_decomposition_type::MaxDim) {
      std::cout << "MaxDim\n";
    } else if (ddtype == domain_decomposition_type::Variable) {
      std::cout << "Variable\n";
    } else {
      std::cout << "Block\n";
    }
    std::cout << "Decomposed Dim: " << domain_decomposed_dim << "\n";
    std::cout << "Decomposed Size: " << domain_decomposed_size << "\n";
  } else {
    std::cout << "No\n";
  }
  std::cout << "Error bound mode: ";
  if (ebtype == error_bound_type::REL) {
    std::cout << "REL\n";
  } else if (ebtype == error_bound_type::ABS) {
    std::cout << "ABS\n";
  }
  std::cout << "Norm type: ";
  if (ntype == norm_type::L_Inf) {
    std::cout << "L_Inf\n";
  } else if (ntype == norm_type::L_2) {
    std::cout << "L_2\n";
  }
  std::cout << "Norm: " << norm << "\n";
  std::cout << "tol: " << tol << "\n";
  std::cout << "s: " << s << "\n";

  std::cout << "Lossless:  ";
  if (ltype == mgard_x::lossless_type::Huffman) {
    std::cout << "Huffman\n";
    std::cout << "Huffman dictionary size: " << huff_dict_size << "\n";
    std::cout << "Huffman block size: " << huff_block_size << "\n";
  } else if (ltype == mgard_x::lossless_type::Huffman_LZ4) {
    std::cout << "Huffman_LZ4\n";
    std::cout << "Huffman dictionary size: " << huff_dict_size << "\n";
    std::cout << "Huffman block size: " << huff_block_size << "\n";
  } else if (ltype == mgard_x::lossless_type::Huffman_Zstd) {
    std::cout << "Huffman_Zstd\n";
    std::cout << "Huffman dictionary size: " << huff_dict_size << "\n";
    std::cout << "Huffman block size: " << huff_block_size << "\n";
  } else if (ltype == mgard_x::lossless_type::CPU_Lossless) {
    std::cout << "CPU_Lossless\n";
  }

  std::cout << "Backend:  ";
  if (ptype == processor_type::X_SERIAL) {
    std::cout << "X_SERIAL\n";
  } else if (ptype == processor_type::X_CUDA) {
    std::cout << "X_OPENMP\n";
  } else if (ptype == processor_type::X_OPENMP) {
    std::cout << "X_CUDA\n";
  } else if (ptype == processor_type::X_HIP) {
    std::cout << "X_HIP\n";
  } else if (ptype == processor_type::X_SYCL) {
    std::cout << "X_SYCL\n";
  }
}

std::uint64_t MetadataBase::SerializePreambleSize() {
  std::uint64_t preamble_size = 0;
  preamble_size += mgard::SIGNATURE.size();
  preamble_size += sizeof(uint64_t); // header size
  preamble_size += sizeof(uint32_t); // crc32 size
  return preamble_size;
}

std::vector<SERIALIZED_TYPE> MetadataBase::Serialize() {
  mgard::pb::Header header;

  { // Version Number
    software_version[0] = MGARD_VERSION_MAJOR;
    software_version[1] = MGARD_VERSION_MINOR;
    software_version[2] = MGARD_VERSION_PATCH;

    mgard::pb::VersionNumber &mgard_version_number =
        *header.mutable_mgard_version();
    mgard_version_number.set_major_(MGARD_VERSION_MAJOR);
    mgard_version_number.set_minor_(MGARD_VERSION_MINOR);
    mgard_version_number.set_patch_(MGARD_VERSION_PATCH);

    file_version[0] = MGARD_FILE_VERSION_MAJOR;
    file_version[1] = MGARD_FILE_VERSION_MINOR;
    file_version[2] = MGARD_FILE_VERSION_PATCH;

    mgard::pb::VersionNumber &format_version_number =
        *header.mutable_file_format_version();
    mgard_version_number.set_major_(MGARD_FILE_VERSION_MAJOR);
    mgard_version_number.set_minor_(MGARD_FILE_VERSION_MINOR);
    mgard_version_number.set_patch_(MGARD_FILE_VERSION_PATCH);
  }

  { // Domain
    mgard::pb::Domain &domain = *header.mutable_domain();
    domain.set_topology(mgard::pb::Domain::CARTESIAN_GRID);
    mgard::pb::CartesianGridTopology &cartesian_grid_topology =
        *domain.mutable_cartesian_grid_topology();
    cartesian_grid_topology.set_dimension(total_dims);
    google::protobuf::RepeatedField<google::protobuf::uint64> &shape_ =
        *cartesian_grid_topology.mutable_shape();
    shape_.Resize(total_dims, 0);
    std::copy(shape.begin(), shape.end(), shape_.mutable_data());
    mgard::pb::Domain::Geometry geometry;
    if (dstype == data_structure_type::Cartesian_Grid_Uniform) {
      geometry = mgard::pb::Domain::UNIT_CUBE;
    } else {
      geometry = mgard::pb::Domain::EXPLICIT_CUBE;
      mgard::pb::ExplicitCubeGeometry &explicit_cube_geometry =
          *domain.mutable_explicit_cube_geometry();
      google::protobuf::RepeatedField<double> &coordinates_ =
          *explicit_cube_geometry.mutable_coordinates();

      uint64_t totel_len = 0;
      for (DIM d = 0; d < total_dims; d++)
        totel_len += shape[d];
      coordinates_.Resize(totel_len, 0);
      double *p = coordinates_.mutable_data();
      for (DIM d = 0; d < total_dims; d++) {
        std::copy(coords[d].begin(), coords[d].end(), p);
        p += shape[d];
      }
    }
    domain.set_geometry(geometry);
  }

  { // Dataset
    mgard::pb::Dataset &dataset = *header.mutable_dataset();
    if (dtype == data_type::Float) {
      dataset.set_type(mgard::pb::Dataset::FLOAT);
    } else if (dtype == data_type::Double) {
      dataset.set_type(mgard::pb::Dataset::DOUBLE);
    }
    dataset.set_dimension(1);
  }

  { // Error control
    mgard::pb::ErrorControl &error = *header.mutable_error_control();
    if (ebtype == error_bound_type::ABS) {
      error.set_mode(mgard::pb::ErrorControl::ABSOLUTE);
    } else if (ebtype == error_bound_type::REL) {
      error.set_mode(mgard::pb::ErrorControl::RELATIVE);
      error.set_norm_of_original_data(norm);
    }
    if (ntype == norm_type::L_Inf) {
      error.set_norm(mgard::pb::ErrorControl::L_INFINITY);
      error.set_s(s);
    } else {
      error.set_norm(mgard::pb::ErrorControl::S_NORM);
      error.set_s(s);
    }
    error.set_tolerance(tol);
  }

  { // Domain Decomposition
    mgard::pb::DomainDecomposition &domainDecomposition =
        *header.mutable_domain_decomposition();
    if (domain_decomposed) {
      if (ddtype == domain_decomposition_type::MaxDim) {
        domainDecomposition.set_method(
            mgard::pb::DomainDecomposition::MAX_DIMENSION);
      } else if (ddtype == domain_decomposition_type::Block) {
        domainDecomposition.set_method(mgard::pb::DomainDecomposition::BLOCK);
      } else if (ddtype == domain_decomposition_type::Variable) {
        domainDecomposition.set_method(
            mgard::pb::DomainDecomposition::VARIABLE);
      }
    } else {
      domainDecomposition.set_method(
          mgard::pb::DomainDecomposition::NOOP_METHOD);
    }
    domainDecomposition.set_decomposition_dimension(domain_decomposed_dim);
    domainDecomposition.set_decomposition_size(domain_decomposed_size);
  }

  { // Function Decomposition
    mgard::pb::FunctionDecomposition &function_decomposition =
        *header.mutable_function_decomposition();
    function_decomposition.set_transform(
        mgard::pb::FunctionDecomposition::MULTILEVEL_COEFFICIENTS);
    if (decomposition == decomposition_type::MultiDim) {
      function_decomposition.set_hierarchy(
          mgard::pb::FunctionDecomposition::MULTIDIMENSION_WITH_GHOST_NODES);
    } else if (decomposition == decomposition_type::SingleDim) {
      function_decomposition.set_hierarchy(
          mgard::pb::FunctionDecomposition::ONE_DIM_AT_A_TIME_WITH_GHOST_NODES);
    } else if (decomposition == decomposition_type::Hybrid) {
      function_decomposition.set_hierarchy(
          mgard::pb::FunctionDecomposition::HYBRID_HIERARCHY);
    }
    function_decomposition.set_l_target(l_target);
  }

  { // Quantization
    mgard::pb::Quantization &quantization = *header.mutable_quantization();
    if (otype == operation_type::Compression) {
      quantization.set_method(mgard::pb::Quantization::COEFFICIENTWISE_LINEAR);
      quantization.set_bin_widths(mgard::pb::Quantization::PER_COEFFICIENT);
      quantization.set_type(mgard::pb::Quantization::INT64_T);
      quantization.set_big_endian(big_endian<std::int64_t>());
      if (big_endian<std::int64_t>()) {
        etype = endiness_type::Big_Endian;
      } else {
        etype = endiness_type::Little_Endian;
      }
    } else {
      quantization.set_method(mgard::pb::Quantization::NOOP_QUANTIZATION);
    }
  }

  { // MDR
    mgard::pb::BitplaneEncoding &bitplane_encoding =
        *header.mutable_bitplane_encoding();
    if (otype == operation_type::MDR) {
      bitplane_encoding.set_method(
          mgard::pb::BitplaneEncoding::GROUPED_BITPLANE_ENCODING);
      bitplane_encoding.set_type(mgard::pb::BitplaneEncoding::INT32_T);
      bitplane_encoding.set_number_bitplanes(number_bitplanes);
      bitplane_encoding.set_big_endian(big_endian<std::int64_t>());
    } else {
      bitplane_encoding.set_method(
          mgard::pb::BitplaneEncoding::NOOP_BITPLANE_ENCODING);
    }
  }

  { // Encoding
    mgard::pb::Encoding &encoding = *header.mutable_encoding();
    if (reorder == 0) {
      encoding.set_preprocessor(mgard::pb::Encoding::NOOP_PREPROCESSOR);
    } else {
      encoding.set_preprocessor(mgard::pb::Encoding::SHUFFLE);
    }
    if (ltype == mgard_x::lossless_type::Huffman) {
      encoding.set_compressor(mgard::pb::Encoding::X_HUFFMAN);
      encoding.set_huffman_dictionary_size(huff_dict_size);
      encoding.set_huffman_block_size(huff_block_size);
    } else if (ltype == mgard_x::lossless_type::Huffman_LZ4) {
      encoding.set_compressor(mgard::pb::Encoding::X_HUFFMAN_LZ4);
      encoding.set_huffman_dictionary_size(huff_dict_size);
      encoding.set_huffman_block_size(huff_block_size);
    } else if (ltype == mgard_x::lossless_type::Huffman_Zstd) {
      encoding.set_compressor(mgard::pb::Encoding::X_HUFFMAN_ZSTD);
      encoding.set_huffman_dictionary_size(huff_dict_size);
      encoding.set_huffman_block_size(huff_block_size);
    } else if (ltype == mgard_x::lossless_type::CPU_Lossless) {
      encoding.set_compressor(mgard::pb::Encoding::CPU_HUFFMAN_ZSTD);
    }
  }

  { // Device
    mgard::pb::Device &device = *header.mutable_device();
    if (ptype == processor_type::X_SERIAL) {
      device.set_backend(mgard::pb::Device::X_SERIAL);
    } else if (ptype == processor_type::X_OPENMP) {
      device.set_backend(mgard::pb::Device::X_OPENMP);
    } else if (ptype == processor_type::X_CUDA) {
      device.set_backend(mgard::pb::Device::X_CUDA);
    } else if (ptype == processor_type::X_HIP) {
      device.set_backend(mgard::pb::Device::X_HIP);
    } else if (ptype == processor_type::X_SYCL) {
      device.set_backend(mgard::pb::Device::X_SYCL);
    }
  }

  // Serialize protobuf
  std::vector<SERIALIZED_TYPE> header_bytes = SerializeProtoBuf(header);
  uint64_t header_size = header_bytes.size();
  uint32_t header_crc32 = ComputeCRC32(header_bytes);

  metadata_size = SerializePreambleSize() + header_size;

  // start serializing
  std::vector<SERIALIZED_TYPE> serialized_data;
  serialized_data.reserve(metadata_size);

  ::SerializeSignature(serialized_data);
  ::Serialize(header_size, serialized_data);
  ::Serialize(header_crc32, serialized_data);
  ::SerializeBytes(header_bytes, serialized_data);

  return serialized_data;
}

uint64_t MetadataBase::DeserializeSize(
    std::vector<SERIALIZED_TYPE>::const_iterator &iter) {
  uint64_t header_size = 0;
  ::DeserializeSignature(iter);
  ::Deserialize(header_size, iter);

  header_size += SerializePreambleSize();

  return header_size;
}

void MetadataBase::Deserialize(
    const std::vector<SERIALIZED_TYPE> &serialized_data) {
  metadata_size = serialized_data.size();

  SerializedIter iter = serialized_data.cbegin();
  DeserializeSize(iter);

  uint32_t header_crc32 = 0;
  ::Deserialize(header_crc32, iter);

  // Actual serialized data is after header starting at index offset.
  uint64_t offset = std::distance(serialized_data.begin(), iter);

  if (header_crc32 != ComputeCRC32(serialized_data, offset)) {
    std::cout << log::log_err << "header CRC32 mismatch.\n";
    exit(-1);
  }

  mgard::pb::Header header = DeserializeProtoBuf(serialized_data, offset);

  { // Version Number
    const mgard::pb::VersionNumber mgard_version_number =
        header.mgard_version();
    software_version[0] = mgard_version_number.major_();
    software_version[1] = mgard_version_number.minor_();
    software_version[2] = mgard_version_number.patch_();
    if (software_version[0] > MGARD_VERSION_MAJOR) {
      std::cout << log::log_err << "MGARD version mismatch.\n";
      exit(-1);
    }

    const mgard::pb::VersionNumber format_version_number =
        header.file_format_version();
    file_version[0] = format_version_number.major_();
    file_version[1] = format_version_number.minor_();
    file_version[2] = format_version_number.patch_();
    if (file_version[0] > MGARD_FILE_VERSION_MAJOR) {
      std::cout << log::log_err << "MGARD file format version mismatch.\n";
      exit(-1);
    }
  }

  { // Domain
    const mgard::pb::Domain &domain = header.domain();
    const mgard::pb::CartesianGridTopology cartesian_grid_topology =
        domain.cartesian_grid_topology();
    total_dims = cartesian_grid_topology.dimension();
    const google::protobuf::RepeatedField<google::protobuf::uint64> shape_ =
        cartesian_grid_topology.shape();
    if (total_dims != shape_.size()) {
      std::cout << log::log_err
                << "grid shape does not match given dimension.\n";
      exit(-1);
    }
    shape = std::vector<uint64_t>(total_dims);
    std::copy(shape_.begin(), shape_.end(), shape.begin());

    const mgard::pb::Domain::Geometry geometry = domain.geometry();
    if (geometry == mgard::pb::Domain::UNIT_CUBE) {
      dstype = data_structure_type::Cartesian_Grid_Uniform;
    } else if (geometry == mgard::pb::Domain::EXPLICIT_CUBE) {
      dstype = data_structure_type::Cartesian_Grid_Non_Uniform;
      const mgard::pb::ExplicitCubeGeometry explicit_cube_geometry =
          domain.explicit_cube_geometry();
      const google::protobuf::RepeatedField<double> coordinates =
          explicit_cube_geometry.coordinates();
      uint64_t totel_len = 0;
      for (DIM d = 0; d < total_dims; d++)
        totel_len += shape[d];
      if (totel_len != coordinates.size()) {
        std::cout << log::log_err
                  << "mismatch between number of node coordinates and grid "
                     "shape.\n";
        exit(-1);
      }
      using It = google::protobuf::RepeatedField<double>::const_iterator;
      It p = coordinates.begin();
      coords = std::vector<std::vector<double>>(total_dims);
      for (size_t d = 0; d < total_dims; d++) {
        const It q = p + shape[d];
        coords[d] = std::vector<double>(shape[d]);
        std::copy(p, q, coords[d].begin());
        p = q;
      }
      assert(p == coordinates.end());
    }
  }

  { // Dataset
    const mgard::pb::Dataset dataset = header.dataset();
    if (dataset.type() == mgard::pb::Dataset::FLOAT) {
      dtype = data_type::Float;
    } else if (dataset.type() == mgard::pb::Dataset::DOUBLE) {
      dtype = data_type::Double;
    }
    assert(dataset.dimension() == 1);
  }

  { // Error control
    const mgard::pb::ErrorControl error = header.error_control();
    if (error.mode() == mgard::pb::ErrorControl::ABSOLUTE) {
      ebtype = error_bound_type::ABS;
    } else if (error.mode() == mgard::pb::ErrorControl::RELATIVE) {
      ebtype = error_bound_type::REL;
      norm = error.norm_of_original_data();
    }

    if (error.norm() == mgard::pb::ErrorControl::L_INFINITY) {
      ntype = norm_type::L_Inf;
      s = std::numeric_limits<double>::infinity();
    } else if (error.norm() == mgard::pb::ErrorControl::S_NORM) {
      ntype = norm_type::L_2;
      s = error.s();
    }
    tol = error.tolerance();
  }

  { // Domain Decomposition
    const mgard::pb::DomainDecomposition domainDecomposition =
        header.domain_decomposition();
    if (domainDecomposition.method() !=
        mgard::pb::DomainDecomposition::NOOP_METHOD) {
      domain_decomposed = true;
      if (domainDecomposition.method() ==
          mgard::pb::DomainDecomposition::MAX_DIMENSION) {
        ddtype = domain_decomposition_type::MaxDim;
      } else if (domainDecomposition.method() ==
                 mgard::pb::DomainDecomposition::BLOCK) {
        ddtype = domain_decomposition_type::Block;
      } else if (domainDecomposition.method() ==
                 mgard::pb::DomainDecomposition::VARIABLE) {
        ddtype = domain_decomposition_type::Variable;
      }

      domain_decomposed_dim = domainDecomposition.decomposition_dimension();
      domain_decomposed_size = domainDecomposition.decomposition_size();
    } else {
      domain_decomposed = false;
    }
  }

  { // Function Decomposition
    const mgard::pb::FunctionDecomposition function_decomposition =
        header.function_decomposition();
    assert(function_decomposition.transform() ==
           mgard::pb::FunctionDecomposition::MULTILEVEL_COEFFICIENTS);
    if (function_decomposition.hierarchy() ==
        mgard::pb::FunctionDecomposition::MULTIDIMENSION_WITH_GHOST_NODES) {
      decomposition = decomposition_type::MultiDim;
    } else if (function_decomposition.hierarchy() ==
               mgard::pb::FunctionDecomposition::
                   ONE_DIM_AT_A_TIME_WITH_GHOST_NODES) {
      decomposition = decomposition_type::SingleDim;
    } else if (function_decomposition.hierarchy() ==
               mgard::pb::FunctionDecomposition::HYBRID_HIERARCHY) {
      decomposition = decomposition_type::Hybrid;
    } else {
      std::cout << log::log_err
                << "this decomposition hierarchy mismatch the hierarchy used "
                   "in MGARD-X.\n";
      exit(-1);
    }
    l_target = function_decomposition.l_target();
  }

  { // Quantization
    const mgard::pb::Quantization quantization = header.quantization();
    if (quantization.method() != mgard::pb::Quantization::NOOP_QUANTIZATION) {
      assert(quantization.bin_widths() ==
             mgard::pb::Quantization::PER_COEFFICIENT);
      assert(quantization.type() == mgard::pb::Quantization::INT64_T);
      assert(quantization.big_endian() == big_endian<std::int64_t>());
      if (big_endian<std::int64_t>()) {
        etype = endiness_type::Big_Endian;
      } else {
        etype = endiness_type::Little_Endian;
      }
    }
  }

  { // MDR
    const mgard::pb::BitplaneEncoding bitplane_encoding =
        header.bitplane_encoding();
    if (bitplane_encoding.method() !=
        mgard::pb::BitplaneEncoding::NOOP_BITPLANE_ENCODING) {
      number_bitplanes = bitplane_encoding.number_bitplanes();
      assert(bitplane_encoding.big_endian() == big_endian<std::int64_t>());
    }
  }

  {
    const mgard::pb::Quantization quantization = header.quantization();
    const mgard::pb::BitplaneEncoding bitplane_encoding =
        header.bitplane_encoding();
    if (quantization.method() != mgard::pb::Quantization::NOOP_QUANTIZATION &&
            bitplane_encoding.method() !=
                mgard::pb::BitplaneEncoding::NOOP_BITPLANE_ENCODING ||
        quantization.method() == mgard::pb::Quantization::NOOP_QUANTIZATION &&
            bitplane_encoding.method() ==
                mgard::pb::BitplaneEncoding::NOOP_BITPLANE_ENCODING) {
      std::cout << log::log_err
                << "cannot determine whether this is compressed or "
                   "refactored data.\n";
      exit(-1);
    }
  }

  if (otype == operation_type::MDR) { // MDR
    mgard::pb::BitplaneEncoding &bitplane_encoding =
        *header.mutable_bitplane_encoding();
    bitplane_encoding.set_method(
        mgard::pb::BitplaneEncoding::GROUPED_BITPLANE_ENCODING);
    bitplane_encoding.set_type(mgard::pb::BitplaneEncoding::INT32_T);
    bitplane_encoding.set_number_bitplanes(number_bitplanes);
    bitplane_encoding.set_big_endian(big_endian<std::int64_t>());
  }

  { // Encoding
    const mgard::pb::Encoding encoding = header.encoding();
    if (encoding.preprocessor() == mgard::pb::Encoding::SHUFFLE) {
      reorder = 1;
    } else {
      reorder = 0;
    }
    if (encoding.compressor() == mgard::pb::Encoding::X_HUFFMAN) {
      ltype = mgard_x::lossless_type::Huffman;
      huff_dict_size = encoding.huffman_dictionary_size();
      huff_block_size = encoding.huffman_block_size();
    } else if (encoding.compressor() == mgard::pb::Encoding::X_HUFFMAN_LZ4) {
      ltype = mgard_x::lossless_type::Huffman_LZ4;
      huff_dict_size = encoding.huffman_dictionary_size();
      huff_block_size = encoding.huffman_block_size();
    } else if (encoding.compressor() == mgard::pb::Encoding::X_HUFFMAN_ZSTD) {
      ltype = mgard_x::lossless_type::Huffman_Zstd;
      huff_dict_size = encoding.huffman_dictionary_size();
      huff_block_size = encoding.huffman_block_size();
    } else if (encoding.compressor() == mgard::pb::Encoding::CPU_HUFFMAN_ZSTD) {
      ltype = mgard_x::lossless_type::CPU_Lossless;
    } else {
      std::cout << log::log_err << "unknown lossless compressor type.\n";
      exit(-1);
    }
  }

  { // Device
    const mgard::pb::Device device = header.device();
    if (device.backend() == mgard::pb::Device::X_SERIAL) {
      ptype = processor_type::X_SERIAL;
    } else if (device.backend() == mgard::pb::Device::X_OPENMP) {
      ptype = processor_type::X_OPENMP;
    } else if (device.backend() == mgard::pb::Device::X_CUDA) {
      ptype = processor_type::X_CUDA;
    } else if (device.backend() == mgard::pb::Device::X_HIP) {
      ptype = processor_type::X_HIP;
    } else if (device.backend() == mgard::pb::Device::X_SYCL) {
      ptype = processor_type::X_SYCL;
    } else if (device.backend() == mgard::pb::Device::CPU) {
      std::cout << log::log_err
                << "this data was not compressed with MGARD-X.\n";
      exit(-1);
    }
  }
}

} // namespace mgard_x
