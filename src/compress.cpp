#include "compress.hpp"

namespace {

using namespace mgard;

template <std::size_t N, typename Real>
std::unique_ptr<unsigned char const []>
decompress_internal(unsigned char *b, unsigned char *cb_copy, const Real s,
                    const Real tol, uint64_t *shape, GridType grid_type,
                    std::size_t compressed_size, uint32_t metadata_size) {
  std::array<std::size_t, N> dims = {};
  for (std::size_t i = 0; i < N; i++) {
    dims.at(i) = shape[i];
  }

  std::array<std::vector<Real>, N> coordinates =
      default_node_coordinates<N, Real>(dims);

  // For non-uniform grid, retrieve the coordinates from metadata
  if (grid_type == NONUNIFORM) {
    for (std::size_t i = 0; i < N; i++) {
      Real *file_buffer = (Real *)std::malloc(sizeof(Real) * shape[i]);
      std::memcpy(file_buffer, b, sizeof(Real) * shape[i]);
      b += sizeof(Real) * shape[i];
      std::copy(file_buffer, file_buffer + shape[i], coordinates.at(i).begin());
      std::free(file_buffer);
    }
  }

  TensorMeshHierarchy<N, Real> hierarchy(dims, coordinates);
  const mgard::CompressedDataset<N, Real> compressed(
      hierarchy, s, tol, cb_copy, compressed_size - metadata_size);
  const mgard::DecompressedDataset<N, Real> decompressed =
      mgard::decompress(compressed);
  unsigned char *const decompressed_buffer =
      new unsigned char[hierarchy.ndof() * sizeof(Real)];
  std::memcpy(decompressed_buffer, decompressed.data(),
              hierarchy.ndof() * sizeof(Real));

  return std::unique_ptr<unsigned char const[]>(decompressed_buffer);
}

} // anonymous namespace

namespace mgard {

std::unique_ptr<unsigned char const []>
decompress(void const *const compressed_buffer, std::size_t compressed_size) {
  unsigned char *b = (unsigned char *)compressed_buffer;

  char sig_str[SIGNATURE_STR.size() + 1];
  std::memcpy(sig_str, b, SIGNATURE_STR.size());
  b += SIGNATURE_STR.size();
  sig_str[SIGNATURE_STR.size()] = '\0';
  if (strcmp(sig_str, SIGNATURE_STR.c_str()) != 0) {
    throw std::invalid_argument("Data was not compressed by MGARD.");
  }

  // Software version major, minor and patch
  uint8_t sv_major = *(uint8_t *)b;
  b += 1;
  uint8_t sv_minor = *(uint8_t *)b;
  b += 1;
  uint8_t sv_patch = *(uint8_t *)b;
  b += 1;

  // File major, minor and patch
  uint8_t fv_major = *(uint8_t *)b;
  b += 1;
  uint8_t fv_minor = *(uint8_t *)b;
  b += 1;
  uint8_t fv_patch = *(uint8_t *)b;
  b += 1;

  // Size of metadata
  uint32_t metadata_size = *(uint32_t *)b;
  b += 4;

  // Type
  uint8_t type = *(uint8_t *)b;
  b += 1;

  // Number of dims
  uint8_t ndims = *(uint8_t *)b;
  b += 1;

  uint64_t shape[ndims];
  for (uint8_t i = 0; i < ndims; i++) {
    shape[i] = *(uint64_t *)b;
    b += 8;
  }

  // Tolerance
  double tol = *(double *)b;
  b += 8;

  // S
  double s = *(double *)b;
  b += 8;

  // L-inf norm or S-norm
  double norm = *(double *)b;
  b += 8;

  // Target level
  uint32_t target_level = *(uint32_t *)b;
  b += 4;

  // Grid type
  GridType grid_type = GetGridType(*(uint8_t *)b);
  b += 1;

#if 0
  std::cout << "ndims = " << (unsigned)ndims << " tol = " << tol << " s = " << s
            << " target_level = " << target_level
            << " grid_type = " << grid_type << "\n";
#endif
  unsigned char *cb_copy = static_cast<unsigned char *>(
      std::malloc(compressed_size - metadata_size));
  std::memcpy(
      cb_copy,
      static_cast<unsigned char *>(const_cast<void *>(compressed_buffer)) +
          metadata_size,
      compressed_size - metadata_size);

  std::unique_ptr<unsigned char const[]> decompressed_buffer;

  switch (ndims) {
  case 1:
    if (type == 0) {
      decompressed_buffer = decompress_internal<1, double>(
          b, cb_copy, s, tol, shape, grid_type, compressed_size, metadata_size);
    } else if (type == 1) {
      decompressed_buffer = decompress_internal<1, float>(
          b, cb_copy, s, tol, shape, grid_type, compressed_size, metadata_size);
    }
    break;
  case 2:
    if (type == 0) {
      decompressed_buffer = decompress_internal<2, double>(
          b, cb_copy, s, tol, shape, grid_type, compressed_size, metadata_size);
    } else if (type == 1) {
      decompressed_buffer = decompress_internal<2, float>(
          b, cb_copy, s, tol, shape, grid_type, compressed_size, metadata_size);
    }
    break;
  case 3:
    if (type == 0) {
      decompressed_buffer = decompress_internal<3, double>(
          b, cb_copy, s, tol, shape, grid_type, compressed_size, metadata_size);
    } else if (type == 1) {
      decompressed_buffer = decompress_internal<3, float>(
          b, cb_copy, s, tol, shape, grid_type, compressed_size, metadata_size);
    }
    break;
  case 4:
    if (type == 0) {
      decompressed_buffer = decompress_internal<4, double>(
          b, cb_copy, s, tol, shape, grid_type, compressed_size, metadata_size);
    } else if (type == 1) {
      decompressed_buffer = decompress_internal<4, float>(
          b, cb_copy, s, tol, shape, grid_type, compressed_size, metadata_size);
    }
    break;
  case 5:
    if (type == 0) {
      decompressed_buffer = decompress_internal<5, double>(
          b, cb_copy, s, tol, shape, grid_type, compressed_size, metadata_size);
    } else if (type == 1) {
      decompressed_buffer = decompress_internal<5, float>(
          b, cb_copy, s, tol, shape, grid_type, compressed_size, metadata_size);
    }
    break;
  case 6:
    if (type == 0) {
      decompressed_buffer = decompress_internal<6, double>(
          b, cb_copy, s, tol, shape, grid_type, compressed_size, metadata_size);
    } else if (type == 1) {
      decompressed_buffer = decompress_internal<6, float>(
          b, cb_copy, s, tol, shape, grid_type, compressed_size, metadata_size);
    }
    break;
  case 7:
    if (type == 0) {
      decompressed_buffer = decompress_internal<7, double>(
          b, cb_copy, s, tol, shape, grid_type, compressed_size, metadata_size);
    } else if (type == 1) {
      decompressed_buffer = decompress_internal<7, float>(
          b, cb_copy, s, tol, shape, grid_type, compressed_size, metadata_size);
    }
    break;
  default:
    throw std::invalid_argument(
        "The number of dimensions beyond 7 is not supported.");
  }
#if 0
  if (metadata_size != b - (unsigned char *)compressed_buffer) {
std::cout << "metadata_size = " << metadata_size << " and = " << b - (unsigned char *)compressed_buffer << "\n";
    throw std::invalid_argument("Error in parsing metadata. Likely, this is "
                                "due to the incompability of MGARD versions");
  }
#endif
  return decompressed_buffer;
}

} // namespace mgard
