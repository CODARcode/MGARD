#include <climits>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include "mgard-x/RuntimeX/DataTypes.h"

#include "mgard-x/ExternalCompressionLowLevel/ZFP/shared.h"

namespace mgard_x {

namespace zfp {

size_t calc_device_mem1d(const int dim_x, const int maxbits) {

  const size_t vals_per_block = 4;
  size_t total_blocks = dim_x / vals_per_block;
  if (dim_x % vals_per_block != 0) {
    total_blocks++;
  }
  const size_t bits_per_block = maxbits;
  const size_t bits_per_word = sizeof(ZFPWord) * 8;
  const size_t total_bits = bits_per_block * total_blocks;
  size_t alloc_size = total_bits / bits_per_word;
  if (total_bits % bits_per_word != 0)
    alloc_size++;
  // ensure we have zeros
  return alloc_size * sizeof(ZFPWord);
}

size_t calc_device_mem2d(const int dim_y, const int dim_x, const int maxbits) {

  const size_t vals_per_block = 16;
  size_t total_blocks = (dim_x * dim_y) / vals_per_block;
  if ((dim_x * dim_y) % vals_per_block != 0)
    total_blocks++;
  const size_t bits_per_block = maxbits;
  const size_t bits_per_word = sizeof(ZFPWord) * 8;
  const size_t total_bits = bits_per_block * total_blocks;
  size_t alloc_size = total_bits / bits_per_word;
  if (total_bits % bits_per_word != 0)
    alloc_size++;
  return alloc_size * sizeof(ZFPWord);
}

size_t calc_device_mem3d(const int encoded_dim_z, const int encoded_dim_y,
                         const int encoded_dim_x, const int bits_per_block) {
  const size_t vals_per_block = 64;
  const size_t size = encoded_dim_x * encoded_dim_y * encoded_dim_z;
  size_t total_blocks = size / vals_per_block;
  const size_t bits_per_word = sizeof(ZFPWord) * 8;
  const size_t total_bits = bits_per_block * total_blocks;
  const size_t alloc_size = total_bits / bits_per_word;
  return alloc_size * sizeof(ZFPWord);
}

std::vector<SIZE> get_max_grid_dims() {
  // static cudaDeviceProp prop;
  static bool firstTime = true;

  if (firstTime) {
    firstTime = false;

    int device = 0;
    // cudaGetDeviceProperties(&prop, device);
  }

  std::vector<SIZE> grid_dims(3);
  grid_dims[0] = 2147483647; // prop.maxGridSize[0];
  grid_dims[1] = 65535;      // prop.maxGridSize[1];
  grid_dims[2] = 65535;      // prop.maxGridSize[2];
  return grid_dims;
}

// size is assumed to have a pad to the nearest cuda block size
std::vector<SIZE> calculate_grid_size(size_t size, size_t cuda_block_size) {
  size_t grids = size / cuda_block_size; // because of pad this will be exact
  std::vector<SIZE> max_grid_dims = get_max_grid_dims();
  int dims = 1;
  // check to see if we need to add more grids
  if (grids > max_grid_dims[0]) {
    dims = 2;
  }
  if (grids > max_grid_dims[0] * max_grid_dims[1]) {
    dims = 3;
  }

  std::vector<SIZE> grid_size(3);
  grid_size[0] = 1;
  grid_size[1] = 1;
  grid_size[2] = 1;

  if (dims == 1) {
    grid_size[0] = grids;
  }

  if (dims == 2) {
    float sq_r = sqrt((float)grids);
    float intpart = 0;
    std::modf(sq_r, &intpart);
    uint base = intpart;
    grid_size[0] = base;
    grid_size[1] = base;
    // figure out how many y to add
    uint rem = (size - base * base);
    uint y_rows = rem / base;
    if (rem % base != 0)
      y_rows++;
    grid_size[1] += y_rows;
  }

  if (dims == 3) {
    float cub_r = pow((float)grids, 1.f / 3.f);
    ;
    float intpart = 0;
    std::modf(cub_r, &intpart);
    int base = intpart;
    grid_size[0] = base;
    grid_size[1] = base;
    grid_size[2] = base;
    // figure out how many z to add
    uint rem = (size - base * base * base);
    uint z_rows = rem / (base * base);
    if (rem % (base * base) != 0)
      z_rows++;
    grid_size[2] += z_rows;
  }

  return grid_size;
}

} // namespace zfp

} // namespace mgard_x