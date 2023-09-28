#ifndef MGARD_X_ZFP_HPP
#define MGARD_X_ZFP_HPP

#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "../../RuntimeX/RuntimeX.h"
#include "../../Utilities/Types.h"

#include "encode1.h"
#include "encode2.h"
#include "encode3.h"
#include "shared.h"

#include "decode1.h"
#include "decode2.h"
#include "decode3.h"

namespace mgard_x {

namespace zfp {

template <DIM D, typename T, typename DeviceType>
void encode(Array<D, T, DeviceType> &original_data,
            Array<1, ZFPWord, DeviceType> &compressed_data, int bits_per_block,
            int queue_idx) {
  // ErrorCheck errors;
  size_t stream_size = 0;
  if (D == 1) {
    int dim_x = original_data.shape(0);
    int stride_x = 1;
    uint zfp_pad_x = dim_x;
    if (zfp_pad_x % 4 != 0)
      zfp_pad_x += 4 - dim_x % 4;
    size_t stream_bytes = calc_device_mem1d(zfp_pad_x, bits_per_block);
    compressed_data.resize({(SIZE)stream_bytes / sizeof(ZFPWord)});
    compressed_data.memset(0, queue_idx);
    DeviceLauncher<DeviceType>::Execute(
        Encode1Kernel<T, DeviceType>(original_data.data(),
                                     compressed_data.data(), dim_x, stride_x,
                                     bits_per_block),
        queue_idx);
  } else if (D == 2) {
    int dim_x = original_data.shape(1);
    int dim_y = original_data.shape(0);
    int stride_x = 1;
    int stride_y = original_data.ld(0);
    uint zfp_pad_x = dim_x, zfp_pad_y = dim_y;
    if (zfp_pad_x % 4 != 0)
      zfp_pad_x += 4 - dim_x % 4;
    if (zfp_pad_y % 4 != 0)
      zfp_pad_y += 4 - dim_y % 4;
    size_t stream_bytes =
        calc_device_mem2d(zfp_pad_x, zfp_pad_y, bits_per_block);
    compressed_data.resize({(SIZE)stream_bytes / sizeof(ZFPWord)});
    compressed_data.memset(0, queue_idx);
    DeviceLauncher<DeviceType>::Execute(
        Encode2Kernel<T, DeviceType>(original_data.data(),
                                     compressed_data.data(), dim_x, dim_y,
                                     stride_x, stride_y, bits_per_block),
        queue_idx);
  } else if (D == 3) {
    int dim_x = original_data.shape(2);
    int dim_y = original_data.shape(1);
    int dim_z = original_data.shape(0);
    int stride_x = 1;
    int stride_y = original_data.ld(1);
    int stride_z = original_data.ld(0);
    uint zfp_pad_x = dim_x, zfp_pad_y = dim_y, zfp_pad_z = dim_z;
    if (zfp_pad_x % 4 != 0)
      zfp_pad_x += 4 - dim_x % 4;
    if (zfp_pad_y % 4 != 0)
      zfp_pad_y += 4 - dim_y % 4;
    if (zfp_pad_z % 4 != 0)
      zfp_pad_z += 4 - dim_z % 4;
    size_t stream_bytes =
        calc_device_mem3d(zfp_pad_x, zfp_pad_y, zfp_pad_z, bits_per_block);
    compressed_data.resize({(SIZE)stream_bytes / sizeof(ZFPWord)});
    compressed_data.memset(0, queue_idx);
    DeviceLauncher<DeviceType>::Execute(
        Encode3Kernel<T, DeviceType>(
            original_data.data(), compressed_data.data(), dim_x, dim_y, dim_z,
            stride_x, stride_y, stride_z, bits_per_block),
        queue_idx);
  }
  // errors.chk("Encode");
}

template <DIM D, typename T, typename DeviceType>
void decode(Array<1, ZFPWord, DeviceType> &compressed_data,
            Array<D, T, DeviceType> &decompressed_data, int bits_per_block,
            int queue_idx) {
  // ErrorCheck errors;
  size_t stream_size = 0;
  if (D == 1) {
    int dim_x = decompressed_data.shape(0);
    int stride_x = 1;
    DeviceLauncher<DeviceType>::Execute(
        Decode1Kernel<T, DeviceType>(compressed_data.data(),
                                     decompressed_data.data(), dim_x, stride_x,
                                     bits_per_block),
        queue_idx);
  } else if (D == 2) {
    int dim_x = decompressed_data.shape(1);
    int dim_y = decompressed_data.shape(0);
    int stride_x = 1;
    int stride_y = decompressed_data.ld(0);
    DeviceLauncher<DeviceType>::Execute(
        Decode2Kernel<T, DeviceType>(compressed_data.data(),
                                     decompressed_data.data(), dim_x, dim_y,
                                     stride_x, stride_y, bits_per_block),
        queue_idx);
  } else if (D == 3) {
    int dim_x = decompressed_data.shape(2);
    int dim_y = decompressed_data.shape(1);
    int dim_z = decompressed_data.shape(0);
    int stride_x = 1;
    int stride_y = decompressed_data.ld(1);
    int stride_z = decompressed_data.ld(0);
    DeviceLauncher<DeviceType>::Execute(
        Decode3Kernel<T, DeviceType>(
            compressed_data.data(), decompressed_data.data(), dim_x, dim_y,
            dim_z, stride_x, stride_y, stride_z, bits_per_block),
        queue_idx);
  }
  // errors.chk("Encode");
}

} // namespace zfp
} // namespace mgard_x

#endif