/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "mgard-x/CompressionLowLevel/CompressionLowLevel.hpp"
#include "mgard-x/Hierarchy/Hierarchy.hpp"
#include "mgard-x/RuntimeX/DataStructures/Array.hpp"
#include "mgard-x/RuntimeX/RuntimeX.h"

namespace mgard_x {
//!\file
//!\brief Low level compression and decompression API.

//! Compress a function on an N-D tensor product grid
//!
//!\param[in] hierarchy Hierarchy type for storing pre-computed hierarchy
//! information
//!\param[in] in_array Dataset to be compressed.
//!\param[in] type Error bound type: REL or ABS.
//!\param[in] tol Error tolerance.
//!\param[in] s Smoothness parameter.
//!\param[out] norm Norm of the original data.
//!\param[in] config Configuring the compression process
//!
//!\return Compressed dataset.
template <DIM D, typename T, typename DeviceType>
void compress(Hierarchy<D, T, DeviceType> &hierarchy,
              Array<D, T, DeviceType> &in_array, enum error_bound_type type,
              T tol, T s, T &norm, Config config,
              CompressionLowLevelWorkspace<D, T, DeviceType> &workspace,
              Array<1, Byte, DeviceType> &compressed_array);

//! Decompress a function on an N-D tensor product grid
//!
//!\param[in] hierarchy Hierarchy type for storing pre-computed hierarchy
//! information.
//!\param[in] compressed_array Compressed dataset.
//!\param[in] type Error bound type: REL or ABS.
//!\param[in] tol Error tolerance.
//!\param[in] s Smoothness parameter.
//!\param[in] norm Norm of the original data.
//!\param[in] config Configuring the decompression process
//!
//!\return Decompressed dataset.
template <DIM D, typename T, typename DeviceType>
void decompress(Hierarchy<D, T, DeviceType> &hierarchy,
                Array<1, unsigned char, DeviceType> &compressed_array,
                enum error_bound_type type, T tol, T s, T norm, Config config,
                CompressionLowLevelWorkspace<D, T, DeviceType> &workspace,
                Array<D, T, DeviceType> &decompressed_array);

} // namespace mgard_x