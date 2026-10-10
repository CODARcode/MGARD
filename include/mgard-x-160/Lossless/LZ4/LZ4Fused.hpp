/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 */

#ifndef MGARD_X_160_LZ4_FUSED_HPP
#define MGARD_X_160_LZ4_FUSED_HPP

#include "LZ4Kernels.hpp" // lz4_coop:: cooperative codec (SubGroup<> lives in the DeviceAdapters)

// GPU launch wrappers for the portable LZ4 backend: one sub-group (warp on
// CUDA, wavefront on HIP) per chunk, running the SAME lz4_coop::compress_chunk
// / decompress_chunk used by every backend -- here with the GPU sub-group, so
// the hash table lives in shared memory and the match scan / byte copies run
// across lanes. The portable functor path (LZ4Kernels) runs the identical codec
// with a size-1 SubGroupScalar. Both __global__ kernels are templated so their
// definitions have weak linkage and don't clash across GPU TUs (a non-template
// __global__ in a header -> nvlink "merge_elf failed"). chunk_size must be
// <= 65535 (uint16 hash slot). The HIP path is UNTESTED (no AMD hardware here).

#if defined(MGARDX_COMPILE_CUDA) || defined(MGARDX_COMPILE_HIP)

namespace mgard_x_160 {
namespace lz4_fused {

// Per-backend sub-group type and stream type.
#if defined(MGARDX_COMPILE_CUDA)
using FusedSubGroup = SubGroup<CUDA>;
using gpuStream_t = cudaStream_t;
#else
using FusedSubGroup = SubGroup<HIP>;
using gpuStream_t = hipStream_t;
#endif

static constexpr int WARPS_PER_BLOCK = 4;
static constexpr int SG = FusedSubGroup::size();

template <int WPB>
__global__ void compress_kernel(const Byte *__restrict__ input, SIZE n,
                                int chunk_size, SIZE nchunks,
                                SIZE max_chunk_out, Byte *__restrict__ scratch,
                                size_t *__restrict__ comp_bytes) {
  constexpr int HS = 1 << lz4::HASH_LOG;
  __shared__ uint16_t s_ht[WPB][HS];
  const int warp = threadIdx.x / FusedSubGroup::size();
  const SIZE c = (SIZE)blockIdx.x * WPB + warp;
  if (c >= nchunks)
    return;
  const SIZE start = c * (SIZE)chunk_size;
  const int len =
      (int)((SIZE)chunk_size < (n - start) ? (SIZE)chunk_size : (n - start));
  FusedSubGroup sg;
  int cb = lz4_coop::compress_chunk(
      sg, input + start, len, scratch + (size_t)c * max_chunk_out, s_ht[warp]);
  if (sg.lane() == 0)
    comp_bytes[c] = (size_t)cb;
}

template <int WPB>
__global__ void decompress_kernel(const Byte *__restrict__ packed,
                                  const size_t *__restrict__ byte_offset,
                                  SIZE n, int chunk_size, SIZE nchunks,
                                  Byte *__restrict__ output) {
  const int warp = threadIdx.x / FusedSubGroup::size();
  const SIZE c = (SIZE)blockIdx.x * WPB + warp;
  if (c >= nchunks)
    return;
  const SIZE start = c * (SIZE)chunk_size;
  const int outLen =
      (int)((SIZE)chunk_size < (n - start) ? (SIZE)chunk_size : (n - start));
  FusedSubGroup sg;
  lz4_coop::decompress_chunk(sg, packed + byte_offset[c], output + start,
                             outLen);
}

inline void launch_compress(const Byte *input, SIZE n, int chunk_size,
                            SIZE nchunks, SIZE max_chunk_out, Byte *scratch,
                            size_t *comp_bytes, gpuStream_t stream) {
  dim3 block(WARPS_PER_BLOCK * SG);
  dim3 grid((unsigned)((nchunks + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK));
  compress_kernel<WARPS_PER_BLOCK><<<grid, block, 0, stream>>>(
      input, n, chunk_size, nchunks, max_chunk_out, scratch, comp_bytes);
}

inline void launch_decompress(const Byte *packed, const size_t *byte_offset,
                              SIZE n, int chunk_size, SIZE nchunks,
                              Byte *output, gpuStream_t stream) {
  dim3 block(WARPS_PER_BLOCK * SG);
  dim3 grid((unsigned)((nchunks + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK));
  decompress_kernel<WARPS_PER_BLOCK><<<grid, block, 0, stream>>>(
      packed, byte_offset, n, chunk_size, nchunks, output);
}

// Fused chunks are valid only when a position fits the uint16 hash slot.
inline bool fused_ok(SIZE chunk_size) { return chunk_size <= 65535; }

} // namespace lz4_fused
} // namespace mgard_x_160

#endif // CUDA || HIP

// ---------------------------------------------------------------------------
// SYCL path (oneAPI/DPC++). UNTESTED -- no Intel GPU / SYCL toolchain here.
// SYCL can't use <<<>>>; it submits an nd_range parallel_for with a local
// accessor for the shared hash table. One sub-group per chunk runs the SAME
// lz4_coop codec via SubGroup<SYCL>. reqd_sub_group_size pins the width to 32.
// ---------------------------------------------------------------------------
#if defined(MGARDX_COMPILE_SYCL)
namespace mgard_x_160 {
namespace lz4_fused {

static constexpr int WARPS_PER_BLOCK = 4;
static constexpr int SYCL_SG = 32;

inline bool fused_ok(SIZE chunk_size) { return chunk_size <= 65535; }

inline void launch_compress(const Byte *input, SIZE n, int chunk_size,
                            SIZE nchunks, SIZE max_chunk_out, Byte *scratch,
                            size_t *comp_bytes, sycl::queue q) {
  constexpr int HS = 1 << lz4::HASH_LOG;
  constexpr int WPB = WARPS_PER_BLOCK;
  size_t groups = (nchunks + WPB - 1) / WPB;
  sycl::range<1> global(groups * WPB * SYCL_SG), local(WPB * SYCL_SG);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<uint16_t, 1> ht(sycl::range<1>(WPB * HS), h);
    h.parallel_for(
        sycl::nd_range<1>(global, local),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SYCL_SG)]] {
          sycl::sub_group nsg = it.get_sub_group();
          int warp = (int)nsg.get_group_id()[0];
          SIZE c = (SIZE)it.get_group(0) * WPB + warp;
          if (c >= nchunks)
            return;
          SIZE start = c * (SIZE)chunk_size;
          int len = (int)((SIZE)chunk_size < (n - start) ? (SIZE)chunk_size
                                                         : (n - start));
          SubGroup<SYCL> sg(nsg);
          uint16_t *htp = &ht[(size_t)warp * HS];
          int cb = lz4_coop::compress_chunk(
              sg, input + start, len, scratch + (size_t)c * max_chunk_out, htp);
          if (sg.lane() == 0)
            comp_bytes[c] = (size_t)cb;
        });
  });
}

inline void launch_decompress(const Byte *packed, const size_t *byte_offset,
                              SIZE n, int chunk_size, SIZE nchunks,
                              Byte *output, sycl::queue q) {
  constexpr int WPB = WARPS_PER_BLOCK;
  size_t groups = (nchunks + WPB - 1) / WPB;
  sycl::range<1> global(groups * WPB * SYCL_SG), local(WPB * SYCL_SG);
  q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<1>(global, local),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SYCL_SG)]] {
          sycl::sub_group nsg = it.get_sub_group();
          int warp = (int)nsg.get_group_id()[0];
          SIZE c = (SIZE)it.get_group(0) * WPB + warp;
          if (c >= nchunks)
            return;
          SIZE start = c * (SIZE)chunk_size;
          int outLen = (int)((SIZE)chunk_size < (n - start) ? (SIZE)chunk_size
                                                            : (n - start));
          SubGroup<SYCL> sg(nsg);
          lz4_coop::decompress_chunk(sg, packed + byte_offset[c],
                                     output + start, outLen);
        });
  });
}

} // namespace lz4_fused
} // namespace mgard_x_160
#endif // SYCL

#endif
