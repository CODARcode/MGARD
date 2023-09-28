#ifndef MGARD_X_ZFP_ENCODE2_CUH
#define MGARD_X_ZFP_ENCODE2_CUH

// #include "cuZFP.h"
#include "encode.h"
#include "shared.h"
// #include "ErrorCheck.h"
#include "type_info.h"

#define ZFP_2D_BLOCK_SIZE 16

namespace mgard_x {

namespace zfp {

template <typename Scalar>
MGARDX_CONT_EXEC void gather_partial2(Scalar *q, const Scalar *p, int nx,
                                      int ny, int sx, int sy) {
  uint x, y;
  for (y = 0; y < 4; y++)
    if (y < ny) {
      for (x = 0; x < 4; x++)
        if (x < nx) {
          q[4 * y + x] = *p; //[x * sx];
          p += sx;
        }
      pad_block(q + 4 * y, nx, 1);
      p += sy - nx * sx;
    }
  for (x = 0; x < 4; x++)
    pad_block(q + x, ny, 4);
}

template <typename Scalar>
MGARDX_CONT_EXEC void gather2(Scalar *q, const Scalar *p, int sx, int sy) {
  uint x, y;
  for (y = 0; y < 4; y++, p += sy - 4 * sx)
    for (x = 0; x < 4; x++, p += sx)
      *q++ = *p;
}

template <typename Scalar, typename DeviceType>
class Encode2Functor : public Functor<DeviceType> {
public:
  MGARDX_CONT Encode2Functor() {}
  MGARDX_CONT Encode2Functor(Scalar *scalars, ZFPWord *stream, uint dim_x,
                             uint dim_y, int stride_x, int stride_y,
                             uint padded_dim_x, uint padded_dim_y,
                             uint total_blocks, uint maxbits)
      : scalars(scalars), stream(stream), dim_x(dim_x), dim_y(dim_y),
        stride_x(stride_x), stride_y(stride_y), padded_dim_x(padded_dim_x),
        padded_dim_y(padded_dim_y), total_blocks(total_blocks),
        maxbits(maxbits) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    typedef unsigned long long int ull;
    typedef long long int ll;
    const ull blockId = FunctorBase<DeviceType>::GetBlockIdX() +
                        FunctorBase<DeviceType>::GetBlockIdY() *
                            FunctorBase<DeviceType>::GetGridDimX() +
                        FunctorBase<DeviceType>::GetGridDimX() *
                            FunctorBase<DeviceType>::GetGridDimY() *
                            FunctorBase<DeviceType>::GetBlockIdZ();

    // each thread gets a block so the block index is
    // the global thread index
    const uint block_idx = blockId * FunctorBase<DeviceType>::GetBlockDimX() +
                           FunctorBase<DeviceType>::GetThreadIdX();

    if (block_idx >= total_blocks) {
      // we can't launch the exact number of blocks
      // so just exit if this isn't real
      return;
    }

    uint block_dim_x, block_dim_y;
    block_dim_x = padded_dim_x >> 2;
    block_dim_y = padded_dim_y >> 2;

    // logical pos in 3d array
    uint block_x, block_y;
    block_x = (block_idx % block_dim_x) * 4;
    block_y = ((block_idx / block_dim_x) % block_dim_y) * 4;

    const ll offset = (ll)block_x * stride_x + (ll)block_y * stride_y;

    Scalar fblock[ZFP_2D_BLOCK_SIZE];

    bool partial = false;
    if (block_x + 4 > dim_x)
      partial = true;
    if (block_y + 4 > dim_y)
      partial = true;

    if (partial) {
      const uint nx = block_x + 4 > dim_x ? dim_x - block_x : 4;
      const uint ny = block_y + 4 > dim_y ? dim_y - block_y : 4;
      gather_partial2(fblock, scalars + offset, nx, ny, stride_x, stride_y);

    } else {
      gather2(fblock, scalars + offset, stride_x, stride_y);
    }

    zfp_encode_block<Scalar, ZFP_2D_BLOCK_SIZE, DeviceType>(fblock, maxbits,
                                                            block_idx, stream);
  }

private:
  Scalar *scalars;
  ZFPWord *stream;
  uint dim_x;
  uint dim_y;
  int stride_x;
  int stride_y;
  uint padded_dim_x;
  uint padded_dim_y;
  uint total_blocks;
  uint maxbits;
};

template <typename Scalar, typename DeviceType>
class Encode2Kernel : public Kernel {
public:
  constexpr static std::string_view Name = "ZFP_Encode2";
  constexpr static bool EnableAutoTuning() { return false; }
  MGARDX_CONT
  Encode2Kernel(Scalar *scalars, ZFPWord *stream, uint dim_x, uint dim_y,
                int stride_x, int stride_y, uint maxbits)
      : scalars(scalars), stream(stream), dim_x(dim_x), dim_y(dim_y),
        stride_x(stride_x), stride_y(stride_y), maxbits(maxbits) {}

  MGARDX_CONT Task<Encode2Functor<Scalar, DeviceType>> GenTask(int queue_idx) {
    const int device_block_size = 128;

    uint zfp_pad_x = dim_x, zfp_pad_y = dim_y;
    if (zfp_pad_x % 4 != 0)
      zfp_pad_x += 4 - dim_x % 4;
    if (zfp_pad_y % 4 != 0)
      zfp_pad_y += 4 - dim_y % 4;

    const uint zfp_blocks = (zfp_pad_x * zfp_pad_y) / 16;
    //
    // we need to ensure that we launch a multiple of the
    // cuda block size
    //
    int block_pad = 0;
    if (zfp_blocks % device_block_size != 0) {
      block_pad = device_block_size - zfp_blocks % device_block_size;
    }
    size_t total_blocks = block_pad + zfp_blocks;

    std::vector<SIZE> grid_size =
        calculate_grid_size(total_blocks, device_block_size);

    using FunctorType = Encode2Functor<Scalar, DeviceType>;
    FunctorType functor(scalars, stream, dim_x, dim_y, stride_x, stride_y,
                        zfp_pad_x, zfp_pad_y, total_blocks, maxbits);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = 0;
    tbz = 1;
    tby = 1;
    tbx = device_block_size;
    gridz = grid_size[2];
    gridy = grid_size[1];
    gridx = grid_size[0];
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  Scalar *scalars;
  ZFPWord *stream;
  uint dim_x;
  uint dim_y;
  int stride_x;
  int stride_y;
  uint maxbits;
};
/*
template<class Scalar>
__global__
void
cudaEncode2(const uint maxbits,
           const Scalar* scalars,
           Word *stream,
           const uint2 dims,
           const int2 stride,
           const uint2 padded_dims,
           const uint total_blocks)
{

  typedef unsigned long long int ull;
  typedef long long int ll;
  const ull blockId = blockIdx.x +
                      blockIdx.y * gridDim.x +
                      gridDim.x * gridDim.y * blockIdx.z;

  // each thread gets a block so the block index is
  // the global thread index
  const uint block_idx = blockId * blockDim.x + threadIdx.x;

  if(block_idx >= total_blocks)
  {
    // we can't launch the exact number of blocks
    // so just exit if this isn't real
    return;
  }

  uint2 block_dims;
  block_dims.x = padded_dims.x >> 2;
  block_dims.y = padded_dims.y >> 2;

  // logical pos in 3d array
  uint2 block;
  block.x = (block_idx % block_dims.x) * 4;
  block.y = ((block_idx/ block_dims.x) % block_dims.y) * 4;

  const ll offset = (ll)block.x * stride.x + (ll)block.y * stride.y;

  Scalar fblock[ZFP_2D_BLOCK_SIZE];

  bool partial = false;
  if(block.x + 4 > dims.x) partial = true;
  if(block.y + 4 > dims.y) partial = true;

  if(partial)
  {
    const uint nx = block.x + 4 > dims.x ? dims.x - block.x : 4;
    const uint ny = block.y + 4 > dims.y ? dims.y - block.y : 4;
    gather_partial2(fblock, scalars + offset, nx, ny, stride.x, stride.y);

  }
  else
  {
    gather2(fblock, scalars + offset, stride.x, stride.y);
  }

  zfp_encode_block<Scalar, ZFP_2D_BLOCK_SIZE>(fblock, maxbits, block_idx,
stream);

}

//
// Launch the encode kernel
//
template<class Scalar>
size_t encode2launch(uint2 dims,
                     int2 stride,
                     const Scalar *d_data,
                     Word *stream,
                     const int maxbits)
{
  const int cuda_block_size = 128;
  dim3 block_size = dim3(cuda_block_size, 1, 1);

  uint2 zfp_pad(dims);
  if(zfp_pad.x % 4 != 0) zfp_pad.x += 4 - dims.x % 4;
  if(zfp_pad.y % 4 != 0) zfp_pad.y += 4 - dims.y % 4;

  const uint zfp_blocks = (zfp_pad.x * zfp_pad.y) / 16;

  //
  // we need to ensure that we launch a multiple of the
  // cuda block size
  //
  int block_pad = 0;
  if(zfp_blocks % cuda_block_size != 0)
  {
    block_pad = cuda_block_size - zfp_blocks % cuda_block_size;
  }

  size_t total_blocks = block_pad + zfp_blocks;

  dim3 grid_size = calculate_grid_size(total_blocks, cuda_block_size);

  //
  size_t stream_bytes = calc_device_mem2d(zfp_pad, maxbits);
  // ensure we have zeros
  cudaMemset(stream, 0, stream_bytes);

#ifdef CUDA_ZFP_RATE_PRINT
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
#endif

  cudaEncode2<Scalar> <<<grid_size, block_size>>>
    (maxbits,
     d_data,
     stream,
     dims,
     stride,
     zfp_pad,
     zfp_blocks);

#ifdef CUDA_ZFP_RATE_PRINT
  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaStreamSynchronize(0);

  float milliseconds = 0.f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  float seconds = milliseconds / 1000.f;
  float mb = (float(dims.x * dims.y) * sizeof(Scalar)) / (1024.f * 1024.f
*1024.f); float rate = mb / seconds; printf("Encode elapsed time: %.5f (s)\n",
seconds); printf("# encode2 rate: %.2f (GB / sec) %d\n", rate, maxbits); #endif
  return stream_bytes;
}

template<class Scalar>
size_t encode2(uint2 dims,
               int2 stride,
               Scalar *d_data,
               Word *stream,
               const int maxbits)
{
  return encode2launch<Scalar>(dims, stride, d_data, stream, maxbits);
}
*/
} // namespace zfp
} // namespace mgard_x

#endif
