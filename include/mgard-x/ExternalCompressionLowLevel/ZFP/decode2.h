#ifndef MGARD_X_ZFP_DECODE2_CUH
#define MGARD_X_ZFP_DECODE2_CUH

#include "decode.h"
#include "shared.h"
#include "type_info.h"

namespace mgard_x {

namespace zfp {

template <typename Scalar>
MGARDX_CONT_EXEC void scatter_partial2(const Scalar *q, Scalar *p, int nx,
                                       int ny, int sx, int sy) {
  uint x, y;
  for (y = 0; y < 4; y++)
    if (y < ny) {
      for (x = 0; x < 4; x++)
        if (x < nx) {
          *p = q[4 * y + x];
          p += sx;
        }
      p += sy - nx * sx;
    }
}

template <typename Scalar>
MGARDX_CONT_EXEC void scatter2(const Scalar *q, Scalar *p, int sx, int sy) {
  uint x, y;
  for (y = 0; y < 4; y++, p += sy - 4 * sx)
    for (x = 0; x < 4; x++, p += sx)
      *p = *q++;
}

template <typename Scalar, typename DeviceType>
class Decode2Functor : public Functor<DeviceType> {
public:
  MGARDX_CONT Decode2Functor() {}
  MGARDX_CONT Decode2Functor(ZFPWord *stream, Scalar *scalars, uint dim_x,
                             uint dim_y, int stride_x, int stride_y,
                             uint padded_dim_x, uint padded_dim_y,
                             uint total_blocks, uint maxbits)
      : stream(stream), scalars(scalars), dim_x(dim_x), dim_y(dim_y),
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

    BlockReader<ZFP_2D_BLOCK_SIZE, DeviceType> reader(stream, maxbits,
                                                      block_idx, total_blocks);

    Scalar result[ZFP_2D_BLOCK_SIZE];
    memset(result, 0, sizeof(Scalar) * ZFP_2D_BLOCK_SIZE);

    zfp_decode<Scalar, ZFP_2D_BLOCK_SIZE, DeviceType>(reader, result, maxbits);

    // logical block dims
    uint block_dim_x, block_dim_y;
    block_dim_x = padded_dim_x >> 2;
    block_dim_y = padded_dim_y >> 2;
    // logical pos in 3d array
    uint block_x, block_y;
    block_x = (block_idx % block_dim_x) * 4;
    block_y = ((block_idx / block_dim_x) % block_dim_y) * 4;

    const ll offset = (ll)block_x * stride_x + (ll)block_y * stride_y;

    bool partial = false;
    if (block_x + 4 > dim_x)
      partial = true;
    if (block_y + 4 > dim_y)
      partial = true;

    if (partial) {
      const uint nx = block_x + 4 > dim_x ? dim_x - block_x : 4;
      const uint ny = block_y + 4 > dim_y ? dim_y - block_y : 4;
      scatter_partial2(result, scalars + offset, nx, ny, stride_x, stride_y);
    } else {
      scatter2(result, scalars + offset, stride_x, stride_y);
    }
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
class Decode2Kernel : public Kernel {
public:
  constexpr static std::string_view Name = "ZFP_Decode2";
  constexpr static bool EnableAutoTuning() { return false; }
  MGARDX_CONT
  Decode2Kernel(ZFPWord *stream, Scalar *scalars, uint dim_x, uint dim_y,
                int stride_x, int stride_y, uint maxbits)
      : stream(stream), scalars(scalars), dim_x(dim_x), dim_y(dim_y),
        stride_x(stride_x), stride_y(stride_y), maxbits(maxbits) {}

  MGARDX_CONT Task<Decode2Functor<Scalar, DeviceType>> GenTask(int queue_idx) {
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

    using FunctorType = Decode2Functor<Scalar, DeviceType>;
    FunctorType functor(stream, scalars, dim_x, dim_y, stride_x, stride_y,
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
  ZFPWord *stream;
  Scalar *scalars;
  uint dim_x;
  uint dim_y;
  int stride_x;
  int stride_y;
  uint maxbits;
};

/*
template<class Scalar, int BlockSize>
__global__
void
cudaDecode2(Word *blocks,
            Scalar *out,
            const uint2 dims,
            const int2 stride,
            const uint2 padded_dims,
            uint maxbits)
{
  typedef unsigned long long int ull;
  typedef long long int ll;
  const ull blockId = blockIdx.x +
                      blockIdx.y * gridDim.x +
                      gridDim.x * gridDim.y * blockIdx.z;

  // each thread gets a block so the block index is
  // the global thread index
  const ull block_idx = blockId * blockDim.x + threadIdx.x;

  const int total_blocks = (padded_dims.x * padded_dims.y) / 16;

  if(block_idx >= total_blocks)
  {
    return;
  }

  BlockReader<BlockSize> reader(blocks, maxbits, block_idx, total_blocks);

  Scalar result[BlockSize];
  memset(result, 0, sizeof(Scalar) * BlockSize);

  zfp_decode(reader, result, maxbits);

  // logical block dims
  uint2 block_dims;
  block_dims.x = padded_dims.x >> 2;
  block_dims.y = padded_dims.y >> 2;
  // logical pos in 3d array
  uint2 block;
  block.x = (block_idx % block_dims.x) * 4;
  block.y = ((block_idx/ block_dims.x) % block_dims.y) * 4;

  const ll offset = (ll)block.x * stride.x + (ll)block.y * stride.y;

  bool partial = false;
  if(block.x + 4 > dims.x) partial = true;
  if(block.y + 4 > dims.y) partial = true;
  if(partial)
  {
    const uint nx = block.x + 4 > dims.x ? dims.x - block.x : 4;
    const uint ny = block.y + 4 > dims.y ? dims.y - block.y : 4;
    scatter_partial2(result, out + offset, nx, ny, stride.x, stride.y);
  }
  else
  {
    scatter2(result, out + offset, stride.x, stride.y);
  }
}

template<class Scalar>
size_t decode2launch(uint2 dims,
                     int2 stride,
                     Word *stream,
                     Scalar *d_data,
                     uint maxbits)
{
  const int cuda_block_size = 128;
  dim3 block_size;
  block_size = dim3(cuda_block_size, 1, 1);

  uint2 zfp_pad(dims);
  // ensure that we have block sizes
  // that are a multiple of 4
  if(zfp_pad.x % 4 != 0) zfp_pad.x += 4 - dims.x % 4;
  if(zfp_pad.y % 4 != 0) zfp_pad.y += 4 - dims.y % 4;

  const int zfp_blocks = (zfp_pad.x * zfp_pad.y) / 16;


  //
  // we need to ensure that we launch a multiple of the
  // cuda block size
  //
  int block_pad = 0;
  if(zfp_blocks % cuda_block_size != 0)
  {
    block_pad = cuda_block_size - zfp_blocks % cuda_block_size;
  }


  size_t stream_bytes = calc_device_mem2d(zfp_pad, maxbits);
  size_t total_blocks = block_pad + zfp_blocks;
  dim3 grid_size = calculate_grid_size(total_blocks, cuda_block_size);

#ifdef CUDA_ZFP_RATE_PRINT
  // setup some timing code
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
#endif

  cudaDecode2<Scalar, 16> << < grid_size, block_size >> >
    (stream,
                 d_data,
     dims,
     stride,
     zfp_pad,
     maxbits);

#ifdef CUDA_ZFP_RATE_PRINT
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
        cudaStreamSynchronize(0);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  float seconds = milliseconds / 1000.f;
  float rate = (float(dims.x * dims.y) * sizeof(Scalar) ) / seconds;
  rate /= 1024.f;
  rate /= 1024.f;
  rate /= 1024.f;
  printf("Decode elapsed time: %.5f (s)\n", seconds);
  printf("# decode2 rate: %.2f (GB / sec) %d\n", rate, maxbits);
#endif
  return stream_bytes;
}

template<class Scalar>
size_t decode2(uint2 dims,
               int2 stride,
               Word *stream,
               Scalar *d_data,
               uint maxbits)
{
        return decode2launch<Scalar>(dims, stride, stream, d_data, maxbits);
}
*/
} // namespace zfp
} // namespace mgard_x

#endif
