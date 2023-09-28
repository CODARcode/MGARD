#ifndef MGARD_X_ZFP_DECODE1_CUH
#define MGARD_X_ZFP_DECODE1_CUH

#include "decode.h"
#include "shared.h"
#include "type_info.h"

namespace mgard_x {

namespace zfp {

template <typename Scalar>
MGARDX_CONT_EXEC void scatter_partial1(const Scalar *q, Scalar *p, int nx,
                                       int sx) {
  uint x;
  for (x = 0; x < 4; x++)
    if (x < nx)
      p[x * sx] = q[x];
}

template <typename Scalar>
MGARDX_CONT_EXEC void scatter1(const Scalar *q, Scalar *p, int sx) {
  uint x;
  for (x = 0; x < 4; x++, p += sx)
    *p = *q++;
}

template <typename Scalar, typename DeviceType>
class Decode1Functor : public Functor<DeviceType> {
public:
  MGARDX_CONT Decode1Functor() {}
  MGARDX_CONT Decode1Functor(ZFPWord *stream, Scalar *scalars, uint dim_x,
                             int stride_x, uint padded_dim_x, uint total_blocks,
                             uint maxbits)
      : scalars(scalars), stream(stream), dim_x(dim_x), stride_x(stride_x),
        padded_dim_x(padded_dim_x), total_blocks(total_blocks),
        maxbits(maxbits) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    typedef unsigned long long int ull;
    typedef long long int ll;
    typedef typename zfp_traits<Scalar>::UInt UInt;
    typedef typename zfp_traits<Scalar>::Int Int;

    const int intprec = get_precision<Scalar>();

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

    BlockReader<ZFP_1D_BLOCK_SIZE, DeviceType> reader(stream, maxbits,
                                                      block_idx, total_blocks);
    Scalar result[ZFP_1D_BLOCK_SIZE] = {0, 0, 0, 0};

    zfp_decode<Scalar, ZFP_1D_BLOCK_SIZE, DeviceType>(reader, result, maxbits);

    uint block;
    block = block_idx * 4ull;
    const ll offset = (ll)block * stride_x;

    bool partial = false;
    if (block + 4 > dim_x)
      partial = true;
    if (partial) {
      const uint nx = 4u - (padded_dim_x - dim_x);
      scatter_partial1(result, scalars + offset, nx, stride_x);
    } else {
      scatter1(result, scalars + offset, stride_x);
    }
  }

private:
  ZFPWord *stream;
  Scalar *scalars;
  uint dim_x;
  int stride_x;
  uint padded_dim_x;
  uint total_blocks;
  uint maxbits;
};

template <typename Scalar, typename DeviceType>
class Decode1Kernel : public Kernel {
public:
  constexpr static std::string_view Name = "ZFP_Decode1";
  constexpr static bool EnableAutoTuning() { return false; }
  MGARDX_CONT
  Decode1Kernel(ZFPWord *stream, Scalar *scalars, uint dim_x, int stride_x,
                uint maxbits)
      : stream(stream), scalars(scalars), dim_x(dim_x), stride_x(stride_x),
        maxbits(maxbits) {}

  MGARDX_CONT Task<Decode1Functor<Scalar, DeviceType>> GenTask(int queue_idx) {
    const int device_block_size = 128;
    // dim3 block_size = dim3(device_block_size, 1, 1);

    uint zfp_pad_x = dim_x;
    if (zfp_pad_x % 4 != 0)
      zfp_pad_x += 4 - dim_x % 4;

    const uint zfp_blocks = (zfp_pad_x) / 4;
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

    using FunctorType = Decode1Functor<Scalar, DeviceType>;
    FunctorType functor(stream, scalars, dim_x, stride_x, zfp_pad_x,
                        total_blocks, maxbits);

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
  int stride_x;
  uint maxbits;
};
/*
template<class Scalar>
__global__
void
cudaDecode1(Word *blocks,
            Scalar *out,
            const uint dim,
            const int stride,
            const uint padded_dim,
            const uint total_blocks,
            uint maxbits)
{
  typedef unsigned long long int ull;
  typedef long long int ll;
  typedef typename zfp_traits<Scalar>::UInt UInt;
  typedef typename zfp_traits<Scalar>::Int Int;

  const int intprec = get_precision<Scalar>();

  const ull blockId = blockIdx.x +
                      blockIdx.y * gridDim.x +
                      gridDim.x  * gridDim.y * blockIdx.z;

  // each thread gets a block so the block index is
  // the global thread index
  const ull block_idx = blockId * blockDim.x + threadIdx.x;

  if(block_idx >= total_blocks) return;

  BlockReader<4> reader(blocks, maxbits, block_idx, total_blocks);
  Scalar result[4] = {0,0,0,0};

  zfp_decode(reader, result, maxbits);

  uint block;
  block = block_idx * 4ull;
  const ll offset = (ll)block * stride;

  bool partial = false;
  if(block + 4 > dim) partial = true;
  if(partial)
  {
    const uint nx = 4u - (padded_dim - dim);
    scatter_partial1(result, out + offset, nx, stride);
  }
  else
  {
    scatter1(result, out + offset, stride);
  }
}

template<class Scalar>
size_t decode1launch(uint dim,
                     int stride,
                     Word *stream,
                     Scalar *d_data,
                     uint maxbits)
{
  const int cuda_block_size = 128;

  uint zfp_pad(dim);
  if(zfp_pad % 4 != 0) zfp_pad += 4 - dim % 4;

  uint zfp_blocks = (zfp_pad) / 4;

  if(dim % 4 != 0)  zfp_blocks = (dim + (4 - dim % 4)) / 4;

  int block_pad = 0;
  if(zfp_blocks % cuda_block_size != 0)
  {
    block_pad = cuda_block_size - zfp_blocks % cuda_block_size;
  }

  size_t total_blocks = block_pad + zfp_blocks;
  size_t stream_bytes = calc_device_mem1d(zfp_pad, maxbits);

  dim3 block_size = dim3(cuda_block_size, 1, 1);
  dim3 grid_size = calculate_grid_size(total_blocks, cuda_block_size);

#ifdef CUDA_ZFP_RATE_PRINT
  // setup some timing code
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
#endif

  cudaDecode1<Scalar> << < grid_size, block_size >> >
    (stream,
                 d_data,
     dim,
     stride,
     zfp_pad,
     zfp_blocks, // total blocks to decode
     maxbits);

#ifdef CUDA_ZFP_RATE_PRINT
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
        cudaStreamSynchronize(0);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  float seconds = milliseconds / 1000.f;
  float rate = (float(dim) * sizeof(Scalar) ) / seconds;
  rate /= 1024.f;
  rate /= 1024.f;
  rate /= 1024.f;
  printf("Decode elapsed time: %.5f (s)\n", seconds);
  printf("# decode1 rate: %.2f (GB / sec) %d\n", rate, maxbits);
#endif
  return stream_bytes;
}

template<class Scalar>
size_t decode1(int dim,
               int stride,
               Word *stream,
               Scalar *d_data,
               uint maxbits)
{
        return decode1launch<Scalar>(dim, stride, stream, d_data, maxbits);
}
*/
} // namespace zfp
} // namespace mgard_x

#endif
