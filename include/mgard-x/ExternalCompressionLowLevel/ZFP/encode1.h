#ifndef MGARD_X_ZFP_ENCODE1_CUH
#define MGARD_X_ZFP_ENCODE1_CUH

// #include "cuZFP.h"
#include "encode.h"
#include "shared.h"
#include "type_info.h"

#include <iostream>
#define ZFP_1D_BLOCK_SIZE 4

namespace mgard_x {

namespace zfp {

template <typename Scalar>
MGARDX_CONT_EXEC void gather_partial1(Scalar *q, const Scalar *p, int nx,
                                      int sx) {
  uint x;
  for (x = 0; x < 4; x++)
    if (x < nx)
      q[x] = p[x * sx];
  pad_block(q, nx, 1);
}

template <typename Scalar>
MGARDX_CONT_EXEC void gather1(Scalar *q, const Scalar *p, int sx) {
  uint x;
  for (x = 0; x < 4; x++, p += sx)
    *q++ = *p;
}

template <typename Scalar, typename DeviceType>
class Encode1Functor : public Functor<DeviceType> {
public:
  MGARDX_CONT Encode1Functor() {}
  MGARDX_CONT Encode1Functor(Scalar *scalars, ZFPWord *stream, uint dim_x,
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

    uint block_dim;
    block_dim = padded_dim_x >> 2;

    // logical pos in 3d array
    uint block;
    block = (block_idx % block_dim) * 4;

    const ll offset = (ll)block * stride_x;

    Scalar fblock[ZFP_1D_BLOCK_SIZE];

    bool partial = false;
    if (block + 4 > dim_x)
      partial = true;

    if (partial) {
      uint nx = 4 - (padded_dim_x - dim_x);
      gather_partial1(fblock, scalars + offset, nx, stride_x);
    } else {
      gather1(fblock, scalars + offset, stride_x);
    }

    zfp_encode_block<Scalar, ZFP_1D_BLOCK_SIZE, DeviceType>(fblock, maxbits,
                                                            block_idx, stream);
  }

private:
  Scalar *scalars;
  ZFPWord *stream;
  uint dim_x;
  int stride_x;
  uint padded_dim_x;
  uint total_blocks;
  uint maxbits;
};

template <typename Scalar, typename DeviceType>
class Encode1Kernel : public Kernel {
public:
  constexpr static std::string_view Name = "ZFP_Encode1";
  constexpr static bool EnableAutoTuning() { return false; }
  MGARDX_CONT
  Encode1Kernel(Scalar *scalars, ZFPWord *stream, uint dim_x, int stride_x,
                uint maxbits)
      : scalars(scalars), stream(stream), dim_x(dim_x), stride_x(stride_x),
        maxbits(maxbits) {}

  MGARDX_CONT Task<Encode1Functor<Scalar, DeviceType>> GenTask(int queue_idx) {
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

    using FunctorType = Encode1Functor<Scalar, DeviceType>;
    FunctorType functor(scalars, stream, dim_x, stride_x, zfp_pad_x,
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
  Scalar *scalars;
  ZFPWord *stream;
  uint dim_x;
  int stride_x;
  uint maxbits;
};

/*

template<class Scalar>
__global__
void
cudaEncode1(const uint maxbits,
           const Scalar* scalars,
           Word *stream,
           const uint dim,
           const int sx,
           const uint padded_dim,
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

  uint block_dim;
  block_dim = padded_dim >> 2;

  // logical pos in 3d array
  uint block;
  block = (block_idx % block_dim) * 4;

  const ll offset = (ll)block * sx;

  Scalar fblock[ZFP_1D_BLOCK_SIZE];

  bool partial = false;
  if(block + 4 > dim) partial = true;

  if(partial)
  {
    uint nx = 4 - (padded_dim - dim);
    gather_partial1(fblock, scalars + offset, nx, sx);
  }
  else
  {
    gather1(fblock, scalars + offset, sx);
  }

  zfp_encode_block<Scalar, ZFP_1D_BLOCK_SIZE>(fblock, maxbits, block_idx,
stream);

}
//
// Launch the encode kernel
//
template<class Scalar>
size_t encode1launch(uint dim,
                     int sx,
                     const Scalar *d_data,
                     Word *stream,
                     const int maxbits)
{
  const int cuda_block_size = 128;
  dim3 block_size = dim3(cuda_block_size, 1, 1);

  uint zfp_pad(dim);
  if(zfp_pad % 4 != 0) zfp_pad += 4 - dim % 4;

  const uint zfp_blocks = (zfp_pad) / 4;
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
  size_t stream_bytes = calc_device_mem1d(zfp_pad, maxbits);
  // ensure we have zeros
  cudaMemset(stream, 0, stream_bytes);

#ifdef CUDA_ZFP_RATE_PRINT
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
#endif

  cudaEncode1<Scalar> <<<grid_size, block_size>>>
    (maxbits,
     d_data,
     stream,
     dim,
     sx,
     zfp_pad,
     zfp_blocks);

#ifdef CUDA_ZFP_RATE_PRINT
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaStreamSynchronize(0);

  float milliseconds = 0.f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  float seconds = milliseconds / 1000.f;
  float gb = (float(dim) * float(sizeof(Scalar))) / (1024.f * 1024.f * 1024.f);
  float rate = gb / seconds;
  printf("Encode elapsed time: %.5f (s)\n", seconds);
  printf("# encode1 rate: %.2f (GB / sec) %d\n", rate, maxbits);
#endif
  return stream_bytes;
}

//
// Encode a host vector and output a encoded device vector
//
template<class Scalar>
size_t encode1(int dim,
               int sx,
               Scalar *d_data,
               Word *stream,
               const int maxbits)
{
  return encode1launch<Scalar>(dim, sx, d_data, stream, maxbits);
}
*/
} // namespace zfp
} // namespace mgard_x

#endif
