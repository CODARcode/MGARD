#ifndef MGARD_X_ZFP_ENCODE3_CUH
#define MGARD_X_ZFP_ENCODE3_CUH

// #include "cuZFP.h"
#include "encode.h"
#include "shared.h"
#include "type_info.h"

#define ZFP_3D_BLOCK_SIZE 64
namespace mgard_x {

namespace zfp {

template <typename Scalar>
MGARDX_CONT_EXEC void gather_partial3(Scalar *q, const Scalar *p, int nx,
                                      int ny, int nz, int sx, int sy, int sz) {
  uint x, y, z;
  for (z = 0; z < 4; z++)
    if (z < nz) {
      for (y = 0; y < 4; y++)
        if (y < ny) {
          for (x = 0; x < 4; x++)
            if (x < nx) {
              q[16 * z + 4 * y + x] = *p;
              p += sx;
            }
          p += sy - nx * sx;
          pad_block(q + 16 * z + 4 * y, nx, 1);
        }
      for (x = 0; x < 4; x++)
        pad_block(q + 16 * z + x, ny, 4);
      p += sz - ny * sy;
    }
  for (y = 0; y < 4; y++)
    for (x = 0; x < 4; x++)
      pad_block(q + 4 * y + x, nz, 16);
}

template <typename Scalar>
MGARDX_CONT_EXEC void gather3(Scalar *q, const Scalar *p, int sx, int sy,
                              int sz) {
  uint x, y, z;
  for (z = 0; z < 4; z++, p += sz - 4 * sy)
    for (y = 0; y < 4; y++, p += sy - 4 * sx)
      for (x = 0; x < 4; x++, p += sx)
        *q++ = *p;
}

template <typename Scalar, typename DeviceType>
class Encode3Functor : public Functor<DeviceType> {
public:
  MGARDX_CONT Encode3Functor() {}
  MGARDX_CONT Encode3Functor(Scalar *scalars, ZFPWord *stream, uint dim_x,
                             uint dim_y, uint dim_z, int stride_x, int stride_y,
                             int stride_z, uint padded_dim_x, uint padded_dim_y,
                             uint padded_dim_z, uint total_blocks, uint maxbits)
      : scalars(scalars), stream(stream), dim_x(dim_x), dim_y(dim_y),
        dim_z(dim_z), stride_x(stride_x), stride_y(stride_y),
        stride_z(stride_z), padded_dim_x(padded_dim_x),
        padded_dim_y(padded_dim_y), padded_dim_z(padded_dim_z),
        total_blocks(total_blocks), maxbits(maxbits) {
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

    uint block_dim_x, block_dim_y, block_dim_z;
    block_dim_x = padded_dim_x >> 2;
    block_dim_y = padded_dim_y >> 2;
    block_dim_z = padded_dim_z >> 2;

    // logical pos in 3d array
    uint block_x, block_y, block_z;
    block_x = (block_idx % block_dim_x) * 4;
    block_y = ((block_idx / block_dim_x) % block_dim_y) * 4;
    block_z = (block_idx / (block_dim_x * block_dim_y)) * 4;

    // default strides
    ll offset = (ll)block_x * stride_x + (ll)block_y * stride_y +
                (ll)block_z * stride_z;
    Scalar fblock[ZFP_3D_BLOCK_SIZE];

    bool partial = false;
    if (block_x + 4 > dim_x)
      partial = true;
    if (block_y + 4 > dim_y)
      partial = true;
    if (block_z + 4 > dim_z)
      partial = true;

    if (partial) {
      const uint nx = block_x + 4 > dim_x ? dim_x - block_x : 4;
      const uint ny = block_y + 4 > dim_y ? dim_y - block_y : 4;
      const uint nz = block_z + 4 > dim_z ? dim_z - block_z : 4;
      gather_partial3(fblock, scalars + offset, nx, ny, nz, stride_x, stride_y,
                      stride_z);

    } else {
      gather3(fblock, scalars + offset, stride_x, stride_y, stride_z);
    }
    zfp_encode_block<Scalar, ZFP_3D_BLOCK_SIZE, DeviceType>(fblock, maxbits,
                                                            block_idx, stream);
  }

private:
  Scalar *scalars;
  ZFPWord *stream;
  uint dim_x;
  uint dim_y;
  uint dim_z;
  int stride_x;
  int stride_y;
  int stride_z;
  uint padded_dim_x;
  uint padded_dim_y;
  uint padded_dim_z;
  uint total_blocks;
  uint maxbits;
};

template <typename Scalar, typename DeviceType>
class Encode3Kernel : public Kernel {
public:
  constexpr static std::string_view Name = "ZFP_Encode3";
  constexpr static bool EnableAutoTuning() { return false; }
  MGARDX_CONT
  Encode3Kernel(Scalar *scalars, ZFPWord *stream, uint dim_x, uint dim_y,
                uint dim_z, int stride_x, int stride_y, int stride_z,
                uint maxbits)
      : scalars(scalars), stream(stream), dim_x(dim_x), dim_y(dim_y),
        dim_z(dim_z), stride_x(stride_x), stride_y(stride_y),
        stride_z(stride_z), maxbits(maxbits) {}

  MGARDX_CONT Task<Encode3Functor<Scalar, DeviceType>> GenTask(int queue_idx) {
    const int device_block_size = 128;

    uint zfp_pad_x = dim_x, zfp_pad_y = dim_y, zfp_pad_z = dim_z;
    if (zfp_pad_x % 4 != 0)
      zfp_pad_x += 4 - dim_x % 4;
    if (zfp_pad_y % 4 != 0)
      zfp_pad_y += 4 - dim_y % 4;
    if (zfp_pad_z % 4 != 0)
      zfp_pad_z += 4 - dim_z % 4;

    const uint zfp_blocks = (zfp_pad_x * zfp_pad_y * zfp_pad_z) / 64;
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

    using FunctorType = Encode3Functor<Scalar, DeviceType>;
    FunctorType functor(scalars, stream, dim_x, dim_y, dim_z, stride_x,
                        stride_y, stride_z, zfp_pad_x, zfp_pad_y, zfp_pad_z,
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
  uint dim_y;
  uint dim_z;
  int stride_x;
  int stride_y;
  int stride_z;
  uint maxbits;
};
/*
template<class Scalar>
__global__
void
cudaEncode(const uint maxbits,
           const Scalar* scalars,
           Word *stream,
           const uint3 dims,
           const int3 stride,
           const uint3 padded_dims,
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

  uint3 block_dims;
  block_dims.x = padded_dims.x >> 2;
  block_dims.y = padded_dims.y >> 2;
  block_dims.z = padded_dims.z >> 2;

  // logical pos in 3d array
  uint3 block;
  block.x = (block_idx % block_dims.x) * 4;
  block.y = ((block_idx/ block_dims.x) % block_dims.y) * 4;
  block.z = (block_idx/ (block_dims.x * block_dims.y)) * 4;

  // default strides
  ll offset = (ll)block.x * stride.x + (ll)block.y * stride.y + (ll)block.z *
stride.z; Scalar fblock[ZFP_3D_BLOCK_SIZE];

  bool partial = false;
  if(block.x + 4 > dims.x) partial = true;
  if(block.y + 4 > dims.y) partial = true;
  if(block.z + 4 > dims.z) partial = true;

  if(partial)
  {
    const uint nx = block.x + 4 > dims.x ? dims.x - block.x : 4;
    const uint ny = block.y + 4 > dims.y ? dims.y - block.y : 4;
    const uint nz = block.z + 4 > dims.z ? dims.z - block.z : 4;
    gather_partial3(fblock, scalars + offset, nx, ny, nz, stride.x, stride.y,
stride.z);

  }
  else
  {
    gather3(fblock, scalars + offset, stride.x, stride.y, stride.z);
  }
  zfp_encode_block<Scalar, ZFP_3D_BLOCK_SIZE>(fblock, maxbits, block_idx,
stream);

}

//
// Launch the encode kernel
//
template<class Scalar>
size_t encode3launch(uint3 dims,
                     int3 stride,
                     const Scalar *d_data,
                     Word *stream,
                     const int maxbits)
{

  const int cuda_block_size = 128;
  dim3 block_size = dim3(cuda_block_size, 1, 1);

  uint3 zfp_pad(dims);
  if(zfp_pad.x % 4 != 0) zfp_pad.x += 4 - dims.x % 4;
  if(zfp_pad.y % 4 != 0) zfp_pad.y += 4 - dims.y % 4;
  if(zfp_pad.z % 4 != 0) zfp_pad.z += 4 - dims.z % 4;

  const uint zfp_blocks = (zfp_pad.x * zfp_pad.y * zfp_pad.z) / 64;

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

  size_t stream_bytes = calc_device_mem3d(zfp_pad, maxbits);
  //ensure we start with 0s
  cudaMemset(stream, 0, stream_bytes);

#ifdef CUDA_ZFP_RATE_PRINT
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
#endif

  cudaEncode<Scalar> <<<grid_size, block_size>>>
    (maxbits,
     d_data,
     stream,
     dims,
     stride,
     zfp_pad,
     zfp_blocks);

#ifdef CUDA_ZFP_RATE_PRINT
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaStreamSynchronize(0);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  float seconds = milliseconds / 1000.f;
  float rate = (float(dims.x * dims.y * dims.z) * sizeof(Scalar) ) / seconds;
  rate /= 1024.f;
  rate /= 1024.f;
  rate /= 1024.f;
  printf("Encode elapsed time: %.5f (s)\n", seconds);
  printf("# encode3 rate: %.2f (GB / sec) \n", rate);
#endif
  return stream_bytes;
}

//
// Just pass the raw pointer to the "real" encode
//
template<class Scalar>
size_t encode(uint3 dims,
              int3 stride,
              Scalar *d_data,
              Word *stream,
              const int bits_per_block)
{
  return encode3launch<Scalar>(dims, stride, d_data, stream, bits_per_block);
}
*/
} // namespace zfp
} // namespace mgard_x
#endif
