#ifndef MGARD_X_ZFP_SHARED_H
#define MGARD_X_ZFP_SHARED_H

//#define CUDA_ZFP_RATE_PRINT 1
typedef unsigned long long ZFPWord;
#define ZFPWsize ((uint)(CHAR_BIT * sizeof(ZFPWord)))

#include "type_info.h"
//#include "zfp.h"
#include "constants.h"
#include <stdio.h>

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define bitsize(x) ((uint)(CHAR_BIT * sizeof(x)))

#define LDEXP(x, e) ldexp(x, e)

#define NBMASK 0xaaaaaaaaaaaaaaaaull

namespace mgard_x {

namespace zfp {

template <typename T> MGARDX_EXEC void print_bits(const T &bits) {
  const int bit_size = sizeof(T) * 8;

  for (int i = bit_size - 1; i >= 0; --i) {
    T one = 1;
    T mask = one << i;
    T val = (bits & mask) >> i;
    printf("%d", (int)val);
  }
  printf("\n");
}

size_t calc_device_mem1d(const int dim_x, const int maxbits);

size_t calc_device_mem2d(const int dim_y, const int dim_x, const int maxbits);

size_t calc_device_mem3d(const int encoded_dim_z, const int encoded_dim_y,
                         const int encoded_dim_x, const int bits_per_block);

std::vector<SIZE> get_max_grid_dims();

// size is assumed to have a pad to the nearest cuda block size
std::vector<SIZE> calculate_grid_size(size_t size, size_t cuda_block_size);

// map two's complement signed integer to negabinary unsigned integer
template <typename DeviceType>
MGARDX_EXEC unsigned long long int int2uint(const long long int x) {
  return (x + (unsigned long long int)0xaaaaaaaaaaaaaaaaull) ^
         (unsigned long long int)0xaaaaaaaaaaaaaaaaull;
}

template <typename DeviceType> MGARDX_EXEC unsigned int int2uint(const int x) {
  return (x + (unsigned int)0xaaaaaaaau) ^ (unsigned int)0xaaaaaaaau;
}

// map two's complement signed integer to negabinary unsigned integer
template <typename DeviceType>
MGARDX_EXEC long long int uint2int(unsigned long long int x) {
  return (x ^ 0xaaaaaaaaaaaaaaaaull) - 0xaaaaaaaaaaaaaaaaull;
}

template <typename DeviceType> MGARDX_EXEC int uint2int(unsigned int x) {
  return (x ^ 0xaaaaaaaau) - 0xaaaaaaaau;
}

template <typename Int, typename Scalar>
MGARDX_EXEC Scalar dequantize(const Int &x, const int &e);

template <>
MGARDX_EXEC double dequantize<long long int, double>(const long long int &x,
                                                     const int &e) {
  return LDEXP((double)x, e - ((int)(CHAR_BIT * scalar_sizeof<double>()) - 2));
}

template <>
MGARDX_EXEC float dequantize<int, float>(const int &x, const int &e) {
  return LDEXP((float)x, e - ((int)(CHAR_BIT * scalar_sizeof<float>()) - 2));
}

template <> MGARDX_EXEC int dequantize<int, int>(const int &x, const int &e) {
  return 1;
}

template <>
MGARDX_EXEC long long int
dequantize<long long int, long long int>(const long long int &x, const int &e) {
  return 1;
}

/* inverse lifting transform of 4-vector */
template <class Int, uint s> MGARDX_EXEC static void inv_lift(Int *p) {
  Int x, y, z, w;
  x = *p;
  p += s;
  y = *p;
  p += s;
  z = *p;
  p += s;
  w = *p;
  p += s;

  /*
  ** non-orthogonal transform
  **       ( 4  6 -4 -1) (x)
  ** 1/4 * ( 4  2  4  5) (y)
  **       ( 4 -2  4 -5) (z)
  **       ( 4 -6 -4  1) (w)
  */
  y += w >> 1;
  w -= y >> 1;
  y += w;
  w <<= 1;
  w -= y;
  z += x;
  x <<= 1;
  x -= z;
  y += z;
  z <<= 1;
  z -= y;
  w += x;
  x <<= 1;
  x -= w;

  p -= s;
  *p = w;
  p -= s;
  *p = z;
  p -= s;
  *p = y;
  p -= s;
  *p = x;
}

template <int BlockSize> MGARDX_EXEC const unsigned char *get_perm();

template <> MGARDX_EXEC const unsigned char *get_perm<64>() { return perm_3d; }

template <> MGARDX_EXEC const unsigned char *get_perm<16>() { return perm_2; }

template <> MGARDX_EXEC const unsigned char *get_perm<4>() { return perm_1; }

} // namespace zfp
} // namespace mgard_x
#endif
