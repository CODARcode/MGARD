#ifndef MGARD_X_ZFP_TYPE_INFO
#define MGARD_X_ZFP_TYPE_INFO

#include <cfloat>

namespace mgard_x {

namespace zfp {

template <typename T> MGARDX_CONT_EXEC int get_ebias();
template <> MGARDX_CONT_EXEC int get_ebias<double>() { return 1023; }
template <> MGARDX_CONT_EXEC int get_ebias<float>() { return 127; }
template <> MGARDX_CONT_EXEC int get_ebias<long long int>() { return 0; }
template <> MGARDX_CONT_EXEC int get_ebias<int>() { return 0; }

template <typename T> MGARDX_CONT_EXEC int get_ebits();
template <> MGARDX_CONT_EXEC int get_ebits<double>() { return 11; }
template <> MGARDX_CONT_EXEC int get_ebits<float>() { return 8; }
template <> MGARDX_CONT_EXEC int get_ebits<int>() { return 0; }
template <> MGARDX_CONT_EXEC int get_ebits<long long int>() { return 0; }

template <typename T> MGARDX_CONT_EXEC int get_precision();
template <> MGARDX_CONT_EXEC int get_precision<double>() { return 64; }
template <> MGARDX_CONT_EXEC int get_precision<long long int>() { return 64; }
template <> MGARDX_CONT_EXEC int get_precision<float>() { return 32; }
template <> MGARDX_CONT_EXEC int get_precision<int>() { return 32; }

template <typename T> MGARDX_CONT_EXEC int get_min_exp();
template <> MGARDX_CONT_EXEC int get_min_exp<double>() { return -1074; }
template <> MGARDX_CONT_EXEC int get_min_exp<float>() { return -1074; }
template <> MGARDX_CONT_EXEC int get_min_exp<long long int>() { return 0; }
template <> MGARDX_CONT_EXEC int get_min_exp<int>() { return 0; }

template <typename T> MGARDX_CONT_EXEC T get_scalar_min();
template <> MGARDX_CONT_EXEC float get_scalar_min<float>() { return FLT_MIN; }
template <> MGARDX_CONT_EXEC double get_scalar_min<double>() { return DBL_MIN; }
template <> MGARDX_CONT_EXEC long long int get_scalar_min<long long int>() {
  return 0;
}
template <> MGARDX_CONT_EXEC int get_scalar_min<int>() { return 0; }

template <typename T> MGARDX_CONT_EXEC int scalar_sizeof();
template <> MGARDX_CONT_EXEC int scalar_sizeof<double>() { return 8; }
template <> MGARDX_CONT_EXEC int scalar_sizeof<long long int>() { return 8; }
template <> MGARDX_CONT_EXEC int scalar_sizeof<float>() { return 4; }
template <> MGARDX_CONT_EXEC int scalar_sizeof<int>() { return 4; }

template <typename T> MGARDX_CONT_EXEC T get_nbmask();
template <> MGARDX_CONT_EXEC unsigned int get_nbmask<unsigned int>() {
  return 0xaaaaaaaau;
}
template <>
MGARDX_CONT_EXEC unsigned long long int get_nbmask<unsigned long long int>() {
  return 0xaaaaaaaaaaaaaaaaull;
}

template <typename T> struct zfp_traits;

template <> struct zfp_traits<double> {
  typedef unsigned long long int UInt;
  typedef long long int Int;
};

template <> struct zfp_traits<long long int> {
  typedef unsigned long long int UInt;
  typedef long long int Int;
};

template <> struct zfp_traits<float> {
  typedef unsigned int UInt;
  typedef int Int;
};

template <> struct zfp_traits<int> {
  typedef unsigned int UInt;
  typedef int Int;
};

template <typename T> MGARDX_CONT_EXEC bool is_int() { return false; }

template <> MGARDX_CONT_EXEC bool is_int<int>() { return true; }

template <> MGARDX_CONT_EXEC bool is_int<long long int>() { return true; }

#if 0
template<int T> struct block_traits;

template<> struct block_traits<1>
{
  typedef unsigned char PlaneType;
};

template<> struct block_traits<2>
{
  typedef unsigned short PlaneType;
};
#endif

} // namespace zfp

} // namespace mgard_x

#endif
