#include <vector>

/// Use this version of mgard_compress to compress your data with a tolerance
/// measured in  relative L-infty norm, version 1 for equispaced grids, 1a for
/// tensor product grids with arbitrary spacing

template <typename Real>
unsigned char *mgard_compress(int itype_flag, Real *data, int &out_size, int n1,
                              int n2, int n3, Real tol); // ...  1

template <typename Real>
unsigned char *mgard_compress(int itype_flag, Real *data, int &out_size, int n1,
                              int n2, int n3, std::vector<Real> &coords_x,
                              std::vector<Real> &coords_y,
                              std::vector<Real> &coords_z,
                              Real tol); // ... 1a

// Use this version of mgard_compress to compress your data with a tolerance
// measured in  relative s-norm.
// Set s=0 for L2-norm
// 2)
template <typename Real>
unsigned char *mgard_compress(int itype_flag, Real *data, int &out_size, int n1,
                              int n2, int n3, Real tol,
                              Real s); // ... 2

template <typename Real>
unsigned char *mgard_compress(int itype_flag, Real *data, int &out_size, int n1,
                              int n2, int n3, std::vector<Real> &coords_x,
                              std::vector<Real> &coords_y,
                              std::vector<Real> &coords_z, Real tol,
                              Real s); // ... 2a

// Use this version of mgard_compress to compress your data to preserve the
// error in a given quantity of interest Here qoi denotes the quantity of
// interest  which is a bounded linear functional in s-norm. This version
// recomputes the s-norm of the supplied linear functional every time it is
// invoked. If the same functional is to be reused for different sets of data
// then you are recommended to use one of the functions below (4, 5) to compute
// and store the norm and call MGARD using (6).
//

template <typename Real>
unsigned char *mgard_compress(int itype_flag, Real *data, int &out_size, int n1,
                              int n2, int n3, Real tol,
                              Real (*qoi)(int, int, int, Real *),
                              Real s); // ... 3

template <typename Real>
unsigned char *mgard_compress(int itype_flag, Real *data, int &out_size, int n1,
                              int n2, int n3, std::vector<Real> &coords_x,
                              std::vector<Real> &coords_y,
                              std::vector<Real> &coords_z, Real tol,
                              Real (*qoi)(int, int, int, Real *),
                              Real s); // ... 3a

// Use this version of mgard_compress to compute the  s-norm of a quantity of
// interest. Store this for further use if you wish to work with the same qoi in
// the future for different datasets.
template <typename Real>
Real mgard_compress(int n1, int n2, int n3,
                    Real (*qoi)(int, int, int, std::vector<Real>),
                    Real s); // ... 4

template <typename Real>
Real mgard_compress(int n1, int n2, int n3, std::vector<Real> &coords_x,
                    std::vector<Real> &coords_y, std::vector<Real> &coords_z,
                    Real (*qoi)(int, int, int, std::vector<Real>),
                    Real s); // ... 4a

// c-compatible version
template <typename Real>
Real mgard_compress(int n1, int n2, int n3, Real (*qoi)(int, int, int, Real *),
                    Real s); // ... 5

template <typename Real>
Real mgard_compress(int n1, int n2, int n3, std::vector<Real> &coords_x,
                    std::vector<Real> &coords_y, std::vector<Real> &coords_z,
                    Real (*qoi)(int, int, int, Real *),
                    Real s); // ... 5a

// Use this version of mgard_compress to compress your data with a tolerance in
// -s norm with given s-norm of quantity of interest qoi
template <typename Real>
unsigned char *mgard_compress(int itype_flag, Real *data, int &out_size, int n1,
                              int n2, int n3, Real tol, Real norm_of_qoi,
                              Real s); // ... 6

template <typename Real>
unsigned char *mgard_compress(int itype_flag, Real *data, int &out_size, int n1,
                              int n2, int n3, std::vector<Real> &coords_x,
                              std::vector<Real> &coords_y,
                              std::vector<Real> &coords_z, Real tol,
                              Real norm_of_qoi, Real s); // ... 6a

template <typename Real>
Real *mgard_decompress(int itype_flag, Real &quantizer, unsigned char *data,
                       int data_len, int n1, int n2,
                       int n3); // decompress L-infty compressed data

template <typename Real>
Real *mgard_decompress(
    int itype_flag, Real &quantizer, unsigned char *data, int data_len, int n1,
    int n2, int n3, std::vector<Real> &coords_x, std::vector<Real> &coords_y,
    std::vector<Real> &coords_z); // decompress L-infty compressed data

template <typename Real>
Real *mgard_decompress(int itype_flag, Real &quantizer, unsigned char *data,
                       int data_len, int n1, int n2, int n3,
                       Real s); // decompress s-norm

template <typename Real>
Real *mgard_decompress(int itype_flag, Real &quantizer, unsigned char *data,
                       int data_len, int n1, int n2, int n3,
                       std::vector<Real> &coords_x, std::vector<Real> &coords_y,
                       std::vector<Real> &coords_z,
                       Real s); // decompress s-norm

#include "mgard_api.tpp"
