// Copyright 2017, Brown University, Providence, RI.
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney
// Corresponding Author: Ozan Tugluk
//
// version: 0.0.0.2
// See LICENSE for details.
#ifndef MGARD_API_H
#define MGARD_API_H
//!\file
//!\brief Compression and decompression API.

#include <vector>

//! Compress a function on an equispaced 3D tensor product grid while
//! controlling the error as measured in the \f$ L^{\infty} \f$ norm.
//!
//!\param[in] itype_flag Flag to specify the datatype. Unused.
//!\param[in] data Dataset to be compressed.
//!\param[out] out_size Size in bytes of the compressed dataset.
//!\param[in] n1 Size of the dataset in the first dimension.
//!\param[in] n2 Size of the dataset in the second dimension.
//!\param[in] n3 Size of the dataset in the third dimension.
//!\param[in] tol Relative error tolerance.
//!
//!\return Compressed dataset.
template <typename Real>
unsigned char *mgard_compress(int itype_flag, Real *data, int &out_size, int n1,
                              int n2, int n3, Real tol);

//! Compress a function on a 3D tensor product grid (with arbitrary node
//! spacing) while controlling the error as measured in the \f$ L^{\infty} \f$
//! norm.
//!
//!\param[in] itype_flag Flag to specify the datatype. Unused.
//!\param[in] data Dataset to be compressed.
//!\param[out] out_size Size in bytes of the compressed dataset.
//!\param[in] n1 Size of the dataset in the first dimension.
//!\param[in] n2 Size of the dataset in the second dimension.
//!\param[in] n3 Size of the dataset in the third dimension.
//!\param[in] coords_x First coordinates of the nodes of the grid.
//!\param[in] coords_y Second coordinates of the nodes of the grid.
//!\param[in] coords_z Third coordinates of the nodes of the grid.
//!\param[in] tol Relative error tolerance.
//!
//!\return Compressed dataset.
template <typename Real>
unsigned char *mgard_compress(int itype_flag, Real *data, int &out_size, int n1,
                              int n2, int n3, std::vector<Real> &coords_x,
                              std::vector<Real> &coords_y,
                              std::vector<Real> &coords_z, Real tol);

//! Compress a function on an equispaced 3D tensor product grid while
//! controlling the error as measured in the `s` norm.
//!
//!\note Set `s` to zero to control the error as measured in the \f$ L^{2} \f$
//! norm.
//!
//!\param[in] itype_flag Flag to specify the datatype. Unused.
//!\param[in] data Dataset to be compressed.
//!\param[out] out_size Size in bytes of the compressed dataset.
//!\param[in] n1 Size of the dataset in the first dimension.
//!\param[in] n2 Size of the dataset in the second dimension.
//!\param[in] n3 Size of the dataset in the third dimension.
//!\param[in] tol Relative error tolerance.
//!\param[in] s Smoothness parameter. Determines the error norm used when
//! compressing the data.
//!
//!\return Compressed dataset.
template <typename Real>
unsigned char *mgard_compress(int itype_flag, Real *data, int &out_size, int n1,
                              int n2, int n3, Real tol, Real s);

// TODO: the meaning of `s` changes when a functional is passed in. Roughly,
// without a functional `s` is the smoothness assumed of the data, and with a
// functional `s` is the smoothness assumed of the Riesz representative of the
// functional. In the latter case, then, the data is assumed to have smoothness
// *`-s`*. This should be changed.

//! Compress a function on an equispaced 3D tensor product grid while
//! controlling the error in a quantity of interest.
//!
//!\note This function recomputes the operator norm of the supplied
//! linear functional every time it is invoked. If the same functional is to be
//! reused for different sets of data, it is recommended to compute and save the
//! norm of the functional and to then compress using the overload that takes
// the functional norm as a parameter.
//!
//!\param[in] itype_flag Flag to specify the datatype. Unused.
//!\param[in] data Dataset to be compressed.
//!\param[out] out_size Size in bytes of the compressed dataset.
//!\param[in] n1 Size of the dataset in the first dimension.
//!\param[in] n2 Size of the dataset in the second dimension.
//!\param[in] n3 Size of the dataset in the third dimension.
//!\param[in] tol Error tolerance.
//!\param[in] qoi Quantity of interest to be preserved.
//!\param[in] s Smoothness parameter. Determines the norm used when computing
//! the norm of the functional; `-s` is used when compressing the data.
//!
//!\return Compressed dataset.
template <typename Real>
unsigned char *mgard_compress(int itype_flag, Real *data, int &out_size, int n1,
                              int n2, int n3, Real tol,
                              Real (*qoi)(int, int, int, Real *), Real s);

//! Compute the operator norm of a linear functional.
//!
//!\param n1 Size of the domain grid in the first dimension.
//!\param n2 Size of the domain grid in the second dimension.
//!\param n3 Size of the domain grid in the third dimension.
//!\param qoi Quantity of interest whose norm is to be computed.
//!\param s Smoothness parameter. The norm of the Riesz representative of the
//! functional will be computed using the `s` norm.
template <typename Real>
Real mgard_compress(int n1, int n2, int n3,
                    Real (*qoi)(int, int, int, std::vector<Real>), Real s);

//! Compute the operator norm of a linear functional.
//!
//!\note This is a C-compatible overload of the above function, differing only
//! in the type of `qoi`.
//!
//!\param n1 Size of the domain grid in the first dimension.
//!\param n2 Size of the domain grid in the second dimension.
//!\param n3 Size of the domain grid in the third dimension.
//!\param qoi Quantity of interest whose norm is to be computed.
//!\param s Smoothness parameter. The norm of the Riesz representative of the
//! functional will be computed using the `s` norm.
template <typename Real>
Real mgard_compress(int n1, int n2, int n3, Real (*qoi)(int, int, int, Real *),
                    Real s);

//! Compress a function on an equispaced 3D tensor product grid while
//! controlling the error in a quantity of interest.
//!
//!\param[in] itype_flag Flag to specify the datatype. Unused.
//!\param[in] data Dataset to be compressed.
//!\param[out] out_size Size in bytes of the compressed dataset.
//!\param[in] n1 Size of the dataset in the first dimension.
//!\param[in] n2 Size of the dataset in the second dimension.
//!\param[in] n3 Size of the dataset in the third dimension.
//!\param[in] tol Error tolerance.
//!\param[in] norm_of_qoi `s` operator norm of the quantity of interest to be
//! preserved.
//!\param[in] s Smoothness parameter. The data will be compressed using
//! smoothness parameter `-s`.
//!
//!\return Compressed dataset.
template <typename Real>
unsigned char *mgard_compress(int itype_flag, Real *data, int &out_size, int n1,
                              int n2, int n3, Real tol, Real norm_of_qoi,
                              Real s);

//! Decompress a function on an equispaced 3D tensor product grid which was
//! compressed while controlling the error as measured in the \f$ L^{\infty} \f$
//! norm.
//!
//!\param[in] itype_flag Flag to specify the datatype. Unused.
//!\param[in, out] quantizer Unused.
//!\param[in] data Compressed dataset.
//!\param[in] data_len Size in bytes of the compressed dataset.
//!\param[in] n1 Size of the dataset in the first dimension.
//!\param[in] n2 Size of the dataset in the second dimension.
//!\param[in] n3 Size of the dataset in the third dimension.
//!
//!\return Decompressed dataset.
template <typename Real>
Real *mgard_decompress(int itype_flag, Real &quantizer, unsigned char *data,
                       int data_len, int n1, int n2, int n3);

//! Decompress a function on an equispaced 3D tensor product grid which was
//! compressed while controlling the error as measured in the `s` norm.
//!
//!\param[in] itype_flag Flag to specify the datatype. Unused.
//!\param[in, out] quantizer Unused.
//!\param[in] data Compressed dataset.
//!\param[in] data_len Size in bytes of the compressed dataset.
//!\param[in] n1 Size of the dataset in the first dimension.
//!\param[in] n2 Size of the dataset in the second dimension.
//!\param[in] n3 Size of the dataset in the third dimension.
//!\param[in] s Smoothness parameter used when compressing the data.
//!
//!\return Decompressed dataset.
template <typename Real>
Real *mgard_decompress(int itype_flag, Real &quantizer, unsigned char *data,
                       int data_len, int n1, int n2, int n3, Real s);

#endif
