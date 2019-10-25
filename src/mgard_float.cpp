// Copyright 2017, Brown University, Providence, RI.
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney
// Corresponding Author: Ozan Tugluk
//
// version: 0.0.0.2
// See LICENSE for details.

#include "mgard_float.h"
#include "mgard_nuni_float.h"

namespace mgard {

unsigned char *refactor_qz(int nrow, int ncol, int nfib, const float *u,
                           int &outsize, float tol) {
  int nlevel;
  std::vector<float> v(u, u + nrow * ncol * nfib), work(nrow * ncol * nfib),
      work2d(nrow * ncol); // duplicate data and create work array
  std::vector<float> coords_x(ncol), coords_y(nrow),
      coords_z(nfib); // coordinate arrays

  int nlevel_x = std::log2(ncol - 1);
  int nc = std::pow(2, nlevel_x) + 1; // ncol new

  int nlevel_y = std::log2(nrow - 1);
  int nr = std::pow(2, nlevel_y) + 1; // nrow new

  int nlevel_z = std::log2(nfib - 1);
  int nf = std::pow(2, nlevel_z) + 1; // nfib new

  nlevel = std::min(nlevel_x, nlevel_y);
  nlevel = std::min(nlevel, nlevel_z);

  int l_target = nlevel - 1;

  // std::cout  << "Linfinity\n" ;
  // dummy equispaced coordinates
  std::iota(std::begin(coords_x), std::end(coords_x), 0);
  std::iota(std::begin(coords_y), std::end(coords_y), 0);
  std::iota(std::begin(coords_z), std::end(coords_z), 0);

  //  float norm =  mgard_gen::ml2_norm3(0,  nrow,  ncol,  nfib ,  nrow,  ncol,
  //  nfib,   v, coords_x, coords_y, coords_z);

  float norm = mgard_common::max_norm(v);

  tol /= nlevel + 2;

  mgard_gen::prep_3D(nr, nc, nf, nrow, ncol, nfib, l_target, v.data(), work,
                     work2d, coords_x, coords_y, coords_z);

  mgard_gen::refactor_3D(nr, nc, nf, nrow, ncol, nfib, l_target, v.data(), work,
                         work2d, coords_x, coords_y, coords_z);

  work.clear();
  work2d.clear();

  int size_ratio = sizeof(float) / sizeof(int);
  std::vector<int> qv(nrow * ncol * nfib + size_ratio);

  mgard::quantize_2D_interleave(
      nrow, ncol * nfib, v.data(), qv, norm,
      tol); // rename this to quantize Linfty or smthng!!!!

  std::vector<unsigned char> out_data;

  mgard::compress_memory_z(qv.data(), sizeof(int) * qv.size(), out_data);

  outsize = out_data.size();
  unsigned char *buffer = (unsigned char *)malloc(outsize);
  std::copy(out_data.begin(), out_data.end(), buffer);
  return buffer;
}

unsigned char *refactor_qz(int nrow, int ncol, int nfib,
                           std::vector<float> &coords_x,
                           std::vector<float> &coords_y,
                           std::vector<float> &coords_z, const float *u,
                           int &outsize, float tol) {
  int nlevel;
  std::vector<float> v(u, u + nrow * ncol * nfib), work(nrow * ncol * nfib),
      work2d(nrow * ncol); // duplicate data and create work array

  int nlevel_x = std::log2(ncol - 1);
  int nc = std::pow(2, nlevel_x) + 1; // ncol new

  int nlevel_y = std::log2(nrow - 1);
  int nr = std::pow(2, nlevel_y) + 1; // nrow new

  int nlevel_z = std::log2(nfib - 1);
  int nf = std::pow(2, nlevel_z) + 1; // nfib new

  nlevel = std::min(nlevel_x, nlevel_y);
  nlevel = std::min(nlevel, nlevel_z);

  int l_target = nlevel - 1;

  float norm = mgard_common::max_norm(v);

  tol /= nlevel + 2;

  mgard_gen::prep_3D(nr, nc, nf, nrow, ncol, nfib, l_target, v.data(), work,
                     work2d, coords_x, coords_y, coords_z);

  mgard_gen::refactor_3D(nr, nc, nf, nrow, ncol, nfib, l_target, v.data(), work,
                         work2d, coords_x, coords_y, coords_z);

  work.clear();
  work2d.clear();

  int size_ratio = sizeof(float) / sizeof(int);
  std::vector<int> qv(nrow * ncol * nfib + size_ratio);

  mgard::quantize_2D_interleave(
      nrow, ncol * nfib, v.data(), qv, norm,
      tol); // rename this to quantize Linfty or smthng!!!!

  std::vector<unsigned char> out_data;

  mgard::compress_memory_z(qv.data(), sizeof(int) * qv.size(), out_data);

  outsize = out_data.size();
  unsigned char *buffer = (unsigned char *)malloc(outsize);
  std::copy(out_data.begin(), out_data.end(), buffer);
  return buffer;
}

unsigned char *refactor_qz(int nrow, int ncol, int nfib, const float *u,
                           int &outsize, float tol, float s) {
  int nlevel;
  std::vector<float> v(u, u + nrow * ncol * nfib), work(nrow * ncol * nfib),
      work2d(nrow * ncol); // duplicate data and create work array
  std::vector<float> coords_x(ncol), coords_y(nrow),
      coords_z(nfib); // coordinate arrays

  int nlevel_x = std::log2(ncol - 1);
  int nc = std::pow(2, nlevel_x) + 1; // ncol new

  int nlevel_y = std::log2(nrow - 1);
  int nr = std::pow(2, nlevel_y) + 1; // nrow new

  int nlevel_z = std::log2(nfib - 1);
  int nf = std::pow(2, nlevel_z) + 1; // nfib new

  nlevel = std::min(nlevel_x, nlevel_y);
  nlevel = std::min(nlevel, nlevel_z);

  int l_target = nlevel - 1;

  // dummy equispaced coordinates
  std::iota(std::begin(coords_x), std::end(coords_x), 0);
  std::iota(std::begin(coords_y), std::end(coords_y), 0);
  std::iota(std::begin(coords_z), std::end(coords_z), 0);

  // float norm =  mgard_gen::ml2_norm3(0,  nrow,  ncol,  nfib ,  nrow,  ncol,
  // nfib,   v, coords_x, coords_y, coords_z);

  float norm = 1.0;

  if (std::abs(0) < 1e-10) {
    norm =
        mgard_gen::ml2_norm3(0, nrow, ncol, nfib, nrow, ncol, nfib, v, coords_x,
                             coords_y, coords_z); // mgard_common::max_norm(v);
    norm = std::sqrt(norm) / std::sqrt(nrow * ncol * nfib);
  }

  // std::cout  << "My 2-norm is: " << norm << "\n";

  mgard_gen::prep_3D(nr, nc, nf, nrow, ncol, nfib, l_target, v.data(), work,
                     work2d, coords_x, coords_y, coords_z);

  mgard_gen::refactor_3D(nr, nc, nf, nrow, ncol, nfib, l_target, v.data(), work,
                         work2d, coords_x, coords_y, coords_z);

  work.clear();
  work2d.clear();

  int size_ratio = sizeof(float) / sizeof(int);
  std::vector<int> qv(nrow * ncol * nfib + size_ratio);
  // qv.reserve(nrow * ncol * nfib + size_ratio);
  //  qv[0] = 0; qv[1] =0;

  mgard_gen::quantize_3D(nr, nc, nf, nrow, ncol, nfib, nlevel, v.data(), qv,
                         coords_x, coords_y, coords_z, s, norm, tol);

  std::vector<unsigned char> out_data;

  mgard::compress_memory_z(qv.data(), sizeof(int) * qv.size(), out_data);

  outsize = out_data.size();
  unsigned char *buffer = (unsigned char *)malloc(outsize);
  std::copy(out_data.begin(), out_data.end(), buffer);
  return buffer;
}

unsigned char *refactor_qz(int nrow, int ncol, int nfib,
                           std::vector<float> &coords_x,
                           std::vector<float> &coords_y,
                           std::vector<float> &coords_z, const float *u,
                           int &outsize, float tol, float s) {
  int nlevel;
  std::vector<float> v(u, u + nrow * ncol * nfib), work(nrow * ncol * nfib),
      work2d(nrow * ncol); // duplicate data and create work array

  int nlevel_x = std::log2(ncol - 1);
  int nc = std::pow(2, nlevel_x) + 1; // ncol new

  int nlevel_y = std::log2(nrow - 1);
  int nr = std::pow(2, nlevel_y) + 1; // nrow new

  int nlevel_z = std::log2(nfib - 1);
  int nf = std::pow(2, nlevel_z) + 1; // nfib new

  nlevel = std::min(nlevel_x, nlevel_y);
  nlevel = std::min(nlevel, nlevel_z);

  int l_target = nlevel - 1;

  float norm = 1.0;

  if (std::abs(0) < 1e-10) {
    norm =
        mgard_gen::ml2_norm3(0, nrow, ncol, nfib, nrow, ncol, nfib, v, coords_x,
                             coords_y, coords_z); // mgard_common::max_norm(v);

    norm = std::sqrt(norm) / std::sqrt(nrow * ncol * nfib);
  }

  mgard_gen::prep_3D(nr, nc, nf, nrow, ncol, nfib, l_target, v.data(), work,
                     work2d, coords_x, coords_y, coords_z);

  mgard_gen::refactor_3D(nr, nc, nf, nrow, ncol, nfib, l_target, v.data(), work,
                         work2d, coords_x, coords_y, coords_z);

  work.clear();
  work2d.clear();

  int size_ratio = sizeof(float) / sizeof(int);
  std::vector<int> qv(nrow * ncol * nfib + size_ratio);

  mgard_gen::quantize_3D(nr, nc, nf, nrow, ncol, nfib, nlevel, v.data(), qv,
                         coords_x, coords_y, coords_z, s, norm, tol);

  std::vector<unsigned char> out_data;

  mgard::compress_memory_z(qv.data(), sizeof(int) * qv.size(), out_data);

  outsize = out_data.size();
  unsigned char *buffer = (unsigned char *)malloc(outsize);
  std::copy(out_data.begin(), out_data.end(), buffer);
  return buffer;
}

unsigned char *refactor_qz(int nrow, int ncol, int nfib, const float *u,
                           int &outsize, float tol,
                           float (*qoi)(int, int, int, std::vector<float>),
                           float s) {
  int nlevel;
  std::vector<float> v(u, u + nrow * ncol * nfib), work(nrow * ncol * nfib),
      work2d(nrow * ncol); // duplicate data and create work array
  std::vector<float> coords_x(ncol), coords_y(nrow),
      coords_z(nfib); // coordinate arrays

  int nlevel_x = std::log2(ncol - 1);
  int nc = std::pow(2, nlevel_x) + 1; // ncol new

  int nlevel_y = std::log2(nrow - 1);
  int nr = std::pow(2, nlevel_y) + 1; // nrow new

  int nlevel_z = std::log2(nfib - 1);
  int nf = std::pow(2, nlevel_z) + 1; // nfib new

  nlevel = std::min(nlevel_x, nlevel_y);
  nlevel = std::min(nlevel, nlevel_z);

  int l_target = nlevel - 1;

  // dummy equispaced coordinates
  std::iota(std::begin(coords_x), std::end(coords_x), 0);
  std::iota(std::begin(coords_y), std::end(coords_y), 0);
  std::iota(std::begin(coords_z), std::end(coords_z), 0);

  //  float norm =  mgard_gen::ml2_norm3(0,  nrow,  ncol,  nfib ,  nrow,  ncol,
  //  nfib,   v, coords_x, coords_y, coords_z);

  // float norm = mgard_common::max_norm(v);

  //  float norm = 1.0; // absolute s-norm, need a switch for relative errors
  //  tol /= nlevel + 1 ;
  //  float s = 0; // Defaulting to L8' compression for a start.

  //  norm = std::sqrt(norm/(nrow*nfib*ncol)); <- quant scaling goes here for s
  //  != 8'

  float norm = mgard_gen::ml2_norm3(0, nrow, ncol, nfib, nrow, ncol, nfib, v,
                                    coords_x, coords_y, coords_z);

  norm = std::sqrt(norm) / std::sqrt(nrow * ncol * nfib);

  mgard_gen::prep_3D(nr, nc, nf, nrow, ncol, nfib, l_target, v.data(), work,
                     work2d, coords_x, coords_y, coords_z);

  mgard_gen::refactor_3D(nr, nc, nf, nrow, ncol, nfib, l_target, v.data(), work,
                         work2d, coords_x, coords_y, coords_z);

  work.clear();
  work2d.clear();

  int size_ratio = sizeof(float) / sizeof(int);
  std::vector<int> qv(nrow * ncol * nfib + size_ratio);

  mgard_gen::quantize_3D(nr, nc, nf, nrow, ncol, nfib, nlevel, v.data(), qv,
                         coords_x, coords_y, coords_z, s, norm, tol);

  std::vector<unsigned char> out_data;

  mgard::compress_memory_z(qv.data(), sizeof(int) * qv.size(), out_data);

  outsize = out_data.size();
  unsigned char *buffer = (unsigned char *)malloc(outsize);
  std::copy(out_data.begin(), out_data.end(), buffer);
  return buffer;
}

float *recompose_udq(float dummyf, int nrow, int ncol, int nfib,
                     unsigned char *data, int data_len) {
  int nlevel;
  int size_ratio = sizeof(float) / sizeof(int);
  std::vector<float> coords_x(ncol), coords_y(nrow),
      coords_z(nfib); // coordinate arrays
  std::vector<int> out_data(nrow * ncol * nfib + size_ratio);
  std::vector<float> work(nrow * ncol * nfib),
      work2d(nrow * ncol); // duplicate data and create work array

  //      dummy equispaced coordinates
  std::iota(std::begin(coords_x), std::end(coords_x), 0);
  std::iota(std::begin(coords_y), std::end(coords_y), 0);
  std::iota(std::begin(coords_z), std::end(coords_z), 0);

  //    //std::cout  <<"**** coord check : "  << coords_x[4] << "\n";

  int nlevel_x = std::log2(ncol - 1);
  int nc = std::pow(2, nlevel_x) + 1; // ncol new

  int nlevel_y = std::log2(nrow - 1);
  int nr = std::pow(2, nlevel_y) + 1; // nrow new

  int nlevel_z = std::log2(nfib - 1);
  int nf = std::pow(2, nlevel_z) + 1; // nfib new

  nlevel = std::min(nlevel_x, nlevel_y);
  nlevel = std::min(nlevel, nlevel_z);

  int l_target = nlevel - 1;

  mgard::decompress_memory_z(data, data_len, out_data.data(),
                             out_data.size() *
                                 sizeof(int)); // decompress input buffer
  float *v = (float *)malloc(nrow * ncol * nfib * sizeof(float));

  mgard::dequantize_2D_interleave(nrow, ncol * nfib, v, out_data);

  mgard_gen::recompose_3D(nr, nc, nf, nrow, ncol, nfib, l_target, v, work,
                          work2d, coords_x, coords_y, coords_z);

  mgard_gen::postp_3D(nr, nc, nf, nrow, ncol, nfib, l_target, v, work, coords_x,
                      coords_y, coords_z);

  return v;
}

float *recompose_udq(int nrow, int ncol, int nfib, std::vector<float> &coords_x,
                     std::vector<float> &coords_y, std::vector<float> &coords_z,
                     unsigned char *data, int data_len) {
  int nlevel;
  int size_ratio = sizeof(float) / sizeof(int);
  std::vector<int> out_data(nrow * ncol * nfib + size_ratio);
  std::vector<float> work(nrow * ncol * nfib),
      work2d(nrow * ncol); // duplicate data and create work array

  int nlevel_x = std::log2(ncol - 1);
  int nc = std::pow(2, nlevel_x) + 1; // ncol new

  int nlevel_y = std::log2(nrow - 1);
  int nr = std::pow(2, nlevel_y) + 1; // nrow new

  int nlevel_z = std::log2(nfib - 1);
  int nf = std::pow(2, nlevel_z) + 1; // nfib new

  nlevel = std::min(nlevel_x, nlevel_y);
  nlevel = std::min(nlevel, nlevel_z);

  int l_target = nlevel - 1;

  mgard::decompress_memory_z(data, data_len, out_data.data(),
                             out_data.size() *
                                 sizeof(int)); // decompress input buffer
  float *v = (float *)malloc(nrow * ncol * nfib * sizeof(float));

  mgard::dequantize_2D_interleave(nrow, ncol * nfib, v, out_data);

  mgard_gen::recompose_3D(nr, nc, nf, nrow, ncol, nfib, l_target, v, work,
                          work2d, coords_x, coords_y, coords_z);

  mgard_gen::postp_3D(nr, nc, nf, nrow, ncol, nfib, l_target, v, work, coords_x,
                      coords_y, coords_z);

  return v;
}

float *recompose_udq(int nrow, int ncol, int nfib, unsigned char *data,
                     int data_len, float s) {
  int nlevel;
  int size_ratio = sizeof(float) / sizeof(int);
  std::vector<float> coords_x(ncol), coords_y(nrow),
      coords_z(nfib); // coordinate arrays
  std::vector<int> out_data(nrow * ncol * nfib + size_ratio);
  std::vector<float> work(nrow * ncol * nfib),
      work2d(nrow * ncol); // duplicate data and create work array

  //    float s = 0; // Defaulting to L2 compression for a start.
  float norm = 1; // defaulting to absolute s, may switch to relative

  //      dummy equispaced coordinates
  std::iota(std::begin(coords_x), std::end(coords_x), 0);
  std::iota(std::begin(coords_y), std::end(coords_y), 0);
  std::iota(std::begin(coords_z), std::end(coords_z), 0);

  //    //std::cout  <<"**** coord check : "  << coords_x[4] << "\n";

  int nlevel_x = std::log2(ncol - 1);
  int nc = std::pow(2, nlevel_x) + 1; // ncol new

  int nlevel_y = std::log2(nrow - 1);
  int nr = std::pow(2, nlevel_y) + 1; // nrow new

  int nlevel_z = std::log2(nfib - 1);
  int nf = std::pow(2, nlevel_z) + 1; // nfib new

  nlevel = std::min(nlevel_x, nlevel_y);
  nlevel = std::min(nlevel, nlevel_z);

  int l_target = nlevel - 1;

  mgard::decompress_memory_z(data, data_len, out_data.data(),
                             out_data.size() *
                                 sizeof(int)); // decompress input buffer
  float *v = (float *)malloc(nrow * ncol * nfib * sizeof(float));

  //    outfile.write( reinterpret_cast<char*>( out_data.data() ),
  //    (nrow*ncol*nfib + size_ratio)*sizeof(int) );

  mgard_gen::dequantize_3D(nr, nc, nf, nrow, ncol, nfib, nlevel, v, out_data,
                           coords_x, coords_y, coords_z, s);
  //    mgard::dequantize_2D_interleave(nrow, ncol*nfib, v, out_data) ;

  //    mgard_common::qread_2D_interleave(nrow,  ncol, nlevel, work.data(),
  //    out_file);

  // mgard_gen::dequant_3D(  nr,  nc,  nf,  nrow,  ncol,  nfib,  nlevel, nlevel,
  // v, work.data(), coords_x, coords_y,  coords_z, s );

  // std::ofstream outfile(out_file, std::ios::out | std::ios::binary);

  //    w
  mgard_gen::recompose_3D(nr, nc, nf, nrow, ncol, nfib, l_target, v, work,
                          work2d, coords_x, coords_y, coords_z);

  mgard_gen::postp_3D(nr, nc, nf, nrow, ncol, nfib, l_target, v, work, coords_x,
                      coords_y, coords_z);

  //    outfile.write( reinterpret_cast<char*>( v ),
  //    nrow*ncol*nfib*sizeof(float) ;)
  return v;
}
//}

float *recompose_udq(int nrow, int ncol, int nfib, std::vector<float> &coords_x,
                     std::vector<float> &coords_y, std::vector<float> &coords_z,
                     unsigned char *data, int data_len, float s) {
  int nlevel;
  int size_ratio = sizeof(float) / sizeof(int);

  std::vector<int> out_data(nrow * ncol * nfib + size_ratio);
  std::vector<float> work(nrow * ncol * nfib),
      work2d(nrow * ncol); // duplicate data and create work array

  //    float s = 0; // Defaulting to L2 compression for a start.
  float norm = 1; // defaulting to absolute s, may switch to relative

  int nlevel_x = std::log2(ncol - 1);
  int nc = std::pow(2, nlevel_x) + 1; // ncol new

  int nlevel_y = std::log2(nrow - 1);
  int nr = std::pow(2, nlevel_y) + 1; // nrow new

  int nlevel_z = std::log2(nfib - 1);
  int nf = std::pow(2, nlevel_z) + 1; // nfib new

  nlevel = std::min(nlevel_x, nlevel_y);
  nlevel = std::min(nlevel, nlevel_z);

  int l_target = nlevel - 1;

  mgard::decompress_memory_z(data, data_len, out_data.data(),
                             out_data.size() *
                                 sizeof(int)); // decompress input buffer
  float *v = (float *)malloc(nrow * ncol * nfib * sizeof(float));

  //    outfile.write( reinterpret_cast<char*>( out_data.data() ),
  //    (nrow*ncol*nfib + size_ratio)*sizeof(int) );

  mgard_gen::dequantize_3D(nr, nc, nf, nrow, ncol, nfib, nlevel, v, out_data,
                           coords_x, coords_y, coords_z, s);
  //    mgard::dequantize_2D_interleave(nrow, ncol*nfib, v, out_data) ;

  //    mgard_common::qread_2D_interleave(nrow,  ncol, nlevel, work.data(),
  //    out_file);

  // mgard_gen::dequant_3D(  nr,  nc,  nf,  nrow,  ncol,  nfib,  nlevel, nlevel,
  // v, work.data(), coords_x, coords_y,  coords_z, s );

  // std::ofstream outfile(out_file, std::ios::out | std::ios::binary);

  //    w
  mgard_gen::recompose_3D(nr, nc, nf, nrow, ncol, nfib, l_target, v, work,
                          work2d, coords_x, coords_y, coords_z);

  mgard_gen::postp_3D(nr, nc, nf, nrow, ncol, nfib, l_target, v, work, coords_x,
                      coords_y, coords_z);

  //    outfile.write( reinterpret_cast<char*>( v ),
  //    nrow*ncol*nfib*sizeof(float) ;)
  return v;
}

unsigned char *refactor_qz_2D(int nrow, int ncol, const float *u, int &outsize,
                              float tol) {

  std::vector<float> row_vec(ncol);
  std::vector<float> col_vec(nrow);
  std::vector<float> v(u, u + nrow * ncol), work(nrow * ncol);

  float norm = mgard_2d::mgard_common::max_norm(v);

  if (mgard::is_2kplus1(nrow) &&
      mgard::is_2kplus1(ncol)) // input is (2^q + 1) x (2^p + 1)
  {
    int nlevel;
    mgard::set_number_of_levels(nrow, ncol, nlevel);
    tol /= float(nlevel + 1);

    int l_target = nlevel - 1;
    mgard::refactor(nrow, ncol, l_target, v.data(), work, row_vec, col_vec);
    work.clear();
    row_vec.clear();
    col_vec.clear();

    int size_ratio = sizeof(float) / sizeof(int);
    std::vector<int> qv(nrow * ncol + size_ratio);

    mgard::quantize_2D_interleave(nrow, ncol, v.data(), qv, norm, tol);

    std::vector<unsigned char> out_data;

    mgard::compress_memory_z(qv.data(), sizeof(int) * qv.size(), out_data);
    outsize = out_data.size();
    unsigned char *buffer = (unsigned char *)malloc(outsize);
    std::copy(out_data.begin(), out_data.end(), buffer);
    return buffer;
  } else {

    std::vector<float> coords_x(ncol), coords_y(nrow);

    std::iota(std::begin(coords_x), std::end(coords_x), 0);
    std::iota(std::begin(coords_y), std::end(coords_y), 0);

    int nlevel_x = std::log2(ncol - 1);
    int nc = std::pow(2, nlevel_x) + 1; // ncol new

    int nlevel_y = std::log2(nrow - 1);
    int nr = std::pow(2, nlevel_y) + 1; // nrow new

    int nlevel = std::min(nlevel_x, nlevel_y);
    tol /= nlevel + 1;

    int l_target = nlevel - 1;
    l_target = 0;
    mgard_2d::mgard_gen::prep_2D(nr, nc, nrow, ncol, l_target, v.data(), work,
                                 coords_x, coords_y, row_vec, col_vec);

    mgard_2d::mgard_gen::refactor_2D(nr, nc, nrow, ncol, l_target, v.data(),
                                     work, coords_x, coords_y, row_vec,
                                     col_vec);

    work.clear();
    col_vec.clear();
    row_vec.clear();

    int size_ratio = sizeof(float) / sizeof(int);
    std::vector<int> qv(nrow * ncol + size_ratio);

    tol /= nlevel + 1;
    mgard::quantize_2D_interleave(nrow, ncol, v.data(), qv, norm, tol);

    std::vector<unsigned char> out_data;

    mgard::compress_memory_z(qv.data(), sizeof(int) * qv.size(), out_data);

    outsize = out_data.size();
    unsigned char *buffer = (unsigned char *)malloc(outsize);
    std::copy(out_data.begin(), out_data.end(), buffer);
    return buffer;
  }
}

unsigned char *refactor_qz_2D(int nrow, int ncol, std::vector<float> &coords_x,
                              std::vector<float> &coords_y, const float *u,
                              int &outsize, float tol) {

  std::vector<float> row_vec(ncol);
  std::vector<float> col_vec(nrow);
  std::vector<float> v(u, u + nrow * ncol), work(nrow * ncol);

  float norm = mgard_2d::mgard_common::max_norm(v);

  int nlevel_x = std::log2(ncol - 1);
  int nc = std::pow(2, nlevel_x) + 1; // ncol new

  int nlevel_y = std::log2(nrow - 1);
  int nr = std::pow(2, nlevel_y) + 1; // nrow new

  int nlevel = std::min(nlevel_x, nlevel_y);
  tol /= nlevel + 2;

  int l_target = nlevel - 1;
  l_target = 0;
  mgard_2d::mgard_gen::prep_2D(nr, nc, nrow, ncol, l_target, v.data(), work,
                               coords_x, coords_y, row_vec, col_vec);

  mgard_2d::mgard_gen::refactor_2D(nr, nc, nrow, ncol, l_target, v.data(), work,
                                   coords_x, coords_y, row_vec, col_vec);

  work.clear();
  col_vec.clear();
  row_vec.clear();

  int size_ratio = sizeof(float) / sizeof(int);
  std::vector<int> qv(nrow * ncol + size_ratio);

  tol /= nlevel + 1;
  mgard::quantize_2D_interleave(nrow, ncol, v.data(), qv, norm, tol);

  std::vector<unsigned char> out_data;

  mgard::compress_memory_z(qv.data(), sizeof(int) * qv.size(), out_data);

  outsize = out_data.size();
  unsigned char *buffer = (unsigned char *)malloc(outsize);
  std::copy(out_data.begin(), out_data.end(), buffer);
  return buffer;
}

unsigned char *refactor_qz_2D(int nrow, int ncol, const float *u, int &outsize,
                              float tol, float s) {

  std::vector<float> row_vec(ncol);
  std::vector<float> col_vec(nrow);
  std::vector<float> v(u, u + nrow * ncol), work(nrow * ncol);

  float norm = mgard_2d::mgard_common::max_norm(v);

  if (mgard::is_2kplus1(nrow) &&
      mgard::is_2kplus1(ncol)) // input is (2^q + 1) x (2^p + 1)
  {
    int nlevel;
    mgard::set_number_of_levels(nrow, ncol, nlevel);
    tol /= float(nlevel + 1);

    int l_target = nlevel - 1;
    mgard::refactor(nrow, ncol, l_target, v.data(), work, row_vec, col_vec);
    work.clear();
    row_vec.clear();
    col_vec.clear();

    std::vector<float> coords_x(ncol), coords_y(nrow);

    std::iota(std::begin(coords_x), std::end(coords_x), 0);
    std::iota(std::begin(coords_y), std::end(coords_y), 0);

    int size_ratio = sizeof(float) / sizeof(int);
    std::vector<int> qv(nrow * ncol + size_ratio);

    // mgard::quantize_2D_interleave (nrow, ncol, v.data(), qv, norm, tol);

    mgard_gen::quantize_2D(nrow, ncol, nrow, ncol, nlevel, v.data(), qv,
                           coords_x, coords_y, s, norm, tol);

    std::vector<unsigned char> out_data;

    mgard::compress_memory_z(qv.data(), sizeof(int) * qv.size(), out_data);
    outsize = out_data.size();
    unsigned char *buffer = (unsigned char *)malloc(outsize);
    std::copy(out_data.begin(), out_data.end(), buffer);
    return buffer;
  } else {

    std::vector<float> coords_x(ncol), coords_y(nrow);

    std::iota(std::begin(coords_x), std::end(coords_x), 0);
    std::iota(std::begin(coords_y), std::end(coords_y), 0);

    int nlevel_x = std::log2(ncol - 1);
    int nc = std::pow(2, nlevel_x) + 1; // ncol new

    int nlevel_y = std::log2(nrow - 1);
    int nr = std::pow(2, nlevel_y) + 1; // nrow new

    int nlevel = std::min(nlevel_x, nlevel_y);
    tol /= nlevel + 1;

    int l_target = nlevel - 1;
    l_target = 0;
    mgard_2d::mgard_gen::prep_2D(nr, nc, nrow, ncol, l_target, v.data(), work,
                                 coords_x, coords_y, row_vec, col_vec);

    mgard_2d::mgard_gen::refactor_2D(nr, nc, nrow, ncol, l_target, v.data(),
                                     work, coords_x, coords_y, row_vec,
                                     col_vec);

    work.clear();
    col_vec.clear();
    row_vec.clear();

    int size_ratio = sizeof(float) / sizeof(int);
    std::vector<int> qv(nrow * ncol + size_ratio);

    mgard_gen::quantize_2D(nr, nc, nrow, ncol, nlevel, v.data(), qv, coords_x,
                           coords_y, s, norm, tol);

    std::vector<unsigned char> out_data;

    mgard::compress_memory_z(qv.data(), sizeof(int) * qv.size(), out_data);

    outsize = out_data.size();
    unsigned char *buffer = (unsigned char *)malloc(outsize);
    std::copy(out_data.begin(), out_data.end(), buffer);
    return buffer;
  }
}

unsigned char *refactor_qz_2D(int nrow, int ncol, std::vector<float> &coords_x,
                              std::vector<float> &coords_y, const float *u,
                              int &outsize, float tol, float s) {

  std::vector<float> row_vec(ncol);
  std::vector<float> col_vec(nrow);
  std::vector<float> v(u, u + nrow * ncol), work(nrow * ncol);

  float norm = mgard_2d::mgard_common::max_norm(v);

  int nlevel_x = std::log2(ncol - 1);
  int nc = std::pow(2, nlevel_x) + 1; // ncol new

  int nlevel_y = std::log2(nrow - 1);
  int nr = std::pow(2, nlevel_y) + 1; // nrow new

  int nlevel = std::min(nlevel_x, nlevel_y);
  tol /= nlevel + 1;

  int l_target = nlevel - 1;
  l_target = 0;
  mgard_2d::mgard_gen::prep_2D(nr, nc, nrow, ncol, l_target, v.data(), work,
                               coords_x, coords_y, row_vec, col_vec);

  mgard_2d::mgard_gen::refactor_2D(nr, nc, nrow, ncol, l_target, v.data(), work,
                                   coords_x, coords_y, row_vec, col_vec);

  work.clear();
  col_vec.clear();
  row_vec.clear();

  int size_ratio = sizeof(float) / sizeof(int);
  std::vector<int> qv(nrow * ncol + size_ratio);

  mgard_gen::quantize_2D(nr, nc, nrow, ncol, nlevel, v.data(), qv, coords_x,
                         coords_y, s, norm, tol);

  std::vector<unsigned char> out_data;

  mgard::compress_memory_z(qv.data(), sizeof(int) * qv.size(), out_data);

  outsize = out_data.size();
  unsigned char *buffer = (unsigned char *)malloc(outsize);
  std::copy(out_data.begin(), out_data.end(), buffer);
  return buffer;
}

float *recompose_udq_2D(float dummyf, int nrow, int ncol, unsigned char *data,
                        int data_len) {
  int size_ratio = sizeof(float) / sizeof(int);

  if (mgard::is_2kplus1(nrow) &&
      mgard::is_2kplus1(ncol)) // input is (2^q + 1) x (2^p + 1)
  {
    int ncol_new = ncol;
    int nrow_new = nrow;

    int nlevel_new;
    mgard::set_number_of_levels(nrow_new, ncol_new, nlevel_new);
    int l_target = nlevel_new - 1;

    std::vector<int> out_data(nrow_new * ncol_new + size_ratio);

    mgard::decompress_memory_z(data, data_len, out_data.data(),
                               out_data.size() *
                                   sizeof(int)); // decompress input buffer

    float *v = (float *)malloc(nrow_new * ncol_new * sizeof(float));

    mgard::dequantize_2D_interleave(nrow_new, ncol_new, v, out_data);
    out_data.clear();

    std::vector<float> row_vec(ncol_new);
    std::vector<float> col_vec(nrow_new);
    std::vector<float> work(nrow_new * ncol_new);

    mgard::recompose(nrow_new, ncol_new, l_target, v, work, row_vec, col_vec);

    return v;

  } else {
    std::vector<float> coords_x(ncol), coords_y(nrow);

    std::iota(std::begin(coords_x), std::end(coords_x), 0);
    std::iota(std::begin(coords_y), std::end(coords_y), 0);

    int nlevel_x = std::log2(ncol - 1);
    int nc = std::pow(2, nlevel_x) + 1; // ncol new

    int nlevel_y = std::log2(nrow - 1);
    int nr = std::pow(2, nlevel_y) + 1; // nrow new

    int nlevel = std::min(nlevel_x, nlevel_y);

    //      int l_target = nlevel-1;

    int l_target = 0;
    std::vector<int> out_data(nrow * ncol + size_ratio);

    mgard::decompress_memory_z(data, data_len, out_data.data(),
                               out_data.size() *
                                   sizeof(int)); // decompress input buffer

    float *v = (float *)malloc(nrow * ncol * sizeof(float));

    mgard::dequantize_2D_interleave(nrow, ncol, v, out_data);

    std::vector<float> row_vec(ncol);
    std::vector<float> col_vec(nrow);
    std::vector<float> work(nrow * ncol);

    mgard_2d::mgard_gen::recompose_2D(nr, nc, nrow, ncol, l_target, v, work,
                                      coords_x, coords_y, row_vec, col_vec);

    mgard_2d::mgard_gen::postp_2D(nr, nc, nrow, ncol, l_target, v, work,
                                  coords_x, coords_y, row_vec, col_vec);

    return v;
  }
}

float *recompose_udq_2D(int nrow, int ncol, std::vector<float> &coords_x,
                        std::vector<float> &coords_y, unsigned char *data,
                        int data_len) {
  int size_ratio = sizeof(float) / sizeof(int);

  int nlevel_x = std::log2(ncol - 1);
  int nc = std::pow(2, nlevel_x) + 1; // ncol new

  int nlevel_y = std::log2(nrow - 1);
  int nr = std::pow(2, nlevel_y) + 1; // nrow new

  int nlevel = std::min(nlevel_x, nlevel_y);

  //      int l_target = nlevel-1;

  int l_target = 0;
  std::vector<int> out_data(nrow * ncol + size_ratio);

  mgard::decompress_memory_z(data, data_len, out_data.data(),
                             out_data.size() *
                                 sizeof(int)); // decompress input buffer

  float *v = (float *)malloc(nrow * ncol * sizeof(float));

  mgard::dequantize_2D_interleave(nrow, ncol, v, out_data);

  std::vector<float> row_vec(ncol);
  std::vector<float> col_vec(nrow);
  std::vector<float> work(nrow * ncol);

  mgard_2d::mgard_gen::recompose_2D(nr, nc, nrow, ncol, l_target, v, work,
                                    coords_x, coords_y, row_vec, col_vec);

  mgard_2d::mgard_gen::postp_2D(nr, nc, nrow, ncol, l_target, v, work, coords_x,
                                coords_y, row_vec, col_vec);

  return v;
}

float *recompose_udq_2D(int nrow, int ncol, unsigned char *data, int data_len,
                        float s) {
  int size_ratio = sizeof(float) / sizeof(int);

  if (mgard::is_2kplus1(nrow) &&
      mgard::is_2kplus1(ncol)) // input is (2^q + 1) x (2^p + 1)
  {
    int ncol_new = ncol;
    int nrow_new = nrow;

    int nlevel_new;
    mgard::set_number_of_levels(nrow_new, ncol_new, nlevel_new);
    int l_target = nlevel_new - 1;

    std::vector<float> coords_x(ncol), coords_y(nrow);

    std::iota(std::begin(coords_x), std::end(coords_x), 0);
    std::iota(std::begin(coords_y), std::end(coords_y), 0);

    std::vector<int> out_data(nrow_new * ncol_new + size_ratio);

    mgard::decompress_memory_z(data, data_len, out_data.data(),
                               out_data.size() *
                                   sizeof(int)); // decompress input buffer

    float *v = (float *)malloc(nrow_new * ncol_new * sizeof(float));

    //      mgard::dequantize_2D_interleave(nrow_new, ncol_new, v, out_data) ;
    mgard_gen::dequantize_2D(nrow, ncol, nrow, ncol, nlevel_new, v, out_data,
                             coords_x, coords_y, s);
    out_data.clear();

    std::vector<float> row_vec(ncol_new);
    std::vector<float> col_vec(nrow_new);
    std::vector<float> work(nrow_new * ncol_new);

    mgard::recompose(nrow_new, ncol_new, l_target, v, work, row_vec, col_vec);

    return v;

  } else {
    std::vector<float> coords_x(ncol), coords_y(nrow);

    std::iota(std::begin(coords_x), std::end(coords_x), 0);
    std::iota(std::begin(coords_y), std::end(coords_y), 0);

    int nlevel_x = std::log2(ncol - 1);
    int nc = std::pow(2, nlevel_x) + 1; // ncol new

    int nlevel_y = std::log2(nrow - 1);
    int nr = std::pow(2, nlevel_y) + 1; // nrow new

    int nlevel = std::min(nlevel_x, nlevel_y);

    //      int l_target = nlevel-1;

    int l_target = 0;
    std::vector<int> out_data(nrow * ncol + size_ratio);

    mgard::decompress_memory_z(data, data_len, out_data.data(),
                               out_data.size() * sizeof(int));

    float *v = (float *)malloc(nrow * ncol * sizeof(float));

    mgard_gen::dequantize_2D(nr, nc, nrow, ncol, nlevel, v, out_data, coords_x,
                             coords_y, s);

    std::vector<float> row_vec(ncol);
    std::vector<float> col_vec(nrow);
    std::vector<float> work(nrow * ncol);

    mgard_2d::mgard_gen::recompose_2D(nr, nc, nrow, ncol, l_target, v, work,
                                      coords_x, coords_y, row_vec, col_vec);

    mgard_2d::mgard_gen::postp_2D(nr, nc, nrow, ncol, l_target, v, work,
                                  coords_x, coords_y, row_vec, col_vec);

    return v;
  }
}

float *recompose_udq_2D(int nrow, int ncol, std::vector<float> &coords_x,
                        std::vector<float> &coords_y, unsigned char *data,
                        int data_len, float s) {
  int size_ratio = sizeof(float) / sizeof(int);

  int nlevel_x = std::log2(ncol - 1);
  int nc = std::pow(2, nlevel_x) + 1; // ncol new

  int nlevel_y = std::log2(nrow - 1);
  int nr = std::pow(2, nlevel_y) + 1; // nrow new

  int nlevel = std::min(nlevel_x, nlevel_y);

  //      int l_target = nlevel-1;

  int l_target = 0;
  std::vector<int> out_data(nrow * ncol + size_ratio);

  mgard::decompress_memory_z(data, data_len, out_data.data(),
                             out_data.size() * sizeof(int));

  float *v = (float *)malloc(nrow * ncol * sizeof(float));

  mgard_gen::dequantize_2D(nr, nc, nrow, ncol, nlevel, v, out_data, coords_x,
                           coords_y, s);

  std::vector<float> row_vec(ncol);
  std::vector<float> col_vec(nrow);
  std::vector<float> work(nrow * ncol);

  mgard_2d::mgard_gen::recompose_2D(nr, nc, nrow, ncol, l_target, v, work,
                                    coords_x, coords_y, row_vec, col_vec);

  mgard_2d::mgard_gen::postp_2D(nr, nc, nrow, ncol, l_target, v, work, coords_x,
                                coords_y, row_vec, col_vec);

  return v;
}

//   unsigned char *
//   refactor_qz_1D (int nrow,  const float *u, int &outsize, float tol)
//   {

//   std::vector<float> v(u, u+nrow), work(nrow);

//   float norm = mgard_common::max_norm(v);

//   std::vector<float> coords_x;
//   int nlevel = std::log2(nrow-1);
//   int nr = std::pow(2, nlevel ) + 1; //ncol new
//   int l_target = nlevel-1;

//   std::iota(std::begin(coords_x), std::end(coords_x), 0);

//   //  mgard::refactor_1D(nlevel-1, v,   work, coords_x,  nr,  nrow );

//   work.clear ();

//   int size_ratio = sizeof (float) / sizeof (int);
//   std::vector<int> qv (nrow * ncol + size_ratio);

//   tol /= nlevel + 1;
//   mgard::quantize_2D_interleave (nrow, 1, v.data (), qv, norm, tol);

//   std::vector<unsigned char> out_data;
//   mgard::compress_memory (qv.data (), sizeof (int) * qv.size (), out_data);

//   outsize = out_data.size ();
//   unsigned char *buffer = (unsigned char *)malloc (outsize);
//   std::copy (out_data.begin (), out_data.end (), buffer);
//   return buffer;
// }

//  float* recompose_udq_1D(int nrow,  unsigned char *data, int data_len)
//   {
//   int size_ratio = sizeof(float)/sizeof(int);
//     {
//       std::vector<float> coords_x;

//       int nlevel = std::log2(nrow-1);
//       int nc = std::pow(2, nlevel ) + 1; //ncol new
// k
//       int l_target = nlevel-1;

//       std::vector<int> out_data(nrow + size_ratio);

//       std::iota(std::begin(coords_x), std::end(coords_x), 0);

//       mgard::decompress_memory(data, data_len, out_data.data(),
//       out_data.size()*sizeof(int)); // decompress input buffer

//       float *v = (float *)malloc (nrow*sizeof(float));

//       mgard::dequantize_2D_interleave(nrow, 1, v, out_data) ;

//       std::vector<float> work(nrow);

//       mgard::recompose_1D(nlevel, v,   work, coords_x,  nr,  nrow );

//       return v;
//     }
// }

inline int get_index(const int ncol, const int i, const int j) {
  return ncol * i + j;
}

bool is_2kplus1(float num) {
  float frac_part, f_level, int_part;
  if (num == 1)
    return 1;

  f_level = std::log2(num - 1);
  frac_part = modff(f_level, &int_part);

  if (frac_part == 0) {
    return 1;
  } else {
    return 0;
  }
}

int parse_cmdl(int argc, char **argv, int &nrow, int &ncol, float &tol,
               std::string &in_file) {
  if (argc >= 5) {
    in_file = argv[1];
    nrow = strtol((argv[2]), NULL, 0); // number of rows
    ncol = strtol((argv[3]), NULL, 0); // number of columns
    tol = strtod((argv[4]), 0);        // error tolerance

    assert(in_file.size() != 0);
    assert(ncol > 3);
    assert(nrow >= 1);
    assert(tol >= 1e-8);

    struct stat file_stats;
    int flag = stat(in_file.c_str(), &file_stats);

    if (flag != 0) // can't stat file somehow
    {
      throw std::runtime_error(
          "Cannot stat input file! Nothing to be done, exiting...");
    }

    return 1;
  } else {
    std::cerr << "Usage: " << argv[0] << " inputfile nrow ncol tol"
              << "\n";
    throw std::runtime_error("Too few arguments, exiting...");
  }
}

void mass_matrix_multiply(const int l, std::vector<float> &v) {

  int stride = std::pow(2, l);
  float temp1, temp2;
  float fac = 0.5;
  // Mass matrix times nodal value-vec
  temp1 = v.front(); // save u(0) for later use
  v.front() = fac * (2.0 * temp1 + v.at(stride));
  for (auto it = v.begin() + stride; it < v.end() - stride; it += stride) {
    temp2 = *it;
    *it = fac * (temp1 + 4 * temp2 + *(it + stride));
    temp1 = temp2; // save u(n) for later use
  }
  v.back() = fac * (2 * v.back() + temp1);
}

void solve_tridiag_M(const int l, std::vector<float> &v) {

  //  int my_level = nlevel - l;
  int stride = std::pow(2, l); // current stride

  float am, bm;

  am = 2.0; // first element of upper diagonal U.

  bm = 1.0 / am;

  int nlevel = static_cast<int>(std::log2(v.size() - 1));
  int n = std::pow(2, nlevel - l) + 1;
  std::vector<float> coeff(n);
  int counter = 1;
  coeff.front() = am;

  // forward sweep
  for (auto it = std::begin(v) + stride; it < std::end(v) - stride;
       it += stride) {
    *(it) -= *(it - stride) / am;

    am = 4.0 - bm;
    bm = 1.0 / am;

    coeff.at(counter) = am;
    ++counter;
  }
  am = 2.0 - bm; // a_n = 2 - b_(n-1)

  auto it = v.end() - stride - 1;
  v.back() -= (*it) * bm; // last element

  coeff.at(counter) = am;

  // backward sweep

  v.back() /= am;
  --counter;

  for (auto it = v.rbegin() + stride; it <= v.rend(); it += stride) {

    *(it) = (*(it) - *(it - stride)) / coeff.at(counter);
    --counter;
    bm = 4.0 - am; // maybe assign 1/am -> bm?
    am = 1.0 / bm;
  }
}

void restriction(const int l, std::vector<float> &v) {
  int stride = std::pow(2, l);
  int Pstride = stride / 2;

  // calculate the result of restrictionion
  auto it = v.begin() + Pstride;
  v.front() += 0.5 * (*it); // first element
  for (auto it = std::begin(v) + stride; it <= std::end(v) - stride;
       it += stride) {
    *(it) += 0.5 * (*(it - Pstride) + *(it + Pstride));
  }
  it = v.end() - Pstride - 1;
  v.back() += 0.5 * (*it); // last element
}

void interpolate_from_level_nMl(const int l, std::vector<float> &v) {

  int stride = std::pow(2, l);
  int Pstride = stride / 2;

  for (auto it = std::begin(v) + stride; it < std::end(v); it += stride) {
    *(it - Pstride) = 0.5 * (*(it - stride) + *it);
  }
}

void print_level_2D(const int nrow, const int ncol, const int l, float *v) {

  int stride = std::pow(2, l);

  for (int irow = 0; irow < nrow; irow += stride) {
    // std::cout  << "\n";
    for (int jcol = 0; jcol < ncol; jcol += stride) {
      // std::cout  << v[get_index (ncol, irow, jcol)] << "\t";
    }
    // std::cout  << "\n";
  }
}

void write_level_2D(const int nrow, const int ncol, const int l, float *v,
                    std::ofstream &outfile) {
  int stride = std::pow(2, l);
  //  int nrow = std::pow(2, nlevel_row) + 1;
  // int ncol = std::pow(2, nlevel_col) + 1;

  for (int irow = 0; irow < nrow; irow += stride) {
    for (int jcol = 0; jcol < ncol; jcol += stride) {
      outfile.write(reinterpret_cast<char *>(&v[get_index(ncol, irow, jcol)]),
                    sizeof(float));
    }
  }
}

void write_level_2D_exc(const int nrow, const int ncol, const int l, float *v,
                        std::ofstream &outfile) {
  // Write P_l\P_{l-1}

  int stride = std::pow(2, l);
  int Cstride = stride * 2;

  int row_counter = 0;

  for (int irow = 0; irow < nrow; irow += stride) {
    if (row_counter % 2 == 0) {
      for (int jcol = Cstride; jcol < ncol; jcol += Cstride) {
        outfile.write(
            reinterpret_cast<char *>(&v[get_index(ncol, irow, jcol - stride)]),
            sizeof(float));
      }
    } else {
      for (int jcol = 0; jcol < ncol; jcol += stride) {
        outfile.write(reinterpret_cast<char *>(&v[get_index(ncol, irow, jcol)]),
                      sizeof(float));
      }
    }
    ++row_counter;
  }
}

void pi_lminus1(const int l, std::vector<float> &v0) {
  int nlevel = static_cast<int>(std::log2(v0.size() - 1));
  int my_level = nlevel - l;
  int stride = std::pow(2, l); // current stride
  //  int Pstride = stride/2; //finer stride
  int Cstride = stride * 2; // coarser stride

  if (my_level != 0) {
    for (auto it0 = v0.begin() + Cstride; it0 < v0.end(); it0 += Cstride) {
      *(it0 - stride) -= 0.5 * (*it0 + *(it0 - Cstride));
    }
  }
}

void pi_Ql(const int nrow, const int ncol, const int l, float *v,
           std::vector<float> &row_vec, std::vector<float> &col_vec) {
  // Restrict data to coarser level

  int stride = std::pow(2, l); // current stride
  //  int Pstride = stride/2; //finer stride
  int Cstride = stride * 2; // coarser stride

  //  std::vector<float> row_vec(ncol), col_vec(nrow)   ;

  for (int irow = 0; irow < nrow;
       irow += Cstride) // Do the rows existing  in the coarser level
  {
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = v[get_index(ncol, irow, jcol)];
    }

    pi_lminus1(l, row_vec);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      v[get_index(ncol, irow, jcol)] = row_vec[jcol];
    }
  }

  if (nrow > 1) {
    for (int jcol = 0; jcol < ncol;
         jcol += Cstride) // Do the columns existing  in the coarser level
    {
      for (int irow = 0; irow < nrow; ++irow) {
        col_vec[irow] = v[get_index(ncol, irow, jcol)];
      }

      pi_lminus1(l, col_vec);

      for (int irow = 0; irow < nrow; ++irow) {
        v[get_index(ncol, irow, jcol)] = col_vec[irow];
      }
    }

    // Now the new-new stuff

    for (int irow = Cstride; irow <= nrow - 1 - Cstride; irow += 2 * Cstride) {
      for (int jcol = Cstride; jcol <= ncol - 1 - Cstride;
           jcol += 2 * Cstride) {
        v[get_index(ncol, irow - stride, jcol - stride)] -=
            0.25 * (v[get_index(ncol, irow - Cstride, jcol - Cstride)] +
                    v[get_index(ncol, irow - Cstride, jcol)] +
                    v[get_index(ncol, irow, jcol)] +
                    v[get_index(ncol, irow, jcol - Cstride)]);

        v[get_index(ncol, irow - stride, jcol + stride)] -=
            0.25 * (v[get_index(ncol, irow - Cstride, jcol)] +
                    v[get_index(ncol, irow - Cstride, jcol + Cstride)] +
                    v[get_index(ncol, irow, jcol + Cstride)] +
                    v[get_index(ncol, irow, jcol)]);

        v[get_index(ncol, irow + stride, jcol + stride)] -=
            0.25 * (v[get_index(ncol, irow, jcol)] +
                    v[get_index(ncol, irow, jcol + Cstride)] +
                    v[get_index(ncol, irow + Cstride, jcol + Cstride)] +
                    v[get_index(ncol, irow + Cstride, jcol)]);

        v[get_index(ncol, irow + stride, jcol - stride)] -=
            0.25 * (v[get_index(ncol, irow, jcol - Cstride)] +
                    v[get_index(ncol, irow, jcol)] +
                    v[get_index(ncol, irow + Cstride, jcol)] +
                    v[get_index(ncol, irow + Cstride, jcol - Cstride)]);
      }
    }
  }
}

void assign_num_level(const int nrow, const int ncol, const int l, float *v,
                      float num) {
  // set the value of nodal values at level l to number num

  int stride = std::pow(2, l); // current stride

  for (int irow = 0; irow < nrow; irow += stride) {
    for (int jcol = 0; jcol < ncol; jcol += stride) {
      v[get_index(ncol, irow, jcol)] = num;
    }
  }
}

void copy_level(const int nrow, const int ncol, const int l, float *v,
                std::vector<float> &work) {

  int stride = std::pow(2, l); // current stride

  for (int irow = 0; irow < nrow; irow += stride) {
    for (int jcol = 0; jcol < ncol; jcol += stride) {
      work[get_index(ncol, irow, jcol)] = v[get_index(ncol, irow, jcol)];
    }
  }
}

void add_level(const int nrow, const int ncol, const int l, float *v,
               float *work) {
  // v += work at level l

  int stride = std::pow(2, l); // current stride

  for (int irow = 0; irow < nrow; irow += stride) {
    for (int jcol = 0; jcol < ncol; jcol += stride) {
      v[get_index(ncol, irow, jcol)] += work[get_index(ncol, irow, jcol)];
    }
  }
}

void subtract_level(const int nrow, const int ncol, const int l, float *v,
                    float *work) {
  // v += work at level l
  int stride = std::pow(2, l); // current stride

  for (int irow = 0; irow < nrow; irow += stride) {
    for (int jcol = 0; jcol < ncol; jcol += stride) {
      v[get_index(ncol, irow, jcol)] -= work[get_index(ncol, irow, jcol)];
    }
  }
}

void compute_correction_loadv(const int l, std::vector<float> &v) {
  int stride = std::pow(2, l); // current stride
  int Pstride = stride / 2;    // finer stride

  auto it = v.begin() + Pstride;
  v.front() += 0.25 * (*it); // first element
  for (auto it = std::begin(v) + stride; it <= std::end(v) - stride;
       it += stride) {
    *(it) += 0.25 * (*(it - Pstride) + *(it + Pstride));
  }
  it = v.end() - Pstride - 1;
  v.back() += 0.25 * (*it); // last element
}

void qwrite_level_2D(const int nrow, const int ncol, const int nlevel,
                     const int l, float *v, float tol,
                     const std::string outfile) {

  int stride = std::pow(2, l);

  float norm = 0;

  for (int irow = 0; irow < nrow; irow += stride) {
    for (int jcol = 0; jcol < ncol; jcol += stride) {
      float ntest = std::abs(v[get_index(ncol, irow, jcol)]);
      if (ntest > norm)
        norm = ntest;
    }
  }

  tol /= (float)(nlevel + 1);
  float coeff = norm * tol;

  gzFile out_file = gzopen(outfile.c_str(), "w9b");
  gzwrite(out_file, &coeff, sizeof(float));

  int prune_count = 0;

  for (int l = 0; l <= nlevel; l++) {
    int stride = std::pow(2, l);
    int Cstride = stride * 2;
    int row_counter = 0;

    for (int irow = 0; irow < nrow; irow += stride) {
      if (row_counter % 2 == 0 && l != nlevel) {
        for (int jcol = Cstride; jcol < ncol; jcol += Cstride) {
          int quantum = (int)(v[get_index(ncol, irow, jcol - stride)] / coeff);
          if (quantum == 0)
            ++prune_count;
          gzwrite(out_file, &quantum, sizeof(int));
        }
      } else {
        for (int jcol = 0; jcol < ncol; jcol += stride) {
          int quantum = (int)(v[get_index(ncol, irow, jcol)] / coeff);
          if (quantum == 0)
            ++prune_count;
          gzwrite(out_file, &quantum, sizeof(int));
        }
      }
      ++row_counter;
    }
  }

  // std::cout  << "Pruned : " << prune_count << " Reduction : "
  //            << (float)nrow * ncol / (nrow * ncol - prune_count) << "\n";
  gzclose(out_file);
}

void quantize_2D_interleave(const int nrow, const int ncol, float *v,
                            std::vector<int> &work, float norm, float tol) {
  //  //std::cout  << "Tolerance: " << tol << "\n";
  int size_ratio = sizeof(float) / sizeof(int);

  ////std::cout  << "Norm of sorts: " << norm << "\n";

  //    float quantizer = 2.0*norm * tol;
  float quantizer = norm * tol;
  ////std::cout  << "Quantization factor: " << quantizer << "\n";
  std::memcpy(work.data(), &quantizer, sizeof(float));

  int prune_count = 0;

  for (int index = 0; index < ncol * nrow; ++index) {
    int quantum = (int)(v[index] / quantizer);
    work[index + size_ratio] = quantum;
    if (quantum == 0)
      ++prune_count;
  }

  ////std::cout  << "Pruned : " << prune_count << " Reduction : "
  //          << (float)2 * nrow * ncol / (nrow * ncol - prune_count) << "\n";
}

void dequantize_2D_interleave(const int nrow, const int ncol, float *v,
                              const std::vector<int> &work) {
  int size_ratio = sizeof(float) / sizeof(int);
  float quantizer;

  std::memcpy(&quantizer, work.data(), sizeof(float));

  for (int index = 0; index < nrow * ncol; ++index) {
    v[index] = quantizer * float(work[index + size_ratio]);
  }
}

void qwrite_2D_interleave(const int nrow, const int ncol, const int nlevel,
                          const int l, float *v, float tol,
                          const std::string outfile) {

  int stride = std::pow(2, l); // current stride

  float norm = 0;

  for (int irow = 0; irow < nrow; irow += stride) {
    for (int jcol = 0; jcol < ncol; jcol += stride) {
      float ntest = std::abs(v[get_index(ncol, irow, jcol)]);
      if (ntest > norm)
        norm = ntest;
    }
  }

  tol /= (float)(nlevel + 1);

  float coeff = norm * tol;
  // std::cout  << "Quantization factor: " << coeff << "\n";

  gzFile out_file = gzopen(outfile.c_str(), "w6b");
  int prune_count = 0;
  gzwrite(out_file, &coeff, sizeof(float));

  for (auto index = 0; index < ncol * nrow; ++index) {
    int quantum = (int)(v[index] / coeff);
    if (quantum == 0)
      ++prune_count;
    gzwrite(out_file, &quantum, sizeof(int));
  }

  // std::cout  << "Pruned : " << prune_count << " Reduction : "
  //            << (float)nrow * ncol / (nrow * ncol - prune_count) << "\n";
  gzclose(out_file);
}

void qread_level_2D(const int nrow, const int ncol, const int nlevel, float *v,
                    std::string infile) {
  int buff_size = 128 * 1024;
  unsigned char unzip_buffer[buff_size];
  int int_buffer[buff_size / sizeof(int)];
  unsigned int unzipped_bytes, total_bytes = 0;
  float coeff;

  gzFile in_file_z = gzopen(infile.c_str(), "r");
  // std::cout  << in_file_z << "\n";

  unzipped_bytes = gzread(in_file_z, unzip_buffer,
                          sizeof(float)); // read the quantization constant
  std::memcpy(&coeff, &unzip_buffer, unzipped_bytes);

  int last = 0;
  while (true) {
    unzipped_bytes = gzread(in_file_z, unzip_buffer, buff_size);
    // std::cout  << unzipped_bytes << "\n";
    if (unzipped_bytes > 0) {
      total_bytes += unzipped_bytes;
      int num_int = unzipped_bytes / sizeof(int);

      std::memcpy(&int_buffer, &unzip_buffer, unzipped_bytes);
      for (int i = 0; i < num_int; ++i) {
        v[last] = float(int_buffer[i]) * coeff;
        ++last;
      }
    } else {
      break;
    }
  }

  gzclose(in_file_z);
}

//       unzippedBytes = gzread(inFileZ, unzipBuffer, buff_size);
//       //std::cout  << "Read: "<< unzippedBytes <<"\n";
//       std::memcpy(&v[irow][0], &unzipBuffer, unzippedBytes);
//     }

//   gzclose(inFileZ);
// }

void refactor(const int nrow, const int ncol, const int l_target, float *v,
              std::vector<float> &work, std::vector<float> &row_vec,
              std::vector<float> &col_vec) {
  // refactor
  //  //std::cout  << "refactoring" << "\n";

  for (int l = 0; l < l_target; ++l) {

    int stride = std::pow(2, l); // current stride
    int Cstride = stride * 2;    // coarser stride

    pi_Ql(nrow, ncol, l, v, row_vec,
          col_vec); // rename!. v@l has I-\Pi_l Q_l+1 u
    copy_level(nrow, ncol, l, v,
               work); // copy the nodal values of v on l  to matrix work

    assign_num_level(nrow, ncol, l + 1, work.data(), 0.0);

    // row-sweep
    for (int irow = 0; irow < nrow; ++irow) {
      for (int jcol = 0; jcol < ncol; ++jcol) {
        row_vec[jcol] = work[get_index(ncol, irow, jcol)];
      }

      mass_matrix_multiply(l, row_vec);

      restriction(l + 1, row_vec);

      solve_tridiag_M(l + 1, row_vec);

      for (int jcol = 0; jcol < ncol; ++jcol) {
        work[get_index(ncol, irow, jcol)] = row_vec[jcol];
      }
    }

    // column-sweep
    if (nrow > 1) // do this if we have an 2-dimensional array
    {
      for (int jcol = 0; jcol < ncol; jcol += Cstride) {
        for (int irow = 0; irow < nrow; ++irow) {
          col_vec[irow] = work[get_index(ncol, irow, jcol)];
        }

        mass_matrix_multiply(l, col_vec);

        restriction(l + 1, col_vec);
        solve_tridiag_M(l + 1, col_vec);

        for (int irow = 0; irow < nrow; ++irow) {
          work[get_index(ncol, irow, jcol)] = col_vec[irow];
        }
      }
    }

    // Solved for (z_l, phi_l) = (c_{l+1}, vl)

    add_level(nrow, ncol, l + 1, v,
              work.data()); // Qu_l = \Pi_l Q_{l+1}u + z_l
  }
}

void recompose(const int nrow, const int ncol, const int l_target, float *v,
               std::vector<float> &work, std::vector<float> &row_vec,
               std::vector<float> &col_vec) {

  // recompose

  for (int l = l_target; l > 0; --l) {

    int stride = std::pow(2, l); // current stride
    int Pstride = stride / 2;

    copy_level(nrow, ncol, l - 1, v, work); // copy the nodal values of cl
                                            // on l-1 (finer level)  to
                                            // matrix work
    assign_num_level(nrow, ncol, l, work.data(),
                     0.0); // zero out nodes of l on cl

    // row-sweep
    for (int irow = 0; irow < nrow; ++irow) {
      for (int jcol = 0; jcol < ncol; ++jcol) {
        row_vec[jcol] = work[get_index(ncol, irow, jcol)];
      }

      mass_matrix_multiply(l - 1, row_vec);

      restriction(l, row_vec);
      solve_tridiag_M(l, row_vec);

      for (int jcol = 0; jcol < ncol; ++jcol) {
        work[get_index(ncol, irow, jcol)] = row_vec[jcol];
      }
    }

    // column-sweep, this is the slow one! Need something like column_copy
    if (nrow > 1) // check if we have 1-D array..
    {
      for (int jcol = 0; jcol < ncol; jcol += stride) {
        for (int irow = 0; irow < nrow; ++irow) {
          col_vec[irow] = work[get_index(ncol, irow, jcol)];
        }

        mass_matrix_multiply(l - 1, col_vec);

        restriction(l, col_vec);
        solve_tridiag_M(l, col_vec);

        for (int irow = 0; irow < nrow; ++irow) {
          work[get_index(ncol, irow, jcol)] = col_vec[irow];
        }
      }
    }
    subtract_level(nrow, ncol, l, work.data(), v); // do -(Qu - zl)

    // row-sweep
    for (int irow = 0; irow < nrow; irow += stride) {
      for (int jcol = 0; jcol < ncol; ++jcol) {
        row_vec[jcol] = work[get_index(ncol, irow, jcol)];
      }

      interpolate_from_level_nMl(l, row_vec);

      for (int jcol = 0; jcol < ncol; ++jcol) {
        work[get_index(ncol, irow, jcol)] = row_vec[jcol];
      }
    }

    // column-sweep, this is the slow one! Need something like column_copy
    if (nrow > 1) {
      for (int jcol = 0; jcol < ncol; jcol += Pstride) {
        for (int irow = 0; irow < nrow; ++irow) // copy all rows
        {
          col_vec[irow] = work[get_index(ncol, irow, jcol)];
        }

        interpolate_from_level_nMl(l, col_vec);

        for (int irow = 0; irow < nrow; ++irow) {
          work[get_index(ncol, irow, jcol)] = col_vec[irow];
        }
      }
    }
    assign_num_level(nrow, ncol, l, v, 0.0); // zero out nodes of l on cl
    subtract_level(nrow, ncol, l - 1, v, work.data());
  }
}

inline float interp_2d(float q11, float q12, float q21, float q22, float x1,
                       float x2, float y1, float y2, float x, float y) {
  float x2x1, y2y1, x2x, y2y, yy1, xx1;
  x2x1 = x2 - x1;
  y2y1 = y2 - y1;
  x2x = x2 - x;
  y2y = y2 - y;
  yy1 = y - y1;
  xx1 = x - x1;
  return 1.0 / (x2x1 * y2y1) *
         (q11 * x2x * y2y + q21 * xx1 * y2y + q12 * x2x * yy1 +
          q22 * xx1 * yy1);
}

inline float interp_0d(const float x1, const float x2, const float y1,
                       const float y2, const float x) {
  // do a linear interpolation between (x1, y1) and (x2, y2)
  return (((x2 - x) * y1 + (x - x1) * y2) / (x2 - x1));
}

void resample_1d(const float *inbuf, float *outbuf, const int ncol,
                 const int ncol_new) {
  float hx_o = 1.0 / float(ncol - 1);
  float hx = 1.0 / float(ncol_new - 1); // x-spacing
  float hx_ratio = (hx_o / hx);         // ratio of x-spacing resampled/orig

  for (int icol = 0; icol < ncol_new - 1; ++icol) {
    int i_left = floor(icol / hx_ratio);
    int i_right = i_left + 1;

    float x1 = float(i_left) * hx_o;
    float x2 = float(i_right) * hx_o;

    float y1 = inbuf[i_left];
    float y2 = inbuf[i_right];
    float x = float(icol) * hx;
    //      //std::cout  <<  x1 << "\t" << x2 << "\t" << x << "\t"<< "\n";
    // //std::cout  <<  y1 << "\t" << y2 << "\t" << "\n";

    outbuf[icol] = interp_0d(x1, x2, y1, y2, x);
    //      std:: cout << mgard_interp_0d( x1,  x2,  y1,  y2,  x) << "\n";
  }

  outbuf[ncol_new - 1] = inbuf[ncol - 1];
}

void resample_1d_inv2(const float *inbuf, float *outbuf, const int ncol,
                      const int ncol_new) {
  float hx_o = 1.0 / float(ncol - 1);
  float hx = 1.0 / float(ncol_new - 1); // x-spacing
  float hx_ratio = (hx_o / hx);         // ratio of x-spacing resampled/orig

  for (int icol = 0; icol < ncol_new - 1; ++icol) {
    int i_left = floor(icol / hx_ratio);
    int i_right = i_left + 1;

    float x1 = float(i_left) * hx_o;
    float x2 = float(i_right) * hx_o;

    float y1 = inbuf[i_left];
    float y2 = inbuf[i_right];
    float x = float(icol) * hx;

    float d1 = std::pow(x1 - x, 4.0);
    float d2 = std::pow(x2 - x, 4.0);

    if (d1 == 0) {
      outbuf[icol] = y1;
    } else if (d2 == 0) {
      outbuf[icol] = y2;
    } else {
      float dsum = 1.0 / d1 + 1.0 / d2;
      outbuf[icol] = (y1 / d1 + y2 / d2) / dsum;
    }
  }

  outbuf[ncol_new - 1] = inbuf[ncol - 1];
}

void resample_2d(const float *inbuf, float *outbuf, const int nrow,
                 const int ncol, const int nrow_new, const int ncol_new) {
  float hx_o = 1.0 / float(ncol - 1);
  float hx = 1.0 / float(ncol_new - 1); // x-spacing
  float hx_ratio = (hx_o / hx);         // ratio of x-spacing resampled/orig

  float hy_o = 1.0 / float(nrow - 1);
  float hy = 1.0 / float(nrow_new - 1); // x-spacing
  float hy_ratio = (hy_o / hy);         // ratio of x-spacing resampled/orig

  for (int irow = 0; irow < nrow_new - 1; ++irow) {
    int i_bot = floor(irow / hy_ratio);
    int i_top = i_bot + 1;

    float y = float(irow) * hy;
    float y1 = float(i_bot) * hy_o;
    float y2 = float(i_top) * hy_o;

    for (int jcol = 0; jcol < ncol_new - 1; ++jcol) {
      int j_left = floor(jcol / hx_ratio);
      int j_right = j_left + 1;

      float x = float(jcol) * hx;
      float x1 = float(j_left) * hx_o;
      float x2 = float(j_right) * hx_o;

      float q11 = inbuf[get_index(ncol, i_bot, j_left)];
      float q12 = inbuf[get_index(ncol, i_top, j_left)];
      float q21 = inbuf[get_index(ncol, i_bot, j_right)];
      float q22 = inbuf[get_index(ncol, i_top, j_right)];

      outbuf[get_index(ncol_new, irow, jcol)] =
          interp_2d(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
    }

    // last column
    float q1 = inbuf[get_index(ncol, i_bot, ncol - 1)];
    float q2 = inbuf[get_index(ncol, i_top, ncol - 1)];
    outbuf[get_index(ncol_new, irow, ncol_new - 1)] =
        interp_0d(y1, y2, q1, q2, y);
  }

  // last-row
  resample_1d(&inbuf[get_index(ncol, nrow - 1, 0)],
              &outbuf[get_index(ncol_new, nrow_new - 1, 0)], ncol, ncol_new);
}

void resample_2d_inv2(const float *inbuf, float *outbuf, const int nrow,
                      const int ncol, const int nrow_new, const int ncol_new) {
  float hx_o = 1.0 / float(ncol - 1);
  float hx = 1.0 / float(ncol_new - 1); // x-spacing
  float hx_ratio = (hx_o / hx);         // ratio of x-spacing resampled/orig

  float hy_o = 1.0 / float(nrow - 1);
  float hy = 1.0 / float(nrow_new - 1); // x-spacing
  float hy_ratio = (hy_o / hy);         // ratio of x-spacing resampled/orig

  for (int irow = 0; irow < nrow_new - 1; ++irow) {
    int i_bot = floor(irow / hy_ratio);
    int i_top = i_bot + 1;

    float y = float(irow) * hy;
    float y1 = float(i_bot) * hy_o;
    float y2 = float(i_top) * hy_o;

    for (int jcol = 0; jcol < ncol_new - 1; ++jcol) {
      int j_left = floor(jcol / hx_ratio);
      int j_right = j_left + 1;

      float x = float(jcol) * hx;
      float x1 = float(j_left) * hx_o;
      float x2 = float(j_right) * hx_o;

      float q11 = inbuf[get_index(ncol, i_bot, j_left)];
      float q12 = inbuf[get_index(ncol, i_top, j_left)];
      float q21 = inbuf[get_index(ncol, i_bot, j_right)];
      float q22 = inbuf[get_index(ncol, i_top, j_right)];

      float d11 = (std::pow(x1 - x, 2.0) + std::pow(y1 - y, 2.0));
      float d12 = (std::pow(x1 - x, 2.0) + std::pow(y2 - y, 2.0));
      float d21 = (std::pow(x2 - x, 2.0) + std::pow(y1 - y, 2.0));
      float d22 = (std::pow(x2 - x, 2.0) + std::pow(y2 - y, 2.0));

      if (d11 == 0) {
        outbuf[get_index(ncol_new, irow, jcol)] = q11;
      } else if (d12 == 0) {
        outbuf[get_index(ncol_new, irow, jcol)] = q12;
      } else if (d21 == 0) {
        outbuf[get_index(ncol_new, irow, jcol)] = q21;
      } else if (d22 == 0) {
        outbuf[get_index(ncol_new, irow, jcol)] = q22;
      } else {
        d11 = std::pow(d11, 1.5);
        d12 = std::pow(d12, 1.5);
        d21 = std::pow(d21, 1.5);
        d22 = std::pow(d22, 1.5);

        float dsum = 1.0 / (d11) + 1.0 / (d12) + 1.0 / (d21) + 1.0 / (d22);
        ////std::cout  <<  (q11/d11 + q12/d12 + q21/d21 + q22/d22)/dsum << "\n";
        //              //std::cout  <<  dsum << "\n";

        outbuf[get_index(ncol_new, irow, jcol)] =
            (q11 / d11 + q12 / d12 + q21 / d21 + q22 / d22) / dsum;
      }
    }

    // last column
    float q1 = inbuf[get_index(ncol, i_bot, ncol - 1)];
    float q2 = inbuf[get_index(ncol, i_top, ncol - 1)];

    float d1 = std::pow(y1 - y, 4.0);
    float d2 = std::pow(y2 - y, 4.0);

    if (d1 == 0) {
      outbuf[get_index(ncol_new, irow, ncol_new - 1)] = q1;
    } else if (d2 == 0) {
      outbuf[get_index(ncol_new, irow, ncol_new - 1)] = q2;
    } else {
      float dsum = 1.0 / d1 + 1.0 / d2;
      outbuf[get_index(ncol_new, irow, ncol_new - 1)] =
          (q1 / d1 + q2 / d2) / dsum;
    }
  }

  // last-row
  resample_1d_inv2(&inbuf[get_index(ncol, nrow - 1, 0)],
                   &outbuf[get_index(ncol_new, nrow_new - 1, 0)], ncol,
                   ncol_new);
}

} // end namespace mgard
