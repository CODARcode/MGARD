// Copyright 2017, Brown University, Providence, RI.
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney
// Corresponding Author: Ozan Tugluk
//
// version: 0.0.0.2
// See LICENSE for details.
#ifndef MGARD_NUNI_TPP
#define MGARD_NUNI_TPP

#include "mgard_nuni.h"

#include <cassert>
#include <cmath>
#include <cstring>

#include <zlib.h>

#include <fstream>
#include <iostream>

#include "interpolation.hpp"
#include "mgard_mesh.hpp"

#include "LinearQuantizer.hpp"

namespace mgard_common {

template <typename Real> Real max_norm(const std::vector<Real> &v) {
  Real norm = 0;

  for (int i = 0; i < v.size(); ++i) {
    Real ntest = std::abs(v[i]);
    if (ntest > norm)
      norm = ntest;
  }
  return norm;
}

template <typename Real>
Real get_h(const std::vector<Real> &coords, int i, int stride) {
  return (coords[i + stride] - coords[i]);
}

template <typename Real>
Real get_dist(const std::vector<Real> &coords, int i, int j) {
  return (coords[j] - coords[i]);
}

template <typename Real>
void qread_2D_interleave(const int nrow, const int ncol, const int nlevel,
                         Real *v, std::string infile) {
  int buff_size = 128 * 1024;
  unsigned char unzip_buffer[buff_size];
  int int_buffer[buff_size / sizeof(int)];
  unsigned int unzipped_bytes, total_bytes = 0;

  Real quantum;
  gzFile in_file_z = gzopen(infile.c_str(), "r");
  // std::cout  << "File to oppen:" << in_file_z <<"\n";

  unzipped_bytes = gzread(in_file_z, unzip_buffer,
                          sizeof(Real)); // read the quantization constant
  std::memcpy(&quantum, &unzip_buffer, unzipped_bytes);
  const mgard::LinearDequantizer<int, Real> dequantizer(quantum);
  int last = 0;
  while (true) {
    unzipped_bytes = gzread(in_file_z, unzip_buffer, buff_size);
    //      //std::cout  << unzipped_bytes <<"\n";
    if (unzipped_bytes > 0) {
      total_bytes += unzipped_bytes;
      int num_int = unzipped_bytes / sizeof(int);

      std::memcpy(&int_buffer, &unzip_buffer, unzipped_bytes);
      for (int i = 0; i < num_int; ++i) {
        v[last] = dequantizer(int_buffer[i]);
        ++last;
      }

    } else {
      break;
    }
  }

  gzclose(in_file_z);
}

template <typename Real>
void qwrite_2D_interleave(const int nrow, const int ncol, const int nlevel,
                          const int l, Real *v, const Real tol, const Real norm,
                          const std::string outfile) {

  int stride = std::pow(2, l); // current stride

  const mgard::LinearQuantizer<Real, int> quantizer(2 * norm * tol /
                                                    (nlevel + 1));

  gzFile out_file = gzopen(outfile.c_str(), "w6b");
  int prune_count = 0;
  gzwrite(out_file, &quantizer.quantum, sizeof(Real));

  for (auto index = 0; index < ncol * nrow; ++index) {
    const int n = quantizer(v[index]);
    if (n == 0)
      ++prune_count;
    gzwrite(out_file, &n, sizeof(int));
  }

  // std::cout  << "Pruned : "<< prune_count << " Reduction : " << (Real)
  // nrow*ncol /(nrow*ncol - prune_count) << "\n";
  gzclose(out_file);
}

template <typename Real>
void qwrite_3D_interleave(const int nrow, const int ncol, const int nfib,
                          const int nlevel, const int l, Real *v,
                          const Real tol, const Real norm,
                          const std::string outfile) {

  int stride = std::pow(2, l); // current stride

  const mgard::LinearQuantizer<Real, int> quantizer(2 * norm * tol /
                                                    (nlevel + 1));

  gzFile out_file = gzopen(outfile.c_str(), "w6b");
  int prune_count = 0;
  gzwrite(out_file, &quantizer.quantum, sizeof(Real));

  for (auto index = 0; index < ncol * nrow * nfib; ++index) {
    const int n = quantizer(v[index]);
    if (n == 0)
      ++prune_count;
    gzwrite(out_file, &n, sizeof(int));
  }

  // std::cout  << "Pruned : "<< prune_count << " Reduction : " << (Real)
  // nrow*ncol*nfib /(nrow*ncol*nfib - prune_count) << "\n";
  gzclose(out_file);
}

template <typename Real>
void qwrite_3D_interleave2(const int nrow, const int ncol, const int nfib,
                           const int nlevel, const int l, Real *v,
                           const Real tol, const Real norm,
                           const std::string outfile) {

  int stride = std::pow(2, l); // current stride

  // There was a commented-out line here dividing the tolerance by `nlevel + 1`.
  const mgard::LinearQuantizer<Real, int> quantizer(norm * tol);

  gzFile out_file = gzopen(outfile.c_str(), "w6b");
  int prune_count = 0;
  gzwrite(out_file, &quantizer.quantum, sizeof(Real));

  for (auto index = 0; index < ncol * nrow * nfib; ++index) {
    const int n = quantizer(v[index]);
    if (n == 0)
      ++prune_count;
    gzwrite(out_file, &n, sizeof(int));
  }

  // std::cout  << "Pruned : "<< prune_count << " Reduction : " << (Real)
  // nrow*ncol*nfib /(nrow*ncol*nfib - prune_count) << "\n";
  gzclose(out_file);
}

template <typename Real>
void copy_slice(Real *work, std::vector<Real> &work2d, int nrow, int ncol,
                int nfib, int is) {
  for (int i = 0; i < nrow; ++i) {
    for (int j = 0; j < ncol; ++j) {
      work2d[mgard::get_index(ncol, i, j)] =
          work[mgard::get_index3(ncol, nfib, i, j, is)];
    }
  }
}

template <typename Real>
void copy_from_slice(Real *work, std::vector<Real> &work2d, int nrow, int ncol,
                     int nfib, int is) {
  for (int i = 0; i < nrow; ++i) {
    for (int j = 0; j < ncol; ++j) {
      work[mgard::get_index3(ncol, nfib, i, j, is)] =
          work2d[mgard::get_index(ncol, i, j)];
    }
  }
}
} // namespace mgard_common

namespace mgard_cannon {

template <typename Real>
void assign_num_level(const int nrow, const int ncol, const int l, Real *v,
                      Real num) {
  // set the value of nodal values at level l to number num

  int stride = std::pow(2, l); // current stride

  for (int irow = 0; irow < nrow; irow += stride) {
    for (int jcol = 0; jcol < ncol; jcol += stride) {
      v[mgard::get_index(ncol, irow, jcol)] = num;
    }
  }
}

template <typename Real>
void subtract_level(const int nrow, const int ncol, const int l, Real *v,
                    Real *work) {
  // v += work at level l
  int stride = std::pow(2, l); // current stride

  for (int irow = 0; irow < nrow; irow += stride) {
    for (int jcol = 0; jcol < ncol; jcol += stride) {
      v[mgard::get_index(ncol, irow, jcol)] -=
          work[mgard::get_index(ncol, irow, jcol)];
    }
  }
}

template <typename Real>
void pi_lminus1(const int l, std::vector<Real> &v,
                const std::vector<Real> &coords) {
  int n = v.size();
  int nlevel = static_cast<int>(std::log2(v.size() - 1));
  int my_level = nlevel - l;
  int stride = std::pow(2, l); // current stride
  //  int Pstride = stride/2; //finer stride
  int Cstride = stride * 2; // coarser stride

  if (my_level != 0) {
    for (int i = Cstride; i < n; i += Cstride) {
      Real h1 = mgard_common::get_h(coords, i - Cstride, stride);
      Real h2 = mgard_common::get_h(coords, i - stride, stride);
      Real hsum = h1 + h2;
      v[i - stride] -= (h1 * v[i] + h2 * v[i - Cstride]) / hsum;
    }
  }
}

template <typename Real>
void restriction(const int l, std::vector<Real> &v,
                 const std::vector<Real> &coords) {
  int stride = std::pow(2, l);
  int Pstride = stride / 2; // finer stride
  int n = v.size();

  // calculate the result of restrictionion

  Real h1 = mgard_common::get_h(coords, 0, Pstride);
  Real h2 = mgard_common::get_h(coords, Pstride, Pstride);
  Real hsum = h1 + h2;

  v.front() += h2 * v[Pstride] / hsum; // first element

  for (int i = stride; i <= n - stride; i += stride) {
    v[i] += h1 * v[i - Pstride] / hsum;
    h1 = mgard_common::get_h(coords, i, Pstride);
    h2 = mgard_common::get_h(coords, i + Pstride, Pstride);
    hsum = h1 + h2;
    v[i] += h2 * v[i + Pstride] / hsum;
  }
  v.back() += h1 * v[n - Pstride - 1] / hsum; // last element
}

template <typename Real>
void prolongate(const int l, std::vector<Real> &v,
                const std::vector<Real> &coords) {

  int stride = std::pow(2, l);
  int Pstride = stride / 2;
  int n = v.size();

  for (int i = stride; i < n; i += stride) {
    Real h1 = mgard_common::get_h(coords, i - stride, Pstride);
    Real h2 = mgard_common::get_h(coords, i - Pstride, Pstride);

    v[i - Pstride] = (h2 * v[i - stride] + h1 * v[i]) / (h1 + h2);
  }
}

template <typename Real>
void solve_tridiag_M(const int l, std::vector<Real> &v,
                     const std::vector<Real> &coords) {

  //  int my_level = nlevel - l;
  int stride = std::pow(2, l); // current stride

  Real am, bm, h1, h2;
  int n = v.size();

  am = 2.0 * mgard_common::get_h(coords, 0,
                                 stride); // first element of upper diagonal U.

  //    bm = mgard_common::get_h(coords, 0, stride) / am;
  bm = mgard_common::get_h(coords, 0, stride) / am;
  int nlevel = static_cast<int>(std::log2(v.size() - 1));
  //    //std::cout  << nlevel;
  int nc = std::pow(2, nlevel - l) + 1;
  std::vector<Real> coeff(n);
  int counter = 1;
  coeff.front() = am;

  // forward sweep
  for (int i = stride; i < n - 1; i += stride) {
    h1 = mgard_common::get_h(coords, i - stride, stride);
    h2 = mgard_common::get_h(coords, i, stride);
    //        //std::cout  << i<< "\t"<< v[i-stride] << "\t" << h1<< "\t"<<
    //        h2<<"\n";
    v[i] -= v[i - stride] * bm;

    am = 2.0 * (h1 + h2) - bm * h1;
    bm = h2 / am;
    //        //std::cout  <<  am<< "\t"<< bm<<"\n";

    coeff.at(counter) = am;
    ++counter;
  }

  h2 = mgard_common::get_h(coords, n - 1 - stride, stride);
  am = 2.0 * h2 - bm * h2; // a_n = 2 - b_(n-1)
  //    //std::cout  << h1 << "\t"<< h2<<"\n";
  v[n - 1] -= v[n - 1 - stride] * bm;

  coeff.at(counter) = am;

  // backward sweep

  v[n - 1] /= am;
  --counter;

  for (int i = n - 1 - stride; i >= 0; i -= stride) {
    // h1 = mgard_common::get_h(coords, i-stride, stride);
    h2 = mgard_common::get_h(coords, i, stride);
    v[i] = (v[i] - h2 * v[i + stride]) / coeff.at(counter);
    --counter;
    //        bm = (2.0*(h1+h2) - am) / h1 ;
    // am = 1.0 / bm;
  }
  // h1 = mgard_common::get_h(coords, 0, stride);
  //    //std::cout  << h1 << "\n";
  //    v[0] = (v[0] - h1*v[1])/coeff[0];
}

template <typename Real>
void mass_matrix_multiply(const int l, std::vector<Real> &v,
                          const std::vector<Real> &coords) {

  int stride = std::pow(2, l);
  int n = v.size();
  Real temp1, temp2;

  // Mass matrix times nodal value-vec
  temp1 = v.front(); // save u(0) for later use
  v.front() = 2.0 * mgard_common::get_h(coords, 0, stride) * temp1 +
              mgard_common::get_h(coords, 0, stride) * v[stride];
  for (int i = stride; i <= n - 1 - stride; i += stride) {
    temp2 = v[i];
    v[i] = mgard_common::get_h(coords, i - stride, stride) * temp1 +
           2 *
               (mgard_common::get_h(coords, i - stride, stride) +
                mgard_common::get_h(coords, i, stride)) *
               temp2 +
           mgard_common::get_h(coords, i, stride) * v[i + stride];
    temp1 = temp2; // save u(n) for later use
  }
  v[n - 1] = mgard_common::get_h(coords, n - stride - 1, stride) * temp1 +
             2 * mgard_common::get_h(coords, n - stride - 1, stride) * v[n - 1];
}

template <typename Real>
void write_level_2D(const int nrow, const int ncol, const int l, Real *v,
                    std::ofstream &outfile) {
  int stride = std::pow(2, l);
  //  int nrow = std::pow(2, nlevel_row) + 1;
  // int ncol = std::pow(2, nlevel_col) + 1;

  for (int irow = 0; irow < nrow; irow += stride) {
    for (int jcol = 0; jcol < ncol; jcol += stride) {
      outfile.write(
          reinterpret_cast<char *>(&v[mgard::get_index(ncol, irow, jcol)]),
          sizeof(Real));
    }
  }
}

template <typename Real>
void copy_level(const int nrow, const int ncol, const int l, Real *v,
                std::vector<Real> &work) {

  int stride = std::pow(2, l); // current stride

  for (int irow = 0; irow < nrow; irow += stride) {
    for (int jcol = 0; jcol < ncol; jcol += stride) {
      work[mgard::get_index(ncol, irow, jcol)] =
          v[mgard::get_index(ncol, irow, jcol)];
    }
  }
}

template <typename Real>
void copy_level3(const int nrow, const int ncol, const int nfib, const int l,
                 Real *v, std::vector<Real> &work) {

  int stride = std::pow(2, l); // current stride

  for (int irow = 0; irow < nrow; irow += stride) {
    for (int jcol = 0; jcol < ncol; jcol += stride) {
      for (int kfib = 0; kfib < nfib; kfib += stride) {
        work[mgard::get_index3(ncol, nfib, irow, jcol, kfib)] =
            v[mgard::get_index3(ncol, nfib, irow, jcol, kfib)];
      }
    }
  }
}
} // namespace mgard_cannon

namespace mgard_gen {
template <typename Real>
Real *get_ref(std::vector<Real> &v, const int n, const int no,
              const int i) // return reference to logical element
{
  // no: original number of points
  // n : number of points at next coarser level (L-1) with  2^k+1 nodes
  // may not work for the last element!
  if (i != n - 1) {
    return &v[floor(((Real)no - 2.0) / ((Real)n - 2.0) * i)];
  }
  // else if( i == n-1 )
  //   {
  return &v[no - 1];
  //      }

  //    return &v[floor(((no-2)/(n-2))*i ) ];
}

template <typename Real>
Real get_h_l(const std::vector<Real> &coords, const int n, const int no, int i,
             int stride) {

  //    return (*get_ref(coords, n, no, i+stride) - *get_ref(coords, n, no, i));
  return (coords[mgard::get_lindex(n, no, i + stride)] -
          coords[mgard::get_lindex(n, no, i)]);
}

template <typename Real>
void write_level_2D_l(const int l, Real *v, std::ofstream &outfile, int nr,
                      int nc, int nrow, int ncol) {
  int stride = std::pow(2, l);
  //  int nrow = std::pow(2, nlevel_row) + 1;
  // int ncol = std::pow(2, nlevel_col) + 1;

  for (int irow = 0; irow < nr; irow += stride) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < nc; jcol += stride) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      outfile.write(
          reinterpret_cast<char *>(&v[mgard::get_index(ncol, ir, jr)]),
          sizeof(Real));
    }
  }
}

template <typename Real>
void qwrite_3D(const int nr, const int nc, const int nf, const int nrow,
               const int ncol, const int nfib, const int nlevel, const int l,
               Real *v, const std::vector<Real> &coords_x,
               const std::vector<Real> &coords_y,
               const std::vector<Real> &coords_z, const Real tol, const Real s,
               const Real norm, const std::string outfile) {

  const Real coeff = tol; /// 4.322;//

  gzFile out_file = gzopen(outfile.c_str(), "w6b");
  int prune_count = 0;
  gzwrite(out_file, &coeff, sizeof(Real));

  //    Real s = 0.0;
  int count = 0;

  // level -1, first level for non 2^k+1
  {

    const Real dx = mgard_gen::get_h_l(coords_x, ncol, ncol, 0, 1);
    const Real dy = mgard_gen::get_h_l(coords_y, nrow, nrow, 0, 1);
    const Real dz = mgard_gen::get_h_l(coords_z, nfib, nfib, 0, 1);
    const Real vol = dx * dy * dz;

    const mgard::LinearQuantizer<Real, int> quantizer(
        coeff / (std::sqrt(vol) * std::pow(2, s * nlevel)));

    for (int kfib = 0; kfib < nf - 1; ++kfib) {
      int kf = mgard::get_lindex(nf, nfib, kfib);
      int kfp = mgard::get_lindex(nf, nfib, kfib + 1);

      if (kfp != kf + 1) // skipped a plane
      {
        for (int irow = 0; irow < nrow; ++irow) {
          for (int jcol = 0; jcol < ncol; ++jcol) {
            Real val = v[mgard::get_index3(ncol, nfib, irow, jcol, kf + 1)];
            const int n = quantizer(val);
            gzwrite(out_file, &n, sizeof(int));
            ++count;
          }
        }
      }
    }

    int count_row = 0;
    int count_col = 0;
    int count_sol = 0;

    for (int kfib = 0; kfib < nf; ++kfib) {
      int kf = mgard::get_lindex(nf, nfib, kfib);
      for (int irow = 0; irow < nr - 1; ++irow) {
        int ir = mgard::get_lindex(nr, nrow, irow);
        int irP = mgard::get_lindex(nr, nrow, irow + 1);
        if (irP != ir + 1) // skipped a row
        {
          //  //std::cout  <<"Skipped row: "  << ir + 1 << "\n";
          for (int jcol = 0; jcol < ncol; ++jcol) {
            Real val = v[mgard::get_index3(ncol, nfib, ir + 1, jcol, kf)];
            const int n = quantizer(val);
            gzwrite(out_file, &n, sizeof(int));
            ++count_row;
            ++count;
          }
        }
      }

      for (int irow = 0; irow < nr; ++irow) {
        int ir = mgard::get_lindex(nr, nrow, irow);

        //      //std::cout  <<"Non skipped row: "  << ir  << "\n";
        for (int jcol = 0; jcol < nc - 1; ++jcol) {
          int jc = mgard::get_lindex(nc, ncol, jcol);
          int jcP = mgard::get_lindex(nc, ncol, jcol + 1);
          if (jcP != jc + 1) // skipped a column
          {
            Real val = v[mgard::get_index3(ncol, nfib, ir, jc + 1, kf)];
            const int n = quantizer(val);
            gzwrite(out_file, &n, sizeof(int));
            ++count_col;
            ++count;
            //                    //std::cout  <<"Skipped col: " << ir << "\t"
            //                    << jc + 1 << "\n";
          }
        }
      }
    }
  }

  // // 2^k+1 part //
  for (int ilevel = 0; ilevel < nlevel; ++ilevel) {
    int stride = std::pow(2, ilevel);
    int Cstride = 2 * stride;

    int fib_counter = 0;

    const Real dx = get_h_l(coords_x, nc, ncol, 0, stride);
    const Real dy = get_h_l(coords_y, nr, nrow, 0, stride);
    const Real dz = get_h_l(coords_z, nf, nfib, 0, stride);
    const Real vol = dx * dy * dz;
    const mgard::LinearQuantizer<Real, int> quantizer(
        coeff / (std::sqrt(vol) * std::pow(2, s * (nlevel - ilevel))));
    // std::cout  << "Volume : " << ilevel << "\t"<< vol << std::endl;
    // //std::cout  << "Stride : " << stride << "\t"<< vol << std::endl;

    for (int kfib = 0; kfib < nf; kfib += stride) {
      int kf = mgard::get_lindex(nf, nfib, kfib);
      int row_counter = 0;

      if (fib_counter % 2 == 0) {
        for (int irow = 0; irow < nr; irow += stride) {
          int ir = mgard::get_lindex(nr, nrow, irow);
          if (row_counter % 2 == 0) {
            for (int jcol = Cstride; jcol < nc; jcol += Cstride) {
              int jc = mgard::get_lindex(nc, ncol, jcol);
              Real val = v[mgard::get_index3(ncol, nfib, ir, jc - stride, kf)];
              const int n = quantizer(val);
              gzwrite(out_file, &n, sizeof(int));
              ++count;
              //                          outfile.write(reinterpret_cast<char*>(
              //                          &v[mgard::get_index3(ncol,
              //                          nfib, ir,jc - stride, kf)] ),
              //                          sizeof(Real) );
              //                  //std::cout  <<  v[irow][icol - stride] <<
              //                  "\t";
            }

          } else {
            for (int jcol = 0; jcol < nc; jcol += stride) {
              int jc = mgard::get_lindex(nc, ncol, jcol);
              Real val = v[mgard::get_index3(ncol, nfib, ir, jc, kf)];
              const int n = quantizer(val);
              gzwrite(out_file, &n, sizeof(int));
              ++count;
              //         outfile.write(reinterpret_cast<char*>(
              //         &v[mgard::get_index3(ncol, nfib, ir, jc, kf)] ),
              //         sizeof(Real) );
              //                  //std::cout  <<  v[irow][icol] << "\t";
            }
          }
          ++row_counter;
        }
      } else {
        for (int irow = 0; irow < nr; irow += stride) {
          int ir = mgard::get_lindex(nr, nrow, irow);
          for (int jcol = 0; jcol < nc; jcol += stride) {
            int jc = mgard::get_lindex(nc, ncol, jcol);
            Real val = v[mgard::get_index3(ncol, nfib, ir, jc, kf)];
            const int n = quantizer(val);
            gzwrite(out_file, &n, sizeof(int));
            ++count;
            //                      outfile.write(reinterpret_cast<char*>(
            //                      &v[mgard::get_index3(ncol, nfib, ir,
            //                      jc, kf)] ), sizeof(Real) );
          }
        }
      }
      ++fib_counter;
    }
  }

  // last level -> L=0
  {

    int stride = std::pow(2, nlevel);
    const Real dx = get_h_l(coords_x, nc, ncol, 0, stride);
    const Real dy = get_h_l(coords_y, nr, nrow, 0, stride);
    const Real dz = get_h_l(coords_z, nf, nfib, 0, stride);
    const Real vol = dx * dy * dz;

    // Square root of volume is scaled by `std::pow(2, 0)`.
    const mgard::LinearQuantizer<Real, int> quantizer(coeff / std::sqrt(vol));
    // //std::cout  << "Volume : " << nlevel << "\t"<< vol << std::endl;
    // //std::cout  << "Stride : " << stride << "\t"<< vol << std::endl;

    for (int irow = 0; irow < nr; irow += stride) {
      int ir = mgard::get_lindex(nr, nrow, irow);
      for (int jcol = 0; jcol < nc; jcol += stride) {
        int jc = mgard::get_lindex(nc, ncol, jcol);
        for (int kfib = 0; kfib < nf; kfib += stride) {
          int kf = mgard::get_lindex(nf, nfib, kfib);
          Real val = v[mgard::get_index3(ncol, nfib, ir, jc, kf)];
          const int n = quantizer(val);
          gzwrite(out_file, &n, sizeof(int));
          ++count;
        }
      }
    }
  }

  // //std::cout  << "Pruned : "<< prune_count << " Reduction : " << (Real)
  // nrow*ncol*nfib /(nrow*ncol*nfib - prune_count) << "\n";
  // std::cout  << "Wrote : "<< count << "\n";
  gzclose(out_file);
}

template <typename Real>
void copy_level_l(const int l, Real *v, Real *work, int nr, int nc, int nrow,
                  int ncol) {
  // work_l = v_l
  int stride = std::pow(2, l); // current stride

  for (int irow = 0; irow < nr; irow += stride) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < nc; jcol += stride) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      work[mgard::get_index(ncol, ir, jr)] = v[mgard::get_index(ncol, ir, jr)];
    }
  }
}

template <typename Real>
void subtract_level_l(const int l, Real *v, Real *work, int nr, int nc,
                      int nrow, int ncol) {
  // v -= work at level l

  int stride = std::pow(2, l); // current stride

  for (int irow = 0; irow < nr; irow += stride) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < nc; jcol += stride) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      v[mgard::get_index(ncol, ir, jr)] -= work[mgard::get_index(ncol, ir, jr)];
    }
  }
}

template <typename Real>
void pi_lminus1_l(const int l, std::vector<Real> &v,
                  const std::vector<Real> &coords, int n, int no) {
  int nlevel = static_cast<int>(std::log2(n - 1));
  int my_level = nlevel - l;
  int stride = std::pow(2, l); // current stride
  //  int Pstride = stride/2; //finer stride
  int Cstride = stride * 2; // coarser stride

  if (my_level != 0) {
    for (int i = Cstride; i < n - 1; i += Cstride) {
      Real h1 = get_h_l(coords, n, no, i - Cstride, stride);
      Real h2 = get_h_l(coords, n, no, i - stride, stride);
      Real hsum = h1 + h2;
      *get_ref(v, n, no, i - stride) -=
          (h1 * (*get_ref(v, n, no, i)) +
           h2 * (*get_ref(v, n, no, i - Cstride))) /
          hsum;
    }

    Real h1 = get_h_l(coords, n, no, n - 1 - Cstride, stride);
    Real h2 = get_h_l(coords, n, no, n - 1 - stride, stride);
    Real hsum = h1 + h2;
    *get_ref(v, n, no, n - 1 - stride) -=
        (h1 * (v.back()) + h2 * (*get_ref(v, n, no, n - 1 - Cstride))) / hsum;
  }
}

template <typename Real>
void pi_lminus1_first(std::vector<Real> &v, const std::vector<Real> &coords,
                      int n, int no) {

  for (int i = 0; i < n - 1; ++i) {
    int i_logic = mgard::get_lindex(n, no, i);
    int i_logicP = mgard::get_lindex(n, no, i + 1);

    if (i_logicP != i_logic + 1) {

      Real h1 = mgard_common::get_dist(coords, i_logic, i_logic + 1);
      Real h2 = mgard_common::get_dist(coords, i_logic + 1, i_logicP);
      Real hsum = h1 + h2;

      v[i_logic + 1] -= (h2 * v[i_logic] + h1 * v[i_logicP]) / hsum;
    }
  }
}

template <typename Real>
void pi_Ql_first(const int nr, const int nc, const int nrow, const int ncol,
                 const int l, Real *v, const std::vector<Real> &coords_x,
                 const std::vector<Real> &coords_y, std::vector<Real> &row_vec,
                 std::vector<Real> &col_vec) {
  // Restrict data to coarser level

  int stride = 1; // current stride
  //  int Pstride = stride/2; //finer stride
  int Cstride = 2; // coarser stride

  for (int irow = 0; irow < nr;
       irow += stride) // Do the rows existing  in the coarser level
  {
    int irow_r = mgard::get_lindex(
        nr, nrow, irow); // get the real location of logical index irow

    for (int jcol = 0; jcol < ncol; ++jcol) {
      // int jcol_r = mgard::get_lindex(nc, ncol, jcol);
      //            std::cerr <<  mgard::get_index(ncol, irow_r, jcol) <<
      //            "\n";

      row_vec[jcol] = v[mgard::get_index(ncol, irow_r, jcol)];
    }

    pi_lminus1_first(row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      //            int jcol_r = mgard::get_lindex(nc, ncol, jcol);
      v[mgard::get_index(ncol, irow_r, jcol)] = row_vec[jcol];
    }

    // if( irP != ir +1) //are we skipping the next row?
    //   {
    //     ++irow;
    //   }
  }

  if (nrow > 1) {
    for (int jcol = 0; jcol < nc;
         jcol += stride) // Do the columns existing  in the coarser level
    {
      int jcol_r = mgard::get_lindex(nc, ncol, jcol);
      int jr = mgard::get_lindex(nc, ncol, jcol);
      int jrP = mgard::get_lindex(nc, ncol, jcol + 1);

      for (int irow = 0; irow < nrow; ++irow) {
        int irow_r = mgard::get_lindex(nr, nrow, irow);
        col_vec[irow] = v[mgard::get_index(ncol, irow, jcol_r)];
      }

      pi_lminus1_first(col_vec, coords_y, nr, nrow);

      for (int irow = 0; irow < nrow; ++irow) {
        int irow_r = mgard::get_lindex(nr, nrow, irow);
        v[mgard::get_index(ncol, irow, jcol_r)] = col_vec[irow];
      }

      // if( jrP != jr +1) //are we skipping the next row?
      //   {
      //     ++jcol;
      //   }
    }
  }

  //        Now the new-new stuff
  for (int irow = 0; irow < nr - 1; ++irow) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    int irP = mgard::get_lindex(nr, nrow, irow + 1);

    for (int jcol = 0; jcol < nc - 1; ++jcol) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      int jrP = mgard::get_lindex(nc, ncol, jcol + 1);

      if ((irP != ir + 1) &&
          (jrP != jr + 1)) // we skipped both a row and a column
      {

        Real q11 = v[mgard::get_index(ncol, ir, jr)];
        Real q12 = v[mgard::get_index(ncol, irP, jr)];
        Real q21 = v[mgard::get_index(ncol, ir, jrP)];
        Real q22 = v[mgard::get_index(ncol, irP, jrP)];

        Real x1 = 0.0;
        Real y1 = 0.0;

        Real x2 = mgard_common::get_dist(coords_x, jr, jrP);
        Real y2 = mgard_common::get_dist(coords_y, ir, irP);

        Real x = mgard_common::get_dist(coords_x, jr, jr + 1);
        Real y = mgard_common::get_dist(coords_y, ir, ir + 1);

        Real temp =
            mgard::interpolate(q11, q12, q21, q22, x1, x2, y1, y2, x, y);

        v[mgard::get_index(ncol, ir + 1, jr + 1)] -= temp;
      }
    }
  }
}

template <typename Real>
void pi_Ql(const int nr, const int nc, const int nrow, const int ncol,
           const int l, Real *v, const std::vector<Real> &coords_x,
           const std::vector<Real> &coords_y, std::vector<Real> &row_vec,
           std::vector<Real> &col_vec) {
  // Restrict data to coarser level

  int stride = std::pow(2, l); // current stride
  //  int Pstride = stride/2; //finer stride
  int Cstride = stride * 2; // coarser stride

  //  std::vector<Real> row_vec(ncol), col_vec(nrow)   ;

  for (int irow = 0; irow < nr;
       irow += Cstride) // Do the rows existing  in the coarser level
  {
    int ir =
        mgard::get_lindex(nr, nrow,
                          irow); // get the real location of logical index irow
    for (int jcol = 0; jcol < ncol; ++jcol) {
      //            int jcol_r = mgard::get_lindex(nc, ncol, jcol);
      row_vec[jcol] = v[mgard::get_index(ncol, ir, jcol)];
    }

    //        mgard_cannon::pi_lminus1(l, row_vec, coords_x);
    pi_lminus1_l(l, row_vec, coords_x, nc, ncol);
    for (int jcol = 0; jcol < ncol; ++jcol) {
      v[mgard::get_index(ncol, ir, jcol)] = row_vec[jcol];
    }
  }

  if (nrow > 1) {
    for (int jcol = 0; jcol < nc;
         jcol += Cstride) // Do the columns existing  in the coarser level
    {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      for (int irow = 0; irow < nrow; ++irow) {
        //                int irow_r = mgard::get_lindex(nr, nrow, irow);
        col_vec[irow] = v[mgard::get_index(ncol, irow, jr)];
      }

      pi_lminus1_l(l, col_vec, coords_y, nr, nrow);
      for (int irow = 0; irow < nrow; ++irow) {
        //                int irow_r = mgard::get_lindex(nr, nrow, irow);
        v[mgard::get_index(ncol, irow, jr)] = col_vec[irow];
      }
    }

    // Now the new-new stuff
    for (int irow = stride; irow < nr; irow += Cstride) {
      int ir1 = mgard::get_lindex(nr, nrow, irow - stride);
      int ir = mgard::get_lindex(nr, nrow, irow);
      int ir2 = mgard::get_lindex(nr, nrow, irow + stride);

      for (int jcol = stride; jcol < nc; jcol += Cstride) {

        int jr1 = mgard::get_lindex(nc, ncol, jcol - stride);
        int jr = mgard::get_lindex(nc, ncol, jcol);
        int jr2 = mgard::get_lindex(nc, ncol, jcol + stride);

        Real q11 = v[mgard::get_index(ncol, ir1, jr1)];
        Real q12 = v[mgard::get_index(ncol, ir2, jr1)];
        Real q21 = v[mgard::get_index(ncol, ir1, jr2)];
        Real q22 = v[mgard::get_index(ncol, ir2, jr2)];

        Real x1 = 0.0; // relative coordinate axis centered at irow - Cstride,
                       // jcol - Cstride
        Real y1 = 0.0;
        Real x2 = mgard_common::get_dist(coords_x, jr1, jr2);
        Real y2 = mgard_common::get_dist(coords_y, ir1, ir2);

        Real x = mgard_common::get_dist(coords_x, jr1, jr);
        Real y = mgard_common::get_dist(coords_y, ir1, ir);
        Real temp =
            mgard::interpolate(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
        //              //std::cout  << temp <<"\n";
        v[mgard::get_index(ncol, ir, jr)] -= temp;
      }
    }
  }
}

template <typename Real>
void pi_Ql3D(const int nr, const int nc, const int nf, const int nrow,
             const int ncol, const int nfib, const int l, Real *v,
             const std::vector<Real> &coords_x,
             const std::vector<Real> &coords_y,
             const std::vector<Real> &coords_z, std::vector<Real> &row_vec,
             std::vector<Real> &col_vec, std::vector<Real> &fib_vec) {

  int stride = std::pow(2, l); // current stride
  //  int Pstride = stride/2; //finer stride
  int Cstride = stride * 2; // coarser stride

  //  std::vector<Real> row_vec(ncol), col_vec(nrow)   ;

  for (int kfib = 0; kfib < nf; kfib += Cstride) {
    int kf =
        mgard::get_lindex(nf, nfib,
                          kfib); // get the real location of logical index irow
    for (int irow = 0; irow < nr;
         irow += Cstride) // Do the rows existing  in the coarser level
    {
      int ir = mgard::get_lindex(
          nr, nrow,
          irow); // get the real location of logical index irow
      for (int jcol = 0; jcol < ncol; ++jcol) {
        row_vec[jcol] = v[mgard::get_index3(ncol, nfib, ir, jcol, kf)];
      }

      pi_lminus1_l(l, row_vec, coords_x, nc, ncol);

      for (int jcol = 0; jcol < ncol; ++jcol) {
        v[mgard::get_index3(ncol, nfib, ir, jcol, kf)] = row_vec[jcol];
      }
    }
  }

  if (nrow > 1) {
    for (int kfib = 0; kfib < nf; kfib += Cstride) {
      int kf = mgard::get_lindex(nf, nfib, kfib);
      for (int jcol = 0; jcol < nc;
           jcol += Cstride) // Do the columns existing  in the coarser level
      {
        int jr = mgard::get_lindex(nc, ncol, jcol);
        for (int irow = 0; irow < nrow; ++irow) {
          //                int irow_r = mgard::get_lindex(nr, nrow, irow);
          col_vec[irow] = v[mgard::get_index3(ncol, nfib, irow, jr, kf)];
        }
        pi_lminus1_l(l, col_vec, coords_y, nr, nrow);
        for (int irow = 0; irow < nrow; ++irow) {
          v[mgard::get_index3(ncol, nfib, irow, jr, kf)] = col_vec[irow];
        }
      }
    }
  }

  if (nfib > 1) {
    for (int irow = 0; irow < nr;
         irow += Cstride) // Do the columns existing  in the coarser level
    {
      int ir = mgard::get_lindex(
          nr, nrow,
          irow); // get the real location of logical index irow
      for (int jcol = 0; jcol < nc; jcol += Cstride) {
        int jr = mgard::get_lindex(nc, ncol, jcol);
        for (int kfib = 0; kfib < nfib; ++kfib) {
          fib_vec[kfib] = v[mgard::get_index3(ncol, nfib, ir, jr, kfib)];
        }
        pi_lminus1_l(l, fib_vec, coords_z, nf, nfib);
        for (int kfib = 0; kfib < nfib; ++kfib) {
          v[mgard::get_index3(ncol, nfib, ir, jr, kfib)] = fib_vec[kfib];
        }
      }
    }
  }

  //        Now the new-new stuff, xy-plane
  for (int kfib = 0; kfib < nf; kfib += Cstride) {
    int kf = mgard::get_lindex(nf, nfib, kfib);
    for (int irow = stride; irow < nr; irow += Cstride) {
      int ir1 = mgard::get_lindex(nr, nrow, irow - stride);
      int ir = mgard::get_lindex(nr, nrow, irow);
      int ir2 = mgard::get_lindex(nr, nrow, irow + stride);

      for (int jcol = stride; jcol < nc; jcol += Cstride) {

        int jr1 = mgard::get_lindex(nc, ncol, jcol - stride);
        int jr = mgard::get_lindex(nc, ncol, jcol);
        int jr2 = mgard::get_lindex(nc, ncol, jcol + stride);

        Real q11 = v[mgard::get_index3(ncol, nfib, ir1, jr1, kf)];
        Real q12 = v[mgard::get_index3(ncol, nfib, ir2, jr1, kf)];
        Real q21 = v[mgard::get_index3(ncol, nfib, ir1, jr2, kf)];
        Real q22 = v[mgard::get_index3(ncol, nfib, ir2, jr2, kf)];

        Real x1 = 0.0; // relative coordinate axis centered at irow - Cstride,
                       // jcol - Cstride
        Real y1 = 0.0;
        Real x2 = mgard_common::get_dist(coords_x, jr1, jr2);
        Real y2 = mgard_common::get_dist(coords_y, ir1, ir2);

        Real x = mgard_common::get_dist(coords_x, jr1, jr);
        Real y = mgard_common::get_dist(coords_y, ir1, ir);
        Real temp =
            mgard::interpolate(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
        //              //std::cout  << temp <<"\n";
        v[mgard::get_index3(ncol, nfib, ir, jr, kf)] -= temp;
      }
    }
  }

  // // //        Now the new-new stuff, xz-plane
  for (int irow = 0; irow < nr; irow += Cstride) {
    int irr = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = stride; jcol < nc; jcol += Cstride) {
      int ir1 = mgard::get_lindex(nc, ncol, jcol - stride);
      int ir = mgard::get_lindex(nc, ncol, jcol);
      int ir2 = mgard::get_lindex(nc, ncol, jcol + stride);

      for (int kfib = stride; kfib < nf; kfib += Cstride) {
        int jr1 = mgard::get_lindex(nf, nfib, kfib - stride);
        int jr = mgard::get_lindex(nf, nfib, kfib);
        int jr2 = mgard::get_lindex(nf, nfib, kfib + stride);

        Real q11 = v[mgard::get_index3(ncol, nfib, irr, ir1, jr1)];
        Real q12 = v[mgard::get_index3(ncol, nfib, irr, ir2, jr1)];
        Real q21 = v[mgard::get_index3(ncol, nfib, irr, ir1, jr2)];
        Real q22 = v[mgard::get_index3(ncol, nfib, irr, ir2, jr2)];

        Real x1 = 0.0; // relative coordinate axis centered at irow - Cstride,
                       // jcol - Cstride
        Real y1 = 0.0;
        Real x2 = mgard_common::get_dist(coords_z, jr1, jr2);
        Real y2 = mgard_common::get_dist(coords_x, ir1, ir2);

        Real x = mgard_common::get_dist(coords_z, jr1, jr);
        Real y = mgard_common::get_dist(coords_x, ir1, ir);
        Real temp =
            mgard::interpolate(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
        //              //std::cout  << temp <<"\n";
        v[mgard::get_index3(ncol, nfib, irr, ir, jr)] -= temp;
      }
    }
  }

  //     //        Now the new-new stuff, yz-plane
  for (int jcol = 0; jcol < nc; jcol += Cstride) {
    int jrr = mgard::get_lindex(nc, ncol, jcol);
    for (int irow = stride; irow < nr; irow += Cstride) {
      int ir1 = mgard::get_lindex(nr, nrow, irow - stride);
      int ir = mgard::get_lindex(nr, nrow, irow);
      int ir2 = mgard::get_lindex(nr, nrow, irow + stride);

      for (int kfib = stride; kfib < nf; kfib += Cstride) {
        int jr1 = mgard::get_lindex(nf, nfib, kfib - stride);
        int jr = mgard::get_lindex(nf, nfib, kfib);
        int jr2 = mgard::get_lindex(nf, nfib, kfib + stride);

        Real q11 = v[mgard::get_index3(ncol, nfib, ir1, jrr, jr1)];
        Real q12 = v[mgard::get_index3(ncol, nfib, ir2, jrr, jr1)];
        Real q21 = v[mgard::get_index3(ncol, nfib, ir1, jrr, jr2)];
        Real q22 = v[mgard::get_index3(ncol, nfib, ir2, jrr, jr2)];

        Real x1 = 0.0; // relative coordinate axis centered at irow - Cstride,
                       // jcol - Cstride
        Real y1 = 0.0;
        Real x2 = mgard_common::get_dist(coords_z, jr1, jr2);
        Real y2 = mgard_common::get_dist(coords_y, ir1, ir2);

        Real x = mgard_common::get_dist(coords_z, jr1, jr);
        Real y = mgard_common::get_dist(coords_y, ir1, ir);
        Real temp =
            mgard::interpolate(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
        //              //std::cout  << temp <<"\n";
        v[mgard::get_index3(ncol, nfib, ir, jrr, jr)] -= temp;
      }
    }
  }

  // ///    new-new-new stuff

  for (int irow = stride; irow < nr; irow += Cstride) {
    const int ir1 = mgard::get_lindex(nr, nrow, irow - stride);
    const int ir = mgard::get_lindex(nr, nrow, irow);
    const int ir2 = mgard::get_lindex(nr, nrow, irow + stride);

    for (int jcol = stride; jcol < nc; jcol += Cstride) {
      const int jr1 = mgard::get_lindex(nc, ncol, jcol - stride);
      const int jr = mgard::get_lindex(nc, ncol, jcol);
      const int jr2 = mgard::get_lindex(nc, ncol, jcol + stride);

      for (int kfib = stride; kfib < nf; kfib += Cstride) {

        const int kr1 = mgard::get_lindex(nf, nfib, kfib - stride);
        const int kr = mgard::get_lindex(nf, nfib, kfib);
        const int kr2 = mgard::get_lindex(nf, nfib, kfib + stride);

        const Real x1 = 0.0;
        const Real y1 = 0.0;
        const Real z1 = 0.0;

        const Real x2 = mgard_common::get_dist(coords_x, jr1, jr2);
        const Real y2 = mgard_common::get_dist(coords_y, ir1, ir2);
        const Real z2 = mgard_common::get_dist(coords_z, kr1, kr2);

        const Real x = mgard_common::get_dist(coords_x, jr1, jr);
        const Real y = mgard_common::get_dist(coords_y, ir1, ir);
        const Real z = mgard_common::get_dist(coords_z, kr1, kr);

        // The `q` indices are ordered '`x`, `y`, `z`.'
        const Real q000 = v[mgard::get_index3(ncol, nfib, ir1, jr1, kr1)];
        const Real q001 = v[mgard::get_index3(ncol, nfib, ir1, jr1, kr2)];
        const Real q010 = v[mgard::get_index3(ncol, nfib, ir2, jr1, kr1)];
        const Real q011 = v[mgard::get_index3(ncol, nfib, ir2, jr1, kr2)];
        const Real q100 = v[mgard::get_index3(ncol, nfib, ir1, jr2, kr1)];
        const Real q101 = v[mgard::get_index3(ncol, nfib, ir1, jr2, kr2)];
        const Real q110 = v[mgard::get_index3(ncol, nfib, ir2, jr2, kr1)];
        const Real q111 = v[mgard::get_index3(ncol, nfib, ir2, jr2, kr2)];

        v[mgard::get_index3(ncol, nfib, ir, jr, kr)] -=
            mgard::interpolate(q000, q001, q010, q011, q100, q101, q110, q111,
                               x1, x2, y1, y2, z1, z2, x, y, z);
      }
    }
  }
}

template <typename Real>
void pi_Ql3D_first(const int nr, const int nc, const int nf, const int nrow,
                   const int ncol, const int nfib, const int l, Real *v,
                   const std::vector<Real> &coords_x,
                   const std::vector<Real> &coords_y,
                   const std::vector<Real> &coords_z,
                   std::vector<Real> &row_vec, std::vector<Real> &col_vec,
                   std::vector<Real> &fib_vec) {

  int stride = 1; // current stride
  //  int Pstride = stride/2; //finer stride
  int Cstride = stride * 2; // coarser stride

  //  std::vector<Real> row_vec(ncol), col_vec(nrow)   ;

  for (int kfib = 0; kfib < nf; kfib += stride) {
    int kf =
        mgard::get_lindex(nf, nfib,
                          kfib); // get the real location of logical index irow
    for (int irow = 0; irow < nr;
         irow += stride) // Do the rows existing  in the coarser level
    {
      int ir = mgard::get_lindex(
          nr, nrow,
          irow); // get the real location of logical index irow
      for (int jcol = 0; jcol < ncol; ++jcol) {
        row_vec[jcol] = v[mgard::get_index3(ncol, nfib, ir, jcol, kf)];
      }

      pi_lminus1_first(row_vec, coords_x, nc, ncol);

      for (int jcol = 0; jcol < ncol; ++jcol) {
        v[mgard::get_index3(ncol, nfib, ir, jcol, kf)] = row_vec[jcol];
      }
    }
  }

  if (nrow > 1) {
    for (int kfib = 0; kfib < nf; kfib += stride) {
      int kf = mgard::get_lindex(nf, nfib, kfib);
      for (int jcol = 0; jcol < nc;
           jcol += stride) // Do the columns existing  in the coarser level
      {
        int jr = mgard::get_lindex(nc, ncol, jcol);
        for (int irow = 0; irow < nrow; ++irow) {
          //                int irow_r = mgard::get_lindex(nr, nrow, irow);
          col_vec[irow] = v[mgard::get_index3(ncol, nfib, irow, jr, kf)];
        }
        pi_lminus1_first(col_vec, coords_y, nr, nrow);
        for (int irow = 0; irow < nrow; ++irow) {
          v[mgard::get_index3(ncol, nfib, irow, jr, kf)] = col_vec[irow];
        }
      }
    }
  }

  if (nfib > 1) {
    for (int irow = 0; irow < nr;
         irow += stride) // Do the columns existing  in the coarser level
    {
      int ir = mgard::get_lindex(
          nr, nrow,
          irow); // get the real location of logical index irow
      for (int jcol = 0; jcol < nc; jcol += stride) {
        int jr = mgard::get_lindex(nc, ncol, jcol);
        for (int kfib = 0; kfib < nfib; ++kfib) {
          fib_vec[kfib] = v[mgard::get_index3(ncol, nfib, ir, jr, kfib)];
        }
        pi_lminus1_first(fib_vec, coords_z, nf, nfib);
        for (int kfib = 0; kfib < nfib; ++kfib) {
          v[mgard::get_index3(ncol, nfib, ir, jr, kfib)] = fib_vec[kfib];
        }
      }
    }
  }

  //        Now the new-new stuff, xy-plane
  for (int kfib = 0; kfib < nf; kfib += stride) {
    int kf = mgard::get_lindex(nf, nfib, kfib);
    for (int irow = 0; irow < nr - 1; irow += stride) {
      int ir = mgard::get_lindex(nr, nrow, irow);
      int irP = mgard::get_lindex(nr, nrow, irow + stride);

      for (int jcol = 0; jcol < nc - 1; jcol += stride) {

        int jr = mgard::get_lindex(nc, ncol, jcol);
        int jrP = mgard::get_lindex(nc, ncol, jcol + stride);

        if ((irP != ir + 1) &&
            (jrP != jr + 1)) // we skipped both a row and a column
        {
          Real q11 = v[mgard::get_index3(ncol, nfib, ir, jr, kf)];
          Real q12 = v[mgard::get_index3(ncol, nfib, irP, jr, kf)];
          Real q21 = v[mgard::get_index3(ncol, nfib, ir, jrP, kf)];
          Real q22 = v[mgard::get_index3(ncol, nfib, irP, jrP, kf)];

          Real x1 = 0.0; // relative coordinate axis centered at irow -
                         // Cstride, jcol - Cstride
          Real y1 = 0.0;
          Real x2 = mgard_common::get_dist(coords_x, jr, jrP);
          Real y2 = mgard_common::get_dist(coords_y, ir, irP);

          Real x = mgard_common::get_dist(coords_x, jr, jr + 1);
          Real y = mgard_common::get_dist(coords_y, ir, ir + 1);
          Real temp =
              mgard::interpolate(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
          //              //std::cout  << temp <<"\n";
          v[mgard::get_index3(ncol, nfib, ir + 1, jr + 1, kf)] -= temp;
        }
      }
    }
  }

  // // //        Now the new-new stuff, xz-plane
  for (int irow = 0; irow < nr; irow += stride) {
    int irr = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < nc - 1; jcol += stride) {
      int ir = mgard::get_lindex(nc, ncol, jcol);
      int irP = mgard::get_lindex(nc, ncol, jcol + stride);

      for (int kfib = 0; kfib < nf - 1; kfib += stride) {

        int jr = mgard::get_lindex(nf, nfib, kfib);
        int jrP = mgard::get_lindex(nf, nfib, kfib + stride);

        if ((irP != ir + 1) &&
            (jrP != jr + 1)) // we skipped both a row and a column
        {

          Real q11 = v[mgard::get_index3(ncol, nfib, irr, ir, jr)];
          Real q12 = v[mgard::get_index3(ncol, nfib, irr, irP, jr)];
          Real q21 = v[mgard::get_index3(ncol, nfib, irr, ir, jrP)];
          Real q22 = v[mgard::get_index3(ncol, nfib, irr, irP, jrP)];

          Real x1 = 0.0; // relative coordinate axis centered at irow -
                         // Cstride, jcol - Cstride
          Real y1 = 0.0;
          Real x2 = mgard_common::get_dist(coords_z, jr, jrP);
          Real y2 = mgard_common::get_dist(coords_x, ir, irP);

          Real x = mgard_common::get_dist(coords_z, jr, jr + 1);
          Real y = mgard_common::get_dist(coords_x, ir, ir + 1);
          Real temp =
              mgard::interpolate(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
          //              //std::cout  << temp <<"\n";
          v[mgard::get_index3(ncol, nfib, irr, ir + 1, jr + 1)] -= temp;
        }
      }
    }
  }

  //     //        Now the new-new stuff, yz-plane
  for (int jcol = 0; jcol < nc; jcol += stride) {
    int jrr = mgard::get_lindex(nc, ncol, jcol);
    for (int irow = 0; irow < nr - 1; irow += stride) {
      int ir = mgard::get_lindex(nr, nrow, irow);
      int irP = mgard::get_lindex(nr, nrow, irow + stride);
      for (int kfib = 0; kfib < nf - 1; kfib += stride) {
        int jr = mgard::get_lindex(nf, nfib, kfib);
        int jrP = mgard::get_lindex(nf, nfib, kfib + stride);

        if ((irP != ir + 1) &&
            (jrP != jr + 1)) // we skipped both a row and a column
        {
          Real q11 = v[mgard::get_index3(ncol, nfib, ir, jrr, jr)];
          Real q12 = v[mgard::get_index3(ncol, nfib, irP, jrr, jr)];
          Real q21 = v[mgard::get_index3(ncol, nfib, ir, jrr, jrP)];
          Real q22 = v[mgard::get_index3(ncol, nfib, irP, jrr, jrP)];

          Real x1 = 0.0; // relative coordinate axis centered at irow -
                         // Cstride, jcol - Cstride
          Real y1 = 0.0;
          Real x2 = mgard_common::get_dist(coords_z, jr, jrP);
          Real y2 = mgard_common::get_dist(coords_y, ir, irP);

          Real x = mgard_common::get_dist(coords_z, jr, jr + 1);
          Real y = mgard_common::get_dist(coords_y, ir, ir + 1);
          Real temp =
              mgard::interpolate(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
          //              //std::cout  << temp <<"\n";
          v[mgard::get_index3(ncol, nfib, ir + 1, jrr, jr + 1)] -= temp;
        }
      }
    }
  }

  ///    new-new-new stuff

  for (int irow = 0; irow < nr - 1; irow += stride) {
    const int ir = mgard::get_lindex(nr, nrow, irow);
    const int irP = mgard::get_lindex(nr, nrow, irow + stride);

    for (int jcol = 0; jcol < nc - 1; jcol += stride) {
      const int jr = mgard::get_lindex(nc, ncol, jcol);
      const int jrP = mgard::get_lindex(nc, ncol, jcol + stride);

      for (int kfib = 0; kfib < nf - 1; kfib += stride) {
        const int kr = mgard::get_lindex(nf, nfib, kfib);
        const int krP = mgard::get_lindex(nf, nfib, kfib + stride);

        if ((irP != ir + 1) && (jrP != jr + 1) &&
            (krP != kr + 1)) // we skipped both a row and a column
        {
          const Real x1 = 0.0;
          const Real y1 = 0.0;
          const Real z1 = 0.0;

          const Real x2 = mgard_common::get_dist(coords_x, jr, jrP);
          const Real y2 = mgard_common::get_dist(coords_y, ir, irP);
          const Real z2 = mgard_common::get_dist(coords_z, kr, krP);

          const Real x = mgard_common::get_dist(coords_x, jr, jr + 1);
          const Real y = mgard_common::get_dist(coords_y, ir, ir + 1);
          const Real z = mgard_common::get_dist(coords_z, kr, kr + 1);

          // The `q` indices are ordered '`x`, `y`, `z`.'
          const Real q000 = v[mgard::get_index3(ncol, nfib, ir, jr, kr)];
          const Real q001 = v[mgard::get_index3(ncol, nfib, ir, jr, krP)];
          const Real q010 = v[mgard::get_index3(ncol, nfib, irP, jr, kr)];
          const Real q011 = v[mgard::get_index3(ncol, nfib, irP, jr, krP)];
          const Real q100 = v[mgard::get_index3(ncol, nfib, ir, jrP, kr)];
          const Real q101 = v[mgard::get_index3(ncol, nfib, ir, jrP, krP)];
          const Real q110 = v[mgard::get_index3(ncol, nfib, irP, jrP, kr)];
          const Real q111 = v[mgard::get_index3(ncol, nfib, irP, jrP, krP)];

          v[mgard::get_index3(ncol, nfib, ir + 1, jr + 1, kr + 1)] -=
              mgard::interpolate(q000, q001, q010, q011, q100, q101, q110, q111,
                                 x1, x2, y1, y2, z1, z2, x, y, z);
        }
      }
    }
  }
}

template <typename Real>
void assign_num_level(const int l, std::vector<Real> &v, Real num, int n,
                      int no) {
  // int stride = std::pow(2,l);//current stride
  // for (int i = 0; i< n-1; i += stride)
  //   {
  //    *get_ref(v, n, no, i) = num;
  //   }
  // v.back() = num;
  int stride = std::pow(2, l);
  for (int i = 0; i < n; i += stride) {
    int il = mgard::get_lindex(n, no, i);
    v[il] = num;
  }
}

template <typename Real>
void assign_num_level_l(const int l, Real *v, Real num, int nr, int nc,
                        const int nrow, const int ncol) {
  // set the value of nodal values at level l to number num

  int stride = std::pow(2, l); // current stride

  for (int irow = 0; irow < nr; irow += stride) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < nc; jcol += stride) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      v[mgard::get_index(ncol, ir, jr)] = num;
    }
  }
}

template <typename Real>
void restriction_first(std::vector<Real> &v, const std::vector<Real> &coords,
                       const int n, const int no) {
  // calculate the result of restrictionion

  for (int i = 0; i < n - 1; ++i) // loop over the logical array
  {
    int i_logic = mgard::get_lindex(n, no, i);
    int i_logicP = mgard::get_lindex(n, no, i + 1);

    if (i_logicP != i_logic + 1) // next real memory location was jumped over,
                                 // so need to restriction
    {
      Real h1 = mgard_common::get_h(coords, i_logic, 1);
      Real h2 = mgard_common::get_h(coords, i_logic + 1, 1);
      Real hsum = h1 + h2;
      // v[i_logic]  = 0.5*v[i_logic]  + 0.5*h2*v[i_logic+1]/hsum;
      // v[i_logicP] = 0.5*v[i_logicP] + 0.5*h1*v[i_logic+1]/hsum;
      v[i_logic] += h2 * v[i_logic + 1] / hsum;
      v[i_logicP] += h1 * v[i_logic + 1] / hsum;
    }
  }
}

template <typename Real>
void solve_tridiag_M_l(const int l, std::vector<Real> &v,
                       const std::vector<Real> &coords, const int n,
                       const int no) {

  //  int my_level = nlevel - l;
  int stride = std::pow(2, l); // current stride

  Real am, bm, h1, h2;
  am = 2.0 *
       get_h_l(coords, n, no, 0, stride); // first element of upper diagonal U.

  //    bm = get_h(coords, 0, stride) / am;
  bm = get_h_l(coords, n, no, 0, stride) / am;
  int nlevel = static_cast<int>(std::log2(n - 1));
  //    //std::cout  << nlevel;
  int nc = std::pow(2, nlevel - l) + 1;
  std::vector<Real> coeff(nc);
  int counter = 1;
  coeff.front() = am;

  // forward sweep
  for (int i = stride; i < n - 1; i += stride) {
    h1 = get_h_l(coords, n, no, i - stride, stride);
    h2 = get_h_l(coords, n, no, i, stride);

    *get_ref(v, n, no, i) -= *get_ref(v, n, no, i - stride) * bm;

    am = 2.0 * (h1 + h2) - bm * h1;
    bm = h2 / am;

    coeff.at(counter) = am;
    ++counter;
  }

  h2 = get_h_l(coords, n, no, n - 1 - stride, stride);
  am = 2.0 * h2 - bm * h2;

  //    *get_ref(v, n, no, n-1) -= *get_ref(v, n, no, n-1-stride)*bm;
  v.back() -= *get_ref(v, n, no, n - 1 - stride) * bm;
  coeff.at(counter) = am;

  // backward sweep

  //    *get_ref(v, n, no, n-1) /= am;
  v.back() /= am;
  --counter;

  for (int i = n - 1 - stride; i >= 0; i -= stride) {
    h2 = get_h_l(coords, n, no, i, stride);
    *get_ref(v, n, no, i) =
        (*get_ref(v, n, no, i) - h2 * (*get_ref(v, n, no, i + stride))) /
        coeff.at(counter);

    //        *get_ref(v, n, no, i) = 3  ;

    --counter;
  }
}

template <typename Real>
void add_level_l(const int l, Real *v, Real *work, int nr, int nc, int nrow,
                 int ncol) {
  // v += work at level l

  int stride = std::pow(2, l); // current stride

  for (int irow = 0; irow < nr; irow += stride) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < nc; jcol += stride) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      v[mgard::get_index(ncol, ir, jr)] += work[mgard::get_index(ncol, ir, jr)];
    }
  }
}

template <typename Real>
void add3_level_l(const int l, Real *v, Real *work, int nr, int nc, int nf,
                  int nrow, int ncol, int nfib) {
  // v += work at level l

  int stride = std::pow(2, l); // current stride

  for (int irow = 0; irow < nr; irow += stride) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < nc; jcol += stride) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      for (int kfib = 0; kfib < nf; kfib += stride) {
        int kr = mgard::get_lindex(nf, nfib, kfib);
        v[mgard::get_index3(ncol, nfib, ir, jr, kr)] +=
            work[mgard::get_index3(ncol, nfib, ir, jr, kr)];
      }
    }
  }
}

template <typename Real>
void sub3_level_l(const int l, Real *v, Real *work, int nr, int nc, int nf,
                  int nrow, int ncol, int nfib) {
  // v += work at level l

  int stride = std::pow(2, l); // current stride

  for (int irow = 0; irow < nr; irow += stride) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < nc; jcol += stride) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      for (int kfib = 0; kfib < nf; kfib += stride) {
        int kr = mgard::get_lindex(nf, nfib, kfib);
        v[mgard::get_index3(ncol, nfib, ir, jr, kr)] -=
            work[mgard::get_index3(ncol, nfib, ir, jr, kr)];
      }
    }
  }
}

template <typename Real>
void sub3_level(const int l, Real *v, Real *work, int nrow, int ncol,
                int nfib) {
  // v += work at level l

  int stride = std::pow(2, l); // current stride

  for (int irow = 0; irow < nrow; irow += stride) {
    for (int jcol = 0; jcol < ncol; jcol += stride) {
      for (int kfib = 0; kfib < nfib; kfib += stride) {
        v[mgard::get_index3(ncol, nfib, irow, jcol, kfib)] -=
            work[mgard::get_index3(ncol, nfib, irow, jcol, kfib)];
      }
    }
  }
}

template <typename Real>
void sub_level_l(const int l, Real *v, Real *work, int nr, int nc, int nf,
                 int nrow, int ncol, int nfib) {
  // v += work at level l

  int stride = std::pow(2, l); // current stride

  for (int irow = 0; irow < nr; irow += stride) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < nc; jcol += stride) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      for (int kfib = 0; kfib < nf; kfib += stride) {
        int kr = mgard::get_lindex(nf, nfib, kfib);
        v[mgard::get_index3(ncol, nfib, ir, jr, kr)] -=
            work[mgard::get_index3(ncol, nfib, ir, jr, kr)];
      }
    }
  }
}

template <typename Real>
void prep_2D(const int nr, const int nc, const int nrow, const int ncol,
             const int l_target, Real *v, std::vector<Real> &work,
             std::vector<Real> &coords_x, std::vector<Real> &coords_y,
             std::vector<Real> &row_vec, std::vector<Real> &col_vec) {
  int l = 0;
  int stride = 1;

  pi_Ql_first(nr, nc, nrow, ncol, l, v, coords_x, coords_y, row_vec,
              col_vec); //(I-\Pi u) this is the initial move to 2^k+1 nodes

  mgard_cannon::copy_level(nrow, ncol, l, v, work);
  mgard_gen::assign_num_level_l(0, work.data(), static_cast<Real>(0.0), nr, nc,
                                nrow, ncol);

  // row-sweep
  for (int irow = 0; irow < nrow; ++irow) {
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[mgard::get_index(ncol, irow, jcol)];
    }

    mgard_cannon::mass_matrix_multiply(0, row_vec, coords_x);

    restriction_first(row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[mgard::get_index(ncol, irow, jcol)] = row_vec[jcol];
    }
  }

  for (int irow = 0; irow < nr; ++irow) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[mgard::get_index(ncol, ir, jcol)];
    }
    solve_tridiag_M_l(0, row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[mgard::get_index(ncol, ir, jcol)] = row_vec[jcol];
    }
  }

  //      //std::cout  << "Row sweep done!"<<"\n";

  // column-sweep
  if (nrow > 1) // do this if we have an 2-dimensional array
  {
    for (int jcol = 0; jcol < ncol; jcol += stride) {
      // int jr  = mgard::get_lindex(nc, ncol, jcol);
      for (int irow = 0; irow < nrow; ++irow) {
        col_vec[irow] = work[mgard::get_index(ncol, irow, jcol)];
      }

      mgard_cannon::mass_matrix_multiply(0, col_vec, coords_y);

      restriction_first(col_vec, coords_y, nr, nrow);

      for (int irow = 0; irow < nrow; ++irow) {
        work[mgard::get_index(ncol, irow, jcol)] = col_vec[irow];
      }
    }

    for (int jcol = 0; jcol < nc; ++jcol) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      for (int irow = 0; irow < nrow; ++irow) {
        col_vec[irow] = work[mgard::get_index(ncol, irow, jr)];
      }
      solve_tridiag_M_l(0, col_vec, coords_y, nr, nrow);

      for (int irow = 0; irow < nrow; ++irow) {
        work[mgard::get_index(ncol, irow, jr)] = col_vec[irow];
      }
    }
  }
  // Solved for (z_l, phi_l) = (c_{l+1}, vl)
  add_level_l(0, v, work.data(), nr, nc, nrow, ncol);
}

template <typename Real>
void mass_mult_l(const int l, std::vector<Real> &v,
                 const std::vector<Real> &coords, const int n, const int no) {

  int stride = std::pow(2, l);
  Real temp1, temp2;
  Real h1, h2;

  // Mass matrix times nodal value-vec
  temp1 = v.front(); // save u(0) for later use

  h1 = get_h_l(coords, n, no, 0, stride);

  v.front() = 2.0 * h1 * temp1 + h1 * (*get_ref(v, n, no, stride));

  for (int i = stride; i <= n - 1 - stride; i += stride) {
    temp2 = *get_ref(v, n, no, i);
    h1 = get_h_l(coords, n, no, i - stride, stride);
    h2 = get_h_l(coords, n, no, i, stride);

    *get_ref(v, n, no, i) = h1 * temp1 + 2 * (h1 + h2) * temp2 +
                            h2 * (*get_ref(v, n, no, i + stride));
    temp1 = temp2; // save u(n) for later use
  }
  v.back() = get_h_l(coords, n, no, n - stride - 1, stride) * temp1 +
             2 * get_h_l(coords, n, no, n - stride - 1, stride) * v.back();
}

template <typename Real>
void restriction_l(const int l, std::vector<Real> &v,
                   const std::vector<Real> &coords, const int n, const int no) {
  int stride = std::pow(2, l);
  int Pstride = stride / 2; // finer stride

  // calculate the result of restrictionion

  Real h1 = get_h_l(coords, n, no, 0, Pstride);
  Real h2 = get_h_l(coords, n, no, Pstride, Pstride);
  Real hsum = h1 + h2;

  v.front() += h2 * (*get_ref(v, n, no, Pstride)) / hsum; // first element

  for (int i = stride; i <= n - stride; i += stride) {
    *get_ref(v, n, no, i) += h1 * (*get_ref(v, n, no, i - Pstride)) / hsum;
    h1 = get_h_l(coords, n, no, i, Pstride);
    h2 = get_h_l(coords, n, no, i + Pstride, Pstride);
    hsum = h1 + h2;
    *get_ref(v, n, no, i) += h2 * (*get_ref(v, n, no, i + Pstride)) / hsum;
  }
  v.back() += h1 * (*get_ref(v, n, no, n - Pstride - 1)) / hsum; // last element
}

template <typename Real>
Real ml2_norm3(const int l, const int nr, const int nc, const int nf,
               const int nrow, const int ncol, const int nfib,
               const std::vector<Real> &v, const std::vector<Real> &coords_x,
               const std::vector<Real> &coords_y,
               const std::vector<Real> &coords_z) {

  int stride = std::pow(2, l);
  int Cstride = stride;
  std::vector<Real> work(v);
  std::vector<Real> row_vec(ncol), col_vec(nrow), fib_vec(nfib);

  for (int kfib = 0; kfib < nf; kfib += stride) {
    int kf = mgard::get_lindex(nf, nfib, kfib);
    for (int irow = 0; irow < nr; irow += stride) {
      int ir = mgard::get_lindex(nr, nrow, irow);
      for (int jcol = 0; jcol < ncol; ++jcol) {
        row_vec[jcol] = work[mgard::get_index3(ncol, nfib, ir, jcol, kf)];
      }
      mgard_gen::mass_mult_l(l, row_vec, coords_x, nc, ncol);
      for (int jcol = 0; jcol < ncol; ++jcol) {
        work[mgard::get_index3(ncol, nfib, ir, jcol, kf)] = row_vec[jcol];
      }
    }
  }

  for (int kfib = 0; kfib < nf; kfib += stride) {
    int kf = mgard::get_lindex(nf, nfib, kfib);
    for (int jcol = 0; jcol < nc; jcol += stride) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      for (int irow = 0; irow < nrow; ++irow) {
        col_vec[irow] = work[mgard::get_index3(ncol, nfib, irow, jr, kf)];
      }
      mgard_gen::mass_mult_l(l, col_vec, coords_y, nr, nrow);
      for (int irow = 0; irow < nrow; ++irow) {
        work[mgard::get_index3(ncol, nfib, irow, jr, kf)] = col_vec[irow];
      }
    }
  }

  for (int irow = 0; irow < nr; irow += stride) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < nc; jcol += stride) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      for (int kfib = 0; kfib < nfib; ++kfib) {
        fib_vec[kfib] = work[mgard::get_index3(ncol, nfib, ir, jr, kfib)];
      }
      mgard_gen::mass_mult_l(l, fib_vec, coords_z, nf, nfib);
      for (int kfib = 0; kfib < nfib; ++kfib) {
        work[mgard::get_index3(ncol, nfib, ir, jr, kfib)] = fib_vec[kfib];
      }
    }
  }

  //    Real norm = ( std::inner_product( v.begin(), v.end(), work.begin(),
  //    0.0d)  );

  Real norm = 0;
  for (int irow = 0; irow < nr; irow += stride) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < nc; jcol += stride) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      for (int kfib = 0; kfib < nf; kfib += stride) {
        int kf = mgard::get_lindex(nf, nfib, kfib);

        norm += work[mgard::get_index3(ncol, nfib, ir, jr, kf)] *
                v[mgard::get_index3(ncol, nfib, ir, jr, kf)];
      }
    }
  }

  return norm / 216.0; // account for missing 1/6 factors in M_{x,y,z}
}

template <typename Real>
void prolongate_l(const int l, std::vector<Real> &v, std::vector<Real> &coords,
                  int n, int no) {

  int stride = std::pow(2, l);
  int Pstride = stride / 2;

  for (int i = stride; i < n; i += stride) {
    Real h1 = get_h_l(coords, n, no, i - stride, Pstride);
    Real h2 = get_h_l(coords, n, no, i - Pstride, Pstride);
    Real hsum = h1 + h2;

    *get_ref(v, n, no, i - Pstride) =
        (h2 * (*get_ref(v, n, no, i - stride)) + h1 * (*get_ref(v, n, no, i))) /
        hsum;
  }

  // Real h1 = get_h_l(coords, n, no, n-1-stride,  Pstride);
  // Real h2 = get_h_l(coords, n, no, n-1-Pstride, Pstride);
  // Real hsum = h1+h2;

  // *get_ref(v, n,  no,  n-1-Pstride) = ( h2*(*get_ref(v, n,  no,  n-1-stride))
  // + h1*(v.back()) )/hsum;
}

template <typename Real>
void refactor_2D(const int nr, const int nc, const int nrow, const int ncol,
                 const int l_target, Real *v, std::vector<Real> &work,
                 std::vector<Real> &coords_x, std::vector<Real> &coords_y,
                 std::vector<Real> &row_vec, std::vector<Real> &col_vec) {
  // refactor

  // for (int l = l_target; l < l_target + 1; ++l)
  //   {
  int l = l_target;
  int stride = std::pow(2, l); // current stride
  int Cstride = stride * 2;    // coarser stride

  // -> change funcs in pi_QL to use _l functions, otherwise distances are
  // wrong!!!
  //       pi_Ql(nr, nc, nrow, ncol, l, v, coords_x, coords_y, row_vec,
  //       col_vec); //rename!. v@l has I-\Pi_l Q_l+1 u

  // copy_level_l(l,  v,  work.data(),  nr,  nc,  nrow,  ncol);
  // assign_num_level_l(l+1, work.data(),  0.0,  nr,  nc,  nrow,  ncol);

  // row-sweep
  for (int irow = 0; irow < nr; ++irow) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[mgard::get_index(ncol, ir, jcol)];
    }

    mgard_gen::mass_mult_l(l, row_vec, coords_x, nc, ncol);

    mgard_gen::restriction_l(l + 1, row_vec, coords_x, nc, ncol);

    mgard_gen::solve_tridiag_M_l(l + 1, row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[mgard::get_index(ncol, ir, jcol)] = row_vec[jcol];
    }
  }

  // column-sweep
  if (nrow > 1) // do this if we have an 2-dimensional array
  {
    for (int jcol = 0; jcol < nc; jcol += Cstride) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      for (int irow = 0; irow < nrow; ++irow) {
        col_vec[irow] = work[mgard::get_index(ncol, irow, jr)];
      }

      mgard_gen::mass_mult_l(l, col_vec, coords_y, nr, nrow);
      mgard_gen::restriction_l(l + 1, col_vec, coords_y, nr, nrow);
      mgard_gen::solve_tridiag_M_l(l + 1, col_vec, coords_y, nr, nrow);

      for (int irow = 0; irow < nrow; ++irow) {
        work[mgard::get_index(ncol, irow, jr)] = col_vec[irow];
      }
    }
  }

  // Solved for (z_l, phi_l) = (c_{l+1}, vl)
  //        add_level_l(l+1, v, work.data(),  nr,  nc,  nrow,  ncol);
  //}
}

template <typename Real>
void refactor_2D_first(const int nr, const int nc, const int nrow,
                       const int ncol, const int l_target, Real *v,
                       std::vector<Real> &work, std::vector<Real> &coords_x,
                       std::vector<Real> &coords_y, std::vector<Real> &row_vec,
                       std::vector<Real> &col_vec) {
  // refactor

  for (int irow = 0; irow < nrow; ++irow) {
    //        int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[mgard::get_index(ncol, irow, jcol)];
    }

    mgard_cannon::mass_matrix_multiply(0, row_vec, coords_x);

    restriction_first(row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[mgard::get_index(ncol, irow, jcol)] = row_vec[jcol];
    }
  }

  for (int irow = 0; irow < nr; ++irow) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[mgard::get_index(ncol, ir, jcol)];
    }

    mgard_gen::solve_tridiag_M_l(0, row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[mgard::get_index(ncol, ir, jcol)] = row_vec[jcol];
    }
  }

  //   //   //std::cout  << "recomposing-colsweep" << "\n";

  //     // column-sweep, this is the slow one! Need something like column_copy
  if (nrow > 1) // check if we have 1-D array..
  {
    for (int jcol = 0; jcol < ncol; ++jcol) {
      //      int jr  = mgard::get_lindex(nc,  ncol,  jcol);
      for (int irow = 0; irow < nrow; ++irow) {
        col_vec[irow] = work[mgard::get_index(ncol, irow, jcol)];
      }

      mgard_cannon::mass_matrix_multiply(0, col_vec, coords_y);

      mgard_gen::restriction_first(col_vec, coords_y, nr, nrow);

      for (int irow = 0; irow < nrow; ++irow) {
        work[mgard::get_index(ncol, irow, jcol)] = col_vec[irow];
      }
    }

    for (int jcol = 0; jcol < nc; ++jcol) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      for (int irow = 0; irow < nrow; ++irow) {
        col_vec[irow] = work[mgard::get_index(ncol, irow, jr)];
      }

      mgard_gen::solve_tridiag_M_l(0, col_vec, coords_y, nr, nrow);
      for (int irow = 0; irow < nrow; ++irow) {
        work[mgard::get_index(ncol, irow, jr)] = col_vec[irow];
      }
    }
  }
}

template <typename Real>
void copy3_level_l(const int l, Real const *const v, Real *const work,
                   const int nr, const int nc, const int nf, const int nrow,
                   const int ncol, const int nfib) {
  // work_l = v_l
  int stride = std::pow(2, l); // current stride

  for (int irow = 0; irow < nr; irow += stride) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < nc; jcol += stride) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      for (int kfib = 0; kfib < nf; kfib += stride) {
        int kr = mgard::get_lindex(nf, nfib, kfib);
        work[mgard::get_index3(ncol, nfib, ir, jr, kr)] =
            v[mgard::get_index3(ncol, nfib, ir, jr, kr)];
      }
    }
  }
}

template <typename Real>
void copy3_level(const int l, Real const *const v, Real *const work,
                 const int nrow, const int ncol, const int nfib) {
  // work_l = v_l
  int stride = std::pow(2, l); // current stride

  for (int irow = 0; irow < nrow; irow += stride) {
    for (int jcol = 0; jcol < ncol; jcol += stride) {
      for (int kfib = 0; kfib < nfib; kfib += stride) {
        work[mgard::get_index3(ncol, nfib, irow, jcol, kfib)] =
            v[mgard::get_index3(ncol, nfib, irow, jcol, kfib)];
      }
    }
  }
}

template <typename Real>
void assign3_level_l(const int l, Real *v, Real num, int nr, int nc, int nf,
                     int nrow, int ncol, int nfib) {
  // work_l = v_l
  int stride = std::pow(2, l); // current stride

  for (int irow = 0; irow < nr; irow += stride) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < nc; jcol += stride) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      for (int kfib = 0; kfib < nf; kfib += stride) {
        int kr = mgard::get_lindex(nf, nfib, kfib);
        v[mgard::get_index3(ncol, nfib, ir, jr, kr)] = num;
      }
    }
  }
}

template <typename Real>
void refactor_3D(const int nr, const int nc, const int nf, const int nrow,
                 const int ncol, const int nfib, const int l_target, Real *v,
                 std::vector<Real> &work, std::vector<Real> &work2d,
                 std::vector<Real> &coords_x, std::vector<Real> &coords_y,
                 std::vector<Real> &coords_z) {

  std::vector<Real> v2d(nrow * ncol), fib_vec(nfib);
  std::vector<Real> row_vec(ncol);
  std::vector<Real> col_vec(nrow);

  for (int l = 0; l < l_target; ++l) {
    int stride = std::pow(2, l);
    int Cstride = 2 * stride;

    pi_Ql3D(nr, nc, nf, nrow, ncol, nfib, l, v, coords_x, coords_y, coords_z,
            row_vec, col_vec, fib_vec);

    mgard_gen::copy3_level_l(l, v, work.data(), nr, nc, nf, nrow, ncol, nfib);
    mgard_gen::assign3_level_l(l + 1, work.data(), static_cast<Real>(0.0), nr,
                               nc, nf, nrow, ncol, nfib);

    //       for (int kfib = 0; kfib < nfib; ++kfib)
    for (int kfib = 0; kfib < nf; kfib += stride) {
      //           int kf = kfib;
      int kf = mgard::get_lindex(
          nf, nfib,
          kfib); // get the real location of logical index irow
      mgard_common::copy_slice(work.data(), work2d, nrow, ncol, nfib, kf);
      mgard_gen::refactor_2D(nr, nc, nrow, ncol, l, v2d.data(), work2d,
                             coords_x, coords_y, row_vec, col_vec);
      mgard_common::copy_from_slice(work.data(), work2d, nrow, ncol, nfib, kf);
    }

    for (int irow = 0; irow < nr; irow += Cstride) {
      int ir = mgard::get_lindex(nr, nrow, irow);
      for (int jcol = 0; jcol < nc; jcol += Cstride) {
        int jc = mgard::get_lindex(nc, ncol, jcol);
        for (int kfib = 0; kfib < nfib; ++kfib) {
          fib_vec[kfib] = work[mgard::get_index3(ncol, nfib, ir, jc, kfib)];
        }
        mgard_gen::mass_mult_l(l, fib_vec, coords_z, nf, nfib);
        mgard_gen::restriction_l(l + 1, fib_vec, coords_z, nf, nfib);
        mgard_gen::solve_tridiag_M_l(l + 1, fib_vec, coords_z, nf, nfib);
        for (int kfib = 0; kfib < nfib; ++kfib) {
          work[mgard::get_index3(ncol, nfib, ir, jc, kfib)] = fib_vec[kfib];
        }
      }
    }

    add3_level_l(l + 1, v, work.data(), nr, nc, nf, nrow, ncol, nfib);
  }
}

template <typename Real>
void compute_zl(const int nr, const int nc, const int nrow, const int ncol,
                const int l_target, std::vector<Real> &work,
                std::vector<Real> &coords_x, std::vector<Real> &coords_y,
                std::vector<Real> &row_vec, std::vector<Real> &col_vec) {
  // recompose
  //    //std::cout  << "recomposing" << "\n";
  //    for (int l = l_target ; l > 0 ; --l)
  //  {
  int l = l_target;

  int stride = std::pow(2, l); // current stride
  int Pstride = stride / 2;

  //        copy_level_l(l-1,  v,  work.data(),  nr,  nc,  nrow,  ncol);

  // assign_num_level_l(l, work.data(),  0.0,  nr,  nc,  nrow,  ncol);

  //        //std::cout  << "recomposing-rowsweep" << "\n";
  //  l = 0;
  // row-sweep
  for (int irow = 0; irow < nr; irow += 1) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[mgard::get_index(ncol, ir, jcol)];
    }

    mgard_gen::mass_mult_l(l - 1, row_vec, coords_x, nc, ncol);

    mgard_gen::restriction_l(l, row_vec, coords_x, nc, ncol);

    mgard_gen::solve_tridiag_M_l(l, row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[mgard::get_index(ncol, ir, jcol)] = row_vec[jcol];
    }
  }

  //   //std::cout  << "recomposing-colsweep" << "\n";

  // column-sweep, this is the slow one! Need something like column_copy
  if (nrow > 1) // check if we have 1-D array..
  {
    for (int jcol = 0; jcol < nc; jcol += stride) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      for (int irow = 0; irow < nrow; ++irow) {
        col_vec[irow] = work[mgard::get_index(ncol, irow, jr)];
      }

      mgard_gen::mass_mult_l(l - 1, col_vec, coords_y, nr, nrow);

      mgard_gen::restriction_l(l, col_vec, coords_y, nr, nrow);

      mgard_gen::solve_tridiag_M_l(l, col_vec, coords_y, nr, nrow);

      for (int irow = 0; irow < nrow; ++irow) {
        work[mgard::get_index(ncol, irow, jr)] = col_vec[irow];
      }
    }
  }
}

template <typename Real>
void compute_zl_last(const int nr, const int nc, const int nrow, const int ncol,
                     const int l_target, std::vector<Real> &work,
                     std::vector<Real> &coords_x, std::vector<Real> &coords_y,
                     std::vector<Real> &row_vec, std::vector<Real> &col_vec) {

  for (int irow = 0; irow < nr; ++irow) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[mgard::get_index(ncol, ir, jcol)];
    }

    mgard_cannon::mass_matrix_multiply(0, row_vec, coords_x);

    restriction_first(row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[mgard::get_index(ncol, ir, jcol)] = row_vec[jcol];
    }
  }

  for (int irow = 0; irow < nr; ++irow) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[mgard::get_index(ncol, ir, jcol)];
    }

    mgard_gen::solve_tridiag_M_l(0, row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[mgard::get_index(ncol, ir, jcol)] = row_vec[jcol];
    }
  }

  //   //   //std::cout  << "recomposing-colsweep" << "\n";

  //     // column-sweep, this is the slow one! Need something like column_copy
  if (nrow > 1) // check if we have 1-D array..
  {
    for (int jcol = 0; jcol < ncol; ++jcol) {
      //      int jr  = mgard::get_lindex(nc,  ncol,  jcol);
      for (int irow = 0; irow < nrow; ++irow) {
        col_vec[irow] = work[mgard::get_index(ncol, irow, jcol)];
      }

      mgard_cannon::mass_matrix_multiply(0, col_vec, coords_y);

      mgard_gen::restriction_first(col_vec, coords_y, nr, nrow);

      for (int irow = 0; irow < nrow; ++irow) {
        work[mgard::get_index(ncol, irow, jcol)] = col_vec[irow];
      }
    }

    for (int jcol = 0; jcol < nc; ++jcol) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      for (int irow = 0; irow < nrow; ++irow) {
        col_vec[irow] = work[mgard::get_index(ncol, irow, jr)];
      }

      mgard_gen::solve_tridiag_M_l(0, col_vec, coords_y, nr, nrow);
      for (int irow = 0; irow < nrow; ++irow) {
        work[mgard::get_index(ncol, irow, jr)] = col_vec[irow];
      }
    }
  }
}

template <typename Real>
void prolongate_last(std::vector<Real> &v, std::vector<Real> &coords, int n,
                     int no) {
  // calculate the result of restrictionion

  for (int i = 0; i < n - 1; ++i) // loop over the logical array
  {
    int i_logic = mgard::get_lindex(n, no, i);
    int i_logicP = mgard::get_lindex(n, no, i + 1);

    if (i_logicP != i_logic + 1) // next real memory location was jumped over,
                                 // so need to restriction
    {
      Real h1 = mgard_common::get_h(coords, i_logic, 1);
      Real h2 = mgard_common::get_h(coords, i_logic + 1, 1);
      Real hsum = h1 + h2;
      v[i_logic + 1] = (h2 * v[i_logic] + h1 * v[i_logicP]) / hsum;
      //             v[i_logic+1] = 2*(h1*v[i_logicP])/hsum;
    }
  }
}

template <typename Real>
void prolong_add_2D(const int nr, const int nc, const int nrow, const int ncol,
                    const int l_target, std::vector<Real> &work,
                    std::vector<Real> &coords_x, std::vector<Real> &coords_y,
                    std::vector<Real> &row_vec, std::vector<Real> &col_vec) {
  int l = l_target;

  int stride = std::pow(2, l); // current stride
  int Pstride = stride / 2;

  for (int irow = 0; irow < nr; irow += stride) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[mgard::get_index(ncol, ir, jcol)];
    }

    mgard_gen::prolongate_l(l, row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[mgard::get_index(ncol, ir, jcol)] = row_vec[jcol];
    }
  }

  //   //std::cout  << "recomposing-colsweep2" << "\n";
  // column-sweep, this is the slow one! Need something like column_copy
  if (nrow > 1) {
    for (int jcol = 0; jcol < nc; jcol += Pstride) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      for (int irow = 0; irow < nrow; ++irow) // copy all rows
      {
        col_vec[irow] = work[mgard::get_index(ncol, irow, jr)];
      }

      mgard_gen::prolongate_l(l, col_vec, coords_y, nr, nrow);

      for (int irow = 0; irow < nrow; ++irow) {
        work[mgard::get_index(ncol, irow, jr)] = col_vec[irow];
      }
    }
  }
}

template <typename Real>
void prolong_add_2D_last(const int nr, const int nc, const int nrow,
                         const int ncol, const int l_target,
                         std::vector<Real> &work, std::vector<Real> &coords_x,
                         std::vector<Real> &coords_y,
                         std::vector<Real> &row_vec,
                         std::vector<Real> &col_vec) {
  int l = 0;

  int stride = 1;
  std::pow(2, l); // current stride
  //   int Pstride = stride/2;

  for (int irow = 0; irow < nr; ++irow) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[mgard::get_index(ncol, ir, jcol)];
    }

    mgard_gen::prolongate_last(row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[mgard::get_index(ncol, ir, jcol)] = row_vec[jcol];
    }
  }

  //   //   //std::cout  << "recomposing-colsweep2" << "\n";
  //     // column-sweep, this is the slow one! Need something like column_copy
  if (nrow > 1) {
    for (int jcol = 0; jcol < ncol; ++jcol) {
      // int jr  = mgard::get_lindex(nc,  ncol,  jcol);
      for (int irow = 0; irow < nrow; ++irow) // copy all rows
      {
        col_vec[irow] = work[mgard::get_index(ncol, irow, jcol)];
      }

      mgard_gen::prolongate_last(col_vec, coords_y, nr, nrow);

      for (int irow = 0; irow < nrow; ++irow) {
        work[mgard::get_index(ncol, irow, jcol)] = col_vec[irow];
      }
    }
  }
}

template <typename Real>
void prep_3D(const int nr, const int nc, const int nf, const int nrow,
             const int ncol, const int nfib, const int l_target, Real *v,
             std::vector<Real> &work, std::vector<Real> &work2d,
             std::vector<Real> &coords_x, std::vector<Real> &coords_y,
             std::vector<Real> &coords_z) {
  int l = 0;
  int stride = 1;

  std::vector<Real> v2d(nrow * ncol), fib_vec(nfib);
  std::vector<Real> row_vec(ncol);
  std::vector<Real> col_vec(nrow);

  pi_Ql3D_first(nr, nc, nf, nrow, ncol, nfib, l, v, coords_x, coords_y,
                coords_z, row_vec, col_vec, fib_vec);

  mgard_gen::copy3_level(0, v, work.data(), nrow, ncol, nfib);
  mgard_gen::assign3_level_l(0, work.data(), static_cast<Real>(0.0), nr, nc, nf,
                             nrow, ncol, nfib);

  for (int kfib = 0; kfib < nfib; kfib += stride) {
    mgard_common::copy_slice(work.data(), work2d, nrow, ncol, nfib, kfib);
    mgard_gen::refactor_2D_first(nr, nc, nrow, ncol, l, v2d.data(), work2d,
                                 coords_x, coords_y, row_vec, col_vec);
    mgard_common::copy_from_slice(work.data(), work2d, nrow, ncol, nfib, kfib);
  }

  for (int irow = 0; irow < nr; irow += stride) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < nc; jcol += stride) {
      int jc = mgard::get_lindex(nc, ncol, jcol);
      for (int kfib = 0; kfib < nfib; ++kfib) {
        fib_vec[kfib] = work[mgard::get_index3(ncol, nfib, ir, jc, kfib)];
      }
      mgard_cannon::mass_matrix_multiply(l, fib_vec, coords_z);
      mgard_gen::restriction_first(fib_vec, coords_z, nf, nfib);
      mgard_gen::solve_tridiag_M_l(l, fib_vec, coords_z, nf, nfib);
      for (int kfib = 0; kfib < nfib; ++kfib) {
        work[mgard::get_index3(ncol, nfib, ir, jc, kfib)] = fib_vec[kfib];
      }
    }
  }

  add3_level_l(0, v, work.data(), nr, nc, nf, nrow, ncol, nfib);
}

template <typename Real>
void recompose_3D(const int nr, const int nc, const int nf, const int nrow,
                  const int ncol, const int nfib, const int l_target, Real *v,
                  std::vector<Real> &work, std::vector<Real> &work2d,
                  std::vector<Real> &coords_x, std::vector<Real> &coords_y,
                  std::vector<Real> &coords_z) {
  // recompose

  std::vector<Real> v2d(nrow * ncol), fib_vec(nfib);
  std::vector<Real> row_vec(ncol);
  std::vector<Real> col_vec(nrow);

  //    //std::cout  << "recomposing" << "\n";
  for (int l = l_target; l > 0; --l) {

    int stride = std::pow(2, l); // current stride
    int Pstride = stride / 2;

    mgard_gen::copy3_level_l(l - 1, v, work.data(), nr, nc, nf, nrow, ncol,
                             nfib);

    mgard_gen::assign3_level_l(l, work.data(), static_cast<Real>(0.0), nr, nc,
                               nf, nrow, ncol, nfib);

    //        for (int kfib = 0; kfib < nfib; ++kfib)
    for (int kfib = 0; kfib < nf; kfib += Pstride) {
      //    int kf =kfib;
      int kf = mgard::get_lindex(nf, nfib, kfib);
      mgard_common::copy_slice(work.data(), work2d, nrow, ncol, nfib, kf);
      //            mgard_gen::compute_zl(nr, nc, nrow, ncol, l,  work2d,
      //            coords_x, coords_y, row_vec, col_vec);
      mgard_gen::refactor_2D(nr, nc, nrow, ncol, l - 1, v2d.data(), work2d,
                             coords_x, coords_y, row_vec, col_vec);
      mgard_common::copy_from_slice(work.data(), work2d, nrow, ncol, nfib, kf);
    }

    for (int irow = 0; irow < nr; irow += stride) {
      int ir = mgard::get_lindex(nr, nrow, irow);
      for (int jcol = 0; jcol < nc; jcol += stride) {
        int jc = mgard::get_lindex(nc, ncol, jcol);
        for (int kfib = 0; kfib < nfib; ++kfib) {
          fib_vec[kfib] = work[mgard::get_index3(ncol, nfib, ir, jc, kfib)];
        }

        mgard_gen::mass_mult_l(l - 1, fib_vec, coords_z, nf, nfib);

        mgard_gen::restriction_l(l, fib_vec, coords_z, nf, nfib);

        mgard_gen::solve_tridiag_M_l(l, fib_vec, coords_z, nf, nfib);

        for (int kfib = 0; kfib < nfib; ++kfib) {
          work[mgard::get_index3(ncol, nfib, ir, jc, kfib)] = fib_vec[kfib];
        }
      }
    }

    //- computed zl -//

    sub3_level_l(l, work.data(), v, nr, nc, nf, nrow, ncol,
                 nfib); // do -(Qu - zl)

    //        for (int is = 0; is < nfib; ++is)
    for (int kfib = 0; kfib < nf; kfib += stride) {
      int kf = mgard::get_lindex(nf, nfib, kfib);
      //            int kf = kfib;
      mgard_common::copy_slice(work.data(), work2d, nrow, ncol, nfib, kf);
      mgard_gen::prolong_add_2D(nr, nc, nrow, ncol, l, work2d, coords_x,
                                coords_y, row_vec, col_vec);
      mgard_common::copy_from_slice(work.data(), work2d, nrow, ncol, nfib, kf);
    }

    for (int irow = 0; irow < nr; irow += Pstride) {
      int ir = mgard::get_lindex(nr, nrow, irow);
      for (int jcol = 0; jcol < nc; jcol += Pstride) {
        int jc = mgard::get_lindex(nc, ncol, jcol);
        for (int kfib = 0; kfib < nfib; ++kfib) {
          fib_vec[kfib] = work[mgard::get_index3(ncol, nfib, ir, jc, kfib)];
        }

        mgard_gen::prolongate_l(l, fib_vec, coords_z, nf, nfib);

        for (int kfib = 0; kfib < nfib; ++kfib) {
          work[mgard::get_index3(ncol, nfib, ir, jc, kfib)] = fib_vec[kfib];
        }
      }
    }

    mgard_gen::assign3_level_l(l, v, static_cast<Real>(0.0), nr, nc, nf, nrow,
                               ncol, nfib);
    mgard_gen::sub3_level_l(l - 1, v, work.data(), nr, nc, nf, nrow, ncol,
                            nfib);
  }
}

template <typename Real>
void postp_3D(const int nr, const int nc, const int nf, const int nrow,
              const int ncol, const int nfib, const int l_target, Real *v,
              std::vector<Real> &work, std::vector<Real> &coords_x,
              std::vector<Real> &coords_y, std::vector<Real> &coords_z) {
  std::vector<Real> work2d(nrow * ncol), fib_vec(nfib), v2d(nrow * ncol);
  std::vector<Real> row_vec(ncol);
  std::vector<Real> col_vec(nrow);

  int l = 0;
  int stride = 1; // current stride
  int Pstride = stride / 2;

  // mgard_gen::copy3_level_l(l,  v,  work.data(),  nrow,  ncol, nfib,  nrow,
  // ncol, nfib);
  mgard_gen::copy3_level(l, v, work.data(), nrow, ncol, nfib);
  mgard_gen::assign3_level_l(l, work.data(), static_cast<Real>(0.0), nr, nc, nf,
                             nrow, ncol, nfib);

  for (int kfib = 0; kfib < nfib; kfib += stride) {
    mgard_common::copy_slice(work.data(), work2d, nrow, ncol, nfib, kfib);
    mgard_gen::refactor_2D_first(nr, nc, nrow, ncol, l, v2d.data(), work2d,
                                 coords_x, coords_y, row_vec, col_vec);
    mgard_common::copy_from_slice(work.data(), work2d, nrow, ncol, nfib, kfib);
  }

  for (int irow = 0; irow < nr; irow += stride) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < nc; jcol += stride) {
      int jc = mgard::get_lindex(nc, ncol, jcol);
      for (int kfib = 0; kfib < nfib; ++kfib) {
        fib_vec[kfib] = work[mgard::get_index3(ncol, nfib, ir, jc, kfib)];
      }
      mgard_cannon::mass_matrix_multiply(l, fib_vec, coords_z);
      mgard_gen::restriction_first(fib_vec, coords_z, nf, nfib);
      mgard_gen::solve_tridiag_M_l(l, fib_vec, coords_z, nf, nfib);
      for (int kfib = 0; kfib < nfib; ++kfib) {
        work[mgard::get_index3(ncol, nfib, ir, jc, kfib)] = fib_vec[kfib];
      }
    }
  }

  // for(int irow = 0; irow < nr; irow += 1)
  //   {
  //     //        int ir = irow;
  //     int ir  = mgard::get_lindex(nr,  nrow,  irow);
  //     for(int kfib = 0; kfib < nf; ++kfib)
  //       {
  //         //    int kf = kfib;
  //         int kf = mgard::get_lindex(nf, nfib, kfib);
  //         for(int jcol = 0; jcol < ncol; jcol += stride)
  //           {
  //             row_vec[jcol] =
  //             work[mgard::get_index3(ncol,nfib,ir,jcol,kf)];
  //           }

  //         //            mgard_gen::mass_mult_l(l, fib_vec, coords_z, nfib,
  //         nfib );
  //         //            assign_num_level(0, row_vec, 0.0, nr, nrow);
  //         mgard_cannon::mass_matrix_multiply(l, row_vec, coords_x);
  //         mgard_gen::restriction_first(row_vec, coords_x, nr, nrow );

  //         mgard_gen::solve_tridiag_M_l(l, row_vec, coords_x, nr, nrow );

  //         for(int jcol = 0; jcol < nc; jcol += stride)
  //           {
  //             int jr = mgard::get_lindex(nc, ncol, jcol);
  //             work[mgard::get_index3(ncol,nfib,ir,jr,kf)] =
  //             row_vec[jr] ;
  //           }

  //       }
  //   }

  // for(int jcol = 0; jcol < nc; jcol += stride)
  //   {
  //     int jc = mgard::get_lindex(nc, ncol, jcol);
  //     for(int kfib = 0; kfib < nf; ++kfib)
  //       {
  //         //            int kf = kfib;
  //         int kf = mgard::get_lindex(nf, nfib, kfib);
  //         for(int ir = 0; ir < nrow; ir += stride)
  //           {
  //             col_vec[ir] =
  //             work[mgard::get_index3(ncol,nfib,ir,jc,kf)];
  //           }
  //         //            assign_num_level(0, col_vec, 0.0, nc, ncol);
  //         mgard_cannon::mass_matrix_multiply(0, col_vec, coords_y);
  //         mgard_gen::restriction_first(col_vec, coords_y, nc, ncol);
  //         mgard_gen::solve_tridiag_M_l(0,  col_vec, coords_y, nc, ncol);
  //         for(int irow = 0; irow < nr; irow += stride)
  //           {
  //             int ir = mgard::get_lindex(nr, nrow, irow);
  //             work[mgard::get_index3(ncol,nfib,ir,jc,kf)] =
  //             col_vec[ir] ;
  //           }

  //       }
  //   }

  // for(int irow = 0; irow < nr; irow += stride)
  //   {
  //     int ir  = mgard::get_lindex(nr,  nrow,  irow);
  //     for(int jcol = 0; jcol < nc; jcol += stride)
  //       {
  //         int jc  = mgard::get_lindex(nc,  ncol,  jcol);
  //         for(int kfib = 0; kfib < nfib; ++kfib)
  //           {
  //             fib_vec[kfib] =
  //             work[mgard::get_index3(ncol,nfib,ir,jc,kfib)];
  //           }

  //         //            mgard_gen::mass_mult_l(l, fib_vec, coords_z, nfib,
  //         nfib );
  //         //            assign_num_level(0, fib_vec, 0.0, nf, nfib);
  //         mgard_cannon::mass_matrix_multiply(l, fib_vec, coords_z);
  //         mgard_gen::restriction_first(fib_vec, coords_z, nf, nfib );

  //         mgard_gen::solve_tridiag_M_l(l, fib_vec, coords_z, nf, nfib );

  //         for(int kfib = 0; kfib < nf; ++kfib)
  //           {
  //             int kf = mgard::get_lindex(nf, nfib, kfib);
  //             work[mgard::get_index3(ncol,nfib,ir,jc,kf)] =
  //             fib_vec[kf] ;

  //           }

  //       }
  //   }

  //- computed zl -//

  sub3_level_l(0, work.data(), v, nr, nc, nf, nrow, ncol, nfib); // do -(Qu -
                                                                 // zl)

  //    for (int kf = 0; kf < nfib; ++kf)
  for (int kfib = 0; kfib < nf; kfib += stride) {
    int kf = mgard::get_lindex(nf, nfib, kfib);

    mgard_common::copy_slice(work.data(), work2d, nrow, ncol, nfib, kf);
    mgard_gen::prolong_add_2D_last(nr, nc, nrow, ncol, l, work2d, coords_x,
                                   coords_y, row_vec, col_vec);
    mgard_common::copy_from_slice(work.data(), work2d, nrow, ncol, nfib, kf);
  }

  for (int irow = 0; irow < nrow; irow += stride) {
    for (int jcol = 0; jcol < ncol; jcol += stride) {

      for (int kfib = 0; kfib < nfib; ++kfib) {
        fib_vec[kfib] = work[mgard::get_index3(ncol, nfib, irow, jcol, kfib)];
      }

      mgard_gen::prolongate_last(fib_vec, coords_z, nf, nfib);

      for (int kfib = 0; kfib < nfib; ++kfib) {
        work[mgard::get_index3(ncol, nfib, irow, jcol, kfib)] = fib_vec[kfib];
      }
    }
  }

  mgard_gen::assign3_level_l(0, v, static_cast<Real>(0.0), nr, nc, nf, nrow,
                             ncol, nfib);
  mgard_gen::sub3_level(0, v, work.data(), nrow, ncol, nfib);
  //    mgard_gen::sub3_level(l, v, work.data(), nrow,  ncol,  nfib);
}

template <typename Real>
void recompose_2D(const int nr, const int nc, const int nrow, const int ncol,
                  const int l_target, Real *v, std::vector<Real> &work,
                  std::vector<Real> &coords_x, std::vector<Real> &coords_y,
                  std::vector<Real> &row_vec, std::vector<Real> &col_vec) {
  // recompose
  //    //std::cout  << "recomposing" << "\n";
  //    for (int l = l_target ; l > 0 ; --l)
  //  {
  int l = l_target;

  int stride = std::pow(2, l); // current stride
  int Pstride = stride / 2;

  //        copy_level_l(l-1,  v,  work.data(),  nr,  nc,  nrow,  ncol);

  // assign_num_level_l(l, work.data(),  0.0,  nr,  nc,  nrow,  ncol);

  //        //std::cout  << "recomposing-rowsweep" << "\n";
  //  l = 0;
  // row-sweep
  for (int irow = 0; irow < nr; ++irow) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[mgard::get_index(ncol, ir, jcol)];
    }

    mgard_gen::mass_mult_l(l - 1, row_vec, coords_x, nc, ncol);

    mgard_gen::restriction_l(l, row_vec, coords_x, nc, ncol);

    mgard_gen::solve_tridiag_M_l(l, row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[mgard::get_index(ncol, ir, jcol)] = row_vec[jcol];
    }
  }

  //   //std::cout  << "recomposing-colsweep" << "\n";

  // column-sweep, this is the slow one! Need something like column_copy
  if (nrow > 1) // check if we have 1-D array..
  {
    for (int jcol = 0; jcol < nc; jcol += stride) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      for (int irow = 0; irow < nrow; ++irow) {
        col_vec[irow] = work[mgard::get_index(ncol, irow, jr)];
      }

      mgard_gen::mass_mult_l(l - 1, col_vec, coords_y, nr, nrow);

      mgard_gen::restriction_l(l, col_vec, coords_y, nr, nrow);

      mgard_gen::solve_tridiag_M_l(l, col_vec, coords_y, nr, nrow);

      for (int irow = 0; irow < nrow; ++irow) {
        work[mgard::get_index(ncol, irow, jr)] = col_vec[irow];
      }
    }
  }
  //        subtract_level_l(l, work.data(), v,  nr,  nc,  nrow,  ncol); //do
  //        -(Qu - zl)
  //        //std::cout  << "recomposing-rowsweep2" << "\n";

  //   //int Pstride = stride/2; //finer stride

  //   // row-sweep
  for (int irow = 0; irow < nr; irow += stride) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[mgard::get_index(ncol, ir, jcol)];
    }

    mgard_gen::prolongate_l(l, row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[mgard::get_index(ncol, ir, jcol)] = row_vec[jcol];
    }
  }

  //   //std::cout  << "recomposing-colsweep2" << "\n";
  // column-sweep, this is the slow one! Need something like column_copy
  if (nrow > 1) {
    for (int jcol = 0; jcol < nc; jcol += Pstride) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      for (int irow = 0; irow < nrow; ++irow) // copy all rows
      {
        col_vec[irow] = work[mgard::get_index(ncol, irow, jr)];
      }

      mgard_gen::prolongate_l(l, col_vec, coords_y, nr, nrow);

      for (int irow = 0; irow < nrow; ++irow) {
        work[mgard::get_index(ncol, irow, jr)] = col_vec[irow];
      }
    }
  }
  // //std::cout  << "last step" << "\n";

  // assign_num_level_l(l, v, 0.0, nr, nc, nrow, ncol);
  // subtract_level_l(l-1, v, work.data(),  nr,  nc,  nrow,  ncol);
  //      }
}

template <typename Real>
void recompose_2D_full(const int nr, const int nc, const int nrow,
                       const int ncol, const int l_target, Real *v,
                       std::vector<Real> &work, std::vector<Real> &coords_x,
                       std::vector<Real> &coords_y, std::vector<Real> &row_vec,
                       std::vector<Real> &col_vec) {
  for (int l = l_target; l > 0; --l) {

    int stride = std::pow(2, l); // current stride
    int Pstride = stride / 2;

    copy_level_l(l - 1, v, work.data(), nr, nc, nrow, ncol);
    assign_num_level_l(l, work.data(), static_cast<Real>(0.0), nr, nc, nrow,
                       ncol);

    for (int irow = 0; irow < nr; ++irow) {
      int ir = mgard::get_lindex(nr, nrow, irow);
      for (int jcol = 0; jcol < ncol; ++jcol) {
        row_vec[jcol] = work[mgard::get_index(ncol, ir, jcol)];
      }

      mgard_gen::mass_mult_l(l - 1, row_vec, coords_x, nc, ncol);

      mgard_gen::restriction_l(l, row_vec, coords_x, nc, ncol);

      mgard_gen::solve_tridiag_M_l(l, row_vec, coords_x, nc, ncol);

      for (int jcol = 0; jcol < ncol; ++jcol) {
        work[mgard::get_index(ncol, ir, jcol)] = row_vec[jcol];
      }
    }

    // column-sweep, this is the slow one! Need something like column_copy
    if (nrow > 1) // check if we have 1-D array..
    {
      for (int jcol = 0; jcol < nc; jcol += stride) {
        int jr = mgard::get_lindex(nc, ncol, jcol);
        for (int irow = 0; irow < nrow; ++irow) {
          col_vec[irow] = work[mgard::get_index(ncol, irow, jr)];
        }

        mgard_gen::mass_mult_l(l - 1, col_vec, coords_y, nr, nrow);

        mgard_gen::restriction_l(l, col_vec, coords_y, nr, nrow);

        mgard_gen::solve_tridiag_M_l(l, col_vec, coords_y, nr, nrow);

        for (int irow = 0; irow < nrow; ++irow) {
          work[mgard::get_index(ncol, irow, jr)] = col_vec[irow];
        }
      }
    }
    subtract_level_l(l, work.data(), v, nr, nc, nrow, ncol); // do -(Qu - zl)

    //   // row-sweep
    for (int irow = 0; irow < nr; irow += stride) {
      int ir = mgard::get_lindex(nr, nrow, irow);
      for (int jcol = 0; jcol < ncol; ++jcol) {
        row_vec[jcol] = work[mgard::get_index(ncol, ir, jcol)];
      }

      mgard_gen::prolongate_l(l, row_vec, coords_x, nc, ncol);

      for (int jcol = 0; jcol < ncol; ++jcol) {
        work[mgard::get_index(ncol, ir, jcol)] = row_vec[jcol];
      }
    }

    // column-sweep, this is the slow one! Need something like column_copy
    if (nrow > 1) {
      for (int jcol = 0; jcol < nc; jcol += Pstride) {
        int jr = mgard::get_lindex(nc, ncol, jcol);
        for (int irow = 0; irow < nrow; ++irow) // copy all rows
        {
          col_vec[irow] = work[mgard::get_index(ncol, irow, jr)];
        }

        mgard_gen::prolongate_l(l, col_vec, coords_y, nr, nrow);

        for (int irow = 0; irow < nrow; ++irow) {
          work[mgard::get_index(ncol, irow, jr)] = col_vec[irow];
        }
      }
    }

    assign_num_level_l(l, v, static_cast<Real>(0.0), nr, nc, nrow, ncol);
    subtract_level_l(l - 1, v, work.data(), nr, nc, nrow, ncol);
  }
}

template <typename Real>
void postp_2D(const int nr, const int nc, const int nrow, const int ncol,
              const int l_target, Real *v, std::vector<Real> &work,
              std::vector<Real> &coords_x, std::vector<Real> &coords_y,
              std::vector<Real> &row_vec, std::vector<Real> &col_vec) {
  mgard_cannon::copy_level(nrow, ncol, 0, v, work);

  assign_num_level_l(0, work.data(), static_cast<Real>(0.0), nr, nc, nrow,
                     ncol);

  for (int irow = 0; irow < nrow; ++irow) {
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[mgard::get_index(ncol, irow, jcol)];
    }

    mgard_cannon::mass_matrix_multiply(0, row_vec, coords_x);

    restriction_first(row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[mgard::get_index(ncol, irow, jcol)] = row_vec[jcol];
    }
  }

  for (int irow = 0; irow < nr; ++irow) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[mgard::get_index(ncol, ir, jcol)];
    }

    mgard_gen::solve_tridiag_M_l(0, row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[mgard::get_index(ncol, ir, jcol)] = row_vec[jcol];
    }
  }

  //     // column-sweep, this is the slow one! Need something like column_copy
  if (nrow > 1) // check if we have 1-D array..
  {
    for (int jcol = 0; jcol < ncol; ++jcol) {
      //      int jr  = mgard::get_lindex(nc,  ncol,  jcol);
      for (int irow = 0; irow < nrow; ++irow) {
        col_vec[irow] = work[mgard::get_index(ncol, irow, jcol)];
      }

      mgard_cannon::mass_matrix_multiply(0, col_vec, coords_y);

      mgard_gen::restriction_first(col_vec, coords_y, nr, nrow);

      for (int irow = 0; irow < nrow; ++irow) {
        work[mgard::get_index(ncol, irow, jcol)] = col_vec[irow];
      }
    }

    for (int jcol = 0; jcol < nc; ++jcol) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      for (int irow = 0; irow < nrow; ++irow) {
        col_vec[irow] = work[mgard::get_index(ncol, irow, jr)];
      }

      mgard_gen::solve_tridiag_M_l(0, col_vec, coords_y, nr, nrow);
      for (int irow = 0; irow < nrow; ++irow) {
        work[mgard::get_index(ncol, irow, jr)] = col_vec[irow];
      }
    }
  }

  subtract_level_l(0, work.data(), v, nr, nc, nrow, ncol); // do -(Qu - zl)
  //        //std::cout  << "recomposing-rowsweep2" << "\n";

  //     //   //int Pstride = stride/2; //finer stride

  //   //   // row-sweep
  for (int irow = 0; irow < nr; ++irow) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[mgard::get_index(ncol, ir, jcol)];
    }

    mgard_gen::prolongate_last(row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[mgard::get_index(ncol, ir, jcol)] = row_vec[jcol];
    }
  }

  //   //   //std::cout  << "recomposing-colsweep2" << "\n";
  //     // column-sweep, this is the slow one! Need something like column_copy
  if (nrow > 1) {
    for (int jcol = 0; jcol < ncol; ++jcol) {
      // int jr  = mgard::get_lindex(nc,  ncol,  jcol);
      for (int irow = 0; irow < nrow; ++irow) // copy all rows
      {
        col_vec[irow] = work[mgard::get_index(ncol, irow, jcol)];
      }

      mgard_gen::prolongate_last(col_vec, coords_y, nr, nrow);

      for (int irow = 0; irow < nrow; ++irow) {
        work[mgard::get_index(ncol, irow, jcol)] = col_vec[irow];
      }
    }
  }

  //     //std::cout  << "last step" << "\n";

  assign_num_level_l(0, v, static_cast<Real>(0.0), nr, nc, nrow, ncol);
  mgard_cannon::subtract_level(nrow, ncol, 0, v, work.data());
}

template <typename Real>
void qwrite_2D_l(const int nr, const int nc, const int nrow, const int ncol,
                 const int nlevel, const int l, Real *v, const Real tol,
                 const Real norm, const std::string outfile) {

  int stride = std::pow(2, l); // current stride
  int Cstride = 2 * stride;

  const mgard::LinearQuantizer<Real, int> quantizer(2 * norm * tol /
                                                    (nlevel + 1));

  gzFile out_file = gzopen(outfile.c_str(), "w6b");
  int prune_count = 0;
  gzwrite(out_file, &quantizer.quantum, sizeof(Real));

  // level L+1, finest first level
  for (int irow = 0; irow < nr; ++irow) // loop over the logical array
  {
    int ir = mgard::get_lindex(nr, nrow, irow);

    for (int jcol = 0; jcol < nc - 1; ++jcol) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      int jrP = mgard::get_lindex(nc, ncol, jcol + 1);

      if (jrP != jr + 1) // next real memory location was jumped over, so this
                         // is level L+1
      {
        const int n = quantizer(v[mgard::get_index(ncol, ir, jr + 1)]);
        if (n == 0)
          ++prune_count;
        gzwrite(out_file, &n, sizeof(int));
      }
    }
  }

  for (int jcol = 0; jcol < nc; ++jcol) {
    int jr = mgard::get_lindex(nc, ncol, jcol);
    for (int irow = 0; irow < nr - 1; ++irow) // loop over the logical array
    {
      int ir = mgard::get_lindex(nr, nrow, irow);
      int irP = mgard::get_lindex(nr, nrow, irow + 1);
      if (irP != ir + 1) // next real memory location was jumped over, so this
                         // is level L+1
      {
        const int n = quantizer(v[mgard::get_index(ncol, ir + 1, jr)]);
        if (n == 0)
          ++prune_count;
        gzwrite(out_file, &n, sizeof(int));
      }
    }
  }

  for (int irow = 0; irow < nr - 1; ++irow) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    int irP = mgard::get_lindex(nr, nrow, irow + 1);

    for (int jcol = 0; jcol < nc - 1; ++jcol) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      int jrP = mgard::get_lindex(nc, ncol, jcol + 1);
      if ((irP != ir + 1) &&
          (jrP != jr + 1)) // we skipped both a row and a column
      {
        const int n = quantizer(v[mgard::get_index(ncol, ir + 1, jr + 1)]);
        if (n == 0)
          ++prune_count;
        gzwrite(out_file, &n, sizeof(int));
      }
    }
  }

  // levels from L->0 in 2^k+1
  for (int l = 0; l <= nlevel; l++) {
    int stride = std::pow(2, l);
    int Cstride = stride * 2;
    int row_counter = 0;

    for (int irow = 0; irow < nr; irow += stride) {
      int ir = mgard::get_lindex(nr, nrow, irow);
      if (row_counter % 2 == 0 && l != nlevel) {
        for (int jcol = Cstride; jcol < nc; jcol += Cstride) {
          int jr = mgard::get_lindex(nc, ncol, jcol);
          const int n = quantizer(v[mgard::get_index(ncol, ir, jr - stride)]);
          if (n == 0)
            ++prune_count;
          gzwrite(out_file, &n, sizeof(int));
        }

      } else {
        for (int jcol = 0; jcol < nc; jcol += stride) {
          int jr = mgard::get_lindex(nc, ncol, jcol);
          const int n = quantizer(v[mgard::get_index(ncol, ir, jr)]);
          if (n == 0)
            ++prune_count;
          gzwrite(out_file, &n, sizeof(int));
        }
      }
      ++row_counter;
    }
  }

  // std::cout  << "Pruned : "<< prune_count << " Reduction : " << (Real)
  // nrow*ncol /(nrow*ncol - prune_count) << "\n";
  gzclose(out_file);
}

template <typename Real>
void quantize_2D(const int nr, const int nc, const int nrow, const int ncol,
                 const int nlevel, Real *v, std::vector<int> &work,
                 const std::vector<Real> &coords_x,
                 const std::vector<Real> &coords_y, Real s, Real norm,
                 Real tol) {

  // s-norm version of per-level quantizer.
  const Real coeff = norm * tol;
  std::memcpy(work.data(), &coeff, sizeof(Real));
  const int size_ratio = sizeof(Real) / sizeof(int);
  int prune_count = 0;

  //    Real s = 0.0;
  int count = 0;
  count += size_ratio;
  //  //std::cout  << "2D quantar  starting " << count << "\n";
  // level -1, first level for non 2^k+1
  {

    const Real dx = mgard_gen::get_h_l(coords_x, ncol, ncol, 0, 1);
    const Real dy = mgard_gen::get_h_l(coords_y, nrow, nrow, 0, 1);
    const Real vol = dx * dy;
    const mgard::LinearQuantizer<Real, int> quantizer(
        coeff / (std::sqrt(vol) * std::pow(2, s * nlevel)));

    for (int irow = 0; irow < nr - 1; ++irow) {
      int ir = mgard::get_lindex(nr, nrow, irow);
      int irP = mgard::get_lindex(nr, nrow, irow + 1);
      if (irP != ir + 1) // skipped a row
      {
        for (int jcol = 0; jcol < ncol; ++jcol) {
          Real val = v[mgard::get_index(ncol, ir + 1, jcol)];
          const int n = quantizer(val);
          //	      //std::cout  << "writing  " << count << "\n";
          work[count] = n;
          ++count;
        }
      }
    }

    for (int irow = 0; irow < nr; ++irow) {
      int ir = mgard::get_lindex(nr, nrow, irow);

      for (int jcol = 0; jcol < nc - 1; ++jcol) {
        int jc = mgard::get_lindex(nc, ncol, jcol);
        int jcP = mgard::get_lindex(nc, ncol, jcol + 1);
        if (jcP != jc + 1) // skipped a column
        {
          Real val = v[mgard::get_index(ncol, ir, jc + 1)];
          const int n = quantizer(val);
          work[count] = n;
          //	      //std::cout  << "writing  " << count << "\n";
          ++count;
        }
      }
    }
  }

  // std::cout  <<"Level -1 " << count << "\n";
  // // 2^k+1 part //
  for (int ilevel = 0; ilevel < nlevel; ++ilevel) {
    int stride = std::pow(2, ilevel);
    int Cstride = 2 * stride;

    const Real dx = get_h_l(coords_x, nc, ncol, 0, stride);
    const Real dy = get_h_l(coords_y, nr, nrow, 0, stride);
    const Real vol = dx * dy;

    const mgard::LinearQuantizer<Real, int> quantizer(
        coeff / (std::sqrt(vol) * std::pow(2, s * (nlevel - ilevel))));

    int row_counter = 0;

    for (int irow = 0; irow < nr; irow += stride) {
      int ir = mgard::get_lindex(nr, nrow, irow);
      if (row_counter % 2 == 0) {
        for (int jcol = Cstride; jcol < nc; jcol += Cstride) {
          int jc = mgard::get_lindex(nc, ncol, jcol - stride);
          Real val = v[mgard::get_index(ncol, ir, jc)];
          const int n = quantizer(val);
          work[count] = n;
          //		    //std::cout  << "writing  " << count << "\n";
          ++count;
        }

      } else {
        for (int jcol = 0; jcol < nc; jcol += stride) {
          int jc = mgard::get_lindex(nc, ncol, jcol);
          Real val = v[mgard::get_index(ncol, ir, jc)];
          const int n = quantizer(val);
          work[count] = n;
          //		    //std::cout  << "writing  " << count << "\n";
          ++count;
        }
      }
      ++row_counter;
    }
  }

  // last level -> L=0
  {

    int stride = std::pow(2, nlevel);
    const Real dx = get_h_l(coords_x, nc, ncol, 0, stride);
    const Real dy = get_h_l(coords_y, nr, nrow, 0, stride);
    const Real vol = dx * dy;
    const mgard::LinearQuantizer<Real, int> quantizer(coeff / std::sqrt(vol));

    for (int irow = 0; irow < nr; irow += stride) {
      int ir = mgard::get_lindex(nr, nrow, irow);
      for (int jcol = 0; jcol < nc; jcol += stride) {
        int jc = mgard::get_lindex(nc, ncol, jcol);

        Real val = v[mgard::get_index(ncol, ir, jc)];
        const int n = quantizer(val);
        //	    //std::cout  << "writing  " << count << "\n";
        work[count] = n;
        ++count;
      }
    }
  }

  // std::cout  << "Wrote out 2d: " << count <<"\n";
}

template <typename Real>
void quantize_3D(const int nr, const int nc, const int nf, const int nrow,
                 const int ncol, const int nfib, const int nlevel, Real *v,
                 std::vector<int> &work, const std::vector<Real> &coords_x,
                 const std::vector<Real> &coords_y,
                 const std::vector<Real> &coords_z, const Real norm,
                 const Real tol) {

  // L-infty version of per level quantizer, reorders MGARDized coeffs. per
  // level
  // account for the  possible projection to 2^k+1
  const mgard::LinearQuantizer<Real, int> quantizer(norm * tol / (nlevel + 2));
  std::memcpy(work.data(), &quantizer.quantum, sizeof(Real));
  const int size_ratio = sizeof(Real) / sizeof(int);
  int prune_count = 0;

  int count = 0;
  count += size_ratio;

  // level -1, first level for non 2^k+1

  for (int kfib = 0; kfib < nf - 1; ++kfib) {
    int kf = mgard::get_lindex(nf, nfib, kfib);
    int kfp = mgard::get_lindex(nf, nfib, kfib + 1);

    if (kfp != kf + 1) // skipped a plane
    {
      for (int irow = 0; irow < nrow; ++irow) {
        for (int jcol = 0; jcol < ncol; ++jcol) {
          Real val = v[mgard::get_index3(ncol, nfib, irow, jcol, kf + 1)];
          const int n = quantizer(val);
          work[count] = n;
          ++count;
        }
      }
    }
  }

  for (int kfib = 0; kfib < nf; ++kfib) {
    int kf = mgard::get_lindex(nf, nfib, kfib);
    for (int irow = 0; irow < nr - 1; ++irow) {
      int ir = mgard::get_lindex(nr, nrow, irow);
      int irP = mgard::get_lindex(nr, nrow, irow + 1);
      if (irP != ir + 1) // skipped a row
      {
        for (int jcol = 0; jcol < ncol; ++jcol) {
          Real val = v[mgard::get_index3(ncol, nfib, ir + 1, jcol, kf)];
          const int n = quantizer(val);
          work[count] = n;
          ++count;
        }
      }
    }

    for (int irow = 0; irow < nr; ++irow) {
      int ir = mgard::get_lindex(nr, nrow, irow);

      for (int jcol = 0; jcol < nc - 1; ++jcol) {
        int jc = mgard::get_lindex(nc, ncol, jcol);
        int jcP = mgard::get_lindex(nc, ncol, jcol + 1);
        if (jcP != jc + 1) // skipped a column
        {
          Real val = v[mgard::get_index3(ncol, nfib, ir, jc + 1, kf)];
          const int n = quantizer(val);
          work[count] = n;
          ++count;
        }
      }
    }
  }

  // // 2^k+1 part //

  for (int ilevel = 0; ilevel < nlevel; ++ilevel) {
    int stride = std::pow(2, ilevel);
    int Cstride = 2 * stride;

    int fib_counter = 0;

    for (int kfib = 0; kfib < nf; kfib += stride) {
      int kf = mgard::get_lindex(nf, nfib, kfib);
      int row_counter = 0;

      if (fib_counter % 2 == 0) {
        for (int irow = 0; irow < nr; irow += stride) {
          int ir = mgard::get_lindex(nr, nrow, irow);
          if (row_counter % 2 == 0) {
            for (int jcol = Cstride; jcol < nc; jcol += Cstride) {
              int jc = mgard::get_lindex(nc, ncol, jcol - stride);
              Real val = v[mgard::get_index3(ncol, nfib, ir, jc, kf)];
              const int n = quantizer(val);
              work[count] = n;
              ++count;
            }

          } else {
            for (int jcol = 0; jcol < nc; jcol += stride) {
              int jc = mgard::get_lindex(nc, ncol, jcol);
              Real val = v[mgard::get_index3(ncol, nfib, ir, jc, kf)];
              const int n = quantizer(val);
              work[count] = n;
              ++count;
              //         outfile.write(reinterpret_cast<char*>(
              //         &v[mgard::get_index3(ncol, nfib, ir, jc, kf)] ),
              //         sizeof(Real) );
              //                  //std::cout  <<  v[irow][icol] << "\t";
            }
          }
          ++row_counter;
        }
      } else {
        for (int irow = 0; irow < nr; irow += stride) {
          int ir = mgard::get_lindex(nr, nrow, irow);
          for (int jcol = 0; jcol < nc; jcol += stride) {
            int jc = mgard::get_lindex(nc, ncol, jcol);
            Real val = v[mgard::get_index3(ncol, nfib, ir, jc, kf)];
            const int n = quantizer(val);
            work[count] = n;
            ++count;
            //                      outfile.write(reinterpret_cast<char*>(
            //                      &v[mgard::get_index3(ncol, nfib, ir,
            //                      jc, kf)] ), sizeof(Real) );
          }
        }
      }
      ++fib_counter;
    }
  }

  // last level -> L=0
  int stride = std::pow(2, nlevel);

  for (int irow = 0; irow < nr; irow += stride) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < nc; jcol += stride) {
      int jc = mgard::get_lindex(nc, ncol, jcol);
      for (int kfib = 0; kfib < nf; kfib += stride) {
        int kf = mgard::get_lindex(nf, nfib, kfib);
        Real val = v[mgard::get_index3(ncol, nfib, ir, jc, kf)];
        const int n = quantizer(val);
        work[count] = n;
        ++count;
      }
    }
  }

  // std::cout  << "Wrote out: " << count <<"\n";
}

template <typename Real>
void quantize_3D(const int nr, const int nc, const int nf, const int nrow,
                 const int ncol, const int nfib, const int nlevel, Real *v,
                 std::vector<int> &work, const std::vector<Real> &coords_x,
                 const std::vector<Real> &coords_y,
                 const std::vector<Real> &coords_z, const Real s,
                 const Real norm, const Real tol) {

  // s-norm version of per-level quantizer.
  const Real coeff = norm * tol;
  std::memcpy(work.data(), &coeff, sizeof(Real));
  const int size_ratio = sizeof(Real) / sizeof(int);
  int prune_count = 0;

  //    Real s = 0.0;
  int count = 0;
  count += size_ratio;

  // level -1, first level for non 2^k+1
  {

    const Real dx = mgard_gen::get_h_l(coords_x, ncol, ncol, 0, 1);
    const Real dy = mgard_gen::get_h_l(coords_y, nrow, nrow, 0, 1);
    const Real dz = mgard_gen::get_h_l(coords_z, nfib, nfib, 0, 1);
    const Real vol = dx * dy * dz;

    const mgard::LinearQuantizer<Real, int> quantizer(
        coeff / (std::sqrt(vol) * std::pow(2, s * nlevel)));
    // std::cout  << "Volume -1: " << vol << std::endl;
    ////std::cout  << "quantizer "  << coeff << std::endl;

    for (int kfib = 0; kfib < nf - 1; ++kfib) {
      int kf = mgard::get_lindex(nf, nfib, kfib);
      int kfp = mgard::get_lindex(nf, nfib, kfib + 1);

      if (kfp != kf + 1) // skipped a plane
      {
        for (int irow = 0; irow < nrow; ++irow) {
          for (int jcol = 0; jcol < ncol; ++jcol) {
            Real val = v[mgard::get_index3(ncol, nfib, irow, jcol, kf + 1)];
            const int n = quantizer(val);
            ////std::cout  << "quantized "  << val << std::endl;
            work[count] = n;
            ++count;
          }
        }
      }
    }

    int count_row = 0;
    int count_col = 0;
    int count_sol = 0;

    for (int kfib = 0; kfib < nf; ++kfib) {
      int kf = mgard::get_lindex(nf, nfib, kfib);
      for (int irow = 0; irow < nr - 1; ++irow) {
        int ir = mgard::get_lindex(nr, nrow, irow);
        int irP = mgard::get_lindex(nr, nrow, irow + 1);
        if (irP != ir + 1) // skipped a row
        {
          //  //std::cout  <<"Skipped row: "  << ir + 1 << "\n";
          for (int jcol = 0; jcol < ncol; ++jcol) {
            Real val = v[mgard::get_index3(ncol, nfib, ir + 1, jcol, kf)];
            const int n = quantizer(val);
            //                    //std::cout  << "quantized "  << val <<
            //                    std::endl;
            work[count] = n;
            ++count_row;
            ++count;
          }
        }
      }

      for (int irow = 0; irow < nr; ++irow) {
        int ir = mgard::get_lindex(nr, nrow, irow);

        //      //std::cout  <<"Non skipped row: "  << ir  << "\n";
        for (int jcol = 0; jcol < nc - 1; ++jcol) {
          int jc = mgard::get_lindex(nc, ncol, jcol);
          int jcP = mgard::get_lindex(nc, ncol, jcol + 1);
          if (jcP != jc + 1) // skipped a column
          {
            Real val = v[mgard::get_index3(ncol, nfib, ir, jc + 1, kf)];
            const int n = quantizer(val);
            work[count] = n;
            ++count_col;
            ++count;
            //                    //std::cout  <<"Skipped col: " << ir << "\t"
            //                    << jc + 1 << "\n";
          }
        }
      }
    }
  }

  // // 2^k+1 part //
  for (int ilevel = 0; ilevel < nlevel; ++ilevel) {
    int stride = std::pow(2, ilevel);
    int Cstride = 2 * stride;

    int fib_counter = 0;

    const Real dx = get_h_l(coords_x, nc, ncol, 0, stride);
    const Real dy = get_h_l(coords_y, nr, nrow, 0, stride);
    const Real dz = get_h_l(coords_z, nf, nfib, 0, stride);
    const Real vol = dx * dy * dz;

    const mgard::LinearQuantizer<Real, int> quantizer(
        coeff / (std::sqrt(vol) * std::pow(2, s * (nlevel - ilevel))));
    // std::cout  << "Volume : " << ilevel << "\t"<< vol << std::endl;
    // //std::cout  << "Stride : " << stride << "\t"<< vol << std::endl;

    for (int kfib = 0; kfib < nf; kfib += stride) {
      int kf = mgard::get_lindex(nf, nfib, kfib);
      int row_counter = 0;

      if (fib_counter % 2 == 0) {
        for (int irow = 0; irow < nr; irow += stride) {
          int ir = mgard::get_lindex(nr, nrow, irow);
          if (row_counter % 2 == 0) {
            for (int jcol = Cstride; jcol < nc; jcol += Cstride) {
              int jc = mgard::get_lindex(nc, ncol, jcol - stride);
              Real val = v[mgard::get_index3(ncol, nfib, ir, jc, kf)];
              const int n = quantizer(val);
              work[count] = n;
              ++count;
              //                          outfile.write(reinterpret_cast<char*>(
              //                          &v[mgard::get_index3(ncol,
              //                          nfib, ir,jc - stride, kf)] ),
              //                          sizeof(Real) );
              //                  //std::cout  <<  v[irow][icol - stride] <<
              //                  "\t";
            }

          } else {
            for (int jcol = 0; jcol < nc; jcol += stride) {
              int jc = mgard::get_lindex(nc, ncol, jcol);
              Real val = v[mgard::get_index3(ncol, nfib, ir, jc, kf)];
              const int n = quantizer(val);
              work[count] = n;
              ++count;
              //         outfile.write(reinterpret_cast<char*>(
              //         &v[mgard::get_index3(ncol, nfib, ir, jc, kf)] ),
              //         sizeof(Real) );
              //                  //std::cout  <<  v[irow][icol] << "\t";
            }
          }
          ++row_counter;
        }
      } else {
        for (int irow = 0; irow < nr; irow += stride) {
          int ir = mgard::get_lindex(nr, nrow, irow);
          for (int jcol = 0; jcol < nc; jcol += stride) {
            int jc = mgard::get_lindex(nc, ncol, jcol);
            Real val = v[mgard::get_index3(ncol, nfib, ir, jc, kf)];
            const int n = quantizer(val);
            work[count] = n;
            ++count;
            //                      outfile.write(reinterpret_cast<char*>(
            //                      &v[mgard::get_index3(ncol, nfib, ir,
            //                      jc, kf)] ), sizeof(Real) );
          }
        }
      }
      ++fib_counter;
    }
  }

  // last level -> L=0
  {

    int stride = std::pow(2, nlevel);
    const Real dx = get_h_l(coords_x, nc, ncol, 0, stride);
    const Real dy = get_h_l(coords_y, nr, nrow, 0, stride);
    const Real dz = get_h_l(coords_z, nf, nfib, 0, stride);
    const Real vol = dx * dy * dz;

    const mgard::LinearQuantizer<Real, int> quantizer(coeff / std::sqrt(vol));
    // std::cout  << "Volume : " << nlevel << "\t"<< vol << std::endl;
    // //std::cout  << "Stride : " << stride << "\t"<< vol << std::endl;

    for (int irow = 0; irow < nr; irow += stride) {
      int ir = mgard::get_lindex(nr, nrow, irow);
      for (int jcol = 0; jcol < nc; jcol += stride) {
        int jc = mgard::get_lindex(nc, ncol, jcol);
        for (int kfib = 0; kfib < nf; kfib += stride) {
          int kf = mgard::get_lindex(nf, nfib, kfib);
          Real val = v[mgard::get_index3(ncol, nfib, ir, jc, kf)];
          const int n = quantizer(val);
          work[count] = n;
          ++count;
        }
      }
    }
  }

  // std::cout  << "Wrote out: " << count <<"\n";
}

template <typename Real>
void dequantize_3D(const int nr, const int nc, const int nf, const int nrow,
                   const int ncol, const int nfib, const int nlevel, Real *v,
                   std::vector<int> &work, const std::vector<Real> &coords_x,
                   const std::vector<Real> &coords_y,
                   const std::vector<Real> &coords_z) {

  const int size_ratio = sizeof(Real) / sizeof(int);
  Real quantum; // quantizing factor

  std::memcpy(&quantum, work.data(), sizeof(Real));
  const mgard::LinearDequantizer<int, Real> dequantizer(quantum);

  int imeg = 0; // mega-counter

  imeg += size_ratio;

  for (int kfib = 0; kfib < nf - 1; ++kfib) {
    int kf = mgard::get_lindex(nf, nfib, kfib);
    int kfp = mgard::get_lindex(nf, nfib, kfib + 1);

    if (kfp != kf + 1) // skipped a plane
    {
      for (int irow = 0; irow < nrow; ++irow) {
        for (int jcol = 0; jcol < ncol; ++jcol) {
          v[mgard::get_index3(ncol, nfib, irow, jcol, kf + 1)] =
              dequantizer(work[imeg]);
          ++imeg;
        }
      }
    }
  }

  for (int kfib = 0; kfib < nf; ++kfib) {
    int kf = mgard::get_lindex(nf, nfib, kfib);
    for (int irow = 0; irow < nr - 1; ++irow) {
      int ir = mgard::get_lindex(nr, nrow, irow);
      int irP = mgard::get_lindex(nr, nrow, irow + 1);
      if (irP != ir + 1) // skipped a row
      {
        //  //std::cout  <<"Skipped row: "  << ir + 1 << "\n";
        for (int jcol = 0; jcol < ncol; ++jcol) {
          v[mgard::get_index3(ncol, nfib, ir + 1, jcol, kf)] =
              dequantizer(work[imeg]);
          ++imeg;
        }
      }
    }

    for (int irow = 0; irow < nr; ++irow) {
      int ir = mgard::get_lindex(nr, nrow, irow);

      //      //std::cout  <<"Non skipped row: "  << ir  << "\n";
      for (int jcol = 0; jcol < nc - 1; ++jcol) {
        int jc = mgard::get_lindex(nc, ncol, jcol);
        int jcP = mgard::get_lindex(nc, ncol, jcol + 1);
        if (jcP != jc + 1) // skipped a column
        {
          v[mgard::get_index3(ncol, nfib, ir, jc + 1, kf)] =
              dequantizer(work[imeg]);
          ++imeg;
        }
      }
    }
  }

  // // 2^k+1 part //

  for (int ilevel = 0; ilevel < nlevel; ++ilevel) {

    int stride = std::pow(2, ilevel);
    int Cstride = 2 * stride;

    int fib_counter = 0;

    for (int kfib = 0; kfib < nf; kfib += stride) {
      int kf = mgard::get_lindex(nf, nfib, kfib);
      int row_counter = 0;

      if (fib_counter % 2 == 0) {
        for (int irow = 0; irow < nr; irow += stride) {
          int ir = mgard::get_lindex(nr, nrow, irow);
          if (row_counter % 2 == 0) {
            for (int jcol = Cstride; jcol < nc; jcol += Cstride) {
              int jc = mgard::get_lindex(nc, ncol, jcol - stride);
              v[mgard::get_index3(ncol, nfib, ir, jc, kf)] =
                  dequantizer(work[imeg]);
              ++imeg;
            }

          } else {
            for (int jcol = 0; jcol < nc; jcol += stride) {
              int jc = mgard::get_lindex(nc, ncol, jcol);
              v[mgard::get_index3(ncol, nfib, ir, jc, kf)] =
                  dequantizer(work[imeg]);
              ++imeg;
            }
          }
          ++row_counter;
        }
      } else {
        for (int irow = 0; irow < nr; irow += stride) {
          int ir = mgard::get_lindex(nr, nrow, irow);
          for (int jcol = 0; jcol < nc; jcol += stride) {
            int jc = mgard::get_lindex(nc, ncol, jcol);
            v[mgard::get_index3(ncol, nfib, ir, jc, kf)] =
                dequantizer(work[imeg]);
            ++imeg;
          }
        }
      }
      ++fib_counter;
    }
  }

  // last level
  int stride = std::pow(2, nlevel);

  for (int irow = 0; irow < nr; irow += stride) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < nc; jcol += stride) {
      int jc = mgard::get_lindex(nc, ncol, jcol);
      for (int kfib = 0; kfib < nf; kfib += stride) {
        int kf = mgard::get_lindex(nf, nfib, kfib);
        v[mgard::get_index3(ncol, nfib, ir, jc, kf)] = dequantizer(work[imeg]);
        ++imeg;
      }
    }
  }

  // std::cout  << "Mega count : "<< imeg << "\n";
}

template <typename Real>
void dequantize_3D(const int nr, const int nc, const int nf, const int nrow,
                   const int ncol, const int nfib, const int nlevel, Real *v,
                   std::vector<int> &work, const std::vector<Real> &coords_x,
                   const std::vector<Real> &coords_y,
                   const std::vector<Real> &coords_z, Real s) {
  int imeg = 0; // mega-counter

  const int size_ratio = sizeof(Real) / sizeof(int);
  Real q; // quantizing factor

  std::memcpy(&q, work.data(), sizeof(Real));
  imeg += size_ratio;

  // std::cout  << "Read quantizeredert " << q << "\n";

  {

    const Real dx = mgard_gen::get_h_l(coords_x, ncol, ncol, 0, 1);
    const Real dy = mgard_gen::get_h_l(coords_y, nrow, nrow, 0, 1);
    const Real dz = mgard_gen::get_h_l(coords_z, nfib, nfib, 0, 1);
    const Real vol = dx * dy * dz;

    const mgard::LinearDequantizer<int, Real> dequantizer(
        q / (std::sqrt(vol) * std::pow(2, s * (nlevel))));

    for (int kfib = 0; kfib < nf - 1; ++kfib) {
      int kf = mgard::get_lindex(nf, nfib, kfib);
      int kfp = mgard::get_lindex(nf, nfib, kfib + 1);

      if (kfp != kf + 1) // skipped a plane
      {
        for (int irow = 0; irow < nrow; ++irow) {
          for (int jcol = 0; jcol < ncol; ++jcol) {
            v[mgard::get_index3(ncol, nfib, irow, jcol, kf + 1)] =
                dequantizer(work[imeg]);
            ++imeg;
          }
        }
      }
    }

    int count_row = 0;
    int count_col = 0;
    int count_sol = 0;

    for (int kfib = 0; kfib < nf; ++kfib) {
      int kf = mgard::get_lindex(nf, nfib, kfib);
      for (int irow = 0; irow < nr - 1; ++irow) {
        int ir = mgard::get_lindex(nr, nrow, irow);
        int irP = mgard::get_lindex(nr, nrow, irow + 1);
        if (irP != ir + 1) // skipped a row
        {
          //  //std::cout  <<"Skipped row: "  << ir + 1 << "\n";
          for (int jcol = 0; jcol < ncol; ++jcol) {
            v[mgard::get_index3(ncol, nfib, ir + 1, jcol, kf)] =
                dequantizer(work[imeg]);
            ++imeg;
          }
        }
      }

      for (int irow = 0; irow < nr; ++irow) {
        int ir = mgard::get_lindex(nr, nrow, irow);

        //      //std::cout  <<"Non skipped row: "  << ir  << "\n";
        for (int jcol = 0; jcol < nc - 1; ++jcol) {
          int jc = mgard::get_lindex(nc, ncol, jcol);
          int jcP = mgard::get_lindex(nc, ncol, jcol + 1);
          if (jcP != jc + 1) // skipped a column
          {
            v[mgard::get_index3(ncol, nfib, ir, jc + 1, kf)] =
                dequantizer(work[imeg]);
            ++imeg;
          }
        }
      }
    }
  }

  // // 2^k+1 part //
  for (int ilevel = 0; ilevel < nlevel; ++ilevel) {
    int stride = std::pow(2, ilevel);
    int Cstride = 2 * stride;

    int fib_counter = 0;

    const Real dx = get_h_l(coords_x, nc, ncol, 0, stride);
    const Real dy = get_h_l(coords_y, nr, nrow, 0, stride);
    const Real dz = get_h_l(coords_z, nf, nfib, 0, stride);
    const Real vol = dx * dy * dz;

    const mgard::LinearDequantizer<int, Real> dequantizer(
        q / (std::sqrt(vol) * std::pow(2, s * (nlevel - ilevel))));
    // std::cout  << "Volume : " << ilevel << "\t"<< vol << std::endl;
    // std::cout  << "Stride : " << stride << "\t"<< vol << std::endl;

    for (int kfib = 0; kfib < nf; kfib += stride) {
      int kf = mgard::get_lindex(nf, nfib, kfib);
      int row_counter = 0;

      if (fib_counter % 2 == 0) {
        for (int irow = 0; irow < nr; irow += stride) {
          int ir = mgard::get_lindex(nr, nrow, irow);
          if (row_counter % 2 == 0) {
            for (int jcol = Cstride; jcol < nc; jcol += Cstride) {
              int jc = mgard::get_lindex(nc, ncol, jcol - stride);
              v[mgard::get_index3(ncol, nfib, ir, jc, kf)] =
                  dequantizer(work[imeg]);
              ++imeg;
            }

          } else {
            for (int jcol = 0; jcol < nc; jcol += stride) {
              int jc = mgard::get_lindex(nc, ncol, jcol);
              v[mgard::get_index3(ncol, nfib, ir, jc, kf)] =
                  dequantizer(work[imeg]);
              ++imeg;
            }
          }
          ++row_counter;
        }
      } else {
        for (int irow = 0; irow < nr; irow += stride) {
          int ir = mgard::get_lindex(nr, nrow, irow);
          for (int jcol = 0; jcol < nc; jcol += stride) {
            int jc = mgard::get_lindex(nc, ncol, jcol);
            v[mgard::get_index3(ncol, nfib, ir, jc, kf)] =
                dequantizer(work[imeg]);
            ++imeg;
          }
        }
      }
      ++fib_counter;
    }
  }

  // last level
  {

    int stride = std::pow(2, nlevel);
    const Real dx = get_h_l(coords_x, nc, ncol, 0, stride);
    const Real dy = get_h_l(coords_y, nr, nrow, 0, stride);
    const Real dz = get_h_l(coords_z, nf, nfib, 0, stride);
    const Real vol = dx * dy * dz;

    const mgard::LinearDequantizer<int, Real> dequantizer(q / std::sqrt(vol));
    // std::cout  << "Volume : " << nlevel << "\t"<< vol << std::endl;
    // std::cout  << "Stride : " << stride << "\t"<< vol << std::endl;

    for (int irow = 0; irow < nr; irow += stride) {
      int ir = mgard::get_lindex(nr, nrow, irow);
      for (int jcol = 0; jcol < nc; jcol += stride) {
        int jc = mgard::get_lindex(nc, ncol, jcol);
        for (int kfib = 0; kfib < nf; kfib += stride) {
          int kf = mgard::get_lindex(nf, nfib, kfib);
          v[mgard::get_index3(ncol, nfib, ir, jc, kf)] =
              dequantizer(work[imeg]);
          ++imeg;
        }
      }
    }
  }

  // std::cout  << "Mega count : "<< imeg << "\n";
}

template <typename Real>
void dequantize_2D(const int nr, const int nc, const int nrow, const int ncol,
                   const int nlevel, Real *v, std::vector<int> &work,
                   const std::vector<Real> &coords_x,
                   const std::vector<Real> &coords_y, Real s) {

  const int size_ratio = sizeof(Real) / sizeof(int);
  Real q; // quantizing factor

  std::memcpy(&q, work.data(), sizeof(Real));

  // std::cout  << "Read quantizeredert val" << q << "\n";

  int imeg = 0;
  imeg += size_ratio;

  // level -1, first level for non 2^k+1
  {

    const Real dx = mgard_gen::get_h_l(coords_x, ncol, ncol, 0, 1);
    const Real dy = mgard_gen::get_h_l(coords_y, nrow, nrow, 0, 1);
    const Real vol = dx * dy;

    const mgard::LinearDequantizer<int, Real> dequantizer(
        q / (std::sqrt(vol) * std::pow(2, s * nlevel)));

    int count_row = 0;
    int count_col = 0;

    for (int irow = 0; irow < nr - 1; ++irow) {
      int ir = mgard::get_lindex(nr, nrow, irow);
      int irP = mgard::get_lindex(nr, nrow, irow + 1);
      if (irP != ir + 1) // skipped a row
      {
        for (int jcol = 0; jcol < ncol; ++jcol) {
          v[mgard::get_index(ncol, ir + 1, jcol)] = dequantizer(work[imeg]);
          ++imeg;
        }
      }
    }

    for (int irow = 0; irow < nr; ++irow) {
      int ir = mgard::get_lindex(nr, nrow, irow);

      for (int jcol = 0; jcol < nc - 1; ++jcol) {
        int jc = mgard::get_lindex(nc, ncol, jcol);
        int jcP = mgard::get_lindex(nc, ncol, jcol + 1);
        if (jcP != jc + 1) // skipped a column
        {
          v[mgard::get_index(ncol, ir, jc + 1)] = dequantizer(work[imeg]);
          ++imeg;
        }
      }
    }
  }

  // // 2^k+1 part //
  for (int ilevel = 0; ilevel < nlevel; ++ilevel) {
    int stride = std::pow(2, ilevel);
    int Cstride = 2 * stride;

    const Real dx = get_h_l(coords_x, nc, ncol, 0, stride);
    const Real dy = get_h_l(coords_y, nr, nrow, 0, stride);
    const Real vol = dx * dy;

    const mgard::LinearDequantizer<int, Real> dequantizer(
        q / (std::sqrt(vol) * std::pow(2, s * (nlevel - ilevel))));

    int row_counter = 0;

    for (int irow = 0; irow < nr; irow += stride) {
      int ir = mgard::get_lindex(nr, nrow, irow);
      if (row_counter % 2 == 0) {
        for (int jcol = Cstride; jcol < nc; jcol += Cstride) {
          int jc = mgard::get_lindex(nc, ncol, jcol - stride);
          v[mgard::get_index(ncol, ir, jc)] = dequantizer(work[imeg]);
          ++imeg;
        }

      } else {
        for (int jcol = 0; jcol < nc; jcol += stride) {
          int jc = mgard::get_lindex(nc, ncol, jcol);
          v[mgard::get_index(ncol, ir, jc)] = dequantizer(work[imeg]);
          ++imeg;
        }
      }
      ++row_counter;
    }
  }

  // last level -> L=0
  {

    int stride = std::pow(2, nlevel);
    const Real dx = get_h_l(coords_x, nc, ncol, 0, stride);
    const Real dy = get_h_l(coords_y, nr, nrow, 0, stride);
    const Real vol = dx * dy;

    const mgard::LinearDequantizer<int, Real> dequantizer(q / std::sqrt(vol));

    for (int irow = 0; irow < nr; irow += stride) {
      int ir = mgard::get_lindex(nr, nrow, irow);
      for (int jcol = 0; jcol < nc; jcol += stride) {
        int jc = mgard::get_lindex(nc, ncol, jcol);
        v[mgard::get_index(ncol, ir, jc)] = dequantizer(work[imeg]);
        ++imeg;
      }
    }
  }

  // std::cout  << "Read in: " << imeg <<"\n";
}

} // namespace mgard_gen

namespace mgard_2d {

namespace mgard_common {

template <typename Real> Real max_norm(const std::vector<Real> &v) {
  Real norm = 0;

  for (int i = 0; i < v.size(); ++i) {
    Real ntest = std::abs(v[i]);
    if (ntest > norm)
      norm = ntest;
  }
  return norm;
}

template <typename Real>
Real get_h(const std::vector<Real> &coords, int i, int stride) {
  return (i + stride - i);
}

template <typename Real>
Real get_dist(const std::vector<Real> &coords, int i, int j) {
  return (j - i);
}

template <typename Real>
void qread_2D_interleave(const int nrow, const int ncol, const int nlevel,
                         Real *v, std::string infile) {
  int buff_size = 128 * 1024;
  unsigned char unzip_buffer[buff_size];
  int int_buffer[buff_size / sizeof(int)];
  unsigned int unzipped_bytes, total_bytes = 0;
  Real quantum;

  gzFile in_file_z = gzopen(infile.c_str(), "r");
  // std::cout  << in_file_z <<"\n";

  unzipped_bytes = gzread(in_file_z, unzip_buffer,
                          sizeof(Real)); // read the quantization constant
  std::memcpy(&quantum, &unzip_buffer, unzipped_bytes);
  const mgard::LinearDequantizer<int, Real> dequantizer(quantum);

  int last = 0;
  while (true) {
    unzipped_bytes = gzread(in_file_z, unzip_buffer, buff_size);
    if (unzipped_bytes > 0) {
      total_bytes += unzipped_bytes;
      int num_int = unzipped_bytes / sizeof(int);

      std::memcpy(&int_buffer, &unzip_buffer, unzipped_bytes);
      for (int i = 0; i < num_int; ++i) {
        v[last] = dequantizer(int_buffer[i]);
        ++last;
      }

    } else {
      break;
    }
  }

  gzclose(in_file_z);
}

template <typename Real>
void qwrite_2D_interleave(const int nrow, const int ncol, const int nlevel,
                          const int l, Real *v, const Real tol, const Real norm,
                          const std::string outfile) {

  //    int stride = std::pow(2,l);//current stride

  const mgard::LinearQuantizer<Real, int> quantizer(2 * norm * tol /
                                                    (nlevel + 1));

  gzFile out_file = gzopen(outfile.c_str(), "w6b");
  int prune_count = 0;
  gzwrite(out_file, &quantizer.quantum, sizeof(Real));

  for (auto index = 0; index < ncol * nrow; ++index) {
    const int n = quantizer(v[index]);
    if (n == 0)
      ++prune_count;
    gzwrite(out_file, &n, sizeof(int));
  }

  // std::cout  << "Pruned : "<< prune_count << " Reduction : " << (Real)
  // nrow*ncol /(nrow*ncol - prune_count) << "\n";
  gzclose(out_file);
}

} // namespace mgard_common

namespace mgard_cannon {

template <typename Real>
void assign_num_level(const int nrow, const int ncol, const int l, Real *v,
                      Real num) {
  // set the value of nodal values at level l to number num

  int stride = std::pow(2, l); // current stride

  for (int irow = 0; irow < nrow; irow += stride) {
    for (int jcol = 0; jcol < ncol; jcol += stride) {
      v[mgard::get_index(ncol, irow, jcol)] = num;
    }
  }
}

template <typename Real>
void subtract_level(const int nrow, const int ncol, const int l, Real *v,
                    Real *work) {
  // v += work at level l
  int stride = std::pow(2, l); // current stride

  for (int irow = 0; irow < nrow; irow += stride) {
    for (int jcol = 0; jcol < ncol; jcol += stride) {
      v[mgard::get_index(ncol, irow, jcol)] -=
          work[mgard::get_index(ncol, irow, jcol)];
    }
  }
}

template <typename Real>
void pi_lminus1(const int l, std::vector<Real> &v,
                const std::vector<Real> &coords) {
  int n = v.size();
  int nlevel = static_cast<int>(std::log2(v.size() - 1));
  int my_level = nlevel - l;
  int stride = std::pow(2, l); // current stride
  //  int Pstride = stride/2; //finer stride
  int Cstride = stride * 2; // coarser stride

  if (my_level != 0) {
    for (int i = Cstride; i < n; i += Cstride) {
      Real h1 = mgard_common::get_h(coords, i - Cstride, stride);
      Real h2 = mgard_common::get_h(coords, i - stride, stride);
      Real hsum = h1 + h2;
      v[i - stride] -= (h1 * v[i] + h2 * v[i - Cstride]) / hsum;
    }
  }
}

template <typename Real>
void restriction(const int l, std::vector<Real> &v,
                 const std::vector<Real> &coords) {
  int stride = std::pow(2, l);
  int Pstride = stride / 2; // finer stride
  int n = v.size();

  // calculate the result of restrictionion

  Real h1 = mgard_common::get_h(coords, 0, Pstride);
  Real h2 = mgard_common::get_h(coords, Pstride, Pstride);
  Real hsum = h1 + h2;

  v.front() += h2 * v[Pstride] / hsum; // first element

  for (int i = stride; i <= n - stride; i += stride) {
    v[i] += h1 * v[i - Pstride] / hsum;
    h1 = mgard_common::get_h(coords, i, Pstride);
    h2 = mgard_common::get_h(coords, i + Pstride, Pstride);
    hsum = h1 + h2;
    v[i] += h2 * v[i + Pstride] / hsum;
  }
  v.back() += h1 * v[n - Pstride - 1] / hsum; // last element
}

template <typename Real>
void prolongate(const int l, std::vector<Real> &v,
                const std::vector<Real> &coords) {

  int stride = std::pow(2, l);
  int Pstride = stride / 2;
  int n = v.size();

  for (int i = stride; i < n; i += stride) {
    Real h1 = mgard_common::get_h(coords, i - stride, Pstride);
    Real h2 = mgard_common::get_h(coords, i - Pstride, Pstride);

    v[i - Pstride] = (h2 * v[i - stride] + h1 * v[i]) / (h1 + h2);
  }
}

template <typename Real>
void solve_tridiag_M(const int l, std::vector<Real> &v,
                     const std::vector<Real> &coords) {

  //  int my_level = nlevel - l;
  int stride = std::pow(2, l); // current stride

  Real am, bm, h1, h2;
  int n = v.size();

  am = 2.0 * mgard_common::get_h(coords, 0,
                                 stride); // first element of upper diagonal U.

  //    bm = mgard_common::get_h(coords, 0, stride) / am;
  bm = mgard_common::get_h(coords, 0, stride) / am;
  int nlevel = static_cast<int>(std::log2(v.size() - 1));
  //    //std::cout  << nlevel;
  int nc = std::pow(2, nlevel - l) + 1;
  std::vector<Real> coeff(n);
  int counter = 1;
  coeff.front() = am;

  // forward sweep
  for (int i = stride; i < n - 1; i += stride) {
    h1 = mgard_common::get_h(coords, i - stride, stride);
    h2 = mgard_common::get_h(coords, i, stride);
    //        //std::cout  << i<< "\t"<< v[i-stride] << "\t" << h1<< "\t"<<
    //        h2<<"\n";
    v[i] -= v[i - stride] * bm;

    am = 2.0 * (h1 + h2) - bm * h1;
    bm = h2 / am;
    //        //std::cout  <<  am<< "\t"<< bm<<"\n";

    coeff.at(counter) = am;
    ++counter;
  }

  h2 = mgard_common::get_h(coords, n - 1 - stride, stride);
  am = 2.0 * h2 - bm * h2; // a_n = 2 - b_(n-1)
  //    //std::cout  << h1 << "\t"<< h2<<"\n";
  v[n - 1] -= v[n - 1 - stride] * bm;

  coeff.at(counter) = am;

  // backward sweep

  v[n - 1] /= am;
  --counter;

  for (int i = n - 1 - stride; i >= 0; i -= stride) {
    // h1 = mgard_common::get_h(coords, i-stride, stride);
    h2 = mgard_common::get_h(coords, i, stride);
    v[i] = (v[i] - h2 * v[i + stride]) / coeff.at(counter);
    --counter;
    //        bm = (2.0*(h1+h2) - am) / h1 ;
    // am = 1.0 / bm;
  }
  // h1 = mgard_common::get_h(coords, 0, stride);
  //    //std::cout  << h1 << "\n";
  //    v[0] = (v[0] - h1*v[1])/coeff[0];
}

template <typename Real>
void mass_matrix_multiply(const int l, std::vector<Real> &v,
                          const std::vector<Real> &coords) {

  int stride = std::pow(2, l);
  int n = v.size();
  Real temp1, temp2;

  // Mass matrix times nodal value-vec
  temp1 = v.front(); // save u(0) for later use
  v.front() = 2.0 * mgard_common::get_h(coords, 0, stride) * temp1 +
              mgard_common::get_h(coords, 0, stride) * v[stride];
  for (int i = stride; i <= n - 1 - stride; i += stride) {
    temp2 = v[i];
    v[i] = mgard_common::get_h(coords, i - stride, stride) * temp1 +
           2 *
               (mgard_common::get_h(coords, i - stride, stride) +
                mgard_common::get_h(coords, i, stride)) *
               temp2 +
           mgard_common::get_h(coords, i, stride) * v[i + stride];
    temp1 = temp2; // save u(n) for later use
  }
  v[n - 1] = mgard_common::get_h(coords, n - stride - 1, stride) * temp1 +
             2 * mgard_common::get_h(coords, n - stride - 1, stride) * v[n - 1];
}

template <typename Real>
void write_level_2D(const int nrow, const int ncol, const int l, Real *v,
                    std::ofstream &outfile) {
  int stride = std::pow(2, l);
  //  int nrow = std::pow(2, nlevel_row) + 1;
  // int ncol = std::pow(2, nlevel_col) + 1;

  for (int irow = 0; irow < nrow; irow += stride) {
    for (int jcol = 0; jcol < ncol; jcol += stride) {
      outfile.write(
          reinterpret_cast<char *>(&v[mgard::get_index(ncol, irow, jcol)]),
          sizeof(Real));
    }
  }
}

template <typename Real>
void copy_level(const int nrow, const int ncol, const int l, Real *v,
                std::vector<Real> &work) {

  int stride = std::pow(2, l); // current stride

  for (int irow = 0; irow < nrow; irow += stride) {
    for (int jcol = 0; jcol < ncol; jcol += stride) {
      work[mgard::get_index(ncol, irow, jcol)] =
          v[mgard::get_index(ncol, irow, jcol)];
    }
  }
}

} // namespace mgard_cannon

namespace mgard_gen {

template <typename Real>
Real *get_ref(std::vector<Real> &v, const int n, const int no,
              const int i) // return reference to logical element
{
  // no: original number of points
  // n : number of points at next coarser level (L-1) with  2^k+1 nodes
  // may not work for the last element!
  Real *ref;
  if (i != n - 1) {
    ref = &v[floor(((Real)no - 2.0) / ((Real)n - 2.0) * i)];
  } else if (i == n - 1) {
    ref = &v[no - 1];
  }
  return ref;
  //    return &v[floor(((no-2)/(n-2))*i ) ];
}

template <typename Real>
Real get_h_l(const std::vector<Real> &coords, const int n, const int no, int i,
             int stride) {

  //    return (*get_ref(coords, n, no, i+stride) - *get_ref(coords, n, no, i));
  return (mgard::get_lindex(n, no, i + stride) - mgard::get_lindex(n, no, i));
}

template <typename Real>
void write_level_2D_l(const int l, Real *v, std::ofstream &outfile, int nr,
                      int nc, int nrow, int ncol) {
  int stride = std::pow(2, l);
  //  int nrow = std::pow(2, nlevel_row) + 1;
  // int ncol = std::pow(2, nlevel_col) + 1;

  for (int irow = 0; irow < nr; irow += stride) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < nc; jcol += stride) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      outfile.write(
          reinterpret_cast<char *>(&v[mgard::get_index(ncol, ir, jr)]),
          sizeof(Real));
    }
  }
}

template <typename Real>
void copy_level_l(const int l, Real *v, Real *work, int nr, int nc, int nrow,
                  int ncol) {
  // work_l = v_l
  int stride = std::pow(2, l); // current stride

  for (int irow = 0; irow < nr; irow += stride) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < nc; jcol += stride) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      work[mgard::get_index(ncol, ir, jr)] = v[mgard::get_index(ncol, ir, jr)];
    }
  }
}

template <typename Real>
void subtract_level_l(const int l, Real *v, Real *work, int nr, int nc,
                      int nrow, int ncol) {
  // v -= work at level l

  int stride = std::pow(2, l); // current stride

  for (int irow = 0; irow < nr; irow += stride) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < nc; jcol += stride) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      v[mgard::get_index(ncol, ir, jr)] -= work[mgard::get_index(ncol, ir, jr)];
    }
  }
}

template <typename Real>
void pi_lminus1_l(const int l, std::vector<Real> &v,
                  const std::vector<Real> &coords, int n, int no) {
  int nlevel = static_cast<int>(std::log2(n - 1));
  int my_level = nlevel - l;
  int stride = std::pow(2, l); // current stride
  //  int Pstride = stride/2; //finer stride
  int Cstride = stride * 2; // coarser stride

  if (my_level != 0) {
    for (int i = Cstride; i < n - 1; i += Cstride) {
      Real h1 = get_h_l(coords, n, no, i - Cstride, stride);
      Real h2 = get_h_l(coords, n, no, i - stride, stride);
      Real hsum = h1 + h2;
      *get_ref(v, n, no, i - stride) -=
          (h1 * (*get_ref(v, n, no, i)) +
           h2 * (*get_ref(v, n, no, i - Cstride))) /
          hsum;
    }

    Real h1 = get_h_l(coords, n, no, n - 1 - Cstride, stride);
    Real h2 = get_h_l(coords, n, no, n - 1 - stride, stride);
    Real hsum = h1 + h2;
    *get_ref(v, n, no, n - 1 - stride) -=
        (h1 * (v.back()) + h2 * (*get_ref(v, n, no, n - 1 - Cstride))) / hsum;
  }
}

template <typename Real>
void pi_lminus1_first(std::vector<Real> &v, const std::vector<Real> &coords,
                      int n, int no) {

  for (int i = 0; i < n - 1; ++i) {
    int i_logic = mgard::get_lindex(n, no, i);
    int i_logicP = mgard::get_lindex(n, no, i + 1);

    if (i_logicP != i_logic + 1) {
      //          //std::cout  << i_logic +1 << "\t" << i_logicP<<"\n";
      Real h1 = mgard_common::get_dist(coords, i_logic, i_logic + 1);
      Real h2 = mgard_common::get_dist(coords, i_logic + 1, i_logicP);
      Real hsum = h1 + h2;
      v[i_logic + 1] -= (h2 * v[i_logic] + h1 * v[i_logicP]) / hsum;
    }
  }
}

template <typename Real>
void pi_Ql_first(const int nc, const int ncol, const int l, Real *v,
                 const std::vector<Real> &coords_x,
                 std::vector<Real> &row_vec) {
  // Restrict data to coarser level

  int stride = 1; // current stride
  //  int Pstride = stride/2; //finer stride
  //    int Cstride = 2; // coarser stride

  for (int jcol = 0; jcol < ncol; ++jcol) {
    // int jcol_r = mgard::get_lindex(nc, ncol, jcol);
    // std::cerr << irow_r << "\t"<< jcol_r << "\n";
    row_vec[jcol] = v[jcol];
  }

  pi_lminus1_first(row_vec, coords_x, nc, ncol);

  for (int jcol = 0; jcol < ncol; ++jcol) {
    //            int jcol_r = mgard::get_lindex(nc, ncol, jcol);
    v[jcol] = row_vec[jcol];
  }
}

template <typename Real>
void pi_Ql_first(const int nr, const int nc, const int nrow, const int ncol,
                 const int l, Real *v, const std::vector<Real> &coords_x,
                 const std::vector<Real> &coords_y, std::vector<Real> &row_vec,
                 std::vector<Real> &col_vec) {
  // Restrict data to coarser level

  int stride = 1; // current stride
  //  int Pstride = stride/2; //finer stride
  //    int Cstride = 2; // coarser stride

  for (int irow = 0; irow < nr;
       irow += stride) // Do the rows existing  in the coarser level
  {
    int irow_r = mgard::get_lindex(
        nr, nrow, irow); // get the real location of logical index irow

    for (int jcol = 0; jcol < ncol; ++jcol) {
      // int jcol_r = mgard::get_lindex(nc, ncol, jcol);
      // std::cerr << irow_r << "\t"<< jcol_r << "\n";

      row_vec[jcol] = v[mgard::get_index(ncol, irow_r, jcol)];
    }

    pi_lminus1_first(row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      //            int jcol_r = mgard::get_lindex(nc, ncol, jcol);
      v[mgard::get_index(ncol, irow_r, jcol)] = row_vec[jcol];
    }

    // if( irP != ir +1) //are we skipping the next row?
    //   {
    //     ++irow;
    //   }
  }

  if (nrow > 1) {
    for (int jcol = 0; jcol < nc;
         jcol += stride) // Do the columns existing  in the coarser level
    {
      int jcol_r = mgard::get_lindex(nc, ncol, jcol);
      //            int jr  = mgard::get_lindex(nc, ncol, jcol);
      // int jrP = mgard::get_lindex(nc, ncol, jcol+1);

      for (int irow = 0; irow < nrow; ++irow) {
        col_vec[irow] = v[mgard::get_index(ncol, irow, jcol_r)];
      }

      pi_lminus1_first(col_vec, coords_y, nr, nrow);

      for (int irow = 0; irow < nrow; ++irow) {
        v[mgard::get_index(ncol, irow, jcol_r)] = col_vec[irow];
      }
    }
  }

  //        Now the new-new stuff
  for (int irow = 0; irow < nr - 1; ++irow) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    int irP = mgard::get_lindex(nr, nrow, irow + 1);

    for (int jcol = 0; jcol < nc - 1; ++jcol) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      int jrP = mgard::get_lindex(nc, ncol, jcol + 1);

      if ((irP != ir + 1) &&
          (jrP != jr + 1)) // we skipped both a row and a column
      {

        Real q11 = v[mgard::get_index(ncol, ir, jr)];
        Real q12 = v[mgard::get_index(ncol, irP, jr)];
        Real q21 = v[mgard::get_index(ncol, ir, jrP)];
        Real q22 = v[mgard::get_index(ncol, irP, jrP)];

        Real x1 = 0.0;
        Real y1 = 0.0;

        Real x2 = mgard_common::get_dist(coords_x, jr, jrP);
        Real y2 = mgard_common::get_dist(coords_y, ir, irP);

        Real x = mgard_common::get_dist(coords_x, jr, jr + 1);
        Real y = mgard_common::get_dist(coords_y, ir, ir + 1);

        Real temp =
            mgard::interpolate(q11, q12, q21, q22, x1, x2, y1, y2, x, y);

        v[mgard::get_index(ncol, ir + 1, jr + 1)] -= temp;
      }
    }
  }
}

template <typename Real>
void pi_Ql(const int nc, const int ncol, const int l, Real *v,
           const std::vector<Real> &coords_x, std::vector<Real> &row_vec) {
  // Restrict data to coarser level

  int stride = std::pow(2, l); // current stride
  //  int Pstride = stride/2; //finer stride
  int Cstride = stride * 2; // coarser stride

  //  std::vector<Real> row_vec(ncol), col_vec(nrow)   ;

  {
    int ir = mgard::get_lindex(
        1, 1, 0); // get the real location of logical index irow
    for (int jcol = 0; jcol < ncol; ++jcol) {
      //            int jcol_r = mgard::get_lindex(nc, ncol, jcol);
      row_vec[jcol] = v[mgard::get_index(ncol, ir, jcol)];
    }

    //        mgard_cannon::pi_lminus1(l, row_vec, coords_x);
    pi_lminus1_l(l, row_vec, coords_x, nc, ncol);
    for (int jcol = 0; jcol < ncol; ++jcol) {
      v[jcol] = row_vec[jcol];
    }
  }
}

template <typename Real>
void pi_Ql(const int nr, const int nc, const int nrow, const int ncol,
           const int l, Real *v, const std::vector<Real> &coords_x,
           const std::vector<Real> &coords_y, std::vector<Real> &row_vec,
           std::vector<Real> &col_vec) {
  // Restrict data to coarser level

  int stride = std::pow(2, l); // current stride
  //  int Pstride = stride/2; //finer stride
  int Cstride = stride * 2; // coarser stride

  //  std::vector<Real> row_vec(ncol), col_vec(nrow)   ;

  for (int irow = 0; irow < nr;
       irow += Cstride) // Do the rows existing  in the coarser level
  {
    int ir =
        mgard::get_lindex(nr, nrow,
                          irow); // get the real location of logical index irow
    for (int jcol = 0; jcol < ncol; ++jcol) {
      //            int jcol_r = mgard::get_lindex(nc, ncol, jcol);
      row_vec[jcol] = v[mgard::get_index(ncol, ir, jcol)];
    }

    //        mgard_cannon::pi_lminus1(l, row_vec, coords_x);
    pi_lminus1_l(l, row_vec, coords_x, nc, ncol);
    for (int jcol = 0; jcol < ncol; ++jcol) {
      v[mgard::get_index(ncol, ir, jcol)] = row_vec[jcol];
    }
  }

  if (nrow > 1) {
    for (int jcol = 0; jcol < nc;
         jcol += Cstride) // Do the columns existing  in the coarser level
    {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      for (int irow = 0; irow < nrow; ++irow) {
        //                int irow_r = mgard::get_lindex(nr, nrow, irow);
        col_vec[irow] = v[mgard::get_index(ncol, irow, jr)];
      }

      pi_lminus1_l(l, col_vec, coords_y, nr, nrow);
      for (int irow = 0; irow < nrow; ++irow) {
        //                int irow_r = mgard::get_lindex(nr, nrow, irow);
        v[mgard::get_index(ncol, irow, jr)] = col_vec[irow];
      }
    }

    // Now the new-new stuff
    for (int irow = stride; irow < nr; irow += Cstride) {
      int ir1 = mgard::get_lindex(nr, nrow, irow - stride);
      int ir = mgard::get_lindex(nr, nrow, irow);
      int ir2 = mgard::get_lindex(nr, nrow, irow + stride);

      for (int jcol = stride; jcol < nc; jcol += Cstride) {

        int jr1 = mgard::get_lindex(nc, ncol, jcol - stride);
        int jr = mgard::get_lindex(nc, ncol, jcol);
        int jr2 = mgard::get_lindex(nc, ncol, jcol + stride);

        Real q11 = v[mgard::get_index(ncol, ir1, jr1)];
        Real q12 = v[mgard::get_index(ncol, ir2, jr1)];
        Real q21 = v[mgard::get_index(ncol, ir1, jr2)];
        Real q22 = v[mgard::get_index(ncol, ir2, jr2)];

        Real x1 = 0.0; // relative coordinate axis centered at irow - Cstride,
                       // jcol - Cstride
        Real y1 = 0.0;
        Real x2 = mgard_common::get_dist(coords_x, jr1, jr2);
        Real y2 = mgard_common::get_dist(coords_y, ir1, ir2);

        Real x = mgard_common::get_dist(coords_x, jr1, jr);
        Real y = mgard_common::get_dist(coords_y, ir1, ir);
        Real temp =
            mgard::interpolate(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
        //              //std::cout  << temp <<"\n";
        v[mgard::get_index(ncol, ir, jr)] -= temp;
      }
    }
  }
}

template <typename Real>
void assign_num_level_l(const int l, Real *v, Real num, int nr, int nc,
                        const int nrow, const int ncol) {
  // set the value of nodal values at level l to number num

  int stride = std::pow(2, l); // current stride

  for (int irow = 0; irow < nr; irow += stride) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < nc; jcol += stride) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      v[mgard::get_index(ncol, ir, jr)] = num;
    }
  }
}

template <typename Real>
void restriction_first(std::vector<Real> &v, const std::vector<Real> &coords,
                       const int n, const int no) {
  // calculate the result of restrictionion

  for (int i = 0; i < n - 1; ++i) // loop over the logical array
  {
    int i_logic = mgard::get_lindex(n, no, i);
    int i_logicP = mgard::get_lindex(n, no, i + 1);

    if (i_logicP != i_logic + 1) // next real memory location was jumped over,
                                 // so need to restriction
    {
      Real h1 = mgard_common::get_h(coords, i_logic, 1);
      Real h2 = mgard_common::get_h(coords, i_logic + 1, 1);
      Real hsum = h1 + h2;
      // v[i_logic]  = 0.5*v[i_logic]  + 0.5*h2*v[i_logic+1]/hsum;
      // v[i_logicP] = 0.5*v[i_logicP] + 0.5*h1*v[i_logic+1]/hsum;
      v[i_logic] += h2 * v[i_logic + 1] / hsum;
      v[i_logicP] += h1 * v[i_logic + 1] / hsum;
    }
  }
}

template <typename Real>
void solve_tridiag_M_l(const int l, std::vector<Real> &v,
                       const std::vector<Real> &coords, const int n,
                       const int no) {

  //  int my_level = nlevel - l;
  int stride = std::pow(2, l); // current stride

  Real am, bm, h1, h2;
  am = 2.0 *
       get_h_l(coords, n, no, 0, stride); // first element of upper diagonal U.

  //    bm = get_h(coords, 0, stride) / am;
  bm = get_h_l(coords, n, no, 0, stride) / am;
  int nlevel = static_cast<int>(std::log2(n - 1));
  //    //std::cout  << nlevel;
  int nc = std::pow(2, nlevel - l) + 1;
  std::vector<Real> coeff(v.size());
  int counter = 1;
  coeff.front() = am;

  // forward sweep
  for (int i = stride; i < n - 1; i += stride) {
    h1 = get_h_l(coords, n, no, i - stride, stride);
    h2 = get_h_l(coords, n, no, i, stride);

    *get_ref(v, n, no, i) -= *get_ref(v, n, no, i - stride) * bm;

    am = 2.0 * (h1 + h2) - bm * h1;
    bm = h2 / am;

    coeff.at(counter) = am;
    ++counter;
  }

  h2 = get_h_l(coords, n, no, n - 1 - stride, stride);
  am = 2.0 * h2 - bm * h2;

  //    *get_ref(v, n, no, n-1) -= *get_ref(v, n, no, n-1-stride)*bm;
  v.back() -= *get_ref(v, n, no, n - 1 - stride) * bm;
  coeff.at(counter) = am;

  // backward sweep

  //    *get_ref(v, n, no, n-1) /= am;
  v.back() /= am;
  --counter;

  for (int i = n - 1 - stride; i >= 0; i -= stride) {
    h2 = get_h_l(coords, n, no, i, stride);
    *get_ref(v, n, no, i) =
        (*get_ref(v, n, no, i) - h2 * (*get_ref(v, n, no, i + stride))) /
        coeff.at(counter);

    //        *get_ref(v, n, no, i) = 3  ;

    --counter;
  }
}

template <typename Real>
void add_level_l(const int l, Real *v, Real *work, int nr, int nc, int nrow,
                 int ncol) {
  // v += work at level l

  int stride = std::pow(2, l); // current stride

  for (int irow = 0; irow < nr; irow += stride) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < nc; jcol += stride) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      v[mgard::get_index(ncol, ir, jr)] += work[mgard::get_index(ncol, ir, jr)];
    }
  }
}

template <typename Real>
void project_first(const int nr, const int nc, const int nrow, const int ncol,
                   const int l_target, Real *v, std::vector<Real> &work,
                   std::vector<Real> &coords_x, std::vector<Real> &coords_y,
                   std::vector<Real> &row_vec, std::vector<Real> &col_vec) {}

template <typename Real>
void prep_1D(const int nc, const int ncol, const int l_target, Real *v,
             std::vector<Real> &work, std::vector<Real> &coords_x,
             std::vector<Real> &row_vec) {

  int l = 0;
  //    int stride = 1;
  pi_Ql_first(nc, ncol, l, v, coords_x,
              row_vec); //(I-\Pi)u this is the initial move to 2^k+1 nodes

  mgard_cannon::copy_level(1, ncol, 0, v, work);

  assign_num_level_l(0, work.data(), static_cast<Real>(0.0), 1, nc, 1, ncol);

  for (int jcol = 0; jcol < ncol; ++jcol) {
    row_vec[jcol] = work[jcol];
  }

  mgard_cannon::mass_matrix_multiply(0, row_vec, coords_x);

  restriction_first(row_vec, coords_x, nc, ncol);

  mgard_gen::solve_tridiag_M_l(0, row_vec, coords_x, nc, ncol);

  add_level_l(0, v, row_vec.data(), 1, nc, 1, ncol);
}

template <typename Real>
void prep_2D(const int nr, const int nc, const int nrow, const int ncol,
             const int l_target, Real *v, std::vector<Real> &work,
             std::vector<Real> &coords_x, std::vector<Real> &coords_y,
             std::vector<Real> &row_vec, std::vector<Real> &col_vec) {

  int l = 0;
  //    int stride = 1;
  pi_Ql_first(nr, nc, nrow, ncol, l, v, coords_x, coords_y, row_vec,
              col_vec); //(I-\Pi)u this is the initial move to 2^k+1 nodes

  mgard_cannon::copy_level(nrow, ncol, 0, v, work);

  assign_num_level_l(0, work.data(), static_cast<Real>(0.0), nr, nc, nrow,
                     ncol);

  for (int irow = 0; irow < nrow; ++irow) {
    //        int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[mgard::get_index(ncol, irow, jcol)];
    }

    mgard_cannon::mass_matrix_multiply(0, row_vec, coords_x);

    restriction_first(row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[mgard::get_index(ncol, irow, jcol)] = row_vec[jcol];
    }
  }

  for (int irow = 0; irow < nr; ++irow) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[mgard::get_index(ncol, ir, jcol)];
    }

    mgard_gen::solve_tridiag_M_l(0, row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[mgard::get_index(ncol, ir, jcol)] = row_vec[jcol];
    }
  }

  //   //   //std::cout  << "recomposing-colsweep" << "\n";

  //     // column-sweep, this is the slow one! Need something like column_copy
  if (nrow > 1) // check if we have 1-D array..
  {
    for (int jcol = 0; jcol < ncol; ++jcol) {
      //      int jr  = mgard::get_lindex(nc,  ncol,  jcol);
      for (int irow = 0; irow < nrow; ++irow) {
        col_vec[irow] = work[mgard::get_index(ncol, irow, jcol)];
      }

      mgard_cannon::mass_matrix_multiply(0, col_vec, coords_y);

      mgard_gen::restriction_first(col_vec, coords_y, nr, nrow);

      for (int irow = 0; irow < nrow; ++irow) {
        work[mgard::get_index(ncol, irow, jcol)] = col_vec[irow];
      }
    }

    for (int jcol = 0; jcol < nc; ++jcol) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      for (int irow = 0; irow < nrow; ++irow) {
        col_vec[irow] = work[mgard::get_index(ncol, irow, jr)];
      }

      mgard_gen::solve_tridiag_M_l(0, col_vec, coords_y, nr, nrow);
      for (int irow = 0; irow < nrow; ++irow) {
        work[mgard::get_index(ncol, irow, jr)] = col_vec[irow];
      }
    }
  }
  add_level_l(0, v, work.data(), nr, nc, nrow, ncol);
}

template <typename Real>
void mass_mult_l(const int l, std::vector<Real> &v,
                 const std::vector<Real> &coords, const int n, const int no) {

  int stride = std::pow(2, l);
  Real temp1, temp2;
  Real h1, h2;

  // Mass matrix times nodal value-vec
  temp1 = v.front(); // save u(0) for later use

  h1 = get_h_l(coords, n, no, 0, stride);

  v.front() = 2.0 * h1 * temp1 + h1 * (*get_ref(v, n, no, stride));

  for (int i = stride; i <= n - 1 - stride; i += stride) {
    temp2 = *get_ref(v, n, no, i);
    h1 = get_h_l(coords, n, no, i - stride, stride);
    h2 = get_h_l(coords, n, no, i, stride);

    *get_ref(v, n, no, i) = h1 * temp1 + 2 * (h1 + h2) * temp2 +
                            h2 * (*get_ref(v, n, no, i + stride));
    temp1 = temp2; // save u(n) for later use
  }
  v.back() = get_h_l(coords, n, no, n - stride - 1, stride) * temp1 +
             2 * get_h_l(coords, n, no, n - stride - 1, stride) * v.back();
}

template <typename Real>
void restriction_l(const int l, std::vector<Real> &v,
                   const std::vector<Real> &coords, const int n, const int no) {
  int stride = std::pow(2, l);
  int Pstride = stride / 2; // finer stride

  // calculate the result of restrictionion

  Real h1 = get_h_l(coords, n, no, 0, Pstride);
  Real h2 = get_h_l(coords, n, no, Pstride, Pstride);
  Real hsum = h1 + h2;

  v.front() += h2 * (*get_ref(v, n, no, Pstride)) / hsum; // first element

  for (int i = stride; i <= n - stride; i += stride) {
    *get_ref(v, n, no, i) += h1 * (*get_ref(v, n, no, i - Pstride)) / hsum;
    h1 = get_h_l(coords, n, no, i, Pstride);
    h2 = get_h_l(coords, n, no, i + Pstride, Pstride);
    hsum = h1 + h2;
    *get_ref(v, n, no, i) += h2 * (*get_ref(v, n, no, i + Pstride)) / hsum;
  }
  v.back() += h1 * (*get_ref(v, n, no, n - Pstride - 1)) / hsum; // last element
}

template <typename Real>
void prolongate_l(const int l, std::vector<Real> &v, std::vector<Real> &coords,
                  int n, int no) {

  int stride = std::pow(2, l);
  int Pstride = stride / 2;

  for (int i = stride; i < n; i += stride) {
    Real h1 = get_h_l(coords, n, no, i - stride, Pstride);
    Real h2 = get_h_l(coords, n, no, i - Pstride, Pstride);
    Real hsum = h1 + h2;

    *get_ref(v, n, no, i - Pstride) =
        (h2 * (*get_ref(v, n, no, i - stride)) + h1 * (*get_ref(v, n, no, i))) /
        hsum;
  }

  // Real h1 = get_h_l(coords, n, no, n-1-stride,  Pstride);
  // Real h2 = get_h_l(coords, n, no, n-1-Pstride, Pstride);
  // Real hsum = h1+h2;

  // *get_ref(v, n,  no,  n-1-Pstride) = ( h2*(*get_ref(v, n,  no,  n-1-stride))
  // + h1*(v.back()) )/hsum;
}

// Gary old branch.
template <typename Real>
void refactor_1D(const int nc, const int ncol, const int l_target, Real *v,
                 std::vector<Real> &work, std::vector<Real> &coords_x,
                 std::vector<Real> &row_vec) {
  for (int l = 0; l < l_target; ++l) {
    int stride = std::pow(2, l); // current stride
    int Cstride = stride * 2;    // coarser stride

    // -> change funcs in pi_QL to use _l functions, otherwise distances are
    // wrong!!!
    pi_Ql(nc, ncol, l, v, coords_x,
          row_vec); // rename!. v@l has I-\Pi_l Q_l+1 u

    copy_level_l(l, v, work.data(), 1, nc, 1, ncol);
    assign_num_level_l(l + 1, work.data(), static_cast<Real>(0.0), 1, nc, 1,
                       ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[jcol];
    }

    mgard_gen::mass_mult_l(l, row_vec, coords_x, nc, ncol);

    mgard_gen::restriction_l(l + 1, row_vec, coords_x, nc, ncol);

    mgard_gen::solve_tridiag_M_l(l + 1, row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[jcol] = row_vec[jcol];
    }

    // Solved for (z_l, phi_l) = (c_{l+1}, vl)
    add_level_l(l + 1, v, work.data(), 1, nc, 1, ncol);
  }
}

template <typename Real>
void refactor_2D(const int nr, const int nc, const int nrow, const int ncol,
                 const int l_target, Real *v, std::vector<Real> &work,
                 std::vector<Real> &coords_x, std::vector<Real> &coords_y,
                 std::vector<Real> &row_vec, std::vector<Real> &col_vec) {
  // refactor
  //    //std::cout  << "I am the general refactorer!" <<"\n";
  for (int l = 0; l < l_target; ++l) {
    int stride = std::pow(2, l); // current stride
    int Cstride = stride * 2;    // coarser stride

    // -> change funcs in pi_QL to use _l functions, otherwise distances are
    // wrong!!!
    pi_Ql(nr, nc, nrow, ncol, l, v, coords_x, coords_y, row_vec,
          col_vec); // rename!. v@l has I-\Pi_l Q_l+1 u

    copy_level_l(l, v, work.data(), nr, nc, nrow, ncol);
    assign_num_level_l(l + 1, work.data(), static_cast<Real>(0.0), nr, nc, nrow,
                       ncol);

    // row-sweep
    for (int irow = 0; irow < nr; ++irow) {
      int ir = mgard::get_lindex(nr, nrow, irow);
      for (int jcol = 0; jcol < ncol; ++jcol) {
        row_vec[jcol] = work[mgard::get_index(ncol, ir, jcol)];
      }

      mgard_gen::mass_mult_l(l, row_vec, coords_x, nc, ncol);

      mgard_gen::restriction_l(l + 1, row_vec, coords_x, nc, ncol);

      mgard_gen::solve_tridiag_M_l(l + 1, row_vec, coords_x, nc, ncol);

      for (int jcol = 0; jcol < ncol; ++jcol) {
        work[mgard::get_index(ncol, ir, jcol)] = row_vec[jcol];
      }
    }

    // column-sweep
    if (nrow > 1) // do this if we have an 2-dimensional array
    {
      for (int jcol = 0; jcol < nc; jcol += Cstride) {
        int jr = mgard::get_lindex(nc, ncol, jcol);
        for (int irow = 0; irow < nrow; ++irow) {
          col_vec[irow] = work[mgard::get_index(ncol, irow, jr)];
        }

        mgard_gen::mass_mult_l(l, col_vec, coords_y, nr, nrow);
        mgard_gen::restriction_l(l + 1, col_vec, coords_y, nr, nrow);
        mgard_gen::solve_tridiag_M_l(l + 1, col_vec, coords_y, nr, nrow);

        for (int irow = 0; irow < nrow; ++irow) {
          work[mgard::get_index(ncol, irow, jr)] = col_vec[irow];
        }
      }
    }

    // Solved for (z_l, phi_l) = (c_{l+1}, vl)
    add_level_l(l + 1, v, work.data(), nr, nc, nrow, ncol);
  }
}

template <typename Real>
void recompose_1D(const int nc, const int ncol, const int l_target, Real *v,
                  std::vector<Real> &work, std::vector<Real> &coords_x,
                  std::vector<Real> &row_vec) {
  for (int l = l_target; l > 0; --l) {

    int stride = std::pow(2, l); // current stride
    int Pstride = stride / 2;

    copy_level_l(l - 1, v, work.data(), 1, nc, 1, ncol);

    assign_num_level_l(l, work.data(), static_cast<Real>(0.0), 1, nc, 1, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[jcol];
    }

    mgard_gen::mass_mult_l(l - 1, row_vec, coords_x, nc, ncol);

    mgard_gen::restriction_l(l, row_vec, coords_x, nc, ncol);

    mgard_gen::solve_tridiag_M_l(l, row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[jcol] = row_vec[jcol];
    }

    subtract_level_l(l, work.data(), v, 1, nc, 1, ncol); // do -(Qu - zl)
    //        //std::cout  << "recomposing-rowsweep2" << "\n";

    //   //int Pstride = stride/2; //finer stride

    //   // row-sweep
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[jcol];
    }

    mgard_gen::prolongate_l(l, row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[jcol] = row_vec[jcol];
    }

    //   //std::cout  << "recomposing-colsweep2" << "\n";
    assign_num_level_l(l, v, static_cast<Real>(0.0), 1, nc, 1, ncol);
    subtract_level_l(l - 1, v, work.data(), 1, nc, 1, ncol);
  }
  //    //std::cout  << "last step" << "\n";
}

template <typename Real>
void recompose_2D(const int nr, const int nc, const int nrow, const int ncol,
                  const int l_target, Real *v, std::vector<Real> &work,
                  std::vector<Real> &coords_x, std::vector<Real> &coords_y,
                  std::vector<Real> &row_vec, std::vector<Real> &col_vec) {
  // recompose
  //    //std::cout  << "recomposing" << "\n";
  for (int l = l_target; l > 0; --l) {

    int stride = std::pow(2, l); // current stride
    int Pstride = stride / 2;

    copy_level_l(l - 1, v, work.data(), nr, nc, nrow, ncol);

    assign_num_level_l(l, work.data(), static_cast<Real>(0.0), nr, nc, nrow,
                       ncol);

    //        //std::cout  << "recomposing-rowsweep" << "\n";
    //  l = 0;
    // row-sweep
    for (int irow = 0; irow < nr; ++irow) {
      int ir = mgard::get_lindex(nr, nrow, irow);
      for (int jcol = 0; jcol < ncol; ++jcol) {
        row_vec[jcol] = work[mgard::get_index(ncol, ir, jcol)];
      }

      mgard_gen::mass_mult_l(l - 1, row_vec, coords_x, nc, ncol);

      mgard_gen::restriction_l(l, row_vec, coords_x, nc, ncol);

      mgard_gen::solve_tridiag_M_l(l, row_vec, coords_x, nc, ncol);

      for (int jcol = 0; jcol < ncol; ++jcol) {
        work[mgard::get_index(ncol, ir, jcol)] = row_vec[jcol];
      }
    }

    //   //std::cout  << "recomposing-colsweep" << "\n";

    // column-sweep, this is the slow one! Need something like column_copy
    if (nrow > 1) // check if we have 1-D array..
    {
      for (int jcol = 0; jcol < nc; jcol += stride) {
        int jr = mgard::get_lindex(nc, ncol, jcol);
        for (int irow = 0; irow < nrow; ++irow) {
          col_vec[irow] = work[mgard::get_index(ncol, irow, jr)];
        }

        mgard_gen::mass_mult_l(l - 1, col_vec, coords_y, nr, nrow);

        mgard_gen::restriction_l(l, col_vec, coords_y, nr, nrow);

        mgard_gen::solve_tridiag_M_l(l, col_vec, coords_y, nr, nrow);

        for (int irow = 0; irow < nrow; ++irow) {
          work[mgard::get_index(ncol, irow, jr)] = col_vec[irow];
        }
      }
    }
    subtract_level_l(l, work.data(), v, nr, nc, nrow, ncol); // do -(Qu - zl)
    //        //std::cout  << "recomposing-rowsweep2" << "\n";

    //   //int Pstride = stride/2; //finer stride

    //   // row-sweep
    for (int irow = 0; irow < nr; irow += stride) {
      int ir = mgard::get_lindex(nr, nrow, irow);
      for (int jcol = 0; jcol < ncol; ++jcol) {
        row_vec[jcol] = work[mgard::get_index(ncol, ir, jcol)];
      }

      mgard_gen::prolongate_l(l, row_vec, coords_x, nc, ncol);

      for (int jcol = 0; jcol < ncol; ++jcol) {
        work[mgard::get_index(ncol, ir, jcol)] = row_vec[jcol];
      }
    }

    //   //std::cout  << "recomposing-colsweep2" << "\n";
    // column-sweep, this is the slow one! Need something like column_copy
    if (nrow > 1) {
      for (int jcol = 0; jcol < nc; jcol += Pstride) {
        int jr = mgard::get_lindex(nc, ncol, jcol);
        for (int irow = 0; irow < nrow; ++irow) // copy all rows
        {
          col_vec[irow] = work[mgard::get_index(ncol, irow, jr)];
        }

        mgard_gen::prolongate_l(l, col_vec, coords_y, nr, nrow);

        for (int irow = 0; irow < nrow; ++irow) {
          work[mgard::get_index(ncol, irow, jr)] = col_vec[irow];
        }
      }
    }

    assign_num_level_l(l, v, static_cast<Real>(0.0), nr, nc, nrow, ncol);
    subtract_level_l(l - 1, v, work.data(), nr, nc, nrow, ncol);
  }
  //    //std::cout  << "last step" << "\n";
}

template <typename Real>
void prolongate_last(std::vector<Real> &v, std::vector<Real> &coords, int n,
                     int no) {
  // calculate the result of restrictionion

  for (int i = 0; i < n - 1; ++i) // loop over the logical array
  {
    int i_logic = mgard::get_lindex(n, no, i);
    int i_logicP = mgard::get_lindex(n, no, i + 1);

    if (i_logicP != i_logic + 1) // next real memory location was jumped over,
                                 // so need to restriction
    {
      Real h1 = mgard_common::get_h(coords, i_logic, 1);
      Real h2 = mgard_common::get_h(coords, i_logic + 1, 1);
      Real hsum = h1 + h2;
      v[i_logic + 1] = (h2 * v[i_logic] + h1 * v[i_logicP]) / hsum;
      //             v[i_logic+1] = 2*(h1*v[i_logicP])/hsum;
    }
  }
}

template <typename Real>
void postp_1D(const int nc, const int ncol, const int l_target, Real *v,
              std::vector<Real> &work, std::vector<Real> &coords_x,
              std::vector<Real> &row_vec) {
  mgard_cannon::copy_level(1, ncol, 0, v, work);

  assign_num_level_l(0, work.data(), static_cast<Real>(0.0), 1, nc, 1, ncol);

  for (int jcol = 0; jcol < ncol; ++jcol) {
    row_vec[jcol] = work[jcol];
  }

  mgard_cannon::mass_matrix_multiply(0, row_vec, coords_x);

  restriction_first(row_vec, coords_x, nc, ncol);

  mgard_gen::solve_tridiag_M_l(0, row_vec, coords_x, nc, ncol);

  for (int jcol = 0; jcol < ncol; ++jcol) {
    work[jcol] = row_vec[jcol];
  }

  subtract_level_l(0, work.data(), v, 1, nc, 1, ncol); // do -(Qu - zl)
  //   //   // row-sweep
  for (int jcol = 0; jcol < ncol; ++jcol) {
    row_vec[jcol] = work[jcol];
  }

  mgard_gen::prolongate_last(row_vec, coords_x, nc, ncol);

  for (int jcol = 0; jcol < ncol; ++jcol) {
    work[jcol] = row_vec[jcol];
  }

  assign_num_level_l(0, v, static_cast<Real>(0.0), 1, nc, 1, ncol);
  mgard_cannon::subtract_level(1, ncol, 0, v, work.data());
}

template <typename Real>
void postp_2D(const int nr, const int nc, const int nrow, const int ncol,
              const int l_target, Real *v, std::vector<Real> &work,
              std::vector<Real> &coords_x, std::vector<Real> &coords_y,
              std::vector<Real> &row_vec, std::vector<Real> &col_vec) {
  mgard_cannon::copy_level(nrow, ncol, 0, v, work);

  assign_num_level_l(0, work.data(), static_cast<Real>(0.0), nr, nc, nrow,
                     ncol);

  for (int irow = 0; irow < nrow; ++irow) {
    //        int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[mgard::get_index(ncol, irow, jcol)];
    }

    mgard_cannon::mass_matrix_multiply(0, row_vec, coords_x);

    restriction_first(row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[mgard::get_index(ncol, irow, jcol)] = row_vec[jcol];
    }
  }

  for (int irow = 0; irow < nr; ++irow) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[mgard::get_index(ncol, ir, jcol)];
    }

    mgard_gen::solve_tridiag_M_l(0, row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[mgard::get_index(ncol, ir, jcol)] = row_vec[jcol];
    }
  }

  //   //   //std::cout  << "recomposing-colsweep" << "\n";

  //     // column-sweep, this is the slow one! Need something like column_copy
  if (nrow > 1) // check if we have 1-D array..
  {
    for (int jcol = 0; jcol < ncol; ++jcol) {
      //      int jr  = mgard::get_lindex(nc,  ncol,  jcol);
      for (int irow = 0; irow < nrow; ++irow) {
        col_vec[irow] = work[mgard::get_index(ncol, irow, jcol)];
      }

      mgard_cannon::mass_matrix_multiply(0, col_vec, coords_y);

      mgard_gen::restriction_first(col_vec, coords_y, nr, nrow);

      for (int irow = 0; irow < nrow; ++irow) {
        work[mgard::get_index(ncol, irow, jcol)] = col_vec[irow];
      }
    }

    for (int jcol = 0; jcol < nc; ++jcol) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      for (int irow = 0; irow < nrow; ++irow) {
        col_vec[irow] = work[mgard::get_index(ncol, irow, jr)];
      }

      mgard_gen::solve_tridiag_M_l(0, col_vec, coords_y, nr, nrow);
      for (int irow = 0; irow < nrow; ++irow) {
        work[mgard::get_index(ncol, irow, jr)] = col_vec[irow];
      }
    }
  }

  subtract_level_l(0, work.data(), v, nr, nc, nrow, ncol); // do -(Qu - zl)
  //        //std::cout  << "recomposing-rowsweep2" << "\n";

  //     //   //int Pstride = stride/2; //finer stride

  //   //   // row-sweep
  for (int irow = 0; irow < nr; ++irow) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[mgard::get_index(ncol, ir, jcol)];
    }

    mgard_gen::prolongate_last(row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[mgard::get_index(ncol, ir, jcol)] = row_vec[jcol];
    }
  }

  //     // column-sweep, this is the slow one! Need something like column_copy
  if (nrow > 1) {
    for (int jcol = 0; jcol < ncol; ++jcol) {
      // int jr  = mgard::get_lindex(nc,  ncol,  jcol);
      for (int irow = 0; irow < nrow; ++irow) // copy all rows
      {
        col_vec[irow] = work[mgard::get_index(ncol, irow, jcol)];
      }

      mgard_gen::prolongate_last(col_vec, coords_y, nr, nrow);

      for (int irow = 0; irow < nrow; ++irow) {
        work[mgard::get_index(ncol, irow, jcol)] = col_vec[irow];
      }
    }
  }

  assign_num_level_l(0, v, static_cast<Real>(0.0), nr, nc, nrow, ncol);
  mgard_cannon::subtract_level(nrow, ncol, 0, v, work.data());
}

template <typename Real>
void qwrite_2D_l(const int nr, const int nc, const int nrow, const int ncol,
                 const int nlevel, const int l, Real *v, const Real tol,
                 const Real norm, const std::string outfile) {

  //    int stride = std::pow(2,l);//current stride
  //    int Cstride = 2*stride;
  const mgard::LinearQuantizer<Real, int> quantizer(2 * norm * tol /
                                                    (nlevel + 1));

  gzFile out_file = gzopen(outfile.c_str(), "w6b");
  int prune_count = 0;
  gzwrite(out_file, &quantizer.quantum, sizeof(Real));

  // level L+1, finest first level
  for (int irow = 0; irow < nr; ++irow) // loop over the logical array
  {
    int ir = mgard::get_lindex(nr, nrow, irow);

    for (int jcol = 0; jcol < nc - 1; ++jcol) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      int jrP = mgard::get_lindex(nc, ncol, jcol + 1);

      if (jrP != jr + 1) // next real memory location was jumped over, so this
                         // is level L+1
      {
        const int n = quantizer(v[mgard::get_index(ncol, ir, jr + 1)]);
        if (n == 0)
          ++prune_count;
        gzwrite(out_file, &n, sizeof(int));
      }
    }
  }

  for (int jcol = 0; jcol < nc; ++jcol) {
    int jr = mgard::get_lindex(nc, ncol, jcol);
    for (int irow = 0; irow < nr - 1; ++irow) // loop over the logical array
    {
      int ir = mgard::get_lindex(nr, nrow, irow);
      int irP = mgard::get_lindex(nr, nrow, irow + 1);
      if (irP != ir + 1) // next real memory location was jumped over, so this
                         // is level L+1
      {
        const int n = quantizer(v[mgard::get_index(ncol, ir + 1, jr)]);
        if (n == 0)
          ++prune_count;
        gzwrite(out_file, &n, sizeof(int));
      }
    }
  }

  for (int irow = 0; irow < nr - 1; ++irow) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    int irP = mgard::get_lindex(nr, nrow, irow + 1);

    for (int jcol = 0; jcol < nc - 1; ++jcol) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      int jrP = mgard::get_lindex(nc, ncol, jcol + 1);
      if ((irP != ir + 1) &&
          (jrP != jr + 1)) // we skipped both a row and a column
      {
        const int n = quantizer(v[mgard::get_index(ncol, ir + 1, jr + 1)]);
        if (n == 0)
          ++prune_count;
        gzwrite(out_file, &n, sizeof(int));
      }
    }
  }

  // levels from L->0 in 2^k+1
  for (int l = 0; l <= nlevel; l++) {
    int stride = std::pow(2, l);
    int Cstride = stride * 2;
    int row_counter = 0;

    for (int irow = 0; irow < nr; irow += stride) {
      int ir = mgard::get_lindex(nr, nrow, irow);
      if (row_counter % 2 == 0 && l != nlevel) {
        for (int jcol = Cstride; jcol < nc; jcol += Cstride) {
          int jr = mgard::get_lindex(nc, ncol, jcol);
          const int n = quantizer(v[mgard::get_index(ncol, ir, jr - stride)]);
          if (n == 0)
            ++prune_count;
          gzwrite(out_file, &n, sizeof(int));
        }

      } else {
        for (int jcol = 0; jcol < nc; jcol += stride) {
          int jr = mgard::get_lindex(nc, ncol, jcol);
          const int n = quantizer(v[mgard::get_index(ncol, ir, jr)]);
          if (n == 0)
            ++prune_count;
          gzwrite(out_file, &n, sizeof(int));
        }
      }
      ++row_counter;
    }
  }

  // std::cout  << "Pruned : "<< prune_count << " Reduction : " << (Real)
  // nrow*ncol /(nrow*ncol - prune_count) << "\n";
  gzclose(out_file);
}

} // namespace mgard_gen

} // namespace mgard_2d

#endif
