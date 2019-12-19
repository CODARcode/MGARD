// Copyright 2017, Brown University, Providence, RI.
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney
// Corresponding Author: Ozan Tugluk
//
// version: 0.0.0.2
// See LICENSE for details.
#ifndef MGARD_NUNI_H
#define MGARD_NUNI_H

#include <string>
#include <vector>

namespace mgard_common {

template <typename Real> Real max_norm(const std::vector<Real> &v);

template <typename Real>
inline Real interp_1d(Real x, Real x1, Real x2, Real q00, Real q01);

template <typename Real>
inline Real interp_2d(Real q11, Real q12, Real q21, Real q22, Real x1, Real x2,
                      Real y1, Real y2, Real x, Real y);

template <typename Real>
inline Real interp_3d(Real q000, Real q100, Real q110, Real q010, Real q001,
                      Real q101, Real q111, Real q011, Real x1, Real x2,
                      Real y1, Real y2, Real z1, Real z2, Real x, Real y,
                      Real z);

template <typename Real>
inline Real get_h(const std::vector<Real> &coords, int i, int stride);

template <typename Real>
inline Real get_dist(const std::vector<Real> &coords, int i, int j);

template <typename Real>
void qread_2D_interleave(const int nrow, const int ncol, const int nlevel,
                         Real *v, std::string infile);

template <typename Real> inline short encode(Real x);

template <typename Real> inline Real decode(short x);

template <typename Real>
void qread_2D_bin(const int nrow, const int ncol, const int nlevel, Real *v,
                  std::string infile);

template <typename Real>
void qwrite_2D_bin(const int nrow, const int ncol, const int nlevel,
                   const int l, Real *v, Real tol, Real norm,
                   const std::string outfile);

template <typename Real>
void qwrite_2D_interleave(const int nrow, const int ncol, const int nlevel,
                          const int l, Real *v, Real tol, Real norm,
                          const std::string outfile);

template <typename Real>
void qwrite_3D_interleave(const int nrow, const int ncol, const int nfib,
                          const int nlevel, const int l, Real *v, Real tol,
                          Real norm, const std::string outfile);

template <typename Real>
void qwrite_3D_interleave2(const int nrow, const int ncol, const int nfib,
                           const int nlevel, const int l, Real *v, Real tol,
                           Real norm, const std::string outfile);

template <typename Real>
void copy_slice(Real *work, std::vector<Real> &work2d, int nrow, int ncol,
                int nfib, int is);

template <typename Real>
void copy_from_slice(Real *work, std::vector<Real> &work2d, int nrow, int ncol,
                     int nfib, int is);

} // namespace mgard_common

namespace mgard_cannon {

template <typename Real>
void assign_num_level(const int nrow, const int ncol, const int l, Real *v,
                      Real num);

template <typename Real>
void subtract_level(const int nrow, const int ncol, const int l, Real *v,
                    Real *work);

template <typename Real>
void pi_lminus1(const int l, std::vector<Real> &v,
                const std::vector<Real> &coords);

template <typename Real>
void restriction(const int l, std::vector<Real> &v,
                 const std::vector<Real> &coords);

template <typename Real>
void prolongate(const int l, std::vector<Real> &v,
                const std::vector<Real> &coords);

template <typename Real>
void solve_tridiag_M(const int l, std::vector<Real> &v,
                     const std::vector<Real> &coords);

template <typename Real>
void mass_matrix_multiply(const int l, std::vector<Real> &v,
                          const std::vector<Real> &coords);

template <typename Real>
void write_level_2D(const int nrow, const int ncol, const int l, Real *v,
                    std::ofstream &outfile);

template <typename Real>
void copy_level(const int nrow, const int ncol, const int l, Real *v,
                std::vector<Real> &work);

template <typename Real>
void copy_level3(const int nrow, const int ncol, const int nfib, const int l,
                 Real *v, std::vector<Real> &work);

} // namespace mgard_cannon

namespace mgard_gen {

template <typename Real>
inline Real *get_ref(std::vector<Real> &v, const int n, const int no,
                     const int i); // return reference to logical element

template <typename Real>
inline Real get_h_l(const std::vector<Real> &coords, const int n, const int no,
                    int i, int stride);

template <typename Real>
Real l2_norm(const int l, const int n, const int no, std::vector<Real> &v,
             const std::vector<Real> &x);

template <typename Real>
Real l2_norm2(const int l, int nr, int nc, int nrow, int ncol,
              std::vector<Real> &v, const std::vector<Real> &coords_x,
              const std::vector<Real> &coords_y);

template <typename Real>
Real l2_norm3(const int l, int nr, int nc, int nf, int nrow, int ncol, int nfib,
              std::vector<Real> &v, const std::vector<Real> &coords_x,
              const std::vector<Real> &coords_y,
              const std::vector<Real> &coords_z);

template <typename Real>
void write_level_2D_l(const int l, Real *v, std::ofstream &outfile, int nr,
                      int nc, int nrow, int ncol);

template <typename Real>
void qwrite_3D(const int nr, const int nc, const int nf, const int nrow,
               const int ncol, const int nfib, const int nlevel, const int l,
               Real *v, const std::vector<Real> &coords_x,
               const std::vector<Real> &coords_y,
               const std::vector<Real> &coords_z, Real tol, Real s, Real norm,
               const std::string outfile);

template <typename Real>
void quantize_3D(const int nr, const int nc, const int nf, const int nrow,
                 const int ncol, const int nfib, const int nlevel, Real *v,
                 std::vector<int> &work, const std::vector<Real> &coords_x,
                 const std::vector<Real> &coords_y,
                 const std::vector<Real> &coords_z, Real norm, Real tol);

template <typename Real>
void quantize_3D(const int nr, const int nc, const int nf, const int nrow,
                 const int ncol, const int nfib, const int nlevel, Real *v,
                 std::vector<int> &work, const std::vector<Real> &coords_x,
                 const std::vector<Real> &coords_y,
                 const std::vector<Real> &coords_z, Real s, Real norm,
                 Real tol);

template <typename Real>
void quantize_2D(const int nr, const int nc, const int nrow, const int ncol,
                 const int nlevel, Real *v, std::vector<int> &work,
                 const std::vector<Real> &coords_x,
                 const std::vector<Real> &coords_y, Real s, Real norm,
                 Real tol);

template <typename Real>
void dequantize_3D(const int nr, const int nc, const int nf, const int nrow,
                   const int ncol, const int nfib, const int nlevel, Real *v,
                   std::vector<int> &out_data,
                   const std::vector<Real> &coords_x,
                   const std::vector<Real> &coords_y,
                   const std::vector<Real> &coords_z);

template <typename Real>
void dequantize_3D(const int nr, const int nc, const int nf, const int nrow,
                   const int ncol, const int nfib, const int nlevel, Real *v,
                   std::vector<int> &out_data,
                   const std::vector<Real> &coords_x,
                   const std::vector<Real> &coords_y,
                   const std::vector<Real> &coords_z, Real s);

template <typename Real>
void dequantize_2D(const int nr, const int nc, const int nrow, const int ncol,
                   const int nlevel, Real *v, std::vector<int> &work,
                   const std::vector<Real> &coords_x,
                   const std::vector<Real> &coords_y, Real s);

template <typename Real>
void copy_level_l(const int l, Real *v, Real *work, int nr, int nc, int nrow,
                  int ncol);

template <typename Real>
void subtract_level_l(const int l, Real *v, Real *work, int nr, int nc,
                      int nrow, int ncol);

template <typename Real>
void pi_lminus1_l(const int l, std::vector<Real> &v,
                  const std::vector<Real> &coords, int n, int no);

template <typename Real>
void pi_lminus1_first(std::vector<Real> &v, const std::vector<Real> &coords,
                      int n, int no);

template <typename Real>
void pi_Ql_first(const int nr, const int nc, const int nrow, const int ncol,
                 const int l, Real *v, const std::vector<Real> &coords_x,
                 const std::vector<Real> &coords_y, std::vector<Real> &row_vec,
                 std::vector<Real> &col_vec);

template <typename Real>
void pi_Ql(const int nr, const int nc, const int nrow, const int ncol,
           const int l, Real *v, const std::vector<Real> &coords_x,
           const std::vector<Real> &coords_y, std::vector<Real> &row_vec,
           std::vector<Real> &col_vec);

template <typename Real>
void pi_Ql3D(const int nr, const int nc, const int nf, const int nrow,
             const int ncol, const int nfib, const int l, Real *v,
             const std::vector<Real> &coords_x,
             const std::vector<Real> &coords_y,
             const std::vector<Real> &coords_z, std::vector<Real> &row_vec,
             std::vector<Real> &col_vec, std::vector<Real> &fib_vec);

template <typename Real>
void pi_Ql3D_first(const int nr, const int nc, const int nf, const int nrow,
                   const int ncol, const int nfib, const int l, Real *v,
                   const std::vector<Real> &coords_x,
                   const std::vector<Real> &coords_y,
                   const std::vector<Real> &coords_z,
                   std::vector<Real> &row_vec, std::vector<Real> &col_vec,
                   std::vector<Real> &fib_vec);

template <typename Real>
void assign_num_level(const int l, std::vector<Real> &v, Real num, int n,
                      int no);

template <typename Real>
void assign_num_level_l(const int l, Real *v, Real num, int nr, int nc,
                        const int nrow, const int ncol);

template <typename Real>
void restriction_first(std::vector<Real> &v, std::vector<Real> &coords, int n,
                       int no);

template <typename Real>
void solve_tridiag_M_l(const int l, std::vector<Real> &v,
                       std::vector<Real> &coords, int n, int no);

template <typename Real>
void add_level_l(const int l, Real *v, Real *work, int nr, int nc, int nrow,
                 int ncol);

template <typename Real>
void add3_level_l(const int l, Real *v, Real *work, int nr, int nc, int nf,
                  int nrow, int ncol, int nfib);

template <typename Real>
void sub3_level_l(const int l, Real *v, Real *work, int nr, int nc, int nf,
                  int nrow, int ncol, int nfib);

template <typename Real>
void sub3_level(const int l, Real *v, Real *work, int nrow, int ncol, int nfib);

template <typename Real>
void sub_level_l(const int l, Real *v, Real *work, int nr, int nc, int nf,
                 int nrow, int ncol, int nfib);

template <typename Real>
void prep_2D(const int nr, const int nc, const int nrow, const int ncol,
             const int l_target, Real *v, std::vector<Real> &work,
             std::vector<Real> &coords_x, std::vector<Real> &coords_y,
             std::vector<Real> &row_vec, std::vector<Real> &col_vec);

template <typename Real>
void mass_mult_l(const int l, std::vector<Real> &v, std::vector<Real> &coords,
                 const int n, const int no);

template <typename Real>
void restriction_l(const int l, std::vector<Real> &v, std::vector<Real> &coords,
                   int n, int no);

template <typename Real>
Real ml2_norm3(const int l, int nr, int nc, int nf, int nrow, int ncol,
               int nfib, const std::vector<Real> &v,
               std::vector<Real> &coords_x, std::vector<Real> &coords_y,
               std::vector<Real> &coords_z);

template <typename Real>
void prolongate_l(const int l, std::vector<Real> &v, std::vector<Real> &coords,
                  int n, int no);

template <typename Real>
void refactor_2D(const int nr, const int nc, const int nrow, const int ncol,
                 const int l_target, Real *v, std::vector<Real> &work,
                 std::vector<Real> &coords_x, std::vector<Real> &coords_y,
                 std::vector<Real> &row_vec, std::vector<Real> &col_vec);

template <typename Real>
void refactor_2D_first(const int nr, const int nc, const int nrow,
                       const int ncol, const int l_target, Real *v,
                       std::vector<Real> &work, std::vector<Real> &coords_x,
                       std::vector<Real> &coords_y, std::vector<Real> &row_vec,
                       std::vector<Real> &col_vec);

template <typename Real>
void copy3_level_l(const int l, Real *v, Real *work, int nr, int nc, int nf,
                   int nrow, int ncol, int nfib);

template <typename Real>
void copy3_level(const int l, Real *v, Real *work, int nrow, int ncol,
                 int nfib);

template <typename Real>
void assign3_level_l(const int l, Real *v, Real num, int nr, int nc, int nf,
                     int nrow, int ncol, int nfib);

template <typename Real>
void refactor_3D(const int nr, const int nc, const int nf, const int nrow,
                 const int ncol, const int nfib, const int l_target, Real *v,
                 std::vector<Real> &work, std::vector<Real> &work2d,
                 std::vector<Real> &coords_x, std::vector<Real> &coords_y,
                 std::vector<Real> &coords_z);

template <typename Real>
void compute_zl(const int nr, const int nc, const int nrow, const int ncol,
                const int l_target, std::vector<Real> &work,
                std::vector<Real> &coords_x, std::vector<Real> &coords_y,
                std::vector<Real> &row_vec, std::vector<Real> &col_vec);

template <typename Real>
void compute_zl_last(const int nr, const int nc, const int nrow, const int ncol,
                     const int l_target, std::vector<Real> &work,
                     std::vector<Real> &coords_x, std::vector<Real> &coords_y,
                     std::vector<Real> &row_vec, std::vector<Real> &col_vec);

template <typename Real>
void prolongate_last(std::vector<Real> &v, std::vector<Real> &coords, int n,
                     int no);

template <typename Real>
void prolong_add_2D(const int nr, const int nc, const int nrow, const int ncol,
                    const int l_target, std::vector<Real> &work,
                    std::vector<Real> &coords_x, std::vector<Real> &coords_y,
                    std::vector<Real> &row_vec, std::vector<Real> &col_vec);

template <typename Real>
void prolong_add_2D_last(const int nr, const int nc, const int nrow,
                         const int ncol, const int l_target,
                         std::vector<Real> &work, std::vector<Real> &coords_x,
                         std::vector<Real> &coords_y,
                         std::vector<Real> &row_vec,
                         std::vector<Real> &col_vec);

template <typename Real>
void prep_3D(const int nr, const int nc, const int nf, const int nrow,
             const int ncol, const int nfib, const int l_target, Real *v,
             std::vector<Real> &work, std::vector<Real> &work2d,
             std::vector<Real> &coords_x, std::vector<Real> &coords_y,
             std::vector<Real> &coords_z);

template <typename Real>
void recompose_3D(const int nr, const int nc, const int nf, const int nrow,
                  const int ncol, const int nfib, const int l_target, Real *v,
                  std::vector<Real> &work, std::vector<Real> &work2d,
                  std::vector<Real> &coords_x, std::vector<Real> &coords_y,
                  std::vector<Real> &coords_z);

template <typename Real>
void postp_3D(const int nr, const int nc, const int nf, const int nrow,
              const int ncol, const int nfib, const int l_target, Real *v,
              std::vector<Real> &work, std::vector<Real> &coords_x,
              std::vector<Real> &coords_y, std::vector<Real> &coords_z);

template <typename Real>
void recompose_2D(const int nr, const int nc, const int nrow, const int ncol,
                  const int l_target, Real *v, std::vector<Real> &work,
                  std::vector<Real> &coords_x, std::vector<Real> &coords_y,
                  std::vector<Real> &row_vec, std::vector<Real> &col_vec);

template <typename Real>
void recompose_2D_full(const int nr, const int nc, const int nrow,
                       const int ncol, const int l_target, Real *v,
                       std::vector<Real> &work, std::vector<Real> &coords_x,
                       std::vector<Real> &coords_y, std::vector<Real> &row_vec,
                       std::vector<Real> &col_vec);

template <typename Real>
void postp_2D(const int nr, const int nc, const int nrow, const int ncol,
              const int l_target, Real *v, std::vector<Real> &work,
              std::vector<Real> &coords_x, std::vector<Real> &coords_y,
              std::vector<Real> &row_vec, std::vector<Real> &col_vec);

template <typename Real>
void qwrite_2D_l(const int nr, const int nc, const int nrow, const int ncol,
                 const int nlevel, const int l, Real *v, Real tol, Real norm,
                 const std::string outfile);

template <typename Real>
Real qoi_norm(int nrow, int ncol, int nfib, std::vector<Real> &coords_x,
              std::vector<Real> &coords_y, std::vector<Real> &coords_z,
              Real (*qoi)(int, int, int, std::vector<Real>), Real s);

template <typename Real>
Real qoi_norm(int nrow, int ncol, int nfib, std::vector<Real> &coords_x,
              std::vector<Real> &coords_y, std::vector<Real> &coords_z,
              Real (*qoi)(int, int, int, Real *), Real s);

} // namespace mgard_gen

namespace mgard_2d {

namespace mgard_common {

template <typename Real> Real max_norm(const std::vector<Real> &v);

template <typename Real>
inline Real interp_2d(Real q11, Real q12, Real q21, Real q22, Real x1, Real x2,
                      Real y1, Real y2, Real x, Real y);

template <typename Real>
inline Real get_h(const std::vector<Real> &coords, int i, int stride);

template <typename Real>
inline Real get_dist(const std::vector<Real> &coords, int i, int j);

template <typename Real>
void qread_2D_interleave(const int nrow, const int ncol, const int nlevel,
                         Real *v, std::string infile);

template <typename Real>
void qwrite_2D_interleave(const int nrow, const int ncol, const int nlevel,
                          const int l, Real *v, Real tol, Real norm,
                          const std::string outfile);

} // namespace mgard_common

namespace mgard_cannon {

template <typename Real>
void assign_num_level(const int nrow, const int ncol, const int l, Real *v,
                      Real num);

template <typename Real>
void subtract_level(const int nrow, const int ncol, const int l, Real *v,
                    Real *work);

template <typename Real>
void pi_lminus1(const int l, std::vector<Real> &v,
                const std::vector<Real> &coords);

template <typename Real>
void restriction(const int l, std::vector<Real> &v,
                 const std::vector<Real> &coords);

template <typename Real>
void prolongate(const int l, std::vector<Real> &v,
                const std::vector<Real> &coords);

template <typename Real>
void solve_tridiag_M(const int l, std::vector<Real> &v,
                     const std::vector<Real> &coords);

template <typename Real>
void mass_matrix_multiply(const int l, std::vector<Real> &v,
                          const std::vector<Real> &coords);

template <typename Real>
void write_level_2D(const int nrow, const int ncol, const int l, Real *v,
                    std::ofstream &outfile);

template <typename Real>
void copy_level(const int nrow, const int ncol, const int l, Real *v,
                std::vector<Real> &work);

} // namespace mgard_cannon

namespace mgard_gen {

template <typename Real>
inline Real *get_ref(std::vector<Real> &v, const int n, const int no,
                     const int i);

template <typename Real>
inline Real get_h_l(const std::vector<Real> &coords, const int n, const int no,
                    int i, int stride);

template <typename Real>
void write_level_2D_l(const int l, Real *v, std::ofstream &outfile, int nr,
                      int nc, int nrow, int ncol);

template <typename Real>
void copy_level_l(const int l, Real *v, Real *work, int nr, int nc, int nrow,
                  int ncol);

template <typename Real>
void subtract_level_l(const int l, Real *v, Real *work, int nr, int nc,
                      int nrow, int ncol);

template <typename Real>
void pi_lminus1_l(const int l, std::vector<Real> &v,
                  const std::vector<Real> &coords, int n, int no);

template <typename Real>
void pi_lminus1_first(std::vector<Real> &v, const std::vector<Real> &coords,
                      int n, int no);

template <typename Real>
void pi_Ql_first(const int nr, const int nc, const int nrow, const int ncol,
                 const int l, Real *v, const std::vector<Real> &coords_x,
                 const std::vector<Real> &coords_y, std::vector<Real> &row_vec,
                 std::vector<Real> &col_vec);

template <typename Real>
void pi_Ql(const int nr, const int nc, const int nrow, const int ncol,
           const int l, Real *v, const std::vector<Real> &coords_x,
           const std::vector<Real> &coords_y, std::vector<Real> &row_vec,
           std::vector<Real> &col_vec);

template <typename Real>
void assign_num_level_l(const int l, Real *v, Real num, int nr, int nc,
                        const int nrow, const int ncol);

template <typename Real>
void restriction_first(std::vector<Real> &v, std::vector<Real> &coords, int n,
                       int no);

template <typename Real>
void solve_tridiag_M_l(const int l, std::vector<Real> &v,
                       std::vector<Real> &coords, int n, int no);

template <typename Real>
void add_level_l(const int l, Real *v, Real *work, int nr, int nc, int nrow,
                 int ncol);

template <typename Real>
void prep_2D(const int nr, const int nc, const int nrow, const int ncol,
             const int l_target, Real *v, std::vector<Real> &work,
             std::vector<Real> &coords_x, std::vector<Real> &coords_y,
             std::vector<Real> &row_vec, std::vector<Real> &col_vec);

template <typename Real>
void mass_mult_l(const int l, std::vector<Real> &v, std::vector<Real> &coords,
                 const int n, const int no);

template <typename Real>
void restriction_l(const int l, std::vector<Real> &v, std::vector<Real> &coords,
                   int n, int no);

template <typename Real>
void prolongate_l(const int l, std::vector<Real> &v, std::vector<Real> &coords,
                  int n, int no);

template <typename Real>
void refactor_2D(const int nr, const int nc, const int nrow, const int ncol,
                 const int l_target, Real *v, std::vector<Real> &work,
                 std::vector<Real> &coords_x, std::vector<Real> &coords_y,
                 std::vector<Real> &row_vec, std::vector<Real> &col_vec);

template <typename Real>
void recompose_2D(const int nr, const int nc, const int nrow, const int ncol,
                  const int l_target, Real *v, std::vector<Real> &work,
                  std::vector<Real> &coords_x, std::vector<Real> &coords_y,
                  std::vector<Real> &row_vec, std::vector<Real> &col_vec);

template <typename Real>
void prolongate_last(std::vector<Real> &v, std::vector<Real> &coords, int n,
                     int no);

template <typename Real>
void postp_2D(const int nr, const int nc, const int nrow, const int ncol,
              const int l_target, Real *v, std::vector<Real> &work,
              std::vector<Real> &coords_x, std::vector<Real> &coords_y,
              std::vector<Real> &row_vec, std::vector<Real> &col_vec);

template <typename Real>
void qwrite_2D_l(const int nr, const int nc, const int nrow, const int ncol,
                 const int nlevel, const int l, Real *v, Real tol, Real norm,
                 const std::string outfile);

} // namespace mgard_gen

} // namespace mgard_2d

#endif
