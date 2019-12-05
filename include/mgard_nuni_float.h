// Copyright 2017, Brown University, Providence, RI.
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney
// Corresponding Author: Ozan Tugluk
//
// version: 0.0.0.2
// See LICENSE for details.

#ifndef MGARD_NUNI_FLOAT_H
#define MGARD_NUNI_FLOAT_H

#include <string>
#include <vector>

namespace mgard_common {

int parse_cmdl(int argc, char **argv, int &nrow, int &ncol, int &nfib,
               float &tol, float &s, std::string &in_file,
               std::string &coord_file);

bool is_2kplus1(float num);

inline int get_index(const int ncol, const int i, const int j);

inline int get_index3(const int ncol, const int nfib, const int i, const int j,
                      const int k);

float max_norm(const std::vector<float> &v);

inline float interp_1d(float x, float x1, float x2, float q00, float q01);

inline float interp_2d(float q11, float q12, float q21, float q22, float x1,
                       float x2, float y1, float y2, float x, float y);

inline float interp_3d(float q000, float q100, float q110, float q010,
                       float q001, float q101, float q111, float q011, float x1,
                       float x2, float y1, float y2, float z1, float z2,
                       float x, float y, float z);

inline float get_h(const std::vector<float> &coords, int i, int stride);

inline float get_dist(const std::vector<float> &coords, int i, int j);

void qread_2D_interleave(const int nrow, const int ncol, const int nlevel,
                         float *v, std::string infile);

void qread_2D_bin(const int nrow, const int ncol, const int nlevel, float *v,
                  std::string infile);

void qwrite_2D_bin(const int nrow, const int ncol, const int nlevel,
                   const int l, float *v, float tol, float norm,
                   const std::string outfile);

void qwrite_2D_interleave(const int nrow, const int ncol, const int nlevel,
                          const int l, float *v, float tol, float norm,
                          const std::string outfile);

void qwrite_3D_interleave(const int nrow, const int ncol, const int nfib,
                          const int nlevel, const int l, float *v, float tol,
                          float norm, const std::string outfile);

void qwrite_3D_interleave2(const int nrow, const int ncol, const int nfib,
                           const int nlevel, const int l, float *v, float tol,
                           float norm, const std::string outfile);

void copy_slice(float *work, std::vector<float> &work2d, int nrow, int ncol,
                int nfib, int is);

void copy_from_slice(float *work, std::vector<float> &work2d, int nrow,
                     int ncol, int nfib, int is);

} // namespace mgard_common

namespace mgard_cannon {

void assign_num_level(const int nrow, const int ncol, const int l, float *v,
                      float num);

void subtract_level(const int nrow, const int ncol, const int l, float *v,
                    float *work);

void pi_lminus1(const int l, std::vector<float> &v,
                const std::vector<float> &coords);

void restriction(const int l, std::vector<float> &v,
                 const std::vector<float> &coords);

void prolongate(const int l, std::vector<float> &v,
                const std::vector<float> &coords);

void solve_tridiag_M(const int l, std::vector<float> &v,
                     const std::vector<float> &coords);

void mass_matrix_multiply(const int l, std::vector<float> &v,
                          const std::vector<float> &coords);

void write_level_2D(const int nrow, const int ncol, const int l, float *v,
                    std::ofstream &outfile);

void copy_level(const int nrow, const int ncol, const int l, float *v,
                std::vector<float> &work);

void copy_level3(const int nrow, const int ncol, const int nfib, const int l,
                 float *v, std::vector<float> &work);

} // namespace mgard_cannon

namespace mgard_gen {
inline float *get_ref(std::vector<float> &v, const int n, const int no,
                      const int i); // return reference to logical element

inline int get_lindex(const int n, const int no, const int i);

inline float get_h_l(const std::vector<float> &coords, const int n,
                     const int no, int i, int stride);

float l2_norm(const int l, const int n, const int no, std::vector<float> &v,
              const std::vector<float> &x);

float l2_norm2(const int l, int nr, int nc, int nrow, int ncol,
               std::vector<float> &v, const std::vector<float> &coords_x,
               const std::vector<float> &coords_y);

float l2_norm3(const int l, int nr, int nc, int nf, int nrow, int ncol,
               int nfib, std::vector<float> &v,
               const std::vector<float> &coords_x,
               const std::vector<float> &coords_y,
               const std::vector<float> &coords_z);

void write_level_2D_l(const int l, float *v, std::ofstream &outfile, int nr,
                      int nc, int nrow, int ncol);

void qwrite_3D(const int nr, const int nc, const int nf, const int nrow,
               const int ncol, const int nfib, const int nlevel, const int l,
               float *v, const std::vector<float> &coords_x,
               const std::vector<float> &coords_y,
               const std::vector<float> &coords_z, float tol, float s,
               float norm, const std::string outfile);

void quantize_3D(const int nr, const int nc, const int nf, const int nrow,
                 const int ncol, const int nfib, const int nlevel, float *v,
                 std::vector<int> &work, const std::vector<float> &coords_x,
                 const std::vector<float> &coords_y,
                 const std::vector<float> &coords_z, float norm, float tol);

void quantize_3D(const int nr, const int nc, const int nf, const int nrow,
                 const int ncol, const int nfib, const int nlevel, float *v,
                 std::vector<int> &work, const std::vector<float> &coords_x,
                 const std::vector<float> &coords_y,
                 const std::vector<float> &coords_z, float s, float norm,
                 float tol);

void quantize_2D(const int nr, const int nc, const int nrow, const int ncol,
                 const int nlevel, float *v, std::vector<int> &work,
                 const std::vector<float> &coords_x,
                 const std::vector<float> &coords_y, float s, float norm,
                 float tol);

void dequantize_3D(const int nr, const int nc, const int nf, const int nrow,
                   const int ncol, const int nfib, const int nlevel, float *v,
                   std::vector<int> &out_data,
                   const std::vector<float> &coords_x,
                   const std::vector<float> &coords_y,
                   const std::vector<float> &coords_z);

void dequantize_3D(const int nr, const int nc, const int nf, const int nrow,
                   const int ncol, const int nfib, const int nlevel, float *v,
                   std::vector<int> &out_data,
                   const std::vector<float> &coords_x,
                   const std::vector<float> &coords_y,
                   const std::vector<float> &coords_z, float s);

void dequantize_2D(const int nr, const int nc, const int nrow, const int ncol,
                   const int nlevel, float *v, std::vector<int> &work,
                   const std::vector<float> &coords_x,
                   const std::vector<float> &coords_y, float s);

void dequant_3D(const int nr, const int nc, const int nf, const int nrow,
                const int ncol, const int nfib, const int nlevel, const int l,
                float *v, float *work, const std::vector<float> &coords_x,
                const std::vector<float> &coords_y,
                const std::vector<float> &coords_z, float s);

void copy_level_l(const int l, float *v, float *work, int nr, int nc, int nrow,
                  int ncol);

void subtract_level_l(const int l, float *v, float *work, int nr, int nc,
                      int nrow, int ncol);

void pi_lminus1_l(const int l, std::vector<float> &v,
                  const std::vector<float> &coords, int n, int no);

void pi_lminus1_first(std::vector<float> &v, const std::vector<float> &coords,
                      int n, int no);

void pi_Ql_first(const int nr, const int nc, const int nrow, const int ncol,
                 const int l, float *v, const std::vector<float> &coords_x,
                 const std::vector<float> &coords_y,
                 std::vector<float> &row_vec, std::vector<float> &col_vec);

void pi_Ql(const int nr, const int nc, const int nrow, const int ncol,
           const int l, float *v, const std::vector<float> &coords_x,
           const std::vector<float> &coords_y, std::vector<float> &row_vec,
           std::vector<float> &col_vec);

void pi_Ql3D(const int nr, const int nc, const int nf, const int nrow,
             const int ncol, const int nfib, const int l, float *v,
             const std::vector<float> &coords_x,
             const std::vector<float> &coords_y,
             const std::vector<float> &coords_z, std::vector<float> &row_vec,
             std::vector<float> &col_vec, std::vector<float> &fib_vec);

void pi_Ql3D_first(const int nr, const int nc, const int nf, const int nrow,
                   const int ncol, const int nfib, const int l, float *v,
                   const std::vector<float> &coords_x,
                   const std::vector<float> &coords_y,
                   const std::vector<float> &coords_z,
                   std::vector<float> &row_vec, std::vector<float> &col_vec,
                   std::vector<float> &fib_vec);

void assign_num_level(const int l, std::vector<float> &v, float num, int n,
                      int no);

void assign_num_level_l(const int l, float *v, float num, int nr, int nc,
                        const int nrow, const int ncol);

void restriction_first(std::vector<float> &v, std::vector<float> &coords, int n,
                       int no);

void solve_tridiag_M_l(const int l, std::vector<float> &v,
                       std::vector<float> &coords, int n, int no);

void add_level_l(const int l, float *v, float *work, int nr, int nc, int nrow,
                 int ncol);

void add3_level_l(const int l, float *v, float *work, int nr, int nc, int nf,
                  int nrow, int ncol, int nfib);

void sub3_level_l(const int l, float *v, float *work, int nr, int nc, int nf,
                  int nrow, int ncol, int nfib);

void sub3_level(const int l, float *v, float *work, int nrow, int ncol,
                int nfib);

void sub_level_l(const int l, float *v, float *work, int nr, int nc, int nf,
                 int nrow, int ncol, int nfib);

void project_first(const int nr, const int nc, const int nrow, const int ncol,
                   const int l_target, float *v, std::vector<float> &work,
                   std::vector<float> &coords_x, std::vector<float> &coords_y,
                   std::vector<float> &row_vec, std::vector<float> &col_vec);

void prep_2D(const int nr, const int nc, const int nrow, const int ncol,
             const int l_target, float *v, std::vector<float> &work,
             std::vector<float> &coords_x, std::vector<float> &coords_y,
             std::vector<float> &row_vec, std::vector<float> &col_vec);

void mass_mult_l(const int l, std::vector<float> &v, std::vector<float> &coords,
                 const int n, const int no);

void restriction_l(const int l, std::vector<float> &v,
                   std::vector<float> &coords, int n, int no);

float ml2_norm3(const int l, int nr, int nc, int nf, int nrow, int ncol,
                int nfib, const std::vector<float> &v,
                std::vector<float> &coords_x, std::vector<float> &coords_y,
                std::vector<float> &coords_z);

void prolongate_l(const int l, std::vector<float> &v,
                  std::vector<float> &coords, int n, int no);

void refactor_1D(const int l_target, std::vector<float> &v,
                 std::vector<float> &work, std::vector<float> &coords, int n,
                 int no);

void refactor_2D(const int nr, const int nc, const int nrow, const int ncol,
                 const int l_target, float *v, std::vector<float> &work,
                 std::vector<float> &coords_x, std::vector<float> &coords_y,
                 std::vector<float> &row_vec, std::vector<float> &col_vec);

void refactor_2D_full(const int nr, const int nc, const int nrow,
                      const int ncol, const int l_target, float *v,
                      std::vector<float> &work, std::vector<float> &coords_x,
                      std::vector<float> &coords_y, std::vector<float> &row_vec,
                      std::vector<float> &col_vec);

void refactor_2D_first(const int nr, const int nc, const int nrow,
                       const int ncol, const int l_target, float *v,
                       std::vector<float> &work, std::vector<float> &coords_x,
                       std::vector<float> &coords_y,
                       std::vector<float> &row_vec,
                       std::vector<float> &col_vec);

void copy3_level_l(const int l, float *v, float *work, int nr, int nc, int nf,
                   int nrow, int ncol, int nfib);

void copy3_level(const int l, float *v, float *work, int nrow, int ncol,
                 int nfib);

void assign3_level_l(const int l, float *v, float num, int nr, int nc, int nf,
                     int nrow, int ncol, int nfib);

void refactor_3D(const int nr, const int nc, const int nf, const int nrow,
                 const int ncol, const int nfib, const int l_target, float *v,
                 std::vector<float> &work, std::vector<float> &work2d,
                 std::vector<float> &coords_x, std::vector<float> &coords_y,
                 std::vector<float> &coords_z);

void compute_zl(const int nr, const int nc, const int nrow, const int ncol,
                const int l_target, std::vector<float> &work,
                std::vector<float> &coords_x, std::vector<float> &coords_y,
                std::vector<float> &row_vec, std::vector<float> &col_vec);

void compute_zl_last(const int nr, const int nc, const int nrow, const int ncol,
                     const int l_target, std::vector<float> &work,
                     std::vector<float> &coords_x, std::vector<float> &coords_y,
                     std::vector<float> &row_vec, std::vector<float> &col_vec);

void prolongate_last(std::vector<float> &v, std::vector<float> &coords, int n,
                     int no);

void prolong_add_2D(const int nr, const int nc, const int nrow, const int ncol,
                    const int l_target, std::vector<float> &work,
                    std::vector<float> &coords_x, std::vector<float> &coords_y,
                    std::vector<float> &row_vec, std::vector<float> &col_vec);

void prolong_add_2D_last(const int nr, const int nc, const int nrow,
                         const int ncol, const int l_target,
                         std::vector<float> &work, std::vector<float> &coords_x,
                         std::vector<float> &coords_y,
                         std::vector<float> &row_vec,
                         std::vector<float> &col_vec);

void prep_3D(const int nr, const int nc, const int nf, const int nrow,
             const int ncol, const int nfib, const int l_target, float *v,
             std::vector<float> &work, std::vector<float> &work2d,
             std::vector<float> &coords_x, std::vector<float> &coords_y,
             std::vector<float> &coords_z);

void recompose_3D(const int nr, const int nc, const int nf, const int nrow,
                  const int ncol, const int nfib, const int l_target, float *v,
                  std::vector<float> &work, std::vector<float> &work2d,
                  std::vector<float> &coords_x, std::vector<float> &coords_y,
                  std::vector<float> &coords_z);

void postp_3D(const int nr, const int nc, const int nf, const int nrow,
              const int ncol, const int nfib, const int l_target, float *v,
              std::vector<float> &work, std::vector<float> &coords_x,
              std::vector<float> &coords_y, std::vector<float> &coords_z);

void recompose_2D(const int nr, const int nc, const int nrow, const int ncol,
                  const int l_target, float *v, std::vector<float> &work,
                  std::vector<float> &coords_x, std::vector<float> &coords_y,
                  std::vector<float> &row_vec, std::vector<float> &col_vec);

void recompose_2D_full(const int nr, const int nc, const int nrow,
                       const int ncol, const int l_target, float *v,
                       std::vector<float> &work, std::vector<float> &coords_x,
                       std::vector<float> &coords_y,
                       std::vector<float> &row_vec,
                       std::vector<float> &col_vec);

void postp_2D(const int nr, const int nc, const int nrow, const int ncol,
              const int l_target, float *v, std::vector<float> &work,
              std::vector<float> &coords_x, std::vector<float> &coords_y,
              std::vector<float> &row_vec, std::vector<float> &col_vec);

void qwrite_2D_l(const int nr, const int nc, const int nrow, const int ncol,
                 const int nlevel, const int l, float *v, float tol, float norm,
                 const std::string outfile);

float qoi_norm(int nrow, int ncol, int nfib, std::vector<float> &coords_x,
               std::vector<float> &coords_y, std::vector<float> &coords_z,
               float (*qoi)(int, int, int, std::vector<float>), float s);

float qoi_norm(int nrow, int ncol, int nfib, std::vector<float> &coords_x,
               std::vector<float> &coords_y, std::vector<float> &coords_z,
               float (*qoi)(int, int, int, float *), float s);
} // namespace mgard_gen

namespace mgard_2d {
namespace mgard_common {

int parse_cmdl(int argc, char **argv, int &nrow, int &ncol, float &tol,
               std::string &in_file, std::string &coord_file);

bool is_2kplus1(float num);

inline int get_index(const int ncol, const int i, const int j);

float max_norm(const std::vector<float> &v);

inline float interp_2d(float q11, float q12, float q21, float q22, float x1,
                       float x2, float y1, float y2, float x, float y);

inline float get_h(const std::vector<float> &coords, int i, int stride);

inline float get_dist(const std::vector<float> &coords, int i, int j);

void qread_2D_interleave(const int nrow, const int ncol, const int nlevel,
                         float *v, std::string infile);

void qwrite_2D_interleave(const int nrow, const int ncol, const int nlevel,
                          const int l, float *v, float tol, float norm,
                          const std::string outfile);

} // namespace mgard_common

namespace mgard_cannon {

void assign_num_level(const int nrow, const int ncol, const int l, float *v,
                      float num);

void subtract_level(const int nrow, const int ncol, const int l, float *v,
                    float *work);

void pi_lminus1(const int l, std::vector<float> &v,
                const std::vector<float> &coords);

void restriction(const int l, std::vector<float> &v,
                 const std::vector<float> &coords);

void prolongate(const int l, std::vector<float> &v,
                const std::vector<float> &coords);

void solve_tridiag_M(const int l, std::vector<float> &v,
                     const std::vector<float> &coords);

void mass_matrix_multiply(const int l, std::vector<float> &v,
                          const std::vector<float> &coords);

void write_level_2D(const int nrow, const int ncol, const int l, float *v,
                    std::ofstream &outfile);

void copy_level(const int nrow, const int ncol, const int l, float *v,
                std::vector<float> &work);

} // namespace mgard_cannon

namespace mgard_gen {
inline float *get_ref(std::vector<float> &v, const int n, const int no,
                      const int i);

inline int get_lindex(const int n, const int no, const int i);

inline float get_h_l(const std::vector<float> &coords, const int n,
                     const int no, int i, int stride);

void write_level_2D_l(const int l, float *v, std::ofstream &outfile, int nr,
                      int nc, int nrow, int ncol);

void copy_level_l(const int l, float *v, float *work, int nr, int nc, int nrow,
                  int ncol);

void subtract_level_l(const int l, float *v, float *work, int nr, int nc,
                      int nrow, int ncol);

void pi_lminus1_l(const int l, std::vector<float> &v,
                  const std::vector<float> &coords, int n, int no);

void pi_lminus1_first(std::vector<float> &v, const std::vector<float> &coords,
                      int n, int no);

void pi_Ql_first(const int nr, const int nc, const int nrow, const int ncol,
                 const int l, float *v, const std::vector<float> &coords_x,
                 const std::vector<float> &coords_y,
                 std::vector<float> &row_vec, std::vector<float> &col_vec);

void pi_Ql(const int nr, const int nc, const int nrow, const int ncol,
           const int l, float *v, const std::vector<float> &coords_x,
           const std::vector<float> &coords_y, std::vector<float> &row_vec,
           std::vector<float> &col_vec);

void assign_num_level_l(const int l, float *v, float num, int nr, int nc,
                        const int nrow, const int ncol);

void restriction_first(std::vector<float> &v, std::vector<float> &coords, int n,
                       int no);

void solve_tridiag_M_l(const int l, std::vector<float> &v,
                       std::vector<float> &coords, int n, int no);

void add_level_l(const int l, float *v, float *work, int nr, int nc, int nrow,
                 int ncol);

void project_first(const int nr, const int nc, const int nrow, const int ncol,
                   const int l_target, float *v, std::vector<float> &work,
                   std::vector<float> &coords_x, std::vector<float> &coords_y,
                   std::vector<float> &row_vec, std::vector<float> &col_vec);

void prep_2D(const int nr, const int nc, const int nrow, const int ncol,
             const int l_target, float *v, std::vector<float> &work,
             std::vector<float> &coords_x, std::vector<float> &coords_y,
             std::vector<float> &row_vec, std::vector<float> &col_vec);

void mass_mult_l(const int l, std::vector<float> &v, std::vector<float> &coords,
                 const int n, const int no);

void restriction_l(const int l, std::vector<float> &v,
                   std::vector<float> &coords, int n, int no);

void prolongate_l(const int l, std::vector<float> &v,
                  std::vector<float> &coords, int n, int no);

void refactor_2D(const int nr, const int nc, const int nrow, const int ncol,
                 const int l_target, float *v, std::vector<float> &work,
                 std::vector<float> &coords_x, std::vector<float> &coords_y,
                 std::vector<float> &row_vec, std::vector<float> &col_vec);

void recompose_2D(const int nr, const int nc, const int nrow, const int ncol,
                  const int l_target, float *v, std::vector<float> &work,
                  std::vector<float> &coords_x, std::vector<float> &coords_y,
                  std::vector<float> &row_vec, std::vector<float> &col_vec);

void prolongate_last(std::vector<float> &v, std::vector<float> &coords, int n,
                     int no);

void postp_2D(const int nr, const int nc, const int nrow, const int ncol,
              const int l_target, float *v, std::vector<float> &work,
              std::vector<float> &coords_x, std::vector<float> &coords_y,
              std::vector<float> &row_vec, std::vector<float> &col_vec);

void qwrite_2D_l(const int nr, const int nc, const int nrow, const int ncol,
                 const int nlevel, const int l, float *v, float tol, float norm,
                 const std::string outfile);

} // namespace mgard_gen

} // namespace mgard_2d

#endif
