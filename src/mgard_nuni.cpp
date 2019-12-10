#include <string>
#include <vector>

#include "mgard_nuni.h"

namespace mgard_common {

template <>
int parse_cmdl<double>(int argc, char **argv, int &nrow, int &ncol, int &nfib,
                       double &tol, double &s, std::string &in_file,
                       std::string &coord_file);

template <> double max_norm<double>(const std::vector<double> &v);

template <>
inline double interp_1d<double>(double x, double x1, double x2, double q00,
                                double q01);

template <>
inline double interp_2d<double>(double q11, double q12, double q21, double q22,
                                double x1, double x2, double y1, double y2,
                                double x, double y);

template <>
inline double interp_3d<double>(double q000, double q100, double q110,
                                double q010, double q001, double q101,
                                double q111, double q011, double x1, double x2,
                                double y1, double y2, double z1, double z2,
                                double x, double y, double z);

template <>
inline double get_h<double>(const std::vector<double> &coords, int i,
                            int stride);

template <>
inline double get_dist<double>(const std::vector<double> &coords, int i, int j);

template <>
void qread_2D_interleave<double>(const int nrow, const int ncol,
                                 const int nlevel, double *v,
                                 std::string infile);

template <> inline short encode<double>(double x);

template <> inline double decode<double>(short x);

template <>
void qread_2D_bin<double>(const int nrow, const int ncol, const int nlevel,
                          double *v, std::string infile);

template <>
void qwrite_2D_bin<double>(const int nrow, const int ncol, const int nlevel,
                           const int l, double *v, double tol, double norm,
                           const std::string outfile);

template <>
void qwrite_2D_interleave<double>(const int nrow, const int ncol,
                                  const int nlevel, const int l, double *v,
                                  double tol, double norm,
                                  const std::string outfile);

template <>
void qwrite_3D_interleave<double>(const int nrow, const int ncol,
                                  const int nfib, const int nlevel, const int l,
                                  double *v, double tol, double norm,
                                  const std::string outfile);

template <>
void qwrite_3D_interleave2<double>(const int nrow, const int ncol,
                                   const int nfib, const int nlevel,
                                   const int l, double *v, double tol,
                                   double norm, const std::string outfile);

template <>
void copy_slice<double>(double *work, std::vector<double> &work2d, int nrow,
                        int ncol, int nfib, int is);

template <>
void copy_from_slice<double>(double *work, std::vector<double> &work2d,
                             int nrow, int ncol, int nfib, int is);

} // namespace mgard_common

namespace mgard_cannon {

template <>
void assign_num_level<double>(const int nrow, const int ncol, const int l,
                              double *v, double num);

template <>
void subtract_level<double>(const int nrow, const int ncol, const int l,
                            double *v, double *work);

template <>
void pi_lminus1<double>(const int l, std::vector<double> &v,
                        const std::vector<double> &coords);

template <>
void restriction<double>(const int l, std::vector<double> &v,
                         const std::vector<double> &coords);

template <>
void prolongate<double>(const int l, std::vector<double> &v,
                        const std::vector<double> &coords);

template <>
void solve_tridiag_M<double>(const int l, std::vector<double> &v,
                             const std::vector<double> &coords);

template <>
void mass_matrix_multiply<double>(const int l, std::vector<double> &v,
                                  const std::vector<double> &coords);

template <>
void write_level_2D<double>(const int nrow, const int ncol, const int l,
                            double *v, std::ofstream &outfile);

template <>
void copy_level<double>(const int nrow, const int ncol, const int l, double *v,
                        std::vector<double> &work);

template <>
void copy_level3<double>(const int nrow, const int ncol, const int nfib,
                         const int l, double *v, std::vector<double> &work);

} // namespace mgard_cannon

namespace mgard_gen {

template <>
inline double *
get_ref<double>(std::vector<double> &v, const int n, const int no,
                const int i); // return reference to logical element

template <>
inline double get_h_l<double>(const std::vector<double> &coords, const int n,
                              const int no, int i, int stride);

template <>
double l2_norm<double>(const int l, const int n, const int no,
                       std::vector<double> &v, const std::vector<double> &x);

template <>
double l2_norm2<double>(const int l, int nr, int nc, int nrow, int ncol,
                        std::vector<double> &v,
                        const std::vector<double> &coords_x,
                        const std::vector<double> &coords_y);

template <>
double l2_norm3<double>(const int l, int nr, int nc, int nf, int nrow, int ncol,
                        int nfib, std::vector<double> &v,
                        const std::vector<double> &coords_x,
                        const std::vector<double> &coords_y,
                        const std::vector<double> &coords_z);

template <>
void write_level_2D_l<double>(const int l, double *v, std::ofstream &outfile,
                              int nr, int nc, int nrow, int ncol);

template <>
void qwrite_3D<double>(const int nr, const int nc, const int nf, const int nrow,
                       const int ncol, const int nfib, const int nlevel,
                       const int l, double *v,
                       const std::vector<double> &coords_x,
                       const std::vector<double> &coords_y,
                       const std::vector<double> &coords_z, double tol,
                       double s, double norm, const std::string outfile);

template <>
void quantize_3D<double>(const int nr, const int nc, const int nf,
                         const int nrow, const int ncol, const int nfib,
                         const int nlevel, double *v, std::vector<int> &work,
                         const std::vector<double> &coords_x,
                         const std::vector<double> &coords_y,
                         const std::vector<double> &coords_z, double norm,
                         double tol);

template <>
void quantize_3D<double>(const int nr, const int nc, const int nf,
                         const int nrow, const int ncol, const int nfib,
                         const int nlevel, double *v, std::vector<int> &work,
                         const std::vector<double> &coords_x,
                         const std::vector<double> &coords_y,
                         const std::vector<double> &coords_z, double s,
                         double norm, double tol);

template <>
void quantize_2D<double>(const int nr, const int nc, const int nrow,
                         const int ncol, const int nlevel, double *v,
                         std::vector<int> &work,
                         const std::vector<double> &coords_x,
                         const std::vector<double> &coords_y, double s,
                         double norm, double tol);

template <>
void dequantize_3D<double>(const int nr, const int nc, const int nf,
                           const int nrow, const int ncol, const int nfib,
                           const int nlevel, double *v,
                           std::vector<int> &out_data,
                           const std::vector<double> &coords_x,
                           const std::vector<double> &coords_y,
                           const std::vector<double> &coords_z);

template <>
void dequantize_3D<double>(const int nr, const int nc, const int nf,
                           const int nrow, const int ncol, const int nfib,
                           const int nlevel, double *v,
                           std::vector<int> &out_data,
                           const std::vector<double> &coords_x,
                           const std::vector<double> &coords_y,
                           const std::vector<double> &coords_z, double s);

template <>
void dequantize_2D<double>(const int nr, const int nc, const int nrow,
                           const int ncol, const int nlevel, double *v,
                           std::vector<int> &work,
                           const std::vector<double> &coords_x,
                           const std::vector<double> &coords_y, double s);

template <>
void dequant_3D<double>(const int nr, const int nc, const int nf,
                        const int nrow, const int ncol, const int nfib,
                        const int nlevel, const int l, double *v, double *work,
                        const std::vector<double> &coords_x,
                        const std::vector<double> &coords_y,
                        const std::vector<double> &coords_z, double s);

template <>
void copy_level_l<double>(const int l, double *v, double *work, int nr, int nc,
                          int nrow, int ncol);

template <>
void subtract_level_l<double>(const int l, double *v, double *work, int nr,
                              int nc, int nrow, int ncol);

template <>
void pi_lminus1_l<double>(const int l, std::vector<double> &v,
                          const std::vector<double> &coords, int n, int no);

template <>
void pi_lminus1_first<double>(std::vector<double> &v,
                              const std::vector<double> &coords, int n, int no);

template <>
void pi_Ql_first<double>(const int nr, const int nc, const int nrow,
                         const int ncol, const int l, double *v,
                         const std::vector<double> &coords_x,
                         const std::vector<double> &coords_y,
                         std::vector<double> &row_vec,
                         std::vector<double> &col_vec);

template <>
void pi_Ql<double>(const int nr, const int nc, const int nrow, const int ncol,
                   const int l, double *v, const std::vector<double> &coords_x,
                   const std::vector<double> &coords_y,
                   std::vector<double> &row_vec, std::vector<double> &col_vec);

template <>
void pi_Ql3D<double>(const int nr, const int nc, const int nf, const int nrow,
                     const int ncol, const int nfib, const int l, double *v,
                     const std::vector<double> &coords_x,
                     const std::vector<double> &coords_y,
                     const std::vector<double> &coords_z,
                     std::vector<double> &row_vec, std::vector<double> &col_vec,
                     std::vector<double> &fib_vec);

template <>
void pi_Ql3D_first<double>(
    const int nr, const int nc, const int nf, const int nrow, const int ncol,
    const int nfib, const int l, double *v, const std::vector<double> &coords_x,
    const std::vector<double> &coords_y, const std::vector<double> &coords_z,
    std::vector<double> &row_vec, std::vector<double> &col_vec,
    std::vector<double> &fib_vec);

template <>
void assign_num_level<double>(const int l, std::vector<double> &v, double num,
                              int n, int no);

template <>
void assign_num_level_l<double>(const int l, double *v, double num, int nr,
                                int nc, const int nrow, const int ncol);

template <>
void restriction_first<double>(std::vector<double> &v,
                               std::vector<double> &coords, int n, int no);

template <>
void solve_tridiag_M_l<double>(const int l, std::vector<double> &v,
                               std::vector<double> &coords, int n, int no);

template <>
void add_level_l<double>(const int l, double *v, double *work, int nr, int nc,
                         int nrow, int ncol);

template <>
void add3_level_l<double>(const int l, double *v, double *work, int nr, int nc,
                          int nf, int nrow, int ncol, int nfib);

template <>
void sub3_level_l<double>(const int l, double *v, double *work, int nr, int nc,
                          int nf, int nrow, int ncol, int nfib);

template <>
void sub3_level<double>(const int l, double *v, double *work, int nrow,
                        int ncol, int nfib);

template <>
void sub_level_l<double>(const int l, double *v, double *work, int nr, int nc,
                         int nf, int nrow, int ncol, int nfib);

template <>
void project_first<double>(const int nr, const int nc, const int nrow,
                           const int ncol, const int l_target, double *v,
                           std::vector<double> &work,
                           std::vector<double> &coords_x,
                           std::vector<double> &coords_y,
                           std::vector<double> &row_vec,
                           std::vector<double> &col_vec);

template <>
void prep_2D<double>(const int nr, const int nc, const int nrow, const int ncol,
                     const int l_target, double *v, std::vector<double> &work,
                     std::vector<double> &coords_x,
                     std::vector<double> &coords_y,
                     std::vector<double> &row_vec,
                     std::vector<double> &col_vec);

template <>
void mass_mult_l<double>(const int l, std::vector<double> &v,
                         std::vector<double> &coords, const int n,
                         const int no);

template <>
void restriction_l<double>(const int l, std::vector<double> &v,
                           std::vector<double> &coords, int n, int no);

template <>
double ml2_norm3<double>(const int l, int nr, int nc, int nf, int nrow,
                         int ncol, int nfib, const std::vector<double> &v,
                         std::vector<double> &coords_x,
                         std::vector<double> &coords_y,
                         std::vector<double> &coords_z);

template <>
void prolongate_l<double>(const int l, std::vector<double> &v,
                          std::vector<double> &coords, int n, int no);

template <>
void refactor_1D<double>(const int l_target, std::vector<double> &v,
                         std::vector<double> &work, std::vector<double> &coords,
                         int n, int no);

template <>
void refactor_2D<double>(const int nr, const int nc, const int nrow,
                         const int ncol, const int l_target, double *v,
                         std::vector<double> &work,
                         std::vector<double> &coords_x,
                         std::vector<double> &coords_y,
                         std::vector<double> &row_vec,
                         std::vector<double> &col_vec);

template <>
void refactor_2D_full<double>(const int nr, const int nc, const int nrow,
                              const int ncol, const int l_target, double *v,
                              std::vector<double> &work,
                              std::vector<double> &coords_x,
                              std::vector<double> &coords_y,
                              std::vector<double> &row_vec,
                              std::vector<double> &col_vec);

template <>
void refactor_2D_first<double>(const int nr, const int nc, const int nrow,
                               const int ncol, const int l_target, double *v,
                               std::vector<double> &work,
                               std::vector<double> &coords_x,
                               std::vector<double> &coords_y,
                               std::vector<double> &row_vec,
                               std::vector<double> &col_vec);

template <>
void copy3_level_l<double>(const int l, double *v, double *work, int nr, int nc,
                           int nf, int nrow, int ncol, int nfib);

template <>
void copy3_level<double>(const int l, double *v, double *work, int nrow,
                         int ncol, int nfib);

template <>
void assign3_level_l<double>(const int l, double *v, double num, int nr, int nc,
                             int nf, int nrow, int ncol, int nfib);

template <>
void refactor_3D<double>(const int nr, const int nc, const int nf,
                         const int nrow, const int ncol, const int nfib,
                         const int l_target, double *v,
                         std::vector<double> &work, std::vector<double> &work2d,
                         std::vector<double> &coords_x,
                         std::vector<double> &coords_y,
                         std::vector<double> &coords_z);

template <>
void compute_zl<double>(const int nr, const int nc, const int nrow,
                        const int ncol, const int l_target,
                        std::vector<double> &work,
                        std::vector<double> &coords_x,
                        std::vector<double> &coords_y,
                        std::vector<double> &row_vec,
                        std::vector<double> &col_vec);

template <>
void compute_zl_last<double>(const int nr, const int nc, const int nrow,
                             const int ncol, const int l_target,
                             std::vector<double> &work,
                             std::vector<double> &coords_x,
                             std::vector<double> &coords_y,
                             std::vector<double> &row_vec,
                             std::vector<double> &col_vec);

template <>
void prolongate_last<double>(std::vector<double> &v,
                             std::vector<double> &coords, int n, int no);

template <>
void prolong_add_2D<double>(const int nr, const int nc, const int nrow,
                            const int ncol, const int l_target,
                            std::vector<double> &work,
                            std::vector<double> &coords_x,
                            std::vector<double> &coords_y,
                            std::vector<double> &row_vec,
                            std::vector<double> &col_vec);

template <>
void prolong_add_2D_last<double>(const int nr, const int nc, const int nrow,
                                 const int ncol, const int l_target,
                                 std::vector<double> &work,
                                 std::vector<double> &coords_x,
                                 std::vector<double> &coords_y,
                                 std::vector<double> &row_vec,
                                 std::vector<double> &col_vec);

template <>
void prep_3D<double>(const int nr, const int nc, const int nf, const int nrow,
                     const int ncol, const int nfib, const int l_target,
                     double *v, std::vector<double> &work,
                     std::vector<double> &work2d, std::vector<double> &coords_x,
                     std::vector<double> &coords_y,
                     std::vector<double> &coords_z);

template <>
void recompose_3D<double>(
    const int nr, const int nc, const int nf, const int nrow, const int ncol,
    const int nfib, const int l_target, double *v, std::vector<double> &work,
    std::vector<double> &work2d, std::vector<double> &coords_x,
    std::vector<double> &coords_y, std::vector<double> &coords_z);

template <>
void postp_3D<double>(const int nr, const int nc, const int nf, const int nrow,
                      const int ncol, const int nfib, const int l_target,
                      double *v, std::vector<double> &work,
                      std::vector<double> &coords_x,
                      std::vector<double> &coords_y,
                      std::vector<double> &coords_z);

template <>
void recompose_2D<double>(const int nr, const int nc, const int nrow,
                          const int ncol, const int l_target, double *v,
                          std::vector<double> &work,
                          std::vector<double> &coords_x,
                          std::vector<double> &coords_y,
                          std::vector<double> &row_vec,
                          std::vector<double> &col_vec);

template <>
void recompose_2D_full<double>(const int nr, const int nc, const int nrow,
                               const int ncol, const int l_target, double *v,
                               std::vector<double> &work,
                               std::vector<double> &coords_x,
                               std::vector<double> &coords_y,
                               std::vector<double> &row_vec,
                               std::vector<double> &col_vec);

template <>
void postp_2D<double>(const int nr, const int nc, const int nrow,
                      const int ncol, const int l_target, double *v,
                      std::vector<double> &work, std::vector<double> &coords_x,
                      std::vector<double> &coords_y,
                      std::vector<double> &row_vec,
                      std::vector<double> &col_vec);

template <>
void qwrite_2D_l<double>(const int nr, const int nc, const int nrow,
                         const int ncol, const int nlevel, const int l,
                         double *v, double tol, double norm,
                         const std::string outfile);

template <>
double
qoi_norm<double>(int nrow, int ncol, int nfib, std::vector<double> &coords_x,
                 std::vector<double> &coords_y, std::vector<double> &coords_z,
                 double (*qoi)(int, int, int, std::vector<double>), double s);

template <>
double
qoi_norm<double>(int nrow, int ncol, int nfib, std::vector<double> &coords_x,
                 std::vector<double> &coords_y, std::vector<double> &coords_z,
                 double (*qoi)(int, int, int, double *), double s);

} // namespace mgard_gen

namespace mgard_2d {

namespace mgard_common {

template <>
int parse_cmdl<double>(int argc, char **argv, int &nrow, int &ncol, double &tol,
                       std::string &in_file, std::string &coord_file);

template <> double max_norm<double>(const std::vector<double> &v);

template <>
inline double interp_2d<double>(double q11, double q12, double q21, double q22,
                                double x1, double x2, double y1, double y2,
                                double x, double y);

template <>
inline double get_h<double>(const std::vector<double> &coords, int i,
                            int stride);

template <>
inline double get_dist<double>(const std::vector<double> &coords, int i, int j);

template <>
void qread_2D_interleave<double>(const int nrow, const int ncol,
                                 const int nlevel, double *v,
                                 std::string infile);

template <>
void qwrite_2D_interleave<double>(const int nrow, const int ncol,
                                  const int nlevel, const int l, double *v,
                                  double tol, double norm,
                                  const std::string outfile);

} // namespace mgard_common

namespace mgard_cannon {

template <>
void assign_num_level<double>(const int nrow, const int ncol, const int l,
                              double *v, double num);

template <>
void subtract_level<double>(const int nrow, const int ncol, const int l,
                            double *v, double *work);

template <>
void pi_lminus1<double>(const int l, std::vector<double> &v,
                        const std::vector<double> &coords);

template <>
void restriction<double>(const int l, std::vector<double> &v,
                         const std::vector<double> &coords);

template <>
void prolongate<double>(const int l, std::vector<double> &v,
                        const std::vector<double> &coords);

template <>
void solve_tridiag_M<double>(const int l, std::vector<double> &v,
                             const std::vector<double> &coords);

template <>
void mass_matrix_multiply<double>(const int l, std::vector<double> &v,
                                  const std::vector<double> &coords);

template <>
void write_level_2D<double>(const int nrow, const int ncol, const int l,
                            double *v, std::ofstream &outfile);

template <>
void copy_level<double>(const int nrow, const int ncol, const int l, double *v,
                        std::vector<double> &work);

} // namespace mgard_cannon

namespace mgard_gen {

template <>
inline double *get_ref<double>(std::vector<double> &v, const int n,
                               const int no, const int i);

template <>
inline double get_h_l<double>(const std::vector<double> &coords, const int n,
                              const int no, int i, int stride);

template <>
void write_level_2D_l<double>(const int l, double *v, std::ofstream &outfile,
                              int nr, int nc, int nrow, int ncol);

template <>
void copy_level_l<double>(const int l, double *v, double *work, int nr, int nc,
                          int nrow, int ncol);

template <>
void subtract_level_l<double>(const int l, double *v, double *work, int nr,
                              int nc, int nrow, int ncol);

template <>
void pi_lminus1_l<double>(const int l, std::vector<double> &v,
                          const std::vector<double> &coords, int n, int no);

template <>
void pi_lminus1_first<double>(std::vector<double> &v,
                              const std::vector<double> &coords, int n, int no);

template <>
void pi_Ql_first<double>(const int nr, const int nc, const int nrow,
                         const int ncol, const int l, double *v,
                         const std::vector<double> &coords_x,
                         const std::vector<double> &coords_y,
                         std::vector<double> &row_vec,
                         std::vector<double> &col_vec);

template <>
void pi_Ql<double>(const int nr, const int nc, const int nrow, const int ncol,
                   const int l, double *v, const std::vector<double> &coords_x,
                   const std::vector<double> &coords_y,
                   std::vector<double> &row_vec, std::vector<double> &col_vec);

template <>
void assign_num_level_l<double>(const int l, double *v, double num, int nr,
                                int nc, const int nrow, const int ncol);

template <>
void restriction_first<double>(std::vector<double> &v,
                               std::vector<double> &coords, int n, int no);

template <>
void solve_tridiag_M_l<double>(const int l, std::vector<double> &v,
                               std::vector<double> &coords, int n, int no);

template <>
void add_level_l<double>(const int l, double *v, double *work, int nr, int nc,
                         int nrow, int ncol);

template <>
void project_first<double>(const int nr, const int nc, const int nrow,
                           const int ncol, const int l_target, double *v,
                           std::vector<double> &work,
                           std::vector<double> &coords_x,
                           std::vector<double> &coords_y,
                           std::vector<double> &row_vec,
                           std::vector<double> &col_vec);

template <>
void prep_2D<double>(const int nr, const int nc, const int nrow, const int ncol,
                     const int l_target, double *v, std::vector<double> &work,
                     std::vector<double> &coords_x,
                     std::vector<double> &coords_y,
                     std::vector<double> &row_vec,
                     std::vector<double> &col_vec);

template <>
void mass_mult_l<double>(const int l, std::vector<double> &v,
                         std::vector<double> &coords, const int n,
                         const int no);

template <>
void restriction_l<double>(const int l, std::vector<double> &v,
                           std::vector<double> &coords, int n, int no);

template <>
void prolongate_l<double>(const int l, std::vector<double> &v,
                          std::vector<double> &coords, int n, int no);

template <>
void refactor_2D<double>(const int nr, const int nc, const int nrow,
                         const int ncol, const int l_target, double *v,
                         std::vector<double> &work,
                         std::vector<double> &coords_x,
                         std::vector<double> &coords_y,
                         std::vector<double> &row_vec,
                         std::vector<double> &col_vec);

template <>
void recompose_2D<double>(const int nr, const int nc, const int nrow,
                          const int ncol, const int l_target, double *v,
                          std::vector<double> &work,
                          std::vector<double> &coords_x,
                          std::vector<double> &coords_y,
                          std::vector<double> &row_vec,
                          std::vector<double> &col_vec);

template <>
void prolongate_last<double>(std::vector<double> &v,
                             std::vector<double> &coords, int n, int no);

template <>
void postp_2D<double>(const int nr, const int nc, const int nrow,
                      const int ncol, const int l_target, double *v,
                      std::vector<double> &work, std::vector<double> &coords_x,
                      std::vector<double> &coords_y,
                      std::vector<double> &row_vec,
                      std::vector<double> &col_vec);

template <>
void qwrite_2D_l<double>(const int nr, const int nc, const int nrow,
                         const int ncol, const int nlevel, const int l,
                         double *v, double tol, double norm,
                         const std::string outfile);

} // namespace mgard_gen

} // namespace mgard_2d

#include "mgard_nuni.tpp"
