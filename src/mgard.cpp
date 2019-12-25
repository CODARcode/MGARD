#include "mgard.h"

namespace mgard {

template <>
inline float interp_2d<float>(float q11, float q12, float q21, float q22,
                              float x1, float x2, float y1, float y2, float x,
                              float y);

template <>
inline double interp_2d<double>(double q11, double q12, double q21, double q22,
                                double x1, double x2, double y1, double y2,
                                double x, double y);

template <>
inline float interp_0d<float>(const float x1, const float x2, const float y1,
                              const float y2, const float x);

template <>
inline double interp_0d<double>(const double x1, const double x2,
                                const double y1, const double y2,
                                const double x);

template <>
void mass_matrix_multiply<float>(const int l, std::vector<float> &v);

template <>
void mass_matrix_multiply<double>(const int l, std::vector<double> &v);

template <> void solve_tridiag_M<float>(const int l, std::vector<float> &v);

template <> void solve_tridiag_M<double>(const int l, std::vector<double> &v);

template <> void restriction<float>(const int l, std::vector<float> &v);

template <> void restriction<double>(const int l, std::vector<double> &v);

template <>
void interpolate_from_level_nMl<float>(const int l, std::vector<float> &v);

template <>
void interpolate_from_level_nMl<double>(const int l, std::vector<double> &v);

template <>
void print_level_2D<float>(const int nrow, const int ncol, const int l,
                           float *v);

template <>
void print_level_2D<double>(const int nrow, const int ncol, const int l,
                            double *v);

template <>
void write_level_2D<float>(const int nrow, const int ncol, const int l,
                           float *v, std::ofstream &outfile);

template <>
void write_level_2D<double>(const int nrow, const int ncol, const int l,
                            double *v, std::ofstream &outfile);

template <>
void write_level_2D_exc<float>(const int nrow, const int ncol, const int l,
                               float *v, std::ofstream &outfile);

template <>
void write_level_2D_exc<double>(const int nrow, const int ncol, const int l,
                                double *v, std::ofstream &outfile);

template <> void pi_lminus1<float>(const int l, std::vector<float> &v0);

template <> void pi_lminus1<double>(const int l, std::vector<double> &v0);

template <>
void pi_Ql<float>(const int nrow, const int ncol, const int l, float *v,
                  std::vector<float> &row_vec, std::vector<float> &col_vec);

template <>
void pi_Ql<double>(const int nrow, const int ncol, const int l, double *v,
                   std::vector<double> &row_vec, std::vector<double> &col_vec);

template <>
void assign_num_level<float>(const int nrow, const int ncol, const int l,
                             float *v, float num);

template <>
void assign_num_level<double>(const int nrow, const int ncol, const int l,
                              double *v, double num);

template <>
void copy_level<float>(const int nrow, const int ncol, const int l, float *v,
                       std::vector<float> &work);

template <>
void copy_level<double>(const int nrow, const int ncol, const int l, double *v,
                        std::vector<double> &work);

template <>
void add_level<float>(const int nrow, const int ncol, const int l, float *v,
                      float *work);

template <>
void add_level<double>(const int nrow, const int ncol, const int l, double *v,
                       double *work);

template <>
void subtract_level<float>(const int nrow, const int ncol, const int l,
                           float *v, float *work);

template <>
void subtract_level<double>(const int nrow, const int ncol, const int l,
                            double *v, double *work);

template <>
void compute_correction_loadv<float>(const int l, std::vector<float> &v);

template <>
void compute_correction_loadv<double>(const int l, std::vector<double> &v);

template <>
void qwrite_level_2D<float>(const int nrow, const int ncol, const int nlevel,
                            const int l, float *v, float tol,
                            const std::string outfile);

template <>
void qwrite_level_2D<double>(const int nrow, const int ncol, const int nlevel,
                             const int l, double *v, double tol,
                             const std::string outfile);

template <>
void quantize_2D_interleave<float>(const int nrow, const int ncol, float *v,
                                   std::vector<int> &work, float norm,
                                   float tol);

template <>
void quantize_2D_interleave<double>(const int nrow, const int ncol, double *v,
                                    std::vector<int> &work, double norm,
                                    double tol);

template <>
void dequantize_2D_interleave<float>(const int nrow, const int ncol, float *v,
                                     const std::vector<int> &work);

template <>
void dequantize_2D_interleave<double>(const int nrow, const int ncol, double *v,
                                      const std::vector<int> &work);

template <>
void qread_level_2D<float>(const int nrow, const int ncol, const int nlevel,
                           float *v, std::string infile);

template <>
void qread_level_2D<double>(const int nrow, const int ncol, const int nlevel,
                            double *v, std::string infile);

template <>
void resample_1d<float>(const float *inbuf, float *outbuf, const int ncol,
                        const int ncol_new);

template <>
void resample_1d<double>(const double *inbuf, double *outbuf, const int ncol,
                         const int ncol_new);

template <>
void resample_2d<float>(const float *inbuf, float *outbuf, const int nrow,
                        const int ncol, const int nrow_new, const int ncol_new);

template <>
void resample_2d<double>(const double *inbuf, double *outbuf, const int nrow,
                         const int ncol, const int nrow_new,
                         const int ncol_new);

template <>
void resample_2d_inv2<float>(const float *inbuf, float *outbuf, const int nrow,
                             const int ncol, const int nrow_new,
                             const int ncol_new);

template <>
void resample_2d_inv2<double>(const double *inbuf, double *outbuf,
                              const int nrow, const int ncol,
                              const int nrow_new, const int ncol_new);

template <>
unsigned char *refactor_qz<float>(int nrow, int ncol, int nfib, const float *v,
                                  int &outsize, float tol);

template <>
unsigned char *refactor_qz<double>(int nrow, int ncol, int nfib,
                                   const double *v, int &outsize, double tol);

template <>
unsigned char *refactor_qz<float>(int nrow, int ncol, int nfib, const float *v,
                                  int &outsize, float tol, float s);

template <>
unsigned char *refactor_qz<double>(int nrow, int ncol, int nfib,
                                   const double *v, int &outsize, double tol,
                                   double s);

template <>
unsigned char *
refactor_qz<float>(int nrow, int ncol, int nfib, std::vector<float> &coords_x,
                   std::vector<float> &coords_y, std::vector<float> &coords_z,
                   const float *v, int &outsize, float tol);

template <>
unsigned char *refactor_qz<double>(int nrow, int ncol, int nfib,
                                   std::vector<double> &coords_x,
                                   std::vector<double> &coords_y,
                                   std::vector<double> &coords_z,
                                   const double *v, int &outsize, double tol);

template <>
unsigned char *
refactor_qz<float>(int nrow, int ncol, int nfib, std::vector<float> &coords_x,
                   std::vector<float> &coords_y, std::vector<float> &coords_z,
                   const float *v, int &outsize, float tol, float s);

template <>
unsigned char *
refactor_qz<double>(int nrow, int ncol, int nfib, std::vector<double> &coords_x,
                    std::vector<double> &coords_y,
                    std::vector<double> &coords_z, const double *v,
                    int &outsize, double tol, double s);

template <>
unsigned char *refactor_qz_2D<float>(int nrow, int ncol, const float *v,
                                     int &outsize, float tol);

template <>
unsigned char *refactor_qz_2D<double>(int nrow, int ncol, const double *v,
                                      int &outsize, double tol);

template <>
unsigned char *refactor_qz_2D<float>(int nrow, int ncol, const float *v,
                                     int &outsize, float tol, float s);

template <>
unsigned char *refactor_qz_2D<double>(int nrow, int ncol, const double *v,
                                      int &outsize, double tol, double s);

template <>
unsigned char *refactor_qz_2D<float>(int nrow, int ncol,
                                     std::vector<float> &coords_x,
                                     std::vector<float> &coords_y,
                                     const float *v, int &outsize, float tol);

template <>
unsigned char *
refactor_qz_2D<double>(int nrow, int ncol, std::vector<double> &coords_x,
                       std::vector<double> &coords_y, const double *v,
                       int &outsize, double tol);

template <>
unsigned char *
refactor_qz_2D<float>(int nrow, int ncol, std::vector<float> &coords_x,
                      std::vector<float> &coords_y, const float *v,
                      int &outsize, float tol, float s);

template <>
unsigned char *
refactor_qz_2D<double>(int nrow, int ncol, std::vector<double> &coords_x,
                       std::vector<double> &coords_y, const double *v,
                       int &outsize, double tol, double s);

template <>
float *recompose_udq<float>(int nrow, int ncol, int nfib, unsigned char *data,
                            int data_len);

template <>
double *recompose_udq<double>(int nrow, int ncol, int nfib, unsigned char *data,
                              int data_len);

template <>
float *
recompose_udq<float>(int nrow, int ncol, int nfib, std::vector<float> &coords_x,
                     std::vector<float> &coords_y, std::vector<float> &coords_z,
                     unsigned char *data, int data_len);

template <>
double *recompose_udq<double>(int nrow, int ncol, int nfib,
                              std::vector<double> &coords_x,
                              std::vector<double> &coords_y,
                              std::vector<double> &coords_z,
                              unsigned char *data, int data_len);

template <>
float *recompose_udq<float>(int nrow, int ncol, int nfib, unsigned char *data,
                            int data_len, float s);

template <>
double *recompose_udq<double>(int nrow, int ncol, int nfib, unsigned char *data,
                              int data_len, double s);

template <>
float *
recompose_udq<float>(int nrow, int ncol, int nfib, std::vector<float> &coords_x,
                     std::vector<float> &coords_y, std::vector<float> &coords_z,
                     unsigned char *data, int data_len, float s);

template <>
double *recompose_udq<double>(int nrow, int ncol, int nfib,
                              std::vector<double> &coords_x,
                              std::vector<double> &coords_y,
                              std::vector<double> &coords_z,
                              unsigned char *data, int data_len, double s);

template <>
float *recompose_udq_2D<float>(int nrow, int ncol, unsigned char *data,
                               int data_len);

template <>
double *recompose_udq_2D<double>(int nrow, int ncol, unsigned char *data,
                                 int data_len);

template <>
float *recompose_udq_2D<float>(int nrow, int ncol, unsigned char *data,
                               int data_len, float s);

template <>
double *recompose_udq_2D<double>(int nrow, int ncol, unsigned char *data,
                                 int data_len, double s);

template <>
float *recompose_udq_2D<float>(int nrow, int ncol, std::vector<float> &coords_x,
                               std::vector<float> &coords_y,
                               unsigned char *data, int data_len);

template <>
double *recompose_udq_2D<double>(int nrow, int ncol,
                                 std::vector<double> &coords_x,
                                 std::vector<double> &coords_y,
                                 unsigned char *data, int data_len);

template <>
float *recompose_udq_2D<float>(int nrow, int ncol, std::vector<float> &coords_x,
                               std::vector<float> &coords_y,
                               unsigned char *data, int data_len, float s);

template <>
double *recompose_udq_2D<double>(int nrow, int ncol,
                                 std::vector<double> &coords_x,
                                 std::vector<double> &coords_y,
                                 unsigned char *data, int data_len, double s);

template <>
unsigned char *refactor_qz_1D<float>(int ncol, const float *v, int &outsize,
                                     float tol);

template <>
unsigned char *refactor_qz_1D<double>(int ncol, const double *v, int &outsize,
                                      double tol);

template <>
float *recompose_udq_1D<float>(int nrow, unsigned char *data, int data_len);

template <>
double *recompose_udq_1D<double>(int nrow, unsigned char *data, int data_len);

template <>
int parse_cmdl<float>(int argc, char **argv, int &nrow, int &ncol, float &tol,
                      std::string &in_file);

template <>
int parse_cmdl<double>(int argc, char **argv, int &nrow, int &ncol, double &tol,
                       std::string &in_file);

template <>
void refactor<float>(const int nrow, const int ncol, const int l_target,
                     float *v, std::vector<float> &work,
                     std::vector<float> &row_vec, std::vector<float> &col_vec);

template <>
void refactor<double>(const int nrow, const int ncol, const int l_target,
                      double *v, std::vector<double> &work,
                      std::vector<double> &row_vec,
                      std::vector<double> &col_vec);

template <>
void recompose<float>(const int nrow, const int ncol, const int l_target,
                      float *v, std::vector<float> &work,
                      std::vector<float> &row_vec, std::vector<float> &col_vec);

template <>
void recompose<double>(const int nrow, const int ncol, const int l_target,
                       double *v, std::vector<double> &work,
                       std::vector<double> &row_vec,
                       std::vector<double> &col_vec);

} // namespace mgard
