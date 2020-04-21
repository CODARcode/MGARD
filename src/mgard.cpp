#include "mgard.h"
#include "mgard.tpp"

namespace mgard {

template void mass_matrix_multiply<float>(const int l, std::vector<float> &v);

template void mass_matrix_multiply<double>(const int l, std::vector<double> &v);

template void solve_tridiag_M<float>(const int l, std::vector<float> &v);

template void solve_tridiag_M<double>(const int l, std::vector<double> &v);

template void restriction<float>(const int l, std::vector<float> &v);

template void restriction<double>(const int l, std::vector<double> &v);

template void interpolate_from_level_nMl<float>(const int l,
                                                std::vector<float> &v);

template void interpolate_from_level_nMl<double>(const int l,
                                                 std::vector<double> &v);

template void pi_lminus1<float>(const int l, std::vector<float> &v0);

template void pi_lminus1<double>(const int l, std::vector<double> &v0);

template void pi_Ql<float>(const int nrow, const int ncol, const int l,
                           float *v, std::vector<float> &row_vec,
                           std::vector<float> &col_vec);

template void pi_Ql<double>(const int nrow, const int ncol, const int l,
                            double *v, std::vector<double> &row_vec,
                            std::vector<double> &col_vec);

template void assign_num_level<float>(const int nrow, const int ncol,
                                      const int l, float *v, float num);

template void assign_num_level<double>(const int nrow, const int ncol,
                                       const int l, double *v, double num);

template void copy_level<float>(const int nrow, const int ncol, const int l,
                                float *v, std::vector<float> &work);

template void copy_level<double>(const int nrow, const int ncol, const int l,
                                 double *v, std::vector<double> &work);

template void add_level<float>(const int nrow, const int ncol, const int l,
                               float *v, float *work);

template void add_level<double>(const int nrow, const int ncol, const int l,
                                double *v, double *work);

template void subtract_level<float>(const int nrow, const int ncol, const int l,
                                    float *v, float *work);

template void subtract_level<double>(const int nrow, const int ncol,
                                     const int l, double *v, double *work);

template void compute_correction_loadv<float>(const int l,
                                              std::vector<float> &v);

template void compute_correction_loadv<double>(const int l,
                                               std::vector<double> &v);

template void quantize_2D_interleave<float>(const int nrow, const int ncol,
                                            float *v, std::vector<int> &work,
                                            float norm, float tol);

template void quantize_2D_interleave<double>(const int nrow, const int ncol,
                                             double *v, std::vector<int> &work,
                                             double norm, double tol);

template void dequantize_2D_interleave<float>(const int nrow, const int ncol,
                                              float *v,
                                              const std::vector<int> &work);

template void dequantize_2D_interleave<double>(const int nrow, const int ncol,
                                               double *v,
                                               const std::vector<int> &work);

template unsigned char *refactor_qz<float>(int nrow, int ncol, int nfib,
                                           const float *v, int &outsize,
                                           float tol);

template unsigned char *refactor_qz<double>(int nrow, int ncol, int nfib,
                                            const double *v, int &outsize,
                                            double tol);

template unsigned char *refactor_qz<float>(int nrow, int ncol, int nfib,
                                           const float *v, int &outsize,
                                           float tol, float s);

template unsigned char *refactor_qz<double>(int nrow, int ncol, int nfib,
                                            const double *v, int &outsize,
                                            double tol, double s);

template unsigned char *
refactor_qz<float>(int nrow, int ncol, int nfib, std::vector<float> &coords_x,
                   std::vector<float> &coords_y, std::vector<float> &coords_z,
                   const float *v, int &outsize, float tol);

template unsigned char *refactor_qz<double>(int nrow, int ncol, int nfib,
                                            std::vector<double> &coords_x,
                                            std::vector<double> &coords_y,
                                            std::vector<double> &coords_z,
                                            const double *v, int &outsize,
                                            double tol);

template unsigned char *
refactor_qz<float>(int nrow, int ncol, int nfib, std::vector<float> &coords_x,
                   std::vector<float> &coords_y, std::vector<float> &coords_z,
                   const float *v, int &outsize, float tol, float s);

template unsigned char *refactor_qz<double>(int nrow, int ncol, int nfib,
                                            std::vector<double> &coords_x,
                                            std::vector<double> &coords_y,
                                            std::vector<double> &coords_z,
                                            const double *v, int &outsize,
                                            double tol, double s);

template unsigned char *refactor_qz_1D<float>(int ncol, const float *u,
                                              int &outsize, float tol);

template unsigned char *refactor_qz_1D<double>(int ncol, const double *u,
                                               int &outsize, double tol);

template unsigned char *refactor_qz_2D<float>(int nrow, int ncol,
                                              const float *v, int &outsize,
                                              float tol);

template unsigned char *refactor_qz_2D<double>(int nrow, int ncol,
                                               const double *v, int &outsize,
                                               double tol);

template unsigned char *refactor_qz_2D<float>(int nrow, int ncol,
                                              const float *v, int &outsize,
                                              float tol, float s);

template unsigned char *refactor_qz_2D<double>(int nrow, int ncol,
                                               const double *v, int &outsize,
                                               double tol, double s);

template unsigned char *refactor_qz_2D<float>(int nrow, int ncol,
                                              std::vector<float> &coords_x,
                                              std::vector<float> &coords_y,
                                              const float *v, int &outsize,
                                              float tol);

template unsigned char *refactor_qz_2D<double>(int nrow, int ncol,
                                               std::vector<double> &coords_x,
                                               std::vector<double> &coords_y,
                                               const double *v, int &outsize,
                                               double tol);

template unsigned char *refactor_qz_2D<float>(int nrow, int ncol,
                                              std::vector<float> &coords_x,
                                              std::vector<float> &coords_y,
                                              const float *v, int &outsize,
                                              float tol, float s);

template unsigned char *refactor_qz_2D<double>(int nrow, int ncol,
                                               std::vector<double> &coords_x,
                                               std::vector<double> &coords_y,
                                               const double *v, int &outsize,
                                               double tol, double s);

template float *recompose_udq<float>(int nrow, int ncol, int nfib,
                                     unsigned char *data, int data_len);

template double *recompose_udq<double>(int nrow, int ncol, int nfib,
                                       unsigned char *data, int data_len);

template float *recompose_udq<float>(int nrow, int ncol, int nfib,
                                     std::vector<float> &coords_x,
                                     std::vector<float> &coords_y,
                                     std::vector<float> &coords_z,
                                     unsigned char *data, int data_len);

template double *recompose_udq<double>(int nrow, int ncol, int nfib,
                                       std::vector<double> &coords_x,
                                       std::vector<double> &coords_y,
                                       std::vector<double> &coords_z,
                                       unsigned char *data, int data_len);

template float *recompose_udq<float>(int nrow, int ncol, int nfib,
                                     unsigned char *data, int data_len,
                                     float s);

template double *recompose_udq<double>(int nrow, int ncol, int nfib,
                                       unsigned char *data, int data_len,
                                       double s);

template float *
recompose_udq<float>(int nrow, int ncol, int nfib, std::vector<float> &coords_x,
                     std::vector<float> &coords_y, std::vector<float> &coords_z,
                     unsigned char *data, int data_len, float s);

template double *recompose_udq<double>(int nrow, int ncol, int nfib,
                                       std::vector<double> &coords_x,
                                       std::vector<double> &coords_y,
                                       std::vector<double> &coords_z,
                                       unsigned char *data, int data_len,
                                       double s);

template float *recompose_udq_1D<float>(int ncol, unsigned char *data,
                                        int data_len);

template double *recompose_udq_1D<double>(int ncol, unsigned char *data,
                                          int data_len);

template float *recompose_udq_1D_huffman<float>(int ncol, unsigned char *data,
                                                int data_len);

template double *recompose_udq_1D_huffman<double>(int ncol, unsigned char *data,
                                                  int data_len);

template float *recompose_udq_2D<float>(int nrow, int ncol, unsigned char *data,
                                        int data_len);

template double *recompose_udq_2D<double>(int nrow, int ncol,
                                          unsigned char *data, int data_len);

template float *recompose_udq_2D<float>(int nrow, int ncol, unsigned char *data,
                                        int data_len, float s);

template double *recompose_udq_2D<double>(int nrow, int ncol,
                                          unsigned char *data, int data_len,
                                          double s);

template float *recompose_udq_2D<float>(int nrow, int ncol,
                                        std::vector<float> &coords_x,
                                        std::vector<float> &coords_y,
                                        unsigned char *data, int data_len);

template double *recompose_udq_2D<double>(int nrow, int ncol,
                                          std::vector<double> &coords_x,
                                          std::vector<double> &coords_y,
                                          unsigned char *data, int data_len);

template float *recompose_udq_2D<float>(int nrow, int ncol,
                                        std::vector<float> &coords_x,
                                        std::vector<float> &coords_y,
                                        unsigned char *data, int data_len,
                                        float s);

template double *recompose_udq_2D<double>(int nrow, int ncol,
                                          std::vector<double> &coords_x,
                                          std::vector<double> &coords_y,
                                          unsigned char *data, int data_len,
                                          double s);

template void refactor<float>(const int nrow, const int ncol,
                              const int l_target, float *v,
                              std::vector<float> &work,
                              std::vector<float> &row_vec,
                              std::vector<float> &col_vec);

template void refactor<double>(const int nrow, const int ncol,
                               const int l_target, double *v,
                               std::vector<double> &work,
                               std::vector<double> &row_vec,
                               std::vector<double> &col_vec);

template void recompose<float>(const int nrow, const int ncol,
                               const int l_target, float *v,
                               std::vector<float> &work,
                               std::vector<float> &row_vec,
                               std::vector<float> &col_vec);

template void recompose<double>(const int nrow, const int ncol,
                                const int l_target, double *v,
                                std::vector<double> &work,
                                std::vector<double> &row_vec,
                                std::vector<double> &col_vec);

} // namespace mgard
