#include "mgard.hpp"
#include "mgard.tpp"

namespace mgard {

template void
mass_matrix_multiply<1, float>(const TensorMeshHierarchy<1, float> &hierarchy,
                               const int l, const std::size_t dimension,
                               float *const v);

template void
mass_matrix_multiply<2, float>(const TensorMeshHierarchy<2, float> &hierarchy,
                               const int l, const std::size_t dimension,
                               float *const v);

template void
mass_matrix_multiply<3, float>(const TensorMeshHierarchy<3, float> &hierarchy,
                               const int l, const std::size_t dimension,
                               float *const v);

template void
mass_matrix_multiply<1, double>(const TensorMeshHierarchy<1, double> &hierarchy,
                                const int l, const std::size_t dimension,
                                double *const v);

template void
mass_matrix_multiply<2, double>(const TensorMeshHierarchy<2, double> &hierarchy,
                                const int l, const std::size_t dimension,
                                double *const v);

template void
mass_matrix_multiply<3, double>(const TensorMeshHierarchy<3, double> &hierarchy,
                                const int l, const std::size_t dimension,
                                double *const v);

template void
solve_tridiag_M<1, float>(const TensorMeshHierarchy<1, float> &hierarchy,
                          const int l, const std::size_t dimension,
                          float *const v);

template void
solve_tridiag_M<2, float>(const TensorMeshHierarchy<2, float> &hierarchy,
                          const int l, const std::size_t dimension,
                          float *const v);

template void
solve_tridiag_M<3, float>(const TensorMeshHierarchy<3, float> &hierarchy,
                          const int l, const std::size_t dimension,
                          float *const v);

template void
solve_tridiag_M<1, double>(const TensorMeshHierarchy<1, double> &hierarchy,
                           const int l, const std::size_t dimension,
                           double *const v);

template void
solve_tridiag_M<2, double>(const TensorMeshHierarchy<2, double> &hierarchy,
                           const int l, const std::size_t dimension,
                           double *const v);

template void
solve_tridiag_M<3, double>(const TensorMeshHierarchy<3, double> &hierarchy,
                           const int l, const std::size_t dimension,
                           double *const v);

template void
restriction<1, float>(const TensorMeshHierarchy<1, float> &hierarchy,
                      const int l, const std::size_t dimension, float *const v);

template void
restriction<2, float>(const TensorMeshHierarchy<2, float> &hierarchy,
                      const int l, const std::size_t dimension, float *const v);

template void
restriction<3, float>(const TensorMeshHierarchy<3, float> &hierarchy,
                      const int l, const std::size_t dimension, float *const v);

template void
restriction<1, double>(const TensorMeshHierarchy<1, double> &hierarchy,
                       const int l, const std::size_t dimension,
                       double *const v);

template void
restriction<2, double>(const TensorMeshHierarchy<2, double> &hierarchy,
                       const int l, const std::size_t dimension,
                       double *const v);

template void
restriction<3, double>(const TensorMeshHierarchy<3, double> &hierarchy,
                       const int l, const std::size_t dimension,
                       double *const v);

template void interpolate_old_to_new_and_overwrite<1, float>(
    const TensorMeshHierarchy<1, float> &hierarchy, const int l,
    const std::size_t dimension, float *const v);

template void interpolate_old_to_new_and_overwrite<2, float>(
    const TensorMeshHierarchy<2, float> &hierarchy, const int l,
    const std::size_t dimension, float *const v);

template void interpolate_old_to_new_and_overwrite<3, float>(
    const TensorMeshHierarchy<3, float> &hierarchy, const int l,
    const std::size_t dimension, float *const v);

template void interpolate_old_to_new_and_overwrite<1, double>(
    const TensorMeshHierarchy<1, double> &hierarchy, const int l,
    const std::size_t dimension, double *const v);

template void interpolate_old_to_new_and_overwrite<2, double>(
    const TensorMeshHierarchy<2, double> &hierarchy, const int l,
    const std::size_t dimension, double *const v);

template void interpolate_old_to_new_and_overwrite<3, double>(
    const TensorMeshHierarchy<3, double> &hierarchy, const int l,
    const std::size_t dimension, double *const v);

template void interpolate_old_to_new_and_subtract<1, float>(
    const TensorMeshHierarchy<1, float> &hierarchy, const int l,
    const std::size_t dimension, float *const v);

template void interpolate_old_to_new_and_subtract<2, float>(
    const TensorMeshHierarchy<2, float> &hierarchy, const int l,
    const std::size_t dimension, float *const v);

template void interpolate_old_to_new_and_subtract<3, float>(
    const TensorMeshHierarchy<3, float> &hierarchy, const int l,
    const std::size_t dimension, float *const v);

template void interpolate_old_to_new_and_subtract<1, double>(
    const TensorMeshHierarchy<1, double> &hierarchy, const int l,
    const std::size_t dimension, double *const v);

template void interpolate_old_to_new_and_subtract<2, double>(
    const TensorMeshHierarchy<2, double> &hierarchy, const int l,
    const std::size_t dimension, double *const v);

template void interpolate_old_to_new_and_subtract<3, double>(
    const TensorMeshHierarchy<3, double> &hierarchy, const int l,
    const std::size_t dimension, double *const v);

template void interpolate_old_to_new_and_subtract<1, float>(
    const TensorMeshHierarchy<1, float> &hierarchy, const int index_difference,
    float *const v);

template void interpolate_old_to_new_and_subtract<1, double>(
    const TensorMeshHierarchy<1, double> &hierarchy, const int index_difference,
    double *const v);

template void interpolate_old_to_new_and_subtract<2, float>(
    const TensorMeshHierarchy<2, float> &hierarchy, const int index_difference,
    float *const v);

template void interpolate_old_to_new_and_subtract<2, double>(
    const TensorMeshHierarchy<2, double> &hierarchy, const int index_difference,
    double *const v);

template void interpolate_old_to_new_and_subtract<3, float>(
    const TensorMeshHierarchy<3, float> &hierarchy, const int index_difference,
    float *const v);

template void interpolate_old_to_new_and_subtract<3, double>(
    const TensorMeshHierarchy<3, double> &hierarchy, const int index_difference,
    double *const v);

template void
assign_num_level<1, float>(const TensorMeshHierarchy<1, float> &hierarchy,
                           const int l, float *const v, const float num);

template void
assign_num_level<2, float>(const TensorMeshHierarchy<2, float> &hierarchy,
                           const int l, float *const v, const float num);

template void
assign_num_level<3, float>(const TensorMeshHierarchy<3, float> &hierarchy,
                           const int l, float *const v, const float num);

template void
assign_num_level<1, double>(const TensorMeshHierarchy<1, double> &hierarchy,
                            const int l, double *const v, const double num);

template void
assign_num_level<2, double>(const TensorMeshHierarchy<2, double> &hierarchy,
                            const int l, double *const v, const double num);

template void
assign_num_level<3, double>(const TensorMeshHierarchy<3, double> &hierarchy,
                            const int l, double *const v, const double num);

template void
copy_level<1, float>(const TensorMeshHierarchy<1, float> &hierarchy,
                     const int l, float const *const v, float *const work);

template void
copy_level<2, float>(const TensorMeshHierarchy<2, float> &hierarchy,
                     const int l, float const *const v, float *const work);

template void
copy_level<3, float>(const TensorMeshHierarchy<3, float> &hierarchy,
                     const int l, float const *const v, float *const work);

template void
copy_level<1, double>(const TensorMeshHierarchy<1, double> &hierarchy,
                      const int l, double const *const v, double *const work);

template void
copy_level<2, double>(const TensorMeshHierarchy<2, double> &hierarchy,
                      const int l, double const *const v, double *const work);

template void
copy_level<3, double>(const TensorMeshHierarchy<3, double> &hierarchy,
                      const int l, double const *const v, double *const work);

template void
add_level<1, float>(const TensorMeshHierarchy<1, float> &hierarchy, const int l,
                    float *const v, float const *const work);

template void
add_level<2, float>(const TensorMeshHierarchy<2, float> &hierarchy, const int l,
                    float *const v, float const *const work);

template void
add_level<3, float>(const TensorMeshHierarchy<3, float> &hierarchy, const int l,
                    float *const v, float const *const work);

template void
add_level<1, double>(const TensorMeshHierarchy<1, double> &hierarchy,
                     const int l, double *const v, double const *const work);

template void
add_level<2, double>(const TensorMeshHierarchy<2, double> &hierarchy,
                     const int l, double *const v, double const *const work);

template void
add_level<3, double>(const TensorMeshHierarchy<3, double> &hierarchy,
                     const int l, double *const v, double const *const work);

template void
subtract_level<1, float>(const TensorMeshHierarchy<1, float> &hierarchy,
                         const int l, float *const v, float const *const work);

template void
subtract_level<2, float>(const TensorMeshHierarchy<2, float> &hierarchy,
                         const int l, float *const v, float const *const work);

template void
subtract_level<3, float>(const TensorMeshHierarchy<3, float> &hierarchy,
                         const int l, float *const v, float const *const work);

template void
subtract_level<1, double>(const TensorMeshHierarchy<1, double> &hierarchy,
                          const int l, double *const v,
                          double const *const work);

template void
subtract_level<2, double>(const TensorMeshHierarchy<2, double> &hierarchy,
                          const int l, double *const v,
                          double const *const work);

template void
subtract_level<3, double>(const TensorMeshHierarchy<3, double> &hierarchy,
                          const int l, double *const v,
                          double const *const work);

// We will probably be removing these instantiations soon.

template void
quantize_interleave<1, float>(const TensorMeshHierarchy<1, float> &hierarchy,
                              float const *const v, int *const work,
                              const float norm, const float tol);

template void
quantize_interleave<2, float>(const TensorMeshHierarchy<2, float> &hierarchy,
                              float const *const v, int *const work,
                              const float norm, const float tol);
template void
quantize_interleave<3, float>(const TensorMeshHierarchy<3, float> &hierarchy,
                              float const *const v, int *const work,
                              const float norm, const float tol);

template void
quantize_interleave<1, double>(const TensorMeshHierarchy<1, double> &hierarchy,
                               double const *const v, int *const work,
                               const double norm, const double tol);

template void
quantize_interleave<2, double>(const TensorMeshHierarchy<2, double> &hierarchy,
                               double const *const v, int *const work,
                               const double norm, const double tol);
template void
quantize_interleave<3, double>(const TensorMeshHierarchy<3, double> &hierarchy,
                               double const *const v, int *const work,
                               const double norm, const double tol);

template void
dequantize_interleave(const TensorMeshHierarchy<1, float> &hierarchy,
                      float *const v, int const *const work);

template void
dequantize_interleave(const TensorMeshHierarchy<2, float> &hierarchy,
                      float *const v, int const *const work);

template void
dequantize_interleave(const TensorMeshHierarchy<3, float> &hierarchy,
                      float *const v, int const *const work);

template void
dequantize_interleave(const TensorMeshHierarchy<1, double> &hierarchy,
                      double *const v, int const *const work);

template void
dequantize_interleave(const TensorMeshHierarchy<2, double> &hierarchy,
                      double *const v, int const *const work);

template void
dequantize_interleave(const TensorMeshHierarchy<3, double> &hierarchy,
                      double *const v, int const *const work);

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

template void decompose(const TensorMeshHierarchy<1, float> &hierarchy,
                        float *const v);

template void decompose(const TensorMeshHierarchy<2, float> &hierarchy,
                        float *const v);

template void decompose(const TensorMeshHierarchy<3, float> &hierarchy,
                        float *const v);

template void decompose(const TensorMeshHierarchy<4, float> &hierarchy,
                        float *const v);

template void decompose(const TensorMeshHierarchy<1, double> &hierarchy,
                        double *const v);

template void decompose(const TensorMeshHierarchy<2, double> &hierarchy,
                        double *const v);

template void decompose(const TensorMeshHierarchy<3, double> &hierarchy,
                        double *const v);

template void decompose(const TensorMeshHierarchy<4, double> &hierarchy,
                        double *const v);

} // namespace mgard
