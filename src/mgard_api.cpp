#include "mgard_api.h"
#include "mgard_api.tpp"

template unsigned char *mgard_compress<float>(float *data, int &out_size,
                                              int n1, int n2, int n3,
                                              float tol);

template unsigned char *mgard_compress<double>(double *data, int &out_size,
                                               int n1, int n2, int n3,
                                               double tol);

template unsigned char *mgard_compress<float>(float *data, int &out_size,
                                              int n1, int n2, int n3,
                                              std::vector<float> &coords_x,
                                              std::vector<float> &coords_y,
                                              std::vector<float> &coords_z,
                                              float tol);

template unsigned char *mgard_compress<double>(double *data, int &out_size,
                                               int n1, int n2, int n3,
                                               std::vector<double> &coords_x,
                                               std::vector<double> &coords_y,
                                               std::vector<double> &coords_z,
                                               double tol);

template unsigned char *mgard_compress<float>(float *data, int &out_size,
                                              int n1, int n2, int n3, float tol,
                                              float s);

template unsigned char *mgard_compress<double>(double *data, int &out_size,
                                               int n1, int n2, int n3,
                                               double tol, double s);

template unsigned char *
mgard_compress<float>(float *data, int &out_size, int n1, int n2, int n3,
                      float tol, float (*qoi)(int, int, int, float *), float s);

template unsigned char *
mgard_compress<double>(double *data, int &out_size, int n1, int n2, int n3,
                       double tol, double (*qoi)(int, int, int, double *),
                       double s);

template unsigned char *mgard_compress<float>(float *data, int &out_size,
                                              int n1, int n2, int n3, float tol,
                                              float norm_of_qoi, float s);

template unsigned char *mgard_compress<double>(double *data, int &out_size,
                                               int n1, int n2, int n3,
                                               double tol, double norm_of_qoi,
                                               double s);

template float *mgard_decompress<float>(unsigned char *data, int data_len,
                                        int n1, int n2, int n3);

template double *mgard_decompress<double>(unsigned char *data, int data_len,
                                          int n1, int n2, int n3);

template float *mgard_decompress<float>(unsigned char *data, int data_len,
                                        int n1, int n2, int n3, float s);

template double *mgard_decompress<double>(unsigned char *data, int data_len,
                                          int n1, int n2, int n3, double s);

namespace mgard {

template class CompressedDataset<1, float>;
template class CompressedDataset<2, float>;
template class CompressedDataset<3, float>;

template class CompressedDataset<1, double>;
template class CompressedDataset<2, double>;
template class CompressedDataset<3, double>;

template class DecompressedDataset<1, float>;
template class DecompressedDataset<2, float>;
template class DecompressedDataset<3, float>;

template class DecompressedDataset<1, double>;
template class DecompressedDataset<2, double>;
template class DecompressedDataset<3, double>;

template CompressedDataset<1, float>
compress(const TensorMeshHierarchy<1, float> &, float *const, const float,
         const float);
template CompressedDataset<2, float>
compress(const TensorMeshHierarchy<2, float> &, float *const, const float,
         const float);
template CompressedDataset<3, float>
compress(const TensorMeshHierarchy<3, float> &, float *const, const float,
         const float);

template CompressedDataset<1, double>
compress(const TensorMeshHierarchy<1, double> &, double *const, const double,
         const double);
template CompressedDataset<2, double>
compress(const TensorMeshHierarchy<2, double> &, double *const, const double,
         const double);
template CompressedDataset<3, double>
compress(const TensorMeshHierarchy<3, double> &, double *const, const double,
         const double);

template DecompressedDataset<1, float>
decompress(const CompressedDataset<1, float> &);
template DecompressedDataset<2, float>
decompress(const CompressedDataset<2, float> &);
template DecompressedDataset<3, float>
decompress(const CompressedDataset<3, float> &);

template DecompressedDataset<1, double>
decompress(const CompressedDataset<1, double> &);
template DecompressedDataset<2, double>
decompress(const CompressedDataset<2, double> &);
template DecompressedDataset<3, double>
decompress(const CompressedDataset<3, double> &);

} // namespace mgard
