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

template float
mgard_compress<float>(int n1, int n2, int n3,
                      float (*qoi)(int, int, int, std::vector<float>), float s);

template double mgard_compress<double>(int n1, int n2, int n3,
                                       double (*qoi)(int, int, int,
                                                     std::vector<double>),
                                       double s);

template float mgard_compress<float>(int n1, int n2, int n3,
                                     float (*qoi)(int, int, int, float *),
                                     float s);

template double mgard_compress<double>(int n1, int n2, int n3,
                                       double (*qoi)(int, int, int, double *),
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
