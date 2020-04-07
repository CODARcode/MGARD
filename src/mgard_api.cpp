#include "mgard_api.h"
#include "mgard_api.tpp"

template unsigned char *mgard_compress<float>(int itype_flag, float *data,
                                              int &out_size, int n1, int n2,
                                              int n3,
                                              float tol); // ...  1

template unsigned char *mgard_compress<double>(int itype_flag, double *data,
                                               int &out_size, int n1, int n2,
                                               int n3,
                                               double tol); // ...  1

template unsigned char *mgard_compress<float>(int itype_flag, float *data,
                                              int &out_size, int n1, int n2,
                                              int n3,
                                              std::vector<float> &coords_x,
                                              std::vector<float> &coords_y,
                                              std::vector<float> &coords_z,
                                              float tol); // ... 1a

template unsigned char *mgard_compress<double>(int itype_flag, double *data,
                                               int &out_size, int n1, int n2,
                                               int n3,
                                               std::vector<double> &coords_x,
                                               std::vector<double> &coords_y,
                                               std::vector<double> &coords_z,
                                               double tol); // ... 1a

template unsigned char *mgard_compress<float>(int itype_flag, float *data,
                                              int &out_size, int n1, int n2,
                                              int n3, float tol,
                                              float s); // ... 2

template unsigned char *mgard_compress<double>(int itype_flag, double *data,
                                               int &out_size, int n1, int n2,
                                               int n3, double tol,
                                               double s); // ... 2

template unsigned char *
mgard_compress<float>(int itype_flag, float *data, int &out_size, int n1,
                      int n2, int n3, float tol,
                      float (*qoi)(int, int, int, float *),
                      float s); // ... 3

template unsigned char *
mgard_compress<double>(int itype_flag, double *data, int &out_size, int n1,
                       int n2, int n3, double tol,
                       double (*qoi)(int, int, int, double *),
                       double s); // ... 3

template float mgard_compress<float>(int n1, int n2, int n3,
                                     float (*qoi)(int, int, int,
                                                  std::vector<float>),
                                     float s); // ... 4

template double mgard_compress<double>(int n1, int n2, int n3,
                                       double (*qoi)(int, int, int,
                                                     std::vector<double>),
                                       double s); // ... 4

template float mgard_compress<float>(int n1, int n2, int n3,
                                     float (*qoi)(int, int, int, float *),
                                     float s); // ... 5

template double mgard_compress<double>(int n1, int n2, int n3,
                                       double (*qoi)(int, int, int, double *),
                                       double s); // ... 5

template unsigned char *mgard_compress<float>(int itype_flag, float *data,
                                              int &out_size, int n1, int n2,
                                              int n3, float tol,
                                              float norm_of_qoi,
                                              float s); // ... 6

template unsigned char *mgard_compress<double>(int itype_flag, double *data,
                                               int &out_size, int n1, int n2,
                                               int n3, double tol,
                                               double norm_of_qoi,
                                               double s); // ... 6

template float *
mgard_decompress<float>(int itype_flag, unsigned char *data, int data_len,
                        int n1, int n2,
                        int n3); // decompress L-infty compressed data

template double *
mgard_decompress<double>(int itype_flag, unsigned char *data, int data_len,
                         int n1, int n2,
                         int n3); // decompress L-infty compressed data

template float *mgard_decompress<float>(int itype_flag, unsigned char *data,
                                        int data_len, int n1, int n2, int n3,
                                        float s); // decompress s-norm

template double *mgard_decompress<double>(int itype_flag, unsigned char *data,
                                          int data_len, int n1, int n2, int n3,
                                          double s); // decompress s-norm
