#include "mgard_api.h"

unsigned char *mgard_compress(int itype_flag, float *data, int &out_size,
                              int n1, int n2, int n3, float tol);

unsigned char *mgard_compress(int itype_flag, float *data, int &out_size,
                              int n1, int n2, int n3,
                              std::vector<float> &coords_x,
                              std::vector<float> &coords_y,
                              std::vector<float> &coords_z, float tol);

unsigned char *mgard_compress(int itype_flag, float *data, int &out_size,
                              int n1, int n2, int n3, float tol, float s);

unsigned char *mgard_compress(int itype_flag, float *data, int &out_size,
                              int n1, int n2, int n3,
                              std::vector<float> &coords_x,
                              std::vector<float> &coords_y,
                              std::vector<float> &coords_z, float tol, float s);

unsigned char *mgard_compress(int itype_flag, float *data, int &out_size,
                              int n1, int n2, int n3, float tol,
                              float (*qoi)(int, int, int, float *), float s);

unsigned char *mgard_compress(int itype_flag, float *data, int &out_size,
                              int n1, int n2, int n3,
                              std::vector<float> &coords_x,
                              std::vector<float> &coords_y,
                              std::vector<float> &coords_z, float tol,
                              float (*qoi)(int, int, int, float *), float s);

float mgard_compress(int n1, int n2, int n3,
                     float (*qoi)(int, int, int, std::vector<float>), float s);

float mgard_compress(int n1, int n2, int n3, std::vector<float> &coords_x,
                     std::vector<float> &coords_y, std::vector<float> &coords_z,
                     float (*qoi)(int, int, int, std::vector<float>), float s);

float mgard_compress(int n1, int n2, int n3,
                     float (*qoi)(int, int, int, float *), float s);

float mgard_compress(int n1, int n2, int n3, std::vector<float> &coords_x,
                     std::vector<float> &coords_y, std::vector<float> &coords_z,
                     float (*qoi)(int, int, int, float *), float s);

unsigned char *mgard_compress(int itype_flag, float *data, int &out_size,
                              int n1, int n2, int n3, float tol,
                              float norm_of_qoi, float s);

unsigned char *mgard_compress(int itype_flag, float *data, int &out_size,
                              int n1, int n2, int n3,
                              std::vector<float> &coords_x,
                              std::vector<float> &coords_y,
                              std::vector<float> &coords_z, float tol,
                              float norm_of_qoi, float s);

float *mgard_decompress(int itype_flag, float &quantizer, unsigned char *data,
                        int data_len, int n1, int n2, int n3);

float *mgard_decompress(int itype_flag, float &quantizer, unsigned char *data,
                        int data_len, int n1, int n2, int n3,
                        std::vector<float> &coords_x,
                        std::vector<float> &coords_y,
                        std::vector<float> &coords_z);

float *mgard_decompress(int itype_flag, float &quantizer, unsigned char *data,
                        int data_len, int n1, int n2, int n3, float s);

float *mgard_decompress(int itype_flag, float &quantizer, unsigned char *data,
                        int data_len, int n1, int n2, int n3,
                        std::vector<float> &coords_x,
                        std::vector<float> &coords_y,
                        std::vector<float> &coords_z, float s);
