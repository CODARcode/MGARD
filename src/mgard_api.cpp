#include "mgard_api.h"

unsigned char *mgard_compress(
    int itype_flag,
    double *data,
    int &out_size,
    int n1,
    int n2,
    int n3,
    double tol
);

unsigned char *mgard_compress(
    int itype_flag,
    double *data,
    int &out_size,
    int n1,
    int n2,
    int n3,
    std::vector<double> &coords_x,
    std::vector<double> &coords_y,
    std::vector<double> &coords_z,
    double tol
);

unsigned char *mgard_compress(
    int itype_flag,
    double *data,
    int &out_size,
    int n1,
    int n2,
    int n3,
    double tol,
    double s
);

unsigned char *mgard_compress(
    int itype_flag,
    double *data,
    int &out_size,
    int n1,
    int n2,
    int n3,
    std::vector<double> &coords_x,
    std::vector<double> &coords_y,
    std::vector<double> &coords_z,
    double tol,
    double s
);

unsigned char *mgard_compress(
    int itype_flag,
    double *data,
    int &out_size,
    int n1,
    int n2,
    int n3,
    double tol,
    double (*qoi)(int, int, int, double *),
    double s
);

unsigned char *mgard_compress(
    int itype_flag,
    double *data,
    int &out_size,
    int n1,
    int n2,
    int n3,
    std::vector<double> &coords_x,
    std::vector<double> &coords_y,
    std::vector<double> &coords_z,
    double tol,
    double (*qoi)(int, int, int, double *),
    double s
);

double mgard_compress(
    int n1,
    int n2,
    int n3,
    double (*qoi)(int, int, int, std::vector<double>),
    double s
);

double mgard_compress(
    int n1,
    int n2,
    int n3,
    std::vector<double> &coords_x,
    std::vector<double> &coords_y,
    std::vector<double> &coords_z,
    double (*qoi)(int, int, int, std::vector<double>),
    double s
);

double mgard_compress(
    int n1,
    int n2,
    int n3,
    double (*qoi)(int, int, int, double *),
    double s
);

double mgard_compress(
    int n1,
    int n2,
    int n3,
    std::vector<double> &coords_x,
    std::vector<double> &coords_y,
    std::vector<double> &coords_z,
    double (*qoi)(int, int, int, double *),
    double s
);

unsigned char *mgard_compress(
    int itype_flag,
    double *data,
    int &out_size,
    int n1,
    int n2,
    int n3,
    double tol,
    double norm_of_qoi,
    double s
);

unsigned char *mgard_compress(
    int itype_flag,
    double *data,
    int &out_size,
    int n1,
    int n2,
    int n3,
    std::vector<double> &coords_x,
    std::vector<double> &coords_y,
    std::vector<double> &coords_z,
    double tol,
    double norm_of_qoi,
    double s
);

double *mgard_decompress(
    int itype_flag,
    double &quantizer,
    unsigned char *data,
    int data_len,
    int n1,
    int n2,
    int n3
);

double *mgard_decompress(
    int itype_flag,
    double &quantizer,
    unsigned char *data,
    int data_len,
    int n1,
    int n2,
    int n3,
    std::vector<double> &coords_x,
    std::vector<double> &coords_y,
    std::vector<double> &coords_z
);

double *mgard_decompress(
    int itype_flag,
    double &quantizer,
    unsigned char *data,
    int data_len,
    int n1,
    int n2,
    int n3,
    double s
);

double *mgard_decompress(
    int itype_flag,
    double &quantizer,
    unsigned char *data,
    int data_len,
    int n1,
    int n2,
    int n3,
    std::vector<double> &coords_x,
    std::vector<double> &coords_y,
    std::vector<double> &coords_z,
    double s
);
