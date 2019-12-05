#include "mgard.h"

namespace mgard {

template <>
unsigned char *refactor_qz<float>(
    int nrow, int ncol, int nfib, const float *u, int &outsize, float tol
);

template <>
unsigned char *refactor_qz<float>(
    int nrow,
    int ncol,
    int nfib,
    std::vector<float> &coords_x,
    std::vector<float> &coords_y,
    std::vector<float> &coords_z,
    const float *u,
    int &outsize,
    float tol
);

template <>
unsigned char *refactor_qz<float>(
    int nrow,
    int ncol,
    int nfib,
    const float *u,
    int &outsize,
    float tol,
    float s
);

template <>
unsigned char *refactor_qz<float>(
    int nrow,
    int ncol,
    int nfib,
    std::vector<float> &coords_x,
    std::vector<float> &coords_y,
    std::vector<float> &coords_z,
    const float *u,
    int &outsize,
    float tol,
    float s
);

template <>
unsigned char *refactor_qz<float>(
    int nrow,
    int ncol,
    int nfib,
    const float *u,
    int &outsize,
    float tol,
    float (*qoi)(int, int, int, std::vector<float>),
    float s
);

template <>
float *recompose_udq<float>(
    int nrow, int ncol, int nfib, unsigned char *data, int data_len
);

template <>
float *recompose_udq<float>(
    int nrow,
    int ncol,
    int nfib,
    std::vector<float> &coords_x,
    std::vector<float> &coords_y,
    std::vector<float> &coords_z,
    unsigned char *data,
    int data_len
);

template <>
float *recompose_udq<float>(
    int nrow, int ncol, int nfib, unsigned char *data, int data_len, float s
);

template <>
float *recompose_udq<float>(
    int nrow,
    int ncol,
    int nfib,
    std::vector<float> &coords_x,
    std::vector<float> &coords_y,
    std::vector<float> &coords_z,
    unsigned char *data,
    int data_len,
    float s
);

template <>
unsigned char *refactor_qz_2D<float>(
    int nrow, int ncol, const float *u, int &outsize, float tol
);

template <>
unsigned char *refactor_qz_2D<float>(
    int nrow,
    int ncol,
    std::vector<float> &coords_x,
    std::vector<float> &coords_y,
    const float *u,
    int &outsize,
    float tol
);

template <>
unsigned char *refactor_qz_2D<float>(
    int nrow, int ncol, const float *u, int &outsize, float tol, float s
);

template <>
unsigned char *refactor_qz_2D<float>(
    int nrow,
    int ncol,
    std::vector<float> &coords_x,
    std::vector<float> &coords_y,
    const float *u,
    int &outsize,
    float tol,
    float s
);

template <>
float *recompose_udq_2D<float>(
    int nrow, int ncol, unsigned char *data, int data_len
);

template <>
float *recompose_udq_2D<float>(
    int nrow,
    int ncol,
    std::vector<float> &coords_x,
    std::vector<float> &coords_y,
    unsigned char *data,
    int data_len
);

template <>
float *recompose_udq_2D<float>(
    int nrow, int ncol, unsigned char *data, int data_len, float s
);

template <>
float *recompose_udq_2D<float>(
    int nrow,
    int ncol,
    std::vector<float> &coords_x,
    std::vector<float> &coords_y,
    unsigned char *data,
    int data_len,
    float s
);

template <>
int parse_cmdl<float>(
    int argc,
    char **argv,
    int &nrow,
    int &ncol,
    float &tol,
    std::string &in_file
);

template <>
void mass_matrix_multiply<float>(const int l, std::vector<float> &v);

template <>
void solve_tridiag_M<float>(const int l, std::vector<float> &v);

template <>
void restriction<float>(const int l, std::vector<float> &v);

template <>
void interpolate_from_level_nMl<float>(const int l, std::vector<float> &v);

template <>
void print_level_2D<float>(
    const int nrow, const int ncol, const int l, float *v
);

template <>
void write_level_2D<float>(
    const int nrow,
    const int ncol,
    const int l,
    float *v,
    std::ofstream &outfile
);

template <>
void write_level_2D_exc<float>(
    const int nrow,
    const int ncol,
    const int l,
    float *v,
    std::ofstream &outfile
);

template <>
void pi_lminus1<float>(const int l, std::vector<float> &v0);

template <>
void pi_Ql<float>(
    const int nrow,
    const int ncol,
    const int l,
    float *v,
    std::vector<float> &row_vec,
    std::vector<float> &col_vec
);

template <>
void assign_num_level<float>(
    const int nrow, const int ncol, const int l, float *v, float num
);

template <>
void copy_level<float>(
    const int nrow,
    const int ncol,
    const int l,
    float *v,
    std::vector<float> &work
);

template <>
void add_level<float>(
    const int nrow, const int ncol, const int l, float *v, float *work
);

template <>
void subtract_level<float>(
    const int nrow, const int ncol, const int l, float *v, float *work
);

template <>
void compute_correction_loadv<float>(const int l, std::vector<float> &v);

template <>
void qwrite_level_2D<float>(
    const int nrow,
    const int ncol,
    const int nlevel,
    const int l,
    float *v,
    float tol,
    const std::string outfile
);

template <>
void quantize_2D_interleave<float>(
    const int nrow,
    const int ncol,
    float *v,
    std::vector<int> &work,
    float norm,
    float tol
);

template <>
void dequantize_2D_interleave<float>(
    const int nrow, const int ncol, float *v, const std::vector<int> &work
);

template <>
void qwrite_2D_interleave<float>(
    const int nrow,
    const int ncol,
    const int nlevel,
    const int l,
    float *v,
    float tol,
    const std::string outfile
);

template <>
void qread_level_2D<float>(
    const int nrow,
    const int ncol,
    const int nlevel,
    float *v,
    std::string infile
);

template <>
void refactor<float>(
    const int nrow,
    const int ncol,
    const int l_target,
    float *v,
    std::vector<float> &work,
    std::vector<float> &row_vec,
    std::vector<float> &col_vec
);

template <>
void recompose<float>(
    const int nrow,
    const int ncol,
    const int l_target,
    float *v,
    std::vector<float> &work,
    std::vector<float> &row_vec,
    std::vector<float> &col_vec
);

template <>
inline float interp_2d<float>(
    float q11,
    float q12,
    float q21,
    float q22,
    float x1,
    float x2,
    float y1,
    float y2,
    float x,
    float y
);

template <>
inline float interp_0d<float>(
    const float x1,
    const float x2,
    const float y1,
    const float y2,
    const float x
);

template <>
void resample_1d<float>(
    const float *inbuf, float *outbuf, const int ncol, const int ncol_new
);

template <>
void resample_1d_inv2<float>(
    const float *inbuf, float *outbuf, const int ncol, const int ncol_new
);

template <>
void resample_2d<float>(
    const float *inbuf,
    float *outbuf,
    const int nrow,
    const int ncol,
    const int nrow_new,
    const int ncol_new
);

template <>
void resample_2d_inv2<float>(
    const float *inbuf,
    float *outbuf,
    const int nrow,
    const int ncol,
    const int nrow_new,
    const int ncol_new
);

} // end namespace mgard
