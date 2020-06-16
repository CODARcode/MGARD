
namespace mgard_common {

}


namespace mgard_cannon {

}

namespace mgard_gen {
   
template <typename T>
void prep_3D_cuda(const int nr, const int nc, const int nf, const int nrow,
             const int ncol, const int nfib, const int l_target, double *v,
             std::vector<double> &work, std::vector<double> &work2d,
             std::vector<double> &coords_x, std::vector<double> &coords_y,
             std::vector<double> &coords_z, 
             int * dirow, int * dicol, int * difib,
             int * dirowP, int * dicolP, int * difibP,
             int * dirowA, int * dicolA, int * difibA,
             T * ddist_r, T * ddist_c, T * ddist_f,
             T * dcoords_r, T * dcoords_c, T * dcoords_f,  
             int B, mgard_cuda_handle & handle, bool profile );

template <typename T>
mgard_cuda_ret 
refactor_3D_cuda_cpt_l2_sm(int l_target,
                           int nrow,           int ncol,           int nfib,
                           int nr,             int nc,             int nf,             
                           int * dirow,        int * dicol,        int * difib,
                           T * dv,        int lddv1,          int lddv2,
                           T * dwork,     int lddwork1,       int lddwork2,
                           T * dcoords_r, T * dcoords_c, T * dcoords_f,
                           int B, mgard_cuda_handle & handle, bool profile);

template <typename T>
mgard_cuda_ret
recompose_3D_cuda_cpt_l2_sm(const int l_target,
                              const int nrow, const int ncol, const int nfib, 
                              const int nr, const int nc, const int nf,
                              int * dirow,        int * dicol,        int * difib,
                              T * dv,        int lddv1,          int lddv2,
                              T * dwork,     int lddwork1,       int lddwork2,
                              T * dcoords_r, T * dcoords_c, T * dcoords_f,
                              int B, mgard_cuda_handle & handle, bool profile);
                              // double *v, 
                              // std::vector<double> &work, std::vector<double> &work2d,
                              // std::vector<double> &coords_x, std::vector<double> &coords_y,
                              // std::vector<double> &coords_z);
   }