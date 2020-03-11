
namespace mgard_common {

}


namespace mgard_cannon {

}

namespace mgard_gen {
mgard_cuda_ret 
refactor_3D_cuda_cpt_l2_sm(int l_target,
                           int nfib,           int nrow,           int ncol,  
                           int nf,             int nr,             int nc, 
                           int * difib,        int * dirow,        int * dicol,
                           double * dv,        int lddv1,          int lddv2,
                           double * dwork,     int lddwork1,       int lddwork2,
                           double * dcoords_z, double * dcoords_y, double * dcoords_x);
}