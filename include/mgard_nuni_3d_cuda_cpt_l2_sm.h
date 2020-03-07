namespace mgard_common {

}


namespace mgard_cannon {

}

namespace mgard_gen {
void 
refactor_3D_cuda(int l_target,
                 int nrow, int ncol, int nfib,  
                 int nr, int nc, int nf, 
                 int * dirow, int * dicol, int * difib,
                 double * dv, int lddv1, int lddv2,
                 double * dcoords_x, double * dcoords_y, double * dcoords_z);
}