#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"


namespace mgard_2d {
namespace mgard_gen {

mgard_cuda_ret
refactor_2D_cuda_compact_l2_sm(const int l_target,
                    const int nrow,     const int ncol,
                    const int nr,       const int nc, 
                    int * dirow,        int * dicol,
                    int * dirowP,       int * dicolP,
                    double * dv,        int lddv, 
                    double * dwork,     int lddwork,
                    double * dcoords_x, double * dcoords_y);
mgard_cuda_ret 
prep_2D_cuda_l2_sm(const int nrow,     const int ncol,
                   const int nr,       const int nc, 
                   int * dirow,        int * dicol,
                   int * dirowP,       int * dicolP,
                   double * dv,        int lddv, 
                   double * dwork,     int lddwork,
                   double * dcoords_x, double * dcoords_y);

void 
recompose_2D_cuda_l2_sm(const int l_target,
                  const int nrow,     const int ncol,
                  const int nr,       const int nc, 
                  int * dirow,        int * dicol,
                  int * dirowP,       int * dicolP,
                  double * dv,        int lddv, 
                  double * dwork,     int lddwork,
                  double * dcoords_x, double * dcoords_y);

void 
postp_2D_cuda_l2_sm(const int nrow,     const int ncol,
                    const int nr,       const int nc, 
                    int * dirow,        int * dicol,
                    int * dirowP,       int * dicolP,
                    double * dv,        int lddv, 
                    double * dwork,     int lddwork,
                    double * dcoords_x, double * dcoords_y);

}
}