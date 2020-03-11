#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_cuda_helper.h"
//#include "mgard_cuda_helper_internal.h"

namespace mgard_2d {
namespace mgard_gen {

void 
prep_2D_cuda(const int nrow,     const int ncol,
             const int nr,       const int nc, 
             int * dirow,        int * dicol,
             int * dirowP,       int * dicolP,
             double * dv,        int lddv, 
             double * dwork,     int lddwork,
             double * dcoords_x, double * dcoords_y);

void 
refactor_2D_cuda(const int l_target,
                 const int nrow,     const int ncol,
                 const int nr,       const int nc, 
                 int * dirow,        int * dicol,
                 int * dirowP,       int * dicolP,
                 double * dv,        int lddv, 
                 double * dwork,     int lddwork,
                 double * dcoords_x, double * dcoords_y);

void 
recompose_2D_cuda(const int l_target,
                  const int nrow,     const int ncol,
                  const int nr,       const int nc, 
                  int * dirow,        int * dicol,
                  int * dirowP,       int * dicolP,
                  double * dv,        int lddv, 
                  double * dwork,     int lddwork,
                  double * dcoords_x, double * dcoords_y);


void 
postp_2D_cuda(const int nrow,     const int ncol,
              const int nr,       const int nc, 
              int * dirow,        int * dicol,
              int * dirowP,       int * dicolP,
              double * dv,        int lddv, 
              double * dwork,     int lddwork,
              double * dcoords_x, double * dcoords_y);
}

}

