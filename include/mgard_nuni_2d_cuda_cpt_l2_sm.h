#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"


namespace mgard_2d {
namespace mgard_gen {

template <typename T>
mgard_cuda_ret
refactor_2D_cuda_compact_l2_sm(const int l_target,
                    const int nrow,     const int ncol,
                    const int nr,       const int nc, 
                    int * dirow,        int * dicol,
                    int * dirowP,       int * dicolP,
                    T * dv,        int lddv, 
                    T * dwork,     int lddwork,
                    T * dcoords_x, T * dcoords_y,
                    int B,
                    mgard_cuda_handle & handle, bool profile);

template <typename T>
mgard_cuda_ret 
prep_2D_cuda_l2_sm(const int nrow,     const int ncol,
                   const int nr,       const int nc, 
                   int * dirow,        int * dicol,
                   int * dirowP,       int * dicolP,
                   T * dv,        int lddv, 
                   T * dwork,     int lddwork,
                   T * dcoords_x, T * dcoords_y,
                   int B,
                   mgard_cuda_handle & handle, bool profile);

template <typename T> mgard_cuda_ret 
recompose_2D_cuda_l2_sm(const int l_target,
                  const int nrow,     const int ncol,
                  const int nr,       const int nc, 
                  int * dirow,        int * dicol,
                  int * dirowP,       int * dicolP,
                  T * dv,        int lddv, 
                  T * dwork,     int lddwork,
                  T * dcoords_x, T * dcoords_y,
                  int B,
                  mgard_cuda_handle & handle, bool profile);

template <typename T>
mgard_cuda_ret
postp_2D_cuda_l2_sm(const int nrow,     const int ncol,
                    const int nr,       const int nc, 
                    int * dirow,        int * dicol,
                    int * dirowP,       int * dicolP,
                    T * dv,        int lddv, 
                    T * dwork,     int lddwork,
                    T * dcoords_x, T * dcoords_y,
                    int B,
                    mgard_cuda_handle & handle, bool profile);

}
}