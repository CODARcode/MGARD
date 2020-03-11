#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda_gen.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_nuni_2d_cuda_mass_mult_l.h"
#include "mgard_cuda_helper.h"
#include "mgard_cuda_compact_helper.h"
#include "mgard_cuda_helper_internal.h"
#include <fstream>
#include <vector>
#include <chrono>

void benchmark_mass_mult_l_row_org(int nrow, int ncol, int l, int l_target) {

  int stride = std::pow(2, l);

  double time_cpu;
  double time_first_cuda;
  double time_l1_compacted_cuda;
  double time_l2_compacted_cuda;
  double time_l2_compacted_sm_cuda;
  double time_l2_compacted_sm_tuned_cuda;
  double time_l2_compacted_sm_tuned_pf_cuda;

  double * v = new double[nrow * ncol];
  int ldv = ncol;

  int nlevel_x = std::log2(ncol-1);
  int nc = std::pow(2, nlevel_x ) + 1; //ncol new

  int nlevel_y = std::log2(nrow-1);
  int nr = std::pow(2, nlevel_y ) + 1; //nrow new

  int * irow  = new int[nr];
  int * icol  = new int[nc];

  double * coords_x = new double[ncol];

  int irow_ptr = 0;
  for (int i = 0; i < nr; i++) {
    int irow_r = mgard_2d::mgard_gen::get_lindex_cuda(nr, nrow, i);
    irow[irow_ptr] = irow_r;
    irow_ptr++;
  }

  int icol_ptr = 0;
  for (int i = 0; i < nc; i++) {
    int icol_r = mgard_2d::mgard_gen::get_lindex_cuda(nc, ncol, i);
    icol[icol_ptr] = icol_r;
    icol_ptr++;
  }

  double * dv;
  size_t dv_pitch;
  cudaMallocPitchHelper((void**)&dv, &dv_pitch, ncol * sizeof(double), nrow);
  int lddv = dv_pitch / sizeof(double);

  int * dirow;
  cudaMallocHelper((void**)&dirow, nr * sizeof(int));
  cudaMemcpyHelper(dirow, irow, nr * sizeof(int), H2D);

  int * dicol;
  cudaMallocHelper((void**)&dicol, nc * sizeof(int));
  cudaMemcpyHelper(dicol, icol, nc * sizeof(int), H2D);

  double * dcoords_x;
  cudaMallocHelper((void**)&dcoords_x, ncol * sizeof(double));
  cudaMemcpyHelper(dcoords_x, coords_x, ncol * sizeof(double), H2D);

  int col_stride = stride;
  int row_stride = 1;

  size_t total_data = 2*(nr / row_stride)*(nc / col_stride)*sizeof(double) + (nc/col_stride) * sizeof(double);

  mgard_cuda_ret ret;
  int repeat_test = 10;
  double time = 0.0;

  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  std::chrono::duration<double> elapsed;

  std::vector<double> row_vec(ncol);
  std::vector<double> coords_x_vec(ncol);
  for (int i = 0; i < ncol; i++) {
    coords_x_vec[i] = coords_x[i];
  }

  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat_test; i++) {

    for (int i = 0; i < nr; ++i) {
      for (int j = 0; j < ncol; ++j) {
        row_vec[j] = v[irow[i] * ldv + j];
      }
      mgard_gen::mass_mult_l(l, row_vec, coords_x_vec, nc, ncol);
      for (int j = 0; j < ncol; ++j) {
        v[irow[i] * ldv + j] = row_vec[j];
      }
    }
  }
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  time = elapsed.count();

  printf("[Original CPU]: %f s ( %f GB/s)\n", 
          time/repeat_test, ((double)total_data/1e9)/(time/repeat_test));
  time_cpu = time;



  time = 0;
  for (int i = 0; i < repeat_test; i++) {
    cudaMemcpy2DHelper(dv, lddv * sizeof(double), 
                     v, ldv  * sizeof(double), 
                     nc * sizeof(double), nr, 
                     H2D);
    ret = mgard_2d::mgard_gen::mass_mult_l_row_cuda(nrow,    ncol,
                                                     nr,    nc,
                                                     row_stride, col_stride,
                                                     dirow, dicol,
                                                     dv,     lddv,
                                                     dcoords_x);
    time += ret.time;
  }




  printf("[First CUDA]: %f s ( %f GB/s)\n", 
          time/repeat_test, ((double)total_data/1e9)/(time/repeat_test));
  time_first_cuda = time;


  double * dcv;
  size_t dcv_pitch;
  cudaMallocPitchHelper((void**)&dcv, &dcv_pitch, nc * sizeof(double), nr);
  int lddcv = dcv_pitch / sizeof(double);

  int * cirow = new int[nr];
  int * cicol = new int[nc];

  for (int i = 0; i < nr; i++) {
    cirow[i] = i;
  }

  for (int i = 0; i < nc; i++) {
    cicol[i] = i;
  }

  int * dcirow;
  cudaMallocHelper((void**)&dcirow, nr * sizeof(int));
  cudaMemcpyHelper(dcirow, cirow, nr * sizeof(int), H2D);

  int * dcicol;
  cudaMallocHelper((void**)&dcicol, nc * sizeof(int));
  cudaMemcpyHelper(dcicol, cicol, nc * sizeof(int), H2D);

  double * ccoords_x = new double[nc];
  for (int i = 0; i < nc; i++) {
    ccoords_x[i] = coords_x[icol[i]];
  }

  double * dccoords_x;
  cudaMallocHelper((void**)&dccoords_x, nc * sizeof(double));
  cudaMemcpyHelper(dccoords_x, ccoords_x, nc * sizeof(double), H2D);

  time = 0;
  for (int i = 0; i < repeat_test; i++) {
    org_to_pow2p1(nrow,  ncol,
                  nr,    nc,
                  dirow, dicol,
                  dv,    lddv,
                  dcv,   lddcv);
    ret = mgard_2d::mgard_gen::mass_mult_l_row_cuda(nrow,    ncol,
                                                     nr,    nc,
                                                     row_stride, col_stride,
                                                     dcirow, dcicol,
                                                     dcv,     lddcv,
                                                     dccoords_x);
    time += ret.time;
  }

  printf("[L1-Compacted CUDA]: %f s ( %f GB/s)\n", 
          time/repeat_test, ((double)total_data/1e9)/(time/repeat_test));
  time_l1_compacted_cuda = time;


  int * nr_l = new int[l_target+1];
  int * nc_l = new int[l_target+1];

  int ** cirow_l = new int*[l_target+1];
  int ** cicol_l = new int*[l_target+1];

  double ** ccoords_x_l = new double*[l_target+1];

  int ** dcirow_l = new int*[l_target+1];
  int ** dcicol_l = new int*[l_target+1];

  double ** dccoords_x_l = new double*[l_target+1];

  for (int l = 0; l < l_target+1; l++) {
    int stride = std::pow(2, l);
    nr_l[l] = ceil((float)nr/std::pow(2, l));
    nc_l[l] = ceil((float)nc/std::pow(2, l));
    cirow_l[l] = new int[nr_l[l]];
    cicol_l[l] = new int[nc_l[l]];

    ccoords_x_l[l] = new double[nc_l[l]];

    for (int i = 0; i < nr_l[l]; i++) {
      cirow_l[l][i] = i;
    }

    for (int i = 0; i < nc_l[l]; i++) {
      cicol_l[l][i] = i;
      ccoords_x_l[l][i] = ccoords_x[i * stride];
    }

    cudaMallocHelper((void**)&(dcirow_l[l]), nr_l[l] * sizeof(int));
    cudaMemcpyHelper(dcirow_l[l], cirow_l[l], nr_l[l] * sizeof(int), H2D);

    cudaMallocHelper((void**)&(dcicol_l[l]), nc_l[l] * sizeof(int));
    cudaMemcpyHelper(dcicol_l[l], cicol_l[l], nc_l[l] * sizeof(int), H2D);

    cudaMallocHelper((void**)&(dccoords_x_l[l]), nc_l[l] * sizeof(double));
    cudaMemcpyHelper(dccoords_x_l[l], ccoords_x_l[l], nc_l[l] * sizeof(double), H2D);
  }

  double * dcv2;
  size_t dcv2_pitch;
  cudaMallocPitchHelper((void**)&dcv2, &dcv2_pitch, nc * sizeof(double), nr);
  int lddcv2 = dcv2_pitch / sizeof(double);

  time = 0;
  for (int i = 0; i < repeat_test; i++) {
    pow2p1_to_cpt(nr,    nc,
                  row_stride, col_stride,
                  dcv,      lddcv,
                  dcv2,     lddcv2);

    ret = mgard_2d::mgard_gen::mass_mult_l_row_cuda(nr_l[0],    nc_l[l],
                                                    nr_l[0],    nc_l[l],
                                                     1, 1,
                                                     dcirow_l[0], dcicol_l[l],
                                                     dcv2,     lddcv2,
                                                     dccoords_x_l[l]);
    time += ret.time;
  }

  printf("[L2-Compacted CUDA]: %f s ( %f GB/s)\n", 
          time/repeat_test, ((double)total_data/1e9)/(time/repeat_test));
  time_l2_compacted_cuda = time;

  time = 0;
  for (int i = 0; i < repeat_test; i++) {
    pow2p1_to_cpt(nr,    nc,
                  row_stride, col_stride,
                  dcv,      lddcv,
                  dcv2,     lddcv2);

    ret = mgard_2d::mgard_gen::mass_mult_l_row_cuda_sm(nr_l[0],    nc_l[l],
                                                     1, 1,
                                                     dcv2,     lddcv2,
                                                     dccoords_x_l[l], 4, 2);
    time += ret.time;
  }

  printf("[L2-Compacted-SM CUDA]: %f s ( %f GB/s)\n", 
          time/repeat_test, ((double)total_data/1e9)/(time/repeat_test));

  time_l2_compacted_sm_cuda = time;


  time_l2_compacted_sm_tuned_cuda = 100000;

  for (int ghost_col = 8; ghost_col <= 32; ghost_col *= 2) {
    for (int B = ghost_col; B <=32; B *=2) {

      time = 0;
      for (int i = 0; i < repeat_test; i++) {
        pow2p1_to_cpt(nr,    nc,
                      row_stride, col_stride,
                      dcv,      lddcv,
                      dcv2,     lddcv2);

        ret = mgard_2d::mgard_gen::mass_mult_l_row_cuda_sm(nr_l[0],    nc_l[l],
                                                         1, 1,
                                                         dcv2,     lddcv2,
                                                         dccoords_x_l[l],
                                                         B, ghost_col);
        time += ret.time;
      }

      printf("[L2-Compacted-SM-Tune (%d, %d) CUDA]: %f s ( %f GB/s)\n", 
              B, ghost_col,
              time/repeat_test, ((double)total_data/1e9)/(time/repeat_test));
      if (time_l2_compacted_sm_tuned_cuda > time) {
        time_l2_compacted_sm_tuned_cuda = time;
      }
    }
  }


  time_l2_compacted_sm_tuned_pf_cuda = 100000;

  for (int ghost_col = 8; ghost_col <= 32; ghost_col *= 2) {
    for (int B = ghost_col; B <=32; B *=2) {

      time = 0;
      for (int i = 0; i < repeat_test; i++) {
        pow2p1_to_cpt(nr,    nc,
                      row_stride, col_stride,
                      dcv,      lddcv,
                      dcv2,     lddcv2);

        ret = mgard_2d::mgard_gen::mass_mult_l_row_cuda_sm_pf(nr_l[0],    nc_l[l],
                                                        nr_l[0],    nc_l[l],
                                                         1, 1,
                                                         dcirow_l[0], dcicol_l[l],
                                                         dcv2,     lddcv2,
                                                         dccoords_x_l[l],
                                                         B, ghost_col);
        time += ret.time;
      }

      printf("[L2-Compacted-SM-Tune-PF (%d, %d) CUDA]: %f s ( %f GB/s)\n", 
              B, ghost_col,
              time/repeat_test, ((double)total_data/1e9)/(time/repeat_test));
      if (time_l2_compacted_sm_tuned_pf_cuda > time) {
        time_l2_compacted_sm_tuned_pf_cuda = time;
      }
    }
  }

  delete [] v;
  delete [] irow;
  delete [] icol;
  cudaFreeHelper(dv);
  cudaFreeHelper(dirow);
  cudaFreeHelper(dicol);
  cudaFreeHelper(dcoords_x);

  cudaFreeHelper(dcv);
  delete [] cirow;
  delete [] cicol;
  cudaFreeHelper(dcirow);
  cudaFreeHelper(dcicol);
  delete [] ccoords_x;
  cudaFreeHelper(dccoords_x);

  for (int l = 0; l < l_target+1; l++) {
    delete [] cirow_l[l];
    delete [] cicol_l[l];
    delete [] ccoords_x_l[l];

    cudaFreeHelper(dcirow_l[l]);
    cudaFreeHelper(dcicol_l[l]);
    cudaFreeHelper(dccoords_x_l[l]);
  }
  delete [] cirow_l;
  delete [] cicol_l;
  delete [] ccoords_x_l;
  delete [] dcirow_l;
  delete [] dcicol_l;
  delete [] dccoords_x_l;

  cudaFreeHelper(dcv2);

  printf("%d,%f,%f,%f,%f,%f,%f,%f\n", l,
          ((double)total_data/1e9)/(time_cpu/repeat_test),
          ((double)total_data/1e9)/(time_first_cuda/repeat_test),
          ((double)total_data/1e9)/(time_l1_compacted_cuda/repeat_test),
          ((double)total_data/1e9)/(time_l2_compacted_cuda/repeat_test),
          ((double)total_data/1e9)/(time_l2_compacted_sm_cuda/repeat_test),
          ((double)total_data/1e9)/(time_l2_compacted_sm_tuned_cuda/repeat_test),
          ((double)total_data/1e9)/(time_l2_compacted_sm_tuned_pf_cuda/repeat_test));

}


void benchmark_mass_mult_l_row(int nr, int nc) {

  double * v = new double[nr * nc];
  int ldv = nc;
  int * irow = new int[nr];
  int * icol = new int[nc];
  double * coords_x = new double[nc];
  int row_stride = 1;
  int col_stride = 1;

  for (int i = 0; i < nr * nc; i++) v[i] = i;
  for (int i = 0; i < nr; i++) irow[i] = i;
  for (int i = 0; i < nc; i++) icol[i] = i;
  for (int i = 0; i < nc; i++) coords_x[i] = i;

  double * dv;
  size_t dv_pitch;
  cudaMallocPitchHelper((void**)&dv, &dv_pitch, nc * sizeof(double), nr);
  int lddv = dv_pitch / sizeof(double);
  cudaMemcpy2DHelper(dv, lddv * sizeof(double), 
                     v, ldv  * sizeof(double), 
                     nc * sizeof(double), nr, 
                     H2D);

  double * dcoords_x;
  cudaMallocHelper((void**)&dcoords_x, nc * sizeof(double));
  cudaMemcpyHelper(dcoords_x, coords_x, nc * sizeof(double), H2D);

  int * dirow;
  cudaMallocHelper((void**)&dirow, nr * sizeof(int));
  cudaMemcpyHelper(dirow, irow, nr * sizeof(int), H2D);

  int * dicol;
  cudaMallocHelper((void**)&dicol, nc * sizeof(int));
  cudaMemcpyHelper(dicol, icol, nc * sizeof(int), H2D);

  mgard_cuda_ret ret;
  int repeat_test = 10;
  double time = 0.0;

  size_t total_data = 2*nr*nc*sizeof(double) + nc * sizeof(double);

  for (int i = 0; i < repeat_test; i++) {
    cudaMemcpy2DHelper(dv, lddv * sizeof(double), 
                     v, ldv  * sizeof(double), 
                     nc * sizeof(double), nr, 
                     H2D);
    ret = mgard_2d::mgard_gen::mass_mult_l_row_cuda(nr,    nc,
                                                   nr,    nc,
                                                   row_stride, col_stride,
                                                   dirow, dicol,
                                                   dv,     lddv,
                                                   dcoords_x);
    time += ret.time;
  }
  printf("[Compacted CUDA]: %f s ( %f GB/s)\n", 
          time/repeat_test, ((double)total_data/1e9)/(time/repeat_test));

  double * dv2;
  size_t dv2_pitch;
  cudaMallocPitchHelper((void**)&dv2, &dv2_pitch, nc * sizeof(double), nr);
  int lddv2 = dv2_pitch / sizeof(double);

  // for (int ghost_col = 8; ghost_col <= 32; ghost_col *= 2) {
  //   for (int B = ghost_col; B <=32; B *=2) {

  //     //int ghost_col = 16;
  //     time = 0.0;
  //     int pass = 0;
  //     for (int i = 0; i < repeat_test; i++) {
  //       cudaMemcpy2DHelper(dv2, lddv2 * sizeof(double), 
  //                          v, ldv  * sizeof(double), 
  //                          nc * sizeof(double), nr, 
  //                          H2D);
  //       ret = mgard_2d::mgard_gen::mass_mult_l_row_cuda_sm(nr,    nc,
  //                                                                nr,    nc,
  //                                                                row_stride, col_stride,
  //                                                                dirow, dicol,
  //                                                                dv2,     lddv2,
  //                                                                dcoords_x,
  //                                                                B, ghost_col);
  //       time += ret.time;

  //       bool correct = compare_matrix_cuda(nr, nc, dv, lddv, dv2, lddv2);
  //       if (correct) pass++;
  //     }
  //     printf("[Compacted-SM CUDA (%d, %d)]: %f s ( %f GB/s) pass(%d/%d)\n", B, ghost_col,
  //           time/repeat_test, ((double)total_data/1e9)/(time/repeat_test),
  //           pass, repeat_test);
  //   }
  // }

  double * dv3;
  size_t dv3_pitch;
  cudaMallocPitchHelper((void**)&dv3, &dv3_pitch, nc * sizeof(double), nr);
  int lddv3 = dv3_pitch / sizeof(double);

  for (int ghost_col = 8; ghost_col <= 32; ghost_col *= 2) {
    for (int B = ghost_col; B <=32; B *=2) {

      //int ghost_col = 16;
      time = 0.0;
      int pass = 0;
      for (int i = 0; i < repeat_test; i++) {
        cudaMemcpy2DHelper(dv3, lddv3 * sizeof(double), 
                           v, ldv  * sizeof(double), 
                           nc * sizeof(double), nr, 
                           H2D);
        ret = mgard_2d::mgard_gen::mass_mult_l_row_cuda_sm_pf(nr,    nc,
                                                                 nr,    nc,
                                                                 row_stride, col_stride,
                                                                 dirow, dicol,
                                                                 dv3,     lddv3,
                                                                 dcoords_x,
                                                                 B, ghost_col);
        time += ret.time;

        bool correct = compare_matrix_cuda(nr, nc, dv, lddv, dv3, lddv3);
        if (correct) pass++;
      }
      printf("[Compacted-SM-PF CUDA (%d, %d)]: %f s ( %f GB/s) pass(%d/%d)\n", B, ghost_col,
            time/repeat_test, ((double)total_data/1e9)/(time/repeat_test),
            pass, repeat_test);
    }
  }

  delete v;
  delete irow;
  delete icol;
  delete coords_x;
  cudaFree(dv);
  cudaFree(dv2);
  cudaFree(dv3);
  cudaFree(dirow);
  cudaFree(dicol);
  cudaFree(dcoords_x);

}


int main(int argc, char *argv[])
{
  //for (int s = 16; s < 16384; s *= 2) 
  int s = 10000;
  {
    // benchmark_mass_mult_l_row(s, s);
    std::cout << "size = " << s << std::endl;
    int nlevel_x = std::log2(s-1);
    int nc = std::pow(2, nlevel_x ) + 1; //ncol new

    int nlevel_y = std::log2(s-1);
    int nr = std::pow(2, nlevel_y ) + 1; //nrow new

    int nlevel = std::min(nlevel_x, nlevel_y);

    int l_target = nlevel-1;

    for (int l = 0; l < l_target; l++){
      std::cout << "l = " << l << std::endl;
    
      benchmark_mass_mult_l_row_org(s, s, l, l_target);
    }
  }
  return 0;
}