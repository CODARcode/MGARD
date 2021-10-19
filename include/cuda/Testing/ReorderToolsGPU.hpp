#ifndef REORDERTOOLSGPU_HPP
#define REORDERTOOLSGPU_HPP

#include "ReorderToolsGPU.h"
#include "../GridProcessingKernel.h"
#include "../GridProcessingKernel3D.h"
#include "../LevelwiseProcessingKernel.h"
#include "../DataRefactoring.h"

#include "../CommonInternal.h"

namespace mgard_cuda {

template <DIM D, typename T>
void ReorderGPU(Handle<D, T> &handle, SubArray<D, T, CUDA> dinput, 
                             SubArray<D, T, CUDA> &doutput, int l_target, int queue_idx) {


  SubArray<D, T, CUDA> dcoarse, dcoeff_f, dcoeff_c, dcoeff_r, 
                 dcoeff_cf, dcoeff_rf, dcoeff_rc,
                 dcoeff_rcf;

  DIM curr_dims[3];
  //handle.l_target = 1;
  for (int l = 0; l < l_target; ++l) {
  	int range_l = std::min(6, (int)std::log2(handle.dofs[0][l]) - 1);
  	int range_lp1 = std::min(6, (int)std::log2(handle.dofs[0][l + 1]) - 1);
	  int unprocessed_idx = 0;
	  printf("reorder 1-3D\n");
	  curr_dims[0] = 0; curr_dims[1] = 1; curr_dims[2] = 2;
	  dinput.project(curr_dims[0], curr_dims[1], curr_dims[2]);
	  doutput.project(curr_dims[0], curr_dims[1], curr_dims[2]); //reuse input1 as temp space
	  calc_coeff_pointers(handle, curr_dims, l, doutput, 
	                      dcoarse, 
	                      dcoeff_f, dcoeff_c, dcoeff_r, 
	                      dcoeff_cf, dcoeff_rf, dcoeff_rc,
	                      dcoeff_rcf);
	  printf("done calc ptrs\n");
	  if (D <= 3) {
	  	gpk_reo<D, D, T, false, false, 1>(
		      handle, handle.shapes_h[l], handle.shapes_d[l],
		      handle.shapes_d[l + 1], dinput.ldvs_d, doutput.ldvs_d,
		      handle.unprocessed_n[unprocessed_idx],
		      handle.unprocessed_dims_d[unprocessed_idx],
		      curr_dims[2], curr_dims[1], curr_dims[0], 
		      handle.ratio[curr_dims[2]][l], handle.ratio[curr_dims[1]][l], handle.ratio[curr_dims[0]][l], 
		      dinput.dv, dinput.lddv1, dinput.lddv2, 
		      dcoarse.dv, dcoarse.lddv1, dcoarse.lddv2, 
		      // null, lddv1, lddv2,
		      dcoeff_f.dv, dcoeff_f.lddv1, dcoeff_f.lddv2, 
		      // null, lddv1, lddv2,
		      dcoeff_c.dv, dcoeff_c.lddv1, dcoeff_c.lddv2, 
		      // null, lddv1, lddv2,
		      dcoeff_r.dv, dcoeff_r.lddv1, dcoeff_r.lddv2, 
		      // null, lddv1, lddv2,
		      dcoeff_cf.dv, dcoeff_cf.lddv1, dcoeff_cf.lddv2, 
		      // null, lddv1, lddv2,
		      dcoeff_rf.dv, dcoeff_rf.lddv1, dcoeff_rf.lddv2, 
		      // null, lddv1, lddv2,
		      dcoeff_rc.dv, dcoeff_rc.lddv1, dcoeff_rc.lddv2, 
		      // null, lddv1, lddv2,
		      dcoeff_rcf.dv, dcoeff_rcf.lddv1, dcoeff_rcf.lddv2, 
		      // null, lddv1, lddv2,
		      queue_idx,
		      handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);
	  		printf("done reo\n");
	  } else {
		  gpk_reo<D, 3, T, false, false, 1>(
		      handle, handle.shapes_h[l], handle.shapes_d[l],
		      handle.shapes_d[l + 1], dinput.ldvs_d, doutput.ldvs_d,
		      handle.unprocessed_n[unprocessed_idx],
		      handle.unprocessed_dims_d[unprocessed_idx],
		      curr_dims[2], curr_dims[1], curr_dims[0], 
		      handle.ratio[curr_dims[2]][l], handle.ratio[curr_dims[1]][l], handle.ratio[curr_dims[0]][l], 
		      dinput.dv, dinput.lddv1, dinput.lddv2, 
		      dcoarse.dv, dcoarse.lddv1, dcoarse.lddv2, 
		      // null, lddv1, lddv2,
		      dcoeff_f.dv, dcoeff_f.lddv1, dcoeff_f.lddv2, 
		      // null, lddv1, lddv2,
		      dcoeff_c.dv, dcoeff_c.lddv1, dcoeff_c.lddv2, 
		      // null, lddv1, lddv2,
		      dcoeff_r.dv, dcoeff_r.lddv1, dcoeff_r.lddv2, 
		      // null, lddv1, lddv2,
		      dcoeff_cf.dv, dcoeff_cf.lddv1, dcoeff_cf.lddv2, 
		      // null, lddv1, lddv2,
		      dcoeff_rf.dv, dcoeff_rf.lddv1, dcoeff_rf.lddv2, 
		      // null, lddv1, lddv2,
		      dcoeff_rc.dv, dcoeff_rc.lddv1, dcoeff_rc.lddv2, 
		      // null, lddv1, lddv2,
		      dcoeff_rcf.dv, dcoeff_rcf.lddv1, dcoeff_rcf.lddv2, 
		      // null, lddv1, lddv2,
		      queue_idx,
		      handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

		  for (DIM d = 3; d < D; d += 2) {
		    //copy back to input for reordering again
		    lwpk<D, T, COPY>(handle, handle.shapes_h[l], handle.shapes_d[l],
		                   doutput.dv, doutput.ldvs_d, dinput.dv, dinput.ldvs_d, queue_idx);
		    printf("reorder-restore %u-%uD\n", d+1, d+2);
		    curr_dims[0] = 0; curr_dims[1] = d; curr_dims[2] = d+1;
		    dinput.project(curr_dims[0], curr_dims[1], curr_dims[2]);
			  doutput.project(curr_dims[0], curr_dims[1], curr_dims[2]); //reuse input1 as temp space
			  calc_coeff_pointers(handle, curr_dims, l, doutput, 
			                      dcoarse, 
			                      dcoeff_f, dcoeff_c, dcoeff_r, 
			                      dcoeff_cf, dcoeff_rf, dcoeff_rc,
			                      dcoeff_rcf);

			  if (D-d == 1) {
			  	unprocessed_idx += 1;
			    gpk_reo<D, 2, T, false, false, 2>(
			      handle, handle.shapes_h[l], handle.shapes_d[l],
			      handle.shapes_d[l + 1], dinput.ldvs_d, doutput.ldvs_d,
			      handle.unprocessed_n[unprocessed_idx],
			      handle.unprocessed_dims_d[unprocessed_idx],
			      curr_dims[2], curr_dims[1], curr_dims[0], 
			      handle.ratio[curr_dims[2]][l], handle.ratio[curr_dims[1]][l], handle.ratio[curr_dims[0]][l], 
			      dinput.dv, dinput.lddv1, dinput.lddv2, 
			      dcoarse.dv, dcoarse.lddv1, dcoarse.lddv2, 
			      // null, lddv1, lddv2,
			      dcoeff_f.dv, dcoeff_f.lddv1, dcoeff_f.lddv2, 
			      // null, lddv1, lddv2,
			      dcoeff_c.dv, dcoeff_c.lddv1, dcoeff_c.lddv2, 
			      // null, lddv1, lddv2,
			      dcoeff_r.dv, dcoeff_r.lddv1, dcoeff_r.lddv2, 
			      // null, lddv1, lddv2,
			      dcoeff_cf.dv, dcoeff_cf.lddv1, dcoeff_cf.lddv2, 
			      // null, lddv1, lddv2,
			      dcoeff_rf.dv, dcoeff_rf.lddv1, dcoeff_rf.lddv2, 
			      // null, lddv1, lddv2,
			      dcoeff_rc.dv, dcoeff_rc.lddv1, dcoeff_rc.lddv2, 
			      // null, lddv1, lddv2,
			      dcoeff_rcf.dv, dcoeff_rcf.lddv1, dcoeff_rcf.lddv2, 
			      // null, lddv1, lddv2,
			      queue_idx,
			      handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);
			  } else {
			    unprocessed_idx += 2;
			    gpk_reo<D, 3, T, false, false, 2>(
			      handle, handle.shapes_h[l], handle.shapes_d[l],
			      handle.shapes_d[l + 1], dinput.ldvs_d, doutput.ldvs_d,
			      handle.unprocessed_n[unprocessed_idx],
			      handle.unprocessed_dims_d[unprocessed_idx],
			      curr_dims[2], curr_dims[1], curr_dims[0], 
			      handle.ratio[curr_dims[2]][l], handle.ratio[curr_dims[1]][l], handle.ratio[curr_dims[0]][l], 
			      dinput.dv, dinput.lddv1, dinput.lddv2, 
			      dcoarse.dv, dcoarse.lddv1, dcoarse.lddv2, 
			      // null, lddv1, lddv2,
			      dcoeff_f.dv, dcoeff_f.lddv1, dcoeff_f.lddv2, 
			      // null, lddv1, lddv2,
			      dcoeff_c.dv, dcoeff_c.lddv1, dcoeff_c.lddv2, 
			      // null, lddv1, lddv2,
			      dcoeff_r.dv, dcoeff_r.lddv1, dcoeff_r.lddv2, 
			      // null, lddv1, lddv2,
			      dcoeff_cf.dv, dcoeff_cf.lddv1, dcoeff_cf.lddv2, 
			      // null, lddv1, lddv2,
			      dcoeff_rf.dv, dcoeff_rf.lddv1, dcoeff_rf.lddv2, 
			      // null, lddv1, lddv2,
			      dcoeff_rc.dv, dcoeff_rc.lddv1, dcoeff_rc.lddv2, 
			      // null, lddv1, lddv2,
			      dcoeff_rcf.dv, dcoeff_rcf.lddv1, dcoeff_rcf.lddv2, 
			      // null, lddv1, lddv2,
			      queue_idx,
			      handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);
			  }
			}
	  }
	}
}

template <DIM D, typename T>
void ReverseReorderGPU(Handle<D, T> &handle, SubArray<D, T, CUDA> dinput, 
                             SubArray<D, T, CUDA> &doutput, int l_target, int queue_idx) {

  SubArray<D, T, CUDA> dcoarse, dcoeff_f, dcoeff_c, dcoeff_r, 
                 dcoeff_cf, dcoeff_rf, dcoeff_rc,
                 dcoeff_rcf;

  DIM curr_dims[3];
  for (int l = 0; l < l_target; ++l) {
  	int range_l = std::min(6, (int)std::log2(handle.dofs[0][l]) - 1);
  	int range_lp1 = std::min(6, (int)std::log2(handle.dofs[0][l + 1]) - 1);
	  int unprocessed_idx = 0;
	  printf("reorder-restore 1-3D\n");
	  curr_dims[0] = 0; curr_dims[1] = 1; curr_dims[2] = 2;
	  dinput.project(curr_dims[0], curr_dims[1], curr_dims[2]);
	  doutput.project(curr_dims[0], curr_dims[1], curr_dims[2]); //reuse input1 as temp space
	  calc_coeff_pointers(handle, curr_dims, l, dinput, 
	                      dcoarse, 
	                      dcoeff_f, dcoeff_c, dcoeff_r, 
	                      dcoeff_cf, dcoeff_rf, dcoeff_rc,
	                      dcoeff_rcf);
	  if (D <= 3) {
	  	gpk_rev<D, D, T, false, false, 1>(
		      handle, handle.shapes_h[l], handle.shapes_d[l],
		      handle.shapes_d[l + 1], doutput.ldvs_d, dinput.ldvs_d,
		      handle.unprocessed_n[unprocessed_idx],
		      handle.unprocessed_dims_d[unprocessed_idx],
		      curr_dims[2], curr_dims[1], curr_dims[0], 
		      handle.ratio[curr_dims[2]][l], handle.ratio[curr_dims[1]][l], handle.ratio[curr_dims[0]][l], 
		      doutput.dv, doutput.lddv1, doutput.lddv2, 
		      dcoarse.dv, dcoarse.lddv1, dcoarse.lddv2, 
		      // null, lddv1, lddv2,
		      dcoeff_f.dv, dcoeff_f.lddv1, dcoeff_f.lddv2, 
		      // null, lddv1, lddv2,
		      dcoeff_c.dv, dcoeff_c.lddv1, dcoeff_c.lddv2, 
		      // null, lddv1, lddv2,
		      dcoeff_r.dv, dcoeff_r.lddv1, dcoeff_r.lddv2, 
		      // null, lddv1, lddv2,
		      dcoeff_cf.dv, dcoeff_cf.lddv1, dcoeff_cf.lddv2, 
		      // null, lddv1, lddv2,
		      dcoeff_rf.dv, dcoeff_rf.lddv1, dcoeff_rf.lddv2, 
		      // null, lddv1, lddv2,
		      dcoeff_rc.dv, dcoeff_rc.lddv1, dcoeff_rc.lddv2, 
		      // null, lddv1, lddv2,
		      dcoeff_rcf.dv, dcoeff_rcf.lddv1, dcoeff_rcf.lddv2, 
		      // null, lddv1, lddv2,
		      0, 0, 0, 
		      handle.dofs[curr_dims[2]][l], handle.dofs[curr_dims[1]][l], handle.dofs[curr_dims[0]][l], 
		      queue_idx,
		      handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);
	  } else {
		  gpk_rev<D, 3, T, false, false, 1>(
		      handle, handle.shapes_h[l], handle.shapes_d[l],
		      handle.shapes_d[l + 1], doutput.ldvs_d, dinput.ldvs_d,
		      handle.unprocessed_n[unprocessed_idx],
		      handle.unprocessed_dims_d[unprocessed_idx],
		      curr_dims[2], curr_dims[1], curr_dims[0], 
		      handle.ratio[curr_dims[2]][l], handle.ratio[curr_dims[1]][l], handle.ratio[curr_dims[0]][l], 
		      doutput.dv, doutput.lddv1, doutput.lddv2, 
		      dcoarse.dv, dcoarse.lddv1, dcoarse.lddv2, 
		      // null, lddv1, lddv2,
		      dcoeff_f.dv, dcoeff_f.lddv1, dcoeff_f.lddv2, 
		      // null, lddv1, lddv2,
		      dcoeff_c.dv, dcoeff_c.lddv1, dcoeff_c.lddv2, 
		      // null, lddv1, lddv2,
		      dcoeff_r.dv, dcoeff_r.lddv1, dcoeff_r.lddv2, 
		      // null, lddv1, lddv2,
		      dcoeff_cf.dv, dcoeff_cf.lddv1, dcoeff_cf.lddv2, 
		      // null, lddv1, lddv2,
		      dcoeff_rf.dv, dcoeff_rf.lddv1, dcoeff_rf.lddv2, 
		      // null, lddv1, lddv2,
		      dcoeff_rc.dv, dcoeff_rc.lddv1, dcoeff_rc.lddv2, 
		      // null, lddv1, lddv2,
		      dcoeff_rcf.dv, dcoeff_rcf.lddv1, dcoeff_rcf.lddv2, 
		      // null, lddv1, lddv2,
		      0, 0, 0, 
		      handle.dofs[curr_dims[2]][l], handle.dofs[curr_dims[1]][l], handle.dofs[curr_dims[0]][l], 
		      queue_idx,
		      handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);

		  for (DIM d = 3; d < D; d += 2) {
		    //copy back to input for reordering again
		    lwpk<D, T, COPY>(handle, handle.shapes_h[l], handle.shapes_d[l],
		                   doutput.dv, doutput.ldvs_d, dinput.dv, dinput.ldvs_d, queue_idx);
		    
		    curr_dims[0] = 0; curr_dims[1] = d; curr_dims[2] = d+1;
		    dinput.project(curr_dims[0], curr_dims[1], curr_dims[2]);
			  doutput.project(curr_dims[0], curr_dims[1], curr_dims[2]); //reuse input1 as temp space
			  calc_coeff_pointers(handle, curr_dims, l, dinput, 
			                      dcoarse, 
			                      dcoeff_f, dcoeff_c, dcoeff_r, 
			                      dcoeff_cf, dcoeff_rf, dcoeff_rc,
			                      dcoeff_rcf);

			  if (D-d == 1) {
			  	printf("reorder-restore %u-%uD\n", d+1, d+1);
			  	unprocessed_idx += 1;
			    gpk_rev<D, 2, T, false, false, 2>(
			      handle, handle.shapes_h[l], handle.shapes_d[l],
			      handle.shapes_d[l + 1], doutput.ldvs_d, dinput.ldvs_d,
			      handle.unprocessed_n[unprocessed_idx],
			      handle.unprocessed_dims_d[unprocessed_idx],
			      curr_dims[2], curr_dims[1], curr_dims[0], 
			      handle.ratio[curr_dims[2]][l], handle.ratio[curr_dims[1]][l], handle.ratio[curr_dims[0]][l], 
			      doutput.dv, doutput.lddv1, doutput.lddv2, 
			      dcoarse.dv, dcoarse.lddv1, dcoarse.lddv2, 
			      // null, lddv1, lddv2,
			      dcoeff_f.dv, dcoeff_f.lddv1, dcoeff_f.lddv2, 
			      // null, lddv1, lddv2,
			      dcoeff_c.dv, dcoeff_c.lddv1, dcoeff_c.lddv2, 
			      // null, lddv1, lddv2,
			      dcoeff_r.dv, dcoeff_r.lddv1, dcoeff_r.lddv2, 
			      // null, lddv1, lddv2,
			      dcoeff_cf.dv, dcoeff_cf.lddv1, dcoeff_cf.lddv2, 
			      // null, lddv1, lddv2,
			      dcoeff_rf.dv, dcoeff_rf.lddv1, dcoeff_rf.lddv2, 
			      // null, lddv1, lddv2,
			      dcoeff_rc.dv, dcoeff_rc.lddv1, dcoeff_rc.lddv2, 
			      // null, lddv1, lddv2,
			      dcoeff_rcf.dv, dcoeff_rcf.lddv1, dcoeff_rcf.lddv2, 
			      // null, lddv1, lddv2,
			      0, 0, 0, 
			      handle.dofs[curr_dims[2]][l], handle.dofs[curr_dims[1]][l], handle.dofs[curr_dims[0]][l], 
			      queue_idx,
			      handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);
			  } else {
			  	printf("reorder-restore %u-%uD\n", d+1, d+2);
			    unprocessed_idx += 2;
			    gpk_rev<D, 3, T, false, false, 2>(
			      handle, handle.shapes_h[l], handle.shapes_d[l],
			      handle.shapes_d[l + 1], doutput.ldvs_d, dinput.ldvs_d,
			      handle.unprocessed_n[unprocessed_idx],
			      handle.unprocessed_dims_d[unprocessed_idx],
			      curr_dims[2], curr_dims[1], curr_dims[0], 
			      handle.ratio[curr_dims[2]][l], handle.ratio[curr_dims[1]][l], handle.ratio[curr_dims[0]][l], 
			      doutput.dv, doutput.lddv1, doutput.lddv2, 
			      dcoarse.dv, dcoarse.lddv1, dcoarse.lddv2, 
			      // null, lddv1, lddv2,
			      dcoeff_f.dv, dcoeff_f.lddv1, dcoeff_f.lddv2, 
			      // null, lddv1, lddv2,
			      dcoeff_c.dv, dcoeff_c.lddv1, dcoeff_c.lddv2, 
			      // null, lddv1, lddv2,
			      dcoeff_r.dv, dcoeff_r.lddv1, dcoeff_r.lddv2, 
			      // null, lddv1, lddv2,
			      dcoeff_cf.dv, dcoeff_cf.lddv1, dcoeff_cf.lddv2, 
			      // null, lddv1, lddv2,
			      dcoeff_rf.dv, dcoeff_rf.lddv1, dcoeff_rf.lddv2, 
			      // null, lddv1, lddv2,
			      dcoeff_rc.dv, dcoeff_rc.lddv1, dcoeff_rc.lddv2, 
			      // null, lddv1, lddv2,
			      dcoeff_rcf.dv, dcoeff_rcf.lddv1, dcoeff_rcf.lddv2, 
			      // null, lddv1, lddv2,
			      0, 0, 0, 
			      handle.dofs[curr_dims[2]][l], handle.dofs[curr_dims[1]][l], handle.dofs[curr_dims[0]][l], 
			      queue_idx,
			      handle.auto_tuning_cc[handle.arch][handle.precision][range_l]);
			  }
			}
	  }
	}
}
}

#endif