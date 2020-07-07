#include <string>
#include <vector>
#include <iomanip> 
#include <numeric>

#ifndef MGRAD_CUDA_HANDLE
#define MGRAD_CUDA_HANDLE

template <typename T>
struct mgard_cuda_handle {
  
  mgard_cuda_handle ();

  /* for Internal use only */
  // mgard_cuda_handle (int num_of_queues);

  /* for general users */
  mgard_cuda_handle (int nrow, int ncol, int nfib);
  mgard_cuda_handle (int nrow, int ncol, int nfib, T * coords_r, T * coords_c, T * coords_f);
  
  /* for performance tuning */
  mgard_cuda_handle (int nrow, int ncol, int nfib, int B, int num_of_queues, int opt);
  mgard_cuda_handle (int nrow, int ncol, int nfib, T * coords_r, T * coords_c, T * coords_f, int B, int num_of_queues, int opt);
  ~mgard_cuda_handle ();

  int get_lindex(const int n, const int no, const int i);
  void * get(int i);
  void sync(int i);
  void sync_all();
  void destory();

  /* CUDA env */
  void * queues;
  int num_of_queues;

  /* Refactoring env */
  int B;
  int opt;
  int l_target;
  int nrow, ncol, nfib, nr, nc, nf;
  T * dcoords_r, * dcoords_c, * dcoords_f;
  int * nr_l, * nc_l, * nf_l;
  int * dirow, * dicol, * difib;
  int * dirow_p, * dicol_p, * difib_p;
  int * dirow_a, * dicol_a, * difib_a;
  T * ddist_r, * ddist_c, * ddist_f;
  T ** ddist_r_l, ** ddist_c_l, ** ddist_f_l;
  T * dwork;
  int lddwork, lddwork1, lddwork2;
  T ** dcwork_2d_rc, ** dcwork_2d_cf;
  int * lddcwork_2d_rc, * lddcwork_2d_cf;
  T ** am_row, ** bm_row, ** am_col, ** bm_col, ** am_fib, ** bm_fib;


private: 
  void init (int nrow, int ncol, int nfib, T * coords_r, T * coords_c, T * coords_f, int B, int num_of_queues, int opt);
  
};

template struct mgard_cuda_handle<double>;
template struct mgard_cuda_handle<float>;

#endif