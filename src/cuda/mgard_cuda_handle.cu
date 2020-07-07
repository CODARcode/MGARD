#include "cuda/mgard_cuda_handle.h"
#include "cuda/mgard_cuda_common_internal.h"
#include "cuda/mgard_cuda_helper.h"
#include "cuda/mgard_cuda_compact_helper.h"

// template <typename T> int
// mgard_cuda_handle<T>::get_lindex(const int n, const int no, const int i) {
//   // no: original number of points
//   // n : number of points at next coarser level (L-1) with  2^k+1 nodes
//   int lindex;
//   //    return floor((no-2)/(n-2)*i);
//   if (i != n - 1) {
//     lindex = floor(((double)no - 2.0) / ((double)n - 2.0) * i);
//   } else if (i == n - 1) {
//     lindex = no - 1;
//   }

//   return lindex;
// }



template <typename T> void
mgard_cuda_handle<T>::init (int nrow, int ncol, int nfib, T * coords_r, T * coords_c, T * coords_f, int B, int num_of_queues, int opt){

  this->B = B;
  this->num_of_queues = num_of_queues;
  cudaStream_t * ptr = new cudaStream_t[num_of_queues];
  for (int i = 0; i < this->num_of_queues; i++) {
    gpuErrchk(cudaStreamCreate(ptr+i));
    // std::cout << "created a stream: " << *(ptr+i) <<"\n";
  }

  this->queues = (void *)ptr;
  this->opt = opt;


  this->nrow = nrow;
  this->ncol = ncol;
  this->nfib = nfib;

  int nlevel_r = std::log2(nrow - 1);
  this->nr = std::pow(2, nlevel_r) + 1;
  int nlevel_c = std::log2(ncol - 1);
  this->nc = std::pow(2, nlevel_c) + 1;
  int nlevel_f = std::log2(nfib - 1);
  this->nf = std::pow(2, nlevel_f) + 1;

  int nlevel = std::min(nlevel_r, nlevel_c);

  if (nfib > 1) nlevel = std::min(nlevel, nlevel_f);

  this->l_target = nlevel - 1;

  this->nr_l = new int[l_target+1];
  this->nc_l = new int[l_target+1];
  if (nfib > 1) this->nf_l = new int[l_target+1];


  for (int l = 0; l < l_target + 1; l++) {
    nr_l[l] = ceil((float)nr/std::pow(2, l));
    nc_l[l] = ceil((float)nc/std::pow(2, l));
    if (nfib > 1) nf_l[l] = ceil((float)nf/std::pow(2, l));
  }

  int * irow  = new int[nr];
  int * irowP = new int[nrow-nr];
  int * irowA = new int[nrow];
  int irow_ptr  = 0;
  int irowP_ptr = 0;

  for (int i = 0; i < nr; i++) {
    int irow_r = mgard_cuda::get_lindex_cuda(nr, nrow, i);
    irow[irow_ptr] = irow_r; 
    if (irow_ptr > 0 && irow[irow_ptr - 1] != irow[irow_ptr] - 1) {
      irowP[irowP_ptr] = irow[irow_ptr] - 1;
      irowP_ptr ++;
    } 
    irow_ptr++;
  }
  for(int i = 0; i < nrow; i++) irowA[i] = i;

  int * icol  = new int[nc];
  int * icolP = new int[ncol-nc];
  int * icolA = new int[ncol];
  int icol_ptr  = 0;
  int icolP_ptr = 0;

  for (int i = 0; i < nc; i++) {
    int icol_c = mgard_cuda::get_lindex_cuda(nc, ncol, i);
    icol[icol_ptr] = icol_c;
    if (icol_ptr > 0 && icol[icol_ptr - 1] != icol[icol_ptr] - 1) {
      icolP[icolP_ptr] = icol[icol_ptr] - 1;
      icolP_ptr ++;
    } 
    icol_ptr++;
  }
  for (int i = 0; i < ncol; i++) icolA[i] = i;
    
  int * ifib  = new int[nf];
  int * ifibP = new int[nfib-nf];
  int * ifibA = new int[nfib];
  int ifib_ptr  = 0;
  int ifibP_ptr = 0;
  if (nfib > 1) {
    for (int i = 0; i < nf; i++) {
      int ifib_f = mgard_cuda::get_lindex_cuda(nf, nfib, i);
      ifib[ifib_ptr] = ifib_f;
      if (ifib_ptr > 0 && ifib[ifib_ptr - 1] != ifib[ifib_ptr] - 1) {
        ifibP[ifibP_ptr] = ifib[ifib_ptr] - 1;
        ifibP_ptr ++;
      } 
      ifib_ptr++;
    }
    for (int i = 0; i < nfib; i++) ifibA[i] = i;
  }

  mgard_cuda::cudaMallocHelper((void**)&(this->dirow), nr * sizeof(int));
  mgard_cuda::cudaMemcpyAsyncHelper(*this, this->dirow, irow, nr * sizeof(int), mgard_cuda::H2D, 0);

  mgard_cuda::cudaMallocHelper((void**)&(this->dicol), nc * sizeof(int));
  mgard_cuda::cudaMemcpyAsyncHelper(*this, this->dicol, icol, nc * sizeof(int), mgard_cuda::H2D, 0);

  if (nfib > 1) {
    mgard_cuda::cudaMallocHelper((void**)&(this->difib), nf * sizeof(int));
    mgard_cuda::cudaMemcpyAsyncHelper(*this, this->difib, ifib, nf * sizeof(int), mgard_cuda::H2D, 0);
  }

  mgard_cuda::cudaMallocHelper((void**)&(this->dirow_p), (nrow-nr) * sizeof(int));
  mgard_cuda::cudaMemcpyAsyncHelper(*this, this->dirow_p, irowP, (nrow-nr) * sizeof(int), mgard_cuda::H2D, 0);

  mgard_cuda::cudaMallocHelper((void**)&(this->dicol_p), (ncol-nc) * sizeof(int));
  mgard_cuda::cudaMemcpyAsyncHelper(*this, this->dicol_p, icolP, (ncol-nc) * sizeof(int), mgard_cuda::H2D, 0);

  if (nfib > 1) { 
    mgard_cuda::cudaMallocHelper((void**)&(this->difib_p), (nfib-nf) * sizeof(int));
    mgard_cuda::cudaMemcpyAsyncHelper(*this, this->difib_p, ifibP, (nfib-nf) * sizeof(int), mgard_cuda::H2D, 0);
  }

  mgard_cuda::cudaMallocHelper((void**)&(this->dirow_a), nrow * sizeof(int));
  mgard_cuda::cudaMemcpyAsyncHelper(*this, this->dirow_a, irowA, nrow * sizeof(int), mgard_cuda::H2D, 0);

  mgard_cuda::cudaMallocHelper((void**)&(this->dicol_a), ncol * sizeof(int));
  mgard_cuda::cudaMemcpyAsyncHelper(*this, this->dicol_a, icolA, ncol * sizeof(int), mgard_cuda::H2D, 0);

  if (nfib > 1) {
    mgard_cuda::cudaMallocHelper((void**)&(this->difib_a), nfib * sizeof(int));
    mgard_cuda::cudaMemcpyAsyncHelper(*this, this->difib_a, ifibA, nfib * sizeof(int), mgard_cuda::H2D, 0);
  }

  
  mgard_cuda::cudaMallocHelper((void**)&(this->dcoords_r), nrow * sizeof(T));
  mgard_cuda::cudaMemcpyAsyncHelper(*this, this->dcoords_r, coords_r, nrow * sizeof(T), mgard_cuda::H2D, 0);
  
  mgard_cuda::cudaMallocHelper((void**)&(this->dcoords_c), ncol * sizeof(T));
  mgard_cuda::cudaMemcpyAsyncHelper(*this, this->dcoords_c, coords_c, ncol * sizeof(T), mgard_cuda::H2D, 0);

  if (nfib > 1) {
    mgard_cuda::cudaMallocHelper((void**)&(this->dcoords_f), nfib * sizeof(T));
    mgard_cuda::cudaMemcpyAsyncHelper(*this, this->dcoords_f, coords_f, nfib * sizeof(T), mgard_cuda::H2D, 0);
  }
  mgard_cuda::cudaMallocHelper((void**)&(this->ddist_r), nrow * sizeof(T));
  mgard_cuda::calc_cpt_dist(*this, nrow, 1, this->dcoords_r, this->ddist_r, 0);

  mgard_cuda::cudaMallocHelper((void**)&(this->ddist_c), ncol * sizeof(T));
  mgard_cuda::calc_cpt_dist(*this, ncol, 1, this->dcoords_c, this->ddist_c, 0);

  if (nfib > 1) {
    mgard_cuda::cudaMallocHelper((void**)&(this->ddist_f), nfib * sizeof(T));
    mgard_cuda::calc_cpt_dist(*this, nfib, 1, this->dcoords_f, this->ddist_f, 0);
  }

  T * dccoords_r, * dccoords_c, * dccoords_f;
  mgard_cuda::cudaMallocHelper((void**)&dccoords_r, nr * sizeof(T));
  mgard_cuda::cudaMallocHelper((void**)&dccoords_c, nc * sizeof(T));
  if (nfib > 1) mgard_cuda::cudaMallocHelper((void**)&dccoords_f, nf * sizeof(T));

  mgard_cuda::org_to_pow2p1(*this, nrow, nr, dirow, this->dcoords_r, dccoords_r, 0);
  mgard_cuda::org_to_pow2p1(*this, ncol, nc, dicol, this->dcoords_c, dccoords_c, 0);
  if (nfib > 1) mgard_cuda::org_to_pow2p1(*this, nfib, nf, difib, this->dcoords_f, dccoords_f, 0);

  this->ddist_r_l = new T*[l_target+1];
  this->ddist_c_l = new T*[l_target+1];
  if (nfib > 1) this->ddist_f_l = new T*[l_target+1];

  for (int l = 0; l < l_target+1; l++) {
    int stride = std::pow(2, l);
    mgard_cuda::cudaMallocHelper((void**)&(this->ddist_r_l[l]), this->nr_l[l] * sizeof(T));
    mgard_cuda::calc_cpt_dist(*this, this->nr, stride, dccoords_r, this->ddist_r_l[l], 0);

    mgard_cuda::cudaMallocHelper((void**)&(this->ddist_c_l[l]), this->nc_l[l] * sizeof(T));
    mgard_cuda::calc_cpt_dist(*this, this->nc, stride, dccoords_c, this->ddist_c_l[l], 0);
    if (nfib > 1) {
      mgard_cuda::cudaMallocHelper((void**)&(this->ddist_f_l[l]), this->nf_l[l] * sizeof(T));
      mgard_cuda::calc_cpt_dist(*this, this->nf, stride, dccoords_f, this->ddist_f_l[l], 0);
    }

  }

  if (nfib > 1) {
    size_t dwork_pitch;
    mgard_cuda::cudaMalloc3DHelper((void**)&(this->dwork), &dwork_pitch, nfib * sizeof(T), ncol, nrow);
    this->lddwork1 = dwork_pitch / sizeof(T);
    this->lddwork2 = ncol;

    this->dcwork_2d_rc = new T*[num_of_queues];
    this->lddcwork_2d_rc = new int[num_of_queues];
    for (int i = 0; i < num_of_queues; i++) {
      size_t dcwork_2d_rc_pitch;
      mgard_cuda::cudaMallocPitchHelper((void**)&dcwork_2d_rc[i], &dcwork_2d_rc_pitch, nc * sizeof(T), nr);
      lddcwork_2d_rc[i] = dcwork_2d_rc_pitch / sizeof(T);
    }

    this->dcwork_2d_cf = new T*[num_of_queues];
    this->lddcwork_2d_cf = new int[num_of_queues];
    for (int i = 0; i < num_of_queues; i++) {
      size_t dcwork_2d_cf_pitch;
      mgard_cuda::cudaMallocPitchHelper((void**)&dcwork_2d_cf[i], &dcwork_2d_cf_pitch, nf * sizeof(T), nc);
      lddcwork_2d_cf[i] = dcwork_2d_cf_pitch / sizeof(T);
    }

    this->am_row = new T*[num_of_queues];
    this->bm_row = new T*[num_of_queues];
    this->am_col = new T*[num_of_queues];
    this->bm_col = new T*[num_of_queues];
    this->am_fib = new T*[num_of_queues];
    this->bm_fib = new T*[num_of_queues];
    for (int i = 0; i < num_of_queues; i++) {
      mgard_cuda::cudaMallocHelper((void**)&am_row[i], nr*sizeof(T));
      mgard_cuda::cudaMallocHelper((void**)&bm_row[i], nr*sizeof(T));
      mgard_cuda::cudaMallocHelper((void**)&am_col[i], nc*sizeof(T));
      mgard_cuda::cudaMallocHelper((void**)&bm_col[i], nc*sizeof(T));
      mgard_cuda::cudaMallocHelper((void**)&am_fib[i], nf*sizeof(T));
      mgard_cuda::cudaMallocHelper((void**)&bm_fib[i], nf*sizeof(T));
    }
  } else {
    size_t dwork_pitch;
    mgard_cuda::cudaMallocPitchHelper((void**)&(this->dwork), &dwork_pitch, ncol * sizeof(T), nrow);
    this->lddwork = dwork_pitch / sizeof(T);

    this->am_row = new T*[1];
    this->bm_row = new T*[1];
    this->am_col = new T*[1];
    this->bm_col = new T*[1];
    this->am_fib = new T*[1];
    this->bm_fib = new T*[1];
    mgard_cuda::cudaMallocHelper((void**)&(this->am_row[0]), nrow*sizeof(T));
    mgard_cuda::cudaMallocHelper((void**)&(this->bm_row[0]), nrow*sizeof(T));
    mgard_cuda::cudaMallocHelper((void**)&(this->am_col[0]), ncol*sizeof(T));
    mgard_cuda::cudaMallocHelper((void**)&(this->bm_col[0]), ncol*sizeof(T));
  }

  delete [] irow;
  delete [] irowP;
  delete [] irowA;

  delete [] icol;
  delete [] icolP;
  delete [] icolA;

  delete [] ifib;
  delete [] ifibP;
  delete [] ifibA;

  mgard_cuda::cudaFreeHelper(dccoords_r);
  mgard_cuda::cudaFreeHelper(dccoords_c);
  if (nfib > 1) mgard_cuda::cudaFreeHelper(dccoords_f);

}

template <typename T>
mgard_cuda_handle<T>::mgard_cuda_handle (){
  std::vector<T> coords_r(5), coords_c(5), coords_f(5);
  init(5,5,5, coords_r.data(), coords_c.data(), coords_f.data(), 16, 1, 1);
}

template <typename T>
mgard_cuda_handle<T>::mgard_cuda_handle (int nrow, int ncol, int nfib){
  std::vector<T> coords_r(nrow), coords_c(ncol), coords_f(nfib);
  std::iota(std::begin(coords_r), std::end(coords_r), 0);
  std::iota(std::begin(coords_c), std::end(coords_c), 0);
  std::iota(std::begin(coords_f), std::end(coords_f), 0);
  int B = 16;
  int num_of_queues = 8;
  int opt = 1;
  init(nrow, ncol, nfib, coords_r.data(), coords_c.data(), coords_f.data(), B, num_of_queues, opt);
}

template <typename T>
mgard_cuda_handle<T>::mgard_cuda_handle (int nrow, int ncol, int nfib, T * coords_r, T * coords_c, T * coords_f){
  int B = 16;
  int num_of_queues = 8;
  int opt = 1;
  init(nrow, ncol, nfib, coords_r, coords_c, coords_f, B, num_of_queues, opt);
}

template <typename T>
mgard_cuda_handle<T>::mgard_cuda_handle (int nrow, int ncol, int nfib, int B, int num_of_queues, int opt){
  std::vector<T> coords_r(nrow), coords_c(ncol), coords_f(nfib);
  std::iota(std::begin(coords_r), std::end(coords_r), 0);
  std::iota(std::begin(coords_c), std::end(coords_c), 0);
  std::iota(std::begin(coords_f), std::end(coords_f), 0);
  init(nrow, ncol, nfib, coords_r.data(), coords_c.data(), coords_f.data(), B, num_of_queues, opt);
}

template <typename T>
mgard_cuda_handle<T>::mgard_cuda_handle (int nrow, int ncol, int nfib, T * coords_r, T * coords_c, T * coords_f, int B, int num_of_queues, int opt){
  init(nrow, ncol, nfib, coords_r, coords_c, coords_f, B, num_of_queues, opt);
}



template <typename T>
void * mgard_cuda_handle<T>::get(int i) {
  cudaStream_t * ptr = (cudaStream_t *)(this->queues);
  // std::cout << "get: " << *(ptr+i) << "\n";
  return (void *)(ptr+i);
}

// template void * mgard_cuda_handle<double>::get(int i);
// template void * mgard_cuda_handle<float>::get(int i);


template <typename T>
void mgard_cuda_handle<T>::sync(int i) {
  cudaStream_t * ptr = (cudaStream_t *)(this->queues);
  gpuErrchk(cudaStreamSynchronize(ptr[i]));
}

// template void mgard_cuda_handle<double>::sync(int i);
// template void mgard_cuda_handle<float>::sync(int i);


template <typename T>
void mgard_cuda_handle<T>::sync_all() {
  cudaStream_t * ptr = (cudaStream_t *)(this->queues);
  for (int i = 0; i < this->num_of_queues; i++) {
    gpuErrchk(cudaStreamSynchronize(ptr[i]));
  }
}

// template void mgard_cuda_handle<double>::sync_all();
// template void mgard_cuda_handle<float>::sync_all();

template <typename T>
mgard_cuda_handle<T>::~mgard_cuda_handle (){
  destory();
}

template <typename T>
void mgard_cuda_handle<T>::destory() {
  cudaStream_t * ptr = (cudaStream_t *)(this->queues);
  for (int i = 0; i < this->num_of_queues; i++) {
    gpuErrchk(cudaStreamDestroy(ptr[i]));
  }
  delete [] this->nr_l;
  delete [] this->nc_l;
  if (this->nfib > 1) delete [] this->nf_l;

  mgard_cuda::cudaFreeHelper(this->dirow);
  mgard_cuda::cudaFreeHelper(this->dicol);
  if (this->nfib > 1) mgard_cuda::cudaFreeHelper(this->difib);

  mgard_cuda::cudaFreeHelper(this->dirow_p);
  mgard_cuda::cudaFreeHelper(this->dicol_p);
  if (this->nfib > 1) mgard_cuda::cudaFreeHelper(this->difib_p);

  mgard_cuda::cudaFreeHelper(this->dirow_a);
  mgard_cuda::cudaFreeHelper(this->dicol_a);
  if (this->nfib > 1) mgard_cuda::cudaFreeHelper(this->difib_a);

  mgard_cuda::cudaFreeHelper(this->dcoords_r);
  mgard_cuda::cudaFreeHelper(this->dcoords_c);
  if (this->nfib > 1) mgard_cuda::cudaFreeHelper(this->dcoords_f);

  mgard_cuda::cudaFreeHelper(this->ddist_r);
  mgard_cuda::cudaFreeHelper(this->ddist_c);
  if (this->nfib > 1) mgard_cuda::cudaFreeHelper(this->ddist_f);

  for (int l = 0; l < this->l_target+1; l++) {
    mgard_cuda::cudaFreeHelper(this->ddist_r_l[l]);
    mgard_cuda::cudaFreeHelper(this->ddist_c_l[l]);
    if (this->nfib > 1) mgard_cuda::cudaFreeHelper(this->ddist_f_l[l]);
  }

  delete [] this->ddist_r_l;
  delete [] this->ddist_c_l;
  if (this->nfib > 1) delete [] this->ddist_f_l;

  mgard_cuda::cudaFreeHelper(this->dwork);

  if (this->nfib > 1) {
    for (int i = 0; i < num_of_queues; i++) {
      mgard_cuda::cudaFreeHelper(this->dcwork_2d_rc[i]);
      mgard_cuda::cudaFreeHelper(this->dcwork_2d_cf[i]);
      mgard_cuda::cudaFreeHelper(this->am_row[i]);
      mgard_cuda::cudaFreeHelper(this->bm_row[i]);
      mgard_cuda::cudaFreeHelper(this->am_col[i]);
      mgard_cuda::cudaFreeHelper(this->bm_col[i]);
      mgard_cuda::cudaFreeHelper(this->am_fib[i]);
      mgard_cuda::cudaFreeHelper(this->bm_fib[i]);
    }
    delete [] this->dcwork_2d_rc;
    delete [] this->lddcwork_2d_rc;
    delete [] this->dcwork_2d_cf;
    delete [] this->lddcwork_2d_cf;

    delete [] this->am_row;
    delete [] this->bm_row;
    delete [] this->am_col;
    delete [] this->bm_col;
    delete [] this->am_fib;
    delete [] this->bm_fib;

  } else {
    for (int i = 0; i < 1; i++) {
      mgard_cuda::cudaFreeHelper(this->am_row[i]);
      mgard_cuda::cudaFreeHelper(this->bm_row[i]);
      mgard_cuda::cudaFreeHelper(this->am_col[i]);
      mgard_cuda::cudaFreeHelper(this->bm_col[i]);
    }

    delete [] this->am_row;
    delete [] this->bm_row;
    delete [] this->am_col;
    delete [] this->bm_col;
    delete [] this->am_fib;
    delete [] this->bm_fib;
  }



}


// template void mgard_cuda_handle<double>::destory_all();
// template void mgard_cuda_handle<float>::destory_all();