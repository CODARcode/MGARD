#include "mgard_norms.hpp"

#include <cmath>

#include "mgard_mesh.hpp"
#include "mgard_nuni.h"

namespace mgard {

template <typename Real>
void project_2D(const int nr, const int nc, const int nrow, const int ncol,
                const int l_target, Real const *const v,
                std::vector<Real> &work, const std::vector<Real> &coords_x,
                const std::vector<Real> &coords_y, std::vector<Real> &row_vec,
                std::vector<Real> &col_vec) {
  // `v` seems not to be used here.
  int l = l_target;
  int stride = std::pow(2, l); // current stride
  int Cstride = stride * 2;    // coarser stride

  // row-sweep
  for (int irow = 0; irow < nr; ++irow) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[mgard::get_index(ncol, ir, jcol)];
    }

    mgard_gen::mass_mult_l(l, row_vec, coords_x, nc, ncol);

    mgard_gen::restriction_l(l + 1, row_vec, coords_x, nc, ncol);

    mgard_gen::solve_tridiag_M_l(l + 1, row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[mgard::get_index(ncol, ir, jcol)] = row_vec[jcol];
    }
  }

  // column-sweep
  if (nrow > 1) // do this if we have an 2-dimensional array
  {
    for (int jcol = 0; jcol < nc; jcol += Cstride) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      for (int irow = 0; irow < nrow; ++irow) {
        col_vec[irow] = work[mgard::get_index(ncol, irow, jr)];
      }

      mgard_gen::mass_mult_l(l, col_vec, coords_y, nr, nrow);
      mgard_gen::restriction_l(l + 1, col_vec, coords_y, nr, nrow);
      mgard_gen::solve_tridiag_M_l(l + 1, col_vec, coords_y, nr, nrow);

      for (int irow = 0; irow < nrow; ++irow) {
        work[mgard::get_index(ncol, irow, jr)] = col_vec[irow];
      }
    }
  }
}

template <typename Real>
void project_2D_non_canon(const int nr, const int nc, const int nrow,
                          const int ncol, const int l_target,
                          Real const *const v, std::vector<Real> &work,
                          const std::vector<Real> &coords_x,
                          const std::vector<Real> &coords_y,
                          std::vector<Real> &row_vec,
                          std::vector<Real> &col_vec) {
  // `v` seems not to be used here.
  for (int irow = 0; irow < nrow; ++irow) {
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[mgard::get_index(ncol, irow, jcol)];
    }

    mgard_cannon::mass_matrix_multiply(0, row_vec, coords_x);

    mgard_gen::restriction_first(row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[mgard::get_index(ncol, irow, jcol)] = row_vec[jcol];
    }
  }

  for (int irow = 0; irow < nr; ++irow) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[mgard::get_index(ncol, ir, jcol)];
    }

    mgard_gen::solve_tridiag_M_l(0, row_vec, coords_x, nc, ncol);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[mgard::get_index(ncol, ir, jcol)] = row_vec[jcol];
    }
  }

  if (nrow > 1) // check if we have 1-D array..
  {
    for (int jcol = 0; jcol < ncol; ++jcol) {
      for (int irow = 0; irow < nrow; ++irow) {
        col_vec[irow] = work[mgard::get_index(ncol, irow, jcol)];
      }

      mgard_cannon::mass_matrix_multiply(0, col_vec, coords_y);

      mgard_gen::restriction_first(col_vec, coords_y, nr, nrow);

      for (int irow = 0; irow < nrow; ++irow) {
        work[mgard::get_index(ncol, irow, jcol)] = col_vec[irow];
      }
    }

    for (int jcol = 0; jcol < nc; ++jcol) {
      int jr = mgard::get_lindex(nc, ncol, jcol);
      for (int irow = 0; irow < nrow; ++irow) {
        col_vec[irow] = work[mgard::get_index(ncol, irow, jr)];
      }

      mgard_gen::solve_tridiag_M_l(0, col_vec, coords_y, nr, nrow);
      for (int irow = 0; irow < nrow; ++irow) {
        work[mgard::get_index(ncol, irow, jr)] = col_vec[irow];
      }
    }
  }
}

template <typename Real>
void project_non_canon(const int nr, const int nc, const int nf, const int nrow,
                       const int ncol, const int nfib, const int l_target,
                       Real const *const v, std::vector<Real> &work,
                       std::vector<Real> &work2d,
                       const std::vector<Real> &coords_x,
                       const std::vector<Real> &coords_y,
                       const std::vector<Real> &coords_z,
                       std::vector<Real> &norm_vec) {
  int l = 0;
  int stride = 1;

  std::vector<Real> v2d(nrow * ncol), fib_vec(nfib);
  std::vector<Real> row_vec(ncol);
  std::vector<Real> col_vec(nrow);

  mgard_gen::copy3_level(0, v, work.data(), nrow, ncol, nfib);

  for (int kfib = 0; kfib < nfib; kfib += stride) {
    mgard_common::copy_slice(work.data(), work2d, nrow, ncol, nfib, kfib);
    project_2D_non_canon(nr, nc, nrow, ncol, l, v2d.data(), work2d, coords_x,
                         coords_y, row_vec, col_vec);
    mgard_common::copy_from_slice(work.data(), work2d, nrow, ncol, nfib, kfib);
  }

  for (int irow = 0; irow < nr; irow += stride) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < nc; jcol += stride) {
      int jc = mgard::get_lindex(nc, ncol, jcol);
      for (int kfib = 0; kfib < nfib; ++kfib) {
        fib_vec[kfib] = work[mgard::get_index3(ncol, nfib, ir, jc, kfib)];
      }
      mgard_cannon::mass_matrix_multiply(l, fib_vec, coords_z);
      mgard_gen::restriction_first(fib_vec, coords_z, nf, nfib);
      mgard_gen::solve_tridiag_M_l(l, fib_vec, coords_z, nf, nfib);
      for (int kfib = 0; kfib < nfib; ++kfib) {
        work[mgard::get_index3(ncol, nfib, ir, jc, kfib)] = fib_vec[kfib];
      }
    }
  }

  // Originally, `work.data()` came before `v`. In the definition, the order is
  // reversed, so I changed it. Haven't checked that this is right.
  mgard_gen::copy3_level_l(0, v, work.data(), nr, nc, nf, nrow, ncol, nfib);
  //  gard_gen::assign3_level_l(0, work.data(),  0.0,  nr,  nc, nf,  nrow, ncol,
  //  nfib);
  // Real norm = mgard_gen::ml2_norm3(0, nr, nc, nf, nrow, ncol, nfib, work,
  // coords_x, coords_y, coords_z);
  // //std::cout  << "sqL2 norm init: "<< std::sqrt(norm) << "\n";
  // norm_vec[0] = norm;
}

template <typename Real>
void project_3D(const int nr, const int nc, const int nf, const int nrow,
                const int ncol, const int nfib, const int l_target,
                Real const *const v, std::vector<Real> &work,
                std::vector<Real> &work2d, const std::vector<Real> &coords_x,
                const std::vector<Real> &coords_y,
                const std::vector<Real> &coords_z,
                std::vector<Real> &norm_vec) {

  std::vector<Real> v2d(nrow * ncol), fib_vec(nfib);
  std::vector<Real> row_vec(ncol);
  std::vector<Real> col_vec(nrow);

  int nlevel_x = std::log2(nc - 1);
  int nlevel_y = std::log2(nr - 1);
  int nlevel_z = std::log2(nf - 1);

  // mgard_gen::copy3_level_l(0,  v,  work.data(),  nr,  nc, nf,  nrow,  ncol,
  // nfib); Real norm = mgard_gen::ml2_norm3_ng(0, nr, nc, nf, nrow, ncol,
  // nfib, work, coords_x, coords_y, coords_z);
  // //std::cout  << "sqL2 norm: "<< (norm) << "\n";
  // norm_vec[0] = norm;

  for (int l = 0; l < l_target; ++l) {
    int stride = std::pow(2, l);
    int Cstride = 2 * stride;

    int xsize = std::pow(2, nlevel_x - l) + 1;
    int ysize = std::pow(2, nlevel_y - l) + 1;
    int zsize = std::pow(2, nlevel_z - l) + 1;

    //       mgard_gen::copy3_level_l(l,  v,  work.data(),  nr,  nc, nf,  nrow,
    //       ncol, nfib);

    for (int kfib = 0; kfib < nf; kfib += stride) {
      int kf = mgard::get_lindex(
          nf, nfib,
          kfib); // get the real location of logical index irow
      mgard_common::copy_slice(work.data(), work2d, nrow, ncol, nfib, kf);
      project_2D(nr, nc, nrow, ncol, l, v2d.data(), work2d, coords_x, coords_y,
                 row_vec, col_vec);
      mgard_common::copy_from_slice(work.data(), work2d, nrow, ncol, nfib, kf);
    }

    for (int irow = 0; irow < nr; irow += Cstride) {
      int ir = mgard::get_lindex(nr, nrow, irow);
      for (int jcol = 0; jcol < nc; jcol += Cstride) {
        int jc = mgard::get_lindex(nc, ncol, jcol);
        for (int kfib = 0; kfib < nfib; ++kfib) {
          fib_vec[kfib] = work[mgard::get_index3(ncol, nfib, ir, jc, kfib)];
        }
        mgard_gen::mass_mult_l(l, fib_vec, coords_z, nf, nfib);
        mgard_gen::restriction_l(l + 1, fib_vec, coords_z, nf, nfib);
        mgard_gen::solve_tridiag_M_l(l + 1, fib_vec, coords_z, nf, nfib);
        for (int kfib = 0; kfib < nfib; ++kfib) {
          work[mgard::get_index3(ncol, nfib, ir, jc, kfib)] = fib_vec[kfib];
        }
      }
    }

    //       mgard_gen::copy3_level_l(l,  work.data(), v,  nr,  nc, nf,  nrow,
    //       ncol, nfib);
    norm_vec[l + 1] = mgard_gen::ml2_norm3(l + 1, nr, nc, nf, nrow, ncol, nfib,
                                           work, coords_x, coords_y, coords_z);
    // std::cout  << "sqL2 norm : "<< l <<"\t"<<(norm_vec[l+1]) << "\n";
  }

  Real norm = mgard_gen::ml2_norm3(l_target + 1, nr, nc, nf, nrow, ncol, nfib,
                                   work, coords_x, coords_y, coords_z);
  // std::cout  << "sqL2 norm lasto 0: "<< std::sqrt(norm) << "\n";
  norm_vec[l_target + 1] = norm;

  int stride = std::pow(2, l_target + 1);
  for (int irow = 0; irow < nr; irow += stride) {
    int ir = mgard::get_lindex(nr, nrow, irow);
    for (int jcol = 0; jcol < nc; jcol += stride) {
      int jc = mgard::get_lindex(nc, ncol, jcol);
      for (int kfib = 0; kfib < nf; kfib += stride) {
        int kf = mgard::get_lindex(nf, nfib, kfib);
        // std::cout  << work[mgard::get_index3(ncol,nfib,ir,jc,kf)] <<
        // "\n";
      }
    }
  }
}

// Could possibly unite these two using `if constexpr` or something.
template <typename Real>
std::vector<Real> qoi_rhs(const int nrow, const int ncol, const int nfib,
                          Real (*const qoi)(int, int, int, std::vector<Real>),
                          std::vector<Real> &work3d) {
  const std::size_t N = nrow * ncol * nfib;
  std::vector<Real> xi(N);
  // `work3d` must also have size `N`.
  std::vector<Real> &p = work3d;
  for (std::size_t i = 0; i < N; ++i) {
    p[i] = 1.0;
    xi[i] = qoi(nrow, ncol, nfib, p);
    p[i] = 0.0;
  }
  return xi;
}

template <typename Real>
std::vector<Real> qoi_rhs(const int nrow, const int ncol, const int nfib,
                          Real (*const qoi)(int, int, int, Real *),
                          std::vector<Real> &work3d) {
  const std::size_t N = nrow * ncol * nfib;
  std::vector<Real> xi(N);
  // `work3d` must also have size `N`.
  Real *const p = work3d.data();
  for (std::size_t i = 0; i < N; ++i) {
    p[i] = 1.0;
    xi[i] = qoi(nrow, ncol, nfib, p);
    p[i] = 0.0;
  }
  return xi;
}

template <typename Real>
Real qoi_rhs_s_norm(const int nrow, const int ncol, const int nfib,
                    const std::vector<Real> &coords_x,
                    const std::vector<Real> &coords_y,
                    const std::vector<Real> &coords_z, std::vector<Real> &xi,
                    const Real s, std::vector<Real> &work3d,
                    std::vector<Real> &work2d) {
  // `workd3d` must have size `nrow * ncol * nfib` and `work2d` must have size
  // `nrow * ncol`.
  const Dimensions2kPlus1<3> dims({nrow, ncol, nfib});
  const int l_target = dims.nlevel - 1;

  std::vector<Real> norm_vec(dims.nlevel + 1);
  project_non_canon(dims.rnded.at(0), dims.rnded.at(1), dims.rnded.at(2),
                    dims.input.at(0), dims.input.at(1), dims.input.at(2),
                    l_target, xi.data(), work3d, work2d, coords_x, coords_y,
                    coords_z, norm_vec);
  // Originally `xi` was passed instead of `work3d`.
  project_3D(dims.rnded.at(0), dims.rnded.at(1), dims.rnded.at(2),
             dims.input.at(0), dims.input.at(1), dims.input.at(2), l_target,
             xi.data(), work3d, work2d, coords_x, coords_y, coords_z, norm_vec);
  norm_vec[dims.nlevel] = 0;
  // Originally this function was called twice, the second time as below (with
  // the result discarded) and the first time using only `dims.input` values
  // (with the result assigned to `norm_vec[0]`). I am assuming that that was an
  // error, but I haven't dug through the code to check.
  norm_vec[0] =
      mgard_gen::ml2_norm3(dims.nlevel, dims.rnded.at(0), dims.rnded.at(1),
                           dims.rnded.at(2), dims.input.at(0), dims.input.at(1),
                           dims.input.at(2), xi, coords_x, coords_y, coords_z);

  Real sum = 0.0;
  for (int i = dims.nlevel; i != 0; --i) {
    const Real norm_lm1 = norm_vec[i - 1];
    const Real norm_l = norm_vec[i];
    const Real diff = norm_l - norm_lm1;
    sum += std::pow(2, 2 * s * (dims.nlevel - i)) * diff;
  }

  // check this is equal to |X_L| (s=0)
  return std::sqrt(std::abs(sum));
}

template <typename Real, typename F>
Real qoi_norm(const int nrow, const int ncol, const int nfib,
              const std::vector<Real> &coords_x,
              const std::vector<Real> &coords_y,
              const std::vector<Real> &coords_z, const F qoi, const Real s) {
  const std::size_t N = nrow * ncol * nfib;
  std::vector<Real> work3d(N);
  std::vector<Real> work2d(nrow * ncol);
  std::vector<Real> row_vec(ncol);
  std::vector<Real> col_vec(nrow);
  std::vector<Real> fib_vec(nfib);
  std::vector<Real> xi = qoi_rhs<Real>(nrow, ncol, nfib, qoi, work3d);

  const int stride = 1;

  for (int kfib = 0; kfib < nfib; kfib += stride) {
    mgard_common::copy_slice(xi.data(), work2d, nrow, ncol, nfib, kfib);
    for (int irow = 0; irow < nrow; ++irow) {
      for (int jcol = 0; jcol < ncol; ++jcol) {
        row_vec[jcol] = work2d[mgard::get_index(ncol, irow, jcol)];
      }

      mgard_cannon::solve_tridiag_M(0, row_vec, coords_x);
      for (int jcol = 0; jcol < ncol; ++jcol) {
        work2d[mgard::get_index(ncol, irow, jcol)] = row_vec[jcol];
      }
    }

    if (nrow > 1) // check if we have 1-D array.
    {
      for (int jcol = 0; jcol < ncol; ++jcol) {
        for (int irow = 0; irow < nrow; ++irow) {
          col_vec[irow] = work2d[mgard::get_index(ncol, irow, jcol)];
        }

        mgard_cannon::solve_tridiag_M(0, col_vec, coords_y);
        for (int irow = 0; irow < nrow; ++irow) {
          work2d[mgard::get_index(ncol, irow, jcol)] = col_vec[irow];
        }
      }
    }

    mgard_common::copy_from_slice(xi.data(), work2d, nrow, ncol, nfib, kfib);
  }

  for (int irow = 0; irow < nrow; irow += stride) {
    for (int jcol = 0; jcol < ncol; jcol += stride) {
      for (int kfib = 0; kfib < nfib; ++kfib) {
        fib_vec[kfib] = xi[mgard::get_index3(ncol, nfib, irow, jcol, kfib)];
      }
      mgard_cannon::solve_tridiag_M(0, fib_vec, coords_z);
      for (int kfib = 0; kfib < nfib; ++kfib) {
        xi[mgard::get_index3(ncol, nfib, irow, jcol, kfib)] = fib_vec[kfib];
      }
    }
  }

  for (std::size_t i = 0; i < N; ++i) {
    xi[i] *= 216.0; // account for the (1/6)^3 factors in the mass matrix
  }

  return qoi_rhs_s_norm(nrow, ncol, nfib, coords_x, coords_y, coords_z, xi, s,
                        work3d, work2d);
}

template <typename Real>
Real qoi_norm(const int nrow, const int ncol, const int nfib,
              const std::vector<Real> &coords_x,
              const std::vector<Real> &coords_y,
              const std::vector<Real> &coords_z,
              Real (*const qoi)(int, int, int, std::vector<Real>),
              const Real s) {
  return qoi_norm<Real, Real (*const)(int, int, int, std::vector<Real>)>(
      nrow, ncol, nfib, coords_x, coords_y, coords_z, qoi, s);
}

template <typename Real>
Real qoi_norm(const int nrow, const int ncol, const int nfib,
              const std::vector<Real> &coords_x,
              const std::vector<Real> &coords_y,
              const std::vector<Real> &coords_z,
              Real (*const qoi)(int, int, int, Real *), const Real s) {
  return qoi_norm<Real, Real (*const)(int, int, int, Real *)>(
      nrow, ncol, nfib, coords_x, coords_y, coords_z, qoi, s);
}

} // namespace mgard
