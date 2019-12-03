#ifndef MGARD_QOI_TPP
#define MGARD_QOI_TPP

namespace mgard_qoi {

template <typename Real>
Real qoi_ave(const int nrow, const int ncol, const int nfib,
               std::vector<Real> u) {
  Real sum = 0;

  for (Real x : u)
    sum += x;

  return sum / u.size();
}

} // namespace mgard_qoi

#endif
