// Copyright 2017, Brown University, Providence, RI.
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney
// Corresponding Author: Ozan Tugluk
//
// version: 0.0.0.2
// See LICENSE for details.
#ifndef MGARD_QOI_H
#define MGARD_QOI_H

#include <vector>

namespace mgard_qoi {

template <typename Real>
Real qoi_ave(const int nrow, const int ncol, const int nfib,
             std::vector<Real> u);

} // namespace mgard_qoi

#include "mgard_qoi.tpp"
#endif
