// Copyright 2017, Brown University, Providence, RI.
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney
// Corresponding Author: Ozan Tugluk
//
// version: 0.0.0.2
// See LICENSE for details.

#include "mgard_qoi.h"


namespace mgard_qoi
{


  double qoi_ave(const int nrow, const int ncol, const int nfib, std::vector<double> u)
  {
    double  sum = 0;

    for ( double x : u ) sum += x;

    return sum/u.size();
  }

  float qoi_ave(const int nrow, const int ncol, const int nfib, std::vector<float> u)
  {
   float  sum = 0;

    for ( double x : u ) sum += x;

    return sum/u.size();
  }


}
