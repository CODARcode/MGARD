#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_cuda_helper.h"
#include "mgard_cuda_helper_internal.h"
#include <fstream>

namespace mgard_common {

template <typename T>
T max_norm_cuda(const std::vector<T> &v) {
  double norm = 0;

  for (int i = 0; i < v.size(); ++i) {
    T ntest = std::abs(v[i]);
    if (ntest > norm)
      norm = ntest;
  }
  return norm;
}

template double max_norm_cuda<double>(const std::vector<double> &v);
template float max_norm_cuda<float>(const std::vector<float> &v);

}