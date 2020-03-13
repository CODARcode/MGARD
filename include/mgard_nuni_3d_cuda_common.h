#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_cuda_helper.h"

namespace mgard_common {

template <typename T>
T max_norm_cuda(const std::vector<T> &v);
}
