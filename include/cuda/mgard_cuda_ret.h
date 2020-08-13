#ifndef MGRAD_CUDA_RET
#define MGRAD_CUDA_RET
struct mgard_cuda_ret {
  int info;
  double time;
  mgard_cuda_ret() : info(0), time(0.0) {}
  mgard_cuda_ret(int info, double time) {
    this->info = info;
    this->time = time;
  }
};
#endif
