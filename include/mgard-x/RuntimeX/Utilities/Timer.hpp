/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_TIMER_HPP
#define MGARD_X_TIMER_HPP

namespace mgard_x {
class Timer {
public:
  void start() { err = clock_gettime(CLOCK_REALTIME, &start_time); }
  void end() {
    err = clock_gettime(CLOCK_REALTIME, &end_time);
    total_time +=
        (double)(end_time.tv_sec - start_time.tv_sec) +
        (double)(end_time.tv_nsec - start_time.tv_nsec) / (double)1000000000;
  }
  double get() {
    double time =
        (double)(end_time.tv_sec - start_time.tv_sec) +
        (double)(end_time.tv_nsec - start_time.tv_nsec) / (double)1000000000;
    return time;
  }

  double get_throughput(SIZE n) { return (double)n / get() / 1e9; }

  void clear() { total_time = 0; }
  void print(std::string s) {
    log::time(s + " time: " + std::to_string(total_time) + " s");
  }

  void print_throughput(std::string s, SIZE n) {
    log::time(s + " throughput: " + std::to_string(get_throughput(n)) +
              " GB/s");
  }

private:
  int err = 0;
  double total_time = 0;
  struct timespec start_time, end_time;
};
} // namespace mgard_x
#endif