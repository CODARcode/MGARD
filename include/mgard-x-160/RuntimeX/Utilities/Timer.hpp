/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_160_TIMER_HPP
#define MGARD_X_160_TIMER_HPP

namespace mgard_x_160 {
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
    return total_time;
  }

  double get_throughput(SIZE num_bytes) {
    return (double)num_bytes / get() / 1e9;
  }

  void clear() { total_time = 0; }
  void print(std::string s, SIZE num_bytes = 0, bool override = false) {
    if (num_bytes == 0) {
      log::time(s + ": " + std::to_string(total_time) + " s", override);
    } else {
      log::time(s + ": " + std::to_string(total_time) + " s (" +
                    std::to_string(get_throughput(num_bytes)) + " GB/s)",
                override);
    }
  }

  void print_throughput(std::string s, SIZE n, bool override = false) {
    log::time(s + " throughput: " + std::to_string(get_throughput(n)) + " GB/s",
              override);
  }

private:
  int err = 0;
  double total_time = 0;
  struct timespec start_time, end_time;
};
} // namespace mgard_x_160
#endif