/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_TIMER_HPP
#define MGARD_X_TIMER_HPP

namespace mgard_x {
class Timer{
  public:
    void start(){
        err = clock_gettime(CLOCK_REALTIME, &start_time);
    }
    void end(){
        err = clock_gettime(CLOCK_REALTIME, &end_time);
        total_time += (double)(end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_nsec - start_time.tv_nsec)/(double)1000000000;
    }
    double get(){
        double time = (double)(end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_nsec - start_time.tv_nsec)/(double)1000000000;
        clear();
        return time;
    }
    void clear(){
        total_time = 0;
    }
    void print(std::string s){
        std::cout << log::log_time << s << " time: " << total_time << "s" << std::endl;
        clear();
    }
  private:
    int err = 0;
    double total_time = 0;
    struct timespec start_time, end_time;
};
}
#endif