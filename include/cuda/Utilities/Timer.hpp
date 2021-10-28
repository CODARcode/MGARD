/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#ifndef MGRAD_CUDA_TIMER_HPP
#define MGRAD_CUDA_TIMER_HPP

namespace mgard_cuda {
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
        std::cout << s << " time: " << total_time << "s" << std::endl;
        clear();
    }
  private:
    int err = 0;
    double total_time = 0;
    struct timespec start_time, end_time;
};
}
#endif