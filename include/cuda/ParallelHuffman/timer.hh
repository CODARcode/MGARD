#ifndef TIMER_HH
#define TIMER_HH

#include <chrono>
#include <iostream>

using std::cerr;
using std::cout;
using std::endl;

using hires = std::chrono::high_resolution_clock;
typedef std::chrono::duration<double> duration_t;
typedef std::chrono::time_point<std::chrono::high_resolution_clock>
    hires_clock_t;

#endif // TIMER_HH
