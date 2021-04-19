#include <iostream>
#include <sstream>
#include <string>

#include "cuda/Message.h"

using std::string;

const string log_null = "       ";
const string log_err  = "\e[31m[ERR]\e[0m  ";
const string log_dbg  = "\e[34m[dbg]\e[0m  ";
const string log_info = "\e[32m[info]\e[0m ";
const string log_warn = "\e[31m[WARN]\e[0m ";

// https://stackoverflow.com/a/26080768/8740097
template <typename T>
void mgard_cuda::log::build(std::ostream& o, T t)
{
    o << t << std::endl;
}

template <typename T, typename... Args>
void mgard_cuda::log::build(std::ostream& o, T t, Args... args)  // recursive variadic function
{
    mgard_cuda::log::build(o, t);
    mgard_cuda::log::build(o, args...);
}

template <typename... Args>
void mgard_cuda::log::print(string log_head, Args... args)
{
    std::ostringstream oss;
    mgard_cuda::log::build(oss, args...);
    std::cout << log_head << oss.str();
}