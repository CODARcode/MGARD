#include <iostream>
#include <sstream>
#include <string>

#include "cuda/parallel_huffman/format.hh"

using std::string;

const string log_null = "       ";
const string log_err = "\e[31m[ERR]\e[0m  ";
const string log_dbg = "\e[34m[dbg]\e[0m  ";
const string log_info = "\e[32m[info]\e[0m ";
const string log_warn = "\e[31m[WARN]\e[0m ";

// https://stackoverflow.com/a/26080768/8740097
template <typename T> void huffman_gpu::log::build(std::ostream &o, T t) {
  o << t << std::endl;
}

template <typename T, typename... Args>
void huffman_gpu::log::build(std::ostream &o, T t,
                             Args... args) // recursive variadic function
{
  huffman_gpu::log::build(o, t);
  huffman_gpu::log::build(o, args...);
}

template <typename... Args>
void huffman_gpu::log::print(string log_head, Args... args) {
  std::ostringstream oss;
  huffman_gpu::log::build(oss, args...);
  std::cout << log_head << oss.str();
}
