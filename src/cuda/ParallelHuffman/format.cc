#include <iostream>
#include <sstream>
#include <string>

#include "cuda/ParallelHuffman/format.hh"

using std::string;

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
