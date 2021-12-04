#ifndef FORMAT_HH
#define FORMAT_HH

#include <iostream>
#include <sstream>
#include <string>

using std::string;

const string log_null = "       ";
const string log_err = "\e[31m[ERR]\e[0m  ";
const string log_dbg = "\e[34m[dbg]\e[0m  ";
const string log_info = "\e[32m[info]\e[0m ";
const string log_warn = "\e[31m[WARN]\e[0m ";

namespace huffman_gpu {
namespace log {

// https://stackoverflow.com/a/26080768/8740097
template <typename T> void build(std::ostream &o, T t);

template <typename T, typename... Args>
void build(std::ostream &o, T t, Args... args);

template <typename... Args> void print(string log_head, Args... args);

} // namespace log

} // namespace huffman_gpu

#endif // FORMAT_HH
