#include <iostream>
#include <sstream>
#include <string>

#include "mgard-x/RuntimeX/Utilities/Message.h"

using std::string;

namespace mgard_x {

namespace log {

bool enable_log_info = true;
bool enable_log_time = false;
bool enable_log_dbg = false;
bool enable_log_warn = false;
bool enable_log_err = false;

const string log_null = "       ";
const string log_info = "\e[32m[info]\e[0m ";
const string log_time = "\e[34m[time]\e[0m ";
const string log_dbg = "\e[34m[dbg]\e[0m  ";
const string log_warn = "\e[31m[WARN]\e[0m ";
const string log_err = "\e[31m[ERR]\e[0m  ";

void info(std::string msg) {
  if (enable_log_info) {
    std::cout << log_info << msg << std::endl;
  }
}

void time(std::string msg) {
  if (enable_log_time) {
    std::cout << log_time << msg << std::endl;
  }
}

void dbg(std::string msg) {
  if (enable_log_dbg) {
    std::cout << log_dbg << msg << std::endl;
  }
}

void warn(std::string msg) {
  if (enable_log_warn) {
    std::cout << log_warn << msg << std::endl;
  }
}

void err(std::string msg) {
  if (enable_log_err) {
    std::cout << log_err << msg << std::endl;
  }
}

// https://stackoverflow.com/a/26080768/8740097
template <typename T> void build(std::ostream &o, T t) { o << t << std::endl; }

template <typename T, typename... Args>
void build(std::ostream &o, T t,
           Args... args) // recursive variadic function
{
  build(o, t);
  build(o, args...);
}

template <typename... Args> void print(string log_head, Args... args) {
  std::ostringstream oss;
  build(oss, args...);
  std::cout << log_head << oss.str();
}

} // namespace log

} // namespace mgard_x