#include <iostream>
#include <sstream>
#include <string>

#include "mgard-x/RuntimeX/Utilities/Log.h"

using std::string;

namespace mgard_x {

namespace log {

int INFO = 1;
int TIME = 2;
int DBG = 4;
int WARN = 8;
int ERR = 16;

int level = ERR;

const string log_null = "       ";
const string log_info = "\e[32m[info]\e[0m ";
const string log_time = "\e[34m[time]\e[0m ";
const string log_dbg = "\e[34m[dbg]\e[0m  ";
const string log_warn = "\e[31m[WARN]\e[0m ";
const string log_err = "\e[31m[ERR]\e[0m  ";

void info(std::string msg, bool override) {
  if (level & INFO || override) {
    std::cout << log_info << msg << std::endl;
  }
}

void time(std::string msg, bool override) {
  if (level & TIME || override) {
    std::cout << log_time << msg << std::endl;
  }
}

void dbg(std::string msg, bool override) {
  if (level & DBG || override) {
    std::cout << log_dbg << msg << std::endl;
  }
}

void warn(std::string msg, bool override) {
  if (level & WARN || override) {
    std::cout << log_warn << msg << std::endl;
  }
}

void err(std::string msg, bool override) {
  if (level & ERR || override) {
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