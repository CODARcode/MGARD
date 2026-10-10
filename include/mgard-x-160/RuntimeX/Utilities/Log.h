#ifndef MGARD_X_160_MESSGAE_HH
#define MGARD_X_160_MESSGAE_HH

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using std::string;

namespace mgard_x_160 {
namespace log {

extern int INFO;
extern int TIME;
extern int DBG;
extern int WARN;
extern int ERR;

extern int level;

extern const string log_null;
extern const string log_info;
extern const string log_time;
extern const string log_dbg;
extern const string log_warn;
extern const string log_err;

void info(std::string msg, bool override = false);
void time(std::string msg, bool override = false);
void dbg(std::string msg, bool override = false);
void warn(std::string msg, bool override = false);
void err(std::string msg, bool override = false);

// https://stackoverflow.com/a/26080768/8740097
template <typename T> void build(std::ostream &o, T t);

template <typename T, typename... Args>
void build(std::ostream &o, T t, Args... args);

template <typename... Args> void print(string log_head, Args... args);

template <typename T> void csv(std::string file, T v) {
  std::ofstream myfile;
  myfile.open(file, std::ios_base::app);
  myfile << v << ", ";
  myfile.close();
}

} // namespace log

} // namespace mgard_x_160

#endif // FORMAT_HH
