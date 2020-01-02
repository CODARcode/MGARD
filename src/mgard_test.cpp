// Copyright 2017, Brown University, Providence, RI.
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney
// Corresponding Author: Ozan Tugluk
// See LICENSE for details.

#include <cassert>
#include <cstdlib>
#include <cstring>

#include <dlfcn.h> // dlopen
#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>

#include "mgard_api.h"
#include "mgard_mesh.hpp"

#define ANSI_RED "\x1b[31m"
#define ANSI_GREEN "\x1b[32m"
#define ANSI_RESET "\x1b[0m"

void print_usage_message(std::ostream &stream) {
  stream << "Usage: mgard_test (float|double) infile nrow ncol nfib "
         << "tolerance [s [shared_obj function_handle]]\n";
}

void print_for_more_details_message(std::ostream &stream) {
  stream << "\nFor more details, run: mgard_test --help\n";
}

struct HelpLine {
  HelpLine(const std::string &name, const std::string &description)
      : name(name), description(description) {}

  const std::string name;
  const std::string description;
};

// I would expect some bugs in here.
std::ostream &operator<<(std::ostream &stream, const HelpLine &line) {
  const std::size_t indentation = 2;
  const std::size_t line_length = 80;
  // Hardcoded. Would need to change if the `name`s got longer.
  const std::size_t description_start = 20;
  const std::size_t N = line.name.length();
  const std::size_t M = line.description.length();
  stream << std::string(indentation, ' ') << line.name
         << std::string(description_start - indentation - N, ' ');
  std::size_t i = 0;
  while (true) {
    std::size_t max_length = line_length - description_start;
    if (i) {
      stream << std::string(description_start + indentation, ' ');
      assert(max_length >= indentation);
      max_length -= indentation;
    }
    const std::size_t j = line.description.rfind(" ", i + max_length);
    if (i + max_length >= M) {
      stream << line.description.substr(i, M - i) << "\n";
      break;
    } else if (j == std::string::npos) {
      stream << line.description.substr(i, max_length) << "\n";
      i += max_length;
    } else {
      stream << line.description.substr(i, j - i) << "\n";
      i = line.description.at(j) == ' ' ? j + 1 : j;
    }
  }
  return stream;
}

void print_help_message(std::ostream &stream) {
  stream
      << "\nparameters:\n"
      << HelpLine("(float|double)", "Type of input array.")
      << HelpLine("infile", "File containing input array.")
      << HelpLine("nrow, ncol, nfib", "Dimensions of input array.")
      << HelpLine("tolerance", "Error tolerance for the compression.")
      << HelpLine(
             "s",
             "Smoothness parameter determining norm used with error tolerance. "
             "If omitted, the `L^inf` (pointwise) norm is used.")
      << HelpLine(
             "shared_obj",
             "Shared object file containing quantity of interest function.")
      << HelpLine("function_handle", "Name of quantity of interest function.");
}

void warning(const std::string &msg) {
  std::cerr << ANSI_RED << "warning: " << msg << ANSI_RESET << "\n";
}

void warning(const std::stringstream &msg) { warning(msg.str()); }

void error(const std::string &msg) {
  std::cerr << ANSI_RED << "error: " << msg << ANSI_RESET << "\n";
}

void error(const std::stringstream &msg) { error(msg.str()); }

double qoi_x(const int nrow, const int ncol, const int nfib,
             std::vector<double> u) {

  int type_indicator = 0;

  for (int irow = 0; irow < nrow; ++irow) {
    for (int jcol = 0; jcol < ncol; ++jcol) {
      for (int kfib = 0; kfib < nfib; ++kfib) {
        if (u[mgard::get_index3(ncol, nfib, irow, jcol, kfib)] != 0)
          return jcol;
      }
    }
  }
  // TODO: adding this just so compiler doesn't complain about reaching end of
  // function. Figure out what should be returned.
  return 0;
}

double qoi_ave(const int nrow, const int ncol, const int nfib,
               std::vector<double> u) {
  double sum = 0;

  for (double x : u)
    sum += x;

  return sum / u.size();
}

double qoi_one(const int nrow, const int ncol, const int nfib,
               std::vector<double> u) {
  double qov = 1.0;

  int type_indicator = 0;
  double h;

  for (int irow = 0; irow < nrow; ++irow) {
    for (int jcol = 0; jcol < ncol; ++jcol) {
      for (int kfib = 0; kfib < nfib; ++kfib) {
        if ((irow == 0 || irow == nrow - 1) &&
            (u[mgard::get_index3(ncol, nfib, irow, jcol, kfib)] != 0))
          ++type_indicator;

        if ((jcol == 0 || jcol == ncol - 1) &&
            (u[mgard::get_index3(ncol, nfib, irow, jcol, kfib)] != 0))
          ++type_indicator;

        if ((kfib == 0 || kfib == nfib - 1) &&
            (u[mgard::get_index3(ncol, nfib, irow, jcol, kfib)] != 0))
          ++type_indicator;
      }
    }
  }

  switch (type_indicator) {
  case 0:
    return 1.0;
  case 1:
    return 0.5;
  case 2:
    return 0.25;
  case 3:
    return 0.125;
  default:
    return 1.0;
  }
}

void announce_unparsed_characters(char const *const p, const std::size_t n) {
  std::stringstream msg;
  msg << "'" << p << "' only parsed up to '" << p[n] << "'";
  warning(msg);
}

int parse_int(char const *const p) {
  std::size_t n;
  const int i = std::stoi(p, &n);
  if (p[n]) {
    announce_unparsed_characters(p, n);
  }
  return i;
}

double parse_double(char const *const p) {
  std::size_t n;
  const double d = std::stod(p, &n);
  if (p[n]) {
    announce_unparsed_characters(p, n);
  }
  return d;
}

template <typename Real> struct Parameters {
  std::string infilename;
  int nrow;
  int ncol;
  int nfib;

  Real tol;

  bool inf_flag;
  Real s;

  bool qoi_flag;
  std::string shared_obj;
  std::string function_handle;
};

template <typename Real>
Parameters<Real> parse_cmdl(const int argc, char const *const *const argv) {
  if (argc < 7 || argc > 10) {
    print_usage_message(std::cerr);
    print_for_more_details_message(std::cerr);
    std::exit(1);
  }

  Parameters<Real> parameters;
  parameters.infilename = argv[2];

  parameters.nrow = parse_int(argv[3]);
  parameters.ncol = parse_int(argv[4]);
  parameters.nfib = parse_int(argv[5]);

  // Parsing `Real`s as `double`s and then casting. Fine as long as `Real`
  // is `float` or `double`.
  parameters.tol = parse_double(argv[6]);

  parameters.inf_flag = argc == 7;
  parameters.qoi_flag = argc == 10;

  if (!parameters.inf_flag) {
    parameters.s = parse_double(argv[7]);
    std::cerr << "compressing using the `s = " << parameters.s << "` norm\n";
  } else {
    std::cerr << "compressing using the `L^inf` norm\n";
    //`s` won't be used in this case (except in filenames).
    parameters.s = std::numeric_limits<Real>::infinity();
  }

  if (parameters.qoi_flag) {
    parameters.shared_obj = argv[8];
    parameters.function_handle = argv[9];
    std::cerr << "using QoI `" << parameters.function_handle << "` from `"
              << parameters.shared_obj << "`\n";
  }

  return parameters;
}

template <typename Real> int get_itype();

template <> int get_itype<float>() { return 0; }

template <> int get_itype<double>() { return 1; }

template <typename Real> int run(const int argc, char const *const *argv) {
  const Parameters<Real> parameters = parse_cmdl<Real>(argc, argv);
  const int itype = get_itype<Real>();

  if (parameters.infilename.size() == 0) {
    error("`infile` must have nonempty name");
    return 1;
  }
  {
    struct stat file_stats;
    const int flag = stat(parameters.infilename.c_str(), &file_stats);
    if (flag != 0) {
      std::stringstream msg;
      msg << "error: cannot stat `" << parameters.infilename << "`";
      error(msg);
      return 1;
    }
  }
  if (parameters.ncol <= 3) {
    error("`ncol` must be greater than 3");
    return 1;
  }
  if (parameters.nrow < 1) {
    error("`nrow` must be positive");
    return 1;
  }
  if (parameters.tol < 1e-8) {
    std::cout << "tol = " << parameters.tol << std::endl;
    error("`tol` must be at least 1e-8");
    return 1;
  }

  const int N = parameters.nrow * parameters.ncol * parameters.nfib;
  std::vector<Real> v_(N);
  Real *const v = v_.data();

  {
    std::ifstream infile(parameters.infilename,
                         std::ios::in | std::ios::binary);
    infile.read(reinterpret_cast<char *>(v), N * sizeof(Real));
    const std::streamsize n = infile.gcount();
    if (n != N * sizeof(Real)) {
      std::stringstream msg;
      msg << "expected " << N * sizeof(Real) << " bytes but read " << n;
      error(msg);
      return 1;
    }
  }

  unsigned char *compressed_data;
  int out_size;
  if (parameters.inf_flag) {
    compressed_data =
        mgard_compress(itype, v, out_size, parameters.nrow, parameters.ncol,
                       parameters.nfib, parameters.tol);
  } else if (parameters.qoi_flag) {
    void *handle = dlopen(parameters.shared_obj.c_str(), RTLD_LAZY);
    if (!handle) {
      std::stringstream msg;
      msg << "[dlopen] " << dlerror();
      error(msg);
      return 1;
    }
    typedef Real (*qoi_t)(int, int, int, Real *);
    // From <https://linux.die.net/man/3/dlopen>:
    //  Since the value of the symbol [returned by `dlerror`] could actually
    //  be `NULL` (so that a `NULL` return from `dlsym()` need not indicate
    //  an error), the correct way to test for an error is to call
    //  `dlerror()` to clear any old error conditions, then call `dlsym()`,
    //  and then call `dlerror()` again, saving its return value into a
    //  variable, and check whether this saved saved value is not `NULL`.
    dlerror();
    qoi_t qoi = reinterpret_cast<qoi_t>(
        dlsym(handle, parameters.function_handle.c_str()));
    const char *dlsym_error = dlerror();
    if (dlsym_error) {
      std::stringstream msg;
      msg << "[dlsym] " << dlsym_error;
      error(msg);
      dlclose(handle);
      return 1;
    }
    compressed_data =
        mgard_compress(itype, v, out_size, parameters.nrow, parameters.ncol,
                       parameters.nfib, parameters.tol, qoi, parameters.s);
  } else {
    compressed_data =
        mgard_compress(itype, v, out_size, parameters.nrow, parameters.ncol,
                       parameters.nfib, parameters.tol, parameters.s);
  }

  {
    const std::string zipfilename =
        parameters.infilename + ".s=" + std::to_string(parameters.s) +
        ".tol=" + std::to_string(parameters.tol) + ".gz";
    std::ofstream zipfile(zipfilename, std::ios::out | std::ios::binary);
    zipfile.write(reinterpret_cast<char *>(compressed_data), out_size);
  }

  Real *dtest;
  // Dummy quantizer.
  Real dummy;

  if (parameters.inf_flag) {
    dtest = mgard_decompress<Real>(itype, dummy, compressed_data, out_size,
                                   parameters.nrow, parameters.ncol,
                                   parameters.nfib);
  } else {
    dtest = mgard_decompress<Real>(itype, dummy, compressed_data, out_size,
                                   parameters.nrow, parameters.ncol,
                                   parameters.nfib, parameters.s);
  }

  {
    const std::string outfilename =
        parameters.infilename + ".s=" + std::to_string(parameters.s) +
        ".tol=" + std::to_string(parameters.tol) + ".mgard";
    std::ofstream outfile(outfilename, std::ios::out | std::ios::binary);
    outfile.write(reinterpret_cast<char *>(dtest), N * sizeof(Real));
  }

  free(dtest);
  free(compressed_data);
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    print_usage_message(std::cerr);
    print_for_more_details_message(std::cerr);
    return 1;
  }

  if (!std::strcmp(argv[1], "--help") || !std::strcmp(argv[1], "-h")) {
    print_usage_message(std::cout);
    print_help_message(std::cout);
    return 0;
  } else if (!std::strcmp(argv[1], "float")) {
    return run<float>(argc, argv);
  } else if (!std::strcmp(argv[1], "double")) {
    return run<double>(argc, argv);
  } else {
    print_usage_message(std::cerr);
    print_for_more_details_message(std::cerr);
    return 1;
  }
}
