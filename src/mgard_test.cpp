// Copyright 2017, Brown University, Providence, RI.
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney
// Corresponding Author: Ozan Tugluk
// See LICENSE for details.

#include <dlfcn.h> // dlopen

#include "mgard_api.h"
#include "mgard_api_float.h"
#include "mgard_mesh.hpp"

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

template <typename DOUBLE>
int parse_cmdl(int argc, char **argv, bool &inf_flag, bool &qoi_flag, int &nrow,
               int &ncol, int &nfib, DOUBLE &tol, DOUBLE &s,
               std::string &in_file, std::string &coord_file,
               std::string &shared_obj, std::string &function_handle) {
  if (argc >= 7) {
    inf_flag = true;  // assume Linfty compression
    qoi_flag = false; // assume no dload qoi

    in_file = argv[2];
    coord_file = argv[3];

    nrow = strtol((argv[4]), NULL, 0); // number of rows
    ncol = strtol((argv[5]), NULL, 0); // number of columns
    nfib = strtol((argv[6]), NULL, 0); // number of columns

    tol = strtod((argv[7]), 0); // error tolerance

    if (argv[8] != NULL) // s-not specified assume L-infty compression
    {
      inf_flag = false;
      s = strtod((argv[8]), 0); // error tolerance
      std::cerr << s << " compression selected \n";
    } else {
      inf_flag = true;
      std::cerr << "L infinity compression selected \n";
    }

    if (argv[9] != NULL) // dynamic load quantity of intereset
    {
      std::cerr << "Using qoi" << function_handle << "\n";
      shared_obj = argv[9];
      function_handle = argv[10];
      qoi_flag = true;
    }

    assert(in_file.size() != 0);
    assert(ncol > 3);
    assert(nrow >= 1);
    assert(tol >= 1e-8);

    struct stat file_stats;
    int flag = stat(in_file.c_str(), &file_stats);

    if (flag != 0) // can't stat file somehow
    {
      throw std::runtime_error(
          "Cannot stat input file! Nothing to be done, exiting...");
    }

    return 1;
  } else {
    std::cerr << "Usage: " << argv[0] << " inputfile nrow ncol tol"
              << "\n";
    throw std::runtime_error("Too few arguments, exiting...");
  }
}

/// --- MAIN ---///

int main(int argc, char **argv) {

  bool inf_flag = true;  // assume Linfty compression
  bool qoi_flag = false; // assume no dload qoi

  double tol, s;
  float tolf, sf;

  int nrow, ncol, nfib, nlevel;
  std::string in_file, coord_file, out_file, zip_file, shared_obj,
      function_handle;

  int out_size, itype;

  unsigned char *compressed_data;

  // -- get commandline params --//

  if (argv[1] != NULL) // we at least have the data type
  {
    itype = strtol((argv[1]), NULL, 0);
  } else {
    std::cerr << "No data type specified, exiting...\n";
    return -1;
  }

  if (itype == 0) // double
  {
    parse_cmdl(argc, argv, inf_flag, qoi_flag, nrow, ncol, nfib, tol, s,
               in_file, coord_file, shared_obj, function_handle);
    std::vector<double> v(nrow * ncol * nfib), coords_x(ncol), coords_y(nrow),
        coords_z(nfib);

    //-- read input file and set dummy coordinates --//
    std::ifstream infile(in_file, std::ios::in | std::ios::binary);
    std::ifstream cordfile(coord_file, std::ios::in | std::ios::binary);

    infile.read(reinterpret_cast<char *>(v.data()),
                nrow * ncol * nfib * sizeof(double));

    //-- set and creat output files -- //
    out_file = in_file + std::to_string(tol) + "_y.dat";
    zip_file = in_file + std::to_string(tol) + ".gz";

    std::ofstream outfile(out_file, std::ios::out | std::ios::binary);
    std::ofstream zipfile(zip_file, std::ios::out | std::ios::binary);

    // compress in memory
    if (inf_flag) {
      compressed_data =
          mgard_compress(0, v.data(), out_size, nrow, ncol, nfib, tol);
    } else {
      if (qoi_flag) {
        //-- dlopen bit --//
        // open library
        void *handle = dlopen(shared_obj.c_str(), RTLD_LAZY);
        if (!handle) {
          std::cerr << "dlopen error: " << dlerror() << '\n';
          return 1;
        }

        // load symbol
        typedef double (*qoi_t)(int, int, int, double *);
        dlerror();
        qoi_t qoi = (qoi_t)dlsym(handle, function_handle.c_str());
        const char *dlsym_error = dlerror();
        if (dlsym_error) {
          std::cerr << "dlsym error: " << dlsym_error << '\n';
          dlclose(handle);
          return 1;
        }

        //-- dlopen --//

        compressed_data = mgard_compress(itype, v.data(), out_size, nrow, ncol,
                                         nfib, tol, qoi, s);
      } else {
        compressed_data =
            mgard_compress(itype, v.data(), out_size, nrow, ncol, nfib, tol, s);
      }
    }
    // std::cout  << "Compressed size" << out_size << "\n";

    zipfile.write(reinterpret_cast<char *>(compressed_data), out_size);

    // decompress in memory
    double *dtest;
    double dummy;

    if (inf_flag) {
      dtest = mgard_decompress(itype, dummy, compressed_data, out_size, nrow,
                               ncol, nfib);
    } else {
      dtest = mgard_decompress(itype, dummy, compressed_data, out_size, nrow,
                               ncol, nfib, s);
    }

    outfile.write(reinterpret_cast<char *>(dtest),
                  nrow * ncol * nfib * sizeof(double));

    free(dtest);
    free(compressed_data);

  }

  else if (itype == 1) // float
  {
    parse_cmdl(argc, argv, inf_flag, qoi_flag, nrow, ncol, nfib, tolf, sf,
               in_file, coord_file, shared_obj, function_handle);

    std::vector<float> v(nrow * ncol * nfib), coords_x(ncol), coords_y(nrow),
        coords_z(nfib);

    //-- read input file and set dummy coordinates --//
    std::ifstream infile(in_file, std::ios::in | std::ios::binary);
    std::ifstream cordfile(coord_file, std::ios::in | std::ios::binary);

    infile.read(reinterpret_cast<char *>(v.data()),
                nrow * ncol * nfib * sizeof(float));

    //-- set and creat output files -- //
    out_file = in_file + std::to_string(tol) + "_y.dat";
    zip_file = in_file + std::to_string(tol) + ".gz";

    std::ofstream outfile(out_file, std::ios::out | std::ios::binary);
    std::ofstream zipfile(zip_file, std::ios::out | std::ios::binary);

    // compress in memory
    if (inf_flag) {
      compressed_data =
          mgard_compress(itype, v.data(), out_size, nrow, ncol, nfib, tolf);
    } else {
      compressed_data =
          mgard_compress(itype, v.data(), out_size, nrow, ncol, nfib, tolf, sf);
    }
    // std::cout  << "Compressed size" << out_size << "\n";

    zipfile.write(reinterpret_cast<char *>(compressed_data), out_size);

    // decompress in memory
    float *dtest;
    float dummy;

    if (inf_flag) {
      dtest = mgard_decompress(itype, dummy, compressed_data, out_size, nrow,
                               ncol, nfib);
    } else {
      dtest = mgard_decompress(itype, dummy, compressed_data, out_size, nrow,
                               ncol, nfib, sf);
    }

    outfile.write(reinterpret_cast<char *>(dtest),
                  nrow * ncol * nfib * sizeof(float));

    free(dtest);
    free(compressed_data);
  }

  return 0;
}
