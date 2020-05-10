#include "mgard_api.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

int main(int argc, char **argv) {
  double tol, result;
  int out_size;
  unsigned char *compressed_data = 0;
  double s;

  if (argc != 3) {
    cerr << "Wrong arugments!\n";
    return 1;
  }

  ifstream datafile(argv[1], ios::in | ios::binary);
  if (!datafile)
    cerr << "Can't open input file!";

  tol = stod(argv[2]);

  datafile.seekg(0, ios::end);
  size_t num_elements = datafile.tellg() / sizeof(double);
  datafile.seekg(0, ios::beg);

  std::vector<double> data(num_elements);
  datafile.read(reinterpret_cast<char *>(&data[0]),
                num_elements * sizeof(double));

  /*
  Parameter s specifies the H^s (Sobolev) norm with which we measure the error.
  s=0 is L^2 norm (can be thought as psnr).
  s=1 means the first derivatives of the function are also preserved to the
  specified tolerance etc.
  */
  s = 0.0;

  compressed_data = mgard_compress(0, data.data(), out_size, 16,
                                   num_elements / 16, 1, tol, s);
  cout << "Original size = " << num_elements * 8 << " out_size = " << out_size
       << " S = " << s << " CR = " << num_elements * 8.0 / out_size << endl;

  double quantizer;
  double *decompressed_data = mgard_decompress(
      0, quantizer, compressed_data, out_size, 16, num_elements/16, 1, s);

  double l2norm_error = 0.0;
  double l2norm = 0.0;
  double abserr = 0.0;
  double max_abserr = 0.0;
  double max = abs(data[0]);

  for (int i = 0; i < num_elements; i++) {
    // We aim to preserve L^2 norm and let us check the resulting L^2 norm
    l2norm_error += pow(decompressed_data[i] - data[i], 2.0);
    l2norm += pow(data[i], 2.0);

    // Also let us check the error of the primary quantity, i.e., L-infinity
    // norm 
    abserr = abs(decompressed_data[i] - data[i]);
    if (max_abserr < abserr) {
      max_abserr = abserr;
    }

    if (max < abs(data[i])) {
      max = abs(data[i]);
    }
  }

  l2norm_error = sqrt(l2norm_error);
  l2norm = sqrt(l2norm);

  cout << "The prescribed L^2 norm = " << tol << endl
       << "The achieved L^2 norm error = " << l2norm_error / l2norm << endl
       << "The acheived L-infinity error = " << max_abserr / max << endl;


  if (tol >= l2norm_error / l2norm) {
    return 0;
  } else {
    return 1;
  }
}
